# SPDX-License-Identifier: Apache-2.0
#
# SCAIL-2 DiT, ported from the official implementation (Apache-2.0):
# https://github.com/zai-org/SCAIL-2/blob/main/wan/modules/model_scail2.py
#
# SCAIL-2 is the Wan2.1-I2V-14B backbone run in context: one token sequence holds
#
#     [additional refs] [reference image] [video being denoised] [driving video]
#
# and every block attends over all of it. The blocks, cross-attention, time/text/image
# embeddings and head are the Wan ones unchanged; SCAIL-2 adds
#   * ``patch_embedding_pose``: embeds the half-resolution driving-video latent,
#   * ``patch_embedding_mask``: embeds the 28-channel colour masks, added to the
#     reference tokens (reference mask) and the driving tokens (driving mask),
#   * a shifted 3D RoPE: the driving tokens sit ``pose_rope_w_shift`` positions to the
#     right of the video and use the 2x2-average-pooled RoPE of the full-resolution
#     grid; in replacement mode the reference sits ``replace_ref_rope_h_shift`` positions
#     down and the video starts at t=0 instead of t=1.
# Only the video tokens are unpatchified.
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.scail2 import SCAIL2Config
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanTransformer3DModel,
    _is_cuda,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Latent channels of the Wan2.1 VAE.
_LATENT_CHANNELS = 16
# Channels appended to every latent before the patch embedding (the i2v mask slot).
_MASK_SLOT_CHANNELS = 4


def scail2_rope_positions(
    num_frames: int,
    grid_h: int,
    grid_w: int,
    *,
    replace_mode: bool,
    num_additional_refs: int = 0,
    pose_w_shift: int = 120,
    ref_h_shift: int = 120,
) -> dict[str, torch.Tensor]:
    """(t, h, w) RoPE positions of each SCAIL-2 token group, as ``[N, 3]`` long tensors.

    ``grid_h`` / ``grid_w`` are the video's token grid (latent size / patch size). The
    driving ("pose") group is returned on the full-resolution grid; its RoPE is
    2x2-average-pooled afterwards (see :func:`pool_pose_rope`). Matches the
    ``rope_T_shift`` / ``rope_H_shift`` / ``rope_W_shift`` tables of the official model.
    """
    base_video_shift = 0 if replace_mode else 1
    h_shift = ref_h_shift if replace_mode else 0

    def grid(f: int, t0: int, h0: int, w0: int) -> torch.Tensor:
        t = torch.arange(t0, t0 + f)
        h = torch.arange(h0, h0 + grid_h)
        w = torch.arange(w0, w0 + grid_w)
        return torch.stack(torch.meshgrid(t, h, w, indexing="ij"), dim=-1).reshape(
            -1, 3
        )

    video_t0 = base_video_shift + num_additional_refs
    positions = {
        "ref": grid(1, num_additional_refs, h_shift, 0),
        "video": grid(num_frames, video_t0, 0, 0),
        "pose": grid(num_frames, video_t0, 0, pose_w_shift),
    }
    if num_additional_refs > 0:
        positions["additional_ref"] = grid(num_additional_refs, 0, h_shift, 0)
    return positions


def pool_pose_rope(
    table: torch.Tensor, num_frames: int, grid_h: int, grid_w: int
) -> torch.Tensor:
    """2x2 average pool of a full-resolution ``[T*H*W, D]`` cos or sin table to the
    driving tokens' ``[T*(H/2)*(W/2), D]``. The official model pools the real and the
    imaginary part of the complex frequencies separately, which is the same thing."""
    d = table.shape[-1]
    table = table.reshape(num_frames, grid_h, grid_w, d).permute(0, 3, 1, 2)
    table = F.avg_pool2d(table, kernel_size=2, stride=2)
    return table.permute(0, 2, 3, 1).reshape(-1, d)


def scail2_rope_cos_sin(
    rotary_emb,
    num_frames: int,
    grid_h: int,
    grid_w: int,
    *,
    replace_mode: bool,
    num_additional_refs: int,
    pose_w_shift: int,
    ref_h_shift: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)``, each ``[N, head_dim / 2]`` fp32, for the whole SCAIL-2 sequence in
    token order ``[additional refs, ref, video, pose]``; ``rotary_emb`` is the Wan
    ``NDRotaryEmbedding``."""
    positions = scail2_rope_positions(
        num_frames,
        grid_h,
        grid_w,
        replace_mode=replace_mode,
        num_additional_refs=num_additional_refs,
        pose_w_shift=pose_w_shift,
        ref_h_shift=ref_h_shift,
    )
    order = (["additional_ref"] if num_additional_refs > 0 else []) + ["ref", "video"]
    cos_parts, sin_parts = [], []
    for name in order:
        cos, sin = rotary_emb.forward_uncached(positions[name].to(device))
        cos_parts.append(cos)
        sin_parts.append(sin)
    cos, sin = rotary_emb.forward_uncached(positions["pose"].to(device))
    cos_parts.append(pool_pose_rope(cos, num_frames, grid_h, grid_w))
    sin_parts.append(pool_pose_rope(sin, num_frames, grid_h, grid_w))
    return (
        torch.cat(cos_parts).float().contiguous(),
        torch.cat(sin_parts).float().contiguous(),
    )


class SCAIL2Transformer3DModel(WanTransformer3DModel):
    """SCAIL-2 14B in-context motion-transfer DiT on the Wan2.1-I2V-14B backbone.

    Attention goes through the Wan blocks' USPAttention, so callers must wrap the
    forward in ``set_forward_context(...)``.
    """

    param_names_mapping = SCAIL2Config().param_names_mapping
    reverse_param_names_mapping = SCAIL2Config().reverse_param_names_mapping
    lora_param_names_mapping = SCAIL2Config().lora_param_names_mapping

    def __init__(
        self,
        config: SCAIL2Config,
        hf_config: dict[str, Any] | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(
            config=config, hf_config=hf_config or {}, quant_config=quant_config
        )
        inner_dim = config.num_attention_heads * config.attention_head_dim
        if config.in_channels != _LATENT_CHANNELS + _MASK_SLOT_CHANNELS:
            raise ValueError(
                "SCAIL-2 expects in_channels=20 (16 latent + 4 mask channels), got "
                f"{config.in_channels}"
            )
        self.patch_embedding_pose = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )
        self.patch_embedding_mask = PatchEmbed(
            in_chans=config.mask_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )
        self.pose_rope_w_shift = config.pose_rope_w_shift
        self.replace_ref_rope_h_shift = config.replace_ref_rope_h_shift
        self._warned_sp_padding = False

    # ------------------------------------------------------------------ RoPE

    @lru_cache(maxsize=4)
    def _scail2_rope(
        self,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        replace_mode: bool,
        num_additional_refs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """See :func:`scail2_rope_cos_sin`. Cached: the geometry is the same for every
        step of a segment and usually for every segment."""
        return scail2_rope_cos_sin(
            self.rotary_emb,
            num_frames,
            grid_h,
            grid_w,
            replace_mode=replace_mode,
            num_additional_refs=num_additional_refs,
            pose_w_shift=self.pose_rope_w_shift,
            ref_h_shift=self.replace_ref_rope_h_shift,
            device=device,
        )

    # ------------------------------------------------------------- embedding

    @staticmethod
    def _with_mask_slot(latent: torch.Tensor, fill: float) -> torch.Tensor:
        """``[B, 16, T, H, W]`` -> ``[B, 20, T, H, W]`` with the 4 mask channels set to ``fill``."""
        b, _, t, h, w = latent.shape
        slot = latent.new_full((b, _MASK_SLOT_CHANNELS, t, h, w), fill)
        return torch.cat([latent, slot], dim=1)

    def _embed_tokens(
        self,
        hidden_states: torch.Tensor,
        reference_latent: torch.Tensor,
        reference_mask: torch.Tensor,
        pose_latent: torch.Tensor,
        driving_mask: torch.Tensor,
        history_mask: torch.Tensor | None,
        additional_ref_latent: torch.Tensor | None,
        additional_ref_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        """Token sequence ``[B, N, C]`` and the token count ahead of the video tokens."""
        dtype = self.patch_embedding.proj.weight.dtype
        # video: noisy latent + history mask (zeros when there is no history)
        if history_mask is None:
            video = self._with_mask_slot(hidden_states, 0.0)
        else:
            video = torch.cat([hidden_states, history_mask.to(hidden_states)], dim=1)
        reference = self._with_mask_slot(reference_latent.to(hidden_states), 1.0)
        pose = self._with_mask_slot(pose_latent.to(hidden_states), 1.0)

        # [ref, video] share the patch embedding; the reference mask covers the reference
        # frame and is zero over the video frames.
        ref_and_video = torch.cat([reference, video], dim=2).to(dtype)
        num_video_frames = hidden_states.shape[2]
        mask = torch.cat(
            [
                reference_mask,
                reference_mask.new_zeros(
                    (
                        reference_mask.shape[0],
                        reference_mask.shape[1],
                        num_video_frames,
                        *reference_mask.shape[3:],
                    )
                ),
            ],
            dim=2,
        ).to(dtype)
        tokens = self.patch_embedding(ref_and_video, tokens=True) + (
            self.patch_embedding_mask(mask, tokens=True)
        )
        pose_tokens = self.patch_embedding_pose(pose.to(dtype), tokens=True) + (
            self.patch_embedding_mask(driving_mask.to(dtype), tokens=True)
        )
        parts = [tokens, pose_tokens]
        num_prefix_tokens = 0
        if additional_ref_latent is not None:
            additional = self._with_mask_slot(
                additional_ref_latent.to(hidden_states), 1.0
            ).to(dtype)
            additional_tokens = self.patch_embedding(additional, tokens=True) + (
                self.patch_embedding_mask(additional_ref_mask.to(dtype), tokens=True)
            )
            parts.insert(0, additional_tokens)
            num_prefix_tokens += additional_tokens.shape[1]
        p_t, p_h, p_w = self.patch_size
        num_prefix_tokens += (
            reference.shape[2] // p_t * (reference.shape[3] // p_h) * (
                reference.shape[4] // p_w
            )
        )
        return torch.cat(parts, dim=1).contiguous(), num_prefix_tokens

    # --------------------------------------------------------------- forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        *,
        reference_latent: torch.Tensor,
        reference_mask: torch.Tensor,
        pose_latent: torch.Tensor,
        driving_mask: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        additional_ref_latent: torch.Tensor | None = None,
        additional_ref_mask: torch.Tensor | None = None,
        replace_mode: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Predict the flow for the video latent.

        Args:
            hidden_states: noisy video latent ``[B, 16, T, H, W]`` (history frames already
                replaced by their clean latents).
            encoder_hidden_states: T5 embeddings ``[B, 512, 4096]`` (zero padded).
            timestep: ``[B]``.
            encoder_hidden_states_image: CLIP ``hidden_states[-2]`` of the reference image,
                ``[B, 257, 1280]``.
            reference_latent: ``[B, 16, 1, H, W]``.
            reference_mask: ``[B, 28, 1, H, W]`` compressed colour mask of the reference.
            pose_latent: driving-video latent at half resolution, ``[B, 16, T, H/2, W/2]``.
            driving_mask: ``[B, 28, T, H/2, W/2]`` compressed colour mask of the driving video.
            history_mask: ``[B, 4, T, H, W]``, ones on the clean history frames; None for
                the first segment.
            additional_ref_latent / additional_ref_mask: ``[B, 16 | 28, N_add, H, W]``.
            replace_mode: replacement (True) or animation (False) RoPE layout.

        Returns:
            ``[B, 16, T, H, W]`` flow prediction for the video tokens only.
        """
        if (additional_ref_latent is None) != (additional_ref_mask is None):
            raise ValueError(
                "additional_ref_latent and additional_ref_mask must be given together"
            )
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image, list):
            encoder_hidden_states_image = (
                encoder_hidden_states_image[0] if encoder_hidden_states_image else None
            )
        if encoder_hidden_states_image is None:
            raise ValueError("SCAIL-2 needs the CLIP embedding of the reference image")

        orig_dtype = hidden_states.dtype
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        grid = (num_frames // p_t, height // p_h, width // p_w)
        if grid[1] % 2 or grid[2] % 2:
            raise ValueError(
                "SCAIL-2 needs an even token grid (height and width multiples of 32 "
                f"pixels); got a {grid[1]}x{grid[2]} token grid"
            )
        expected_pose = (grid[1] // 2 * p_h, grid[2] // 2 * p_w)
        if tuple(pose_latent.shape[3:]) != expected_pose:
            raise ValueError(
                f"pose_latent must be half the video latent size {expected_pose}, got "
                f"{tuple(pose_latent.shape[3:])}"
            )
        num_additional_refs = (
            0 if additional_ref_latent is None else additional_ref_latent.shape[2] // p_t
        )

        hidden_states, video_start = self._embed_tokens(
            hidden_states,
            reference_latent,
            reference_mask,
            pose_latent,
            driving_mask,
            history_mask,
            additional_ref_latent,
            additional_ref_mask,
        )
        num_video_tokens = grid[0] * grid[1] * grid[2]
        seq_len = hidden_states.shape[1]

        freqs_cos, freqs_sin = self._scail2_rope(
            grid[0],
            grid[1],
            grid[2],
            bool(replace_mode),
            num_additional_refs,
            hidden_states.device,
        )
        if freqs_cos.shape[0] != seq_len:
            raise RuntimeError(
                f"SCAIL-2 RoPE covers {freqs_cos.shape[0]} tokens but the sequence has "
                f"{seq_len}"
            )

        # Ulysses / ring SP: shard the whole in-context sequence; each block's
        # USPAttention exchanges the shards, so every token still sees every token.
        sp_size = get_sp_world_size()
        if sp_size > 1:
            if seq_len % sp_size and not self._warned_sp_padding:
                logger.warning(
                    "SCAIL-2 sequence length %d is not divisible by the SP degree %d; "
                    "the tail is zero padded, which perturbs attention slightly.",
                    seq_len,
                    sp_size,
                )
                self._warned_sp_padding = True
            hidden_states, shard = shard_seq(hidden_states, dim=1)
            freqs_cos = shard_like(freqs_cos, shard, dim=0, pad_mode="repeat_last")
            freqs_sin = shard_like(freqs_sin, shard, dim=0, pad_mode="repeat_last")

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1)).float()
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states_image, encoder_hidden_states], dim=1
        )
        if not current_platform.is_amp_supported():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)

        hidden_states = self._run_blocks(
            hidden_states, encoder_hidden_states, timestep_proj, freqs_cos, freqs_sin
        )

        if sp_size > 1:
            hidden_states = gather_seq(hidden_states, seq_len, dim=1)

        # Only the video tokens are denoised; the head is per token, so slice first.
        hidden_states = hidden_states[:, video_start : video_start + num_video_tokens]
        return self._unpatchify(hidden_states, temb, grid)

    def _run_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """The Wan block loop with explicit (cos, sin) tables."""
        freqs_cis = (freqs_cos, freqs_sin)
        complex_freqs = torch.complex(freqs_cos, freqs_sin).unsqueeze(-2)
        rope_cos_sin_cache = (
            torch.cat([freqs_cos, freqs_sin], dim=-1).contiguous() if _is_cuda else None
        )
        block_kwargs = dict(
            rope_cos_sin_cache=rope_cos_sin_cache, complex_freqs=complex_freqs
        )
        if self._chains_ffn_residuals():
            # each block's FFN residual is added by the next block's norm1
            ffn_residual = None
            for block in self.blocks[:-1]:
                hidden_states, ffn_residual = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    ffn_residual=ffn_residual,
                    defer_ffn_residual=True,
                    **block_kwargs,
                )
            return self.blocks[-1](
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                ffn_residual=ffn_residual,
                **block_kwargs,
            )
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                **block_kwargs,
            )
        return hidden_states


EntryClass = SCAIL2Transformer3DModel
