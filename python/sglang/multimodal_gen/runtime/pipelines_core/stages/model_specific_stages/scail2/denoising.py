# SPDX-License-Identifier: Apache-2.0
"""SCAIL-2 denoising and output stages.

SCAIL2DenoisingStage runs the official segment loop
(https://github.com/zai-org/SCAIL-2/blob/main/wan/scail.py, ``generate``) on the shared
denoising skeleton: per segment it encodes the half-resolution driving video and its
mask, encodes the previous segment's last ``segment_overlap`` decoded frames as clean
history (pinned into the latent before every step and after every scheduler step),
denoises with UniPC and cond/uncond CFG, decodes the segment and drops the overlap.
SCAIL2OutputStage emits the assembled frames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import model_parallel_is_initialized
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    run_cfg_parallel,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
    CFGBranch,
    CFGPolicy,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.scail2.before_denoising import (
    SCAIL2_REQUEST_STATE_EXTRA_KEY,
    SCAIL2RequestState,
    make_scheduler_timesteps,
    request_state_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.scail2.preprocess import (
    VAE_SPATIAL_STRIDE,
    compress_mask_to_latent,
    latent_frames,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
    WanAnimate2VaeAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

logger = init_logger(__name__)

# Channels of the history mask the DiT appends to the noisy latent.
_HISTORY_MASK_CHANNELS = 4


def _is_request_state(value: object) -> bool:
    return isinstance(value, SCAIL2RequestState)


def _is_decoded_frames(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 4 and value.dtype == np.uint8


def _to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """``[3, T, H, W]`` in [-1, 1] -> ``[T, H, W, 3]`` uint8."""
    return (
        rearrange((video.clamp(-1, 1) + 1) * 127.5, "c t h w -> t h w c")
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


@dataclass(frozen=True, kw_only=True)
class SCAIL2SegmentConditioning:
    """DiT kwargs of one segment that do not change from step to step."""

    reference_latent: torch.Tensor
    reference_mask: torch.Tensor
    pose_latent: torch.Tensor  # [1, 16, t, h/2, w/2]
    driving_mask: torch.Tensor  # [1, 28, t, h/2, w/2]
    history_mask: torch.Tensor | None  # [1, 4, t, h, w]
    additional_ref_latent: torch.Tensor | None
    additional_ref_mask: torch.Tensor | None
    replace_mode: bool

    def dit_kwargs(self) -> dict:
        return {
            "reference_latent": self.reference_latent,
            "reference_mask": self.reference_mask,
            "pose_latent": self.pose_latent,
            "driving_mask": self.driving_mask,
            "history_mask": self.history_mask,
            "additional_ref_latent": self.additional_ref_latent,
            "additional_ref_mask": self.additional_ref_mask,
            "replace_mode": self.replace_mode,
        }


@dataclass(kw_only=True)
class SCAIL2DenoisingContext(DenoisingContext):
    conditioning: SCAIL2SegmentConditioning
    image_embeds: torch.Tensor
    history_latent: torch.Tensor | None
    cfg_parallel: bool
    guidance_scale: float


def apply_clean_history(
    latent: torch.Tensor, history_latent: torch.Tensor | None
) -> torch.Tensor:
    """Overwrite the leading latent frames of ``[16, T, h, w]`` with the clean history."""
    if history_latent is None:
        return latent
    latent = latent.clone()
    latent[:, : history_latent.shape[1]] = history_latent.to(latent)
    return latent


class SCAIL2DenoisingStage(DenoisingStage):
    """The SCAIL-2 segment loop: condition, denoise, decode and stitch every segment."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        scheduler: FlowUniPCMultistepScheduler,
        pipeline: ComposedPipelineBase,
        vae: WanAnimate2VaeAdapter,
    ) -> None:
        super().__init__(
            transformer=transformer, scheduler=scheduler, pipeline=pipeline, vae=vae
        )
        self.vae: WanAnimate2VaeAdapter = vae
        self.segment_decoder = DecodingStage(vae, pipeline=pipeline)

    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        with self.use_declared_component(component_name="vae", module=self.vae.vae):
            return self.vae.encode([video])[0]

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(4)])
        result.add_check(
            "prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3)
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            V.list_of_tensors_dims(3),
        )
        result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check(
            SCAIL2_REQUEST_STATE_EXTRA_KEY,
            batch.extra.get(SCAIL2_REQUEST_STATE_EXTRA_KEY),
            _is_request_state,
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        state = batch.extra.get(SCAIL2_REQUEST_STATE_EXTRA_KEY)
        result.add_check(
            "decoded_frames",
            state.decoded_frames if isinstance(state, SCAIL2RequestState) else None,
            _is_decoded_frames,
        )
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        # The VAE adapter runs the VAE in fp32 with autocast off; do not let the manager
        # cast it to the configured vae precision on the way in.
        return [
            replace(use, target_dtype=None) if use.component_name == "vae" else use
            for use in super().component_uses(server_args, stage_name)
        ]

    # ------------------------------------------------------------ segments

    def _segment_conditioning(
        self,
        state: SCAIL2RequestState,
        segment: tuple[int, int],
        history_pixels: torch.Tensor | None,
    ) -> tuple[SCAIL2SegmentConditioning, torch.Tensor | None]:
        """Driving-video latent, driving mask and history of one segment, as the reference
        builds them (half-resolution bilinear resize before the VAE and the mask
        compression)."""
        device = state.device
        start, end = segment
        # [t, 3, H, W] uint8 -> [-1, 1], then half resolution
        pose = (state.driving_video[start:end].to(device).float() - 127.5) / 127.5
        pose = F.interpolate(pose, scale_factor=0.5, mode="bilinear", align_corners=False)
        pose_latent = self._vae_encode(rearrange(pose, "t c h w -> c t h w"))
        num_latent_frames = pose_latent.shape[1]

        # [3, t, H, W] uint8 -> [-1, 1], half resolution, 28-channel latent mask
        mask = (state.driving_mask[:, start:end].to(device).float() - 127.5) / 127.5
        mask = F.interpolate(mask, scale_factor=0.5, mode="bilinear", align_corners=False)
        driving_mask = compress_mask_to_latent(mask)

        history_latent = None
        history_mask = None
        if history_pixels is not None:
            # the reference casts the decoded frames to the bf16 param dtype first
            history_latent = self._vae_encode(
                history_pixels.to(device=device, dtype=torch.bfloat16).float()
            )
            history_t = min(history_latent.shape[1], num_latent_frames)
            history_mask = torch.zeros(
                _HISTORY_MASK_CHANNELS,
                num_latent_frames,
                state.height // VAE_SPATIAL_STRIDE,
                state.width // VAE_SPATIAL_STRIDE,
                device=device,
                dtype=torch.float32,
            )
            history_mask[:, :history_t] = 1
            history_mask = history_mask.unsqueeze(0)

        conditioning = SCAIL2SegmentConditioning(
            reference_latent=state.reference_latent,
            reference_mask=state.reference_mask,
            pose_latent=pose_latent.unsqueeze(0),
            driving_mask=driving_mask.unsqueeze(0),
            history_mask=history_mask,
            additional_ref_latent=state.additional_ref_latent,
            additional_ref_mask=state.additional_ref_mask,
            replace_mode=state.inputs.replace_mode,
        )
        return conditioning, history_latent

    @torch.no_grad()
    def _denoise_segment(
        self,
        batch: Req,
        server_args: ServerArgs,
        state: SCAIL2RequestState,
        conditioning: SCAIL2SegmentConditioning,
        history_latent: torch.Tensor | None,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise one segment; returns the clean ``[16, t, h, w]`` latent."""
        inputs = state.inputs
        timesteps = make_scheduler_timesteps(
            self.scheduler, inputs.num_inference_steps, inputs.sample_shift, state.device
        )
        guidance_scale = inputs.guidance_scale
        perform_cfg = guidance_scale > 1.0
        cfg_parallel = (
            perform_cfg
            and bool(server_args.enable_cfg_parallel)
            and model_parallel_is_initialized()
        )
        if cfg_parallel and get_classifier_free_guidance_world_size() != 2:
            logger.warning_once(
                "CFG parallel enabled but the CFG group world size is "
                f"{get_classifier_free_guidance_world_size()} (SCAIL-2 expects 2); "
                "using the sequential two-pass CFG path."
            )
            cfg_parallel = False
        branches = [
            CFGBranch(
                "conditional", True, {"encoder_hidden_states": state.prompt_embeds}
            )
        ]
        if perform_cfg:
            branches.append(
                CFGBranch(
                    "unconditional",
                    False,
                    {"encoder_hidden_states": state.negative_prompt_embeds},
                )
            )

        ctx = SCAIL2DenoisingContext(
            scheduler=self.scheduler,
            extra_step_kwargs={},
            target_dtype=torch.bfloat16,
            autocast_enabled=True,
            timesteps=timesteps,
            num_inference_steps=inputs.num_inference_steps,
            num_warmup_steps=0,
            image_kwargs={},
            pos_cond_kwargs={},
            neg_cond_kwargs={},
            latents=apply_clean_history(noise, history_latent),
            boundary_timestep=None,
            z=None,
            reserved_frames_mask=None,
            seq_len=None,
            guidance=None,
            is_warmup=batch.is_warmup,
            cfg_policy=CFGPolicy(branches=branches),
            # UniPC advances once per timestep
            extra={"progress_step_interval": 1},
            conditioning=conditioning,
            image_embeds=state.image_embeds,
            history_latent=history_latent,
            cfg_parallel=cfg_parallel,
            guidance_scale=guidance_scale,
        )
        # segment trajectories are not part of the assembled-video output
        self._run_denoising_loop(ctx, batch, server_args, collect_trajectory=False)
        return ctx.latents

    def _before_denoising_loop(
        self, ctx: SCAIL2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        # set_timesteps in _denoise_segment already reset the UniPC state.
        self.begin_declared_component_use(
            component_name="transformer", module=self.transformer
        )

    def _prepare_step_state(
        self,
        ctx: SCAIL2DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_host: torch.Tensor,
        timesteps_cpu: torch.Tensor,
    ) -> DenoisingStepState:
        return DenoisingStepState(
            step_index=step_index,
            t_host=t_host,
            t_device=ctx.timesteps[step_index],
            t_int=int(t_host.item()),
            current_model=self.transformer,
            current_guidance_scale=ctx.guidance_scale,
            attn_metadata=None,
        )

    def _run_denoising_step(
        self,
        ctx: SCAIL2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        latent_model_input = apply_clean_history(ctx.latents, ctx.history_latent)
        timestep = step.t_device.reshape(1)
        dit_kwargs = ctx.conditioning.dit_kwargs()

        def predict(branch: CFGBranch) -> torch.Tensor:
            with set_forward_context(
                current_timestep=step.step_index,
                attn_metadata=None,
                forward_batch=batch,
            ):
                prediction = self.transformer(
                    hidden_states=latent_model_input.unsqueeze(0),
                    timestep=timestep,
                    encoder_hidden_states_image=ctx.image_embeds,
                    **branch.kwargs,
                    **dit_kwargs,
                )[0].float()
            return prediction.contiguous() if ctx.cfg_parallel else prediction

        predictions = (
            run_cfg_parallel(ctx.cfg_policy, predict)
            if ctx.cfg_parallel
            else [predict(branch) for branch in ctx.cfg_policy.branches]
        )
        noise_pred = predictions[0]
        if len(predictions) == 2:
            noise_pred = predictions[1] + step.current_guidance_scale * (
                noise_pred - predictions[1]
            )
        latents = ctx.scheduler.step(
            noise_pred.unsqueeze(0),
            step.t_device,
            ctx.latents.unsqueeze(0),
            return_dict=False,
        )[0].squeeze(0)
        ctx.latents = apply_clean_history(latents, ctx.history_latent)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state = request_state_from_batch(batch)
        overlap = state.inputs.segment_overlap
        output_frames: list[np.ndarray] = []
        history_pixels: torch.Tensor | None = None  # [3, overlap, H, W] in [-1, 1]

        # The DiT runs in bf16 autocast like the reference (timestep/norm math in fp32).
        with torch.autocast(
            device_type=current_platform.device_type, dtype=torch.bfloat16
        ):
            for segment_index, segment in enumerate(state.segments):
                if segment_index > 0:
                    self.log_info(
                        "segment %d/%d: frames [%d, %d)",
                        segment_index + 1,
                        len(state.segments),
                        *segment,
                    )
                conditioning, history_latent = self._segment_conditioning(
                    state, segment, history_pixels
                )
                # Segment 0's noise is the generator's first draw (batch.latents); the
                # next segments take the following draws in order, like the reference.
                if segment_index == 0:
                    noise = state.segment_0_noise
                else:
                    noise = torch.randn(
                        state.segment_0_noise.shape[0],
                        latent_frames(segment[1] - segment[0]),
                        *state.segment_0_noise.shape[2:],
                        dtype=torch.float32,
                        device=state.device,
                        generator=state.generator,
                    )
                latents = self._denoise_segment(
                    batch, server_args, state, conditioning, history_latent, noise
                )
                with self.use_declared_component(
                    component_name="vae", module=self.vae.vae
                ):
                    # [1, 3, t, H, W] fp32 in [-1, 1]
                    video = self.segment_decoder.decode_raw(
                        latents.float().unsqueeze(0),
                        server_args,
                        vae_dtype=torch.float32,
                    )[0]
                frames = _to_uint8_frames(video)
                output_frames.extend(frames if segment_index == 0 else frames[overlap:])
                history_pixels = (
                    video[:, -overlap:].contiguous()
                    if segment_index < len(state.segments) - 1
                    else None
                )
                del conditioning, history_latent, latents, video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        batch.extra.pop(DENOISING_LOOP_STATE, None)
        state.decoded_frames = np.stack(output_frames, axis=0)
        return batch


class SCAIL2OutputStage(PipelineStage):
    """Terminal stage: wrap the already-decoded, assembled frames in an OutputBatch.

    Replaces the standard DecodingStage, which would decode a single-segment latent.
    """

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DECODER

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        state = batch.extra.get(SCAIL2_REQUEST_STATE_EXTRA_KEY)
        result.add_check(
            "decoded_frames",
            state.decoded_frames if isinstance(state, SCAIL2RequestState) else None,
            _is_decoded_frames,
        )
        return result

    def verify_output(
        self, batch: OutputBatch, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.list_not_empty)
        return result

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        state = request_state_from_batch(batch)
        if state.decoded_frames is None:
            raise ValueError(
                "SCAIL2OutputStage: SCAIL2DenoisingStage did not assemble the frames."
            )
        # [T, H, W, C] uint8, consumed directly by the save path; the driving video's
        # audio is not carried (the reference drops it too).
        return OutputBatch(output=[state.decoded_frames], metrics=batch.metrics)
