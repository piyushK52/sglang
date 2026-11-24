# SPDX-License-Identifier: Apache-2.0
#
# SCAIL-2 DiT config (https://github.com/zai-org/SCAIL-2).
#
# SCAIL-2 is the Wan2.1-I2V-14B backbone with two extra patch embeddings: a pose
# embedding for the driving video and a mask embedding for the 28-channel colour masks.
# The released checkpoint (zai-org/SCAIL-2, Comfy-Org/SCAIL-2) uses the official Wan
# parameter names, so the Wan-Animate-2 official-name rules are reused and the two new
# embeddings are added in front of them.
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.wan_animate_2 import (
    _WAN_ANIMATE_2_EXTRA_MAPPING,
)
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig

_WAN = WanVideoArchConfig()

_SCAIL2_EXTRA_MAPPING: dict[str, str] = {
    # The two SCAIL-2 embeddings are plain Conv3d in the checkpoint and PatchEmbed here.
    r"^patch_embedding_pose\.(weight|bias)$": r"patch_embedding_pose.proj.\1",
    r"^patch_embedding_mask\.(weight|bias)$": r"patch_embedding_mask.proj.\1",
}


@dataclass
class SCAIL2ArchConfig(WanVideoArchConfig):
    """Wan2.1-I2V-14B dims plus the SCAIL-2 pose / mask embeddings."""

    # SCAIL-2 extras, then the official-name rules, then Wan lora (official -> hf), then
    # Wan main (hf -> native); the loader chains them in insertion order.
    param_names_mapping: dict[str, str] = field(
        default_factory=lambda: {
            **_SCAIL2_EXTRA_MAPPING,
            **_WAN_ANIMATE_2_EXTRA_MAPPING,
            **_WAN.lora_param_names_mapping,
            **_WAN.param_names_mapping,
        }
    )
    reverse_param_names_mapping: dict[str, str] = field(
        default_factory=lambda: dict(_WAN.reverse_param_names_mapping)
    )
    lora_param_names_mapping: dict[str, str] = field(
        default_factory=lambda: dict(_WAN.lora_param_names_mapping)
    )

    # in_channels: 16 latent + 4 mask channels (history mask for the video, ones for the
    # reference and pose). The pose embedding takes the same 20 channels.
    in_channels: int = 20
    out_channels: int = 16
    # 7 colours x 4 frames per latent frame (official ``mask_dim``).
    mask_channels: int = 28
    image_dim: int | None = 1280
    added_kv_proj_dim: int | None = 5120
    num_layers: int = 40

    # RoPE offsets of the official model: pose tokens sit 120 positions to the right of
    # the video along w; in replacement mode the reference sits 120 positions down along h.
    pose_rope_w_shift: int = 120
    replace_ref_rope_h_shift: int = 120

    def __post_init__(self) -> None:
        # A config.json written with the official Wan names (in_dim, dim, num_heads, ...)
        # lands in extra_attrs; translate it like Wan-Animate-2 does.
        extras = self.extra_attrs
        if "num_heads" in extras:
            self.num_attention_heads = extras.pop("num_heads")
        if "dim" in extras:
            self.attention_head_dim = extras.pop("dim") // self.num_attention_heads
        if "in_dim" in extras:
            self.in_channels = extras.pop("in_dim")
        if "out_dim" in extras:
            self.out_channels = extras.pop("out_dim")
        if "mask_dim" in extras:
            self.mask_channels = extras.pop("mask_dim")
        self.patch_size = tuple(self.patch_size)
        super().__post_init__()


@dataclass
class SCAIL2Config(DiTConfig):
    arch_config: SCAIL2ArchConfig = field(default_factory=SCAIL2ArchConfig)
    prefix: str = "SCAIL2"
