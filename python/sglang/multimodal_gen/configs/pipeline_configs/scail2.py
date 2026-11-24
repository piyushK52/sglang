# SPDX-License-Identifier: Apache-2.0
"""SCAIL-2 14B pipeline config (https://github.com/zai-org/SCAIL-2)."""

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits.scail2 import SCAIL2Config
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanI2V480PConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType


@dataclass
class SCAIL2_14B_Config(WanI2V480PConfig):
    """SCAIL-2 14B: reference image + driving video (+ colour masks) -> video.

    Shares the Wan2.1 I2V component bundle (UMT5, CLIP ViT-H, Wan2.1 VAE with the encoder
    loaded) and swaps in the SCAIL-2 DiT. The bespoke stages read ``flow_shift`` as the
    default sampling shift (the reference ``generate.py`` uses 3.0; a request overrides
    it with ``sample_shift``).
    """

    task_type: ModelTaskType = ModelTaskType.I2V
    dit_config: DiTConfig = field(default_factory=SCAIL2Config)
    flow_shift: float | None = 3.0
    # The VAE adapter runs encode and decode in fp32 like the reference WanVAE.
    vae_decode_precision: str = "fp32"

    def get_decode_scale_and_shift(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
        vae: torch.nn.Module | None,
    ) -> tuple[float, None]:
        """``(1.0, None)``: the VAE adapter applies the latent scaling itself."""
        return 1.0, None

    def supports_disaggregation(self) -> bool:
        # The stages share one request-state object (GPU tensors and a torch.Generator)
        # through batch.extra, which the disaggregation transport does not forward.
        return False

    def validate_server_args(self, server_args: Any) -> None:
        attention_backend = (server_args.attention_backend or "").lower()
        if attention_backend == "video_sparse_attn":
            raise ValueError(
                "SCAIL-2 does not support --attention-backend video_sparse_attn: its "
                "token sequence is not a single video grid."
            )
        num_heads = self.dit_config.num_attention_heads
        if num_heads % server_args.tp_size != 0:
            raise ValueError(
                f"SCAIL-2 tensor parallelism needs the {num_heads} attention heads "
                f"divisible by --tp-size {server_args.tp_size}."
            )
        local_num_heads = num_heads // server_args.tp_size
        if local_num_heads % server_args.ulysses_degree != 0:
            raise ValueError(
                f"SCAIL-2 Ulysses SP needs the TP-local head count ({local_num_heads}) "
                f"divisible by --ulysses-degree {server_args.ulysses_degree}."
            )
        super().validate_server_args(server_args)


def _is_scail2_path(path: str) -> bool:
    # Registry probes are lower-cased paths and model_index.json class names.
    return path == "scail2pipeline" or "scail-2" in path or "scail2" in path


def register():
    from sglang.multimodal_gen.configs.sample.scail2 import SCAIL2_14B_SamplingParams
    from sglang.multimodal_gen.registry import register_configs

    # There is no published Diffusers repository; the local directory built by
    # tools/build_scail2_model.py ships model_index.json with
    # _class_name == "SCAIL2Pipeline", which the detector matches.
    register_configs(
        sampling_param_cls=SCAIL2_14B_SamplingParams,
        pipeline_config_cls=SCAIL2_14B_Config,
        model_detectors=[_is_scail2_path],
    )
