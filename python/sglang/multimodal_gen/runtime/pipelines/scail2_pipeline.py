# SPDX-License-Identifier: Apache-2.0
"""SCAIL-2 pipeline (https://github.com/zai-org/SCAIL-2).

Character animation / replacement on Wan2.1-I2V-14B: a reference image (plus its
colour mask) performs the motion of a driving video (plus its per-frame colour mask).

Stages: SCAIL2BeforeDenoisingStage (text, CLIP, reference VAE latent, masks, segment plan)
-> SCAIL2DenoisingStage (per segment: driving latents, clean history, UniPC denoise,
in-loop VAE decode) -> SCAIL2OutputStage (emit frames).

Routed by model_index.json ``_class_name == "SCAIL2Pipeline"``. The checkpoint directory
is the Wan2.1-I2V-14B-480P-Diffusers components with the SCAIL-2 transformer; build it
with ``python -m sglang.multimodal_gen.tools.build_scail2_model``.
"""

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    SchedulerLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.scail2 import (
    SCAIL2BeforeDenoisingStage,
    SCAIL2DenoisingStage,
    SCAIL2OutputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
    WanAnimate2ImageEncoderAdapter,
    WanAnimate2VaeAdapter,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SCAIL2TransformerLoader(TransformerLoader):
    """Every checkpoint key must map: a Wan I2V checkpoint without the SCAIL-2 pose and
    mask embeddings would otherwise load and run with random embeddings."""

    # Pipeline-local: an empty component_names keeps this subclass out of the shared
    # loader registry.
    component_names = []
    strict_checkpoint_keys = True


class SCAIL2SchedulerLoader(SchedulerLoader):
    """The reference samples with FlowUniPCMultistepScheduler(shift=1) and applies the
    sampling shift in ``set_timesteps``; the Wan I2V checkpoint's own scheduler config
    (diffusers UniPC with flow_shift baked in) is not read."""

    component_names = []

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        override = server_args.pipeline_config.scheduler_class_override
        if override:
            logger.warning(
                "--scheduler-class-override %s is ignored for SCAIL-2, which samples "
                "with the reference FlowUniPCMultistepScheduler.",
                override,
            )
        return FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False
        )


class SCAIL2Pipeline(ComposedPipelineBase):
    """SCAIL-2 14B in-context motion transfer: reference image + driving video -> video."""

    pipeline_name = "SCAIL2Pipeline"
    is_video_pipeline = True

    _required_config_modules: list[str] = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "image_encoder",
        "image_processor",
        "scheduler",
    ]
    component_loaders = {
        "transformer": SCAIL2TransformerLoader,
        "scheduler": SCAIL2SchedulerLoader,
    }

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # The adapters keep the official Wan CLIP / VAE numerics (see Wan-Animate-2) over
        # the natively loaded modules, which stay visible to residency management.
        image_encoder = WanAnimate2ImageEncoderAdapter.from_loaded(
            self.get_module("image_encoder"),
            image_processor=self.get_module("image_processor"),
            device=get_local_torch_device(),
        )
        vae = WanAnimate2VaeAdapter(self.get_module("vae"))
        self.add_stage(
            stage_name="scail2_before_denoising",
            stage=SCAIL2BeforeDenoisingStage(
                vae=vae,
                image_encoder=image_encoder,
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
                default_shift=server_args.pipeline_config.flow_shift,
            ),
        )
        self.add_stage(
            stage_name="scail2_denoising",
            stage=SCAIL2DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
                vae=vae,
            ),
        )
        # Replaces the standard DecodingStage: segments are decoded in the loop.
        self.add_stage_factory(
            RoleType.DECODER,
            lambda: SCAIL2OutputStage(),
            "decoding_stage",
        )


EntryClass = SCAIL2Pipeline
