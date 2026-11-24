# SPDX-License-Identifier: Apache-2.0
"""SCAIL-2 before-denoising stage.

Mirrors the segment-independent half of the official ``SCAIL2Pipeline.generate``
(https://github.com/zai-org/SCAIL-2/blob/main/wan/scail.py) and ``generate.py``:
loads and centre-crops the reference image, its mask, the driving video and its mask,
encodes the prompts (T5), the reference image (CLIP + VAE) and the masks, and plans the
segments. The per-segment work (driving-video latents, history) runs in the denoising
stage, which needs the previous segment's decoded frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.scail2.preprocess import (
    VAE_SPATIAL_STRIDE,
    build_segments,
    check_resolution,
    compress_mask_to_latent,
    latent_frames,
    load_image,
    orient_target_size,
    read_video_frames,
    resize_for_rectangle_crop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.preprocess import (
    validate_and_get_single_string,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
        WanAnimate2ImageEncoderAdapter,
        WanAnimate2VaeAdapter,
    )

_LATENT_CHANNELS = 16


@dataclass(frozen=True, kw_only=True)
class SCAIL2Inputs:
    """The per-request inputs read off the Req."""

    reference_image_path: str
    reference_mask_path: str
    driving_video_path: str
    driving_mask_path: str
    additional_ref_image_paths: tuple[str, ...]
    additional_ref_mask_paths: tuple[str, ...]

    prompt: str
    negative_prompt: str

    width: int
    height: int
    replace_mode: bool
    segment_len: int
    segment_overlap: int
    seed: int
    num_inference_steps: int
    guidance_scale: float
    sample_shift: float

    @classmethod
    def from_request(cls, batch: Req, default_shift: float) -> SCAIL2Inputs:
        def paths(value, name: str) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                return (value,)
            if not all(isinstance(v, str) for v in value):
                raise ValueError(f"SCAIL-2 '{name}' must be a list of paths")
            return tuple(value)

        additional_images = paths(
            batch.additional_ref_image_paths, "additional_ref_image_paths"
        )
        additional_masks = paths(
            batch.additional_ref_mask_paths, "additional_ref_mask_paths"
        )
        if len(additional_images) != len(additional_masks):
            raise ValueError(
                "SCAIL-2 needs one mask per additional reference image; got "
                f"{len(additional_images)} images and {len(additional_masks)} masks"
            )
        reference_mask_path = batch.reference_mask_path
        driving_mask_path = batch.driving_mask_path
        if not reference_mask_path or not driving_mask_path:
            raise ValueError(
                "SCAIL-2 needs reference_mask_path (colour-coded reference mask image) "
                "and driving_mask_path (colour-coded driving mask video)"
            )

        seed = batch.seed
        if isinstance(seed, list):
            if len(seed) != 1:
                raise ValueError(
                    f"SCAIL-2 takes exactly one 'seed' per request, got {len(seed)}"
                )
            seed = seed[0]
        sample_shift = batch.sample_shift
        return cls(
            reference_image_path=validate_and_get_single_string(
                batch.image_path, "image_path"
            ),
            reference_mask_path=validate_and_get_single_string(
                reference_mask_path, "reference_mask_path"
            ),
            driving_video_path=validate_and_get_single_string(
                batch.video_path, "video_path"
            ),
            driving_mask_path=validate_and_get_single_string(
                driving_mask_path, "driving_mask_path"
            ),
            additional_ref_image_paths=additional_images,
            additional_ref_mask_paths=additional_masks,
            prompt=validate_and_get_single_string(batch.prompt, "prompt"),
            negative_prompt=validate_and_get_single_string(
                batch.negative_prompt or "", "negative_prompt"
            ),
            width=int(batch.width),
            height=int(batch.height),
            replace_mode=bool(batch.replace_mode),
            segment_len=int(batch.segment_len),
            segment_overlap=int(batch.segment_overlap),
            seed=int(seed),
            num_inference_steps=int(batch.num_inference_steps),
            guidance_scale=float(batch.guidance_scale),
            sample_shift=float(
                default_shift if sample_shift is None else sample_shift
            ),
        )


@dataclass(kw_only=True)
class SCAIL2RequestState:
    """Segment-independent state, computed once and read by every segment."""

    device: torch.device
    generator: torch.Generator
    inputs: SCAIL2Inputs
    height: int
    width: int

    # text, [1, 512, 4096]
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    # CLIP hidden_states[-2] of the reference image, [1, 257, 1280]
    image_embeds: torch.Tensor

    # [1, 16, 1, h, w] / [1, 28, 1, h, w]
    reference_latent: torch.Tensor
    reference_mask: torch.Tensor
    # [1, 16, N, h, w] / [1, 28, N, h, w], None without additional references
    additional_ref_latent: torch.Tensor | None
    additional_ref_mask: torch.Tensor | None

    # Driving inputs at the target size, kept on the host as uint8:
    # video [T, 3, H, W], mask [3, T, H, W].
    driving_video: torch.Tensor
    driving_mask: torch.Tensor

    segments: list[tuple[int, int]]
    # The first segment's noise, drawn here so batch.latents is the real denoising input;
    # later segments draw from the generator in order, as the reference does.
    segment_0_noise: torch.Tensor

    # Assembled by the denoising stage for the output stage, [T, H, W, C] uint8.
    decoded_frames: np.ndarray | None = None


SCAIL2_REQUEST_STATE_EXTRA_KEY = "scail2_request_state"


def request_state_from_batch(batch: Req) -> SCAIL2RequestState:
    state = batch.extra.get(SCAIL2_REQUEST_STATE_EXTRA_KEY)
    if not isinstance(state, SCAIL2RequestState):
        raise ValueError(
            f"batch.extra[{SCAIL2_REQUEST_STATE_EXTRA_KEY!r}] is missing: "
            "SCAIL2BeforeDenoisingStage must run before this stage."
        )
    return state


def make_scheduler_timesteps(
    scheduler: FlowUniPCMultistepScheduler,
    num_inference_steps: int,
    shift: float,
    device: torch.device,
) -> torch.Tensor:
    """The reference's UniPC set-up: the scheduler is built with ``shift=1`` and the
    sampling shift is applied in ``set_timesteps``. Also resets the solver state."""
    scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    return scheduler.timesteps


class SCAIL2BeforeDenoisingStage(TextEncodingStage):
    """Native text encoding plus the SCAIL-2 reference, mask and segment preparation."""

    # Text-only stage dedup must not skip the image/video conditions or the request RNG.
    deduplicated_output_fields = ()

    def __init__(
        self,
        vae: WanAnimate2VaeAdapter,
        image_encoder: WanAnimate2ImageEncoderAdapter,
        text_encoder: torch.nn.Module,
        tokenizer,
        scheduler: FlowUniPCMultistepScheduler,
        default_shift: float,
    ) -> None:
        super().__init__(text_encoders=[text_encoder], tokenizers=[tokenizer])
        self.vae = vae
        self.image_encoder = image_encoder
        # Shared with the denoising stage, which resets it per segment; used here only to
        # derive batch.timesteps on the same grid.
        self.scheduler = scheduler
        self.default_shift = default_shift

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        name = self._component_stage_name(stage_name)
        return super().component_uses(server_args, stage_name) + [
            ComponentUse(stage_name=name, component_name="vae"),
            ComponentUse(stage_name=name, component_name="image_encoder"),
        ]

    def vae_encode(self, videos: list[torch.Tensor]) -> list[torch.Tensor]:
        with self.use_declared_component(component_name="vae", module=self.vae.vae):
            return self.vae.encode(videos)

    def image_embed(self, videos: list[torch.Tensor]) -> torch.Tensor:
        with self.use_declared_component(
            component_name="image_encoder", module=self.image_encoder.model
        ):
            return self.image_encoder.visual(videos)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        inputs = SCAIL2Inputs.from_request(batch, self.default_shift)
        generator = torch.Generator(device=device).manual_seed(inputs.seed)

        state = self._build_request_state(inputs, device, generator, server_args)
        if len(state.segments) > 1:
            self.log_info(
                "driving video splits into %d segments of %d frames (overlap %d)",
                len(state.segments),
                inputs.segment_len,
                inputs.segment_overlap,
            )
        batch.extra[SCAIL2_REQUEST_STATE_EXTRA_KEY] = state

        # Standard DenoisingStage fields, for shared verification and bookkeeping; the
        # segment loop reads the request state.
        batch.prompt_embeds = [state.prompt_embeds]
        batch.negative_prompt_embeds = [state.negative_prompt_embeds]
        batch.image_embeds = [state.image_embeds]
        batch.generator = generator
        batch.timesteps = make_scheduler_timesteps(
            self.scheduler, inputs.num_inference_steps, inputs.sample_shift, device
        )
        batch.sigmas = self.scheduler.sigmas.tolist()
        batch.latents = state.segment_0_noise
        batch.raw_latent_shape = batch.latents.shape
        batch.height, batch.width = state.height, state.width
        return batch

    def _build_request_state(
        self,
        inputs: SCAIL2Inputs,
        device: torch.device,
        generator: torch.Generator,
        server_args: ServerArgs,
    ) -> SCAIL2RequestState:
        # reference image and mask, centre-cropped to the target size
        reference = load_image(inputs.reference_image_path)
        height, width = orient_target_size(
            reference.shape[2], reference.shape[3], inputs.height, inputs.width
        )
        check_resolution(height, width)
        reference = resize_for_rectangle_crop(reference, height, width)[0]
        reference_mask_image = resize_for_rectangle_crop(
            load_image(inputs.reference_mask_path), height, width
        )[0]

        # driving video and mask: every frame, centre-cropped, kept on the host as uint8
        driving_video = resize_for_rectangle_crop(
            read_video_frames(inputs.driving_video_path), height, width
        )
        driving_mask = resize_for_rectangle_crop(
            read_video_frames(inputs.driving_mask_path), height, width
        ).permute(1, 0, 2, 3)
        num_frames = driving_video.shape[0]
        if driving_mask.shape[1] != num_frames:
            raise ValueError(
                "SCAIL-2 driving video and driving mask must have the same frame count; "
                f"got {num_frames} and {driving_mask.shape[1]}"
            )
        segments = build_segments(
            num_frames, inputs.segment_len, inputs.segment_overlap
        )
        if not segments:
            raise ValueError(
                f"SCAIL-2 produced no segment for a {num_frames}-frame driving video; "
                "use a longer video or a smaller segment_len"
            )

        reference = reference.to(device)
        with torch.no_grad():
            # VAE latents in the reference's scaling, [16, 1, h, w]
            reference_latent = self.vae_encode([reference.unsqueeze(1)])[0]
            reference_mask = compress_mask_to_latent(
                reference_mask_image.to(device).unsqueeze(1)
            )
            image_embeds = self.image_embed([reference.unsqueeze(1)]).to(
                torch.bfloat16
            )

            additional_latents, additional_masks = [], []
            for image_path, mask_path in zip(
                inputs.additional_ref_image_paths,
                inputs.additional_ref_mask_paths,
                strict=True,
            ):
                image = resize_for_rectangle_crop(load_image(image_path), height, width)
                mask = resize_for_rectangle_crop(load_image(mask_path), height, width)
                additional_latents.append(
                    self.vae_encode([image[0].to(device).unsqueeze(1)])[0]
                )
                additional_masks.append(
                    compress_mask_to_latent(mask[0].to(device).unsqueeze(1))
                )

            prompt_embeds, negative_prompt_embeds = [
                self.encode_text(text, server_args, device=device)[0][0]
                for text in (inputs.prompt, inputs.negative_prompt)
            ]

        lat_h, lat_w = height // VAE_SPATIAL_STRIDE, width // VAE_SPATIAL_STRIDE
        segment_0_noise = torch.randn(
            _LATENT_CHANNELS,
            latent_frames(segments[0][1] - segments[0][0]),
            lat_h,
            lat_w,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        return SCAIL2RequestState(
            device=device,
            generator=generator,
            inputs=inputs,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            reference_latent=reference_latent.unsqueeze(0),
            reference_mask=reference_mask.unsqueeze(0),
            additional_ref_latent=(
                torch.cat(additional_latents, dim=1).unsqueeze(0)
                if additional_latents
                else None
            ),
            additional_ref_mask=(
                torch.cat(additional_masks, dim=1).unsqueeze(0)
                if additional_masks
                else None
            ),
            driving_video=driving_video,
            driving_mask=driving_mask,
            segments=segments,
            segment_0_noise=segment_0_noise,
        )
