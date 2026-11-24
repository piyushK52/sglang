# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SCAIL-2 14B (https://github.com/zai-org/SCAIL-2).

Defaults follow the reference ``generate.py``: 896x512 (swapped to match the reference
image's orientation), 40 UniPC steps, CFG 5, shift 3, an empty negative prompt, and
81-frame segments overlapping by 5 frames. With the Lightx2v step-distill LoRA use
``num_inference_steps=8, guidance_scale=1.0, sample_shift=1.0``.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_WARMUP_SIZE = 64


@dataclass
class SCAIL2_14B_SamplingParams(SamplingParams):
    """Request fields and defaults for SCAIL-2.

    ``image_path`` is the reference image and ``video_path`` the driving video. The two
    masks are colour coded: each character is painted one of white, red, green, blue,
    yellow, magenta or cyan in both the reference mask and the driving mask, and the
    background is black or white. In animation mode the reference background is white
    (keep) and the driving background black (drop); in replacement mode the opposite.
    """

    height: int = 512
    width: int = 896
    # Nominal only: the output has one frame per generated driving-video frame.
    num_frames: int = 81
    fps: int = 16

    guidance_scale: float = 5.0
    num_inference_steps: int = 40
    negative_prompt: str = ""

    # Colour-coded mask of the reference image (an image path).
    reference_mask_path: str | None = None
    # Colour-coded per-frame mask of the driving video (a video path, same frame count).
    driving_mask_path: str | None = None
    # Replacement mode (put the character into the driving video) instead of animation.
    replace_mode: bool = False
    # Optional extra references (other views of the character, or more characters), each
    # with its own colour mask.
    additional_ref_image_paths: list[str] | None = None
    additional_ref_mask_paths: list[str] | None = None

    # Long driving videos run in segments of segment_len frames (4k+1) that share
    # segment_overlap frames; frames after the last full segment are not generated.
    segment_len: int = 81
    segment_overlap: int = 5
    # Flow-matching shift applied in the UniPC set_timesteps; None uses the pipeline
    # config's flow_shift (3.0, the reference default).
    sample_shift: float | None = None

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(896, 512), (512, 896)]
    )

    def __post_init__(self) -> None:
        if self.enable_teacache or self.enable_spectrum:
            raise ValueError("SCAIL-2 does not support enable_teacache or enable_spectrum.")
        if self.segment_len % 4 != 1 or self.segment_len < 5:
            raise ValueError(
                f"segment_len must be 4k+1 frames (k >= 1), got {self.segment_len}"
            )
        if not 0 < self.segment_overlap < self.segment_len:
            raise ValueError(
                "segment_overlap must be in (0, segment_len), got "
                f"{self.segment_overlap}"
            )
        # The output length follows the driving video, not num_frames.
        self.adjust_frames = False
        super().__post_init__()

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "reference_mask_path",
                "driving_mask_path",
                "replace_mode",
                "additional_ref_image_paths",
                "additional_ref_mask_paths",
                "segment_len",
                "segment_overlap",
                "sample_shift",
            }
        )

    def _validate_with_pipeline_config(self, pipeline_config) -> None:
        super()._validate_with_pipeline_config(pipeline_config)
        if not self.video_path:
            raise ValueError("SCAIL-2 requires 'video_path' (the driving video)")
        if not self.reference_mask_path or not self.driving_mask_path:
            raise ValueError(
                "SCAIL-2 requires 'reference_mask_path' and 'driving_mask_path' "
                "(colour-coded masks of the reference image and the driving video)"
            )

    def prepare_synthetic_warmup_request_for_queue(
        self, req: Any, server_args: Any
    ) -> None:
        """Give the synthetic warmup request the masks and driving video the generic
        builder does not supply: a grey one-segment driving video, a centred blue
        character in both masks, white reference background / black driving background
        (animation mode)."""
        del server_args
        from PIL import Image

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.preprocess import (
            write_reference_video,
        )

        image_path = req.image_path
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else None
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("SCAIL-2 synthetic warmup requires the warmup reference image")
        directory = os.path.dirname(image_path)
        num_frames = self.segment_len
        fps = int(req.fps)

        size = _WARMUP_SIZE
        inner = slice(size // 4, 3 * size // 4)
        reference_mask = np.full((size, size, 3), 255, dtype=np.uint8)
        reference_mask[inner, inner] = (0, 0, 255)
        driving_mask = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
        driving_mask[:, inner, inner] = (0, 0, 255)
        driving_video = np.full((num_frames, size, size, 3), 128, dtype=np.uint8)

        reference_mask_path = os.path.join(directory, "scail2_warmup_ref_mask.png")
        video_path = os.path.join(directory, f"scail2_warmup_driving_{num_frames}f.mp4")
        mask_path = os.path.join(directory, f"scail2_warmup_mask_{num_frames}f.mp4")
        Image.fromarray(reference_mask).save(reference_mask_path)
        write_reference_video(video_path, driving_video, fps=fps)
        write_reference_video(mask_path, driving_mask, fps=fps)

        req.video_path = video_path
        req.reference_mask_path = reference_mask_path
        req.driving_mask_path = mask_path
        req.num_frames = num_frames

    def _adjust_visual_fields(self, server_args: Any, pipeline_config: Any) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        if self.num_frames != type(self).num_frames:
            logger.info(
                "SCAIL-2 ignores num_frames=%s: the output length follows the driving "
                "video",
                self.num_frames,
            )
