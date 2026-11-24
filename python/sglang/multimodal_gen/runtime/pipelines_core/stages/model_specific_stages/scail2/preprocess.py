# SPDX-License-Identifier: Apache-2.0
# Adapted from the SCAIL-2 reference implementation (Apache-2.0):
# https://github.com/zai-org/SCAIL-2 (generate.py, wan/scail.py, wan/utils/scail_utils.py)
#
# Input loading, resizing, mask compression and the segment schedule, kept numerically
# identical to the reference.
from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.srt.multimodal.media_io import get_video_bytes

# Wan2.1 VAE strides.
VAE_TEMPORAL_STRIDE = 4
VAE_SPATIAL_STRIDE = 8
# Token-grid side lengths must be even (the driving tokens pool 2x2), so pixel sides are
# multiples of 8 (VAE) * 2 (patch) * 2 (pool).
RESOLUTION_MULTIPLE = 32

# Colour order of the 28 mask channels (official ``extract_and_compress_mask_to_latent``).
MASK_COLOURS = ("white", "red", "green", "blue", "yellow", "magenta", "cyan")
# A pixel channel counts as "on" at >= 225/255.
_MASK_ON_THRESHOLD = (225.0 - 127.5) / 127.5


def load_image(path: str) -> torch.Tensor:
    """``[1, 3, H, W]`` fp32 in [-1, 1], RGB (official ``load_image_to_tensor_chw_normalized``)."""
    image = Image.open(path).convert("RGB")
    array = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
    return (array.permute(2, 0, 1).float() / 255.0 * 2 - 1).unsqueeze(0)


def read_video_frames(path: str) -> torch.Tensor:
    """Every frame of the video at its native rate, ``[T, 3, H, W]`` uint8 RGB.

    The reference reads all frames with decord and does not resample, so the output has
    as many frames as the driving video. decord2 / PyAV are fallbacks.
    """
    source = (
        BytesIO(get_video_bytes(path))
        if path.startswith(("http://", "https://", "data:"))
        else path
    )
    try:
        from decord import VideoReader
    except ImportError:
        try:
            from decord2 import VideoReader
        except ImportError:
            VideoReader = None

    if VideoReader is not None:
        reader = VideoReader(source)
        frames = reader.get_batch(list(range(len(reader)))).asnumpy()
    else:
        import av

        with av.open(source) as container:
            frames = np.stack(
                [
                    frame.to_ndarray(format="rgb24")
                    for frame in container.decode(video=0)
                ]
            )
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()


def resize_for_rectangle_crop(
    frames: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Bicubic resize to cover ``(height, width)`` then centre crop, on ``[N, C, H, W]``
    (official ``resize_for_rectangle_crop`` with ``reshape_mode="center"``)."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import crop, resize

    src_h, src_w = frames.shape[2], frames.shape[3]
    if src_w / src_h > width / height:
        frames = resize(
            frames,
            size=[height, int(src_w * height / src_h)],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        frames = resize(
            frames,
            size=[int(src_h * width / src_w), width],
            interpolation=InterpolationMode.BICUBIC,
        )
    delta_h = frames.shape[2] - height
    delta_w = frames.shape[3] - width
    return crop(frames, top=delta_h // 2, left=delta_w // 2, height=height, width=width)


def orient_target_size(
    image_height: int, image_width: int, height: int, width: int
) -> tuple[int, int]:
    """Swap the requested size when its orientation disagrees with the reference image's
    (``generate.py`` does this before resizing). Returns ``(height, width)``."""
    if (image_height < image_width and height > width) or (
        image_height > image_width and height < width
    ):
        return width, height
    return height, width


def check_resolution(height: int, width: int) -> None:
    if height % RESOLUTION_MULTIPLE or width % RESOLUTION_MULTIPLE:
        raise ValueError(
            f"SCAIL-2 needs width and height that are multiples of {RESOLUTION_MULTIPLE}; "
            f"got {width}x{height}"
        )


def compress_mask_to_latent(mask_cthw: torch.Tensor) -> torch.Tensor:
    """Colour mask ``[3, T, H, W]`` in [-1, 1] -> binary ``[28, T_lat, H_lat, W_lat]``
    without the VAE (official ``extract_and_compress_mask_to_latent``).

    Each pixel is classified into the 7 colours of :data:`MASK_COLOURS` (black is all
    zeros), area-downsampled 8x and stacked 4 frames per latent frame, the first frame
    repeated 4 times like the causal VAE.
    """
    _, num_frames, height, width = mask_cthw.shape
    mask = mask_cthw.permute(1, 0, 2, 3).float()
    r = (mask[:, 0:1] > _MASK_ON_THRESHOLD).float()
    g = (mask[:, 1:2] > _MASK_ON_THRESHOLD).float()
    b = (mask[:, 2:3] > _MASK_ON_THRESHOLD).float()
    nr, ng, nb = 1 - r, 1 - g, 1 - b
    binary = torch.cat(
        [r * g * b, r * ng * nb, nr * g * nb, nr * ng * b, r * g * nb, r * ng * b, nr * g * b],
        dim=1,
    )
    lat_h, lat_w = height, width
    for _ in range(3):
        lat_h = (lat_h + 1) // 2
        lat_w = (lat_w + 1) // 2
    binary = F.interpolate(binary, size=(lat_h, lat_w), mode="area")
    lat_t = (num_frames - 1) // VAE_TEMPORAL_STRIDE + 1
    padded = torch.cat(
        [binary[:1].repeat(VAE_TEMPORAL_STRIDE, 1, 1, 1), binary[1:]], dim=0
    )
    return padded.view(lat_t, VAE_TEMPORAL_STRIDE * 7, lat_h, lat_w).permute(
        1, 0, 2, 3
    )


def build_segments(
    total_frames: int, segment_len: int, segment_overlap: int
) -> list[tuple[int, int]]:
    """``[start, end)`` pixel-frame windows (official ``build_segments``).

    A video no longer than one segment is cut to the largest 4k+1 frame count. Longer
    videos are split into full ``segment_len`` windows that overlap by
    ``segment_overlap`` frames; frames after the last full window are not generated.
    """
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    if segment_overlap <= 0 or segment_overlap >= segment_len:
        raise ValueError("segment_overlap must be in (0, segment_len)")
    if total_frames <= segment_len:
        keep = (total_frames - 1) // VAE_TEMPORAL_STRIDE * VAE_TEMPORAL_STRIDE + 1
        return [(0, keep)]
    segments = []
    start = 0
    stride = segment_len - segment_overlap
    while start + segment_len <= total_frames:
        segments.append((start, start + segment_len))
        start += stride
    return segments


def latent_frames(num_pixel_frames: int) -> int:
    return (num_pixel_frames - 1) // VAE_TEMPORAL_STRIDE + 1
