# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SCAIL-2: RoPE parity with the official implementation, mask
compression, the segment schedule, checkpoint-name mapping, registry routing and
request validation.

No weights, GPU or network. The reference functions below are ported from
https://github.com/zai-org/SCAIL-2 (wan/modules/model_scail2.py,
wan/utils/scail_utils.py) so the port is checked against the original numerics.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.scail2 import (
    SCAIL2ArchConfig,
    SCAIL2Config,
)
from sglang.multimodal_gen.configs.pipeline_configs.scail2 import SCAIL2_14B_Config
from sglang.multimodal_gen.configs.sample.scail2 import SCAIL2_14B_SamplingParams
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.layers.rotary_embedding import NDRotaryEmbedding
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.scail2 import (
    SCAIL2Transformer3DModel,
    scail2_rope_cos_sin,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.scail2.denoising import (
    apply_clean_history,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.scail2.preprocess import (
    build_segments,
    compress_mask_to_latent,
    orient_target_size,
)

# --------------------------------------------------------------------------------------
# Reference RoPE (official model_scail2.py, complex arithmetic)
# --------------------------------------------------------------------------------------


def _rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _reference_freqs(d):
    return torch.cat(
        [
            _rope_params(1024, d - 4 * (d // 6)),
            _rope_params(1024, 2 * (d // 6)),
            _rope_params(1024, 2 * (d // 6)),
        ],
        dim=1,
    )


def _reference_apply(x, freqs, f, h, w, shift_f, shift_h, shift_w, pool=False):
    n, c = x.size(2), x.size(3) // 2
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    freqs_i = torch.cat(
        [
            parts[0][shift_f : shift_f + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            parts[1][shift_h : shift_h + h].view(1, h, 1, -1).expand(f, h, w, -1),
            parts[2][shift_w : shift_w + w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    )
    if pool:
        real = F.avg_pool2d(freqs_i.real.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)
        imag = F.avg_pool2d(freqs_i.imag.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)
        freqs_i = torch.complex(real, imag)
    seq_len = freqs_i.shape[0] * freqs_i.shape[1] * freqs_i.shape[2]
    freqs_i = freqs_i.reshape(seq_len, 1, -1)
    x_i = torch.view_as_complex(x[0].to(torch.float64).reshape(seq_len, n, -1, 2))
    return torch.view_as_real(x_i * freqs_i).flatten(2).unsqueeze(0).float()


def _reference_rope(x, d, t, h, w, replace_mode, num_additional_refs):
    freqs = _reference_freqs(d)
    base = 0 if replace_mode else 1
    h_shift = 120 if replace_mode else 0
    ref_len, video_len = h * w, t * h * w
    out, i = [], 0
    if num_additional_refs:
        n = num_additional_refs * ref_len
        out.append(
            _reference_apply(x[:, :n], freqs, num_additional_refs, h, w, 0, h_shift, 0)
        )
        i = n
    out.append(
        _reference_apply(
            x[:, i : i + ref_len], freqs, 1, h, w, num_additional_refs, h_shift, 0
        )
    )
    i += ref_len
    video_t0 = base + num_additional_refs
    out.append(
        _reference_apply(x[:, i : i + video_len], freqs, t, h, w, video_t0, 0, 0)
    )
    i += video_len
    out.append(_reference_apply(x[:, i:], freqs, t, h, w, video_t0, 0, 120, pool=True))
    return torch.cat(out, dim=1)


def _apply_interleaved(x, cos, sin):
    """The Wan blocks' rotation: pairs (x[2i], x[2i+1]) rotated by (cos[i], sin[i])."""
    x = x.float()
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out = torch.stack(
        [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
    )
    return out.flatten(-2)


@pytest.mark.parametrize(
    "replace_mode,num_additional_refs", [(False, 0), (True, 0), (False, 2), (True, 1)]
)
def test_rope_matches_official(replace_mode, num_additional_refs):
    head_dim, heads, t, h, w = 128, 2, 3, 4, 6
    rotary_emb = NDRotaryEmbedding(
        rope_dim_list=[head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)],
        rope_theta=10000,
        dtype=torch.float64,
    )
    cos, sin = scail2_rope_cos_sin(
        rotary_emb,
        t,
        h,
        w,
        replace_mode=replace_mode,
        num_additional_refs=num_additional_refs,
        pose_w_shift=120,
        ref_h_shift=120,
        device=torch.device("cpu"),
    )
    n = (num_additional_refs + 1) * h * w + t * h * w + t * (h // 2) * (w // 2)
    assert cos.shape == (n, head_dim // 2)
    x = torch.randn(1, n, heads, head_dim)
    torch.testing.assert_close(
        _apply_interleaved(x, cos, sin),
        _reference_rope(x, head_dim, t, h, w, replace_mode, num_additional_refs),
        atol=1e-4,
        rtol=1e-4,
    )


# --------------------------------------------------------------------------------------
# Masks, segments, history
# --------------------------------------------------------------------------------------


def test_mask_compression_shape_and_colour():
    # 9 all-blue frames -> 3 latent frames; blue is colour index 3 of 7
    mask = torch.zeros(3, 9, 64, 96)
    mask[2] = 1.0
    latent = compress_mask_to_latent(mask * 2 - 1)
    assert latent.shape == (28, 3, 8, 12)
    by_colour = latent.reshape(4, 7, 3, 8, 12)
    assert torch.all(by_colour[:, 3] == 1.0)
    assert latent.sum() == by_colour[:, 3].sum()


def test_mask_compression_black_is_empty_and_white_is_white():
    black = -torch.ones(3, 1, 16, 16)
    assert compress_mask_to_latent(black).sum() == 0
    white = torch.ones(3, 1, 16, 16)
    latent = compress_mask_to_latent(white)
    assert torch.all(latent.reshape(4, 7, 1, 2, 2)[:, 0] == 1.0)
    assert latent.sum() == 4 * 2 * 2


def test_mask_threshold_is_225():
    # 224/255 is off, 226/255 is on (reference _ON_THRESH)
    off = torch.full((3, 1, 8, 8), 224 / 127.5 - 1)
    on = torch.full((3, 1, 8, 8), 226 / 127.5 - 1)
    assert compress_mask_to_latent(off).sum() == 0
    assert compress_mask_to_latent(on).reshape(4, 7, 1, 1, 1)[:, 0].sum() == 4


@pytest.mark.parametrize(
    "total,expected",
    [
        (50, [(0, 49)]),
        (81, [(0, 81)]),
        (200, [(0, 81), (76, 157)]),
        (157, [(0, 81), (76, 157)]),
    ],
)
def test_segments_match_official(total, expected):
    assert build_segments(total, 81, 5) == expected


def test_segments_reject_bad_overlap():
    with pytest.raises(ValueError):
        build_segments(100, 81, 0)
    with pytest.raises(ValueError):
        build_segments(100, 81, 81)


def test_clean_history_is_pinned():
    latent = torch.randn(16, 5, 4, 4)
    history = torch.randn(16, 2, 4, 4)
    out = apply_clean_history(latent, history)
    assert torch.equal(out[:, :2], history)
    assert torch.equal(out[:, 2:], latent[:, 2:])
    assert apply_clean_history(latent, None) is latent


def test_target_size_follows_reference_orientation():
    # portrait reference, landscape request -> swapped (generate.py)
    assert orient_target_size(1000, 600, 512, 896) == (896, 512)
    assert orient_target_size(600, 1000, 512, 896) == (512, 896)
    assert orient_target_size(600, 600, 512, 896) == (512, 896)


# --------------------------------------------------------------------------------------
# Checkpoint names, config, registry, request validation
# --------------------------------------------------------------------------------------

_TOP_LEVEL_KEYS = [
    *(f"{m}.{p}" for m in ("patch_embedding", "patch_embedding_pose", "patch_embedding_mask") for p in ("weight", "bias")),
    *(f"text_embedding.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")),
    *(f"time_embedding.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")),
    "time_projection.1.weight",
    "time_projection.1.bias",
    "head.head.weight",
    "head.head.bias",
    "head.modulation",
    *(f"img_emb.proj.{i}.{p}" for i in (0, 1, 3, 4) for p in ("weight", "bias")),
]
_BLOCK_KEYS = [
    *(f"self_attn.{m}.{p}" for m in "qkvo" for p in ("weight", "bias")),
    "self_attn.norm_q.weight",
    "self_attn.norm_k.weight",
    *(f"cross_attn.{m}.{p}" for m in "qkvo" for p in ("weight", "bias")),
    *(f"cross_attn.{m}.{p}" for m in ("k_img", "v_img") for p in ("weight", "bias")),
    "cross_attn.norm_q.weight",
    "cross_attn.norm_k.weight",
    "cross_attn.norm_k_img.weight",
    "norm3.weight",
    "norm3.bias",
    *(f"ffn.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")),
    "modulation",
]


def _official_checkpoint_keys() -> list[str]:
    keys = list(_TOP_LEVEL_KEYS)
    for block in range(SCAIL2ArchConfig().num_layers):
        keys.extend(f"blocks.{block}.{key}" for key in _BLOCK_KEYS)
    return keys


@pytest.mark.usefixtures("single_process_model_parallel")
def test_every_official_key_maps_onto_a_distinct_dit_parameter():
    """The released SCAIL-2 weights use the official Wan names; every one must land on
    its own DiT parameter, and every parameter must be covered."""
    keys = _official_checkpoint_keys()
    map_fn = get_param_names_mapping(SCAIL2ArchConfig().param_names_mapping)
    mapped = {key: map_fn(key)[0] for key in keys}
    assert sorted(k for k, v in mapped.items() if v == k) == []
    assert len(set(mapped.values())) == len(keys)
    assert mapped["patch_embedding_pose.weight"] == "patch_embedding_pose.proj.weight"
    assert mapped["patch_embedding_mask.bias"] == "patch_embedding_mask.proj.bias"

    with torch.device("meta"):
        model = SCAIL2Transformer3DModel(SCAIL2Config(), hf_config={})
    assert set(mapped.values()) == set(model.state_dict())
    assert model.patch_embedding.proj.weight.shape == (5120, 20, 1, 2, 2)
    assert model.patch_embedding_pose.proj.weight.shape == (5120, 20, 1, 2, 2)
    assert model.patch_embedding_mask.proj.weight.shape == (5120, 28, 1, 2, 2)


def test_built_transformer_config_updates_the_arch_config():
    config = SCAIL2Config()
    config.update_model_arch(
        {
            "in_channels": 20,
            "mask_channels": 28,
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "image_dim": 1280,
            "added_kv_proj_dim": 5120,
            "patch_size": [1, 2, 2],
        }
    )
    assert config.in_channels == 20
    assert config.mask_channels == 28
    assert config.patch_size == (1, 2, 2)
    assert config.hidden_size == 5120


@pytest.mark.parametrize(
    "path", ["/models/SCAIL-2-14B-Diffusers", "ckpts/scail2_local"]
)
def test_detector_resolves_scail2_paths(path):
    with patch(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        return_value={},
    ):
        info = _get_config_info(path)
    assert info is not None
    assert info.pipeline_config_cls is SCAIL2_14B_Config
    assert info.sampling_param_cls is SCAIL2_14B_SamplingParams


def test_model_index_class_name_routes_a_checkpoint_at_any_path():
    with patch(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        return_value={"_class_name": "SCAIL2Pipeline"},
    ):
        info = _get_config_info("/models/ckpt-7f3a")
    assert info is not None
    assert info.pipeline_config_cls is SCAIL2_14B_Config


def test_pipeline_config_defaults():
    config = SCAIL2_14B_Config()
    assert config.flow_shift == 3.0
    assert config.dit_config.arch_config.in_channels == 20
    assert config.get_decode_scale_and_shift(None, None, None) == (1.0, None)
    assert not config.supports_disaggregation()


def test_sampling_params_defaults_and_validation():
    params = SCAIL2_14B_SamplingParams()
    assert (params.width, params.height) == (896, 512)
    assert params.num_inference_steps == 40
    assert params.guidance_scale == 5.0
    assert params.negative_prompt == ""
    assert (params.segment_len, params.segment_overlap) == (81, 5)
    with pytest.raises(ValueError):
        SCAIL2_14B_SamplingParams(segment_len=80)
    with pytest.raises(ValueError):
        SCAIL2_14B_SamplingParams(segment_overlap=81)
    with pytest.raises(ValueError):
        SCAIL2_14B_SamplingParams(enable_teacache=True)
    assert {"reference_mask_path", "driving_mask_path", "replace_mode"} <= (
        SCAIL2_14B_SamplingParams.video_request_extra_fields()
    )


def test_request_needs_masks_and_driving_video():
    config = SCAIL2_14B_Config()
    with pytest.raises(ValueError, match="video_path"):
        SCAIL2_14B_SamplingParams(
            image_path="ref.png",
            reference_mask_path="m.png",
            driving_mask_path="m.mp4",
        )._validate_with_pipeline_config(config)
    with pytest.raises(ValueError, match="mask"):
        SCAIL2_14B_SamplingParams(
            image_path="ref.png", video_path="drive.mp4"
        )._validate_with_pipeline_config(config)
    SCAIL2_14B_SamplingParams(
        image_path="ref.png",
        video_path="drive.mp4",
        reference_mask_path="m.png",
        driving_mask_path="m.mp4",
    )._validate_with_pipeline_config(config)
