# SPDX-License-Identifier: Apache-2.0
"""Assemble a SCAIL-2 model directory that SGLang Diffusion can load.

SCAIL-2 (https://github.com/zai-org/SCAIL-2) ships only the transformer weights, in the
official Wan parameter naming (e.g. Comfy-Org/SCAIL-2
``diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors``). Its text encoder, CLIP image
encoder and VAE are those of Wan2.1-I2V-14B. This tool links the Wan2.1-I2V-14B-480P
Diffusers components next to a ``transformer/`` holding the SCAIL-2 weights and writes a
``model_index.json`` naming ``SCAIL2Pipeline``:

    python -m sglang.multimodal_gen.tools.build_scail2_model \\
        --wan-i2v-path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \\
        --scail2-weights /path/to/wan2.1_14B_SCAIL_2_fp16.safetensors \\
        --output-path /models/SCAIL-2-14B-Diffusers

The weights keep their official names (the SCAIL-2 DiT config maps them at load time),
so they are symlinked, not rewritten, unless they carry a ComfyUI key prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

# Keys every SCAIL-2 checkpoint must have, with the shapes of the 14B model's
# (out, in, kt, kh, kw) patch embeddings.
_REQUIRED_KEYS = {
    "patch_embedding.weight": (5120, 20, 1, 2, 2),
    "patch_embedding_pose.weight": (5120, 20, 1, 2, 2),
    "patch_embedding_mask.weight": (5120, 28, 1, 2, 2),
}
_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"


def _resolve_base(path_or_repo: str) -> pathlib.Path:
    path = pathlib.Path(path_or_repo).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return pathlib.Path(
        snapshot_download(
            path_or_repo,
            allow_patterns=[
                "model_index.json",
                "text_encoder/*",
                "tokenizer/*",
                "vae/*",
                "image_encoder/*",
                "image_processor/*",
                "scheduler/*",
                "transformer/config.json",
            ],
        )
    )


def _inspect_weights(path: pathlib.Path) -> str:
    """Check the checkpoint is SCAIL-2 and return its key prefix ("" if none)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        prefix = next(
            (p for p in _KEY_PREFIXES if keys and all(k.startswith(p) for k in keys)),
            "",
        )
        for key, shape in _REQUIRED_KEYS.items():
            if prefix + key not in keys:
                raise ValueError(
                    f"{path} has no {key!r}: not a SCAIL-2 transformer checkpoint "
                    "(expected the official Wan parameter names)"
                )
            actual = tuple(f.get_slice(prefix + key).get_shape())
            if actual != shape:
                raise ValueError(
                    f"{path}: {key} has shape {actual}, expected {shape} (SCAIL-2 14B)"
                )
    return prefix


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"{dst} already exists")
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def build(
    wan_i2v_path: str,
    scail2_weights: str,
    output_path: str,
    copy: bool = False,
) -> pathlib.Path:
    base = _resolve_base(wan_i2v_path)
    weights = pathlib.Path(scail2_weights).expanduser().resolve()
    out = pathlib.Path(output_path).expanduser()
    prefix = _inspect_weights(weights)

    out.mkdir(parents=True, exist_ok=False)
    for component in (
        "text_encoder",
        "tokenizer",
        "vae",
        "image_encoder",
        "image_processor",
        "scheduler",
    ):
        src = base / component
        if not src.is_dir():
            raise FileNotFoundError(f"{base} has no {component}/ (is it Wan2.1-I2V Diffusers?)")
        _link_or_copy(src, out / component, copy)

    transformer_dir = out / "transformer"
    transformer_dir.mkdir()
    config = json.loads((base / "transformer" / "config.json").read_text())
    config.update(
        {
            "_class_name": "SCAIL2Transformer3DModel",
            "in_channels": 20,
            "mask_channels": 28,
        }
    )
    (transformer_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if prefix:
        # ComfyUI-prefixed keys: rewrite once without the prefix.
        from safetensors.torch import load_file, save_file

        state_dict = load_file(str(weights))
        save_file(
            {k[len(prefix) :]: v for k, v in state_dict.items()},
            str(transformer_dir / _WEIGHTS_NAME),
        )
    else:
        _link_or_copy(weights, transformer_dir / _WEIGHTS_NAME, copy)

    model_index = json.loads((base / "model_index.json").read_text())
    model_index["_class_name"] = "SCAIL2Pipeline"
    model_index["transformer"] = ["diffusers", "SCAIL2Transformer3DModel"]
    model_index.pop("transformer_2", None)
    (out / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--wan-i2v-path",
        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        help="Local directory or HF repo of Wan2.1-I2V-14B-480P-Diffusers.",
    )
    parser.add_argument(
        "--scail2-weights",
        required=True,
        help="SCAIL-2 transformer safetensors with official Wan parameter names.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking them.",
    )
    args = parser.parse_args()
    out = build(args.wan_i2v_path, args.scail2_weights, args.output_path, args.copy)
    print(f"SCAIL-2 model directory written to {out}")


if __name__ == "__main__":
    main()
