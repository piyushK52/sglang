# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 task profiles.

Data table driving request validation and plan resolution for video tasks
(t2va / fl2va / ref2va) and image tasks (t2i / edit). One row per task; stages
and the request projector consume rows instead of branching on task names.

Design summary: keyframes bind target geometry; references remain independent.
Ref2VA also admits keyframes for hybrid checkpoints, matching the released
Comfy workflow where reference presentation is encoded before guide latents
are appended.
"""

from __future__ import annotations

from types import MappingProxyType

import msgspec

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DEFAULT_BRANCHES,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_video_latent_index_at,
)

MINIMAX_H3_CONDITION_ROLE_KEYFRAME = "keyframe"
MINIMAX_H3_CONDITION_ROLE_REFERENCE = "reference"

MINIMAX_H3_TASK_T2VA = "t2va"
MINIMAX_H3_TASK_FL2VA = "fl2va"
MINIMAX_H3_TASK_REF2VA = "ref2va"
MINIMAX_H3_TASK_T2I = "t2i"
MINIMAX_H3_TASK_EDIT = "edit"

# Keep this video-only: the hybrid video pipeline enumerates these tasks.
MINIMAX_H3_TASK_PARTITIONS = MappingProxyType(
    {
        "t2va": "fl2va",
        "fl2va": "fl2va",
        "ref2va": "ref2va",
    }
)

# --model-variant serving both video partitions from one process: t2va / fl2va
# on the FL2VA transformer, ref2va on the Ref2VA one
MINIMAX_H3_DUAL_VARIANT = "fl2va+ref2va"

# One image checkpoint serves every task it declares, in this order.
MINIMAX_H3_IMAGE_PARTITION = "image"
MINIMAX_H3_IMAGE_TASKS = (MINIMAX_H3_TASK_T2I, MINIMAX_H3_TASK_EDIT)


def canonical_minimax_h3_task(task: str) -> str:
    """Normalize a public task name (no aliases are currently defined)."""

    return task


def partition_for_task(task: str) -> str:
    """Return the deployment partition serving *task*."""

    if not isinstance(task, str) or not task.strip():
        raise ValueError("MiniMax H3 task must be a non-empty string")
    normalized = task.strip().lower()
    if normalized in MINIMAX_H3_IMAGE_TASKS:
        return MINIMAX_H3_IMAGE_PARTITION
    try:
        return MINIMAX_H3_TASK_PARTITIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax H3 task {task!r}") from exc


MINIMAX_H3_FINITE_ASPECT_RATIOS = (
    "21:9",
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
)

# ``fl2va`` remains the single public task name for all keyframe-only variants,
# and hybrid ``ref2va`` takes the same keyframes. Keep the accepted frame_index
# rules centralized so validation and every downstream sink reject reversed or
# stale payloads consistently.
MINIMAX_H3_KEYFRAME_SIGNATURE_HINT = (
    "one or more strictly increasing frame_index values (-1 only as the last)"
)


def minimax_h3_is_keyframe_signature(frame_indices: tuple[object, ...]) -> bool:
    """Whether ordered semantic ``frame_indices`` are admissible."""
    if not frame_indices:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in frame_indices
    ):
        return False
    # -1 names the last frame, so it may only close the sequence.
    ordered = frame_indices[:-1] if frame_indices[-1] == -1 else frame_indices
    return all(value >= 0 for value in ordered) and all(
        earlier < later for earlier, later in zip(ordered, ordered[1:])
    )


def minimax_h3_check_keyframe_position(
    *, frame_index: int, frame_count: int, path: str
) -> None:
    """Require a keyframe on the first frame, the last frame, or a latent start.

    ``frame_count`` is the aligned output length and ``frame_index`` a resolved
    index inside it; ``path`` names the offending request field.
    """
    if frame_index == frame_count - 1:
        return
    if minimax_h3_video_latent_index_at(frame_index) is not None:
        return
    valid = [
        candidate
        for candidate in range(frame_count - 1)
        if minimax_h3_video_latent_index_at(candidate) is not None
    ] + [frame_count - 1]
    before = max(candidate for candidate in valid if candidate < frame_index)
    after = min(candidate for candidate in valid if candidate > frame_index)
    raise ValueError(
        f"{path} must be the first frame, the last frame, or the first frame of a "
        f"video latent (17n + 0, 1, 5, 9, 13), got {frame_index}; nearest valid "
        f"frames are {before} and {after}"
    )


# Canonical material-chain names are part of the direct runtime plan.
# Profile validation keeps only the wire-level allowlist here.
MINIMAX_H3_CANONICAL_MATERIAL_CHAINS = frozenset(
    {
        "image.target_canvas",
        "image.reference_preserve",
        "video.reference_preserve",
        "video_audio.reference_preserve",
        "audio",
    }
)


class MiniMaxH3ConditionRule(msgspec.Struct, frozen=True):
    """Per-(role, type) admission + routing rule."""

    role: str
    condition_type: str
    material_chain: str
    requires_frame_index: bool = False
    visual_tokenizer_encode: bool = False
    audio_tokenizer_encode: bool = False


class MiniMaxH3TaskProfile(msgspec.Struct, frozen=True):
    """One row of the task table."""

    task: str
    conditions_required: bool
    condition_rules: tuple[MiniMaxH3ConditionRule, ...]
    branches: tuple[dict, ...]
    default_flow_shift: float | None
    default_audio_flow_shift: float | None
    required_components: tuple[str, ...]
    aspect_ratio_forced_auto: bool
    geometry_source: str  # "explicit_target" | "first_keyframe" | "model_default"
    # For ref2va video, target may omit duration_seconds
    # when an audio reference is present — duration derives from the
    # reference audio probe and is then checked against the shared 4-15s range.
    duration_from_audio_reference: bool = False
    # Some deployments keep video references disabled while retaining the
    # routing rule for projection/contract compatibility.
    video_reference_supported: bool = True
    min_condition_count: int | None = None
    max_condition_count: int | None = None
    # Resolution policy for target.aspect_ratio="auto".  A concrete ratio
    # resolves immediately; None keeps geometry deferred to the named source.
    auto_aspect_ratio: str | None = None
    auto_geometry_source: str | None = None
    output_modality: str = "video"

    def rule_for(self, *, role: str, condition_type: str) -> MiniMaxH3ConditionRule:
        for rule in self.condition_rules:
            if rule.role == role and rule.condition_type == condition_type:
                return rule
        raise ValueError(
            f"task {self.task!r} does not allow condition "
            f"role={role!r} type={condition_type!r}"
        )


_BASE_COMPONENTS = (
    "processor",
    "text_encoder",
    "tokenizer",
    "transformer",
    "video_vae",
    "audio_vae",
)

_BRANCHES = tuple(dict(branch) for branch in MINIMAX_H3_DEFAULT_BRANCHES)
_IMAGE_COMPONENTS = tuple(name for name in _BASE_COMPONENTS if name != "audio_vae")
# Same tuple-of-dicts shape as the video branches; the name keys the image
# loop's text payload.
_IMAGE_BRANCHES = ({"name": "positive"},)
MINIMAX_H3_TASK_PROFILES: dict[str, MiniMaxH3TaskProfile] = {
    MINIMAX_H3_TASK_T2I: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_T2I,
        conditions_required=False,
        condition_rules=(),
        branches=_IMAGE_BRANCHES,
        default_flow_shift=None,
        default_audio_flow_shift=None,
        required_components=_IMAGE_COMPONENTS,
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        output_modality="image",
    ),
    MINIMAX_H3_TASK_EDIT: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_EDIT,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="image",
                material_chain="image.reference_preserve",
                visual_tokenizer_encode=True,
            ),
        ),
        branches=_IMAGE_BRANCHES,
        default_flow_shift=None,
        default_audio_flow_shift=None,
        required_components=_IMAGE_COMPONENTS,
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        min_condition_count=1,
        output_modality="image",
    ),
    MINIMAX_H3_TASK_T2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_T2VA,
        conditions_required=False,
        condition_rules=(),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        auto_aspect_ratio="16:9",
        auto_geometry_source="policy_default",
    ),
    MINIMAX_H3_TASK_FL2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_FL2VA,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
                condition_type="image",
                material_chain="image.target_canvas",
                requires_frame_index=True,
                visual_tokenizer_encode=True,
            ),
        ),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        # FL follows the resolved target geometry projected by the caller.
        # Its keyframes are prepared against that target canvas.
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        min_condition_count=1,
        auto_geometry_source="first_keyframe",
    ),
    MINIMAX_H3_TASK_REF2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_REF2VA,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
                condition_type="image",
                material_chain="image.target_canvas",
                requires_frame_index=True,
                visual_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="image",
                material_chain="image.reference_preserve",
                visual_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="video",
                material_chain="video.reference_preserve",
                visual_tokenizer_encode=True,
                # ref2va video: the reference video's ORIGINAL soundtrack is the
                # audio reference; -17 rows cover the full pre-truncation track.
                # Route the same material into the audio encoder.
                audio_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="video_audio",
                material_chain="video_audio.reference_preserve",
                visual_tokenizer_encode=True,
                audio_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="audio",
                material_chain="audio",
                audio_tokenizer_encode=True,
            ),
        ),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        # ref2va video may use an explicit aspect ratio such as "7:4":
        # ref2va allows explicit aspect ratios; "auto" falls back to the
        # policy default (references never bind target geometry).
        aspect_ratio_forced_auto=False,
        geometry_source="explicit_target",
        duration_from_audio_reference=True,
        auto_aspect_ratio="16:9",
        auto_geometry_source="policy_default",
        # ref2va video uses a subject image plus a reference video with soundtrack.
        video_reference_supported=True,
    ),
}


def minimax_h3_task_profile(task: str) -> MiniMaxH3TaskProfile:
    profile = MINIMAX_H3_TASK_PROFILES.get(task)
    if profile is None:
        raise ValueError(
            f"unknown minimax_h3 task {task!r}; supported: "
            f"{sorted(MINIMAX_H3_TASK_PROFILES)}"
        )
    return profile


def _validate_profiles() -> None:
    for task, profile in MINIMAX_H3_TASK_PROFILES.items():
        if profile.task != task:
            raise ValueError(f"profile key/task mismatch: {task} vs {profile.task}")
        rule_keys = [
            (rule.role, rule.condition_type) for rule in profile.condition_rules
        ]
        if len(rule_keys) != len(set(rule_keys)):
            raise ValueError(f"task {task}: condition rules must be unique")
        if profile.min_condition_count is not None and profile.min_condition_count <= 0:
            raise ValueError(f"task {task}: min_condition_count must be positive")
        if profile.max_condition_count is not None and profile.max_condition_count <= 0:
            raise ValueError(f"task {task}: max_condition_count must be positive")
        if (
            profile.min_condition_count is not None
            and profile.max_condition_count is not None
            and profile.min_condition_count > profile.max_condition_count
        ):
            raise ValueError(
                f"task {task}: min_condition_count exceeds max_condition_count"
            )
        for rule in profile.condition_rules:
            if rule.material_chain not in MINIMAX_H3_CANONICAL_MATERIAL_CHAINS:
                raise ValueError(
                    f"task {task}: unknown material chain {rule.material_chain!r}"
                )


_validate_profiles()

__all__ = [
    "MINIMAX_H3_CONDITION_ROLE_KEYFRAME",
    "MINIMAX_H3_FINITE_ASPECT_RATIOS",
    "MINIMAX_H3_KEYFRAME_SIGNATURE_HINT",
    "MINIMAX_H3_CONDITION_ROLE_REFERENCE",
    "MINIMAX_H3_CANONICAL_MATERIAL_CHAINS",
    "MINIMAX_H3_DUAL_VARIANT",
    "MINIMAX_H3_TASK_EDIT",
    "MINIMAX_H3_TASK_FL2VA",
    "MINIMAX_H3_IMAGE_PARTITION",
    "MINIMAX_H3_IMAGE_TASKS",
    "MINIMAX_H3_TASK_PARTITIONS",
    "MINIMAX_H3_TASK_PROFILES",
    "MINIMAX_H3_TASK_T2I",
    "MINIMAX_H3_TASK_REF2VA",
    "MINIMAX_H3_TASK_T2VA",
    "MiniMaxH3TaskProfile",
    "canonical_minimax_h3_task",
    "minimax_h3_check_keyframe_position",
    "minimax_h3_is_keyframe_signature",
    "minimax_h3_task_profile",
    "partition_for_task",
]
