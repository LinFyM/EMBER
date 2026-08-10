"""Episode-level evidence contract for Expert-Manifold Writer evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_ADAPTER_SCHEMA,
    RANK_RESERVED_CONFIG_SCHEMA,
    RANK_RESERVED_EPISODE_SCHEMA,
)
from ember.expert_manifold.v6_prior_contract import V6_PRIOR_CONFIG_SCHEMA
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    condition_demo_index,
    frame_order_seed,
    reference_demo_index,
    video_selection_seed,
)


EXPERT_MANIFOLD_WRITER_KIND = "expert_manifold_writer"
EXPERT_MANIFOLD_ADAPTER_SCHEMA = (
    "ember_pi05_v6_condition_program_residual_eval_adapter_v8"
)
EXPERT_MANIFOLD_EPISODE_SCHEMA = "ember_pi05_v6_condition_program_residual_episode_v8"


def expert_manifold_episode_schema(adapter: Mapping[str, Any]) -> str:
    """Return the episode schema bound to this exact deployment adapter."""

    schema = adapter.get("schema_version")
    config_schema = adapter.get("config", {}).get("schema")
    legacy_config_schemas = {
        "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_v2",
        "ember_pi05_v6_exact_anchored_reconciliation_program_residual_v3",
        V6_PRIOR_CONFIG_SCHEMA,
    }
    if (
        schema == EXPERT_MANIFOLD_ADAPTER_SCHEMA
        and config_schema in legacy_config_schemas
    ):
        return EXPERT_MANIFOLD_EPISODE_SCHEMA
    if (
        schema == RANK_RESERVED_ADAPTER_SCHEMA
        and config_schema == RANK_RESERVED_CONFIG_SCHEMA
    ):
        return RANK_RESERVED_EPISODE_SCHEMA
    raise ExpertManifoldError("invalid Expert-Manifold deployment adapter")


def expected_expert_manifold_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
) -> dict[str, Any]:
    if adapter.get("kind") != EXPERT_MANIFOLD_WRITER_KIND or not lora_reference:
        raise ExpertManifoldError("invalid Expert-Manifold episode adapter")
    episode_schema = expert_manifold_episode_schema(adapter)
    matches = [
        row
        for row in adapter["task_video_mapping"]
        if row["suite"] == suite and int(row["task_id"]) == task_id
    ]
    if len(matches) != 1:
        raise ExpertManifoldError("episode task is outside Expert-Manifold mapping")
    mapping = matches[0]
    schedule = adapter["video_schedule"]
    seed = int(schedule["seed"])
    mode = str(schedule["sampling_mode"])
    demo_count = int(schedule["demo_count"])
    reference = reference_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=mode,
    )
    selected = condition_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        condition=str(adapter["video_condition"]),
        demo_count=demo_count,
        sampling_mode=mode,
    )
    result = {
        "schema_version": episode_schema,
        "writer_method": EXPERT_MANIFOLD_WRITER_KIND,
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "writer_asset_reference": adapter["writer_asset"]["reference"],
        "writer_checkpoint_kind": adapter["writer_asset"]["kind"],
        "writer_method_macro": int(adapter["writer_asset"]["method_macro"]),
        "writer_parameter_count": int(
            adapter["writer_asset"]["writer_parameter_count"]
        ),
        "writer_deployment_trainable_parameter_count": 0,
        "writer_program_residual_value_count": int(
            adapter["writer_asset"]["program_residual_value_count"]
        ),
        "generated_lora_tensor_count": int(
            adapter["writer_asset"]["generated_lora_tensor_count"]
        ),
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_frames_used": adapter["video_condition"] != "no_video",
        "teacher_video_count": int(adapter["video_condition"] != "no_video"),
        "teacher_video_seed_root": seed,
        "teacher_video_selection_seed": video_selection_seed(
            seed,
            suite,
            task_id,
            init_state_id,
            sampling_mode=mode,
        ),
        "teacher_video_sampling_mode": mode,
        "video_suite": str(mapping["video_suite"]),
        "video_task_id": int(mapping["video_task_id"]),
        "video_global_task_id": int(mapping["video_global_task_id"]),
        "video_split_role": str(mapping["video_split_role"]),
        "teacher_demo_indices": [selected],
        "teacher_reference_demo_indices": [reference],
        "task_video_mapping_reference": adapter["task_video_mapping_reference"],
        "pairing_reference": adapter["pairing_reference"],
        "writer_generation_seed_schedule": (
            "numeric_seedsequence_one_shot_frame_order_v1"
        ),
        "teacher_video_order_seeds": [
            frame_order_seed(seed, suite, task_id, reference)
        ],
    }
    if adapter["video_condition"] == "same_task_other":
        result["teacher_demo_offset"] = SAME_TASK_OTHER_OFFSET
    if adapter["schema_version"] == RANK_RESERVED_ADAPTER_SCHEMA:
        result.update(
            {
                "writer_program_residual_enabled": bool(
                    adapter["writer_asset"]["enable_program_residual"]
                ),
                "writer_qv_base_rank": 14,
                "writer_qv_residual_rank": 2,
                "writer_action_rank": 16,
            }
        )
    return result


def validate_expert_manifold_episode_evidence(
    adapter: Mapping[str, Any],
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    if not isinstance(row, Mapping):
        return False
    try:
        seconds = float(row.get("writer_generation_seconds", float("nan")))
        expected = expected_expert_manifold_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=str(row.get("lora_reference", "")),
        )
    except (ExpertManifoldError, KeyError, TypeError, ValueError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return observed == expected and math.isfinite(seconds) and seconds >= 0
