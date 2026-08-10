"""Static authority and Program reference for the rank-reserved compiler."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import REPO_ROOT, V6_PRIOR_CONFIG_SCHEMA
from ember.pi05_source_checkpoint import read_json


RANK_RESERVED_CONFIG_SCHEMA = "ember_pi05_v6_qv_rank_reserved_native_reward_v1"
RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_program_reference_v1"
)
RANK_RESERVED_ADAPTER_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_native_reward_eval_adapter_v9"
)
RANK_RESERVED_EPISODE_SCHEMA = "ember_pi05_v6_qv_rank_reserved_native_reward_episode_v9"
RANK_RESERVED_FAMILY = "v6_qv_rank_reserved_native_reward_v1"
RANK_RESERVED_CANONICAL_CONFIG = (
    REPO_ROOT / "configs/pi05_v6_qv_rank_reserved_native_reward_v1.json"
)
RANK_RESERVED_PROGRAM_REFERENCE = (
    REPO_ROOT / "configs/pi05_v6_qv_rank_reserved_cycle1_program_load_only_v1.json"
)

_BASE_AUTHORITY = {
    "path": "configs/pi05_v6_reward_credit_program_cotangent_v1.json",
    "bytes": 13_939,
    "schema": V6_PRIOR_CONFIG_SCHEMA,
}
_DESIGN_AUTHORITY = {
    "path": "docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md"
}
_GENERATION_EVIDENCE = {
    "path": (
        "runs/outputs/pi05_reward_qv_pivot_rank14_plus2_transport_v1_"
        "e3857f7_20260811/analysis.json"
    ),
    "bytes": 2_604_840,
    "schema": "ember_reward_qv_pivot_rank14_plus2_native_transport_analysis_v1",
    "source_commit": "e3857f73ce92fa7f790a7e49f8166d7e5ef5b9e5",
}
_METHOD = {
    "name": "frozen_v6_qv_rank_reserved_native_reward_load_only",
    "writer_input": "exact task language plus exactly one action-hidden teacher video",
    "dynamic_value": "one_raw_teacher_video_only",
    "language_only_lora_path": False,
    "deployment_expert_bank_read": False,
    "deployment_output": "one complete 38-target rank16 public LoRA",
    "training_enabled": False,
}
_COMPILER = {
    "name": "qv_pivot_preserving_rank14_plus_condition_local_rank2_v1",
    "public_rank": 16,
    "public_target_count": 38,
    "public_tensor_count": 76,
    "qv_target_count": 36,
    "qv_base_rank": 14,
    "qv_residual_rank": 2,
    "qv_pivot": (
        "deterministic_modified_gram_schmidt_native_b_columns_first_ordinal_tie_break"
    ),
    "qv_base_solve": "fp32_least_squares_to_native_b0a0_then_native_publish",
    "qv_tangent": ("analytic_b0_delta_a_plus_delta_b_a0_without_second_order"),
    "qv_residual_factorization": (
        "condition_local_compact_top2_svd_without_full_t_materialization"
    ),
    "qv_zero_program": "rank14_base_plus_two_physical_zero_a_and_b_slots",
    "action_target_count": 2,
    "action_path": (
        "unchanged_full_rank16_fp32_factor_candidate_with_second_order_cross_term"
    ),
    "no_video_path": "source_identity_fast_path_before_video_or_compiler",
    "native_storage": "72_bf16_plus_4_fp32_public_tensors",
    "training_enabled": False,
}
_ASSETS = {
    "macro0": {
        "kind": "v6_qv_rank14_zero_program_load_only",
        "method_macro": 0,
        "checkpoint": (
            "runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_"
            "seed7_s2400_4efa737_20260729/checkpoints/step_00000400"
        ),
        "enable_program_residual": False,
    },
    "cycle1": {
        "kind": "v6_qv_rank14_plus2_reward_program_load_only",
        "method_macro": 1,
        "checkpoint": (
            "configs/pi05_v6_qv_rank_reserved_cycle1_program_load_only_v1.json"
        ),
        "enable_program_residual": True,
    },
}
_GATES = {
    "macro0_correct_min": 130,
    "macro0_breadth_min": 6,
    "macro0_lost_to_paired_old134_max": 10,
    "cycle1_correct_min": 144,
    "cycle1_breadth_min": 6,
    "cycle1_lost_to_macro0_max": 6,
    "cycle1_gained_must_exceed_lost": True,
    "diagnostic_nonpass_correct_range": [140, 143],
    "goal_correct_strict_min": 151,
}
_REGISTERED_ROOTS = {
    "profile": (
        "runs/outputs/pi05_v6_qv_rank_reserved_native_reward_profile_"
        "b8_b16_b32_20260811"
    ),
    "vertical": (
        "runs/outputs/pi05_v6_qv_rank_reserved_native_reward_vertical_"
        "four_suite_20260811"
    ),
    "macro0_correct": (
        "runs/outputs/pi05_v6_qv_rank_reserved_native_reward_"
        "correct400_macro0000_20260811"
    ),
    "cycle1_correct": (
        "runs/outputs/pi05_v6_qv_rank_reserved_native_reward_"
        "correct400_macro0001_20260811"
    ),
    "cycle1_controls": {
        condition: (
            "runs/outputs/pi05_v6_qv_rank_reserved_native_reward_"
            f"{label}400_macro0001_20260811"
        )
        for condition, label in {
            "same_task_other": "same_task_other",
            "cross_suite_wrong": "cross_suite_wrong",
            "shuffled": "shuffled",
            "reversed": "reversed",
            "no_video": "no_video",
        }.items()
    },
}
_IMMUTABLE_REFERENCES = {
    "old_full_rank_macro0_correct": {
        "root": (
            "runs/outputs/pi05_v6_balanced_causal_condition_residual_correct400_"
            "noreplacement_seed7_method_macro0000_6b5f7a6_20260810"
        ),
        "commit": "6b5f7a6ad6ef1a778205071f38faec9f936cf54e",
        "correct": 134,
        "breadth": 6,
    },
    "old_native_reward_cycle1_correct": {
        "root": (
            "runs/outputs/pi05_v6_reward_credit_program_cotangent_"
            "correct400_cycle0001_20260810"
        ),
        "commit": "e3857f73ce92fa7f790a7e49f8166d7e5ef5b9e5",
        "correct": 134,
        "breadth": 6,
    },
}
_EVALUATION_BASE = {
    "throughput_policy": (
        "highest_measured_end_to_end_loras_per_second_with_memory_headroom"
    ),
    "required_writer_model_batch_sizes": [8, 16, 32],
    "minimum_smoke_writer_model_batch_size": 8,
    "immutable_references": _IMMUTABLE_REFERENCES,
    "registered_roots": _REGISTERED_ROOTS,
    "gates": _GATES,
}
_TOP_LEVEL = {
    "schema_version",
    "status",
    "method",
    "base_writer_authority",
    "design_authority",
    "generation_evidence",
    "compiler",
    "assets",
    "evaluation",
    "content_hash_policy",
}


def _repo_file(relative: str, *, label: str, expected_bytes: int | None = None) -> Path:
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ExpertManifoldError(f"{label} escaped the repository") from error
    if not path.is_file() or path.is_symlink():
        raise ExpertManifoldError(f"{label} is missing or not a regular file")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ExpertManifoldError(f"{label} byte count changed")
    return path


def rank_reserved_output_path(
    value: object,
    *,
    label: str,
    expected_bytes: int | None = None,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    """Resolve one lexical runs/outputs path through a frozen-worktree symlink."""

    if not isinstance(value, str) or not value:
        raise ExpertManifoldError(f"{label} path changed")
    relative = Path(value)
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("runs", "outputs")
        or any(part in {"", ".", ".."} for part in relative.parts)
        or require_file
        and require_directory
    ):
        raise ExpertManifoldError(f"{label} escaped canonical outputs")
    lexical = REPO_ROOT / relative
    path = lexical.resolve()
    outputs = (REPO_ROOT / "runs/outputs").resolve()
    try:
        path.relative_to(outputs)
    except ValueError as error:
        raise ExpertManifoldError(f"{label} escaped canonical outputs") from error
    if lexical.is_symlink():
        raise ExpertManifoldError(f"{label} is an unsupported direct symlink")
    if require_file and not path.is_file():
        raise ExpertManifoldError(f"{label} is missing or not a regular file")
    if require_directory and not path.is_dir():
        raise ExpertManifoldError(f"{label} is missing or not a directory")
    if expected_bytes is not None and (
        not path.is_file() or path.stat().st_size != expected_bytes
    ):
        raise ExpertManifoldError(f"{label} byte count changed")
    return path


_EXPECTED_PROGRAM_SOURCE = {
    "path": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_formal_"
        "cycle0to2_r6_k4_nmc4_b8_balanced_20260810/checkpoints/macro_00000001"
    ),
    "manifest": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_formal_"
        "cycle0to2_r6_k4_nmc4_b8_balanced_20260810/checkpoints/"
        "macro_00000001/manifest.json"
    ),
    "manifest_bytes": 15_880,
    "manifest_schema": "ember_pi05_v6_reward_credit_program_cotangent_checkpoint_v4",
    "run_schema": "ember_pi05_v6_reward_credit_program_cotangent_run_v1",
    "training_commit": "e3857f73ce92fa7f790a7e49f8166d7e5ef5b9e5",
    "config_schema": V6_PRIOR_CONFIG_SCHEMA,
    "world_size": 6,
    "next_macro": 1,
    "metrics_rows": 1,
}
_EXPECTED_PROGRAM_MEMORY = {
    "path": (
        "runs/outputs/pi05_v6_reward_credit_program_cotangent_formal_"
        "cycle0to2_r6_k4_nmc4_b8_balanced_20260810/checkpoints/"
        "macro_00000001/program_memory.safetensors"
    ),
    "bytes": 83_886_184,
    "tensor_count": 1,
    "key": "program_memory.value",
    "dtype": "F32",
    "shape": [256, 320, 256],
    "value_count": 20_971_520,
}


def _read_program_reference(
    path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    path = path.resolve()
    if path != RANK_RESERVED_PROGRAM_REFERENCE.resolve() or not path.is_file():
        raise ExpertManifoldError("non-canonical rank-reserved Program reference")
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid rank-reserved Program reference") from error
    valid = (
        isinstance(reference, dict)
        and set(reference)
        == {
            "schema_version",
            "source_checkpoint",
            "program_memory",
            "copy_or_symlink",
            "optimizer_or_rng_read",
            "content_hash_policy",
        }
        and reference.get("schema_version") == RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA
        and reference.get("copy_or_symlink") is False
        and reference.get("optimizer_or_rng_read") is False
        and reference.get("content_hash_policy") == "disabled_by_owner"
    )
    source = reference.get("source_checkpoint") if isinstance(reference, dict) else None
    memory = reference.get("program_memory") if isinstance(reference, dict) else None
    if (
        not valid
        or not isinstance(source, Mapping)
        or not isinstance(memory, Mapping)
        or dict(source) != _EXPECTED_PROGRAM_SOURCE
        or dict(memory) != _EXPECTED_PROGRAM_MEMORY
    ):
        raise ExpertManifoldError("rank-reserved Program reference changed")
    return reference, source, memory


def _validate_program_manifest(
    source: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> tuple[Path, Path]:
    checkpoint_dir = rank_reserved_output_path(
        source["path"],
        label="rank-reserved source checkpoint",
        require_directory=True,
    )
    manifest_path = rank_reserved_output_path(
        str(source["manifest"]),
        label="rank-reserved source manifest",
        expected_bytes=int(source["manifest_bytes"]),
        require_file=True,
    )
    if manifest_path.parent != checkpoint_dir:
        raise ExpertManifoldError("rank-reserved source checkpoint identity changed")
    manifest = read_json(manifest_path)
    checkpoint_contract = manifest.get("checkpoint_contract", {})
    valid = (
        manifest.get("schema_version") == source["manifest_schema"]
        and isinstance(checkpoint_contract, Mapping)
        and checkpoint_contract.get("run_schema") == source["run_schema"]
        and checkpoint_contract.get("mode") == "formal"
        and checkpoint_contract.get("git_commit") == source["training_commit"]
        and checkpoint_contract.get("config", {}).get("schema")
        == source["config_schema"]
        and int(manifest.get("world_size", -1)) == int(source["world_size"])
        and int(manifest.get("next_macro", -1)) == int(source["next_macro"])
        and int(manifest.get("metrics_rows", -1)) == int(source["metrics_rows"])
        and manifest.get("program_memory_shape") == memory["shape"]
        and int(manifest.get("files", {}).get("program_memory.safetensors", -1))
        == int(memory["bytes"])
    )
    if not valid:
        raise ExpertManifoldError("rank-reserved source manifest changed")
    return checkpoint_dir, manifest_path


def _validate_program_tensor(
    memory: Mapping[str, Any],
    checkpoint_dir: Path,
) -> Path:
    memory_path = rank_reserved_output_path(
        str(memory["path"]),
        label="rank-reserved Program tensor",
        expected_bytes=int(memory["bytes"]),
        require_file=True,
    )
    if memory_path.parent != checkpoint_dir:
        raise ExpertManifoldError("rank-reserved Program tensor ownership changed")
    try:
        with safe_open(str(memory_path), framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            value = handle.get_slice(str(memory["key"])) if keys == (memory["key"],) else None
            valid = (
                value is not None
                and str(value.get_dtype()) == memory["dtype"]
                and list(value.get_shape()) == memory["shape"]
                and math.prod(value.get_shape()) == int(memory["value_count"])
            )
            if not valid:
                raise ExpertManifoldError("rank-reserved Program tensor header changed")
    except ExpertManifoldError:
        raise
    except Exception as error:
        raise ExpertManifoldError(
            "rank-reserved Program tensor header changed"
        ) from error
    return memory_path


def load_rank_reserved_program_reference(path: Path) -> dict[str, Any]:
    """Validate the single sealed Program tensor by metadata and tensor header."""

    path = path.resolve()
    reference, source, memory = _read_program_reference(path)
    checkpoint_dir, manifest_path = _validate_program_manifest(source, memory)
    memory_path = _validate_program_tensor(memory, checkpoint_dir)
    return {
        **reference,
        "path": str(path),
        "program_memory": {**dict(memory), "path": str(memory_path)},
        "source_checkpoint": {
            **dict(source),
            "path": str(checkpoint_dir),
            "manifest": str(manifest_path),
        },
    }
