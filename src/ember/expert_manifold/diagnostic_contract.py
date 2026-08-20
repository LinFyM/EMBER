"""Sealed validation-task expert oracle used only for post-hoc diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_EXPERT_CONFIG_SCHEMA = "ember_pi05_validation_expert_diagnostic_v1"


@dataclass(frozen=True)
class ValidationExpertSpec:
    ordinal: int
    global_task_id: int
    suite: str
    task_id: int
    language: str
    path: Path
    expected_bytes: int


def validation_worker_assignments(
    formal: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    try:
        assignments = tuple(
            tuple(int(value) for value in row) for row in formal["worker_task_indices"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "validation-expert worker assignments are malformed"
        ) from error
    flat = tuple(value for row in assignments for value in row)
    if (
        len(assignments) != int(formal.get("allowed_worker_count", -1))
        or any(tuple(sorted(set(row))) != row for row in assignments)
        or len(flat) != 8
        or set(flat) != set(range(8))
    ):
        raise ValueError("validation-expert workers do not partition validation8")
    return assignments


def validation_expert_config_is_valid(config: Mapping[str, Any]) -> bool:
    experts = config.get("task_experts", {})
    formal = experts.get("formal_run", {})
    profile = formal.get("profile_evidence", {})
    information = config.get("information_wall", {})
    authorities = config.get("authorities", {})
    try:
        validation_worker_assignments(formal)
    except ValueError:
        return False
    return (
        config.get("schema_version") == VALIDATION_EXPERT_CONFIG_SCHEMA
        and config.get("status") == "sealed_validation_expert_diagnostic"
        and set(authorities)
        == {
            "target_data_manifest",
            "evaluation_config",
            "lora_contract",
            "source_base_config",
        }
        and all(
            (REPO_ROOT / str(row.get("path", ""))).is_file()
            for row in authorities.values()
        )
        and information.get("expert_action_role") == "validation_diagnostic"
        and information.get("shared_training_gradient_use") is False
        and information.get("writer_or_decoder_gradient_use") is False
        and information.get("checkpoint_selection_use") is False
        and int(information.get("test_action_reads", -1)) == 0
        and information.get("deployment_uses_privileged_experts") is False
        and int(experts.get("task_count", -1)) == 8
        and int(experts.get("episodes_per_task", -1)) == 50
        and experts.get("demo_indices") == [0, 49]
        and int(experts.get("action_chunk_size", -1)) == 50
        and experts.get("lora_topology") == "configs/pi05_lora_v1.json:38targets:rank16"
        and experts.get("task_parameter_sharing") == "none"
        and formal.get("status") == "sealed"
        and int(formal.get("total_steps", -1)) == 2000
        and int(formal.get("per_task_batch_size", -1)) == 16
        and formal.get("checkpoint_steps") == [1000, 2000]
        and int(formal.get("allowed_worker_count", -1)) == 6
        and int(formal.get("default_stop_step", -1)) == 1000
        and formal.get("stage_stop_steps") == [1000, 2000]
        and int(formal.get("diagnostic_evaluation_step", -1)) == 2000
        and formal.get("checkpoint_policy")
        == "step1000_resume_boundary_step2000_only_diagnostic_no_selection"
        and profile.get("device") == "NVIDIA A40"
        and int(profile.get("per_task_batch_size", -1)) == 16
        and profile.get("workload_shape_matches_train24_experts") is True
        and config.get("content_hash_policy") == "disabled_by_owner"
    )


def validation_expert_rows(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    manifest_path = REPO_ROOT / str(
        config["authorities"]["target_data_manifest"]["path"]
    )
    manifest = read_json(manifest_path)
    selected = sorted(
        (
            row
            for row in manifest.get("tasks", ())
            if row.get("split_role") == "validation"
        ),
        key=lambda row: int(row["global_task_id"]),
    )
    if len(selected) != 8:
        raise ValueError("target manifest did not resolve validation8")
    return tuple(
        {
            "ordinal": ordinal,
            "global_task_id": int(row["global_task_id"]),
            "suite": str(row["suite"]),
            "task_id": int(row["task_id"]),
            "split_role": "validation_diagnostic",
            "language": str(row["language"]),
            "hdf5_relative_path": str(row["hdf5"]["relative_path"]),
            "hdf5_bytes": int(row["hdf5"]["bytes"]),
        }
        for ordinal, row in enumerate(selected)
    )


def load_validation_expert_specs(
    config: Mapping[str, Any], data_root: Path
) -> tuple[ValidationExpertSpec, ...]:
    result = []
    for row in validation_expert_rows(config):
        path = data_root / str(row["hdf5_relative_path"])
        expected_bytes = int(row["hdf5_bytes"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(
                f"validation expert HDF5 path or size changed: "
                f"{int(row['global_task_id'])}"
            )
        result.append(
            ValidationExpertSpec(
                ordinal=int(row["ordinal"]),
                global_task_id=int(row["global_task_id"]),
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=expected_bytes,
            )
        )
    return tuple(result)
