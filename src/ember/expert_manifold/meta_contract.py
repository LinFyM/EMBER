"""Non-held LIBERO-90 authority for privileged meta-task experts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ember.meta_protocol import load_meta_protocol
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
META_EXPERT_CONFIG_SCHEMA = "ember_pi05_nonheld_meta_expert_bank_v1"


@dataclass(frozen=True)
class MetaExpertSpec:
    ordinal: int
    task_id: int
    split_role: str
    language: str
    path: Path
    expected_bytes: int


def _authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def meta_worker_assignments(
    formal: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    try:
        assignments = tuple(
            tuple(int(value) for value in row)
            for row in formal["worker_task_indices"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("meta-expert worker assignments are malformed") from error
    flat = tuple(value for row in assignments for value in row)
    if (
        len(assignments) != int(formal.get("allowed_worker_count", -1))
        or any(tuple(sorted(set(row))) != row for row in assignments)
        or len(flat) != 71
        or set(flat) != set(range(71))
    ):
        raise ValueError("meta-expert workers do not partition all 71 tasks")
    return assignments


def meta_expert_config_is_valid(config: Mapping[str, Any]) -> bool:
    experts = config.get("task_experts", {})
    formal = experts.get("formal_run", {})
    profile = formal.get("profile_evidence", {})
    authorities = config.get("authorities", {})
    information = config.get("information_wall", {})
    try:
        meta_worker_assignments(formal)
    except ValueError:
        return False
    return (
        config.get("schema_version") == META_EXPERT_CONFIG_SCHEMA
        and config.get("status") == "sealed_nonheld_meta_expert_training_contract"
        and set(authorities)
        == {
            "meta_protocol",
            "source_manifest",
            "evaluation_config",
            "lora_contract",
            "source_base_config",
        }
        and all(_authority_path(config, name).is_file() for name in authorities)
        and information.get("expert_action_roles")
        == ["meta_train", "meta_validation_oracle"]
        and int(information.get("target40_action_reads", -1)) == 0
        and information.get("deployment_uses_privileged_experts") is False
        and information.get("task_identity_role") == "sampler_ownership_only"
        and int(experts.get("task_count", -1)) == 71
        and int(experts.get("episodes_per_task", -1)) == 50
        and experts.get("demo_indices") == [0, 49]
        and int(experts.get("action_chunk_size", -1)) == 50
        and experts.get("lora_topology")
        == "configs/pi05_lora_v1.json:38targets:rank16"
        and experts.get("task_parameter_sharing") == "none"
        and formal.get("status") == "sealed"
        and int(formal.get("total_steps", -1)) == 2000
        and int(formal.get("per_task_batch_size", -1)) == 16
        and formal.get("checkpoint_steps") == [1000, 2000]
        and int(formal.get("allowed_worker_count", -1)) == 6
        and int(formal.get("default_stop_step", -1)) == 1000
        and formal.get("stage_stop_steps") == [1000, 2000]
        and formal.get("checkpoint_policy")
        == "uniform_global_stage_no_task_specific_mixing"
        and profile.get("device") == "NVIDIA A40"
        and int(profile.get("per_task_batch_size", -1)) == 16
        and profile.get("workload_shape_matches_train24_experts") is True
        and config.get("content_hash_policy") == "disabled_by_owner"
    )


def load_meta_expert_specs(
    config: Mapping[str, Any], data_root: Path
) -> tuple[MetaExpertSpec, ...]:
    rows = meta_expert_rows(config)
    result = []
    for row in rows:
        path = data_root / str(row["hdf5_filename"])
        expected_bytes = int(row["hdf5_bytes"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(
                f"meta-expert HDF5 path or size changed: {int(row['task_id'])}"
            )
        result.append(
            MetaExpertSpec(
                ordinal=int(row["ordinal"]),
                task_id=int(row["task_id"]),
                split_role=str(row["split_role"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=expected_bytes,
            )
        )
    return tuple(result)


def meta_expert_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Resolve stable bank identities without opening numeric task data."""

    protocol = load_meta_protocol(_authority_path(config, "meta_protocol"))
    manifest = read_json(_authority_path(config, "source_manifest"))
    rows = {int(row["task_index"]): row for row in manifest.get("tasks", [])}
    active_ids = tuple(int(value) for value in protocol["active_source_task_ids"])
    held_fold = int(protocol["roles"]["architecture_validation"]["default_fold"])
    held_ids = {
        int(value)
        for row in protocol["task_level_cross_validation"]["folds"]
        if int(row["fold"]) == held_fold
        for value in row["task_ids"]
    }
    if set(rows) != set(active_ids):
        raise ValueError("source manifest did not resolve non-held meta71")
    result: list[dict[str, Any]] = []
    for ordinal, task_id in enumerate(active_ids):
        row = rows[task_id]
        hdf5 = row["hdf5"]
        result.append(
            {
                "ordinal": ordinal,
                "global_task_id": task_id,
                "suite": "libero_90",
                "task_id": task_id,
                "split_role": (
                    "meta_validation_oracle" if task_id in held_ids else "meta_train"
                ),
                "language": str(row["language"]),
                "hdf5_filename": str(hdf5["filename"]),
                "hdf5_bytes": int(hdf5["bytes"]),
            }
        )
    return tuple(result)
