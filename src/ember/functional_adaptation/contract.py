"""Data roles and task-level folds for the fixed functional-adaptation route."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
META_PROTOCOL_SCHEMA = "ember_libero90_nonheld_meta_v1"


class FunctionalAdaptationContractError(RuntimeError):
    """Raised when the non-held meta-task boundary changes."""


@dataclass(frozen=True)
class MetaTask:
    """One semantically audited LIBERO-90 task available to meta-training."""

    task_id: int
    language: str
    task_name: str
    hdf5_filename: str
    hdf5_bytes: int


@dataclass(frozen=True)
class MetaTaskSplit:
    """One task-level leave-out fold with disjoint train and validation tasks."""

    held_out_fold: int
    train: tuple[MetaTask, ...]
    validation: tuple[MetaTask, ...]


def _authority_path(protocol: Mapping[str, Any], name: str) -> Path:
    try:
        relative = str(protocol["authorities"][name]["path"])
    except (KeyError, TypeError) as error:
        raise FunctionalAdaptationContractError(
            f"missing meta-task authority: {name}"
        ) from error
    path = REPO_ROOT / relative
    if not path.is_file():
        raise FunctionalAdaptationContractError(
            f"meta-task authority does not exist: {relative}"
        )
    return path


def _folds(protocol: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    rows = protocol["task_level_cross_validation"]["folds"]
    folds = {
        int(row["fold"]): tuple(int(task_id) for task_id in row["task_ids"])
        for row in rows
    }
    if set(folds) != set(range(5)):
        raise FunctionalAdaptationContractError("meta-task folds must be 0..4")
    return folds


def _validate_allowlist(
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    overlap_audit: Mapping[str, Any],
) -> tuple[int, ...]:
    active = tuple(int(value) for value in protocol["active_source_task_ids"])
    excluded = tuple(
        int(value) for value in protocol["excluded_target_overlap_task_ids"]
    )
    source_active = tuple(source_manifest["summary"]["active_source_task_ids"])
    audit_active = tuple(overlap_audit["summary"]["active_source_task_ids"])
    audit_excluded = tuple(overlap_audit["summary"]["excluded_source_task_ids"])
    complete_partition = (
        set(active).isdisjoint(excluded)
        and set(active).union(excluded) == set(range(90))
    )
    if (
        len(active) != 71
        or tuple(sorted(set(active))) != active
        or len(excluded) != 19
        or not complete_partition
        or source_active != active
        or audit_active != active
        or audit_excluded != excluded
    ):
        raise FunctionalAdaptationContractError(
            "meta-task allowlist differs from the semantic overlap audit"
        )
    return active


def _validate_fold_partition(
    protocol: Mapping[str, Any], active: tuple[int, ...]
) -> None:
    folds = _folds(protocol)
    folded = tuple(task_id for fold in range(5) for task_id in folds[fold])
    roles = protocol["roles"]
    held_out = int(roles["architecture_validation"]["default_fold"])
    training_folds = {
        int(value)
        for value in roles["video_adaptation_meta_training"][
            "default_training_folds"
        ]
    }
    if len(folded) != len(active) or set(folded) != set(active):
        raise FunctionalAdaptationContractError(
            "meta-task folds do not partition the allowlist"
        )
    if training_folds != set(folds).difference({held_out}):
        raise FunctionalAdaptationContractError(
            "default meta train/validation folds overlap or leave a gap"
        )


def _validate_process_contrasts(
    protocol: Mapping[str, Any], active: tuple[int, ...]
) -> None:
    active_set = set(active)
    groups = (
        group
        for row in protocol["process_contrast_groups"]
        for group in row["task_groups"]
    )
    if any(
        len(group) < 2 or not set(map(int, group)).issubset(active_set)
        for group in groups
    ):
        raise FunctionalAdaptationContractError(
            "process contrast escaped the non-held allowlist"
        )


def load_meta_protocol(path: Path) -> dict[str, Any]:
    """Load the active non-held meta protocol and verify its scientific split."""

    protocol = read_json(path.resolve())
    if (
        protocol.get("schema_version") != META_PROTOCOL_SCHEMA
        or protocol.get("status") != "active_successor_data_contract"
        or protocol.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise FunctionalAdaptationContractError("meta-task protocol changed")

    source_manifest = read_json(_authority_path(protocol, "source_manifest"))
    overlap_audit = read_json(_authority_path(protocol, "overlap_audit"))
    _authority_path(protocol, "target_protocol")
    _authority_path(protocol, "target_manifest")
    _authority_path(protocol, "lora_contract")
    active = _validate_allowlist(protocol, source_manifest, overlap_audit)
    _validate_fold_partition(protocol, active)
    _validate_process_contrasts(protocol, active)
    return protocol


def _meta_tasks(protocol: Mapping[str, Any]) -> dict[int, MetaTask]:
    manifest = read_json(_authority_path(protocol, "source_manifest"))
    active = set(int(value) for value in protocol["active_source_task_ids"])
    tasks = {
        int(row["task_index"]): MetaTask(
            task_id=int(row["task_index"]),
            language=str(row["language"]),
            task_name=str(row["task_name"]),
            hdf5_filename=str(row["hdf5"]["filename"]),
            hdf5_bytes=int(row["hdf5"]["bytes"]),
        )
        for row in manifest["tasks"]
        if int(row["task_index"]) in active
    }
    if set(tasks) != active:
        raise FunctionalAdaptationContractError(
            "source manifest did not resolve every allowed meta task"
        )
    return tasks


def meta_task_split(
    protocol: Mapping[str, Any], held_out_fold: int | None = None
) -> MetaTaskSplit:
    """Resolve one disjoint task-level fold without consulting outcomes."""

    folds = _folds(protocol)
    selected = (
        int(protocol["roles"]["architecture_validation"]["default_fold"])
        if held_out_fold is None
        else int(held_out_fold)
    )
    if selected not in folds:
        raise FunctionalAdaptationContractError("held-out meta fold is outside 0..4")
    tasks = _meta_tasks(protocol)
    validation_ids = set(folds[selected])
    active_order = tuple(int(value) for value in protocol["active_source_task_ids"])
    train = tuple(
        tasks[task_id] for task_id in active_order if task_id not in validation_ids
    )
    validation = tuple(
        tasks[task_id] for task_id in active_order if task_id in validation_ids
    )
    return MetaTaskSplit(
        held_out_fold=selected,
        train=train,
        validation=validation,
    )
