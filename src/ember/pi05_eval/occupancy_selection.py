"""Validate the retained successful-expert occupancy panel."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError


SUCCESSFUL_EXPERT_OCCUPANCY_SELECTION_SCHEMA = (
    "ember_successful_expert_occupancy_selection_v1"
)
SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA = (
    "ember_successful_expert_occupancy_capture_v1"
)
G3_VERIFIED_MEMBER_OCCUPANCY_SELECTION_SCHEMA = (
    "ember_ecp_g3_verified_member_occupancy_selection_v1"
)
G3_VERIFIED_MEMBER_OCCUPANCY_CAPTURE_SCHEMA = (
    "ember_ecp_g3_verified_member_occupancy_capture_v1"
)


def _selection_contract(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, int], set[tuple[str, int, int]]]:
    categories = {
        category: sum(row.get("category") == category for row in rows)
        for category in ("gained", "retained_success", "direct_success_fallback")
    }
    keys = {
        (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("init_state_id", -1)),
        )
        for row in rows
    }
    task_categories: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        task_categories.setdefault(
            (str(row.get("suite")), int(row.get("task_id", -1))), set()
        ).add(str(row.get("category")))
    valid = (
        manifest.get("schema_version") == SUCCESSFUL_EXPERT_OCCUPANCY_SELECTION_SCHEMA
        and len(rows) == 8
        and len(keys) == len(rows)
        and categories
        == {"gained": 4, "retained_success": 4, "direct_success_fallback": 0}
        and len(task_categories) == 4
        and all(
            values == {"gained", "retained_success"}
            for values in task_categories.values()
        )
    )
    if not valid:
        raise Pi05EvaluationError("successful-expert occupancy selection changed")
    return categories, keys


def successful_expert_occupancy_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    adapter_kind: str | None,
    selection_path: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    categories, keys = _selection_contract(manifest, rows)
    if (
        args.mode != "formal"
        or args.role != "nonheld_meta_train"
        or adapter_kind != "task_expert"
    ):
        raise Pi05EvaluationError("successful-expert occupancy request changed")
    selected = []
    covered = set()
    for task in tasks:
        task_key = (str(task.suite), int(task.task_id))
        task_rows = [
            row
            for row in rows
            if (str(row.get("suite")), int(row.get("task_id", -1))) == task_key
        ]
        if task_rows and any(row.get("language") != task.language for row in task_rows):
            raise Pi05EvaluationError("successful-expert task language changed")
        state_ids = tuple(
            state_id
            for state_id in task.init_state_ids
            if (*task_key, int(state_id)) in keys
        )
        if state_ids:
            selected.append(replace(task, init_state_ids=state_ids))
            covered.update((*task_key, int(state_id)) for state_id in state_ids)
    if covered != keys:
        raise Pi05EvaluationError("successful-expert selection is outside meta-train")
    return tuple(selected), {
        "schema_version": SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA,
        "selection_path": str(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "source_results": manifest.get("source_results"),
        "direct_results": manifest.get("direct_results"),
        "category_counts": categories,
        "selected_rows": len(rows),
        "selected_tasks": len(selected),
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "training_gradient_use": False,
        "held_data_use": False,
        "claim_boundary": manifest.get("claim_boundary"),
    }


def g3_verified_member_occupancy_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    adapter_kind: str | None,
    selection_path: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Select one already successful initial state per target-fit member."""

    step = int(manifest.get("member_step", -1))
    expected = 19 if step == 1000 else 18 if step == 2000 else -1
    keys = {
        (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("init_state_id", -1)),
        )
        for row in rows
    }
    source = manifest.get("source_results", {})
    source_path = Path(str(source.get("path", ""))).resolve()
    valid = (
        manifest.get("schema_version")
        == G3_VERIFIED_MEMBER_OCCUPANCY_SELECTION_SCHEMA
        and manifest.get("status") == "sealed_g3_verified_member_occupancy"
        and len(rows) == expected
        and len(keys) == len(rows)
        and len({int(row.get("global_task_id", -1)) for row in rows}) == expected
        and all(
            row.get("success") is True
            and int(row.get("member_step", -1)) == step
            and str(row.get("language", ""))
            for row in rows
        )
        and source_path.is_file()
        and source_path.stat().st_size == int(source.get("bytes", -1))
    )
    if not valid:
        raise Pi05EvaluationError("G3 verified-member occupancy selection changed")
    if (
        args.mode != "formal"
        or args.role != "development_train"
        or adapter_kind != "task_expert"
        or int(getattr(args, "task_expert_step", -1)) != step
    ):
        raise Pi05EvaluationError("G3 verified-member occupancy request changed")

    selected = []
    covered = set()
    for task in tasks:
        task_key = (str(task.suite), int(task.task_id))
        task_rows = [
            row
            for row in rows
            if (str(row.get("suite")), int(row.get("task_id", -1))) == task_key
        ]
        if task_rows and any(
            row.get("language") != task.language for row in task_rows
        ):
            raise Pi05EvaluationError("G3 verified-member language changed")
        state_ids = tuple(
            state_id
            for state_id in task.init_state_ids
            if (*task_key, int(state_id)) in keys
        )
        if state_ids:
            selected.append(replace(task, init_state_ids=state_ids))
            covered.update((*task_key, int(state_id)) for state_id in state_ids)
    if covered != keys:
        raise Pi05EvaluationError("G3 occupancy selection escaped train24")
    return tuple(selected), {
        "schema_version": G3_VERIFIED_MEMBER_OCCUPANCY_CAPTURE_SCHEMA,
        "diagnostic_subset": "g3_verified_member_occupancy",
        "selection_path": str(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "source_results": dict(source),
        "member_step": step,
        "selected_rows": len(rows),
        "selected_tasks": len(selected),
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "training_gradient_use": True,
        "held_data_use": False,
        "claim_boundary": manifest.get("claim_boundary"),
    }
