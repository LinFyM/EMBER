"""Validate and apply successful-expert occupancy selections."""

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
MDCO_OCCUPANCY_SELECTION_SCHEMA = "ember_ecp_stage1_mdco_nonheld_occupancy_selection_v1"
ECP_STAGE1_OCCUPANCY_SELECTION_SCHEMA = (
    "ember_ecp_stage1_occupancy_complete_selection_v1"
)
ECP_STAGE1_OCCUPANCY_CAPTURE_SCHEMA = "ember_ecp_stage1_occupancy_complete_capture_v1"

ECP_STAGE1_HELD_KEYS = (
    ("libero_spatial", 0, 0),
    ("libero_spatial", 9, 9),
    ("libero_object", 8, 18),
    ("libero_goal", 5, 25),
    ("libero_10", 6, 36),
)
ECP_STAGE1_PROFILE_KEYS = (("libero_spatial", 2, 2),)


def _ecp_stage1_panel(panel: str) -> tuple[str, tuple[tuple[str, int, int], ...]]:
    if panel.startswith("profile_"):
        return panel.removeprefix("profile_"), ECP_STAGE1_PROFILE_KEYS
    return panel, ECP_STAGE1_HELD_KEYS


def ecp_stage1_occupancy_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    writer_kind: str | None,
    selection_path: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Bind the three preregistered Stage 1A rollout-observation panels."""

    panel = str(manifest.get("panel"))
    base_panel, expected_keys = _ecp_stage1_panel(panel)
    expected_writer = {
        "source_support": None,
        "independent_particle": "task_expert",
        "candidate_policy": "task_expert",
    }.get(base_panel, "unsupported")
    row_keys = {
        (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("global_task_id", -1)),
            int(row.get("init_state_id", -1)),
        )
        for row in rows
    }
    task_keys = {
        (suite, task_id, global_id) for suite, task_id, global_id in expected_keys
    }
    observed_tasks = {
        (suite, task_id, global_id) for suite, task_id, global_id, _ in row_keys
    }
    category_counts = {
        category: sum(str(row.get("category")) == category for row in rows)
        for category in ("initial", "recovery_candidate", "successful", "candidate")
    }
    valid_panel = (
        (
            base_panel == "source_support"
            and len(rows) == 12 * len(expected_keys)
            and category_counts
            == {
                "initial": 8 * len(expected_keys),
                "recovery_candidate": 4 * len(expected_keys),
                "successful": 0,
                "candidate": 0,
            }
            and all(
                {int(row[3]) for row in row_keys if row[:3] == key}
                == {0, 1, 2, 7, 14, 21, 26, 27, 28, 35, 42, 49}
                for key in task_keys
            )
        )
        or (
            base_panel == "independent_particle"
            and len(rows) == len(expected_keys)
            and category_counts
            == {
                "initial": 0,
                "recovery_candidate": 0,
                "successful": len(expected_keys),
                "candidate": 0,
            }
        )
        or (
            base_panel == "candidate_policy"
            and len(rows) == len(expected_keys)
            and category_counts
            == {
                "initial": 0,
                "recovery_candidate": 0,
                "successful": 0,
                "candidate": len(expected_keys),
            }
            and {row[3] for row in row_keys} == {0}
        )
    )
    if (
        args.mode != "formal"
        or args.role != "development_train"
        or manifest.get("status") != "preregistered_or_fixed_after_member_qualification"
        or writer_kind != expected_writer
        or observed_tasks != task_keys
        or len(row_keys) != len(rows)
        or any(not 0 <= row[3] < 50 for row in row_keys)
        or not valid_panel
    ):
        raise Pi05EvaluationError("ECP Stage 1 occupancy selection changed")

    by_task = {(str(task.suite), int(task.task_id)): task for task in tasks}
    selected = []
    covered = set()
    for suite, task_id, global_id in expected_keys:
        task = by_task.get((suite, task_id))
        if task is None:
            raise Pi05EvaluationError("ECP Stage 1 held task is unavailable")
        state_ids = tuple(
            state_id
            for state_id in task.init_state_ids
            if (suite, task_id, global_id, int(state_id)) in row_keys
        )
        selected.append(replace(task, init_state_ids=state_ids))
        covered.update((suite, task_id, global_id, int(value)) for value in state_ids)
    if covered != row_keys:
        raise Pi05EvaluationError("ECP Stage 1 occupancy rows are outside train24")
    return tuple(selected), {
        "schema_version": ECP_STAGE1_OCCUPANCY_CAPTURE_SCHEMA,
        "selection_path": str(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "panel": panel,
        "selected_rows": len(rows),
        "selected_tasks": len(expected_keys),
        "category_counts": category_counts,
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "training_gradient_use": True,
        "gradient_scope": (
            "fold0 held5 privileged Stage 1B realization oracle only"
            if expected_keys == ECP_STAGE1_HELD_KEYS
            else "fold0 fit-task numerical and resource profile only"
        ),
        "adapter_is_declared_subset": writer_kind == "task_expert",
        "validation_use": False,
        "test_use": False,
    }


def _selection_contract(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[bool, dict[str, int], set[tuple[str, int, int]]]:
    categories = {
        category: sum(row.get("category") == category for row in rows)
        for category in (
            "gained",
            "retained_success",
            "direct_success_fallback",
        )
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
    mdco = manifest.get("schema_version") == MDCO_OCCUPANCY_SELECTION_SCHEMA
    if mdco:
        per_task = {
            key: sum(
                (str(row.get("suite")), int(row.get("task_id", -1))) == key
                for row in rows
            )
            for key in task_categories
        }
        valid = (
            manifest.get("status") == "preregistered_fixed_successful_rows"
            and int(manifest.get("task_count", -1)) == 71
            and int(manifest.get("selected_rows", -1)) == len(rows)
            and len(task_categories) == 71
            and 71 <= len(rows) <= 142
            and all(count in {1, 2} for count in per_task.values())
            and all(row.get("direct_success") is True for row in rows)
            and all(
                str(row.get("category"))
                in {"gained", "retained_success", "direct_success_fallback"}
                for row in rows
            )
        )
    else:
        valid = (
            len(rows) == 8
            and categories
            == {
                "gained": 4,
                "retained_success": 4,
                "direct_success_fallback": 0,
            }
            and len(task_categories) == 4
            and all(
                values == {"gained", "retained_success"}
                for values in task_categories.values()
            )
        )
    return valid, categories, keys


def _selected_tasks(
    tasks: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    keys: set[tuple[str, int, int]],
) -> tuple[Any, ...]:
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
        raise Pi05EvaluationError(
            "successful-expert occupancy selection is outside meta-train"
        )
    return tuple(selected)


def successful_expert_occupancy_tasks(
    args: Any,
    tasks: Sequence[Any],
    *,
    output_dir: Path,
    writer_kind: str | None,
    selection_path: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    valid, categories, keys = _selection_contract(manifest, rows)
    mdco = manifest.get("schema_version") == MDCO_OCCUPANCY_SELECTION_SCHEMA
    expected_role = "nonheld_meta" if mdco else "nonheld_meta_train"
    if (
        args.mode != "formal"
        or args.role != expected_role
        or writer_kind != "task_expert"
        or len(keys) != len(rows)
        or not valid
    ):
        raise Pi05EvaluationError("successful-expert occupancy selection changed")
    selected = _selected_tasks(tasks, rows, keys)
    return selected, {
        "schema_version": SUCCESSFUL_EXPERT_OCCUPANCY_CAPTURE_SCHEMA,
        "selection_path": str(selection_path),
        "selection_bytes": selection_path.stat().st_size,
        "source_results": manifest.get("source_results"),
        "direct_results": manifest.get("direct_results"),
        "category_counts": categories,
        "selected_rows": len(rows),
        "selected_tasks": len(selected),
        "mapping_diverse_compiler_oracle": mdco,
        "trajectory_root": str((output_dir / "occupancy_trajectories").resolve()),
        "training_gradient_use": mdco,
        "gradient_scope": (
            "fit90 privileged Stage 1 policy mapping and support only" if mdco else None
        ),
        "held_data_use": False,
        "claim_boundary": manifest.get("claim_boundary"),
    }
