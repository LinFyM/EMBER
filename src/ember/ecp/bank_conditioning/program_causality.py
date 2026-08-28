"""Pre-registered fit-only correct-vs-wrong Program qualification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSplit,
)
from ember.pi05_source_checkpoint import read_json


PROGRAM_CAUSALITY_SCHEMA = (
    "ember_ecp_shared_compiler_g3_f3_program_causality_v1"
)
FAMILY_NAMES = ("q", "v", "action_in", "action_out")
ROLE_NAMES = ("meta_fit", "target_fit")


@dataclass(frozen=True)
class ProgramCausalityPair:
    """One primary fit condition and a deterministic same-role wrong Program."""

    primary: MappingCondition
    wrong: MappingCondition


def load_program_causality_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path.resolve())
    panel = contract.get("panel", {})
    gate = contract.get("gate", {})
    wall = contract.get("information_wall", {})
    valid = all(
        (
            contract.get("schema_version") == PROGRAM_CAUSALITY_SCHEMA,
            contract.get("status") == "active_preregistered",
            contract.get("model_config_schema")
            == "ember_ecp_shared_compiler_g3_v4",
            panel.get("source") == "mapping_fit_only",
            panel.get("conditions_per_fit_task") == 1,
            panel.get("condition_selection")
            == "lowest_video_demo_without_outcome_reads",
            panel.get("wrong_program_pairing")
            == "next_authority_id_within_same_meta_or_target_role_cyclic",
            panel.get("meta_fit_tasks") == 25,
            panel.get("target_fit_tasks") == 15,
            panel.get("total_pairs") == 40,
            gate.get("role_median_correct_minus_wrong_program_minimum") == 0.1,
            wall.get("teacher_use")
            == "primary_task_fit_teacher_for_both_arms",
            wall.get("wrong_program_native_bank_use") is False,
            wall.get("held_video_or_task_gradient_use") is False,
            wall.get("validation_or_test_use") is False,
            wall.get("shuffled_or_reversed_use") is False,
            wall.get("checkpoint_selection_use") is True,
        )
    )
    if not valid:
        raise ValueError("G3 Program-causality pre-registration changed")
    return contract


def program_causality_pairs(
    split: SharedCompilerMappingSplit,
) -> tuple[ProgramCausalityPair, ...]:
    """Select one outcome-independent fit row per task and rotate within role."""

    selected = {
        task: min(rows, key=lambda row: row.video_demo)
        for task, rows in split.fit_by_task.items()
    }
    pairs = []
    for role in ROLE_NAMES:
        tasks = tuple(
            sorted(
                task
                for task, condition in selected.items()
                if condition.role == role
            )
        )
        expected = 25 if role == "meta_fit" else 15
        if len(tasks) != expected:
            raise ValueError("G3 Program-causality role panel changed")
        for index, task in enumerate(tasks):
            wrong_task = tasks[(index + 1) % len(tasks)]
            pairs.append(
                ProgramCausalityPair(
                    primary=selected[task], wrong=selected[wrong_task]
                )
            )
    result = tuple(sorted(pairs, key=lambda row: row.primary.authority_id))
    if (
        len(result) != 40
        or len({row.primary.authority_id for row in result}) != 40
        or any(
            row.primary.authority_id == row.wrong.authority_id
            or row.primary.role != row.wrong.role
            for row in result
        )
    ):
        raise ValueError("G3 Program-causality pairing changed")
    return result


def program_causality_extra_costs(
    split: SharedCompilerMappingSplit,
) -> dict[tuple[int, int], int]:
    """Proxy the extra wrong-Pass-A plus wrong-compiler work for queue balance."""

    return {
        (pair.primary.authority_id, pair.primary.video_demo): (
            pair.primary.sampled_frames + pair.wrong.sampled_frames
        )
        for pair in program_causality_pairs(split)
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered or not 0.0 <= fraction <= 1.0:
        raise ValueError("Program-causality quantile input changed")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": _quantile(values, 0.5),
        "p10": _quantile(values, 0.1),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_program_causality_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        len(rows) != 40
        or len({int(row["authority_id"]) for row in rows}) != 40
    ):
        raise ValueError("Program-causality evidence did not cover 40 fit tasks")
    result: dict[str, Any] = {}
    for role in ROLE_NAMES:
        selected = [row for row in rows if row["role"] == role]
        expected = 25 if role == "meta_fit" else 15
        if len(selected) != expected:
            raise ValueError("Program-causality role evidence changed")
        differences = [
            float(row["correct"]["mean_best_recovery"])
            - float(row["wrong"]["mean_best_recovery"])
            for row in selected
        ]
        result[role] = {
            "task_count": len(selected),
            "correct_recovery": _distribution(
                [float(row["correct"]["mean_best_recovery"]) for row in selected]
            ),
            "wrong_program_recovery": _distribution(
                [float(row["wrong"]["mean_best_recovery"]) for row in selected]
            ),
            "correct_minus_wrong_program": _distribution(differences),
            "positive_task_fraction": sum(value > 0 for value in differences)
            / len(differences),
            "family_correct_minus_wrong_program": {
                family: _distribution(
                    [
                        float(row["correct"]["best_family_recovery"][family])
                        - float(row["wrong"]["best_family_recovery"][family])
                        for row in selected
                    ]
                )
                for family in FAMILY_NAMES
            },
        }
    return result


def program_causality_checks(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, bool]:
    threshold = float(
        contract["gate"][
            "role_median_correct_minus_wrong_program_minimum"
        ]
    )
    return {
        role: float(summary[role]["correct_minus_wrong_program"]["median"])
        >= threshold
        for role in ROLE_NAMES
    }
