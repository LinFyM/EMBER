"""Strict CPU-only analysis of paired PI05 Writer evaluation results."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.eval_adapters import EXPERT_MANIFOLD_WRITER_KIND, paired_writer_identity
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    task_video_mapping,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_queue import publish_json_exclusive
from ember.pi05_eval_results import AGGREGATE_SCHEMA
from ember.pi05_target_data import SUITE_ORDER


CHECKPOINT_CURVE_SCHEMA = "ember_pi05_v6_prior_checkpoint_curve_analysis_v1"
SIX_ARM_AUDIT_SCHEMA = "ember_pi05_v6_prior_six_arm_paired_analysis_v1"
CHECKPOINT_MACROS = (0, 10, 25, 50)
SIX_ARM_CONDITIONS = (
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "shuffled",
    "reversed",
    "no_video",
)

EpisodeKey = tuple[str, int, int]
TaskKey = tuple[str, int]


def _fail(message: str) -> None:
    raise Pi05EvaluationError(message)


def _suite_sort_key(suite: str) -> tuple[int, str]:
    try:
        return SUITE_ORDER.index(suite), suite
    except ValueError:
        return len(SUITE_ORDER), suite


def _episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])


def _task_key(row: Mapping[str, Any]) -> TaskKey:
    return str(row["suite"]), int(row["task_id"])


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[EpisodeKey, Mapping[str, Any]]:
    indexed: dict[EpisodeKey, Mapping[str, Any]] = {}
    for row in rows:
        key = _episode_key(row)
        if key in indexed or type(row.get("success")) is not bool:
            _fail("paired analysis rows are duplicated or have an invalid outcome")
        indexed[key] = row
    return indexed


def _task_sort_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    suite = str(row["suite"])
    return (*_suite_sort_key(suite), int(row["task_id"]))


def _summary_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    episodes = len(rows)
    return {
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes if episodes else None,
    }


def summarize_panel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one already validated, uniquely keyed rollout panel."""

    indexed = _index_rows(rows)
    task_keys = sorted({_task_key(row) for row in indexed.values()}, key=lambda key: (*_suite_sort_key(key[0]), key[1]))
    per_task: list[dict[str, Any]] = []
    for suite, task_id in task_keys:
        selected = [row for row in indexed.values() if _task_key(row) == (suite, task_id)]
        first = selected[0]
        per_task.append(
            {
                "suite": suite,
                "task_id": task_id,
                "split_role": first.get("split_role"),
                "language": first.get("language"),
                **_summary_row(selected),
            }
        )
    per_suite = []
    for suite in sorted({key[0] for key in task_keys}, key=_suite_sort_key):
        selected = [row for row in indexed.values() if str(row["suite"]) == suite]
        per_suite.append({"suite": suite, **_summary_row(selected)})
    top3 = sorted(
        per_task,
        key=lambda row: (-int(row["successes"]), *_task_sort_key(row)),
    )[:3]
    total_successes = sum(int(row["successes"]) for row in per_task)
    top3_successes = sum(int(row["successes"]) for row in top3)
    return {
        "overall": _summary_row(list(indexed.values())),
        "per_task": per_task,
        "per_suite": per_suite,
        "nonzero_task_breadth": sum(int(row["successes"]) > 0 for row in per_task),
        "top3_tasks": top3,
        "top3_successes": top3_successes,
        "top3_success_share": (
            top3_successes / total_successes if total_successes else None
        ),
    }


def exact_mcnemar_two_sided_p(gained: int, lost: int) -> float:
    """Exact two-sided binomial McNemar p-value for discordant pairs."""

    if gained < 0 or lost < 0:
        _fail("McNemar discordant counts must be non-negative")
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _outcome_counts(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    retained_success = sum(bool(left["success"]) and bool(right["success"]) for left, right in pairs)
    gained = sum(not bool(left["success"]) and bool(right["success"]) for left, right in pairs)
    lost = sum(bool(left["success"]) and not bool(right["success"]) for left, right in pairs)
    retained_failure = len(pairs) - retained_success - gained - lost
    union = retained_success + gained + lost
    return {
        "episodes": len(pairs),
        "retained_success": retained_success,
        "gained": gained,
        "lost": lost,
        "retained_failure": retained_failure,
        "net": gained - lost,
        "churn": gained + lost,
        "churn_rate": (gained + lost) / len(pairs) if pairs else None,
        "success_set_jaccard": retained_success / union if union else 1.0,
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided_p(gained, lost),
    }


def paired_transition_summary(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two panels; gained means left failure became right success."""

    left, right = _index_rows(left_rows), _index_rows(right_rows)
    if set(left) != set(right):
        _fail("paired panels do not have identical episode keys")
    keys = sorted(left, key=lambda key: (*_suite_sort_key(key[0]), key[1], key[2]))
    pairs = [(left[key], right[key]) for key in keys]
    task_keys = sorted({key[:2] for key in keys}, key=lambda key: (*_suite_sort_key(key[0]), key[1]))
    per_task = []
    for suite, task_id in task_keys:
        selected = [pair for key, pair in zip(keys, pairs) if key[:2] == (suite, task_id)]
        per_task.append({"suite": suite, "task_id": task_id, **_outcome_counts(selected)})
    per_suite = []
    for suite in sorted({key[0] for key in task_keys}, key=_suite_sort_key):
        selected = [pair for key, pair in zip(keys, pairs) if key[0] == suite]
        per_suite.append({"suite": suite, **_outcome_counts(selected)})
    return {"overall": _outcome_counts(pairs), "per_task": per_task, "per_suite": per_suite}


def _formal_adapter(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    adapter = result.get("adapter", {})
    paired = result.get("paired_control", {})
    condition = str(adapter.get("video_condition", ""))
    schedule = adapter.get("video_schedule", {})
    wall = adapter.get("information_wall", {})
    expected_wall = {
        "writer_input": "exact task language plus one action-hidden teacher video",
        "video_is_only_dynamic_value": True,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "language_only_lora_path": False,
        "deployment_expert_bank_read": False,
    }
    if (
        result.get("schema_version") != AGGREGATE_SCHEMA
        or result.get("mode") != "formal"
        or result.get("role") != "validation"
        or adapter.get("kind") != EXPERT_MANIFOLD_WRITER_KIND
        or condition not in SIX_ARM_CONDITIONS
        or result.get("arm") != f"expert_manifold_v6_prior_{condition}"
        or paired.get("mode") != "formal"
        or paired.get("role") != "validation"
        or paired.get("writer") != paired_writer_identity(adapter)
        or int(schedule.get("seed", -1)) != 7
        or int(schedule.get("demo_count", -1)) != 50
        or schedule.get("sampling_mode") != "without_replacement"
        or int(schedule.get("videos_per_condition", -1)) != 1
        or schedule.get("paired_between_all_video_conditions") is not True
        or schedule.get("queue_order_independent") is not True
        or any(wall.get(key) != value for key, value in expected_wall.items())
        or adapter.get("lora_contract", {}).get("rank") != 16
        or adapter.get("lora_contract", {}).get("target_count") != 38
    ):
        _fail("analysis requires a formal validation v2 Expert-Manifold panel")
    return adapter, paired, condition


def _formal_tasks(paired: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[TaskKey]]:
    tasks = paired.get("tasks", [])
    task_keys = [(str(task["suite"]), int(task["task_id"])) for task in tasks]
    suite_counts = {suite: sum(key[0] == suite for key in task_keys) for suite in SUITE_ORDER}
    if (
        len(tasks) != 8
        or len(set(task_keys)) != 8
        or suite_counts != {suite: 2 for suite in SUITE_ORDER}
        or any(
            list(map(int, task.get("init_state_ids", []))) != list(range(50))
            or task.get("split_role") != "validation"
            for task in tasks
        )
    ):
        _fail("formal validation panel must contain the sealed 8 tasks and states 0..49")
    return list(tasks), task_keys


def _formal_panel_index(result: Mapping[str, Any]) -> dict[EpisodeKey, Mapping[str, Any]]:
    adapter, paired, condition = _formal_adapter(result)
    tasks, task_keys = _formal_tasks(paired)
    roles = {key: str(task["split_role"]) for key, task in zip(task_keys, tasks)}
    expected_mapping = list(task_video_mapping(task_keys, roles, condition))
    if adapter.get("task_video_mapping") != expected_mapping:
        _fail("Writer task/video mapping is not the sealed condition mapping")
    rows = list(result.get("rows", []))
    indexed = _index_rows(rows)
    expected_keys = {(suite, task_id, state) for suite, task_id in task_keys for state in range(50)}
    if len(indexed) != 400 or set(indexed) != expected_keys:
        _fail("formal validation result must contain exactly 8x50 paired rows")
    mapping = {(row["suite"], int(row["task_id"])): row for row in expected_mapping}
    for row in indexed.values():
        _validate_episode_evidence(row, adapter, mapping[_task_key(row)])
    return indexed


def _validate_episode_evidence(
    row: Mapping[str, Any], adapter: Mapping[str, Any], mapping: Mapping[str, Any]
) -> None:
    writer = row.get("writer", {})
    condition = str(adapter["video_condition"])
    references = list(writer.get("teacher_reference_demo_indices", []))
    selected = list(writer.get("teacher_demo_indices", []))
    same = condition == "same_task_other"
    frames_used = condition != "no_video"
    asset = adapter["writer_asset"]
    expected = {
        "condition": condition,
        "teacher_video_kind": condition,
        "method_arm": adapter["arm"],
        "writer_asset_reference": asset["reference"],
        "writer_method_macro": int(asset["method_macro"]),
        "writer_checkpoint_kind": asset["kind"],
        "pairing_reference": adapter["pairing_reference"],
        "teacher_video_frames_used": frames_used,
        "teacher_video_count": int(frames_used),
        "video_suite": mapping["video_suite"],
        "video_task_id": int(mapping["video_task_id"]),
        "video_global_task_id": int(mapping["video_global_task_id"]),
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_demo_offset": SAME_TASK_OTHER_OFFSET if same else None,
    }
    valid = (
        len(references) == len(selected) == 1
        and 0 <= int(references[0]) < 50
        and 0 <= int(selected[0]) < 50
        and int(selected[0])
        == ((int(references[0]) + SAME_TASK_OTHER_OFFSET) % 50 if same else int(references[0]))
        and len(writer.get("teacher_video_order_seeds", [])) == 1
        and all(writer.get(key) == value for key, value in expected.items())
        and adapter.get("information_wall", {}).get("no_video_counterfactual") is (not frames_used)
    )
    if not valid:
        _fail("Writer episode evidence violates its paired video condition")


def _scientific_projection(
    result: Mapping[str, Any], *, allow_checkpoint_change: bool
) -> dict[str, Any]:
    projection = copy.deepcopy(result["paired_control"])
    projection["parallel"].pop("physical_gpu_ids", None)
    if allow_checkpoint_change:
        asset = projection["writer"]["writer_asset"]
        for key in ("reference", "kind", "method_macro", "checkpoint", "manifest", "training_mode"):
            asset.pop(key, None)
        state = asset.get("writer_state")
        if isinstance(state, dict):
            state.pop("path", None)
    return projection


def _common_noise_prefix(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_seeds = list(left.get("policy_noise_seeds", []))
    right_seeds = list(right.get("policy_noise_seeds", []))
    common = min(len(left_seeds), len(right_seeds))
    return left_seeds[:common] == right_seeds[:common]


def _assert_row_pairing(
    reference: Mapping[EpisodeKey, Mapping[str, Any]],
    candidate: Mapping[EpisodeKey, Mapping[str, Any]],
    *,
    require_same_actual_video: bool,
) -> None:
    if set(reference) != set(candidate):
        _fail("paired roots do not contain the same episode identities")
    row_fields = ("language", "split_role", "env_seed", "policy_seed_root")
    writer_fields = (
        "language_global_task_id",
        "lora_contract_reference",
        "pairing_reference",
        "teacher_reference_demo_indices",
        "teacher_video_seed_root",
        "teacher_video_selection_seed",
        "teacher_video_sampling_mode",
        "teacher_video_order_seeds",
        "writer_generation_seed_schedule",
    )
    if require_same_actual_video:
        writer_fields += (
            "teacher_demo_indices",
            "video_suite",
            "video_task_id",
            "video_global_task_id",
            "video_split_role",
        )
    for key in reference:
        left, right = reference[key], candidate[key]
        left_writer, right_writer = left["writer"], right["writer"]
        if (
            any(left.get(field) != right.get(field) for field in row_fields)
            or any(left_writer.get(field) != right_writer.get(field) for field in writer_fields)
            or not _common_noise_prefix(left, right)
        ):
            _fail("paired roots changed state, RNG, language, or teacher-video identity")


def _method_macro(result: Mapping[str, Any]) -> int:
    asset = result["adapter"]["writer_asset"]
    macro = int(asset.get("method_macro", -1))
    expected_kind = "historical_v6_macro400_load_only" if macro == 0 else "v6_prior_trained_checkpoint"
    if macro not in CHECKPOINT_MACROS or asset.get("kind") != expected_kind:
        _fail("checkpoint curve contains an unexpected method macro or checkpoint kind")
    return macro


def _prefix_rows(indexed: Mapping[EpisodeKey, Mapping[str, Any]], stop: int) -> list[Mapping[str, Any]]:
    selected = [row for key, row in indexed.items() if key[2] < stop]
    states = {(key[0], key[1]): [] for key in indexed}
    for row in selected:
        states[_task_key(row)].append(int(row["init_state_id"]))
    if len(selected) != len(states) * stop or any(sorted(values) != list(range(stop)) for values in states.values()):
        _fail("same-root state-prefix panel is incomplete")
    return selected


def _curve_set_evidence(rows_by_macro: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    indexes = {macro: _index_rows(rows) for macro, rows in rows_by_macro.items()}
    success_sets = {macro: {key for key, row in rows.items() if row["success"]} for macro, rows in indexes.items()}
    union = set().union(*success_sets.values())
    intersection = set.intersection(*success_sets.values())
    task_keys = sorted({key[:2] for key in next(iter(indexes.values()))}, key=lambda key: (*_suite_sort_key(key[0]), key[1]))
    task_rows = []
    for suite, task_id in task_keys:
        counts = {str(macro): sum(key[:2] == (suite, task_id) for key in values) for macro, values in success_sets.items()}
        task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "successes_by_macro": counts,
                "union_successes": sum(key[:2] == (suite, task_id) for key in union),
                "intersection_successes": sum(key[:2] == (suite, task_id) for key in intersection),
                "envelope_successes": max(counts.values()),
            }
        )
    totals = {str(macro): len(values) for macro, values in success_sets.items()}
    envelope = sum(int(row["envelope_successes"]) for row in task_rows)
    best = max(totals.values())
    return {
        "successes_by_macro": totals,
        "union_successes": len(union),
        "intersection_successes": len(intersection),
        "best_single_checkpoint_successes": best,
        "per_task_envelope_successes": envelope,
        "envelope_gap_over_best_single": envelope - best,
        "per_task": task_rows,
        "selection_warning": "union, intersection, and envelope are diagnostics only; they do not define checkpoint fusion",
    }


def checkpoint_curve_analysis(results_by_root: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build strict macro0/10/25/50 analysis from validated result payloads."""

    by_macro: dict[int, tuple[str, Mapping[str, Any], dict[EpisodeKey, Mapping[str, Any]]]] = {}
    for root, result in results_by_root.items():
        indexed = _formal_panel_index(result)
        if result["adapter"]["video_condition"] != "correct":
            _fail("checkpoint curve accepts only correct-video roots")
        macro = _method_macro(result)
        if macro in by_macro:
            _fail("checkpoint curve contains duplicate method macros")
        by_macro[macro] = (root, result, indexed)
    if tuple(sorted(by_macro)) != CHECKPOINT_MACROS:
        _fail("checkpoint curve requires exactly method macros 0, 10, 25, and 50")
    reference_projection = _scientific_projection(by_macro[0][1], allow_checkpoint_change=True)
    reference_rows = by_macro[0][2]
    for macro in CHECKPOINT_MACROS[1:]:
        if _scientific_projection(by_macro[macro][1], allow_checkpoint_change=True) != reference_projection:
            _fail("checkpoint curve changed its scientific or execution contract")
        _assert_row_pairing(reference_rows, by_macro[macro][2], require_same_actual_video=True)
    panels: dict[str, dict[int, list[Mapping[str, Any]]]] = {
        "correct80": {macro: _prefix_rows(by_macro[macro][2], 10) for macro in CHECKPOINT_MACROS},
        "correct400": {macro: list(by_macro[macro][2].values()) for macro in CHECKPOINT_MACROS},
    }
    comparisons = ((0, 10), (10, 25), (25, 50), (0, 25), (0, 50))
    return {
        "schema_version": CHECKPOINT_CURVE_SCHEMA,
        "contract_audit": {
            "formal_validation_8x50": True,
            "same_scientific_contract_except_checkpoint_identity": True,
            "same_state_rng_language_and_correct_video_identity": True,
            "physical_gpu_indices_excluded_from_projection": True,
        },
        "row_selection": {
            "correct80": "same validated correct400 root rows with init_state_id < 10",
            "correct400": "all formal validation rows",
        },
        "roots": [
            {
                "root": by_macro[macro][0],
                "contract_reference": by_macro[macro][1]["contract_reference"],
                "method_macro": macro,
                "writer_asset_reference": by_macro[macro][1]["adapter"]["writer_asset"]["reference"],
            }
            for macro in CHECKPOINT_MACROS
        ],
        "panels": {
            name: {str(macro): summarize_panel(rows[macro]) for macro in CHECKPOINT_MACROS}
            for name, rows in panels.items()
        },
        "comparisons": {
            name: {
                f"{left}_to_{right}": paired_transition_summary(rows[left], rows[right])
                for left, right in comparisons
            }
            for name, rows in panels.items()
        },
        "curve_evidence": {name: _curve_set_evidence(rows) for name, rows in panels.items()},
        "metric_definitions": {
            "gained": "left failure and right success on the identical episode key",
            "lost": "left success and right failure on the identical episode key",
            "churn": "gained plus lost",
            "nonzero_task_breadth": "tasks with at least one success in this exact panel",
            "top3_success_share": "successes from deterministic top-3 tasks divided by all successes",
        },
    }


def _control_outcome_summary(
    correct: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    transition = paired_transition_summary(control, correct)

    def rename(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **{key: value[key] for key in ("episodes", "churn", "churn_rate", "success_set_jaccard", "mcnemar_exact_two_sided_p")},
            "both_success": value["retained_success"],
            "correct_only": value["gained"],
            "control_only": value["lost"],
            "both_failure": value["retained_failure"],
            "correct_minus_control": value["net"],
        }

    per_task = [{"suite": row["suite"], "task_id": row["task_id"], **rename(row)} for row in transition["per_task"]]
    positive = [int(row["correct_minus_control"]) for row in per_task if int(row["correct_minus_control"]) > 0]
    return {
        "overall": rename(transition["overall"]),
        "per_task": per_task,
        "per_suite": [{"suite": row["suite"], **rename(row)} for row in transition["per_suite"]],
        "positive_contributing_task_count": len(positive),
        "largest_positive_task_contribution_share": max(positive) / sum(positive) if positive else None,
    }


def six_arm_paired_analysis(results_by_root: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build a strict single-checkpoint six-arm video-causality audit."""

    by_condition: dict[str, tuple[str, Mapping[str, Any], dict[EpisodeKey, Mapping[str, Any]]]] = {}
    for root, result in results_by_root.items():
        indexed = _formal_panel_index(result)
        condition = str(result["adapter"]["video_condition"])
        if condition in by_condition:
            _fail("six-arm audit contains a duplicate video condition")
        by_condition[condition] = (root, result, indexed)
    if set(by_condition) != set(SIX_ARM_CONDITIONS):
        _fail("six-arm audit requires exactly the canonical six video conditions")
    correct_result = by_condition["correct"][1]
    projection = _scientific_projection(correct_result, allow_checkpoint_change=False)
    correct_rows = by_condition["correct"][2]
    for condition in SIX_ARM_CONDITIONS[1:]:
        result, rows = by_condition[condition][1:]
        if _scientific_projection(result, allow_checkpoint_change=False) != projection:
            _fail("six-arm audit changed its single checkpoint or scientific contract")
        _assert_row_pairing(correct_rows, rows, require_same_actual_video=False)
    asset = correct_result["adapter"]["writer_asset"]
    arm_rows = {condition: list(by_condition[condition][2].values()) for condition in SIX_ARM_CONDITIONS}
    return {
        "schema_version": SIX_ARM_AUDIT_SCHEMA,
        "winner": {
            "method_macro": int(asset["method_macro"]),
            "writer_asset_reference": asset["reference"],
            "checkpoint_kind": asset["kind"],
        },
        "contract_audit": {
            "formal_validation_8x50_each_arm": True,
            "same_single_checkpoint_and_scientific_contract": True,
            "same_state_rng_language_and_reference_video_identity": True,
            "condition_specific_video_mapping_and_order_validated": True,
            "physical_gpu_indices_excluded_from_projection": True,
        },
        "roots": [
            {
                "condition": condition,
                "root": by_condition[condition][0],
                "contract_reference": by_condition[condition][1]["contract_reference"],
            }
            for condition in SIX_ARM_CONDITIONS
        ],
        "arms": {condition: summarize_panel(arm_rows[condition]) for condition in SIX_ARM_CONDITIONS},
        "comparisons_to_correct": {
            condition: {
                "interpretation": "same-task cross-video robustness" if condition == "same_task_other" else "video counterfactual control",
                **_control_outcome_summary(arm_rows["correct"], arm_rows[condition]),
            }
            for condition in SIX_ARM_CONDITIONS
            if condition != "correct"
        },
        "metric_definitions": {
            "correct_only": "correct succeeds and the paired control fails",
            "control_only": "control succeeds and the paired correct arm fails",
            "correct_minus_control": "correct_only minus control_only",
            "same_task_other": "robustness arm, not a negative video arm",
        },
    }


def _validated_roots(roots: Sequence[Path]) -> dict[str, Mapping[str, Any]]:
    normalized = [root.resolve() for root in roots]
    if len(set(normalized)) != len(normalized):
        _fail("analysis roots must be unique")
    results: dict[str, Mapping[str, Any]] = {}
    for root in normalized:
        if not (root / "results.json").is_file():
            _fail(f"analysis root has no immutable results.json: {root}")
        from ember.pi05_eval_results import aggregate_run

        results[str(root)] = aggregate_run(root)
    return results


def analyze_checkpoint_curve(
    roots: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    result = checkpoint_curve_analysis(_validated_roots(roots))
    publish_json_exclusive(output_path.resolve(), result)
    return result


def audit_six_arms(roots: Sequence[Path], output_path: Path) -> dict[str, Any]:
    result = six_arm_paired_analysis(_validated_roots(roots))
    publish_json_exclusive(output_path.resolve(), result)
    return result
