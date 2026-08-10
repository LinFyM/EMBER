"""Strict CPU-only analysis of paired PI05 Writer evaluation results."""

from __future__ import annotations

import copy
import json
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
from ember.pi05_eval.paired_metrics import (
    EpisodeKey,
    TaskKey,
    control_outcome_summary as _control_outcome_summary,
    episode_key as _episode_key,
    exact_mcnemar_two_sided_p,
    index_rows as _index_rows,
    paired_transition_summary,
    suite_sort_key as _suite_sort_key,
    summarize_panel,
    task_key as _task_key,
)
from ember.pi05_eval.writer_family_registry import (
    CHECKPOINT_MACROS,
    HISTORICAL_TRANSITION_BASELINE_FAMILIES,
    HISTORICAL_TRANSITION_CANDIDATE_MACROS,
    PROGRAM_RESIDUAL_WRITER_FAMILIES,
    WRITER_FAMILIES,
)
from ember.pi05_target_data import SUITE_ORDER


CHECKPOINT_CURVE_SCHEMA = "ember_pi05_v6_writer_checkpoint_curve_analysis_v2"
SIX_ARM_AUDIT_SCHEMA = "ember_pi05_v6_writer_six_arm_paired_analysis_v2"
HISTORICAL_BASELINE_TRANSITION_SCHEMA = "ember_pi05_v6_historical_baseline_transition_analysis_v2"
SIX_ARM_CONDITIONS = (
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "shuffled",
    "reversed",
    "no_video",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_DATA_MANIFEST = REPO_ROOT / "configs/pi05_target_data_v1/manifest.json"


def _fail(message: str) -> None:
    raise Pi05EvaluationError(message)


def _writer_family(adapter: Mapping[str, Any]) -> tuple[str, Mapping[str, str]]:
    for name, family in WRITER_FAMILIES.items():
        if (
            adapter.get("schema_version") == family["adapter_schema"]
            and adapter.get("config", {}).get("schema") == family["config_schema"]
        ):
            return name, family
    _fail("Writer result is not a sealed supported method family")


def _formal_adapter(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    adapter = result.get("adapter", {})
    paired = result.get("paired_control", {})
    condition = str(adapter.get("video_condition", ""))
    _, family = _writer_family(adapter)
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
        or result.get("arm") != f"{family['arm_prefix']}{condition}"
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
        or adapter.get("evaluation_authority", {}).get("formal_status") not in family["formal_statuses"]
        or paired.get("git", {}).get("dirty_paths") != []
    ):
        _fail("analysis requires a sealed formal validation Expert-Manifold panel")
    return adapter, paired, condition


def _sealed_validation_tasks() -> list[dict[str, Any]]:
    try:
        manifest = json.loads(TARGET_DATA_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read the sealed target-data manifest: {exc}")
    tasks = [
        {
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "global_task_id": int(task["global_task_id"]),
            "split_role": str(task["split_role"]),
            "language": str(task["language"]),
        }
        for task in manifest.get("tasks", [])
        if task.get("split_role") == "validation"
    ]
    summary_ids = list(map(int, manifest.get("summary", {}).get("roles", {}).get("validation", [])))
    if (
        manifest.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or len(tasks) != 8
        or len({(task["suite"], task["task_id"]) for task in tasks}) != 8
        or [task["global_task_id"] for task in tasks] != summary_ids
    ):
        _fail("sealed target-data manifest does not define the canonical 8 validation tasks")
    return tasks


def _formal_tasks(paired: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[TaskKey]]:
    tasks = paired.get("tasks", [])
    authority = _sealed_validation_tasks()
    expected = [{key: task[key] for key in ("suite", "task_id", "split_role", "language")} for task in authority]
    observed = [
        {
            "suite": str(task.get("suite", "")),
            "task_id": int(task.get("task_id", -1)),
            "split_role": task.get("split_role"),
            "language": task.get("language"),
        }
        for task in tasks
    ]
    if observed != expected or any(list(map(int, task.get("init_state_ids", []))) != list(range(50)) for task in tasks):
        _fail("formal validation panel must exactly match the sealed 8 tasks, languages, and states 0..49")
    task_keys = [(str(task["suite"]), int(task["task_id"])) for task in tasks]
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


def _validate_episode_evidence(row: Mapping[str, Any], adapter: Mapping[str, Any], mapping: Mapping[str, Any]) -> None:
    writer = row.get("writer", {})
    condition = str(adapter["video_condition"])
    family_name, family = _writer_family(adapter)
    references = list(writer.get("teacher_reference_demo_indices", []))
    selected = list(writer.get("teacher_demo_indices", []))
    same = condition == "same_task_other"
    frames_used = condition != "no_video"
    asset = adapter["writer_asset"]
    expected = {
        "schema_version": family["episode_schema"],
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
    if family_name in PROGRAM_RESIDUAL_WRITER_FAMILIES:
        expected.update(
            {
                "writer_parameter_count": int(asset["writer_parameter_count"]),
                "writer_deployment_trainable_parameter_count": 0,
                "writer_program_residual_value_count": int(asset["program_residual_value_count"]),
                "generated_lora_tensor_count": int(asset["generated_lora_tensor_count"]),
            }
        )
    valid = (
        len(references) == len(selected) == 1
        and 0 <= int(references[0]) < 50
        and 0 <= int(selected[0]) < 50
        and int(selected[0]) == ((int(references[0]) + SAME_TASK_OTHER_OFFSET) % 50 if same else int(references[0]))
        and len(writer.get("teacher_video_order_seeds", [])) == 1
        and all(writer.get(key) == value for key, value in expected.items())
        and adapter.get("information_wall", {}).get("no_video_counterfactual") is (not frames_used)
    )
    if not valid:
        _fail("Writer episode evidence violates its paired video condition")


def _scientific_projection(result: Mapping[str, Any], *, allow_checkpoint_change: bool) -> dict[str, Any]:
    projection = copy.deepcopy(result["paired_control"])
    projection.pop("parallel", None)
    if allow_checkpoint_change:
        asset = projection["writer"]["writer_asset"]
        for key in (
            "reference",
            "kind",
            "method_macro",
            "checkpoint",
            "manifest",
            "training_mode",
        ):
            asset.pop(key, None)
        state = asset.get("writer_state")
        if isinstance(state, dict):
            state.pop("path", None)
        residual = asset.get("residual_state")
        if isinstance(residual, dict):
            for key in ("kind", "path", "bytes", "tensor_count", "key"):
                residual.pop(key, None)
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


def _method_macro(
    result: Mapping[str, Any],
    *,
    allowed_macros: Sequence[int] = CHECKPOINT_MACROS,
    context: str = "checkpoint curve",
) -> int:
    adapter = result["adapter"]
    _, family = _writer_family(adapter)
    asset = adapter["writer_asset"]
    macro = int(asset.get("method_macro", -1))
    expected_kind = "historical_v6_macro400_load_only" if macro == 0 else family["trained_checkpoint_kind"]
    if macro not in allowed_macros or asset.get("kind") != expected_kind:
        _fail(f"{context} contains an unexpected method macro or checkpoint kind")
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
    task_keys = sorted(
        {key[:2] for key in next(iter(indexes.values()))},
        key=lambda key: (*_suite_sort_key(key[0]), key[1]),
    )
    task_rows = []
    for suite, task_id in task_keys:
        counts = {
            str(macro): sum(key[:2] == (suite, task_id) for key in values) for macro, values in success_sets.items()
        }
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
    families = {_writer_family(result["adapter"])[0] for _, result, _ in by_macro.values()}
    if len(families) != 1:
        _fail("checkpoint curve cannot mix legacy and current method families")
    method_family = next(iter(families))
    reference_projection = _scientific_projection(by_macro[0][1], allow_checkpoint_change=True)
    reference_rows = by_macro[0][2]
    for macro in CHECKPOINT_MACROS[1:]:
        if _scientific_projection(by_macro[macro][1], allow_checkpoint_change=True) != reference_projection:
            _fail("checkpoint curve changed its scientific contract")
        _assert_row_pairing(reference_rows, by_macro[macro][2], require_same_actual_video=True)
    panels: dict[str, dict[int, list[Mapping[str, Any]]]] = {
        "correct80": {macro: _prefix_rows(by_macro[macro][2], 10) for macro in CHECKPOINT_MACROS},
        "correct400": {macro: list(by_macro[macro][2].values()) for macro in CHECKPOINT_MACROS},
    }
    comparisons = ((0, 10), (10, 25), (25, 50), (0, 25), (0, 50))
    return {
        "schema_version": CHECKPOINT_CURVE_SCHEMA,
        "method_family": method_family,
        "contract_audit": {
            "formal_validation_8x50": True,
            "same_scientific_contract_except_checkpoint_identity": True,
            "same_state_rng_language_and_correct_video_identity": True,
            "execution_parallel_topology_excluded_from_scientific_projection": True,
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
                "parallel_provenance": copy.deepcopy(by_macro[macro][1]["paired_control"].get("parallel", {})),
            }
            for macro in CHECKPOINT_MACROS
        ],
        "panels": {
            name: {str(macro): summarize_panel(rows[macro]) for macro in CHECKPOINT_MACROS}
            for name, rows in panels.items()
        },
        "comparisons": {
            name: {
                f"{left}_to_{right}": paired_transition_summary(rows[left], rows[right]) for left, right in comparisons
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


def _historical_transition_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize provenance and explicit family identity for a cross-family audit."""

    projection = _scientific_projection(result, allow_checkpoint_change=True)
    projection.pop("git", None)
    writer = projection.get("writer")
    tokenizer = projection.get("tokenizer")
    normalization = projection.get("normalization")
    if not all(isinstance(value, dict) for value in (writer, tokenizer, normalization)):
        _fail("historical transition is missing its shared scientific contract")
    for key in (
        "execution_backend",
        "config",
        "writer_asset",
        "evaluation_authority",
    ):
        writer.pop(key, None)
    tokenizer.pop("manifest_path", None)
    normalization.pop("path", None)
    return projection


def historical_baseline_transition_analysis(
    results_by_root: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one sealed candidate with its explicit immutable macro0 baseline."""

    if len(results_by_root) != 2:
        _fail("historical baseline transition requires exactly two roots")
    by_family: dict[
        str,
        tuple[str, Mapping[str, Any], dict[EpisodeKey, Mapping[str, Any]]],
    ] = {}
    for root, result in results_by_root.items():
        indexed = _formal_panel_index(result)
        if result["adapter"]["video_condition"] != "correct":
            _fail("historical baseline transition accepts only correct-video roots")
        family = _writer_family(result["adapter"])[0]
        if family in by_family:
            _fail("historical baseline transition contains a duplicate method family")
        by_family[family] = (root, result, indexed)
    supported_pairs = [
        (baseline_family, candidate_family)
        for candidate_family, baseline_family in (HISTORICAL_TRANSITION_BASELINE_FAMILIES.items())
        if set(by_family) == {baseline_family, candidate_family}
    ]
    if len(supported_pairs) != 1:
        _fail(
            "historical baseline transition requires exactly one supported "
            "baseline and current candidate family pair"
        )

    baseline_family, candidate_family = supported_pairs[0]
    baseline = by_family[baseline_family]
    candidate = by_family[candidate_family]
    _method_macro(
        baseline[1],
        allowed_macros=(0,),
        context="historical baseline transition baseline",
    )
    candidate_macros = HISTORICAL_TRANSITION_CANDIDATE_MACROS[candidate_family]
    _method_macro(
        candidate[1],
        allowed_macros=candidate_macros,
        context="historical baseline transition candidate",
    )
    if _historical_transition_projection(baseline[1]) != _historical_transition_projection(candidate[1]):
        _fail("historical baseline transition changed its shared scientific contract")
    _assert_row_pairing(baseline[2], candidate[2], require_same_actual_video=True)

    baseline_rows = list(baseline[2].values())
    candidate_rows = list(candidate[2].values())
    panels = {
        "correct80": {
            "historical_baseline": _prefix_rows(baseline[2], 10),
            "current_candidate": _prefix_rows(candidate[2], 10),
        },
        "correct400": {
            "historical_baseline": baseline_rows,
            "current_candidate": candidate_rows,
        },
    }

    def root_evidence(
        value: tuple[
            str,
            Mapping[str, Any],
            dict[EpisodeKey, Mapping[str, Any]],
        ]
    ) -> dict[str, Any]:
        root, result, _ = value
        adapter = result["adapter"]
        family, family_contract = _writer_family(adapter)
        asset = adapter["writer_asset"]
        return {
            "root": root,
            "method_family": family,
            "method_macro": int(asset["method_macro"]),
            "checkpoint_kind": asset["kind"],
            "adapter_schema": family_contract["adapter_schema"],
            "config_schema": family_contract["config_schema"],
            "contract_reference": result["contract_reference"],
            "git": copy.deepcopy(result["paired_control"]["git"]),
            "parallel_provenance": copy.deepcopy(result["paired_control"].get("parallel", {})),
        }

    return {
        "schema_version": HISTORICAL_BASELINE_TRANSITION_SCHEMA,
        "analysis_scope": ("cross_family_historical_baseline_transition_not_checkpoint_curve"),
        "method_families": {
            "historical_baseline": baseline_family,
            "current_candidate": candidate_family,
        },
        "contract_audit": {
            "native_family_validation_each_root": True,
            "formal_validation_8x50_each": True,
            "same_shared_scientific_contract": True,
            "same_state_rng_language_and_correct_video_identity": True,
            "family_labels_preserved": True,
            "checkpoint_curve_membership_claimed": False,
        },
        "row_selection": {
            "correct80": "same validated correct400 root rows with init_state_id < 10",
            "correct400": "all formal validation rows",
        },
        "roots": {
            "historical_baseline": root_evidence(baseline),
            "current_candidate": root_evidence(candidate),
        },
        "panels": {
            panel: {role: summarize_panel(rows) for role, rows in role_rows.items()}
            for panel, role_rows in panels.items()
        },
        "baseline_to_candidate": {
            panel: paired_transition_summary(
                role_rows["historical_baseline"],
                role_rows["current_candidate"],
            )
            for panel, role_rows in panels.items()
        },
        "metric_definitions": {
            "gained": "historical failure and current success on the identical episode key",
            "lost": "historical success and current failure on the identical episode key",
            "churn": "gained plus lost",
            "nonzero_task_breadth": "tasks with at least one success in this exact panel",
            "cross_family_warning": (
                "native family labels are retained; this artifact is not a within-family checkpoint curve"
            ),
        },
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
    families = {_writer_family(result["adapter"])[0] for _, result, _ in by_condition.values()}
    if len(families) != 1:
        _fail("six-arm audit cannot mix legacy and current method families")
    method_family = next(iter(families))
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
        "method_family": method_family,
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
            "execution_parallel_topology_excluded_from_scientific_projection": True,
        },
        "roots": [
            {
                "condition": condition,
                "root": by_condition[condition][0],
                "contract_reference": by_condition[condition][1]["contract_reference"],
                "parallel_provenance": copy.deepcopy(by_condition[condition][1]["paired_control"].get("parallel", {})),
            }
            for condition in SIX_ARM_CONDITIONS
        ],
        "arms": {condition: summarize_panel(arm_rows[condition]) for condition in SIX_ARM_CONDITIONS},
        "comparisons_to_correct": {
            condition: {
                "interpretation": (
                    "same-task cross-video robustness"
                    if condition == "same_task_other"
                    else "video counterfactual control"
                ),
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
        try:
            stored = json.loads((root / "results.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"analysis root has an invalid results.json: {root}: {exc}")
        legacy = False
        if isinstance(stored, Mapping):
            try:
                legacy = _writer_family(stored.get("adapter", {}))[0].startswith("legacy_")
            except Pi05EvaluationError:
                legacy = False
        if legacy:
            _formal_panel_index(stored)
            results[str(root)] = stored
        else:
            from ember.pi05_eval_results import aggregate_run

            results[str(root)] = aggregate_run(root)
    return results


def analyze_checkpoint_curve(roots: Sequence[Path], output_path: Path) -> dict[str, Any]:
    result = checkpoint_curve_analysis(_validated_roots(roots))
    publish_json_exclusive(output_path.resolve(), result)
    return result


def analyze_historical_baseline_transition(
    legacy_root: Path,
    current_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    result = historical_baseline_transition_analysis(_validated_roots((legacy_root, current_root)))
    publish_json_exclusive(output_path.resolve(), result)
    return result


def audit_six_arms(roots: Sequence[Path], output_path: Path) -> dict[str, Any]:
    validated = _validated_roots(roots)
    result = six_arm_paired_analysis(validated)
    if result["method_family"] == "v6_reward_credit_program_v1":
        from ember.pi05_eval.reward_credit_gate import (
            reward_credit_six_arm_evidence_from_config,
        )

        correct = next(value for value in validated.values() if value["adapter"]["video_condition"] == "correct")
        config = correct["adapter"]["config"]
        result["reward_credit_goal"] = reward_credit_six_arm_evidence_from_config(
            result,
            config_path=Path(str(config["path"])),
            expected_bytes=int(config["bytes"]),
        )
    publish_json_exclusive(output_path.resolve(), result)
    return result
