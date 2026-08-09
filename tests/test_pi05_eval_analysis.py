from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import pytest

from ember.eval_adapters import paired_writer_identity
from ember.expert_manifold.video_schedule import SAME_TASK_OTHER_OFFSET, task_video_mapping
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.analysis import (
    CHECKPOINT_CURVE_SCHEMA,
    HISTORICAL_BASELINE_TRANSITION_SCHEMA,
    SIX_ARM_AUDIT_SCHEMA,
    SIX_ARM_CONDITIONS,
    _formal_tasks,
    analyze_checkpoint_curve,
    analyze_historical_baseline_transition,
    checkpoint_curve_analysis,
    exact_mcnemar_two_sided_p,
    historical_baseline_transition_analysis,
    paired_transition_summary,
    six_arm_paired_analysis,
    summarize_panel,
)
from ember.pi05_eval_results import AGGREGATE_SCHEMA


TASKS = (
    ("libero_spatial", 1),
    ("libero_spatial", 3),
    ("libero_object", 1),
    ("libero_object", 3),
    ("libero_goal", 3),
    ("libero_goal", 6),
    ("libero_10", 1),
    ("libero_10", 2),
)

TASK_LANGUAGES = {
    ("libero_spatial", 1): "pick up the black bowl next to the ramekin and place it on the plate",
    ("libero_spatial", 3): "pick up the black bowl on the cookie box and place it on the plate",
    ("libero_object", 1): "pick up the cream cheese and place it in the basket",
    ("libero_object", 3): "pick up the bbq sauce and place it in the basket",
    ("libero_goal", 3): "open the top drawer and put the bowl inside",
    ("libero_goal", 6): "put the cream cheese in the bowl",
    ("libero_10", 1): "put both the cream cheese box and the butter in the basket",
    ("libero_10", 2): "turn on the stove and put the moka pot on it",
}


def _success_keys(predicate: Callable[[str, int, int], bool]) -> set[tuple[str, int, int]]:
    return {
        (suite, task_id, state)
        for suite, task_id in TASKS
        for state in range(50)
        if predicate(suite, task_id, state)
    }


def _tasks() -> list[dict]:
    return [
        {
            "suite": suite,
            "task_id": task_id,
            "split_role": "validation",
            "language": TASK_LANGUAGES[(suite, task_id)],
            "init_state_ids": list(range(50)),
        }
        for suite, task_id in TASKS
    ]


def _adapter(macro: int, condition: str, *, family: str = "current") -> dict:
    current = family == "current"
    assert current or family == "legacy"
    roles = {key: "validation" for key in TASKS}
    mapping = list(task_video_mapping(TASKS, roles, condition))
    checkpoint_kind = (
        "historical_v6_macro400_load_only"
        if macro == 0
        else ("v6_ecp_trained_checkpoint" if current else "v6_prior_trained_checkpoint")
    )
    writer_asset = {
        "reference": f"writer:m{macro}",
        "kind": checkpoint_kind,
        "training_mode": "historical_v6_task_complete" if macro == 0 else "formal",
        "source_macro": 400,
        "method_macro": macro,
        "checkpoint": f"/writer/macro_{macro}",
        "manifest": {"path": f"/writer/macro_{macro}/manifest.json"},
        "architecture": "v6-prior",
        "writer_parameter_count": 10_775_296,
        "deployment_trainable_parameter_count": 0,
        "generated_lora_tensor_count": 76,
        "writer_state": {
            "path": f"/writer/macro_{macro}/writer.safetensors",
            "bytes": 45_818_648,
            "state_tensor_count": 600,
            "template_lora_storage": {"tensor_count": 76, "rank": 16},
        },
    }
    return {
        "schema_version": (
            "ember_pi05_v6_ecp_eval_adapter_v6"
            if current
            else "ember_pi05_v6_prior_eval_adapter_v5"
        ),
        "kind": "expert_manifold_writer",
        "arm": (
            f"expert_manifold_v6_ecp_{condition}"
            if current
            else f"expert_manifold_v6_prior_{condition}"
        ),
        "execution_backend": "online_writer_then_episode_cache",
        "config": {
            "schema": (
                "ember_pi05_v6_ecp_policy_effective_writer_v2"
                if current
                else "ember_pi05_v6_prior_policy_effective_writer_v1"
            )
        },
        "writer_asset": writer_asset,
        "evaluation_authority": {"formal_status": "sealed"},
        "video_data": {"root": "/videos", "tasks": "sealed-validation-8"},
        "lora_contract": {"reference": "lora-v1", "rank": 16, "target_count": 38},
        "video_schedule": {
            "seed": 7,
            "demo_count": 50,
            "sampling_mode": "without_replacement",
            "videos_per_condition": 1,
            "paired_between_all_video_conditions": True,
            "queue_order_independent": True,
        },
        "pairing_reference": "one-shot-pairing-v1",
        "video_condition": condition,
        "task_video_mapping": mapping,
        "information_wall": {
            "writer_input": "exact task language plus one action-hidden teacher video",
            "video_is_only_dynamic_value": True,
            "no_video_counterfactual": condition == "no_video",
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "language_only_lora_path": False,
            "deployment_expert_bank_read": False,
        },
    }


def _rows(
    macro: int,
    condition: str,
    successes: set[tuple[str, int, int]],
    adapter: dict,
) -> list[dict]:
    writer_asset = adapter["writer_asset"]
    checkpoint_kind = writer_asset["kind"]
    mapping = adapter["task_video_mapping"]
    mapping_by_task = {
        (row["suite"], int(row["task_id"])): row for row in mapping
    }
    rows = []
    for suite, task_id in TASKS:
        task_mapping = mapping_by_task[(suite, task_id)]
        for state in range(50):
            reference = (state + task_id) % 50
            selected = (
                (reference + SAME_TASK_OTHER_OFFSET) % 50
                if condition == "same_task_other"
                else reference
            )
            writer = {
                "schema_version": (
                    "ember_pi05_v6_ecp_episode_v6"
                    if adapter["config"]["schema"].endswith("writer_v2")
                    else "ember_pi05_v6_prior_episode_v5"
                ),
                "condition": condition,
                "teacher_video_kind": condition,
                "method_arm": adapter["arm"],
                "writer_asset_reference": writer_asset["reference"],
                "writer_method_macro": macro,
                "writer_checkpoint_kind": checkpoint_kind,
                "lora_contract_reference": "lora-v1",
                "pairing_reference": "one-shot-pairing-v1",
                "language_global_task_id": task_mapping["language_global_task_id"],
                "teacher_reference_demo_indices": [reference],
                "teacher_demo_indices": [selected],
                "teacher_video_seed_root": 7,
                "teacher_video_selection_seed": state * 100 + task_id,
                "teacher_video_sampling_mode": "without_replacement",
                "teacher_video_order_seeds": [state * 1000 + task_id],
                "writer_generation_seed_schedule": "one-shot-seed-v1",
                "teacher_video_frames_used": condition != "no_video",
                "teacher_video_count": int(condition != "no_video"),
                "video_suite": task_mapping["video_suite"],
                "video_task_id": task_mapping["video_task_id"],
                "video_global_task_id": task_mapping["video_global_task_id"],
                "video_split_role": task_mapping["video_split_role"],
            }
            if condition == "same_task_other":
                writer["teacher_demo_offset"] = SAME_TASK_OTHER_OFFSET
            key = (suite, task_id, state)
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "init_state_id": state,
                    "split_role": "validation",
                    "language": TASK_LANGUAGES[(suite, task_id)],
                    "env_seed": 7,
                    "policy_seed_root": 7,
                    "policy_noise_seeds": [state + 1, state + 1001],
                    "success": key in successes,
                    "writer": writer,
                }
            )
    return rows


def _result(
    macro: int,
    condition: str,
    successes: set[tuple[str, int, int]],
    *,
    physical_gpu_ids: tuple[int, ...] = (0, 1),
    family: str = "current",
) -> dict:
    tasks = _tasks()
    adapter = _adapter(macro, condition, family=family)
    parallel = {
        "physical_gpu_ids": list(physical_gpu_ids),
        "physical_gpu_count": len(physical_gpu_ids),
        "worker_count": len(physical_gpu_ids) * 2,
        "replicas_per_gpu": 2,
        "envs_per_replica": 8,
        "writer_generation_batch_size": 16,
    }
    paired = {
        "schema_version": "ember_pi05_writer_paired_control_v2",
        "mode": "formal",
        "role": "validation",
        "git": {"commit": "formal-commit", "dirty_paths": []},
        "model": {"checkpoint": "/source"},
        "tokenizer": {"path": "/tokenizer"},
        "normalization": {"path": "/normalization"},
        "tasks": tasks,
        "environment": {"fixed_init_state_count": 50},
        "policy": {"replan_steps": 5, "precision": "bfloat16"},
        "rng": {"inference_seed": 7},
        "parallel": parallel,
        "writer": paired_writer_identity(adapter),
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "contract_reference": f"contract:m{macro}:{condition}:{physical_gpu_ids}",
        "arm": adapter["arm"],
        "role": "validation",
        "mode": "formal",
        "adapter": adapter,
        "paired_control": paired,
        "rows": _rows(macro, condition, successes, adapter),
    }


def test_panel_summary_reports_task_suite_breadth_and_deterministic_top3() -> None:
    rows = _result(
        0,
        "correct",
        _success_keys(lambda suite, task_id, state: state < (task_id % 3)),
    )["rows"]
    summary = summarize_panel(rows)
    assert summary["overall"]["episodes"] == 400
    assert [row["episodes"] for row in summary["per_suite"]] == [100] * 4
    assert summary["nonzero_task_breadth"] == 4
    assert len(summary["top3_tasks"]) == 3
    assert summary["top3_tasks"] == sorted(
        summary["top3_tasks"],
        key=lambda row: (
            -row["successes"],
            ("libero_spatial", "libero_object", "libero_goal", "libero_10").index(row["suite"]),
            row["suite"],
            row["task_id"],
        ),
    )


def test_transition_reports_gained_lost_churn_jaccard_and_exact_mcnemar() -> None:
    left = [
        {"suite": "libero_spatial", "task_id": 1, "init_state_id": index, "success": value}
        for index, value in enumerate((True, False, True, False))
    ]
    right = [
        {"suite": "libero_spatial", "task_id": 1, "init_state_id": index, "success": value}
        for index, value in enumerate((True, True, False, False))
    ]
    overall = paired_transition_summary(left, right)["overall"]
    assert overall == {
        "episodes": 4,
        "retained_success": 1,
        "gained": 1,
        "lost": 1,
        "retained_failure": 1,
        "net": 0,
        "churn": 2,
        "churn_rate": 0.5,
        "success_set_jaccard": 1 / 3,
        "mcnemar_exact_two_sided_p": 1.0,
    }
    assert exact_mcnemar_two_sided_p(0, 3) == 0.25


def test_checkpoint_curve_derives_true_same_root_80_and_checkpoint_churn() -> None:
    success_by_macro = {
        0: _success_keys(lambda _suite, _task, state: state == 0),
        10: _success_keys(lambda suite, task, state: state == 0 or (suite == "libero_spatial" and task == 1 and state == 1)),
        25: _success_keys(lambda _suite, _task, state: state in {0, 1}),
        50: _success_keys(lambda _suite, _task, state: state == 49),
    }
    results = {
        f"root-{macro}": _result(macro, "correct", success_by_macro[macro], physical_gpu_ids=(macro + 2, macro + 3))
        for macro in (0, 10, 25, 50)
    }
    analysis = checkpoint_curve_analysis(results)
    assert analysis["schema_version"] == CHECKPOINT_CURVE_SCHEMA
    assert analysis["panels"]["correct80"]["0"]["overall"] == {
        "successes": 8,
        "episodes": 80,
        "success_rate": 0.1,
    }
    assert analysis["panels"]["correct80"]["50"]["overall"]["successes"] == 0
    assert analysis["panels"]["correct400"]["50"]["overall"]["successes"] == 8
    assert analysis["comparisons"]["correct80"]["0_to_10"]["overall"]["gained"] == 1
    assert analysis["curve_evidence"]["correct400"]["union_successes"] == 24


def test_checkpoint_curve_rejects_missing_state_or_pairing_drift() -> None:
    results = {
        f"root-{macro}": _result(macro, "correct", set())
        for macro in (0, 10, 25, 50)
    }
    results["root-10"]["rows"].pop()
    with pytest.raises(Pi05EvaluationError, match="8x50"):
        checkpoint_curve_analysis(results)
    results["root-10"] = _result(10, "correct", set())
    results["root-10"]["rows"][0]["policy_noise_seeds"][0] += 1
    with pytest.raises(Pi05EvaluationError, match="RNG"):
        checkpoint_curve_analysis(results)


def test_checkpoint_curve_keeps_legacy_read_only_and_rejects_mixed_families() -> None:
    legacy = {
        f"legacy-{macro}": _result(macro, "correct", set(), family="legacy")
        for macro in (0, 10, 25, 50)
    }
    assert checkpoint_curve_analysis(legacy)["method_family"] == "legacy_v6_prior_v1"

    mixed = dict(legacy)
    mixed["legacy-10"] = _result(10, "correct", set(), family="current")
    with pytest.raises(Pi05EvaluationError, match="cannot mix"):
        checkpoint_curve_analysis(mixed)


def test_historical_transition_preserves_families_and_pairs_true_rows() -> None:
    baseline_success = _success_keys(
        lambda _suite, _task, state: state == 0
    )
    candidate_success = _success_keys(
        lambda suite, task, state: state == 0
        or (suite == "libero_spatial" and task == 1 and state == 1)
    )
    baseline = _result(0, "correct", baseline_success, family="legacy")
    candidate = _result(10, "correct", candidate_success, family="current")
    baseline["paired_control"]["git"]["commit"] = "legacy-commit"
    candidate["paired_control"]["git"]["commit"] = "current-commit"
    baseline["paired_control"]["tokenizer"]["manifest_path"] = "/legacy/tokenizer.json"
    candidate["paired_control"]["tokenizer"]["manifest_path"] = "/current/tokenizer.json"
    baseline["paired_control"]["normalization"]["path"] = "/legacy/normalization.json"
    candidate["paired_control"]["normalization"]["path"] = "/current/normalization.json"

    analysis = historical_baseline_transition_analysis(
        {"legacy-root": baseline, "current-root": candidate}
    )
    assert analysis["schema_version"] == HISTORICAL_BASELINE_TRANSITION_SCHEMA
    assert analysis["method_families"] == {
        "historical_baseline": "legacy_v6_prior_v1",
        "current_candidate": "v6_ecp_v2",
    }
    assert analysis["contract_audit"]["checkpoint_curve_membership_claimed"] is False
    assert analysis["panels"]["correct400"]["historical_baseline"]["overall"]["successes"] == 8
    transition = analysis["baseline_to_candidate"]["correct400"]["overall"]
    assert transition["retained_success"] == 8
    assert transition["gained"] == 1
    assert transition["lost"] == 0


def test_historical_transition_rejects_wrong_identity_or_scientific_drift() -> None:
    baseline = _result(0, "correct", set(), family="legacy")
    candidate = _result(10, "correct", set(), family="current")
    duplicate_family = {
        "left": _result(0, "correct", set(), family="current"),
        "right": candidate,
    }
    with pytest.raises(Pi05EvaluationError, match="duplicate method family"):
        historical_baseline_transition_analysis(duplicate_family)

    drifted = copy.deepcopy(candidate)
    drifted["paired_control"]["policy"]["replan_steps"] = 4
    with pytest.raises(Pi05EvaluationError, match="shared scientific contract"):
        historical_baseline_transition_analysis(
            {"legacy": baseline, "current": drifted}
        )

    drifted = copy.deepcopy(candidate)
    drifted["rows"][0]["policy_noise_seeds"][0] += 1
    with pytest.raises(Pi05EvaluationError, match="RNG"):
        historical_baseline_transition_analysis(
            {"legacy": baseline, "current": drifted}
        )


def test_historical_transition_reads_legacy_immutable_and_reaggregates_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = tmp_path / "legacy"
    current_root = tmp_path / "current"
    legacy_root.mkdir()
    current_root.mkdir()
    legacy = _result(0, "correct", set(), family="legacy")
    current = _result(10, "correct", set(), family="current")
    (legacy_root / "results.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    (current_root / "results.json").write_text("{}\n", encoding="utf-8")
    calls = []

    def aggregate(root: Path) -> dict:
        calls.append(root)
        return current

    monkeypatch.setattr("ember.pi05_eval_results.aggregate_run", aggregate)
    output = tmp_path / "transition.json"
    result = analyze_historical_baseline_transition(
        legacy_root, current_root, output
    )
    assert result["schema_version"] == HISTORICAL_BASELINE_TRANSITION_SCHEMA
    assert calls == [current_root.resolve()]
    assert output.is_file()
    with pytest.raises(Pi05EvaluationError, match="already exists"):
        analyze_historical_baseline_transition(legacy_root, current_root, output)


@pytest.mark.parametrize(
    ("field", "value"),
    (("task_id", 0), ("language", "a plausible but unsealed task language")),
)
def test_formal_tasks_reject_unsealed_task_identity(field: str, value: object) -> None:
    tasks = _tasks()
    tasks[0][field] = value
    with pytest.raises(Pi05EvaluationError, match="exactly match the sealed 8 tasks"):
        _formal_tasks({"tasks": tasks})


def test_checkpoint_curve_reaggregates_roots_and_publishes_immutably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / f"root-{macro}" for macro in (0, 10, 25, 50)]
    payloads = {}
    for macro, root in zip((0, 10, 25, 50), roots):
        root.mkdir()
        (root / "results.json").write_text("{}\n", encoding="utf-8")
        payloads[root.resolve()] = _result(macro, "correct", set())
    calls = []

    def aggregate(root: Path) -> dict:
        calls.append(root)
        return payloads[root]

    monkeypatch.setattr("ember.pi05_eval_results.aggregate_run", aggregate)
    output = tmp_path / "analysis.json"
    result = analyze_checkpoint_curve(roots, output)
    assert result["schema_version"] == CHECKPOINT_CURVE_SCHEMA
    assert calls == [root.resolve() for root in roots]
    assert output.is_file()
    with pytest.raises(Pi05EvaluationError, match="already exists"):
        analyze_checkpoint_curve(roots, output)


def test_six_arm_audit_validates_pairing_same_offset_and_no_video() -> None:
    correct = _success_keys(lambda _suite, _task, state: state == 0)
    controls = {
        condition: (correct if condition == "same_task_other" else set())
        for condition in SIX_ARM_CONDITIONS
    }
    controls["correct"] = correct
    results = {
        f"root-{condition}": _result(
            50,
            condition,
            controls[condition],
            physical_gpu_ids=(index + 2, index + 3),
        )
        for index, condition in enumerate(SIX_ARM_CONDITIONS)
    }
    analysis = six_arm_paired_analysis(results)
    assert analysis["schema_version"] == SIX_ARM_AUDIT_SCHEMA
    assert analysis["comparisons_to_correct"]["cross_suite_wrong"]["overall"]["correct_only"] == 8
    assert analysis["comparisons_to_correct"]["same_task_other"]["interpretation"] == "same-task cross-video robustness"
    same_writer = results["root-same_task_other"]["rows"][0]["writer"]
    assert same_writer["teacher_demo_indices"][0] == (
        same_writer["teacher_reference_demo_indices"][0] + SAME_TASK_OTHER_OFFSET
    ) % 50
    assert results["root-no_video"]["rows"][0]["writer"]["teacher_video_frames_used"] is False


def test_six_arm_audit_rejects_contract_or_episode_drift() -> None:
    results = {
        f"root-{condition}": _result(50, condition, set())
        for condition in SIX_ARM_CONDITIONS
    }
    drifted = copy.deepcopy(results)
    drifted["root-cross_suite_wrong"]["paired_control"]["parallel"] = {
        "physical_gpu_ids": [7],
        "physical_gpu_count": 1,
        "worker_count": 9,
        "replicas_per_gpu": 9,
        "envs_per_replica": 3,
        "writer_generation_batch_size": 64,
    }
    six_arm_paired_analysis(drifted)
    drifted = copy.deepcopy(results)
    drifted["root-cross_suite_wrong"]["paired_control"]["policy"]["replan_steps"] = 4
    with pytest.raises(Pi05EvaluationError, match="scientific contract"):
        six_arm_paired_analysis(drifted)
    drifted = copy.deepcopy(results)
    drifted["root-no_video"]["rows"][0]["writer"]["teacher_video_frames_used"] = True
    with pytest.raises(Pi05EvaluationError, match="video condition"):
        six_arm_paired_analysis(drifted)
    missing = dict(results)
    missing.pop("root-reversed")
    with pytest.raises(Pi05EvaluationError, match="exactly"):
        six_arm_paired_analysis(missing)
