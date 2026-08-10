from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ember.eval_adapters import paired_writer_identity
from ember.expert_manifold.video_schedule import SAME_TASK_OTHER_OFFSET
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.analysis import (
    CHECKPOINT_CURVE_SCHEMA,
    HISTORICAL_BASELINE_TRANSITION_SCHEMA,
    SIX_ARM_AUDIT_SCHEMA,
    SIX_ARM_CONDITIONS,
    _formal_panel_index,
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
from pi05_eval_analysis_fixture import (
    result as _result,
    success_keys as _success_keys,
    tasks as _tasks,
)


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
    mixed["legacy-10"] = _result(10, "correct", set(), family="ecp")
    with pytest.raises(Pi05EvaluationError, match="cannot mix"):
        checkpoint_curve_analysis(mixed)


def test_checkpoint_curve_accepts_tangent_family_but_not_macro100() -> None:
    tangent = {
        f"tangent-{macro}": _result(
            macro, "correct", set(), family="tangent"
        )
        for macro in (0, 10, 25, 50)
    }
    assert (
        checkpoint_curve_analysis(tangent)["method_family"]
        == "v6_tangent_tube_v3"
    )

    tangent["tangent-50"] = _result(
        100, "correct", set(), family="tangent"
    )
    with pytest.raises(Pi05EvaluationError, match="checkpoint curve"):
        checkpoint_curve_analysis(tangent)


def test_checkpoint_curve_accepts_residual_memory_identity_changes_only() -> None:
    residual = {
        f"residual-{macro}": _result(
            macro,
            "correct",
            set(),
            family="residual",
        )
        for macro in (0, 10, 25, 50)
    }
    assert (
        checkpoint_curve_analysis(residual)["method_family"]
        == "v6_condition_residual_v2"
    )
    residual["residual-25"]["adapter"]["writer_asset"]["residual_state"][
        "shape"
    ] = [128, 320, 256]
    residual["residual-25"]["paired_control"]["writer"] = paired_writer_identity(
        residual["residual-25"]["adapter"]
    )
    with pytest.raises(Pi05EvaluationError, match="scientific contract"):
        checkpoint_curve_analysis(residual)


def test_residual_family_rejects_legacy_deployment_seal_and_missing_v8_state() -> None:
    result = _result(10, "correct", set(), family="residual")
    result["adapter"]["evaluation_authority"]["formal_status"] = (
        "sealed_from_unchanged_v6_deployment_graph"
    )
    result["paired_control"]["writer"] = paired_writer_identity(result["adapter"])
    with pytest.raises(Pi05EvaluationError, match="sealed formal"):
        checkpoint_curve_analysis(
            {
                "m0": _result(0, "correct", set(), family="residual"),
                "m10": result,
                "m25": _result(25, "correct", set(), family="residual"),
                "m50": _result(50, "correct", set(), family="residual"),
            }
        )
    result = _result(10, "correct", set(), family="residual")
    result["rows"][0]["writer"].pop("writer_program_residual_value_count")
    with pytest.raises(Pi05EvaluationError, match="episode evidence"):
        checkpoint_curve_analysis(
            {
                "m0": _result(0, "correct", set(), family="residual"),
                "m10": result,
                "m25": _result(25, "correct", set(), family="residual"),
                "m50": _result(50, "correct", set(), family="residual"),
            }
        )


def test_formal_panel_accepts_anchored_reconciliation_v3_evidence() -> None:
    result = _result(
        10,
        "correct",
        set(),
        family="reconciliation",
    )
    assert len(_formal_panel_index(result)) == 400


@pytest.mark.parametrize("macro", (1, 2))
def test_formal_panel_accepts_reward_credit_program_v1_evidence(macro: int) -> None:
    result = _result(macro, "correct", set(), family="reward")
    assert len(_formal_panel_index(result)) == 400


def test_reward_credit_historical_transition_rejects_unregistered_macro() -> None:
    with pytest.raises(Pi05EvaluationError, match="candidate"):
        historical_baseline_transition_analysis(
            {
                "baseline": _result(0, "correct", set(), family="residual"),
                "current": _result(10, "correct", set(), family="reward"),
            }
        )


@pytest.mark.parametrize(
    (
        "baseline_family",
        "candidate_family",
        "candidate_macro",
        "expected_baseline_family",
        "expected_candidate_family",
    ),
    (
        ("legacy", "ecp", 10, "legacy_v6_prior_v1", "v6_ecp_v2"),
        ("legacy", "ecp", 25, "legacy_v6_prior_v1", "v6_ecp_v2"),
        ("legacy", "ecp", 50, "legacy_v6_prior_v1", "v6_ecp_v2"),
        ("legacy", "tangent", 10, "legacy_v6_prior_v1", "v6_tangent_tube_v3"),
        ("legacy", "tangent", 25, "legacy_v6_prior_v1", "v6_tangent_tube_v3"),
        ("legacy", "tangent", 50, "legacy_v6_prior_v1", "v6_tangent_tube_v3"),
        ("legacy", "residual", 10, "legacy_v6_prior_v1", "v6_condition_residual_v2"),
        ("legacy", "residual", 25, "legacy_v6_prior_v1", "v6_condition_residual_v2"),
        ("legacy", "residual", 50, "legacy_v6_prior_v1", "v6_condition_residual_v2"),
        (
            "residual",
            "reconciliation",
            10,
            "v6_condition_residual_v2",
            "v6_anchored_reconciliation_v3",
        ),
        (
            "residual",
            "reconciliation",
            25,
            "v6_condition_residual_v2",
            "v6_anchored_reconciliation_v3",
        ),
        (
            "residual",
            "reward",
            1,
            "v6_condition_residual_v2",
            "v6_reward_credit_program_v1",
        ),
        (
            "residual",
            "reward",
            2,
            "v6_condition_residual_v2",
            "v6_reward_credit_program_v1",
        ),
    ),
)
def test_historical_transition_preserves_families_and_pairs_true_rows(
    baseline_family: str,
    candidate_family: str,
    candidate_macro: int,
    expected_baseline_family: str,
    expected_candidate_family: str,
) -> None:
    baseline_success = _success_keys(
        lambda _suite, _task, state: state == 0
    )
    candidate_success = _success_keys(
        lambda suite, task, state: state == 0
        or (suite == "libero_spatial" and task == 1 and state == 1)
    )
    baseline = _result(0, "correct", baseline_success, family=baseline_family)
    candidate = _result(
        candidate_macro, "correct", candidate_success, family=candidate_family
    )
    if candidate_family == "tangent":
        candidate["adapter"]["evaluation_authority"]["formal_status"] = (
            "sealed_from_unchanged_v6_deployment_graph"
        )
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
        "historical_baseline": expected_baseline_family,
        "current_candidate": expected_candidate_family,
    }
    assert analysis["contract_audit"]["checkpoint_curve_membership_claimed"] is False
    assert analysis["panels"]["correct400"]["historical_baseline"]["overall"]["successes"] == 8
    transition = analysis["baseline_to_candidate"]["correct400"]["overall"]
    assert transition["retained_success"] == 8
    assert transition["gained"] == 1
    assert transition["lost"] == 0


def test_historical_transition_rejects_wrong_identity_or_scientific_drift() -> None:
    baseline = _result(0, "correct", set(), family="legacy")
    candidate = _result(10, "correct", set(), family="ecp")
    duplicate_family = {
        "left": _result(0, "correct", set(), family="ecp"),
        "right": candidate,
    }
    with pytest.raises(Pi05EvaluationError, match="duplicate method family"):
        historical_baseline_transition_analysis(duplicate_family)

    ecp_and_tangent = {
        "ecp": candidate,
        "tangent": _result(10, "correct", set(), family="tangent"),
    }
    with pytest.raises(Pi05EvaluationError, match="supported baseline"):
        historical_baseline_transition_analysis(ecp_and_tangent)

    wrong_macro = _result(100, "correct", set(), family="ecp")
    with pytest.raises(Pi05EvaluationError, match="candidate"):
        historical_baseline_transition_analysis(
            {"legacy": baseline, "current": wrong_macro}
        )

    wrong_tangent_macro = _result(100, "correct", set(), family="tangent")
    with pytest.raises(Pi05EvaluationError, match="candidate"):
        historical_baseline_transition_analysis(
            {"legacy": baseline, "current": wrong_tangent_macro}
        )

    wrong_reconciliation_macro = _result(
        50, "correct", set(), family="reconciliation"
    )
    with pytest.raises(Pi05EvaluationError, match="candidate"):
        historical_baseline_transition_analysis(
            {
                "baseline": _result(0, "correct", set(), family="residual"),
                "current": wrong_reconciliation_macro,
            }
        )

    with pytest.raises(Pi05EvaluationError, match="supported baseline"):
        historical_baseline_transition_analysis(
            {
                "legacy": baseline,
                "current": _result(
                    10, "correct", set(), family="reconciliation"
                ),
            }
        )

    with pytest.raises(Pi05EvaluationError, match="baseline"):
        historical_baseline_transition_analysis(
            {
                "baseline": _result(10, "correct", set(), family="residual"),
                "current": _result(
                    10, "correct", set(), family="reconciliation"
                ),
            }
        )

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


def test_all_formal_analysis_rejects_unsealed_or_dirty_native_roots() -> None:
    baseline = _result(0, "correct", set(), family="legacy")
    candidate = _result(10, "correct", set(), family="tangent")
    candidate["adapter"]["evaluation_authority"]["formal_status"] = (
        "blocked_until_live_a40_resume_profile_evidence"
    )
    with pytest.raises(Pi05EvaluationError, match="sealed formal validation"):
        historical_baseline_transition_analysis(
            {"legacy": baseline, "current": candidate}
        )

    candidate["adapter"]["evaluation_authority"]["formal_status"] = (
        "sealed_from_unchanged_v6_deployment_graph"
    )
    candidate["paired_control"]["git"]["dirty_paths"] = ["local-change"]
    with pytest.raises(Pi05EvaluationError, match="sealed formal validation"):
        historical_baseline_transition_analysis(
            {"legacy": baseline, "current": candidate}
        )


def test_historical_transition_reads_legacy_immutable_and_reaggregates_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = tmp_path / "legacy"
    current_root = tmp_path / "current"
    legacy_root.mkdir()
    current_root.mkdir()
    legacy = _result(0, "correct", set(), family="legacy")
    current = _result(10, "correct", set(), family="ecp")
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
