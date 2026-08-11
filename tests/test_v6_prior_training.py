from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import ember.expert_manifold.v6_prior_training as prior_training
from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_candidate_guard import PairedTaskEvidence
from ember.expert_manifold.v6_prior_profile import profile_success_key_application
from ember.expert_manifold.v6_prior_runtime import _reconcile_metrics_cursor
from ember.expert_manifold.v6_prior_step import GeneratedConditionGraph
from ember.expert_manifold.v6_prior_training import (
    TaskObjective,
    _gather_full48,
    _profile_passes,
    _profile_task_local_motion,
    _task_record,
    build_parser,
)
from ember.pi05_source_checkpoint import DistributedContext


_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
_HARMFUL = {0, 6}
_STABLE = {12, 18, 19, 20}
_BENEFICIAL = {1, 7}


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"))


def _paired(ordinal: int) -> PairedTaskEvidence:
    if ordinal in _HARMFUL:
        base, candidate = (True, True), (True, False)
    elif ordinal in _STABLE:
        base = candidate = (True, True)
    elif ordinal in _BENEFICIAL:
        base, candidate = (False, False), (True, False)
    else:
        base = candidate = (False, False)
    rows = tuple(
        {
            "arm": arm,
            "rollout_cursor": lane,
            "environment_seed": lane + 10,
            "policy_seed_root": 13,
            "policy_noise_seeds": [lane + 20],
            "success": success[lane],
            "steps": 5,
            "reward_sum": float(success[lane]),
            "replan_count": 1,
        }
        for arm, success in (("base", base), ("candidate", candidate))
        for lane in range(2)
    )
    return PairedTaskEvidence(
        base_success=base,
        candidate_success=candidate,
        trajectory_rows=rows,
        exact_pair_count=2,
        candidate_program_motion_rms=0.1,
        candidate_lora_response_rms=0.2,
        candidate_action_response_rms=0.3,
        rollout_seconds=2.0,
    )


def _objective(ordinal: int) -> TaskObjective:
    correct = torch.zeros(256, dtype=torch.float32)
    negative = torch.zeros(256, dtype=torch.float32)
    correct[ordinal] = 1.0
    negative[ordinal + 24] = 1.0
    task = SimpleNamespace(
        ordinal=ordinal,
        global_task_id=ordinal + 100,
        suite=_SUITES[ordinal // 6],
        task_id=ordinal % 6,
    )
    program = torch.zeros(1, 2, 3)
    graph = GeneratedConditionGraph(
        correct_lora={},
        program_leaf=program,
        program_input_before=program,
        base_program_slots=program,
        residual_before=program,
        correct_feature=correct,
        negative_feature=negative,
        correct_raw_frames=100,
        correct_sampled_frames=21,
        negative_raw_frames=100,
        negative_sampled_frames=21,
    )
    return TaskObjective(
        task=task,
        task_visit=3,
        teacher_demo=4,
        counterfactual_kind=("reversed", "shuffled", "wrong")[ordinal % 3],
        counterfactual_task=None,
        counterfactual_demo=None,
        functional_loss=torch.tensor(float(ordinal), dtype=torch.float32),
        correct_feature=correct,
        negative_feature=negative,
        program_cotangent=torch.full((2, 3), float(ordinal + 1)),
        graph=graph,
        correct_raw_frames=100,
        correct_sampled_frames=21,
        negative_raw_frames=100,
        negative_sampled_frames=21,
        paired=_paired(ordinal),
    )


def test_full48_gather_sorts_train24_and_never_rescales_program_cotangents() -> None:
    local = [_objective(index) for index in reversed(range(24))]
    correct, negative, cotangents = _gather_full48(local, _context())
    assert correct.shape == negative.shape == (24, 256)
    assert cotangents.shape == (24, 2, 3)
    assert torch.equal(correct[:, :24], torch.eye(24))
    assert torch.equal(negative[:, 24:48], torch.eye(24))
    for ordinal in range(24):
        assert torch.equal(
            cotangents[ordinal], torch.full((2, 3), float(ordinal + 1))
        )


def test_full48_gather_rejects_duplicate_or_missing_task_ordinals() -> None:
    local = [_objective(index) for index in range(24)]
    local[-1] = _objective(22)
    with pytest.raises(ExpertManifoldError, match="task order changed"):
        _gather_full48(local, _context())


def test_full48_gather_keeps_cotangents_aligned_across_padded_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = [
        7, 0, 23, -1, 2, 11, 19, 3, 5, 13, 1, 22, 8,
        15, 4, 18, 6, 14, 9, 20, 10, 16, 12, 21, 17,
    ]
    payload = torch.zeros(25, 513, dtype=torch.float32)
    payload[:, 0].fill_(-1)
    cotangents = torch.zeros(25, 2, 3, dtype=torch.float32)
    for row, ordinal in enumerate(order):
        if ordinal < 0:
            continue
        objective = _objective(ordinal)
        payload[row, 0] = ordinal
        payload[row, 1:257] = objective.correct_feature
        payload[row, 257:] = objective.negative_feature
        cotangents[row] = objective.program_cotangent
    gathered = iter((payload, cotangents))
    monkeypatch.setattr(
        prior_training,
        "_all_gather_fixed",
        lambda value, context: next(gathered),
    )

    correct, negative, aligned = _gather_full48(
        [_objective(index) for index in range(5)],
        DistributedContext(0, 0, 5, torch.device("cpu")),
    )

    assert torch.equal(correct[:, :24], torch.eye(24))
    assert torch.equal(negative[:, 24:48], torch.eye(24))
    for ordinal in range(24):
        assert torch.equal(aligned[ordinal], torch.full((2, 3), float(ordinal + 1)))


def _profile_config() -> dict:
    return {
        "profile_run": {
            "throughput_baseline": {
                "step_seconds": 20.0,
                "source_tasks_per_rank": 4,
            },
            "gates": {
                "task_count": 24,
                "video_count": 24,
                "source_action_query_count": 480,
                "paired_state_count": 48,
                "base_rollout_count": 48,
                "candidate_rollout_count": 48,
                "rollout_count": 96,
                "discordant_state_count_min": 4,
                "harmful_task_count_min": 2,
                "harmful_suite_count_min": 2,
                "candidate_gain_count_min": 1,
                "original_feature_rank": 48,
                "projected_feature_rank_min": 24,
                "projected_to_blind_energy_ratio_min": 0.25,
                "final_guard_violation_count": 0,
                "protected_to_unprotected_motion_ratio_max": 1e-5,
                "negative_to_unprotected_motion_rms_max": 0.15,
                "negative_null_task_count_min": 18,
                "negative_null_per_kind_min": 6,
                "predicted_observed_relative_rms_max": 0.005,
                "protected_to_unprotected_lora_response_ratio_max": 1e-5,
                "protected_fixed_action_response_rms_max": 1e-6,
                "unprotected_fixed_action_probe_task_count": 4,
                "production_wall_ratio_max": 1.5,
                "negative_policy_forwards": 0,
                "oom_count": 0,
                "nonfinite_count": 0,
            },
        }
    }


def _profile_row() -> dict:
    protected = _HARMFUL | _STABLE
    records = [_task_record(_objective(ordinal)) for ordinal in range(24)]
    response = {
        suite: {
            "program_motion_rms_max": 0.1,
            "lora_response_rms_max": 0.2,
            "action_response_rms_max": 0.3,
        }
        for suite in _SUITES
    }
    return {
        "blind_update": {
            "current_protected_conditions": 0,
            "anchor_constraint_rows": 0,
            "value_delta_rms": 0.01,
        },
        "candidate_guard_projection": {
            "persisted_guard_rows": 0,
            "current_stable_guard_rows": 4,
            "current_harmful_guard_rows": 2,
            "current_guard_rows": 6,
            "total_guard_rows": 6,
            "guard_rank": 6,
            "original_feature_rank": 48,
            "projected_feature_rank": 42,
            "final_guard_violation_count": 0,
            "projection_changed": True,
            "projected_to_blind_energy_ratio": 0.8,
            "blind_projected_inner_product": 1.0,
            "blind_projected_cosine": 0.9,
            "projected_delta_rms": 0.008,
        },
        "application": {"predicted_observed_relative_rms": 0.001},
        "lora_response": {
            "protected_lora_a_to_unprotected_ratio": 0.0,
            "protected_lora_b_to_unprotected_ratio": 0.0,
            "protected_effective_ba_to_unprotected_ratio": 0.0,
            "protected_fixed_action_probe_task_count": 4,
            "protected_fixed_action_probe_suites": list(_SUITES),
            "protected_fixed_action_response_max": 0.0,
            "unprotected_fixed_action_probe_task_count": 4,
            "unprotected_fixed_action_probe_suites": list(_SUITES),
            "unprotected_fixed_action_passing_task_count": 4,
            "fixed_action_probe_task_count": 8,
            "fixed_action_probe_policy_forwards": 16,
        },
        "task_local_motion": {
            "unprotected_correct_motion_rms": 1.0,
            "protected_to_unprotected_motion_ratio": 0.0,
            "negative_to_unprotected_motion_ratio": 0.1,
            "negative_null_passing_tasks": 24,
            "rows": [
                {
                    "task_ordinal": ordinal,
                    "protected": ordinal in protected,
                    "negative_to_unprotected_motion_rms": 0.1,
                }
                for ordinal in range(24)
            ],
        },
        "success_key_application": {
            "constraint_row_count": 6,
            "current_protected_task_count": 6,
            "anchor_program_motion_rms": 0.0,
            "anchor_program_motion_max_abs": 0.0,
        },
        "paired_outcomes": {
            "paired_states": 48,
            "base_rollouts": 48,
            "candidate_rollouts": 48,
            "base_successes": 8,
            "candidate_successes": 8,
            "losses": 2,
            "gains": 2,
            "discordant_states": 4,
            "harmful_task_count": 2,
            "beneficial_task_count": 2,
            "indifferent_task_count": 20,
            "stable_success_task_count": 4,
            "rollouts": 96,
            "exact_pair_records": 48,
            "harmful_tasks_per_suite": {
                "libero_spatial": 1,
                "libero_object": 1,
                "libero_goal": 0,
                "libero_10": 0,
            },
        },
        "candidate_response_by_suite": response,
        "success_key_bank": {
            "current_stable_success_count": 4,
            "persisted_before_count": 0,
            "newly_stored_count": 4,
            "persisted_after_count": 4,
        },
        "task_records": records,
        "world_size": 6,
        "task_counts_per_rank": [4] * 6,
        "maximum_tasks_per_rank": 4,
        "step_seconds": 20.0,
        "functional_loss": 1.0,
        "program_cotangent_rms": 2.0,
        "negative_policy_forwards": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
    }


def test_mechanism_profile_requires_pairs_guards_closure_and_scaled_wall() -> None:
    passed, evidence = _profile_passes(_profile_config(), _profile_row())
    assert passed is True
    assert all(evidence["checks"].values())
    for section, key, value in (
        ("candidate_guard_projection", "projected_feature_rank", 23),
        ("candidate_guard_projection", "projected_to_blind_energy_ratio", 0.1),
        ("candidate_guard_projection", "final_guard_violation_count", 1),
        ("application", "predicted_observed_relative_rms", 0.01),
        ("task_local_motion", "negative_to_unprotected_motion_ratio", 0.3),
        ("lora_response", "protected_effective_ba_to_unprotected_ratio", 0.1),
        ("paired_outcomes", "discordant_states", 3),
    ):
        row = _profile_row()
        row[section][key] = value
        assert _profile_passes(_profile_config(), row)[0] is False
    row = _profile_row()
    row["maximum_tasks_per_rank"] = 5
    row["task_counts_per_rank"] = [5, 5, 5, 5, 4]
    row["world_size"] = 5
    row["step_seconds"] = 25.0
    assert _profile_passes(_profile_config(), row)[0] is True
    row["step_seconds"] = 38.0
    assert _profile_passes(_profile_config(), row)[0] is False


def test_task_local_profile_closes_guards_and_keeps_unprotected_descent() -> None:
    protected = torch.zeros(24, dtype=torch.bool)
    protected[:6] = True
    cotangents = torch.ones((24, 2, 3), dtype=torch.float32)
    correct = torch.full((24, 2, 3), -0.5)
    correct[protected] = 0
    motion = torch.cat((correct, torch.full((24, 2, 3), 0.05)))
    evidence = _profile_task_local_motion(
        cotangents,
        motion,
        protected,
        _profile_config()["profile_run"]["gates"],
    )
    assert evidence["protected_to_unprotected_motion_ratio"] == 0
    assert evidence["unprotected_descent_passing_tasks"] == 18
    assert evidence["negative_null_passing_tasks"] == 24


def test_success_key_application_reports_final_guarded_write() -> None:
    anchors = torch.eye(3)
    delta = torch.zeros((3, 2, 2))
    delta[2, 0, 0] = 1.0
    evidence = profile_success_key_application(
        anchors[:2], delta, torch.tensor([True, True] + [False] * 22)
    )
    assert evidence["constraint_row_count"] == 2
    assert evidence["anchor_program_motion_rms"] == 0


def test_task_record_reports_b20_exact_pairs_and_no_reward_backward() -> None:
    row = _task_record(_objective(0))
    assert row["source_action_queries"] == 20
    assert row["physical_correct_policy_forwards"] == 2
    assert row["base_rollouts"] == row["candidate_rollouts"] == 2
    assert row["harmful"] is True
    assert row["exact_pair_count"] == 2
    assert row["trajectory_replay_policy_forwards"] == 0
    assert row["trajectory_replay_cfm_forwards"] == 0
    assert row["reward_gradient_count"] == 0
    assert row["candidate_action_response_rms"] > 0


def test_resume_reconciles_post_checkpoint_metrics_into_failure_packet(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"macro": macro, "value": macro}) + "\n"
            for macro in range(1, 18)
        ),
        encoding="utf-8",
    )
    assert _reconcile_metrics_cursor(
        metrics, context=_context(), expected_rows=10
    ) == 10
    retained = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["macro"] for row in retained] == list(range(1, 11))


def test_cli_exposes_only_residual_profile_and_formal_modes() -> None:
    parser = build_parser()
    mode = next(action for action in parser._actions if action.dest == "mode")
    assert tuple(mode.choices) == ("mechanism-profile", "formal")
