from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_profile import (
    profile_success_key_application,
)
from ember.expert_manifold.v6_prior_runtime import _reconcile_metrics_cursor
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


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"))


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
    success_count = 4 if ordinal in {0, 6, 12, 18, 19, 20} else 1
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
        success_count=success_count,
        trajectory_rows=tuple(
            {
                "rollout_cursor": lane,
                "environment_seed": lane + 10,
                "policy_noise_seeds": [lane + 20],
                "success": lane < success_count,
                "steps": 5,
                "reward_sum": float(lane < success_count),
                "replan_count": 1,
                "retained_observation_tensors": 0,
                "retained_action_tensors": 0,
            }
            for lane in range(4)
        ),
        rollout_seconds=2.0,
        correct_raw_frames=100,
        correct_sampled_frames=21,
        negative_raw_frames=100,
        negative_sampled_frames=21,
    )


def test_full48_gather_sorts_train24_and_never_rescales_program_cotangents() -> None:
    local = [_objective(index) for index in reversed(range(24))]
    correct, negative, cotangents, success_counts = _gather_full48(
        local, _context()
    )
    assert correct.shape == negative.shape == (24, 256)
    assert cotangents.shape == (24, 2, 3)
    assert torch.equal(correct[:, :24], torch.eye(24))
    assert torch.equal(negative[:, 24:48], torch.eye(24))
    for ordinal in range(24):
        assert torch.equal(
            cotangents[ordinal], torch.full((2, 3), float(ordinal + 1))
        )
        assert success_counts[ordinal] == _objective(ordinal).success_count


def test_full48_gather_rejects_duplicate_or_missing_task_ordinals() -> None:
    local = [_objective(index) for index in range(24)]
    local[-1] = _objective(22)
    with pytest.raises(ExpertManifoldError, match="task order changed"):
        _gather_full48(local, _context())


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
                "rollout_count": 96,
                "all_success_task_count_min": 6,
                "all_success_task_per_suite_min": 1,
                "original_feature_rank": 48,
                "active_regularized_gram_condition_number_max": 200.0,
                "unprotected_projected_feature_energy_ratio_median_min": 0.2,
                "protected_to_unprotected_motion_ratio_max": 1e-5,
                "negative_to_unprotected_motion_rms_max": 0.15,
                "unprotected_descent_fraction_min": 0.8,
                "negative_null_task_count_min": 18,
                "negative_null_per_kind_min": 6,
                "predicted_observed_relative_rms_max": 0.005,
                "protected_to_unprotected_lora_response_ratio_max": 1e-5,
                "protected_fixed_action_response_rms_max": 1e-6,
                "unprotected_fixed_action_probe_task_count": 4,
                "production_wall_ratio_max": 1.25,
                "negative_policy_forwards": 0,
                "oom_count": 0,
                "nonfinite_count": 0,
            },
        }
    }


def _profile_row() -> dict:
    protected = {0, 6, 12, 18, 19, 20}
    records = []
    for ordinal in range(24):
        record = _task_record(_objective(ordinal))
        records.append(record)
    return {
        "update": {
            "correct_conditions": 24,
            "negative_conditions": 24,
            "current_protected_conditions": 6,
            "unprotected_correct_conditions": 18,
            "anchor_constraint_rows": 6,
            "anchor_rank": 6,
            "original_feature_rank": 48,
            "projected_feature_rank": 42,
            "active_regularized_gram_condition_number": 100.0,
            "correct_cotangent_rms": 2.0,
            "predicted_protected_to_unprotected_ratio": 0.0,
            "predicted_negative_to_unprotected_ratio": 0.1,
            "unprotected_projected_feature_energy_ratio_median": 0.8,
            "value_delta_rms": 0.01,
        },
        "application": {"predicted_observed_relative_rms": 0.001},
        "lora_response": {
            "protected_lora_a_to_unprotected_ratio": 0.0,
            "protected_lora_b_to_unprotected_ratio": 0.0,
            "protected_effective_ba_to_unprotected_ratio": 0.0,
            "unprotected_lora_a_response_rms": 0.01,
            "unprotected_lora_b_response_rms": 0.02,
            "unprotected_effective_ba_response_rms": 0.03,
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
            "task_count": 24,
            "protected_task_count": 6,
            "unprotected_task_count": 18,
            "unprotected_correct_motion_rms": 1.0,
            "protected_to_unprotected_motion_ratio": 0.0,
            "negative_to_unprotected_motion_ratio": 0.1,
            "unprotected_descent_passing_tasks": 18,
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
        "success_outcomes": {
            "rollouts": 96,
            "success_episodes": 42,
            "failure_episodes": 54,
            "all_success_tasks": 6,
            "all_failure_tasks": 0,
            "all_success_tasks_per_suite": {
                "libero_spatial": 1,
                "libero_object": 1,
                "libero_goal": 1,
                "libero_10": 3,
            },
            "retained_observation_tensors": 0,
            "retained_action_tensors": 0,
            "trajectory_replay_policy_forwards": 0,
            "trajectory_replay_cfm_forwards": 0,
            "reward_gradient_count": 0,
        },
        "success_key_bank": {
            "current_all_success_count": 6,
            "persisted_before_count": 0,
            "constraint_row_count": 6,
            "newly_stored_count": 6,
            "persisted_after_count": 6,
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


def test_mechanism_profile_requires_rank_closure_outcomes_and_scaled_wall() -> None:
    passed, evidence = _profile_passes(_profile_config(), _profile_row())
    assert passed is True
    assert all(evidence["checks"].values())
    for section, key, value in (
        ("update", "projected_feature_rank", 41),
        ("update", "active_regularized_gram_condition_number", 201.0),
        ("update", "unprotected_projected_feature_energy_ratio_median", 0.1),
        ("application", "predicted_observed_relative_rms", 0.01),
        ("task_local_motion", "negative_to_unprotected_motion_ratio", 0.3),
        ("lora_response", "protected_effective_ba_to_unprotected_ratio", 0.1),
        ("success_outcomes", "retained_action_tensors", 1),
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
    row["step_seconds"] = 32.0
    assert _profile_passes(_profile_config(), row)[0] is False


def test_task_local_profile_protects_successes_and_keeps_unprotected_descent() -> None:
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


def test_success_key_application_reports_the_final_shared_write() -> None:
    anchors = torch.eye(3)
    delta = torch.zeros((3, 2, 2))
    delta[2, 0, 0] = 1.0
    evidence = profile_success_key_application(
        anchors[:2], delta, torch.tensor([True, True] + [False] * 22)
    )
    assert evidence["constraint_row_count"] == 2
    assert evidence["anchor_program_motion_rms"] == 0


def test_task_record_reports_b20_k4_outcomes_and_no_replay() -> None:
    row = _task_record(_objective(0))
    assert row["source_action_queries"] == 20
    assert row["physical_correct_policy_forwards"] == 2
    assert row["success_count"] == 4
    assert row["all_success"] is True
    assert row["trajectory_replay_policy_forwards"] == 0
    assert row["trajectory_replay_cfm_forwards"] == 0
    assert row["reward_gradient_count"] == 0
    assert row["negative_policy_forwards"] == 0


def test_resume_reconciles_post_checkpoint_metrics_into_failure_packet(
    tmp_path: Path,
) -> None:
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
