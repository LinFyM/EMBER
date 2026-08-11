from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_runtime import _reconcile_metrics_cursor
from ember.expert_manifold.v6_prior_profile import (
    profile_success_guard_application as _profile_success_guard_application,
)
from ember.expert_manifold.v6_reward_credit import (
    SuccessGuardProjectionSummary,
    SuccessRetentionCreditSummary,
)
from ember.expert_manifold.v6_prior_training import (
    TaskObjective,
    _gather_full48,
    _profile_passes,
    _profile_task_local_motion,
    _task_record,
    build_parser,
)
from ember.pi05_source_checkpoint import DistributedContext


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
        suite=f"suite-{ordinal // 6}",
        task_id=ordinal % 6,
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
        source_program_cotangent=torch.full((2, 3), float(ordinal + 1)),
        retention_credit=SuccessRetentionCreditSummary(
            successes=1,
            failures=3,
            success_episode_ids=(0,),
            replay_chunks=2,
            flow_panel_chunks=5,
            flow_panel_row_indices=(0, 1),
            executed_action_steps=7,
            mc_samples=4,
            functional_policy_forwards=4,
            episode_objectives=(0.5,),
            lora_gradient_rms=(0.2,),
            program_cotangent_rms=(0.1,),
        ),
        guard_projection=SuccessGuardProjectionSummary(
            constraint_count=1,
            active_constraint_count=0,
            active_constraint_ordinals=(),
            raw_feasible=True,
            changed=False,
            blind_direction_rms=1.0,
            safe_direction_rms=1.0,
            safe_to_blind_norm_ratio=1.0,
            blind_safe_cosine=1.0,
            source_descent_ratio=1.0,
            maximum_constraint_value=-0.1,
        ),
        retention_program_cotangents=None,
        trajectory_rows=tuple(
            {
                "success": lane == 0,
                "steps": 5,
                "replay_chunks": 2 if lane == 0 else 0,
            }
            for lane in range(4)
        ),
        rollout_seconds=2.0,
        retention_seconds=1.0,
        correct_raw_frames=100,
        correct_sampled_frames=21,
        negative_raw_frames=100,
        negative_sampled_frames=21,
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
            cotangents[ordinal],
            torch.full((2, 3), float(ordinal + 1)),
        )


def test_full48_gather_rejects_duplicate_or_missing_task_ordinals() -> None:
    local = [_objective(index) for index in range(24)]
    local[-1] = _objective(22)
    with pytest.raises(ExpertManifoldError, match="task order changed"):
        _gather_full48(local, _context())


def _profile_config() -> dict:
    return {
        "profile_run": {
            "throughput_baseline": {"step_seconds": 20.0},
            "gates": {
                "feature_rank_min": 48,
                "regularized_gram_condition_number_max": 200.0,
                "correct_motion_to_cotangent_rms_min": 0.25,
                "negative_to_correct_motion_rms_max": 0.15,
                "predicted_observed_relative_rms_max": 0.005,
                "production_wall_ratio_max": 1.75,
                "lora_a_response_rms_min": 0.0,
                "lora_b_response_rms_min": 0.0,
                "fixed_action_response_rms_min": 0.0,
                "fixed_action_probe_task_count": 4,
                "fixed_action_passing_task_count_min": 4,
                "correct_retained_task_count_min": 21,
                "negative_null_task_count_min": 18,
                "negative_null_per_kind_min": 6,
                "extra_negative_policy_forwards": 0,
                "task_count": 24,
                "video_count": 24,
                "source_action_query_count": 480,
                "rollout_count": 96,
                "guarded_task_count_min": 18,
                "all_success_task_count_min": 6,
                "guarded_task_per_suite_min": 2,
                "success_program_cotangent_nonzero": True,
                "failure_replay_gradient_episodes": 0,
                "projection_changed_task_count_min": 1,
                "exact_blind_fallback_required": True,
                "applied_guard_evidence_required": True,
                "oom_count": 0,
                "nonfinite_count": 0,
            },
        }
    }


def _profile_row() -> dict:
    return {
        "update": {
            "feature_rank": 48,
            "regularized_gram_condition_number": 100.0,
            "correct_cotangent_rms": 2.0,
            "predicted_correct_motion_rms": 1.0,
            "predicted_negative_to_correct_ratio": 0.1,
        },
        "application": {"predicted_observed_relative_rms": 0.001},
        "lora_response": {
            "lora_a_response_rms": 0.01,
            "lora_b_response_rms": 0.02,
            "fixed_action_response_rms": 0.03,
            "fixed_action_probe_task_count": 4,
            "fixed_action_probe_policy_forwards": 8,
            "fixed_action_passing_task_count": 4,
        },
        "task_local_motion": {
            "task_count": 24,
            "correct_retained_passing_tasks": 24,
            "negative_null_passing_tasks": 24,
            "rows": [
                {
                    "task_ordinal": value,
                    "negative_to_correct_motion_rms": 0.1,
                }
                for value in range(24)
            ],
        },
        "task_records": [
            {
                "task_ordinal": value,
                "suite": (
                    "libero_spatial",
                    "libero_object",
                    "libero_goal",
                    "libero_10",
                )[value // 6],
                "counterfactual_kind": ("reversed", "shuffled", "wrong")[value % 3],
                "historical_v6_video_encodes": 1,
                "source_action_queries": 20,
                "retention_credit": {
                    "successes": 4 if value < 6 else 1,
                    "replay_chunks": 4 if value < 6 else 1,
                    "flow_panel_chunks": 4,
                    "flow_panel_row_indices": (
                        [0, 1, 2, 3] if value < 6 else [0]
                    ),
                    "program_cotangent_rms": (
                        [0.1, 0.2, 0.3, 0.4] if value < 6 else [0.1]
                    ),
                },
                "guard_projection": {
                    "changed": value == 0,
                    "raw_feasible": value != 0,
                },
            }
            for value in range(24)
        ],
        "success_guard": {
            "rollouts": 96,
            "success_episodes": 42,
            "failure_episodes": 54,
            "guarded_tasks": 24,
            "all_success_tasks": 6,
            "all_failure_tasks": 0,
            "guarded_tasks_per_suite": {
                "libero_spatial": 6,
                "libero_object": 6,
                "libero_goal": 6,
                "libero_10": 6,
            },
            "projection_changed_tasks": 1,
            "raw_feasible_tasks": 23,
            "failure_replay_gradient_episodes": 0,
            "maximum_constraint_value": 0.0,
            "minimum_source_descent_ratio": 0.5,
        },
        "success_guard_application": {
            "task_count": 24,
            "constraint_count": 42,
            "continuous_violating_constraint_count": 0,
            "native_program_violating_constraint_count": 0,
            "maximum_continuous_constraint_value": 0.0,
            "maximum_native_program_constraint_value": 0.0,
            "minimum_continuous_source_descent_ratio": 0.5,
            "minimum_native_program_source_descent_ratio": 0.5,
            "rows": [],
        },
        "profile_task_seconds": 19.0,
        "production_kernel_seconds": 1.0,
        "negative_policy_forwards": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
    }


def test_mechanism_profile_requires_every_predeclared_path_and_throughput_gate() -> (
    None
):
    passed, evidence = _profile_passes(_profile_config(), _profile_row())
    assert passed is True
    assert all(evidence["checks"].values())
    mutations = (
        ("update", "feature_rank", 47),
        ("update", "regularized_gram_condition_number", 201.0),
        ("update", "predicted_correct_motion_rms", 0.1),
        ("update", "predicted_negative_to_correct_ratio", 0.3),
        ("application", "predicted_observed_relative_rms", 0.01),
        ("lora_response", "lora_a_response_rms", 0.0),
        ("lora_response", "lora_b_response_rms", 0.0),
        ("lora_response", "fixed_action_response_rms", 0.0),
        ("lora_response", "lora_a_response_rms", float("inf")),
        ("lora_response", "lora_b_response_rms", float("inf")),
    )
    for section, key, value in mutations:
        row = _profile_row()
        row[section][key] = value
        assert _profile_passes(_profile_config(), row)[0] is False
    row = _profile_row()
    row["production_kernel_seconds"] = 16.1
    assert _profile_passes(_profile_config(), row)[0] is False
    row = _profile_row()
    row["negative_policy_forwards"] = 1
    assert _profile_passes(_profile_config(), row)[0] is False
    row = _profile_row()
    row["success_guard"]["projection_changed_tasks"] = 0
    assert _profile_passes(_profile_config(), row)[0] is False
    row = _profile_row()
    row["success_guard_application"]["constraint_count"] = 41
    assert _profile_passes(_profile_config(), row)[0] is False


def test_task_local_profile_keeps_all_24_retained_and_null_rows() -> None:
    cotangents = torch.ones((24, 2, 3), dtype=torch.float32)
    motion = torch.cat(
        (
            torch.full((24, 2, 3), 0.5),
            torch.full((24, 2, 3), 0.05),
        )
    )
    evidence = _profile_task_local_motion(
        cotangents,
        motion,
        _profile_config()["profile_run"]["gates"],
    )
    assert evidence["task_count"] == 24
    assert evidence["correct_retained_passing_tasks"] == 24
    assert evidence["negative_null_passing_tasks"] == 24
    assert [row["task_ordinal"] for row in evidence["rows"]] == list(range(24))


def test_applied_full48_motion_is_checked_against_every_success_guard() -> None:
    objectives = [
        replace(
            _objective(ordinal),
            retention_program_cotangents=(torch.ones((2, 3)),),
            program_before=torch.zeros((2, 3), dtype=torch.bfloat16),
        )
        for ordinal in range(24)
    ]
    motion = torch.full((24, 2, 3), -0.5)
    evidence = _profile_success_guard_application(objectives, motion, _context())
    assert evidence["constraint_count"] == 24
    assert evidence["continuous_violating_constraint_count"] == 0
    assert evidence["native_program_violating_constraint_count"] == 0
    assert evidence["minimum_continuous_source_descent_ratio"] > 0
    assert evidence["minimum_native_program_source_descent_ratio"] > 0
    violating = motion.clone()
    violating[0].fill_(0.5)
    failed = _profile_success_guard_application(objectives, violating, _context())
    assert failed["continuous_violating_constraint_count"] == 1
    assert failed["native_program_violating_constraint_count"] == 1


def test_task_record_reports_one_correct_b20_and_zero_negative_policy_forwards() -> (
    None
):
    row = _task_record(_objective(0))
    assert row["source_action_queries"] == 20
    assert row["physical_correct_policy_forwards"] == 2
    assert row["retention_policy_forwards"] == 4
    assert row["retention_credit"]["successes"] == 1
    assert row["negative_policy_forwards"] == 0
    assert row["historical_v6_video_encodes"] == 1
    assert row["policy_innovation_key_count"] == 2
    assert row["policy_innovation_unique_video_count"] == 1
    assert row["policy_innovation_duplicate_frame_forwards"] == 0


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
    assert (
        _reconcile_metrics_cursor(
            metrics,
            context=_context(),
            expected_rows=10,
        )
        == 10
    )
    retained = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["macro"] for row in retained] == list(range(1, 11))
    packet = tmp_path / "failure_packets/orphaned_after_step_00000010.jsonl"
    assert [
        json.loads(line)["macro"] for line in packet.read_text().splitlines()
    ] == list(range(11, 18))


def test_cli_exposes_only_residual_profile_and_formal_modes() -> None:
    parser = build_parser()
    mode = next(action for action in parser._actions if action.dest == "mode")
    assert tuple(mode.choices) == ("mechanism-profile", "formal")
    destinations = {action.dest for action in parser._actions}
    assert destinations.isdisjoint(
        {"expert_bank_root", "warm_start", "teacher_audit", "auxiliary_weight"}
    )
