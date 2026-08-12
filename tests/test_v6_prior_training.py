from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import ember.expert_manifold.v6_prior_training as prior_training
from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_runtime import _reconcile_metrics_cursor
from ember.expert_manifold.v6_prior_training import (
    TaskObjective,
    ViewObjective,
    _AtomicTaskClaimQueue,
    _gather_paired_video_rows,
    _profile_passes,
    _profile_task_local_motion,
    _retained_task_cap,
    _task_record,
    build_parser,
)
from ember.pi05_source_checkpoint import DistributedContext


_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _context(world_size: int = 1) -> DistributedContext:
    return DistributedContext(0, 0, world_size, torch.device("cpu"))


def _view(ordinal: int, *, companion: bool) -> ViewObjective:
    correct = torch.zeros(256, dtype=torch.float32)
    negative = torch.zeros(256, dtype=torch.float32)
    offset = 48 if companion else 0
    correct[ordinal + offset] = 1
    negative[ordinal + offset + 96] = 1
    gradient = torch.full(
        (320, 256), float(ordinal + 1 + int(companion)), dtype=torch.float32
    )
    return ViewObjective(
        demo=5 if companion else 4,
        counterfactual_demo=(9 if companion else 8) if ordinal % 3 == 2 else None,
        functional_loss=torch.tensor(float(ordinal + int(companion))),
        correct_feature=correct,
        negative_feature=negative,
        program_cotangent=gradient,
        correct_raw_frames=100,
        correct_sampled_frames=21,
        negative_raw_frames=90,
        negative_sampled_frames=19,
    )


def _objective(ordinal: int) -> TaskObjective:
    return TaskObjective(
        task=SimpleNamespace(
            ordinal=ordinal,
            global_task_id=100 + ordinal,
            suite=_SUITES[ordinal // 6],
            task_id=ordinal % 6,
        ),
        task_visit=3,
        action_query_demos=(0, 1),
        counterfactual_kind=("reversed", "shuffled", "wrong")[ordinal % 3],
        counterfactual_task=(
            SimpleNamespace(global_task_id=200 + ordinal)
            if ordinal % 3 == 2
            else None
        ),
        primary=_view(ordinal, companion=False),
        companion=_view(ordinal, companion=True),
        phase_a_queue_index=ordinal,
        phase_a_rank=0,
        phase_a_finished_seconds=1,
    )


def test_atomic_queue_supports_any_world_size_up_to_six_without_waiting() -> None:
    assert [_retained_task_cap(world) for world in range(1, 7)] == [24, 12, 8, 8, 8, 8]


def test_atomic_work_queue_claims_each_job_once(tmp_path) -> None:
    jobs = tuple((100 + index, 3) for index in range(24))
    cursor = tmp_path / "cursor"
    cursor.write_text("0", encoding="ascii")
    queues = (_AtomicTaskClaimQueue(cursor, jobs), _AtomicTaskClaimQueue(cursor, jobs))
    claimed = [queues[index % 2].claim()[:2] for index in range(24)]
    assert claimed == list(enumerate(jobs))
    assert queues[0].claim()[1] is None


def test_full96_gather_orders_primary_then_companion_without_rescaling() -> None:
    local = [_objective(index) for index in reversed(range(24))]
    correct, negative, cotangents, timing = _gather_paired_video_rows(
        local, _context()
    )
    assert correct.shape == negative.shape == (48, 256)
    assert cotangents.shape == (48, 320, 256)
    assert torch.equal(correct[:24, :24], torch.eye(24))
    assert torch.equal(correct[24:, 48:72], torch.eye(24))
    assert torch.equal(negative[:24, 96:120], torch.eye(24))
    assert torch.equal(negative[24:, 144:168], torch.eye(24))
    assert [row["task_ordinal"] for row in timing] == list(range(24))
    for ordinal in range(24):
        assert torch.equal(
            cotangents[ordinal],
            torch.full((320, 256), float(ordinal + 1)),
        )
        assert torch.equal(
            cotangents[24 + ordinal],
            torch.full((320, 256), float(ordinal + 2)),
        )


def test_full96_gather_rejects_duplicate_task_ordinals() -> None:
    local = [_objective(index) for index in range(24)]
    local[-1] = _objective(22)
    with pytest.raises(ExpertManifoldError, match="task order changed"):
        _gather_paired_video_rows(local, _context())


def test_full96_padded_gather_keeps_all_four_features_and_two_cotangents_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = [7, 0, 23, -1, 2, 11, 19, 3, 5, 13, 1, 22, 8, 15, 4, 18, 6, 14, 9, 20, 10, 16, 12, 21, 17]
    payload = torch.zeros(25, 1031, dtype=torch.float32)
    payload[:, 0].fill_(-1)
    cotangents = torch.zeros(25, 2, 320, 256, dtype=torch.float32)
    for row, ordinal in enumerate(order):
        if ordinal < 0:
            continue
        objective = _objective(ordinal)
        payload[row, 0] = ordinal
        payload[row, 1] = row % 5
        payload[row, 2] = row
        payload[row, 4] = 1
        payload[row, 7:263] = objective.primary.correct_feature
        payload[row, 263:519] = objective.primary.negative_feature
        payload[row, 519:775] = objective.companion.correct_feature
        payload[row, 775:] = objective.companion.negative_feature
        cotangents[row, 0] = objective.primary.program_cotangent
        cotangents[row, 1] = objective.companion.program_cotangent
    gathered = iter((payload, cotangents))
    monkeypatch.setattr(
        prior_training,
        "_all_gather_fixed",
        lambda value, context: next(gathered),
    )
    correct, negative, gradients, timing = _gather_paired_video_rows(
        [_objective(index) for index in range(5)], _context(5)
    )
    assert torch.equal(correct[:24, :24], torch.eye(24))
    assert torch.equal(correct[24:, 48:72], torch.eye(24))
    assert torch.equal(negative[:24, 96:120], torch.eye(24))
    assert torch.equal(negative[24:, 144:168], torch.eye(24))
    assert [row["task_ordinal"] for row in timing] == list(range(24))
    assert gradients.shape == (48, 320, 256)


def _profile_config() -> dict:
    return {
        "profile_run": {
            "gates": {
                "positive_feature_rank_min": 24,
                "full_feature_rank_min": 48,
                "regularized_condition_max": 200,
                "both_view_descent_task_count_min": 12,
                "negative_to_correct_motion_rms_max": 0.15,
                "negative_null_per_kind_min": 12,
                "predicted_observed_relative_rms_max": 0.005,
                "retained_task_cap_max": 24,
                "queue_claim_seconds_max": 1,
                "step_seconds_max": 292.4,
                "oom_count": 0,
                "nonfinite_count": 0,
            }
        }
    }


def _profile_row() -> dict:
    records = [_task_record(_objective(ordinal)) for ordinal in range(24)]
    return {
        "update": {
            "positive_feature_rank": 48,
            "original_feature_rank": 96,
            "regularized_gram_condition_number": 80,
            "value_delta_rms": 0.1,
            "primary_motion_rms": 1.0,
            "companion_motion_rms": 1.0,
        },
        "application": {"predicted_observed_relative_rms": 0.001},
        "task_local_motion": {
            "total_directional_derivative": -3,
            "primary_directional_derivative": -2,
            "companion_directional_derivative": -1,
            "suite_joint_directional_derivatives": {
                suite: -0.5 for suite in _SUITES
            },
            "both_view_descent_task_count": 18,
            "negative_to_correct_motion_ratio": 0.1,
            "negative_null_per_kind": {
                "reversed": 16,
                "shuffled": 16,
                "wrong": 16,
            },
        },
        "lora_response": {
            "probe_rows": 8,
            "policy_forwards": 16,
            "all_program_motion_nonzero": True,
            "all_lora_a_response_nonzero": True,
            "all_lora_b_response_nonzero": True,
            "all_effective_ba_response_nonzero": True,
            "all_fixed_action_response_nonzero": True,
        },
        "task_records": records,
        "correct_condition_rows": 48,
        "negative_condition_rows": 48,
        "logical_source_action_queries": 960,
        "outcome_rollouts": 0,
        "phase_a_task_rows": [
            {
                "task_ordinal": ordinal,
                "rank": ordinal % 6,
                "queue_index": ordinal,
                "started_seconds": ordinal,
                "finished_seconds": ordinal + 1,
            }
            for ordinal in range(24)
        ],
        "queue_claim_seconds": 0.1,
        "phase_a_seconds": 100,
        "step_seconds": 200,
        "world_size": 6,
        "task_counts_per_rank": [4] * 6,
        "functional_loss": 1,
        "program_cotangent_rms": 2,
        "oom_count": 0,
        "nonfinite_count": 0,
    }


def test_profile_requires_continuous_joint_credit_without_outcomes() -> None:
    passed, evidence = _profile_passes(_profile_config(), _profile_row())
    assert passed
    assert all(evidence["checks"].values())
    for section, key, value in (
        ("update", "positive_feature_rank", 23),
        ("update", "regularized_gram_condition_number", 201),
        ("task_local_motion", "primary_directional_derivative", 0),
        ("task_local_motion", "both_view_descent_task_count", 11),
        ("task_local_motion", "negative_to_correct_motion_ratio", 0.2),
        ("application", "predicted_observed_relative_rms", 0.01),
        ("lora_response", "all_effective_ba_response_nonzero", False),
    ):
        row = _profile_row()
        row[section][key] = value
        assert not _profile_passes(_profile_config(), row)[0]
    row = _profile_row()
    row["outcome_rollouts"] = 1
    assert not _profile_passes(_profile_config(), row)[0]


def test_task_local_profile_requires_both_views_and_zero_rhs_negatives() -> None:
    cotangents = torch.ones((48, 2, 3), dtype=torch.float32)
    correct = torch.full((48, 2, 3), -0.5)
    negative = torch.full((48, 2, 3), 0.025)
    evidence = _profile_task_local_motion(
        cotangents,
        torch.cat((correct, negative)),
        0,
        _profile_config()["profile_run"]["gates"],
    )
    assert evidence["both_view_descent_task_count"] == 24
    assert evidence["negative_null_passing_views"] == 48
    assert evidence["negative_null_per_kind"] == {
        "reversed": 16,
        "shuffled": 16,
        "wrong": 16,
    }


def test_task_record_counts_two_complete_views_same_twenty_queries_and_no_reward() -> None:
    row = _task_record(_objective(2))
    assert row["distinct_ordered_correct_videos"] == 2
    assert row["distinct_wrong_videos"] == 2
    assert row["unique_source_action_queries"] == 20
    assert row["logical_source_action_queries"] == 40
    assert row["physical_correct_policy_forwards"] == 4
    assert row["outcome_rollouts"] == row["reward_reads"] == 0
    assert row["teacher_action_reads"] == 0


def test_resume_reconciles_post_checkpoint_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps({"macro": macro}) + "\n" for macro in range(1, 8)),
        encoding="utf-8",
    )
    assert _reconcile_metrics_cursor(metrics, context=_context(), expected_rows=5) == 5
    assert len(metrics.read_text().splitlines()) == 5


def test_cli_exposes_only_profile_and_formal_modes() -> None:
    parser = build_parser()
    mode = next(action for action in parser._actions if action.dest == "mode")
    assert tuple(mode.choices) == ("mechanism-profile", "formal")
