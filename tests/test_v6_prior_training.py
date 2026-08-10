from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ember.expert_manifold.v6_prior_training as training_module
from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_profile import (
    profile_action_panel,
    profile_credit_motion,
    profile_passes,
)
from ember.expert_manifold.v6_prior_runtime import _reconcile_metrics_cursor
from ember.expert_manifold.v6_prior_training import (
    TaskObjective,
    _gather_full48,
    _start_event,
    _task_record,
    build_parser,
)
from ember.expert_manifold.v6_reward_credit import RewardProgramCreditSummary
from ember.pi05_source_checkpoint import DistributedContext
from ember.reward.rollout import RewardTrajectory


def _context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"))


def test_start_event_reads_the_sealed_runtime_recipe() -> None:
    runtime = SimpleNamespace(
        args=SimpleNamespace(mode="mechanism-profile"),
        segment=SimpleNamespace(start_macro=0, stop_macro=1),
        ownership=SimpleNamespace(frozen_parameter_count=123),
        writer=SimpleNamespace(
            program_memory=SimpleNamespace(value=torch.empty((2, 3, 4)))
        ),
        config={
            "optimization": {
                "rollout_policy_batch_size": 7,
                "reward_replay_chunk_batch_size": 11,
            },
            "objective": {"flow_mc_samples": 13},
        },
    )
    event = _start_event(runtime)
    assert event["rollout_policy_batch_size"] == 7
    assert event["reward_replay_chunk_batch_size"] == 11
    assert event["flow_mc_samples"] == 13
    assert "reward_replay_microbatch_size" not in event


def _trajectory(cursor: int, *, success: bool) -> RewardTrajectory:
    return RewardTrajectory(
        suite="libero_spatial",
        task_id=0,
        global_task_id=100,
        adaptation_seed=7,
        rollout_cursor=cursor,
        env_seed=1000 + cursor,
        policy_seed_root=2000,
        success=success,
        steps=5,
        reward_sum=float(success),
        dummy_settling_steps=10,
        policy_noise_seeds=(3000 + cursor,),
        observations=({"observation.state": torch.zeros(1, 7)},),
        action_chunks=(torch.zeros(1, 50, 7),),
        valid_action_steps=(5,),
    )


def _objective(ordinal: int, *, mixed: bool = True) -> TaskObjective:
    correct = torch.zeros(256, dtype=torch.float32)
    negative = torch.zeros(256, dtype=torch.float32)
    correct[ordinal] = 1.0
    negative[ordinal + 24] = 1.0
    task = SimpleNamespace(
        ordinal=ordinal,
        global_task_id=ordinal + 100,
        suite=(
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        )[ordinal // 6],
        task_id=ordinal % 6,
    )
    successes = (True, False, True, False) if mixed else (False,) * 4
    trajectories = tuple(
        _trajectory(ordinal * 4 + lane, success=value)
        for lane, value in enumerate(successes)
    )
    success_count = sum(successes)
    credit = RewardProgramCreditSummary(
        objective=0.25 if mixed else 0.0,
        successes=success_count,
        failures=4 - success_count,
        mixed=mixed,
        positive_episodes=success_count if mixed else 0,
        negative_episodes=4 - success_count if mixed else 0,
        zero_episodes=0 if mixed else 4,
        replay_chunks=4,
        executed_action_steps=20,
        mc_samples=4,
        functional_policy_forwards=8 if mixed else 0,
        program_cotangent_rms=float(ordinal + 1) if mixed else 0.0,
    )
    return TaskObjective(
        task=task,
        task_visit=0,
        teacher_demo=4,
        counterfactual_kind=("reversed", "shuffled", "wrong")[ordinal % 3],
        counterfactual_task=None,
        counterfactual_demo=None,
        correct_feature=correct,
        negative_feature=negative,
        program_cotangent=torch.full((2, 3), float(ordinal + 1) if mixed else 0.0),
        credit=credit,
        trajectory_rows=tuple(
            training_module._trajectory_record(value) for value in trajectories
        ),
        correct_raw_frames=100,
        correct_sampled_frames=21,
        negative_raw_frames=100,
        negative_sampled_frames=21,
        rollout_seconds=1.0,
        credit_seconds=2.0,
    )


def test_full48_gather_sorts_train24_and_never_rescales_program_cotangents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(context=_context())
    monkeypatch.setattr(
        training_module,
        "_credit_ready_rendezvous",
        lambda _runtime, *, macro: (None, None),
    )
    local = [_objective(index) for index in reversed(range(24))]
    correct, negative, cotangents = _gather_full48(runtime, local, macro=0)
    assert correct.shape == negative.shape == (24, 256)
    assert cotangents.shape == (24, 2, 3)
    assert torch.equal(correct[:, :24], torch.eye(24))
    assert torch.equal(negative[:, 24:48], torch.eye(24))
    for ordinal in range(24):
        assert torch.equal(
            cotangents[ordinal],
            torch.full((2, 3), float(ordinal + 1)),
        )


def test_full48_gather_rejects_duplicate_or_missing_task_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(context=_context())
    monkeypatch.setattr(
        training_module,
        "_credit_ready_rendezvous",
        lambda _runtime, *, macro: (None, None),
    )
    local = [_objective(index) for index in range(24)]
    local[-1] = _objective(22)
    with pytest.raises(ExpertManifoldError, match="task order changed"):
        _gather_full48(runtime, local, macro=0)


def _profile_config() -> dict:
    return {
        "profile_run": {
            "gates": {
                "tasks": 24,
                "rollouts": 96,
                "rollouts_per_task": 4,
                "videos": 24,
                "videos_per_task": 1,
                "mixed_tasks_min": 6,
                "homogeneous_tasks_min": 1,
                "mixed_cotangent_nonzero": True,
                "homogeneous_cotangent_exact_zero": True,
                "program_to_lora_response_nonzero": True,
                "program_to_all_mixed_fixed_action_response_nonzero": True,
                "mixed_action_probe_scope": "all_mixed_tasks",
                "mixed_suite_count": 4,
                "fixed_action_queries_per_mixed_task": 4,
                "fixed_action_policy_forwards_per_mixed_task": 2,
                "full48_feature_rank_min": 24,
                "negative_null_motion_ratio_max": 0.25,
                "predicted_observed_relative_rms_max": 0.005,
                "extra_negative_policy_forwards": 0,
                "old_policy_forwards": 0,
                "oom_count": 0,
                "nonfinite_count": 0,
                "watchdog_count": 0,
            }
        }
    }


def _profile_row() -> dict:
    mixed_ordinals = {0, 1, 2, 3, 4, 5, 8, 12, 18}
    records = []
    for ordinal in range(24):
        mixed = ordinal in mixed_ordinals
        records.append(
            {
                "task_ordinal": ordinal,
                "suite": (
                    "libero_spatial",
                    "libero_object",
                    "libero_goal",
                    "libero_10",
                )[ordinal // 6],
                "mixed": mixed,
                "rollouts": 4,
                "videos": 1,
                "program_cotangent_rms": 0.1 if mixed else 0.0,
                "functional_policy_forwards": 8 if mixed else 0,
            }
        )
    return {
        "tasks": 24,
        "rollouts": 96,
        "videos": 24,
        "successes": 12,
        "task_records": records,
        "update": {
            "feature_rank": 24,
            "predicted_negative_to_correct_ratio": 0.1,
        },
        "application": {"predicted_observed_relative_rms": 0.001},
        "lora_response": {
            "lora_a_response_rms": 0.01,
            "lora_b_response_rms": 0.02,
            "fixed_action_response_rms": 0.03,
            "fixed_action_probe_task_count": len(mixed_ordinals),
            "fixed_action_probe_query_count": 4 * len(mixed_ordinals),
            "fixed_action_probe_policy_forwards": 2 * len(mixed_ordinals),
            "fixed_action_passing_task_count": len(mixed_ordinals),
            "fixed_action_task_rows": [
                {
                    "task_ordinal": ordinal,
                    "query_count": 4,
                    "policy_forwards": 2,
                    "lora_a_value_count": 10,
                    "lora_a_response_rms": 0.01,
                    "lora_b_value_count": 10,
                    "lora_b_response_rms": 0.02,
                    "fixed_action_value_count": 1400,
                    "fixed_action_response_rms": 0.03,
                }
                for ordinal in sorted(mixed_ordinals)
            ],
        },
        "negative_policy_forwards": 0,
        "old_policy_forwards": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "watchdog_count": 0,
        "step_seconds": 10.0,
        "max_cuda_allocated_bytes": 30_000_000_000,
        "max_cuda_reserved_bytes": 40_000_000_000,
    }


def test_mechanism_profile_requires_full24_mixed_and_homogeneous_credit_paths() -> None:
    config = _profile_config()
    passed, evidence = profile_passes(config, [_profile_row()])
    assert passed is True
    assert all(evidence["checks"].values())
    mutations = (
        ("tasks", 23),
        ("rollouts", 95),
        ("videos", 23),
        ("negative_policy_forwards", 1),
        ("watchdog_count", 1),
    )
    for key, value in mutations:
        row = _profile_row()
        row[key] = value
        assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["task_records"][0]["program_cotangent_rms"] = 0.0
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["task_records"][6]["functional_policy_forwards"] = 1
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    for record in row["task_records"]:
        record["mixed"] = True
        record["program_cotangent_rms"] = 0.1
        record["functional_policy_forwards"] = 8
    assert profile_passes(config, [row])[0] is False
    for invalid in (None, 0, "false"):
        row = _profile_row()
        row["task_records"][6]["mixed"] = invalid
        with pytest.raises(ExpertManifoldError, match="partition changed"):
            profile_passes(config, [row])
    row = _profile_row()
    del row["task_records"][6]["mixed"]
    with pytest.raises(ExpertManifoldError, match="partition changed"):
        profile_passes(config, [row])
    row = _profile_row()
    row["lora_response"]["fixed_action_response_rms"] = 0.0
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["lora_response"]["fixed_action_task_rows"].pop()
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["lora_response"]["fixed_action_task_rows"][0]["task_ordinal"] = 6
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["lora_response"]["fixed_action_task_rows"][0]["query_count"] = 3
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["lora_response"]["fixed_action_task_rows"][0]["fixed_action_response_rms"] = 0.0
    assert profile_passes(config, [row])[0] is False
    row = _profile_row()
    row["task_records"][18]["suite"] = "libero_goal"
    assert profile_passes(config, [row])[0] is False


def test_profile_action_panel_retains_all_k4_initial_queries_and_original_noise() -> (
    None
):
    mixed = tuple(_trajectory(lane, success=lane == 0) for lane in range(4))
    for trajectory in mixed:
        trajectory.observations[0].update(
            {
                "observation.image": torch.zeros(1, 3, 2, 2),
                "observation.tokens": torch.zeros(1, 2, dtype=torch.long),
                "observation.mask": torch.ones(1, 2, dtype=torch.bool),
            }
        )
    panel = profile_action_panel(mixed)
    assert panel is not None
    assert panel.query["observation.state"].shape == (4, 7)
    assert panel.noise_seeds == (3000, 3001, 3002, 3003)
    homogeneous = tuple(_trajectory(lane, success=True) for lane in range(4))
    assert profile_action_panel(homogeneous) is None


def test_profile_reports_shared_motion_on_exact_zero_credit_tasks() -> None:
    cotangents = torch.zeros(4, 2, 3)
    cotangents[:2] = 1
    motion = torch.zeros_like(cotangents)
    motion[:2] = 2
    motion[2:] = 1
    evidence = profile_credit_motion(cotangents, motion)
    assert evidence == {
        "mixed_correct_motion_rms": 2.0,
        "homogeneous_correct_motion_rms": 1.0,
        "homogeneous_to_mixed_motion_ratio": 0.5,
        "homogeneous_task_count": 2,
        "homogeneous_moving_task_count": 2,
    }


def test_task_record_reports_k4_signed_replay_and_no_retired_forwards() -> None:
    row = _task_record(_objective(0))
    assert row["rollouts"] == 4
    assert row["successes"] == 2
    assert row["failures"] == 2
    assert row["replay_chunks"] == 4
    assert row["executed_action_steps"] == 20
    assert row["videos"] == 1
    assert row["teacher_action_reads"] == 0
    assert row["source_action_reads"] == 0
    assert row["old_policy_forwards"] == 0
    assert row["negative_policy_forwards"] == 0


def test_resume_reconciles_post_checkpoint_metrics_into_failure_packet(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"macro": macro, "value": macro}) + "\n" for macro in (1, 2)
        ),
        encoding="utf-8",
    )
    assert (
        _reconcile_metrics_cursor(
            metrics,
            context=_context(),
            expected_rows=1,
        )
        == 1
    )
    retained = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [row["macro"] for row in retained] == [1]
    packet = tmp_path / "failure_packets/orphaned_after_step_00000001.jsonl"
    assert [json.loads(line)["macro"] for line in packet.read_text().splitlines()] == [
        2
    ]


def test_cli_exposes_only_reward_profile_and_formal_modes() -> None:
    parser = build_parser()
    mode = next(action for action in parser._actions if action.dest == "mode")
    workers = next(action for action in parser._actions if action.dest == "num_workers")
    assert tuple(mode.choices) == ("mechanism-profile", "formal")
    assert workers.default == 0
    destinations = {action.dest for action in parser._actions}
    assert destinations.isdisjoint(
        {
            "expert_bank_root",
            "warm_start",
            "teacher_audit",
            "auxiliary_weight",
            "old_policy",
            "learning_epochs",
        }
    )
