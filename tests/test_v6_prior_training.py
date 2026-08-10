from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
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
            "diagnostic_macros": 3,
            "throughput_baseline": {"step_seconds": 20.0},
            "gates": {
                "feature_rank_min": 24,
                "correct_motion_to_cotangent_rms_min": 0.25,
                "negative_to_correct_motion_rms_max": 0.25,
                "predicted_observed_relative_rms_max": 0.005,
                "production_wall_ratio_max": 1.1,
                "fixed_action_response_rms_min": 0.0,
                "fixed_action_probe_task_count": 4,
                "fixed_action_passing_task_count_min": 4,
                "correct_retained_task_count_min": 18,
                "negative_null_task_count_min": 18,
                "oom_count": 0,
                "nonfinite_count": 0,
                "first_step_blind_ratio_abs_tolerance": 1e-5,
                "old_panel_drift_rms_vs_blind_max": 0.5,
                "old_correct_rows_improved_fraction_min": 0.75,
                "current_correct_motion_vs_blind_min": 0.5,
            }
        }
    }


def _profile_row(index: int) -> dict:
    return {
        "update": {
            "feature_rank": 24,
            "correct_cotangent_rms": 2.0,
            "predicted_correct_motion_rms": 1.0,
            "predicted_negative_to_correct_ratio": 0.1,
            "current_motion_to_blind_ratio": 1.0 if index == 0 else 0.75,
            "reference_to_blind_ratio": 0.0 if index == 0 else 0.4,
            "reference_rows_improved_fraction": 1.0 if index == 0 else 0.8,
            "assimilated_rows_before": index * 48,
            "assimilated_rows_after": (index + 1) * 48,
            "reference_correct_rows": index * 24,
        },
        "application": {"predicted_observed_relative_rms": 0.001},
        "lora_response": None if index < 2 else {
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
            "rows": [{"task_ordinal": value} for value in range(24)],
        },
        "profile_task_seconds": 19.0,
        "production_kernel_seconds": 1.0,
        "negative_policy_forwards": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
    }


def _profile_rows() -> list[dict]:
    return [_profile_row(index) for index in range(3)]


def test_mechanism_profile_requires_every_predeclared_path_and_throughput_gate() -> None:
    passed, evidence = _profile_passes(_profile_config(), _profile_rows())
    assert passed is True
    assert all(evidence["checks"].values())
    mutations = (
        ("update", "feature_rank", 23),
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
        rows = _profile_rows()
        rows[-1][section][key] = value
        assert _profile_passes(_profile_config(), rows)[0] is False
    rows = _profile_rows()
    rows[-1]["production_kernel_seconds"] = 3.1
    assert _profile_passes(_profile_config(), rows)[0] is False
    rows = _profile_rows()
    rows[-1]["negative_policy_forwards"] = 1
    assert _profile_passes(_profile_config(), rows)[0] is False
    for key, value in (
        ("reference_to_blind_ratio", 0.51),
        ("reference_rows_improved_fraction", 0.74),
        ("current_motion_to_blind_ratio", 0.49),
    ):
        rows = _profile_rows()
        rows[1]["update"][key] = value
        assert _profile_passes(_profile_config(), rows)[0] is False


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


def test_task_record_reports_one_correct_b20_and_zero_negative_policy_forwards() -> None:
    row = _task_record(_objective(0))
    assert row["source_action_queries"] == 20
    assert row["physical_correct_policy_forwards"] == 2
    assert row["negative_policy_forwards"] == 0
    assert row["writer_video_encodes"] == 1


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
        metrics,
        context=_context(),
        expected_rows=10,
    ) == 10
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
