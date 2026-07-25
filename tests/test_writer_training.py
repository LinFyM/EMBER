from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.writer.as_contract import (
    load_writer_config,
    parse_checkpoint_steps,
    reconcile_resume_contract,
    resolve_runtime,
    resume_step,
    writer_split_roles,
)
from ember.writer.data import WriterFlowNoiseSchedule
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_as_writer_action_forecast_v4.json"


def test_action_forecast_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert writer["architecture"] == "pi05_action_forecast_anchored_visual_state_v4"
    assert writer["state_slots"] == 32
    assert writer["state_coordinates"] == 8
    assert writer["state_token_generation"].startswith("native_32_token_anchor")
    assert writer["state_token_trainability"].startswith("trainable_initial")
    assert (
        writer["frame_microbatch_remainder"]
        == "repeat_last_pad_to_fixed_size_then_crop"
    )
    assert writer["maximum_revision_count"] == 10
    assert writer["vl_meta_lora_rank"] == 4
    assert writer["action_meta_lora_rank"] == 8
    assert writer["num_flow_steps"] == 10
    assert writer["action_horizon"] == 50
    assert writer["query_count"] == 320
    assert writer["frame_stride"] == 5
    assert writer["frame_microbatch_size"] == 32
    assert writer["belief_alignment"].startswith("one_token_per_absolute")
    assert writer["revision_reference"].startswith("all_earlier_covering")
    assert writer["revision_strength_path"].startswith("stop_gradient_raw")
    assert writer["revision_direction_path"].endswith(
        "signed_mean_and_per_dimension_rms"
    )
    assert writer["belief_layout"].startswith("plan_first_128_revision_second_128")
    assert writer["temporal_value_path"].endswith("without_time_mean_removal")
    assert writer["query_block_order"] == "cross_attention_before_self_attention"
    assert writer_split_roles(config) == ("train",)
    assert config["conditioning_training"] == {
        "method": "normal_positive_functional_action_loss_only",
        "writer_language_contract": (
            "correct_task_language_native_state_action_layout"
        ),
        "policy_language_contract": "correct_action_query_task_language",
        "action_query_batch_owner": (
            "one physical batch per rank with no second policy microbatch"
        ),
        "independent_conditions_per_optimizer_step": 1,
        "normal_loss_weight": 1.0,
    }
    assert "functional_policy_microbatch_size" not in str(config)
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["profile_evidence"]["status"] == "sealed"
    assert config["profile_evidence"]["selected"] == {
        "world_size": 4,
        "frame_microbatch_size": 32,
        "action_query_batch_size_per_rank": 20,
    }
    assert config["specificity_gate"]["fresh_optimizer_steps"] == 75
    assert config["formal_run"]["total_steps"] == 1200


def test_checkpoint_schedule_and_cursor_are_fail_closed() -> None:
    assert parse_checkpoint_steps("2,4,4", 4) == (2, 4)
    assert parse_checkpoint_steps("every:2", 6) == (2, 4, 6)
    assert resume_step(Path("/tmp/step_00000004")) == 4
    with pytest.raises(WriterModelError, match="must end at total_steps"):
        parse_checkpoint_steps("2,3", 4)
    with pytest.raises(WriterModelError, match="not a step checkpoint"):
        resume_step(Path("/tmp/trainer_state.pt"))
    with pytest.raises(WriterModelError, match="must divide"):
        parse_checkpoint_steps("every:4", 6)


def test_profile_and_formal_runtime_require_four_symmetric_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_writer_config(CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    profile = argparse.Namespace(
        mode="profile",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    assert resolve_runtime(profile, config, context) == (
        75,
        20,
        (50, 75),
    )
    wrong_world = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    with pytest.raises(WriterModelError, match="exactly 4"):
        resolve_runtime(profile, config, wrong_world)
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "branch": "main",
            "commit": "a" * 40,
            "origin_main": "a" * 40,
            "dirty_paths": [],
        },
    )
    formal = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    with pytest.raises(WriterModelError, match="not sealed"):
        resolve_runtime(formal, config, context)


def test_flow_noise_is_shared_within_visit_reproducible_and_visit_specific() -> None:
    schedule = WriterFlowNoiseSchedule(seed=29)
    first = schedule.noise_for_visit(7, device="cpu")
    replay = WriterFlowNoiseSchedule(seed=29).noise_for_visit(7, device="cpu")
    later = schedule.noise_for_visit(8, device="cpu")
    assert first.shape == (50, 32)
    assert torch.equal(first, replay)
    assert not torch.equal(first, later)
    identity = schedule.identity_for_visits(7, 9)
    assert identity["start_global_visit"] == 7
    assert identity["stop_global_visit"] == 9
    assert len(identity["identity_sha256"]) == 64


def test_code_compatible_resume_allows_only_recorded_commit_change(
    tmp_path: Path,
) -> None:
    existing = {
        "schema_version": "contract",
        "git": {"branch": "main", "commit": "old"},
        "runtime": {"selected_stop_step": 500, "total_steps": 1200},
    }
    write_json_atomic(tmp_path / "run_contract.json", existing)
    args = argparse.Namespace(
        output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000500",
        allow_contract_compatible_code_resume=True,
    )
    candidate = {**existing, "git": {"branch": "main", "commit": "new"}}
    assert reconcile_resume_contract(args, candidate) == existing
    changed = {
        **candidate,
        "runtime": {"selected_stop_step": 500, "total_steps": 1300},
    }
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, changed)


def test_retired_writer_configs_are_not_active() -> None:
    for name in (
        "writer_cold_start_v1.json",
        "pi05_as_writer_v1.json",
        "pi05_as_writer_v2.json",
        "pi05_as_writer_v3_normal_only.json",
    ):
        with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
            load_writer_config(REPO_ROOT / "configs" / name)
