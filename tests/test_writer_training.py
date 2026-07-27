from __future__ import annotations

import argparse
import copy
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
from ember.writer.as_step import _balanced_action_video_map
from ember.writer.data import TeacherVideoSchedule
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_as_writer_core_causal_v5.json"


def test_action_video_assignment_is_one_to_one_and_balanced() -> None:
    assignment = _balanced_action_video_map(
        16,
        4,
        device=torch.device("cpu"),
    )
    assert assignment.shape == (16, 1)
    assert assignment[:, 0].tolist() == [0, 1, 2, 3] * 4
    assert torch.bincount(assignment[:, 0], minlength=4).tolist() == [4] * 4
    with pytest.raises(WriterModelError, match="divide evenly"):
        _balanced_action_video_map(10, 4, device=torch.device("cpu"))


def test_core_causal_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert writer["architecture"] == "pi05_semantic_core_causal_procedure_v5"
    assert writer["teacher_state_input"] is False
    assert writer["teacher_prompt"] == "Task: {cleaned_task};\nAction: "
    assert writer["core_tokens_per_frame"] == 64
    assert writer["core_order_contract"].startswith("flatten_all_frame")
    assert writer["frame_batching_contract"].startswith("pack_all_four_videos")
    assert writer["vl_meta_lora_rank"] == 4
    assert writer["action_meta_lora_rank"] == 8
    assert writer["action_horizon"] == 50
    assert writer["query_count"] == 320
    assert writer["frame_stride"] == 5
    assert writer["max_frames_per_encoder_call"] == 32
    assert writer["procedure_attention"] == "global_causal_pre_norm_with_valid_mask"
    assert writer["procedure_blocks"] == 2
    assert writer["core_compiler_blocks"] == 1
    assert writer["procedure_refiner_blocks"] == 1
    assert writer["procedure_refiner"].startswith("independent_content_only_delta")
    assert writer["factor_hidden_width"] == 420
    assert writer_split_roles(config) == ("train",)
    conditioning = config["conditioning_training"]
    assert conditioning["teacher_videos_per_task_visit"] == 4
    assert (
        conditioning["logical_pair_batch"]
        == "per_rank_action_batch"
    )
    assert conditioning["action_video_assignment"] == "round_robin_one_action_one_video"
    assert conditioning["pair_loss_reduction"] == "mean_over_all_action_video_pairs"
    assert conditioning["policy_noise_contract"].startswith("one independent")
    assert "optimizer_gradient_accumulation" not in str(config)
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert "state" in config["information_wall"]["writer_forbidden_inputs"]
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["profile_evidence"]["status"] == "sealed"
    assert config["profile_evidence"]["allowed_physical_gpu_ids"] == [4, 5, 6, 7]
    assert config["profile_evidence"]["max_frames_per_encoder_call"] == 32
    assert config["profile_evidence"]["selected"]["per_rank_action_batch_size"] == 16
    assert config["profile_evidence"]["selected"]["policy_forward_calls_per_step"] == 1
    assert config["specificity_gate"]["status"] == "pending"
    assert config["formal_run"]["total_steps"] == 12000
    assert config["formal_run"]["per_rank_batch_size"] == 16
    assert config["formal_run"]["checkpoint_steps"] == "every:50"
    assert config["formal_run"]["selected_stop_step"] == 400


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


def test_profile_and_formal_runtime_require_four_symmetric_ranks() -> None:
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
        12,
        16,
        (1, 2, 4, 8, 12),
    )
    assert profile.stop_after_step == 12
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
    formal = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    pending = copy.deepcopy(config)
    pending["formal_run"]["status"] = "pending_profile"
    with pytest.raises(WriterModelError, match="not sealed"):
        resolve_runtime(formal, pending, context)


def test_shared_multi_video_schedule_is_distinct_reproducible_and_visit_specific() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    )
    first = schedule.demos_for_task_visit(3, 7, 4)
    replay = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    ).demos_for_task_visit(3, 7, 4)
    later = schedule.demos_for_task_visit(3, 8, 4)
    assert len(first) == len(set(first)) == 4
    assert first == replay
    assert first != later
    with pytest.raises(WriterModelError, match="outside the schedule"):
        schedule.demos_for_task_visit(3, 7, 51)


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
