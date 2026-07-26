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
from ember.writer.data import TeacherVideoSchedule
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_as_writer_core_causal_v5.json"


def test_core_causal_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert writer["architecture"] == "pi05_semantic_core_causal_procedure_v5"
    assert writer["teacher_state_input"] is False
    assert writer["teacher_prompt"] == "Task: {cleaned_task};\nAction: "
    assert writer["core_tokens_per_frame"] == 64
    assert writer["core_order_contract"].startswith("flatten_all_frame")
    assert (
        writer["frame_microbatch_remainder"]
        == "repeat_last_pad_to_fixed_size_then_crop"
    )
    assert writer["vl_meta_lora_rank"] == 4
    assert writer["action_meta_lora_rank"] == 8
    assert writer["action_horizon"] == 50
    assert writer["query_count"] == 320
    assert writer["frame_stride"] == 5
    assert writer["frame_microbatch_size"] == 32
    assert writer["procedure_attention"] == "global_causal_pre_norm_with_valid_mask"
    assert writer["procedure_blocks"] == 2
    assert writer["core_compiler_blocks"] == 1
    assert writer["procedure_refiner_blocks"] == 1
    assert writer["procedure_refiner"].startswith("independent_content_only_delta")
    assert writer["factor_hidden_width"] == 420
    assert writer_split_roles(config) == ("train",)
    conditioning = config["conditioning_training"]
    assert conditioning["teacher_videos_per_action"] == 4
    assert conditioning["logical_pair_batch"] == "per_rank_action_batch_times_four"
    assert conditioning["pair_loss_reduction"] == "mean_over_all_action_video_pairs"
    assert conditioning["policy_noise_contract"].startswith("same action target")
    assert "optimizer_gradient_accumulation" not in str(config)
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert "state" in config["information_wall"]["writer_forbidden_inputs"]
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["profile_evidence"]["status"] == "sealed"
    assert config["profile_evidence"]["allowed_physical_gpu_ids"] == [4, 5, 6, 7]
    assert config["profile_evidence"]["selected"] == {
        "world_size": 4,
        "frame_microbatch_size": 32,
        "action_query_batch_size_per_rank": 8,
    }
    assert config["profile_evidence"]["exact_resume"] == (
        "step_2_to_step_12_passed"
    )
    assert config["specificity_gate"]["status"] == "pending"
    assert config["formal_run"]["total_steps"] == 12000
    assert config["formal_run"]["selected_stop_step"] == 60


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
        12,
        8,
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
    sealed = copy.deepcopy(config)
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "branch": "main",
            "commit": "a" * 40,
            "origin_main": "a" * 40,
            "dirty_paths": [],
        },
    )
    total_steps, batch_size, checkpoints = resolve_runtime(
        formal, sealed, context
    )
    assert (total_steps, batch_size) == (12000, 8)
    assert checkpoints[0] == 10 and checkpoints[-1] == 12000
    assert formal.stop_after_step == 60


def test_multi_video_schedule_is_distinct_reproducible_and_action_specific() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    )
    first = schedule.demos_for_action(3, 7, 0, 4)
    replay = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    ).demos_for_action(3, 7, 0, 4)
    later = schedule.demos_for_action(3, 7, 1, 4)
    assert len(first) == len(set(first)) == 4
    assert first == replay
    assert first != later
    with pytest.raises(WriterModelError, match="outside the schedule"):
        schedule.demos_for_action(3, 7, 0, 51)


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
