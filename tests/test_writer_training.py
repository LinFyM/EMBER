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
CONFIG = REPO_ROOT / "configs/pi05_as_writer_language_axial_v5_2.json"


def test_language_axial_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert (
        writer["architecture"]
        == "pi05_language_axial_patch_grounded_core_causal_procedure_slot_fusion_v5_2"
    )
    assert writer["teacher_state_input"] is False
    assert writer["teacher_prompt"] == "Task: {cleaned_task};\nAction: "
    assert writer["text_branch_input"].startswith("bos_plus_exact")
    assert "task_queried_image_position_content" in writer["multimodal_core_value"]
    assert writer["patch_grounding_heads"] == 8
    assert "no_value_projection" in writer["patch_grounding_value"]
    assert writer["frame_batching_contract"].startswith("encode_one_video")
    assert writer["text_meta_lora_rank"] == 4
    assert writer["vl_meta_lora_rank"] == 4
    assert writer["action_meta_lora_rank"] == 4
    assert writer["action_horizon"] == 50
    assert writer["query_count"] == 320
    assert writer["frame_stride"] == 5
    assert writer["max_frames_per_encoder_call"] == 32
    assert writer["frame_attention_initial_lambda"] == 0.05
    assert writer["semantic_core_blocks"] == 2
    assert writer["procedure_attention"] == "global_causal_pre_norm_with_valid_mask"
    assert writer["procedure_blocks"] == 2
    assert writer["slot_fusion"].startswith("zero_initialized")
    assert writer["post_fusion_blocks"] == 1
    assert writer["factor_hidden_width"] == 216
    assert writer_split_roles(config) == ("train",)
    conditioning = config["conditioning_training"]
    assert conditioning["teacher_videos_per_task_visit"] == 1
    assert (
        conditioning["logical_pair_batch"]
        == "per_rank_action_batch"
    )
    assert conditioning["action_video_assignment"] == "all_actions_share_single_video_lora"
    assert conditioning["pair_loss_reduction"] == "mean_over_rank_local_action_batch"
    assert conditioning["policy_noise_contract"].startswith("one independent")
    assert "optimizer_gradient_accumulation" not in str(config)
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert "state" in config["information_wall"]["writer_forbidden_inputs"]
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["profile_evidence"]["status"] == "completed_v5_2_live_profile"
    assert config["profile_evidence"]["allowed_physical_gpu_ids"] == [4, 5, 6, 7]
    assert config["profile_evidence"]["initial_candidate_from_v5_1"] == {
        "max_frames_per_encoder_call": 32,
        "per_rank_action_batch_size": 20,
    }
    assert config["profile_evidence"]["selected"][
        "per_rank_action_batch_size"
    ] == 21
    assert config["profile_evidence"]["upper_bound"][
        "per_rank_action_batch_size"
    ] == 22
    assert "cuda_oom" in config["profile_evidence"]["upper_bound"]["status"]
    assert config["profile_evidence"]["inference_profile"][
        "writer_generators_per_gpu"
    ] == 6
    assert config["profile_evidence"]["teacher_videos_per_task_visit"] == 1
    assert config["specificity_gate"]["status"] == "pending"
    assert config["formal_run"]["status"] == "sealed"
    assert config["formal_run"]["total_steps"] == 12000
    assert config["formal_run"]["per_rank_batch_size"] == 21
    assert config["formal_run"]["selected_stop_step"] == 900
    assert config["formal_run"]["stage_stop_steps"] == "every:100"
    assert "no_automatic_continuation" in config["formal_run"][
        "segment_definition"
    ]


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
        3,
        21,
        (1, 2, 3),
    )
    assert profile.stop_after_step == 3
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


def test_single_video_schedule_is_reproducible_and_cycle_complete() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    )
    first = schedule.demo_for_task_visit(3, 7)
    replay = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    ).demo_for_task_visit(3, 7)
    cycle = [schedule.demo_for_task_visit(3, visit) for visit in range(50)]
    assert first == replay
    assert len(set(cycle)) == 50
    with pytest.raises(WriterModelError, match="outside the schedule"):
        schedule.demo_for_task_visit(99, 7)


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
