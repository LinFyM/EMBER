from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext
from ember.writer.as_config import load_writer_config, resolve_mode_config, writer_split_roles
from ember.writer.as_contract import resolve_runtime
from ember.writer.model import WriterModelError
from ember.writer.training import prepare_runtime
from ember.writer.update_contract import build_update_runtime_contract, checkpoint_state_family


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_as_writer_k4_layer_trace_m2p_bci_v1.json"


def _context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=6,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )


def test_k4_layer_trace_config_seals_video_owned_joint_generation() -> None:
    config = load_writer_config(CONFIG)
    assert writer_split_roles(config) == ("train",)
    assert config["writer"]["architecture"] == (
        "pi05_k4_policy_layer_trace_axis_m2p_v1"
    )
    assert config["writer"]["videos_per_condition"] == 4
    assert config["writer"]["language_value_bypass"] is False
    assert config["writer"]["policy_groups"] == 20
    assert config["writer"]["trace_tokens_per_group_per_condition"] == 64
    assert config["writer"]["memory_slots"] == 68
    assert config["conditioning_training"]["global_tasks_per_optimizer_update"] == 24
    assert config["formal_run"]["checkpoint_steps"] == [25, 50, 75, 100, 125, 150, 175, 200]


def test_profile_uses_independent_video_seed_without_mutating_formal_config() -> None:
    config = load_writer_config(CONFIG)
    profile = resolve_mode_config(config, "profile")
    assert profile["data"]["teacher_video_seed"] == 173
    assert config["data"]["teacher_video_seed"] == 20260722
    args = argparse.Namespace(
        mode="profile",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
    )
    assert resolve_runtime(args, profile, _context()) == (200, 20, (1, 2, 3, 200))
    assert args.stop_after_step == 3


def test_formal_stays_blocked_until_live_layer_trace_profile_and_resume() -> None:
    config = load_writer_config(CONFIG)
    assert config["formal_run"]["status"] == "blocked_on_live_profile"
    assert config["formal_run"]["launch_state"] == "not_ready_before_layer_trace_profile"
    evidence = config["profile_evidence"]
    assert evidence["status"] == "not_run_for_layer_trace_architecture"
    assert evidence["profile_weights_reusable_for_formal"] is False


def test_k4_m2p_rejects_writer_warm_start_before_runtime_construction() -> None:
    args = argparse.Namespace(
        config=CONFIG,
        mode="profile",
        initialize_writer_checkpoint=ROOT / "historical_writer",
    )
    with pytest.raises(WriterModelError, match="warm-start is forbidden"):
        prepare_runtime(args, _context())


def test_runtime_contract_owns_one_joint_k4_full24_update() -> None:
    config = load_writer_config(CONFIG)
    runtime = build_update_runtime_contract(
        config=config,
        context=_context(),
        video_data={"sampled_frame_counts_by_task": {str(i): {} for i in range(24)}},
        total_steps=200,
        stop_step=200,
        batch_size=20,
        batch_cycle=(20,),
        checkpoint_steps=(25, 50, 75, 100, 125, 150, 175, 200),
        num_workers=0,
        rank_topology=tuple(
            {
                "rank": rank,
                "local_rank": rank,
                "device": f"cuda:{rank}",
                "numa_node": 0 if rank < 3 else 1,
                "cpu_affinity": [rank],
            }
            for rank in range(6)
        ),
    )
    assert runtime["checkpoint_state_family"] == (
        "k4_policy_layer_trace_m2p_full24_v1"
    )
    assert runtime["macro_step_axis"] == "full24_end_to_end_k4_layer_trace_m2p_update"
    assert runtime["condition_gradient_unit"] == "one_joint_k4_video_set_to_one_lora_per_task"
    assert runtime["teacher_videos_per_task_visit"] == 4
    assert checkpoint_state_family(config) == runtime["checkpoint_state_family"]
