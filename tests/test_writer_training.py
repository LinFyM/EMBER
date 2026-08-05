from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext
from ember.writer.as_config import (
    load_writer_config,
    resolve_mode_config,
    writer_split_roles,
)
from ember.writer.as_contract import resolve_runtime
from ember.writer.condition_kernel import load_condition_authority
from ember.writer.model import WriterModelError
from ember.writer.update_contract import (
    build_update_runtime_contract,
    checkpoint_state_family,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_as_writer_condition_kernel_memory_bci_v1.json"


def _context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=6,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )


def test_condition_kernel_config_seals_fresh_identity_and_full24_update() -> None:
    config = load_writer_config(CONFIG)
    assert writer_split_roles(config) == ("train",)
    assert config["writer"]["architecture"] == (
        "pi05_factorized_condition_kernel_program_memory_v1"
    )
    assert config["writer"]["condition_feature_width"] == 1024
    assert config["writer"]["program_memory_parameter_count"] == 83_886_080
    assert config["conditioning_training"]["global_tasks_per_optimizer_update"] == 24
    assert config["conditioning_training"]["factor_decoder_train_through_macro"] == 50
    assert config["conditioning_training"]["program_memory_optimizer"] == "none"
    assert config["formal_run"]["checkpoint_steps"] == [50, 100, 150, 200]
    authority = load_condition_authority(
        str(REPO_ROOT / config["authorities"]["condition_address"]["path"])
    )
    assert authority["task_frequencies"].shape == (16, 2048)
    assert authority["video_frequencies"].shape == (16, 512)


def test_profile_uses_its_own_video_seed_without_mutating_formal_config() -> None:
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
    total, batch, checkpoints = resolve_runtime(args, profile, _context())
    assert (total, batch, checkpoints, args.stop_after_step) == (
        3,
        20,
        (1, 2, 3),
        3,
    )


def test_sealed_formal_launch_requires_a_clean_worktree() -> None:
    config = load_writer_config(CONFIG)
    args = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
    )
    with pytest.raises(WriterModelError, match="clean worktree"):
        resolve_runtime(args, config, _context())


def test_runtime_contract_has_one_program_owner_and_frozen_decoder_boundary() -> None:
    config = load_writer_config(CONFIG)
    runtime = build_update_runtime_contract(
        config=config,
        context=_context(),
        video_data={"sampled_frame_cost_sha256": "frames"},
        total_steps=200,
        stop_step=200,
        batch_size=20,
        batch_cycle=(20,),
        checkpoint_steps=(50, 100, 150, 200),
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
        "condition_kernel_program_memory_full24_v1"
    )
    assert runtime["program_memory_optimizer"] == "none"
    assert runtime["factor_decoder_optimizer_updates"] == 50
    assert runtime["program_credit_allgathers_per_macro"] == 3
    assert checkpoint_state_family(config) == runtime["checkpoint_state_family"]
