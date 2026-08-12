from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_config import load_writer_config, parse_macro_boundaries
from ember.writer.as_contract import publish_contract
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.as_step import (
    _pack_condition,
    accumulate_flat_gradient,
    assign_flat_gradient,
    parameter_layout,
    reduce_full24_gradient,
)
from ember.writer.checkpoint import load_writer_checkpoint, save_writer_checkpoint
from ember.writer.data import RawTeacherVideo
from ember.writer.errors import WriterModelError
from ember.writer.update_schedule import build_exposure_scheduler
from ember.pi05_source_checkpoint import DistributedContext


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]


def _dataset_stub() -> _DatasetStub:
    rows: dict[int, dict[int, tuple[int, ...]]] = {}
    frame_index = []
    cursor = 0
    for task_id in range(24):
        rows[task_id] = {}
        for demo in range(50):
            rows[task_id][demo] = (cursor,)
            frame_index.append((task_id, demo, 0))
            cursor += 1
    return _DatasetStub(rows, tuple(frame_index))


def test_rank8_dynamic_k_config_is_mechanical_and_loadable() -> None:
    config = load_writer_config(
        REPO_ROOT
        / "configs/pi05_as_writer_dynamic_k_backbone_memory_rank8_v1.json"
    )
    lora = load_pi05_lora_contract(
        REPO_ROOT / "configs/pi05_lora_rank8_writer_v1.json"
    )
    assert (lora.rank, lora.alpha, lora.parameter_count) == (8, 8, 643_584)
    assert lora.state_tensor_count == 76
    assert config["data"]["dynamic_k_max"] == 4
    assert config["optimization"]["distributed"]["fresh_world_sizes"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert parse_macro_boundaries("every:25", 400) == tuple(
        range(25, 401, 25)
    )
    assert parse_macro_boundaries("1,2,3", 3) == (1, 2, 3)


def test_dynamic_k_schedule_balances_each_macro_and_each_task_cycle() -> None:
    task_ids = tuple(range(24))
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=20260722,
        videos_per_visit=4,
        dynamic_k_max=4,
    )
    for macro in range(9):
        counts = [
            schedule.shot_count_for_task_visit(task_id, macro)
            for task_id in task_ids
        ]
        assert {k: counts.count(k) for k in range(1, 5)} == {
            1: 6,
            2: 6,
            3: 6,
            4: 6,
        }
    for task_id in task_ids:
        assert {
            schedule.shot_count_for_task_visit(task_id, macro)
            for macro in range(4)
        } == {1, 2, 3, 4}


def test_dynamic_k_videos_are_unique_and_in_action_episode_complement() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=tuple(range(24)),
        demo_indices=range(50),
        seed=20260722,
        videos_per_visit=4,
        dynamic_k_max=4,
    )
    excluded = tuple(range(20))
    for task_id in schedule.task_ids:
        for macro in range(4):
            selected = schedule.demos_for_task_visit(
                task_id, macro, excluded=excluded
            )
            assert len(selected) == schedule.shot_count_for_task_visit(
                task_id, macro
            )
            assert len(selected) == len(set(selected))
            assert not set(selected) & set(excluded)


@pytest.mark.parametrize("world_size", range(1, 7))
def test_dynamic_k_sampler_covers_full24_with_uneven_world_sizes(
    world_size: int,
) -> None:
    dataset = _dataset_stub()
    task_ids = tuple(range(24))
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=20260722,
        videos_per_visit=4,
        dynamic_k_max=4,
    )
    samplers = [
        MixedTaskBatchSampler(
            dataset,  # type: ignore[arg-type]
            task_ids=task_ids,
            per_rank_batch_size=20,
            start_step=0,
            stop_step=2,
            rank=rank,
            world_size=world_size,
            seed=20260721,
            tasks_per_rank_per_update=(24 + world_size - 1) // world_size,
            video_schedule=schedule,
            task_video_costs={
                task_id: {demo: 1 + task_id + demo for demo in range(50)}
                for task_id in task_ids
            },
            assignment_strategy="cost_balanced_long_first_dynamic_uneven",
        )
        for rank in range(world_size)
    ]
    for macro in range(2):
        shards = [sampler.tasks_for_step(macro) for sampler in samplers]
        assert {task for shard in shards for task in shard} == set(task_ids)
        assert sum(map(len, shards)) == 24
        assert max(map(len, shards)) - min(map(len, shards)) <= 1
        for sampler, shard in zip(samplers, shards, strict=True):
            for task_id in shard:
                excluded = sampler.action_demo_indices_for_task_visit(
                    task_id, macro
                )
                selected = schedule.demos_for_task_visit(
                    task_id, macro, excluded=excluded
                )
                assert len(excluded) == 20
                assert not set(selected) & set(excluded)
        costs, _, _ = samplers[0]._cost_order_for_task_cycle(macro)
        task_id = task_ids[0]
        excluded = samplers[0].action_demo_indices_for_task_visit(task_id, macro)
        videos = schedule.demos_for_task_visit(
            task_id, macro, excluded=excluded
        )
        assert costs[task_id] == 20 + sum(
            1 + task_id + demo for demo in videos
        )


def test_flat_gradient_accumulates_tasks_then_divides_once_by_24() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False),
        torch.nn.Linear(3, 1, bias=False),
    )
    layout = parameter_layout(module)
    flat = torch.zeros(layout[-1].stop)
    for task in range(24):
        gradients = tuple(
            torch.full_like(item.parameter, float(task + 1)) for item in layout
        )
        accumulate_flat_gradient(flat, gradients, layout)
    expected = sum(range(1, 25)) / 24
    reduce_full24_gradient(flat, world_size=1)
    assert torch.all(flat == expected)
    assign_flat_gradient(flat, layout)
    assert all(torch.all(item.parameter.grad == expected) for item in layout)


def test_flat_gradient_contract_rejects_non_full24_reduction() -> None:
    with pytest.raises(WriterModelError, match="full24"):
        reduce_full24_gradient(torch.ones(2), world_size=1, global_task_count=23)


def test_scheduler_applies_warmup_lr_before_first_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = build_exposure_scheduler(
        optimizer,
        {
            "kind": "cosine_decay_with_warmup",
            "peak_lr": 3e-4,
            "warmup_steps": 17,
            "decay_steps": 400,
            "decay_lr": 1e-5,
        },
        400,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 17)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2 * 3e-4 / 17)


def test_ragged_offsets_remain_cpu_long_while_frames_follow_device() -> None:
    class _Store:
        @staticmethod
        def load(task_id: int, demo: int) -> RawTeacherVideo:
            del task_id
            count = demo + 2
            return RawTeacherVideo(
                frames=np.zeros((count, 3, 4, 4), dtype=np.uint8),
                frame_indices=np.arange(count, dtype=np.int64),
                raw_frame_count=count,
            )

    runtime = SimpleNamespace(
        video_store=_Store(),
        context=SimpleNamespace(device=torch.device("cpu")),
        language_tokens={
            7: (
                torch.ones((1, 3), dtype=torch.long),
                torch.ones((1, 3), dtype=torch.bool),
                torch.ones((1, 3), dtype=torch.bool),
            )
        },
    )
    packed, metrics = _pack_condition(runtime, 7, (0, 2))
    frames, indices, video_offsets, condition_offsets, *_ = packed
    assert frames.device.type == indices.device.type == "cpu"
    assert video_offsets.device.type == condition_offsets.device.type == "cpu"
    assert video_offsets.dtype == condition_offsets.dtype == torch.long
    assert video_offsets.tolist() == [0, 2, 6]
    assert condition_offsets.tolist() == [0, 2]
    assert metrics["total_sampled_frames"] == 6


def test_publish_contract_accepts_only_exact_resume_contract(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    contract = {"schema_version": "test", "runtime": {"world_size": 1}}
    args = argparse.Namespace(
        output_dir=tmp_path / "run",
        resume=None,
        stop_after_macro=1,
    )
    publish_contract(args, context, contract)
    args.resume = tmp_path / "run/checkpoints/macro_00000001"
    publish_contract(args, context, contract)
    with pytest.raises(WriterModelError, match="contract changed"):
        publish_contract(args, context, {**contract, "changed": True})


def test_hashless_checkpoint_restores_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.writer.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "capture_rng", lambda context: {"rank": context.rank})
    monkeypatch.setattr(checkpoint_module, "restore_rng", lambda state, context: None)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    writer = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    contract = {"schema_version": "launch_v1"}
    expected = writer.weight.detach().clone()
    checkpoint = save_writer_checkpoint(
        output_dir=tmp_path,
        macro=1,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
        metrics_rows=1,
    )
    with torch.no_grad():
        writer.weight.zero_()
    macro, rows = load_writer_checkpoint(
        checkpoint=checkpoint,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
    )
    assert (macro, rows) == (1, 1)
    assert torch.equal(writer.weight, expected)
