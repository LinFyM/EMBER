from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
)
from ember.writer.as_config import (
    AS_WRITER_SERIAL4_CONFIG_SCHEMA,
    load_writer_config,
)
from ember.writer.as_contract import (
    AS_WRITER_LAUNCH_SCHEMA,
    resolve_runtime,
)
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.checkpoint import (
    AS_WRITER_SERIAL4_CHECKPOINT_SCHEMA,
    AS_WRITER_SERIAL4_RANK_STATE_SCHEMA,
    _write_rank_state,
    _write_shared_state,
    load_writer_checkpoint,
)
from ember.writer.task_gradient import FlatParameter, compose_raw_mean_gradient
from ember.writer.update_schedule import (
    advance_scheduler_after_update,
    build_exposure_scheduler,
    logical_task_cycle_steps,
)
from ember.writer.update_contract import (
    build_update_runtime_contract,
    serial4_gradient_contract,
)


ROOT = Path(__file__).resolve().parents[1]
SERIAL_CONFIG = (
    ROOT
    / "configs/pi05_as_writer_unified_causal_program_serial4_exposurematched_v1.json"
)


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]


def _dataset() -> _DatasetStub:
    rows: dict[int, dict[int, tuple[int, ...]]] = {}
    frame_index: list[tuple[int, int, int]] = []
    cursor = 0
    for task_id in range(24):
        rows[task_id] = {}
        for demo_index in range(50):
            episode = tuple(range(cursor, cursor + 3))
            rows[task_id][demo_index] = episode
            frame_index.extend(
                (task_id, demo_index, frame) for frame in range(3)
            )
            cursor += 3
    return _DatasetStub(rows, tuple(frame_index))


def _sampler(
    dataset: _DatasetStub,
    *,
    rank: int,
    stop_step: int,
    tasks_per_rank: int,
    updates_per_cycle: int,
) -> MixedTaskBatchSampler:
    schedule = TeacherVideoSchedule(
        task_ids=range(24), demo_indices=range(50), seed=20260722
    )
    return MixedTaskBatchSampler(
        dataset,  # type: ignore[arg-type]
        task_ids=range(24),
        per_rank_batch_size=20,
        start_step=0,
        stop_step=stop_step,
        rank=rank,
        world_size=4,
        seed=20260721,
        tasks_per_rank_per_update=tasks_per_rank,
        optimizer_updates_per_task_cycle=updates_per_cycle,
        video_schedule=schedule,
        task_video_costs={
            task_id: {
                demo_index: 10 + task_id + 3 * demo_index
                for demo_index in range(50)
            }
            for task_id in range(24)
        },
    )


def test_serial4_six_phases_reconstruct_full24_rank_order_and_queries() -> None:
    dataset = _dataset()
    full = [
        _sampler(
            dataset,
            rank=rank,
            stop_step=3,
            tasks_per_rank=6,
            updates_per_cycle=1,
        )
        for rank in range(4)
    ]
    serial = [
        _sampler(
            dataset,
            rank=rank,
            stop_step=18,
            tasks_per_rank=1,
            updates_per_cycle=6,
        )
        for rank in range(4)
    ]

    for cycle in range(3):
        full_assignments = full[0].assignments_for_step(cycle)
        for rank, microtask, task_id, task_visit in full_assignments:
            selected = serial[0].assignments_for_step(cycle * 6 + microtask)
            assert (rank, 0, task_id, task_visit) in selected
            assert task_visit == cycle
        assert {
            task_id
            for phase in range(6)
            for _, _, task_id, _ in serial[0].assignments_for_step(cycle * 6 + phase)
        } == set(range(24))

    for rank in range(4):
        assert list(serial[rank]) == list(full[rank])
        for cycle in range(3):
            full_tasks = full[rank].tasks_for_step(cycle)
            serial_tasks = tuple(
                serial[rank].tasks_for_step(cycle * 6 + phase)[0]
                for phase in range(6)
            )
            assert serial_tasks == full_tasks
            for task_id in full_tasks:
                assert serial[rank].video_schedule.demo_for_task_visit(
                    task_id, cycle
                ) == full[rank].video_schedule.demo_for_task_visit(task_id, cycle)


def test_serial4_1200_updates_match_full24_exposure_counts() -> None:
    sampler = _sampler(
        _dataset(), rank=0, stop_step=1200, tasks_per_rank=1, updates_per_cycle=6
    )
    summary = sampler.video_schedule.consumed_identity_summary(sampler, 0, 1200)

    assert summary["query"]["global_examples"] == 96_000
    assert summary["query"]["min_examples_per_task"] == 4_000
    assert summary["query"]["max_examples_per_task"] == 4_000
    assert summary["min_video_visits_per_task"] == 200
    assert summary["max_video_visits_per_task"] == 200
    assert summary["min_unique_videos_per_task"] == 50


def _optimizer_and_scheduler(config: dict) -> tuple[torch.optim.Optimizer, object]:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [parameter], lr=float(config["optimization"]["scheduler"]["peak_lr"])
    )
    scheduler = build_exposure_scheduler(
        optimizer, config["optimization"]["scheduler"], 400
    )
    return optimizer, scheduler


def _lr_sequence(
    config: dict, updates: int, updates_per_cycle: int
) -> list[float]:
    optimizer, scheduler = _optimizer_and_scheduler(config)
    values = []
    for update in range(updates):
        values.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        advance_scheduler_after_update(
            scheduler,  # type: ignore[arg-type]
            completed_optimizer_updates=update + 1,
            optimizer_updates_per_task_cycle=updates_per_cycle,
        )
    return values


def test_serial4_lr_is_full24_staircase_repeated_six_and_resumable() -> None:
    config = load_writer_config(SERIAL_CONFIG)
    assert logical_task_cycle_steps(config, 2400) == 400
    full = _lr_sequence(config, 400, 1)
    serial = _lr_sequence(config, 2400, 6)
    assert serial == [value for value in full for _ in range(6)]

    optimizer, scheduler = _optimizer_and_scheduler(config)
    advanced = []
    for update in range(7):
        optimizer.step()
        advanced.append(
            advance_scheduler_after_update(
                scheduler,  # type: ignore[arg-type]
                completed_optimizer_updates=update + 1,
                optimizer_updates_per_task_cycle=6,
            )
        )
    assert advanced == [False, False, False, False, False, True, False]
    assert serial[:6] == [full[0]] * 6
    assert serial[6] == full[1]

    before_boundary_optimizer, before_boundary_scheduler = (
        _optimizer_and_scheduler(config)
    )
    for update in range(5):
        before_boundary_optimizer.step()
        assert not advance_scheduler_after_update(
            before_boundary_scheduler,  # type: ignore[arg-type]
            completed_optimizer_updates=update + 1,
            optimizer_updates_per_task_cycle=6,
        )
    optimizer_state = copy.deepcopy(before_boundary_optimizer.state_dict())
    scheduler_state = copy.deepcopy(before_boundary_scheduler.state_dict())
    resumed_optimizer, resumed_scheduler = _optimizer_and_scheduler(config)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    assert float(resumed_optimizer.param_groups[0]["lr"]) == full[0]
    resumed_optimizer.step()
    assert advance_scheduler_after_update(
        resumed_scheduler,  # type: ignore[arg-type]
        completed_optimizer_updates=6,
        optimizer_updates_per_task_cycle=6,
    )
    assert float(resumed_optimizer.param_groups[0]["lr"]) == full[1]
    resumed_optimizer.step()
    assert not advance_scheduler_after_update(
        resumed_scheduler,  # type: ignore[arg-type]
        completed_optimizer_updates=7,
        optimizer_updates_per_task_cycle=6,
    )


def test_serial4_selected4_raw_mean_has_4x4_gram_and_25pct_reference() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    layout = (
        FlatParameter(
            name="factor_heads.weight",
            parameter=parameter,
            start=0,
            stop=4,
            block="factor",
        ),
    )
    direction, metrics = compose_raw_mean_gradient(
        torch.tensor([3, 1, 4, 2], dtype=torch.long),
        torch.eye(4, dtype=torch.float32),
        layout,
    )
    runtime = serial4_gradient_contract(
        context=type("Context", (), {"world_size": 4})(),
        tasks_per_rank=1,
        global_tasks=4,
    )

    assert torch.equal(direction, torch.full((4,), 0.25))
    assert len(metrics["raw_gradient_gram"]) == 4
    assert all(len(row) == 4 for row in metrics["raw_gradient_gram"])
    assert metrics["raw_mean_to_average_task_energy_ratio"] == pytest.approx(0.25)
    assert runtime["gradient_gram_shape_per_optimizer_update"] == [4, 4]
    assert runtime["orthogonal_equal_norm_mean_to_task_energy_reference"] == 0.25
    assert runtime["energy_ratio_interpretation"].endswith(
        "not_scientific_success"
    )

    config = load_writer_config(SERIAL_CONFIG)
    launch = build_update_runtime_contract(
        config=config,
        context=type("Context", (), {"world_size": 4})(),
        video_data={"sampled_frame_cost_sha256": "a" * 64},
        total_steps=2400,
        stop_step=1200,
        batch_size=20,
        batch_cycle=(20,),
        checkpoint_steps=tuple(range(150, 2401, 150)),
        num_workers=2,
        rank_topology=({}, {}, {}, {}),
    )
    assert launch["optimizer_updates_per_task_cycle"] == 6
    assert launch["total_task_cycles"] == 400
    assert launch["global_tasks_per_optimizer_update"] == 4
    assert launch["gradient_gram_shape_per_optimizer_update"] == [4, 4]
    assert launch["global_policy_samples_per_optimizer_update"] == 80
    assert "global_task_gradients_per_macro" not in launch


def test_serial4_config_and_profile_axis_are_fresh_and_fail_closed() -> None:
    config = load_writer_config(SERIAL_CONFIG)
    full24 = load_writer_config(
        ROOT / "configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json"
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    args = argparse.Namespace(
        mode="profile",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )

    assert config["schema_version"] == AS_WRITER_SERIAL4_CONFIG_SCHEMA
    assert (config["writer"], config["authorities"]) == (
        full24["writer"],
        full24["authorities"],
    )
    assert (
        config["data"]["sampler_seed"],
        config["data"]["teacher_video_seed"],
    ) == (20260721, 20260722)
    assert (
        config["optimization"]["scheduler"]["warmup_steps"],
        config["optimization"]["scheduler"]["decay_steps"],
    ) == (17, 400)
    assert resolve_runtime(args, config, context) == (
        18,
        20,
        (1, 3, 5, 6, 7, 12, 18),
    )
    assert config["formal_run"]["one_hour_scale"]["candidate_optimizer_updates"] == [
        300,
        600,
        900,
        1200,
    ]


def test_serial4_midcycle_checkpoint_restores_data_and_scheduler_cursor(
    tmp_path: Path,
) -> None:
    config = load_writer_config(SERIAL_CONFIG)
    sampler = _sampler(
        _dataset(), rank=0, stop_step=18, tasks_per_rank=1, updates_per_cycle=6
    )
    writer = torch.nn.Linear(1, 1, bias=False)
    scheduler_config = config["optimization"]["scheduler"]
    optimizer = torch.optim.AdamW(
        writer.parameters(), lr=float(scheduler_config["peak_lr"])
    )
    scheduler = build_exposure_scheduler(optimizer, scheduler_config, 400)
    first_applied_lr = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    contract = {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "runtime": {
            "world_size": 1,
            "total_steps": 18,
            "tasks_per_rank_per_optimizer_update": 1,
            "teacher_videos_per_task_visit": 1,
            "optimizer_updates_per_task_cycle": 6,
        },
    }
    temporary = tmp_path / ".step.partial"
    final = tmp_path / "step_00000001"
    temporary.mkdir()
    _write_rank_state(
        temporary / "rank_00_state.pt",
        step=1,
        context=DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        ),
        sampler=sampler,
        video_schedule=sampler.video_schedule,
        saved_rng={"cursor": 1},
        videos_per_task_visit=1,
        tasks_per_rank_per_update=1,
        optimizer_updates_per_task_cycle=6,
        rank_state_schema=AS_WRITER_SERIAL4_RANK_STATE_SCHEMA,
    )
    consumed = _write_shared_state(
        temporary,
        final,
        output_dir=tmp_path,
        step=1,
        writer=writer,  # type: ignore[arg-type]
        optimizer=optimizer,
        scheduler=scheduler,  # type: ignore[arg-type]
        sampler=sampler,
        video_schedule=sampler.video_schedule,
        contract=contract,
        require_full_coverage=False,
        metrics_rows=1,
    )
    restored_writer = torch.nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(
        restored_writer.parameters(), lr=float(scheduler_config["peak_lr"])
    )
    restored_scheduler = build_exposure_scheduler(
        restored_optimizer, scheduler_config, 400
    )
    loaded, rng, rows = load_writer_checkpoint(
        checkpoint=final,
        context=DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        ),
        writer=restored_writer,  # type: ignore[arg-type]
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,  # type: ignore[arg-type]
        sampler_seed=sampler.seed,
        teacher_video_seed=sampler.video_schedule.seed,
        per_rank_batch_size=20,
        per_rank_batch_cycle=(20,),
        videos_per_task_visit=1,
        tasks_per_rank_per_update=1,
        optimizer_updates_per_task_cycle=6,
        contract_sha256=canonical_hash(contract),
    )

    assert consumed["next_step"] == loaded == 1
    assert consumed["next_data_step"] == 1
    assert consumed["next_task_cycle"] == 0
    assert consumed["next_task_cycle_phase"] == 1
    assert consumed["scheduler_logical_updates"] == 0
    assert read_json(final / "checkpoint_manifest.json")["schema_version"] == (
        AS_WRITER_SERIAL4_CHECKPOINT_SCHEMA
    )
    assert rng == {"cursor": 1}
    assert rows == 1
    resumed_lrs = []
    for update in range(1, 3):
        resumed_lrs.append(float(restored_optimizer.param_groups[0]["lr"]))
        restored_optimizer.step()
        assert not advance_scheduler_after_update(
            restored_scheduler,  # type: ignore[arg-type]
            completed_optimizer_updates=update + 1,
            optimizer_updates_per_task_cycle=6,
        )
    assert [first_applied_lr, *resumed_lrs] == _lr_sequence(config, 3, 6)
