from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
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
    AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA,
    AS_WRITER_SERIAL4_CONFIG_SCHEMA,
    _validate_formal_schedule,
    load_writer_config,
)
from ember.writer.as_contract import (
    AS_WRITER_LAUNCH_SCHEMA,
    _validate_formal_runtime,
    parse_checkpoint_steps,
    resolve_runtime,
)
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.checkpoint import (
    AS_WRITER_CYCLE_NORMALIZED_GROUP4_CHECKPOINT_SCHEMA,
    AS_WRITER_CYCLE_NORMALIZED_GROUP4_RANK_STATE_SCHEMA,
    AS_WRITER_SERIAL4_CHECKPOINT_SCHEMA,
    AS_WRITER_SERIAL4_RANK_STATE_SCHEMA,
    _state_schemas,
    _write_rank_state,
    _write_shared_state,
    load_writer_checkpoint,
)
from ember.writer.model import WriterModelError
from ember.writer.functional import (
    functional_lora_loss_gradient,
    scoped_policy_randomness,
    task_query_policy_rng_seed,
)
from ember.writer.task_gradient import FlatParameter, compose_raw_mean_gradient
from ember.writer.update_schedule import (
    advance_scheduler_after_update,
    build_exposure_scheduler,
    cycle_matched_weight_decay,
    logical_task_cycle_steps,
    prepare_optimizer_update,
)
from ember.writer.update_contract import (
    build_update_runtime_contract,
    checkpoint_state_family,
    serial4_gradient_contract,
)


ROOT = Path(__file__).resolve().parents[1]
SERIAL_CONFIG = (
    ROOT
    / "configs/pi05_as_writer_contextual_value_dual_read_serial4_exposurematched_v1.json"
)
TASK_QUERY_RAW_CONFIG = (
    ROOT
    / "configs/pi05_as_writer_contextual_value_dual_read_taskquery_rawfull24_v1.json"
)
GROUP4_CONFIG = (
    ROOT
    / "configs/pi05_as_writer_contextual_value_dual_read_cycle_normalized_group4_v1.json"
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
    assignment_strategy: str = "cost_balanced_long_first",
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
        assignment_strategy=assignment_strategy,
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


def test_randomized_group4_is_exactly_phase_balanced_and_cost_independent() -> None:
    dataset = _dataset()
    sampler = _sampler(
        dataset,
        rank=0,
        stop_step=1200,
        tasks_per_rank=1,
        updates_per_cycle=6,
        assignment_strategy="randomized_latin_group4",
    )
    changed_costs = _sampler(
        dataset,
        rank=0,
        stop_step=1200,
        tasks_per_rank=1,
        updates_per_cycle=6,
        assignment_strategy="randomized_latin_group4",
    )
    for task_id, demo_costs in changed_costs.task_video_costs.items():
        for demo_index in demo_costs:
            demo_costs[demo_index] = 100_000 - 97 * task_id - demo_index

    phase_counts = {task_id: [0] * 6 for task_id in range(24)}
    for task_cycle in range(200):
        cycle_tasks = []
        for phase in range(6):
            step = task_cycle * 6 + phase
            assignments = sampler.assignments_for_step(step)
            assert assignments == changed_costs.assignments_for_step(step)
            assert [rank for rank, _, _, _ in assignments] == [0, 1, 2, 3]
            assert len({task_id for _, _, task_id, _ in assignments}) == 4
            for _, microtask, task_id, task_visit in assignments:
                assert microtask == 0
                assert task_visit == task_cycle
                phase_counts[task_id][phase] += 1
                cycle_tasks.append(task_id)
        assert sorted(cycle_tasks) == list(range(24))

    for task_id, counts in phase_counts.items():
        assert sorted(counts) == [33, 33, 33, 33, 34, 34], task_id
        assert sum(phase * count for phase, count in enumerate(counts)) == 500

    assert sampler.assignments_for_step(0) == (
        (0, 0, 18, 0),
        (1, 0, 21, 0),
        (2, 0, 6, 0),
        (3, 0, 7, 0),
    )
    assert sampler.assignments_for_step(1199) == (
        (0, 0, 17, 199),
        (1, 0, 0, 199),
        (2, 0, 3, 199),
        (3, 0, 21, 199),
    )


def test_group4_and_raw_share_exact_task_video_and_query_exposures() -> None:
    dataset = _dataset()
    raw = _sampler(
        dataset,
        rank=0,
        stop_step=200,
        tasks_per_rank=6,
        updates_per_cycle=1,
    )
    group4 = _sampler(
        dataset,
        rank=0,
        stop_step=1200,
        tasks_per_rank=1,
        updates_per_cycle=6,
        assignment_strategy="randomized_latin_group4",
    )
    for task_cycle in (0, 17, 197, 198, 199):
        raw_tasks = {
            task_id: task_visit
            for _, _, task_id, task_visit in raw.assignments_for_step(task_cycle)
        }
        grouped_tasks = {
            task_id: task_visit
            for phase in range(6)
            for _, _, task_id, task_visit in group4.assignments_for_step(
                task_cycle * 6 + phase
            )
        }
        assert raw_tasks == grouped_tasks == {
            task_id: task_cycle for task_id in range(24)
        }
        for task_id in range(24):
            assert raw.video_schedule.demo_for_task_visit(
                task_id, task_cycle
            ) == group4.video_schedule.demo_for_task_visit(task_id, task_cycle)
            raw_rows = tuple(
                raw._sample_for_task_visit(task_id, task_cycle, offset)
                for offset in range(20)
            )
            grouped_rows = tuple(
                group4._sample_for_task_visit(task_id, task_cycle, offset)
                for offset in range(20)
            )
            assert raw_rows == grouped_rows


def test_task_query_policy_rng_is_phase_independent_and_restores_outer_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = task_query_policy_rng_seed(
        optimization_seed=7,
        task_id=11,
        task_visit=37,
        demo_indices=[4, 9, 12],
        frame_indices=[18, 3, 42],
    )
    assert seed == task_query_policy_rng_seed(
        optimization_seed=7,
        task_id=11,
        task_visit=37,
        demo_indices=[4, 9, 12],
        frame_indices=[18, 3, 42],
    )
    assert seed != task_query_policy_rng_seed(
        optimization_seed=7,
        task_id=11,
        task_visit=37,
        demo_indices=[9, 4, 12],
        frame_indices=[3, 18, 42],
    )

    def fake_call(policy, leaves, contract, batch):  # type: ignore[no-untyped-def]
        del policy, contract, batch
        noise = torch.rand_like(leaves["x"])
        return (leaves["x"] * noise).sum(), {"noise": noise.tolist()}

    monkeypatch.setattr("ember.writer.functional.functional_lora_call", fake_call)
    policy = torch.nn.Identity()
    state = {"x": torch.tensor([1.0, 2.0, 3.0])}
    torch.random.default_generator.manual_seed(101)
    outer = torch.get_rng_state().clone()
    first = functional_lora_loss_gradient(
        policy,
        state,
        None,  # type: ignore[arg-type]
        batch={},
        policy_rng_seed=seed,
        policy_rng_device=torch.device("cpu"),
    )
    assert torch.equal(torch.get_rng_state(), outer)
    torch.random.default_generator.manual_seed(999)
    changed_outer = torch.get_rng_state().clone()
    second = functional_lora_loss_gradient(
        policy,
        state,
        None,  # type: ignore[arg-type]
        batch={},
        policy_rng_seed=seed,
        policy_rng_device=torch.device("cpu"),
    )
    assert torch.equal(torch.get_rng_state(), changed_outer)
    assert torch.equal(first[2]["x"], second[2]["x"])
    assert first[0] == second[0]
    with scoped_policy_randomness(seed, torch.device("cpu")):
        direct = torch.rand(3)
    assert torch.equal(direct, first[2]["x"])


def test_cuda_policy_rng_keys_cpu_time_and_cuda_noise_and_restores_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the CUDA branch without requiring a physical CUDA device."""

    cuda_generator = torch.Generator()

    @contextmanager
    def fake_fork_rng(*, devices: list[int], device_type: str):
        assert devices == [0]
        assert device_type == "cuda"
        cpu_state = torch.get_rng_state().clone()
        cuda_state = cuda_generator.get_state().clone()
        try:
            yield
        finally:
            torch.set_rng_state(cpu_state)
            cuda_generator.set_state(cuda_state)

    monkeypatch.setattr(torch.random, "fork_rng", fake_fork_rng)
    monkeypatch.setattr(torch.cuda, "default_generators", (cuda_generator,))

    torch.random.default_generator.manual_seed(101)
    cuda_generator.manual_seed(202)
    outer_cpu = torch.get_rng_state().clone()
    outer_cuda = cuda_generator.get_state().clone()
    with scoped_policy_randomness(303, torch.device("cuda:0")):
        first_time = torch.rand(4)
        first_noise = torch.rand(4, generator=cuda_generator)
    assert torch.equal(torch.get_rng_state(), outer_cpu)
    assert torch.equal(cuda_generator.get_state(), outer_cuda)

    torch.random.default_generator.manual_seed(404)
    cuda_generator.manual_seed(505)
    changed_outer_cpu = torch.get_rng_state().clone()
    changed_outer_cuda = cuda_generator.get_state().clone()
    with scoped_policy_randomness(303, torch.device("cuda:0")):
        second_time = torch.rand(4)
        second_noise = torch.rand(4, generator=cuda_generator)
    assert torch.equal(torch.get_rng_state(), changed_outer_cpu)
    assert torch.equal(cuda_generator.get_state(), changed_outer_cuda)
    assert torch.equal(first_time, second_time)
    assert torch.equal(first_noise, second_noise)


def test_cycle_normalized_optimizer_matches_lr_decay_and_moment_clock() -> None:
    raw_config = load_writer_config(TASK_QUERY_RAW_CONFIG)
    group_config = load_writer_config(GROUP4_CONFIG)
    raw_parameter = torch.nn.Parameter(torch.ones(()))
    group_parameter = torch.nn.Parameter(torch.ones(()))

    def build(config: dict, parameter: torch.nn.Parameter):
        optimizer_config = config["optimization"]["optimizer"]
        optimizer = torch.optim.AdamW(
            [parameter],
            lr=float(config["optimization"]["scheduler"]["peak_lr"]),
            betas=tuple(optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        scheduler = build_exposure_scheduler(
            optimizer, config["optimization"]["scheduler"], 200
        )
        return optimizer, scheduler

    raw_optimizer, raw_scheduler = build(raw_config, raw_parameter)
    group_optimizer, group_scheduler = build(group_config, group_parameter)
    raw_update = prepare_optimizer_update(
        raw_optimizer, raw_scheduler, raw_config
    )
    group_update = prepare_optimizer_update(
        group_optimizer, group_scheduler, group_config
    )
    assert group_update["logical_lr"] == raw_update["logical_lr"]
    assert group_update["applied_lr"] * 6 == pytest.approx(
        raw_update["applied_lr"]
    )
    assert tuple(group_optimizer.param_groups[0]["betas"])[0] ** 6 == pytest.approx(0.9)
    assert tuple(group_optimizer.param_groups[0]["betas"])[1] ** 6 == pytest.approx(0.95)
    assert (
        1.0
        - group_update["applied_lr"]
        * group_update["applied_weight_decay"]
    ) ** 6 == pytest.approx(
        1.0 - raw_update["applied_lr"] * raw_update["applied_weight_decay"],
        abs=1e-15,
    )
    assert group_update["applied_weight_decay"] == pytest.approx(
        cycle_matched_weight_decay(
            group_update["logical_lr"], 1e-4, 6
        )
    )

    raw_parameter.grad = torch.ones_like(raw_parameter)
    raw_optimizer.step()
    assert advance_scheduler_after_update(
        raw_scheduler,
        completed_optimizer_updates=1,
        optimizer_updates_per_task_cycle=1,
    )
    for phase in range(6):
        update = prepare_optimizer_update(
            group_optimizer, group_scheduler, group_config
        )
        assert update["applied_lr"] == group_update["applied_lr"]
        group_parameter.grad = torch.ones_like(group_parameter)
        group_optimizer.step()
        assert advance_scheduler_after_update(
            group_scheduler,
            completed_optimizer_updates=phase + 1,
            optimizer_updates_per_task_cycle=6,
        ) == (phase == 5)
    raw_state = raw_optimizer.state[raw_parameter]
    group_state = group_optimizer.state[group_parameter]
    assert group_state["exp_avg"] == pytest.approx(raw_state["exp_avg"])
    assert group_state["exp_avg_sq"] == pytest.approx(raw_state["exp_avg_sq"])
    assert raw_scheduler.state_dict()["last_epoch"] == 1
    assert group_scheduler.state_dict()["last_epoch"] == 1


def test_cycle_normalized_configs_and_checkpoint_families_fail_closed() -> None:
    raw = load_writer_config(TASK_QUERY_RAW_CONFIG)
    group4 = load_writer_config(GROUP4_CONFIG)
    assert raw["schema_version"] == AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA
    assert group4["schema_version"] == AS_WRITER_CYCLE_NORMALIZED_CONFIG_SCHEMA
    assert raw["formal_run"]["status"] == "sealed"
    assert raw["formal_run"]["launch_state"] == (
        "ready_from_clean_detached_postseal_commit"
    )
    assert group4["formal_run"]["status"] == "pending_profile"
    assert group4["formal_run"]["launch_state"].startswith("blocked_until")
    assert raw["formal_run"]["total_steps"] == 400
    assert group4["formal_run"]["total_steps"] == 2400
    assert raw["formal_run"]["stage_stop_steps"] == [200, 400]
    assert group4["formal_run"]["stage_stop_steps"] == [1200, 2400]
    assert logical_task_cycle_steps(
        raw, raw["formal_run"]["total_steps"]
    ) == 400
    assert logical_task_cycle_steps(
        group4, group4["formal_run"]["total_steps"]
    ) == 400
    compressed = copy.deepcopy(raw)
    compressed["formal_run"]["total_steps"] = 200
    with pytest.raises(WriterModelError, match="would auto-scale"):
        _validate_formal_schedule(compressed)
    truncated = copy.deepcopy(raw)
    truncated["formal_run"]["stage_stop_steps"] = [200]
    with pytest.raises(WriterModelError, match="would auto-scale"):
        _validate_formal_schedule(truncated)
    assert raw["profile_evidence"]["exact_resume_smoke"] is not None
    assert raw["profile_evidence"]["exact_resume_smoke"][
        "step1_all_payload_sha_size_mtime_unchanged"
    ] is True
    assert group4["profile_evidence"]["exact_resume_smoke"] is None
    assert checkpoint_state_family(raw) == "cvadr_task_query_keyed_rawfull24_v2"
    assert checkpoint_state_family(group4) == (
        "cvadr_cycle_normalized_randomized_group4_v2"
    )
    assert _state_schemas(1, checkpoint_state_family(raw))[0] != (
        _state_schemas(1)[0]
    )
    assert _state_schemas(6, checkpoint_state_family(group4)) == (
        AS_WRITER_CYCLE_NORMALIZED_GROUP4_CHECKPOINT_SCHEMA,
        "ember_pi05_contextual_value_dual_read_cycle_normalized_group4_trainer_state_v2",
        AS_WRITER_CYCLE_NORMALIZED_GROUP4_RANK_STATE_SCHEMA,
    )
    with pytest.raises(WriterModelError, match="unsupported"):
        _state_schemas(1, "ucp_task_query_keyed_rawfull24_v1")
    with pytest.raises(WriterModelError, match="unsupported"):
        _state_schemas(6, "ucp_cycle_normalized_randomized_group4_v1")
    with pytest.raises(WriterModelError, match="unsupported"):
        _state_schemas(1, "cvadr_task_query_keyed_rawfull24_v1")
    with pytest.raises(WriterModelError, match="unsupported"):
        _state_schemas(6, "cvadr_cycle_normalized_randomized_group4_v1")

    launch = build_update_runtime_contract(
        config=group4,
        context=type("Context", (), {"world_size": 4})(),
        video_data={"sampled_frame_cost_sha256": "b" * 64},
        total_steps=2400,
        stop_step=1200,
        batch_size=20,
        batch_cycle=(20,),
        checkpoint_steps=tuple(range(150, 2401, 150)),
        num_workers=2,
        rank_topology=({}, {}, {}, {}),
    )
    assert launch["checkpoint_state_family"] == (
        "cvadr_cycle_normalized_randomized_group4_v2"
    )
    assert launch["phase_cost_assignment_input"] == "none"
    assert launch["rank_local_long_first"] == "single_task_trivial_order"
    assert launch["optimizer_cycle_normalization"]["lr_divisor"] == 6


def test_cycle_normalized_formal_runtime_keeps_two_stage_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "dirty_paths": [],
            "commit": "sealed",
            "origin_main": "sealed",
        },
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        numa_node=1,
        cpu_affinity=(38,),
    )
    for path, expected_total, expected_stop, interval in (
        (TASK_QUERY_RAW_CONFIG, 400, 200, 25),
        (GROUP4_CONFIG, 2400, 1200, 150),
    ):
        args = argparse.Namespace(
            mode="formal",
            total_steps=None,
            batch_size=None,
            checkpoint_steps=None,
            stop_after_step=None,
            resume=None,
            skip_data_sha=True,
        )
        config = copy.deepcopy(load_writer_config(path))
        if path == TASK_QUERY_RAW_CONFIG:
            assert config["formal_run"]["status"] == "sealed"
            blocked = copy.deepcopy(config)
            blocked["formal_run"]["status"] = "pending_profile"
            with pytest.raises(WriterModelError, match="not sealed"):
                resolve_runtime(args, blocked, context)
        else:
            assert config["formal_run"]["status"] == "pending_profile"
            with pytest.raises(WriterModelError, match="not sealed"):
                resolve_runtime(args, config, context)
            config["formal_run"]["status"] = "sealed"
        total, batch, checkpoints = resolve_runtime(args, config, context)
        assert (total, batch, args.stop_after_step) == (
            expected_total,
            20,
            expected_stop,
        )
        assert checkpoints == tuple(range(interval, expected_total + 1, interval))


def _optimizer_and_scheduler(config: dict) -> tuple[torch.optim.Optimizer, object]:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [parameter], lr=float(config["optimization"]["scheduler"]["peak_lr"])
    )
    scheduler = build_exposure_scheduler(
        optimizer,
        config["optimization"]["scheduler"],
        logical_task_cycle_steps(
            config, config["formal_run"]["total_steps"]
        ),
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
        ROOT / "configs/pi05_as_writer_contextual_value_dual_read_full24_decay400_v1.json"
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
        config["profile_evidence"]["profile_teacher_video_seed"],
        config["profile_evidence"]["formal_teacher_video_seed_after_profile_seal"],
    ) == (172, 20260722)
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


def test_serial4_formal_boundaries_must_complete_task_cycles() -> None:
    config = load_writer_config(SERIAL_CONFIG)
    config["formal_run"] = copy.deepcopy(config["formal_run"])
    config["formal_run"].update({
        "status": "sealed",
        "checkpoint_steps": [149, 1200, 2400],
    })
    checkpoints = parse_checkpoint_steps(
        config["formal_run"]["checkpoint_steps"], 2400,
    )
    context = DistributedContext(
        rank=0, local_rank=0, world_size=4, device=torch.device("cpu"),
        numa_node=1, cpu_affinity=(48,),
    )
    with pytest.raises(WriterModelError, match="boundaries must complete"):
        _validate_formal_runtime(
            argparse.Namespace(resume=Path("checkpoint")),
            config,
            context,
            total_steps=2400,
            batch_size=20,
            checkpoint_steps=checkpoints,
            default_stop=1200,
            stop_step=1200,
        )


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


def test_cycle_normalized_group4_midcycle_resume_restores_optimizer_clock(
    tmp_path: Path,
) -> None:
    config = load_writer_config(GROUP4_CONFIG)
    sampler = _sampler(
        _dataset(),
        rank=0,
        stop_step=18,
        tasks_per_rank=1,
        updates_per_cycle=6,
        assignment_strategy="randomized_latin_group4",
    )
    writer = torch.nn.Linear(1, 1, bias=False)

    def build_optimizer(module: torch.nn.Module):
        optimizer_config = config["optimization"]["optimizer"]
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=float(config["optimization"]["scheduler"]["peak_lr"]),
            betas=tuple(optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        scheduler = build_exposure_scheduler(
            optimizer, config["optimization"]["scheduler"], 200
        )
        return optimizer, scheduler

    optimizer, scheduler = build_optimizer(writer)
    writer.weight.grad = torch.ones_like(writer.weight)
    first_update = prepare_optimizer_update(optimizer, scheduler, config)
    optimizer.step()
    assert not advance_scheduler_after_update(
        scheduler,
        completed_optimizer_updates=1,
        optimizer_updates_per_task_cycle=6,
    )
    family = checkpoint_state_family(config)
    contract = {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "runtime": {
            "world_size": 1,
            "total_steps": 18,
            "tasks_per_rank_per_optimizer_update": 1,
            "teacher_videos_per_task_visit": 1,
            "optimizer_updates_per_task_cycle": 6,
            "checkpoint_state_family": family,
        },
    }
    temporary = tmp_path / ".group4.partial"
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
        rank_state_schema=AS_WRITER_CYCLE_NORMALIZED_GROUP4_RANK_STATE_SCHEMA,
        checkpoint_state_family=family,
    )
    _write_shared_state(
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
    restored_optimizer, restored_scheduler = build_optimizer(restored_writer)
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
        checkpoint_state_family=family,
    )
    assert loaded == rows == 1
    assert rng == {"cursor": 1}
    assert first_update["applied_lr"] == pytest.approx(
        float(restored_optimizer.param_groups[0]["lr"])
    )
    for step in (1, 2):
        for parameter, active_optimizer, active_scheduler in (
            (writer.weight, optimizer, scheduler),
            (restored_writer.weight, restored_optimizer, restored_scheduler),
        ):
            parameter.grad = torch.ones_like(parameter)
            prepare_optimizer_update(
                active_optimizer, active_scheduler, config
            )
            active_optimizer.step()
            assert not advance_scheduler_after_update(
                active_scheduler,
                completed_optimizer_updates=step + 1,
                optimizer_updates_per_task_cycle=6,
            )
    assert torch.equal(writer.weight, restored_writer.weight)
    assert optimizer.state_dict() == restored_optimizer.state_dict()
    assert scheduler.state_dict() == restored_scheduler.state_dict()
