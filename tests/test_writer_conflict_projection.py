from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ember.writer.conflict_projection import (
    FlatParameter,
    assign_flat_gradient,
    compose_conflict_projected_gradient,
    compose_distributed_conflict_projected_gradient,
    gradient_direction_sketches,
)


def _layout(parameter: torch.nn.Parameter) -> tuple[FlatParameter, ...]:
    return (
        FlatParameter(
            name="factor_heads.test",
            parameter=parameter,
            start=0,
            stop=parameter.numel(),
            block="factor",
        ),
    )


def test_cp24_no_conflict_is_exact_raw_mean() -> None:
    parameter = torch.nn.Parameter(torch.zeros(24))
    gradients = torch.eye(24, dtype=torch.float32)
    direction, metrics = compose_conflict_projected_gradient(
        torch.arange(24, dtype=torch.long),
        gradients,
        _layout(parameter),
        seed=7,
        macro_step=5,
    )
    assert torch.equal(direction, gradients.mean(dim=0))
    assert metrics["no_conflict_exact_raw_mean"] is True
    assert metrics["projection_count"] == 0


def test_cp24_conflict_projection_is_deterministic_and_assignable() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3))
    gradients = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]],
        dtype=torch.float32,
    )
    first, first_metrics = compose_conflict_projected_gradient(
        torch.tensor([3, 7, 11]),
        gradients,
        _layout(parameter),
        seed=13,
        macro_step=9,
    )
    second, second_metrics = compose_conflict_projected_gradient(
        torch.tensor([3, 7, 11]),
        gradients,
        _layout(parameter),
        seed=13,
        macro_step=9,
    )
    assert torch.equal(first, second)
    assert first_metrics == second_metrics
    assert first_metrics["projection_count"] > 0
    assert first_metrics["no_conflict_exact_raw_mean"] is False
    assert torch.isfinite(first).all()
    assign_flat_gradient(first, _layout(parameter))
    assert torch.equal(parameter.grad, first)


def test_gradient_direction_sketch_is_fixed_and_sign_sensitive() -> None:
    parameter = torch.nn.Parameter(torch.zeros(64))
    gradients = torch.arange(128, dtype=torch.float32).reshape(2, 64)
    first = gradient_direction_sketches(
        gradients,
        _layout(parameter),
        dimensions=8,
    )["factor"]
    second = gradient_direction_sketches(
        gradients,
        _layout(parameter),
        dimensions=8,
    )["factor"]
    reversed_direction = gradient_direction_sketches(
        -gradients,
        _layout(parameter),
        dimensions=8,
    )["factor"]
    assert torch.equal(first, second)
    assert torch.equal(reversed_direction, -first)
    assert first.shape == (2, 8)


def test_bounded_cp_composition_matches_sorted_reference_on_one_rank() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    local_ids = torch.tensor([11, 3, 7])
    local_gradients = torch.tensor(
        [
            [0.0, -1.0, 1.0, 0.5],
            [1.0, 0.0, 0.0, 0.5],
            [-1.0, 1.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    order = torch.argsort(local_ids)
    reference, reference_metrics = compose_conflict_projected_gradient(
        local_ids[order],
        local_gradients[order],
        _layout(parameter),
        seed=13,
        macro_step=9,
    )
    bounded, bounded_metrics = compose_distributed_conflict_projected_gradient(
        local_ids,
        local_gradients,
        _layout(parameter),
        expected_task_ids=(3, 7, 11),
        world_size=1,
        rank=0,
        seed=13,
        macro_step=9,
        gram_chunk_elements=2,
    )
    assert torch.allclose(bounded, reference, atol=1e-7, rtol=1e-7)
    for key in (
        "raw_gradient_gram",
        "projected_gradient_gram",
        "projection_count",
        "raw_candidate_negative_tasks",
        "projected_candidate_negative_tasks",
    ):
        assert bounded_metrics[key] == reference_metrics[key]
    assert bounded_metrics["gradient_gram_chunk_allgathers"] == 0
    assert bounded_metrics["distributed_full_gradient_materialized"] is False


def _distributed_cp_worker(
    rank: int,
    world_size: int,
    initialization_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{initialization_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.nn.Parameter(torch.zeros(4))
        task_ids = (
            torch.tensor([7, 3])
            if rank == 0
            else torch.tensor([11, 5])
        )
        gradients = (
            torch.tensor(
                [[-1.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5]],
                dtype=torch.float32,
            )
            if rank == 0
            else torch.tensor(
                [[0.0, -1.0, 1.0, 0.5], [0.5, 0.0, -1.0, 0.5]],
                dtype=torch.float32,
            )
        )
        direction, metrics = compose_distributed_conflict_projected_gradient(
            task_ids,
            gradients,
            _layout(parameter),
            expected_task_ids=(3, 5, 7, 11),
            world_size=world_size,
            rank=rank,
            seed=13,
            macro_step=9,
            gram_chunk_elements=2,
        )
        torch.save(
            {"direction": direction, "metrics": metrics},
            Path(output_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_bounded_cp_composition_matches_reference_across_two_ranks(
    tmp_path: Path,
) -> None:
    initialization_file = tmp_path / "gloo_init"
    mp.spawn(
        _distributed_cp_worker,
        args=(2, str(initialization_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    outputs = [
        torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    sorted_ids = torch.tensor([3, 5, 7, 11])
    sorted_gradients = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.5, 0.0, -1.0, 0.5],
            [-1.0, 1.0, 0.0, 0.5],
            [0.0, -1.0, 1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    parameter = torch.nn.Parameter(torch.zeros(4))
    reference, reference_metrics = compose_conflict_projected_gradient(
        sorted_ids,
        sorted_gradients,
        _layout(parameter),
        seed=13,
        macro_step=9,
    )
    for output in outputs:
        assert torch.allclose(
            output["direction"], reference, atol=1e-7, rtol=1e-7
        )
        assert output["metrics"]["raw_gradient_gram"] == reference_metrics[
            "raw_gradient_gram"
        ]
        assert output["metrics"]["gradient_gram_chunk_allgathers"] == 2
        assert output["metrics"]["gradient_weight_broadcasts"] == 1
        assert output["metrics"]["gradient_weight_authority_rank"] == 0
        assert output["metrics"]["gathered_task_ids"] == [7, 3, 11, 5]


def _distributed_no_conflict_worker(
    rank: int,
    world_size: int,
    initialization_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{initialization_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        parameter = torch.nn.Parameter(torch.zeros(4))
        task_ids = (
            torch.tensor([7, 3]) if rank == 0 else torch.tensor([11, 5])
        )
        gradients_by_task = {
            3: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            5: torch.tensor([0.0, 1.0, 0.0, 0.0]),
            7: torch.tensor([0.0, 0.0, 1.0, 0.0]),
            11: torch.tensor([0.0, 0.0, 0.0, 1.0]),
        }
        gradients = torch.stack(
            [gradients_by_task[int(task_id)] for task_id in task_ids]
        )
        direction, metrics = compose_distributed_conflict_projected_gradient(
            task_ids,
            gradients,
            _layout(parameter),
            expected_task_ids=(3, 5, 7, 11),
            world_size=world_size,
            rank=rank,
            seed=19,
            macro_step=4,
            gram_chunk_elements=2,
        )
        torch.save(
            {"direction": direction, "metrics": metrics},
            Path(output_dir) / f"no_conflict_rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_distributed_no_conflict_is_exact_mean_with_permuted_rank_tasks(
    tmp_path: Path,
) -> None:
    initialization_file = tmp_path / "gloo_no_conflict_init"
    mp.spawn(
        _distributed_no_conflict_worker,
        args=(2, str(initialization_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        output = torch.load(
            tmp_path / f"no_conflict_rank_{rank}.pt", weights_only=True
        )
        assert torch.equal(output["direction"], torch.full((4,), 0.25))
        assert output["metrics"]["no_conflict_exact_raw_mean"] is True
        assert output["metrics"]["projection_count"] == 0
        assert output["metrics"]["gradient_weight_broadcasts"] == 1
