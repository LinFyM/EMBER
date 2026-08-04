from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ember.writer.task_gradient import (
    FlatParameter,
    assign_flat_gradient,
    compose_distributed_raw_mean_gradient,
    compose_raw_mean_gradient,
    gradient_direction_sketches,
    parameter_layout,
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


def _task_gradients() -> dict[int, torch.Tensor]:
    return {
        3: torch.tensor([1.0, 0.0, 0.0, 0.5]),
        5: torch.tensor([0.5, 0.0, -1.0, 0.5]),
        7: torch.tensor([-1.0, 1.0, 0.0, 0.5]),
        11: torch.tensor([0.0, -1.0, 1.0, 0.5]),
    }


def test_parameter_layout_owns_exact_cvadr_blocks() -> None:
    writer = torch.nn.Module()
    writer.semantic_encoder = torch.nn.Linear(1, 1)
    writer.semantic_core = torch.nn.Linear(1, 1)
    writer.visual_transition = torch.nn.Linear(1, 1)
    writer.procedure = torch.nn.Linear(1, 1)
    writer.compiler = torch.nn.Linear(1, 1)
    writer.factor_heads = torch.nn.Linear(1, 1)
    layout = parameter_layout(writer)
    assert {item.block for item in layout} == {
        "semantic_frontend",
        "core",
        "program",
        "compiler",
        "factor",
    }


def test_raw_mean_is_exact_under_task_order_permutation_and_assignable() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    by_task = _task_gradients()
    first_ids = torch.tensor([11, 3, 7, 5], dtype=torch.long)
    second_ids = torch.tensor([5, 7, 3, 11], dtype=torch.long)
    first, first_metrics = compose_raw_mean_gradient(
        first_ids,
        torch.stack([by_task[int(task_id)] for task_id in first_ids]),
        _layout(parameter),
    )
    second, second_metrics = compose_raw_mean_gradient(
        second_ids,
        torch.stack([by_task[int(task_id)] for task_id in second_ids]),
        _layout(parameter),
    )
    expected = torch.stack([by_task[task_id] for task_id in sorted(by_task)]).mean(0)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert first_metrics == second_metrics
    assert torch.isfinite(first).all()
    assert first_metrics["task_ids"] == [3, 5, 7, 11]
    assert first_metrics["raw_candidate_negative_tasks"] >= 0
    assert first_metrics["raw_mean_to_average_task_energy_ratio"] >= 0.0
    assert "projected_gradient_gram" not in first_metrics
    assert "projection_count" not in first_metrics
    assign_flat_gradient(first, _layout(parameter))
    assert torch.equal(parameter.grad, expected)


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


def test_bounded_one_rank_raw_mean_matches_sorted_reference() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    by_task = _task_gradients()
    local_ids = torch.tensor([11, 3, 7, 5], dtype=torch.long)
    local_gradients = torch.stack(
        [by_task[int(task_id)] for task_id in local_ids]
    )
    reference, reference_metrics = compose_raw_mean_gradient(
        local_ids,
        local_gradients,
        _layout(parameter),
    )
    bounded, bounded_metrics = compose_distributed_raw_mean_gradient(
        local_ids,
        local_gradients,
        _layout(parameter),
        expected_task_ids=(3, 5, 7, 11),
        world_size=1,
        rank=0,
        gram_chunk_elements=2,
    )
    assert torch.equal(bounded, reference)
    assert bounded_metrics["raw_gradient_gram"] == reference_metrics[
        "raw_gradient_gram"
    ]
    assert bounded_metrics["blocks"] == reference_metrics["blocks"]
    assert bounded_metrics["gradient_gram_chunk_allgathers"] == 0
    assert bounded_metrics["gradient_gram_chunk_collective_completions"] == 0
    assert bounded_metrics["gradient_gram_chunk_cuda_synchronizations"] == 0
    assert bounded_metrics["distributed_full_gradient_materialized"] is False


def _distributed_raw_worker(
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
        by_task = _task_gradients()
        assignments = (
            ((7, 3), (11, 5)),
            ((11, 5), (3, 7)),
        )
        outputs = []
        for assignment in assignments:
            task_ids = torch.tensor(assignment[rank], dtype=torch.long)
            gradients = torch.stack(
                [by_task[int(task_id)] for task_id in task_ids]
            )
            direction, metrics = compose_distributed_raw_mean_gradient(
                task_ids,
                gradients,
                _layout(parameter),
                expected_task_ids=(3, 5, 7, 11),
                world_size=world_size,
                rank=rank,
                gram_chunk_elements=2,
            )
            assign_flat_gradient(direction, _layout(parameter))
            outputs.append(
                {
                    "direction": direction,
                    "assigned": parameter.grad.clone(),
                    "metrics": metrics,
                }
            )
        torch.save(outputs, Path(output_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_raw_mean_is_exact_under_task_and_rank_permutation(
    tmp_path: Path,
) -> None:
    initialization_file = tmp_path / "gloo_init"
    mp.spawn(
        _distributed_raw_worker,
        args=(2, str(initialization_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    by_task = _task_gradients()
    expected = torch.stack([by_task[task_id] for task_id in sorted(by_task)]).mean(0)
    for rank in range(2):
        outputs = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        assert len(outputs) == 2
        assert torch.equal(outputs[0]["direction"], expected)
        assert torch.equal(outputs[1]["direction"], expected)
        assert torch.equal(outputs[0]["direction"], outputs[1]["direction"])
        assert torch.equal(outputs[0]["assigned"], expected)
        assert torch.equal(outputs[1]["assigned"], expected)
        assert torch.isfinite(outputs[0]["direction"]).all()
        assert outputs[0]["metrics"]["raw_gradient_gram"] == outputs[1][
            "metrics"
        ]["raw_gradient_gram"]
        assert outputs[0]["metrics"]["gradient_gram_chunk_allgathers"] == 2
        assert outputs[0]["metrics"][
            "gradient_gram_chunk_collective_completions"
        ] == 2
        assert outputs[0]["metrics"]["gradient_task_id_allgathers"] == 1
        assert outputs[0]["metrics"]["gradient_collectives"] == 3
        assert "gradient_weight_broadcasts" not in outputs[0]["metrics"]
        assert "gradient_direction_allreduces" not in outputs[0]["metrics"]
