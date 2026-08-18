"""Matched full24 task-gradient aggregation diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.distributed as dist

from ember.writer.errors import WriterModelError


def deterministic_pcgrad_full24(
    local_gradients: torch.Tensor,
    *,
    local_task_ids: Sequence[int],
    task_ids: Sequence[int],
    world_size: int,
    macro: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Project conflicting full24 task gradients in a fixed PCGrad order."""

    expected = tuple(sorted(int(task_id) for task_id in task_ids))
    local_ids = tuple(int(task_id) for task_id in local_task_ids)
    if (
        local_gradients.ndim != 2
        or local_gradients.shape[0] != len(local_ids)
        or len(expected) != 24
        or len(set(expected)) != 24
        or world_size <= 0
        or macro < 0
    ):
        raise WriterModelError("invalid deterministic PCGrad full24 input")

    id_shards: list[Any] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(id_shards, list(local_ids))
    else:
        id_shards[0] = list(local_ids)
    observed = [int(task_id) for shard in id_shards for task_id in shard]
    if (
        len(observed) != 24
        or len(set(observed)) != 24
        or sorted(observed) != list(expected)
    ):
        raise WriterModelError("deterministic PCGrad lost full24 task coverage")

    max_local_tasks = max(len(shard) for shard in id_shards)
    width = local_gradients.shape[1]
    padded = local_gradients.new_zeros((max_local_tasks, width))
    padded[: len(local_ids)].copy_(local_gradients)
    if world_size > 1:
        gathered = local_gradients.new_empty((world_size * max_local_tasks, width))
        dist.all_gather_into_tensor(gathered, padded)
    else:
        gathered = padded
    row_by_task = {
        int(task_id): rank * max_local_tasks + offset
        for rank, shard in enumerate(id_shards)
        for offset, task_id in enumerate(shard)
    }

    arithmetic_mean = local_gradients.new_zeros(width)
    reference_norms = {}
    for task_id in expected:
        reference = gathered[row_by_task[task_id]]
        arithmetic_mean.add_(reference)
        reference_norms[task_id] = float(torch.dot(reference, reference))
    arithmetic_mean.div_(24)

    projected_sum = local_gradients.new_zeros(width)
    local_projection_count = 0
    for row, task_id in enumerate(local_ids):
        projected = local_gradients[row].clone()
        other_ids = [candidate for candidate in expected if candidate != task_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            (
                int(seed) * 1_000_003
                + (macro + 1) * 97_409
                + (task_id + 1) * 65_537
            )
            % (2**63 - 1)
        )
        order = torch.randperm(len(other_ids), generator=generator).tolist()
        for index in order:
            reference_id = other_ids[index]
            denominator = reference_norms[reference_id]
            if denominator == 0.0:
                continue
            reference = gathered[row_by_task[reference_id]]
            dot = float(torch.dot(projected, reference))
            if dot < 0.0:
                projected.add_(reference, alpha=-dot / denominator)
                local_projection_count += 1
        projected_sum.add_(projected)

    if world_size > 1:
        dist.all_reduce(projected_sum, op=dist.ReduceOp.SUM)
        count = torch.tensor(
            local_projection_count,
            dtype=torch.long,
            device=local_gradients.device,
        )
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        projection_count = int(count)
    else:
        projection_count = local_projection_count
    projected_sum.div_(24)
    arithmetic_norm = float(torch.linalg.vector_norm(arithmetic_mean))
    projected_norm = float(torch.linalg.vector_norm(projected_sum))
    cosine = (
        float(torch.dot(arithmetic_mean, projected_sum))
        / (arithmetic_norm * projected_norm)
        if arithmetic_norm > 0.0 and projected_norm > 0.0
        else 0.0
    )
    return projected_sum, {
        "task_gradient_aggregation": "deterministic_pcgrad_v1",
        "pcgrad_projection_count": projection_count,
        "pcgrad_ordered_pair_count": 24 * 23,
        "pcgrad_arithmetic_mean_norm": arithmetic_norm,
        "pcgrad_aggregate_norm": projected_norm,
        "pcgrad_cosine_to_arithmetic_mean": cosine,
    }
