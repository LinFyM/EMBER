"""Deterministic full-24 task-gradient composition for SPG training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist


GRADIENT_DIRECTION_SKETCH_DIMENSIONS = 32


class ConflictProjectionError(RuntimeError):
    """Raised when the CP-24 task-gradient contract is violated."""


@dataclass(frozen=True)
class FlatParameter:
    """One parameter's ownership and slice in a flat Writer gradient."""

    name: str
    parameter: torch.nn.Parameter
    start: int
    stop: int
    block: str


def _parameter_block(name: str) -> str:
    if name.startswith("semantic_encoder."):
        return "semantic_frontend"
    for prefix, block in (
        ("semantic_core.", "core"),
        ("semantic_program.", "program"),
        ("compiler.", "compiler"),
        ("factor_heads.", "factor"),
    ):
        if name.startswith(prefix):
            return block
    raise ConflictProjectionError(f"unowned SPG Writer parameter: {name}")


def parameter_layout(writer: torch.nn.Module) -> tuple[FlatParameter, ...]:
    """Build a stable flat layout for every trainable Writer parameter."""

    result = []
    cursor = 0
    for name, parameter in writer.named_parameters():
        if not parameter.requires_grad:
            continue
        stop = cursor + parameter.numel()
        result.append(
            FlatParameter(
                name=name,
                parameter=parameter,
                start=cursor,
                stop=stop,
                block=_parameter_block(name),
            )
        )
        cursor = stop
    if not result or len({item.name for item in result}) != len(result):
        raise ConflictProjectionError("invalid SPG Writer parameter layout")
    if {item.block for item in result} != {
        "semantic_frontend",
        "core",
        "program",
        "compiler",
        "factor",
    }:
        raise ConflictProjectionError("SPG Writer parameter ownership changed")
    return tuple(result)


def flatten_task_gradient(
    gradients: Sequence[torch.Tensor | None],
    layout: Sequence[FlatParameter],
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flatten one task gradient in parameter registration order."""

    if len(gradients) != len(layout):
        raise ConflictProjectionError("task gradient lost Writer parameters")
    if output is None:
        result = torch.zeros(
            layout[-1].stop,
            dtype=torch.float32,
            device=layout[0].parameter.device,
        )
    else:
        if (
            output.shape != (layout[-1].stop,)
            or output.dtype != torch.float32
            or output.device != layout[0].parameter.device
        ):
            raise ConflictProjectionError("invalid flat task-gradient destination")
        result = output
        result.zero_()
    for gradient, item in zip(gradients, layout, strict=True):
        if gradient is None:
            continue
        if gradient.shape != item.parameter.shape:
            raise ConflictProjectionError(
                f"task gradient shape changed for Writer parameter {item.name}"
            )
        result[item.start : item.stop].copy_(gradient.detach().reshape(-1))
    # One device synchronization per task is sufficient.  Checking every one
    # of the roughly 500 parameter tensors separately makes CP-24 thousands of
    # host/device round trips per macro without strengthening the contract.
    if not bool(torch.isfinite(result).all()):
        raise ConflictProjectionError("non-finite flattened task gradient")
    return result


def _validate_local_gradients(
    local_task_ids: torch.Tensor,
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    world_size: int,
) -> None:
    if (
        local_task_ids.ndim != 1
        or local_task_ids.dtype != torch.long
        or local_gradients.ndim != 2
        or local_gradients.shape
        != (local_task_ids.numel(), layout[-1].stop)
        or local_gradients.dtype != torch.float32
        or world_size <= 0
        or not bool(torch.isfinite(local_gradients).all())
    ):
        raise ConflictProjectionError("invalid local task-gradient matrix")


def _gather_task_ids(
    local_task_ids: torch.Tensor,
    *,
    expected_task_ids: Iterable[int],
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gathered IDs, their sorted order, and the sealed sorted IDs."""

    if world_size == 1:
        gathered = local_task_ids
    else:
        gathered = torch.empty(
            world_size * local_task_ids.numel(),
            dtype=torch.long,
            device=local_task_ids.device,
        )
        dist.all_gather_into_tensor(gathered, local_task_ids.contiguous())
    order = torch.argsort(gathered)
    sorted_ids = gathered.index_select(0, order)
    expected = torch.tensor(
        sorted(int(value) for value in expected_task_ids),
        dtype=torch.long,
        device=sorted_ids.device,
    )
    if not torch.equal(sorted_ids, expected):
        raise ConflictProjectionError("CP-24 task-gradient coverage changed")
    return gathered, order, sorted_ids


def _contiguous_block_ranges(
    layout: Sequence[FlatParameter],
) -> tuple[tuple[str, int, int], ...]:
    """Coalesce adjacent parameters while preserving explicit block ownership."""

    result: list[tuple[str, int, int]] = []
    for item in layout:
        if result and result[-1][0] == item.block and result[-1][2] == item.start:
            block, start, _ = result[-1]
            result[-1] = (block, start, item.stop)
        else:
            result.append((item.block, item.start, item.stop))
    observed = [block for block, _, _ in result]
    if len(observed) != len(set(observed)):
        raise ConflictProjectionError("SPG parameter blocks are not contiguous")
    return tuple(result)


def gradient_direction_sketches(
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    dimensions: int = GRADIENT_DIRECTION_SKETCH_DIMENSIONS,
) -> dict[str, torch.Tensor]:
    """Return fixed CountSketch projections for cross-macro direction QA.

    These small projections are diagnostics only: they never enter CP-24 or
    the optimizer.  Fixed coordinate hashes make the same task/module vectors
    comparable across successive one-video macro visits without retaining a
    full historical task-gradient matrix.
    """

    if (
        local_gradients.ndim != 2
        or local_gradients.shape[1] != layout[-1].stop
        or local_gradients.dtype != torch.float32
        or dimensions <= 0
    ):
        raise ConflictProjectionError("invalid task gradients for direction sketch")
    result: dict[str, torch.Tensor] = {}
    for ordinal, (block, start, stop) in enumerate(_contiguous_block_ranges(layout)):
        width = stop - start
        coordinates = torch.arange(
            width,
            dtype=torch.int64,
            device=local_gradients.device,
        )
        mixed = coordinates * 6_364_136_223_846_793_005
        mixed.add_((ordinal + 1) * 1_442_695_040_888_963_407)
        mixed.bitwise_xor_(torch.bitwise_right_shift(mixed, 33))
        buckets = torch.remainder(mixed, dimensions)
        signs = torch.where(
            torch.bitwise_and(torch.bitwise_right_shift(mixed, 17), 1) == 0,
            1.0,
            -1.0,
        ).to(dtype=torch.float32)
        sketch = torch.zeros(
            local_gradients.shape[0],
            dimensions,
            dtype=torch.float32,
            device=local_gradients.device,
        )
        sketch.scatter_add_(
            1,
            buckets.expand(local_gradients.shape[0], -1),
            local_gradients[:, start:stop] * signs,
        )
        sketch.mul_(math.sqrt(dimensions / width))
        if not bool(torch.isfinite(sketch).all()):
            raise ConflictProjectionError("non-finite task-gradient direction sketch")
        result[block] = sketch
    return result


def _distributed_gradient_grams(
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    world_size: int,
    sorted_order: torch.Tensor,
    chunk_elements: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Accumulate full and block Grams without materializing ``24 x P``.

    Each rank retains only its six task gradients.  A bounded parameter slice is
    exchanged at a time, so the largest collective buffer is
    ``global_tasks x chunk_elements`` rather than the full Writer gradient.
    """

    if chunk_elements <= 0:
        raise ConflictProjectionError("CP-24 Gram chunk must be positive")
    global_tasks = world_size * local_gradients.shape[0]
    gram = torch.zeros(
        global_tasks,
        global_tasks,
        dtype=torch.float32,
        device=local_gradients.device,
    )
    block_grams: dict[str, torch.Tensor] = {}
    for block, start, stop in _contiguous_block_ranges(layout):
        block_gram = torch.zeros_like(gram)
        for left in range(start, stop, chunk_elements):
            right = min(left + chunk_elements, stop)
            local_chunk = local_gradients[:, left:right].contiguous()
            if world_size == 1:
                gathered_chunk = local_chunk
            else:
                gathered_chunk = torch.empty(
                    global_tasks,
                    right - left,
                    dtype=torch.float32,
                    device=local_gradients.device,
                )
                dist.all_gather_into_tensor(gathered_chunk, local_chunk)
                # NCCL's synchronous Python API only guarantees that work is
                # enqueued on the CUDA stream.  Without an explicit device
                # completion boundary, faster ranks can queue every bounded
                # chunk while a co-scheduled rank has not entered the first
                # Gram exchange, leaving persistent NCCL kernels starved.  A
                # per-chunk stream wait makes the memory bound real and keeps
                # all ranks on the same collective under shared-GPU load.
                if local_gradients.is_cuda:
                    torch.cuda.current_stream(
                        local_gradients.device
                    ).synchronize()
            contribution = gathered_chunk @ gathered_chunk.T
            block_gram.add_(contribution)
            del contribution, gathered_chunk, local_chunk
        block_gram = block_gram.index_select(0, sorted_order).index_select(
            1, sorted_order
        )
        gram.add_(block_gram)
        block_grams[block] = block_gram
    if not bool(torch.isfinite(gram).all()):
        raise ConflictProjectionError("non-finite CP-24 Gradient Gram")
    return gram, block_grams


def _cosine_summary(gram: torch.Tensor) -> dict[str, Any]:
    count = gram.shape[0]
    diagonal = gram.diag().clamp_min(0.0)
    denominator = torch.sqrt(diagonal[:, None] * diagonal[None]).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    cosine = gram / denominator
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=gram.device)
    values = cosine[off_diagonal]
    return {
        "negative_pair_fraction": float((values < 0).to(torch.float32).mean()),
        "median_pair_cosine": float(values.median()),
        "minimum_pair_cosine": float(values.min()),
        "mean_gradient_norm": float(torch.sqrt(diagonal).mean()),
    }


def _projection_orders(
    task_ids: torch.Tensor,
    *,
    seed: int,
    macro_step: int,
) -> tuple[torch.Tensor, ...]:
    count = task_ids.numel()
    result = []
    for row, task_id in enumerate(task_ids.detach().cpu().tolist()):
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed)
            + (int(macro_step) + 1) * 1_000_003
            + (int(task_id) + 1) * 9_176
        )
        order = torch.randperm(count, generator=generator)
        order = order[order != row]
        result.append(order.to(task_ids.device))
    return tuple(result)


def _projection_weights_and_metrics(
    task_ids: torch.Tensor,
    gram: torch.Tensor,
    block_grams: Mapping[str, torch.Tensor],
    *,
    seed: int,
    macro_step: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve deterministic PCGrad entirely in the 24-task coefficient space."""

    if (
        task_ids.ndim != 1
        or gram.shape != (task_ids.numel(), task_ids.numel())
        or task_ids.numel() < 2
        or not block_grams
        or any(value.shape != gram.shape for value in block_grams.values())
    ):
        raise ConflictProjectionError("invalid CP-24 Gradient Gram")
    if not bool(torch.isfinite(gram).all()):
        raise ConflictProjectionError("non-finite CP-24 Gradient Gram")
    # PCGrad has a small sequential 24x24 control flow.  Run that control flow
    # on CPU after one bounded Gram transfer; otherwise each scalar branch is a
    # separate CUDA synchronization (hundreds per macro).  The only large
    # tensors, the task gradients, remain on device.
    result_device = gram.device
    task_ids = task_ids.detach().to(device="cpu")
    gram = gram.detach().to(device="cpu")
    block_grams = {
        name: value.detach().to(device="cpu")
        for name, value in block_grams.items()
    }
    count = task_ids.numel()
    coefficients = torch.eye(count, dtype=torch.float32)
    orders = _projection_orders(task_ids, seed=seed, macro_step=macro_step)
    applied = 0
    for row, order in enumerate(orders):
        for column in order.tolist():
            target_norm = gram[column, column]
            if float(target_norm) <= 0.0:
                continue
            dot = coefficients[row] @ gram[:, column]
            current_norm = coefficients[row] @ gram @ coefficients[row]
            tolerance = 1e-12 * torch.sqrt(
                current_norm.clamp_min(0.0) * target_norm
            )
            if bool(dot < -tolerance):
                coefficients[row, column] -= dot / target_norm
                applied += 1

    if applied == 0:
        weights = torch.full(
            (count,),
            1.0 / count,
            dtype=torch.float32,
            device=gram.device,
        )
        projected_gram = gram
    else:
        weights = coefficients.mean(dim=0)
        projected_gram = coefficients @ gram @ coefficients.T
    raw_weights = torch.full_like(weights, 1.0 / count)
    raw_norm = torch.sqrt((raw_weights @ gram @ raw_weights).clamp_min(0.0))
    projected_norm = torch.sqrt((weights @ gram @ weights).clamp_min(0.0))
    cosine = (raw_weights @ gram @ weights) / (
        raw_norm * projected_norm
    ).clamp_min(torch.finfo(torch.float32).tiny)
    raw_candidate_dots = gram.mean(dim=1)
    projected_candidate_dots = gram @ weights
    block_metrics: dict[str, Any] = {}
    for block, block_gram in sorted(block_grams.items()):
        projected_block_gram = coefficients @ block_gram @ coefficients.T
        block_metrics[block] = {
            "raw": _cosine_summary(block_gram),
            "projected": _cosine_summary(projected_block_gram),
            "raw_gram": block_gram.tolist(),
            "projected_gram": projected_block_gram.tolist(),
        }
    metrics = {
        "schema_version": "ember_spg_cp24_gradient_composition_v1",
        "task_ids": task_ids.detach().cpu().tolist(),
        "raw_gradient_gram": gram.detach().cpu().tolist(),
        "projected_gradient_gram": projected_gram.detach().cpu().tolist(),
        "raw": _cosine_summary(gram),
        "projected": _cosine_summary(projected_gram),
        "projection_count": applied,
        "raw_mean_gradient_norm": float(raw_norm),
        "projected_gradient_norm": float(projected_norm),
        "projected_to_raw_norm_ratio": float(
            projected_norm / raw_norm.clamp_min(torch.finfo(torch.float32).tiny)
        ),
        "raw_projected_cosine": float(cosine),
        "raw_candidate_negative_tasks": int((raw_candidate_dots < 0).sum()),
        "projected_candidate_negative_tasks": int(
            (projected_candidate_dots < 0).sum()
        ),
        "raw_candidate_task_dots": raw_candidate_dots.tolist(),
        "projected_candidate_task_dots": projected_candidate_dots.tolist(),
        "no_conflict_exact_raw_mean": applied == 0,
        "coefficient_l1_change": float(
            (coefficients - torch.eye(count, device=gram.device)).abs().sum()
        ),
        "blocks": block_metrics,
    }
    return weights.to(device=result_device), metrics


def compose_conflict_projected_gradient(
    task_ids: torch.Tensor,
    task_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    seed: int,
    macro_step: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reference in-memory composition used by focused unit tests."""

    if (
        task_ids.ndim != 1
        or task_gradients.ndim != 2
        or task_gradients.shape[0] != task_ids.numel()
        or task_gradients.shape[1] != layout[-1].stop
        or task_gradients.dtype != torch.float32
    ):
        raise ConflictProjectionError("invalid global task-gradient matrix")
    gram = task_gradients @ task_gradients.T
    block_grams = {
        block: task_gradients[:, start:stop]
        @ task_gradients[:, start:stop].T
        for block, start, stop in _contiguous_block_ranges(layout)
    }
    weights, metrics = _projection_weights_and_metrics(
        task_ids,
        gram,
        block_grams,
        seed=seed,
        macro_step=macro_step,
    )
    direction = (
        task_gradients.mean(dim=0)
        if metrics["no_conflict_exact_raw_mean"]
        else weights @ task_gradients
    )
    if not bool(torch.isfinite(direction).all()):
        raise ConflictProjectionError("non-finite projected Writer gradient")
    return direction, metrics


def compose_distributed_conflict_projected_gradient(
    local_task_ids: torch.Tensor,
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    expected_task_ids: Iterable[int],
    world_size: int,
    rank: int,
    seed: int,
    macro_step: int,
    gram_chunk_elements: int = 1_048_576,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose CP-24 with bounded collectives and one final direction reduce."""

    _validate_local_gradients(local_task_ids, local_gradients, layout, world_size)
    if not 0 <= rank < world_size:
        raise ConflictProjectionError("invalid distributed CP-24 rank")
    gathered_ids, order, sorted_ids = _gather_task_ids(
        local_task_ids,
        expected_task_ids=expected_task_ids,
        world_size=world_size,
    )
    gram, block_grams = _distributed_gradient_grams(
        local_gradients,
        layout,
        world_size=world_size,
        sorted_order=order,
        chunk_elements=gram_chunk_elements,
    )
    sorted_weights, metrics = _projection_weights_and_metrics(
        sorted_ids,
        gram,
        block_grams,
        seed=seed,
        macro_step=macro_step,
    )
    if world_size > 1:
        # Gram accumulation is numerically equivalent on all ranks, but the
        # sequential PCGrad zero-crossing branches must have one authority.
        # Rank 0's tiny coefficient vector therefore defines the direction.
        dist.broadcast(sorted_weights, src=0)
    if not bool(torch.isfinite(sorted_weights).all()):
        raise ConflictProjectionError("non-finite broadcast CP-24 weights")
    gathered_weights = torch.empty_like(sorted_weights)
    gathered_weights[order] = sorted_weights
    local_count = local_task_ids.numel()
    local_weights = gathered_weights[
        rank * local_count : (rank + 1) * local_count
    ]
    direction = local_weights @ local_gradients
    if world_size > 1:
        dist.all_reduce(direction, op=dist.ReduceOp.SUM)
    if not bool(torch.isfinite(direction).all()):
        raise ConflictProjectionError("non-finite projected Writer gradient")
    metrics["gram_chunk_elements"] = int(gram_chunk_elements)
    metrics["gradient_gram_chunk_allgathers"] = (
        sum(
            (stop - start + gram_chunk_elements - 1) // gram_chunk_elements
            for _, start, stop in _contiguous_block_ranges(layout)
        )
        if world_size > 1
        else 0
    )
    metrics["gradient_gram_chunk_cuda_synchronizations"] = (
        metrics["gradient_gram_chunk_allgathers"]
        if local_gradients.is_cuda
        else 0
    )
    metrics["gradient_task_id_allgathers"] = 1 if world_size > 1 else 0
    metrics["gradient_direction_allreduces"] = 1 if world_size > 1 else 0
    metrics["gradient_weight_broadcasts"] = 1 if world_size > 1 else 0
    metrics["gradient_weight_authority_rank"] = 0
    metrics["gradient_collectives"] = (
        metrics["gradient_task_id_allgathers"]
        + metrics["gradient_gram_chunk_allgathers"]
        + metrics["gradient_weight_broadcasts"]
        + metrics["gradient_direction_allreduces"]
    )
    metrics["distributed_full_gradient_materialized"] = False
    metrics["gathered_task_ids"] = gathered_ids.detach().cpu().tolist()
    return direction, metrics


def assign_flat_gradient(
    direction: torch.Tensor,
    layout: Sequence[FlatParameter],
) -> None:
    """Install one composed flat gradient on the physical Writer parameters."""

    if direction.ndim != 1 or direction.numel() != layout[-1].stop:
        raise ConflictProjectionError("composed gradient shape changed")
    for item in layout:
        value = direction[item.start : item.stop].reshape_as(item.parameter)
        if item.parameter.grad is None:
            item.parameter.grad = value.to(item.parameter.dtype).clone()
        else:
            item.parameter.grad.copy_(value)


def synchronize_writer_state(writer: torch.nn.Module, world_size: int) -> None:
    """Give every CP-24 rank the exact rank-zero Writer state once at startup."""

    if world_size <= 0:
        raise ConflictProjectionError("invalid CP-24 synchronization world size")
    if world_size == 1:
        return
    with torch.no_grad():
        for parameter in writer.parameters():
            dist.broadcast(parameter, src=0)
        for buffer in writer.buffers():
            dist.broadcast(buffer, src=0)
