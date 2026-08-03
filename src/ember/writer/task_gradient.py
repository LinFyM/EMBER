"""Raw task-balanced Writer gradients with read-only coherence diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist


GRADIENT_DIRECTION_SKETCH_DIMENSIONS = 32


class TaskGradientError(RuntimeError):
    """Raised when the task-gradient collection contract is violated."""


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
        ("direction_router.", "factor"),
        ("factor_heads.", "factor"),
    ):
        if name.startswith(prefix):
            return block
    raise TaskGradientError(f"unowned Writer parameter: {name}")


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
        raise TaskGradientError("invalid Writer parameter layout")
    if {item.block for item in result} != {
        "semantic_frontend",
        "core",
        "program",
        "compiler",
        "factor",
    }:
        raise TaskGradientError("Writer parameter ownership changed")
    return tuple(result)


def flatten_task_gradient(
    gradients: Sequence[torch.Tensor | None],
    layout: Sequence[FlatParameter],
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flatten one task gradient in parameter registration order."""

    if len(gradients) != len(layout):
        raise TaskGradientError("task gradient lost Writer parameters")
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
            raise TaskGradientError("invalid flat task-gradient destination")
        result = output
        result.zero_()
    for gradient, item in zip(gradients, layout, strict=True):
        if gradient is None:
            continue
        if gradient.shape != item.parameter.shape:
            raise TaskGradientError(
                f"task gradient shape changed for Writer parameter {item.name}"
            )
        result[item.start : item.stop].copy_(gradient.detach().reshape(-1))
    if not bool(torch.isfinite(result).all()):
        raise TaskGradientError("non-finite flattened task gradient")
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
        or local_gradients.shape != (local_task_ids.numel(), layout[-1].stop)
        or local_gradients.dtype != torch.float32
        or world_size <= 0
        or not bool(torch.isfinite(local_gradients).all())
    ):
        raise TaskGradientError("invalid local task-gradient matrix")


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
        raise TaskGradientError("full-task gradient coverage changed")
    return gathered, order, sorted_ids


def _contiguous_block_ranges(
    layout: Sequence[FlatParameter],
) -> tuple[tuple[str, int, int], ...]:
    """Coalesce adjacent parameters while preserving explicit ownership."""

    result: list[tuple[str, int, int]] = []
    for item in layout:
        if result and result[-1][0] == item.block and result[-1][2] == item.start:
            block, start, _ = result[-1]
            result[-1] = (block, start, item.stop)
        else:
            result.append((item.block, item.start, item.stop))
    observed = [block for block, _, _ in result]
    if len(observed) != len(set(observed)):
        raise TaskGradientError("Writer parameter blocks are not contiguous")
    return tuple(result)


def gradient_direction_sketches(
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    dimensions: int = GRADIENT_DIRECTION_SKETCH_DIMENSIONS,
) -> dict[str, torch.Tensor]:
    """Return fixed CountSketch projections for cross-macro direction QA.

    These projections are diagnostics only. They never enter the optimizer
    direction. Fixed coordinate hashes keep task/module vectors comparable
    across successive one-video macro visits without retaining full history.
    """

    if (
        local_gradients.ndim != 2
        or local_gradients.shape[1] != layout[-1].stop
        or local_gradients.dtype != torch.float32
        or dimensions <= 0
    ):
        raise TaskGradientError("invalid task gradients for direction sketch")
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
            raise TaskGradientError("non-finite task-gradient direction sketch")
        result[block] = sketch
    return result


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


def _raw_metrics(
    task_ids: torch.Tensor,
    gram: torch.Tensor,
    block_grams: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize the raw task geometry without changing the update."""

    count = task_ids.numel()
    if (
        task_ids.ndim != 1
        or count < 2
        or gram.shape != (count, count)
        or not block_grams
        or any(value.shape != gram.shape for value in block_grams.values())
        or not bool(torch.isfinite(gram).all())
    ):
        raise TaskGradientError("invalid full-task Gradient Gram")

    # Metrics are small 24x24 tensors. Moving them to CPU avoids repeated CUDA
    # scalar synchronizations while leaving all full gradients on device.
    task_ids = task_ids.detach().to(device="cpu")
    gram = gram.detach().to(device="cpu")
    block_grams = {
        name: value.detach().to(device="cpu")
        for name, value in block_grams.items()
    }

    def energy_metrics(value: torch.Tensor) -> dict[str, Any]:
        mean_energy = value.mean().clamp_min(0.0)
        average_task_energy = value.diag().clamp_min(0.0).mean()
        ratio = (
            mean_energy / average_task_energy
            if float(average_task_energy) > 0.0
            else torch.zeros((), dtype=torch.float32)
        )
        candidate_dots = value.mean(dim=1)
        return {
            "raw": _cosine_summary(value),
            "raw_gram": value.tolist(),
            "raw_mean_gradient_norm": float(mean_energy.clamp_min(0.0).sqrt()),
            "raw_mean_gradient_energy": float(mean_energy),
            "average_task_gradient_energy": float(average_task_energy),
            "raw_mean_to_average_task_energy_ratio": float(ratio),
            "raw_candidate_negative_tasks": int((candidate_dots < 0).sum()),
            "raw_candidate_task_dots": candidate_dots.tolist(),
        }

    full = energy_metrics(gram)
    return {
        "schema_version": "ember_raw_full_task_gradient_v1",
        "task_ids": task_ids.tolist(),
        "raw_gradient_gram": full.pop("raw_gram"),
        **full,
        "blocks": {
            block: energy_metrics(block_gram)
            for block, block_gram in sorted(block_grams.items())
        },
    }


def compose_raw_mean_gradient(
    task_ids: torch.Tensor,
    task_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reference raw mean composition for focused unit tests."""

    if (
        task_ids.ndim != 1
        or task_ids.dtype != torch.long
        or task_gradients.ndim != 2
        or task_gradients.shape[0] != task_ids.numel()
        or task_gradients.shape[1] != layout[-1].stop
        or task_gradients.dtype != torch.float32
        or task_ids.unique().numel() != task_ids.numel()
        or not bool(torch.isfinite(task_gradients).all())
    ):
        raise TaskGradientError("invalid global task-gradient matrix")
    order = torch.argsort(task_ids)
    sorted_ids = task_ids.index_select(0, order)
    sorted_gradients = task_gradients.index_select(0, order)
    direction = sorted_gradients.mean(dim=0)
    gram = sorted_gradients @ sorted_gradients.T
    block_grams = {
        block: sorted_gradients[:, start:stop]
        @ sorted_gradients[:, start:stop].T
        for block, start, stop in _contiguous_block_ranges(layout)
    }
    if not bool(torch.isfinite(direction).all()):
        raise TaskGradientError("non-finite raw Writer gradient")
    return direction, _raw_metrics(sorted_ids, gram, block_grams)


def _distributed_grams_and_raw_mean(
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    world_size: int,
    sorted_order: torch.Tensor,
    chunk_elements: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], int]:
    """Compose sorted raw mean and Grams without materializing ``24 x P``."""

    if chunk_elements <= 0:
        raise TaskGradientError("Gradient Gram chunk must be positive")
    global_tasks = world_size * local_gradients.shape[0]
    direction = torch.empty(
        layout[-1].stop,
        dtype=torch.float32,
        device=local_gradients.device,
    )
    gram = torch.zeros(
        global_tasks,
        global_tasks,
        dtype=torch.float32,
        device=local_gradients.device,
    )
    block_grams: dict[str, torch.Tensor] = {}
    completed_collectives = 0
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
                if local_gradients.is_cuda:
                    torch.cuda.current_stream(local_gradients.device).synchronize()
                completed_collectives += 1
            sorted_chunk = gathered_chunk.index_select(0, sorted_order)
            direction[left:right].copy_(sorted_chunk.mean(dim=0))
            block_gram.add_(sorted_chunk @ sorted_chunk.T)
            del sorted_chunk, gathered_chunk, local_chunk
        gram.add_(block_gram)
        block_grams[block] = block_gram
    if not bool(torch.isfinite(gram).all()) or not bool(
        torch.isfinite(direction).all()
    ):
        raise TaskGradientError("non-finite raw task-gradient composition")
    return direction, gram, block_grams, completed_collectives


def compose_distributed_raw_mean_gradient(
    local_task_ids: torch.Tensor,
    local_gradients: torch.Tensor,
    layout: Sequence[FlatParameter],
    *,
    expected_task_ids: Iterable[int],
    world_size: int,
    rank: int,
    gram_chunk_elements: int = 1_048_576,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose the exact raw task mean with bounded diagnostic collectives."""

    _validate_local_gradients(local_task_ids, local_gradients, layout, world_size)
    if not 0 <= rank < world_size:
        raise TaskGradientError("invalid distributed task-gradient rank")
    gathered_ids, order, sorted_ids = _gather_task_ids(
        local_task_ids,
        expected_task_ids=expected_task_ids,
        world_size=world_size,
    )
    direction, gram, block_grams, completed = _distributed_grams_and_raw_mean(
        local_gradients,
        layout,
        world_size=world_size,
        sorted_order=order,
        chunk_elements=gram_chunk_elements,
    )
    metrics = _raw_metrics(sorted_ids, gram, block_grams)
    scheduled = (
        sum(
            (stop - start + gram_chunk_elements - 1) // gram_chunk_elements
            for _, start, stop in _contiguous_block_ranges(layout)
        )
        if world_size > 1
        else 0
    )
    if completed != scheduled:
        raise TaskGradientError("Gradient Gram collective completion changed")
    metrics["gram_chunk_elements"] = int(gram_chunk_elements)
    metrics["gradient_gram_chunk_allgathers"] = scheduled
    metrics["gradient_gram_chunk_collective_completions"] = completed
    metrics["gradient_gram_chunk_cuda_synchronizations"] = (
        completed if local_gradients.is_cuda else 0
    )
    metrics["gradient_task_id_allgathers"] = 1 if world_size > 1 else 0
    metrics["gradient_collectives"] = (
        metrics["gradient_task_id_allgathers"] + completed
    )
    metrics["distributed_full_gradient_materialized"] = False
    metrics["gathered_task_ids"] = gathered_ids.detach().cpu().tolist()
    return direction, metrics


def assign_flat_gradient(
    direction: torch.Tensor,
    layout: Sequence[FlatParameter],
) -> None:
    """Install one raw flat gradient on the physical Writer parameters."""

    if direction.ndim != 1 or direction.numel() != layout[-1].stop:
        raise TaskGradientError("composed gradient shape changed")
    if not bool(torch.isfinite(direction).all()):
        raise TaskGradientError("cannot assign non-finite Writer gradient")
    for item in layout:
        value = direction[item.start : item.stop].reshape_as(item.parameter)
        if item.parameter.grad is None:
            item.parameter.grad = value.to(item.parameter.dtype).clone()
        else:
            item.parameter.grad.copy_(value)


def synchronize_writer_state(writer: torch.nn.Module, world_size: int) -> None:
    """Give every task-complete rank the exact rank-zero Writer state."""

    if world_size <= 0:
        raise TaskGradientError("invalid Writer synchronization world size")
    if world_size == 1:
        return
    with torch.no_grad():
        for parameter in writer.parameters():
            dist.broadcast(parameter, src=0)
        for buffer in writer.buffers():
            dist.broadcast(buffer, src=0)
