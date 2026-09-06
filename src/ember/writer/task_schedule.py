"""Task-group and video sampling independent of device placement."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence


def task_group_counts(
    cell: Mapping[str, Any],
    *,
    meta: Sequence[int],
    target: Sequence[int],
) -> tuple[int, int]:
    """Resolve experiment-configured role counts without imposing owner policy."""

    total = int(cell["global_tasks_per_update"])
    configured = cell.get("tasks_per_update_by_role")
    if configured is None:
        if total <= 0 or total % 2:
            raise ValueError(
                "shared Writer unequal role sampling needs tasks_per_update_by_role"
            )
        counts = (total // 2, total // 2)
    else:
        if set(configured) != {"meta", "target"}:
            raise ValueError("shared Writer task sampling roles changed")
        counts = (int(configured["meta"]), int(configured["target"]))
    if (
        total <= 0
        or min(counts) < 0
        or sum(counts) != total
        or counts[0] > len(tuple(meta))
        or counts[1] > len(tuple(target))
        or not any(counts)
    ):
        raise ValueError("shared Writer task sampling counts changed")
    return counts


def counted_task_group(
    groups: Sequence[Sequence[int]],
    counts: Sequence[int],
    optimizer_step: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Cycle any configured task groups with independent counts and equal exposure."""

    normalized = tuple(tuple(map(int, group)) for group in groups)
    requested = tuple(map(int, counts))
    all_tasks = tuple(task for group in normalized for task in group)
    if (
        optimizer_step < 0
        or len(normalized) != len(requested)
        or not normalized
        or len(all_tasks) != len(set(all_tasks))
        or any(
            count < 0 or count > len(group)
            for group, count in zip(normalized, requested, strict=True)
        )
        or not any(requested)
    ):
        raise ValueError("shared Writer counted task schedule changed")
    selected: list[int] = []
    for group_index, (group, count) in enumerate(
        zip(normalized, requested, strict=True)
    ):
        if count == 0:
            continue
        ordered = list(group)
        random.Random(int(seed) + 104729 + group_index * 25609).shuffle(ordered)
        offset = (optimizer_step * count) % len(ordered)
        selected.extend(
            ordered[(offset + index) % len(ordered)] for index in range(count)
        )
    return tuple(selected)


def task_occurrence_schedule(
    groups: Sequence[Sequence[int]],
) -> tuple[dict[int, int], ...]:
    """Give every scheduled task its own zero-based deterministic data cursor."""

    counts: dict[int, int] = {}
    result = []
    for group in groups:
        normalized = tuple(map(int, group))
        if len(normalized) != len(set(normalized)):
            raise ValueError("shared Writer task group repeated a task")
        result.append({task: counts.get(task, 0) for task in normalized})
        for task in normalized:
            counts[task] = counts.get(task, 0) + 1
    return tuple(result)


def training_video_demos(
    fit: Sequence[int],
    *,
    task_occurrence: int,
    task: int,
    cardinalities: Sequence[int],
    seed: int,
) -> tuple[int, ...]:
    """Choose a deterministic K-set from this task's own update cursor."""

    pool = tuple(map(int, fit))
    requested = tuple(map(int, cardinalities))
    allowed = tuple(value for value in requested if value <= len(pool))
    if (
        task_occurrence < 0
        or not pool
        or len(pool) != len(set(pool))
        or not requested
        or tuple(sorted(set(requested))) != requested
        or any(value not in {1, 2, 4} for value in requested)
        or not allowed
    ):
        raise ValueError("shared Writer K-video schedule changed")
    cardinality = allowed[(task_occurrence + int(task) + int(seed)) % len(allowed)]
    offset = (task_occurrence * max(allowed) + int(task) + int(seed)) % len(pool)
    return tuple(pool[(offset + index) % len(pool)] for index in range(cardinality))


def balanced_task_owners(
    costs: Mapping[int, int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    """Give every task one stable evidence-cache owner."""

    if not costs or not 1 <= world_size <= 6 or min(map(int, costs.values())) <= 0:
        raise ValueError("shared Writer task ownership changed")
    rows: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for task, cost in sorted(costs.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        rows[rank].append(int(task))
        loads[rank] += int(cost)
    return tuple(tuple(sorted(row)) for row in rows)
