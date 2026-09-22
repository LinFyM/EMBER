"""Outcome-independent execution planning for shared Writer task batches.

The sampler owns the twelve conditions in each update. Every condition executes
whole on one rank; four ranks receive three conditions and six receive two.
These plans only describe placement. Learned Action Meta responses cannot be
reused across optimizer steps merely because a cache plan is available.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Mapping, Sequence

from ember.writer.learning_data import TASKS_PER_UPDATE


RankTasks = tuple[tuple[int, ...], ...]
MAX_EXACT_ASSIGNMENT_COMBINATIONS = 100_000


def condition_rank_groups(world_size: int) -> RankTasks:
    """A complete condition has one rank; no frame/query process subgroups."""
    if 1 <= world_size <= 6:
        return tuple((rank,) for rank in range(world_size))
    raise ValueError("condition execution supports one through six useful ranks")


def condition_assignment(jobs, costs, *, world_size):
    """Cost-balance complete conditions with equal counts when divisible."""
    condition_rank_groups(world_size)
    ordered = sorted(jobs, key=lambda job: (-costs[job], job))
    assigned, loads = [[] for _ in range(world_size)], [0] * world_size
    for start in range(0, len(ordered), world_size):
        ranks = sorted(range(world_size), key=lambda rank: (loads[rank], rank))
        for rank, job in zip(ranks, ordered[start:start + world_size]):
            assigned[rank].append(job)
            loads[rank] += costs[job]
    return tuple(tuple(group) for group in assigned)


def merge_condition_rows(rows: Sequence[dict], *, main_queries: int, teaching_queries: Sequence[int]) -> list[dict]:
    """Validate one complete, equally weighted exposure row per condition."""
    if (len(rows) != TASKS_PER_UPDATE or sorted(row["job_id"] for row in rows) != list(range(TASKS_PER_UPDATE))
            or len({row["task"] for row in rows}) != TASKS_PER_UPDATE):
        raise ValueError("an update must record exactly twelve distinct complete conditions")
    merged = sorted(rows, key=lambda row: row["job_id"])
    for row, teaching_count in zip(merged, teaching_queries, strict=True):
        if (row["query_offset"] != 0 or row["teaching_query_offset"] != 0
                or row["queries"] != main_queries or row["teaching_queries"] != teaching_count
                or not math.isclose(row["condition_weight"], 1 / TASKS_PER_UPDATE)
                or not math.isclose(row["task_weight"], 1 / TASKS_PER_UPDATE)
                or not math.isclose(row["teaching_weight"], 1 / (3 * TASKS_PER_UPDATE))):
            raise ValueError("condition exposure or equal task weighting changed")
    return [dict(row) for row in merged]


def _normalized_execution_ranks(
    execution_ranks: Mapping[int, Sequence[int]], world_size: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not 1 <= int(world_size) <= 6 or not execution_ranks:
        raise ValueError("shared Writer execution topology changed")
    result = []
    for task, ranks in sorted(execution_ranks.items()):
        normalized = tuple(sorted(set(map(int, ranks))))
        if (
            int(task) < 0
            or not normalized
            or len(normalized) != len(tuple(ranks))
            or min(normalized) < 0
            or max(normalized) >= int(world_size)
        ):
            raise ValueError("shared Writer execution rank eligibility changed")
        result.append((int(task), normalized))
    return tuple(result)


@lru_cache(maxsize=4096)
def _cached_cost_balanced_assignment(
    group: tuple[int, ...],
    costs: tuple[tuple[int, int], ...],
    eligibility: tuple[tuple[int, tuple[int, ...]], ...],
    world_size: int,
) -> RankTasks:
    cost_by_task = dict(costs)
    ranks_by_task = dict(eligibility)
    ordered = tuple(sorted(group, key=lambda task: (-cost_by_task[task], task)))

    # LPT provides a tight deterministic upper bound before exact branch and
    # bound.  Updates are small, so finding the exact minimum makespan costs
    # negligible relative to one frozen-policy forward.
    greedy_loads = [0] * world_size
    greedy_rank: dict[int, int] = {}
    for task in ordered:
        rank = min(
            ranks_by_task[task],
            key=lambda value: (greedy_loads[value], value),
        )
        greedy_rank[task] = rank
        greedy_loads[rank] += cost_by_task[task]

    def score(loads: Sequence[int], assigned: Mapping[int, int]) -> tuple[object, ...]:
        return (
            max(loads),
            sum(value * value for value in loads),
            tuple(assigned[task] for task in group),
        )

    best_rank = dict(greedy_rank)
    best_score = score(greedy_loads, best_rank)

    # A deterministic move/swap refinement keeps larger task batches cheap.
    # Small active batches continue into the exact search below.
    while True:
        improved_rank = best_rank
        improved_loads = greedy_loads
        improved_score = best_score
        for task in ordered:
            source = best_rank[task]
            cost = cost_by_task[task]
            for destination in ranks_by_task[task]:
                if destination == source:
                    continue
                candidate_loads = list(greedy_loads)
                candidate_loads[source] -= cost
                candidate_loads[destination] += cost
                candidate_rank = {**best_rank, task: destination}
                candidate_score = score(candidate_loads, candidate_rank)
                if candidate_score < improved_score:
                    improved_rank = candidate_rank
                    improved_loads = candidate_loads
                    improved_score = candidate_score
        for left_index, left in enumerate(ordered):
            left_rank = best_rank[left]
            for right in ordered[left_index + 1 :]:
                right_rank = best_rank[right]
                if (
                    left_rank == right_rank
                    or right_rank not in ranks_by_task[left]
                    or left_rank not in ranks_by_task[right]
                ):
                    continue
                candidate_loads = list(greedy_loads)
                candidate_loads[left_rank] += cost_by_task[right] - cost_by_task[left]
                candidate_loads[right_rank] += cost_by_task[left] - cost_by_task[right]
                candidate_rank = {
                    **best_rank,
                    left: right_rank,
                    right: left_rank,
                }
                candidate_score = score(candidate_loads, candidate_rank)
                if candidate_score < improved_score:
                    improved_rank = candidate_rank
                    improved_loads = candidate_loads
                    improved_score = candidate_score
        if improved_score >= best_score:
            break
        best_rank = improved_rank
        greedy_loads = improved_loads
        best_score = improved_score

    combinations = 1
    for task in ordered:
        combinations *= len(ranks_by_task[task])
        if combinations > MAX_EXACT_ASSIGNMENT_COMBINATIONS:
            rows = [[] for _ in range(world_size)]
            for task in group:
                rows[best_rank[task]].append(task)
            return tuple(tuple(row) for row in rows)

    loads = [0] * world_size
    assigned: dict[int, int] = {}
    seen: set[tuple[int, tuple[int, ...]]] = set()
    total_cost = sum(cost_by_task[task] for task in ordered)
    global_lower_bound = (total_cost + world_size - 1) // world_size

    def search(index: int) -> None:
        nonlocal best_rank, best_score
        state = (index, tuple(loads))
        if state in seen:
            return
        seen.add(state)
        if max(max(loads), global_lower_bound) > int(best_score[0]):
            return
        if index == len(ordered):
            candidate = score(loads, assigned)
            if candidate < best_score:
                best_score = candidate
                best_rank = dict(assigned)
            return
        task = ordered[index]
        cost = cost_by_task[task]
        for rank in sorted(
            ranks_by_task[task],
            key=lambda value: (loads[value] + cost, loads[value], value),
        ):
            updated = loads[rank] + cost
            if updated > int(best_score[0]):
                continue
            loads[rank] = updated
            assigned[task] = rank
            search(index + 1)
            del assigned[task]
            loads[rank] -= cost

    search(0)
    rows = [[] for _ in range(world_size)]
    for task in group:
        rows[best_rank[task]].append(task)
    return tuple(tuple(row) for row in rows)


def cost_balanced_task_assignment(
    group: Sequence[int],
    costs: Mapping[int, int],
    execution_ranks: Mapping[int, Sequence[int]],
    *,
    world_size: int,
) -> RankTasks:
    """Assign every task once using exact or bounded deterministic balancing."""

    tasks = tuple(map(int, group))
    normalized_costs = tuple(
        sorted((int(task), int(cost)) for task, cost in costs.items())
    )
    eligibility = _normalized_execution_ranks(execution_ranks, world_size)
    cost_by_task = dict(normalized_costs)
    ranks_by_task = dict(eligibility)
    if (
        not tasks
        or len(tasks) != len(set(tasks))
        or set(tasks) != set(cost_by_task)
        or not set(tasks) <= set(ranks_by_task)
        or min(cost_by_task.values(), default=0) <= 0
    ):
        raise ValueError("shared Writer execution task group changed")
    return _cached_cost_balanced_assignment(
        tasks, normalized_costs, eligibility, int(world_size)
    )
