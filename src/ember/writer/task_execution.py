"""Outcome-independent execution planning for shared Writer task batches.

The scientific sampler decides which tasks belong to an optimizer update.  This
module only decides where those already-selected tasks execute. Six ranks use
two groups of three, with disjoint native-frame and action-query work inside
each condition. A condition still contributes exactly once per update.
These plans only describe placement. Learned Action Meta responses cannot be
reused across optimizer steps merely because a cache plan is available.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Mapping, Sequence


RankTasks = tuple[tuple[int, ...], ...]
MAX_EXACT_ASSIGNMENT_COMBINATIONS = 100_000


def condition_rank_groups(world_size: int) -> RankTasks:
    """Keep four logical conditions while giving up to six ranks useful work."""
    if world_size == 6:
        return ((0, 1, 2), (3, 4, 5))
    if world_size == 5:
        return ((0, 1, 2), (3, 4))
    if 1 <= world_size <= 4:
        return tuple((rank,) for rank in range(world_size))
    raise ValueError("condition execution supports one through six useful ranks")


def initialize_condition_group(context):
    """Create subgroups in one global order after deferred NCCL is ready."""
    import torch.distributed as dist

    groups = condition_rank_groups(context.world_size)
    selected = None
    for ranks in groups:
        if len(ranks) > 1:
            group = dist.new_group(ranks=list(ranks))
            if context.rank in ranks:
                selected = group
    return selected


def query_shard(count: int, members: Sequence[int], rank: int) -> tuple[int, int]:
    """Return a contiguous slice of the unchanged logical policy RNG batch."""
    index = tuple(members).index(rank)
    size, remainder = divmod(count, len(members))
    if size <= 0:
        raise ValueError("query partition would leave an idle condition rank")
    return index * size + min(index, remainder), size + int(index < remainder)


def _combine_query_shards(parts: Sequence[dict]) -> dict:
    row = dict(parts[0])
    physical_keys = {key for key in row if key.endswith("_seconds") or key.startswith("input_cache_")}
    physical_keys.update(("seconds", "execution_rank", "query_offset", "queries",
                          "condition_weight", "fm_lora_gradient_norm", "policy_microbatch",
                          "teaching_query_offset", "teaching_queries", "teaching_weight", "teaching_lora_gradient_norm"))
    row["execution_shards"] = [{key: part[key] for key in sorted(physical_keys) if key in part}
                               for part in parts]
    for key in physical_keys:
        row.pop(key, None)
    row["seconds"] = max(part["seconds"] for part in parts)
    for key in ("action_demos", "action_frames", "action_start_indices",
                "teaching_action_demos", "teaching_action_frames", "teaching_action_start_indices"):
        if key in parts[0]:
            row[key] = [value for part in parts for value in part[key]]
    for key in ("source_forward_calls", "compiled_forward_calls", "teaching_compiled_forward_calls"):
        if key in parts[0]:
            row[key] = sum(part[key] for part in parts)
    return row


def merge_condition_rows(rows: Sequence[dict]) -> list[dict]:
    """Reassemble physical query slices into four scientific exposure rows."""
    by_job = defaultdict(list)
    for row in rows:
        by_job[row["job_id"]].append(row)
    if len(by_job) != 4:
        raise ValueError("an update must record exactly four logical conditions")
    merged = []
    for _, parts in sorted(by_job.items()):
        parts.sort(key=lambda row: row["query_offset"])
        cursor, teaching_cursor = 0, 0
        for part in parts:
            if part["query_offset"] != cursor:
                raise ValueError("condition query shards overlap or leave a gap")
            cursor += part["queries"]
            if part["teaching_query_offset"] != teaching_cursor:
                raise ValueError("condition teaching shards overlap or leave a gap")
            teaching_cursor += part["teaching_queries"]
        if cursor != 21 or teaching_cursor != 7:
            raise ValueError("condition query shards must cover all 21 main and seven teaching queries")
        row = _combine_query_shards(parts) if len(parts) > 1 else dict(parts[0])
        row.update(queries=cursor, query_offset=0, condition_weight=.25,
                   flow_loss=sum(part["flow_loss"] * part["queries"] for part in parts) / cursor,
                   teaching_queries=teaching_cursor, teaching_query_offset=0,
                   teaching_weight=sum(part["teaching_weight"] for part in parts),
                   teaching_loss=sum(part["teaching_loss"] * part["teaching_queries"] for part in parts) / teaching_cursor)
        merged.append(row)
    return merged


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
