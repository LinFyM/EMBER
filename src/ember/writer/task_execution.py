"""Outcome-independent execution planning for shared Writer task batches.

The scientific sampler decides which tasks belong to an optimizer update.  This
module only decides where those already-selected tasks execute.  A task may be
available on more than one rank when its frozen CPU evidence cache is
selectively replicated; it is still executed exactly once per update.
These plans only describe placement. Learned Action Meta responses cannot be
reused across optimizer steps merely because a cache plan is available.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Mapping, Sequence


RankTasks = tuple[tuple[int, ...], ...]
StepCosts = tuple[tuple[int, int], ...]
MAX_EXACT_ASSIGNMENT_COMBINATIONS = 100_000
MAX_EXACT_REPLICATION_CANDIDATES = 64


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


def assignment_makespan(
    assignment: Sequence[Sequence[int]], costs: Mapping[int, int]
) -> int:
    """Return the largest predicted rank load for one assignment."""

    tasks = tuple(task for row in assignment for task in row)
    if len(tasks) != len(set(tasks)) or set(tasks) != set(map(int, costs)):
        raise ValueError("shared Writer assignment coverage changed")
    return max(
        (sum(int(costs[task]) for task in row) for row in assignment),
        default=0,
    )


def _execution_objective(
    steps: Counter[StepCosts],
    execution_ranks: Mapping[int, Sequence[int]],
    world_size: int,
) -> tuple[int, int]:
    total = 0
    tail = 0
    for signature, count in steps.items():
        costs = dict(signature)
        if len(costs) <= world_size and all(
            len(execution_ranks[task]) == world_size for task in costs
        ):
            makespan = max(costs.values())
        else:
            assignment = cost_balanced_task_assignment(
                tuple(costs), costs, execution_ranks, world_size=world_size
            )
            makespan = assignment_makespan(assignment, costs)
        total += int(count) * makespan
        tail = max(tail, makespan)
    return total, tail


def _direct_move_replica_gains(
    steps: Counter[StepCosts],
    execution_ranks: Mapping[int, Sequence[int]],
    *,
    world_size: int,
) -> Counter[tuple[int, int]]:
    """Estimate all replica gains from feasible current-assignment moves."""

    gains: Counter[tuple[int, int]] = Counter()
    for signature, count in steps.items():
        costs = dict(signature)
        assignment = cost_balanced_task_assignment(
            tuple(costs), costs, execution_ranks, world_size=world_size
        )
        loads = [sum(costs[value] for value in row) for row in assignment]
        current = max(loads)
        for source, row in enumerate(assignment):
            for task in row:
                cost = costs[task]
                for destination in range(world_size):
                    if destination in execution_ranks[task]:
                        continue
                    moved = list(loads)
                    moved[source] -= cost
                    moved[destination] += cost
                    gain = current - max(moved)
                    if gain > 0:
                        gains[(task, destination)] += int(count) * gain
    return gains


def selective_replication_plan(
    step_costs: Sequence[Mapping[int, int]],
    *,
    base_task_owners: Sequence[Sequence[int]],
    cache_bytes: Mapping[int, int],
    extra_budget_bytes: int,
) -> dict[str, object]:
    """Greedily buy only replicas that reduce finite-run predicted makespan.

    The target is the same objective attainable if every selected task were
    cached on every rank.  The planner stops at that target, when no replica
    helps, or when the launch-specific host-memory budget is exhausted.
    """

    owners = tuple(tuple(map(int, row)) for row in base_task_owners)
    world_size = len(owners)
    owner_by_task = {task: rank for rank, row in enumerate(owners) for task in row}
    normalized_steps: Counter[StepCosts] = Counter(
        tuple(sorted((int(task), int(cost)) for task, cost in row.items()))
        for row in step_costs
    )
    active_tasks = set(task for row in normalized_steps for task, _ in row)
    supplied_sizes = {int(task): int(value) for task, value in cache_bytes.items()}
    sizes = {
        task: supplied_sizes[task] for task in active_tasks if task in supplied_sizes
    }
    if (
        not normalized_steps
        or not 1 <= world_size <= 6
        or len(owner_by_task) != sum(len(row) for row in owners)
        or not active_tasks <= set(owner_by_task)
        or set(sizes) != active_tasks
        or min(sizes.values(), default=0) <= 0
        or int(extra_budget_bytes) < 0
        or any(cost <= 0 for row in normalized_steps for _, cost in row)
    ):
        raise ValueError("shared Writer selective cache replication changed")

    execution = {task: (owner_by_task[task],) for task in sorted(active_tasks)}
    ideal = {task: tuple(range(world_size)) for task in sorted(active_tasks)}
    objective = _execution_objective(normalized_steps, execution, world_size)
    base_objective = objective
    ideal_objective = _execution_objective(normalized_steps, ideal, world_size)
    budget = int(extra_budget_bytes)
    used = 0
    replicas: list[tuple[int, int]] = []

    candidate_count = len(active_tasks) * (world_size - 1)
    exact_candidate_search = candidate_count <= MAX_EXACT_REPLICATION_CANDIDATES
    while objective > ideal_objective:
        if exact_candidate_search:
            best_exact: (
                tuple[
                    tuple[int, int],
                    int,
                    int,
                    dict[int, tuple[int, ...]],
                ]
                | None
            ) = None
            for task in sorted(active_tasks):
                size = sizes[task]
                if used + size > budget:
                    continue
                for rank in range(world_size):
                    if rank in execution[task]:
                        continue
                    candidate_execution = dict(execution)
                    candidate_execution[task] = tuple(sorted((*execution[task], rank)))
                    candidate_objective = _execution_objective(
                        normalized_steps, candidate_execution, world_size
                    )
                    if candidate_objective >= objective:
                        continue
                    candidate = (
                        candidate_objective,
                        size,
                        task * world_size + rank,
                        candidate_execution,
                    )
                    if best_exact is None or candidate[:3] < best_exact[:3]:
                        best_exact = candidate
            if best_exact is None:
                break
            objective, size, encoded, execution = best_exact
        else:
            # Full objective solves for every candidate scale quadratically in
            # the task inventory. Rank candidates by weighted makespan gain per
            # cache byte, then solve the exact finite objective only once for
            # the selected eligibility expansion.
            best_heuristic: tuple[int, int, int] | None = None
            direct_gains = _direct_move_replica_gains(
                normalized_steps, execution, world_size=world_size
            )
            for task in sorted(active_tasks):
                size = sizes[task]
                if used + size > budget:
                    continue
                for rank in range(world_size):
                    if rank in execution[task]:
                        continue
                    gain = direct_gains[(task, rank)]
                    if gain <= 0:
                        continue
                    encoded = task * world_size + rank
                    candidate = (gain, size, encoded)
                    if best_heuristic is None:
                        best_heuristic = candidate
                        continue
                    best_gain, best_size, best_encoded = best_heuristic
                    if gain * best_size > best_gain * size or (
                        gain * best_size == best_gain * size
                        and (gain, -size, -encoded)
                        > (best_gain, -best_size, -best_encoded)
                    ):
                        best_heuristic = candidate
            if best_heuristic is None:
                break
            _, size, encoded = best_heuristic
            task, rank = divmod(encoded, world_size)
            candidate_execution = dict(execution)
            candidate_execution[task] = tuple(sorted((*execution[task], rank)))
            candidate_objective = _execution_objective(
                normalized_steps, candidate_execution, world_size
            )
            if candidate_objective >= objective:
                break
            execution = candidate_execution
            objective = candidate_objective

        task, rank = divmod(encoded, world_size)
        used += size
        replicas.append((task, rank))

    rows = tuple(
        tuple(sorted(task for task, ranks in execution.items() if rank in ranks))
        for rank in range(world_size)
    )
    return {
        "strategy": "finite_schedule_selective_cache_replication_cost_balanced_assignment",
        "execution_ownership": rows,
        "replicas": tuple(replicas),
        "extra_cache_bytes": used,
        "budget_bytes": budget,
        "base_total_cost": base_objective[0],
        "base_tail_cost": base_objective[1],
        "predicted_total_cost": objective[0],
        "predicted_tail_cost": objective[1],
        "ideal_total_cost": ideal_objective[0],
        "ideal_tail_cost": ideal_objective[1],
        "unique_step_signatures": len(normalized_steps),
        "planned_steps": sum(normalized_steps.values()),
        "replica_search": (
            "exact_candidate_objective"
            if exact_candidate_search
            else "direct_move_gain_per_byte_then_exact_objective"
        ),
    }


def shared_mmap_execution_plan(
    step_costs: Sequence[Mapping[int, int]],
    *,
    cache_bytes: Mapping[int, int],
    world_size: int,
) -> dict[str, object]:
    """Plan exact scheduling when every rank maps the same physical cache."""

    normalized_steps: Counter[StepCosts] = Counter(
        tuple(sorted((int(task), int(cost)) for task, cost in row.items()))
        for row in step_costs
    )
    active_tasks = set(task for row in normalized_steps for task, _ in row)
    supplied_sizes = {int(task): int(value) for task, value in cache_bytes.items()}
    sizes = {
        task: supplied_sizes[task]
        for task in active_tasks
        if task in supplied_sizes
    }
    if (
        not normalized_steps
        or not 1 <= int(world_size) <= 6
        or set(sizes) != active_tasks
        or min(sizes.values(), default=0) <= 0
        or any(cost <= 0 for row in normalized_steps for _, cost in row)
    ):
        raise ValueError("shared Writer mmap execution plan changed")
    execution = {
        task: tuple(range(int(world_size))) for task in sorted(active_tasks)
    }
    objective = _execution_objective(normalized_steps, execution, int(world_size))
    rows = tuple(tuple(sorted(active_tasks)) for _ in range(int(world_size)))
    return {
        "strategy": "node_local_single_copy_mmap_cost_balanced_assignment",
        "execution_ownership": rows,
        "replicas": (),
        "extra_cache_bytes": 0,
        "shared_cache_bytes": sum(sizes.values()),
        "budget_bytes": 0,
        "base_total_cost": objective[0],
        "base_tail_cost": objective[1],
        "predicted_total_cost": objective[0],
        "predicted_tail_cost": objective[1],
        "ideal_total_cost": objective[0],
        "ideal_tail_cost": objective[1],
        "unique_step_signatures": len(normalized_steps),
        "planned_steps": sum(normalized_steps.values()),
        "replica_search": "not_applicable_shared_mmap",
    }
