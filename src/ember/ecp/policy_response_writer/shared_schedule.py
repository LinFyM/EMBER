"""Scalable task, video, and cache-owner schedules for shared Writer training."""

from __future__ import annotations

import random
from functools import lru_cache
from math import gcd, lcm
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.bank_conditioning.mapping import load_mapping_split
from ember.pi05_source_checkpoint import read_json


VideoSplit = tuple[tuple[int, ...], int]


def _selected_task_ids(config: Mapping[str, Any]) -> tuple[int, ...]:
    split = config["task_split"]
    groups = tuple(
        tuple(map(int, split[name]))
        for name in (
            "gradient_meta",
            "gradient_target",
            "true_task_held_meta",
            "true_task_held_target",
        )
    )
    values = tuple(task for group in groups for task in group)
    if (
        not all(groups)
        or len(values) != len(set(values))
        or min(values, default=-1) < 0
    ):
        raise ValueError("Policy-Response Writer task split changed")
    return values


def _functional_panel_config(
    config: Mapping[str, Any], *, asset_root: Path
) -> dict[str, Any]:
    """Resolve either one legacy panel config or explicit completed panel roots."""

    authorities = config["authorities"]
    relative = authorities.get("functional_panel_config")
    sources = tuple(authorities.get("functional_panel_sources", ()))
    if (relative is None) == (not sources):
        raise ValueError("Policy-Response Writer functional panel authority changed")
    if relative is not None:
        return read_json((asset_root / str(relative)).resolve())

    selected = set(_selected_task_ids(config))
    records: dict[int, Path] = {}
    for source in sources:
        root = (asset_root / str(source["root"])).resolve()
        completion = root / str(source["completion"])
        if (
            not root.is_dir()
            or not completion.is_file()
            or read_json(completion).get("status") != "complete"
        ):
            raise ValueError("Policy-Response Writer panel source is incomplete")
        candidates = tuple(sorted(root.glob("shard_*/task_*.json")))
        if int(source["task_count"]) != len(candidates):
            raise ValueError("Policy-Response Writer panel source task count changed")
        for path in candidates:
            try:
                task = int(path.stem.removeprefix("task_"))
            except ValueError as error:
                raise ValueError(
                    "Policy-Response Writer panel filename changed"
                ) from error
            if task not in selected:
                continue
            if task in records:
                raise ValueError("Policy-Response Writer panel sources overlap")
            records[task] = path
    if set(records) != selected:
        raise ValueError("Policy-Response Writer panel source lost a selected task")
    return {
        "authorities": {
            "functional_panel_records": {
                str(task): str(path) for task, path in sorted(records.items())
            }
        }
    }


def shared_task_group(
    meta: Sequence[int],
    target: Sequence[int],
    optimizer_step: int,
    *,
    tasks_per_role: int = 3,
    seed: int = 0,
) -> tuple[int, ...]:
    """Cycle a fixed role-balanced batch with equal frequency within each role."""

    left = tuple(map(int, meta))
    right = tuple(map(int, target))
    if (
        optimizer_step < 0
        or tasks_per_role <= 0
        or len(left) < tasks_per_role
        or len(right) < tasks_per_role
        or len(set((*left, *right))) != len(left) + len(right)
    ):
        raise ValueError("shared Writer task-role schedule changed")

    def selected(values: tuple[int, ...], salt: int) -> tuple[int, ...]:
        ordered = list(values)
        random.Random(int(seed) + salt).shuffle(ordered)
        offset = (optimizer_step * tasks_per_role) % len(ordered)
        return tuple(
            ordered[(offset + index) % len(ordered)]
            for index in range(tasks_per_role)
        )

    return (*selected(left, 104729), *selected(right, 130363))


def training_video_demos(
    fit: Sequence[int],
    *,
    optimizer_step: int,
    task: int,
    cardinalities: Sequence[int],
    seed: int,
) -> tuple[int, ...]:
    """Choose one deterministic K-set while retaining all supported K in training."""

    pool = tuple(map(int, fit))
    requested = tuple(map(int, cardinalities))
    allowed = tuple(value for value in requested if value <= len(pool))
    if (
        optimizer_step < 0
        or not pool
        or len(pool) != len(set(pool))
        or not requested
        or tuple(sorted(set(requested))) != requested
        or any(value not in {1, 2, 4} for value in requested)
        or not allowed
    ):
        raise ValueError("shared Writer K-video schedule changed")
    cardinality = allowed[(optimizer_step + int(task) + int(seed)) % len(allowed)]
    offset = (
        optimizer_step * max(allowed) + int(task) + int(seed)
    ) % len(pool)
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


def role_balanced_task_owners(
    costs: Mapping[int, int],
    *,
    meta: Sequence[int],
    target: Sequence[int],
    held: Sequence[int],
    world_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Balance each gradient role across cache owners, then place held diagnostics."""

    roles = (tuple(map(int, meta)), tuple(map(int, target)))
    held_tasks = tuple(map(int, held))
    tasks = tuple(task for role in roles for task in role) + held_tasks
    if (
        not tasks
        or len(tasks) != len(set(tasks))
        or set(tasks) != set(map(int, costs))
        or not 1 <= world_size <= 6
    ):
        raise ValueError("shared Writer role-balanced ownership changed")
    rows: list[list[int]] = [[] for _ in range(world_size)]
    total_loads = [0] * world_size
    if world_size < 6:
        shared_ranks = tuple(range(world_size))
        role_ranks = (shared_ranks, shared_ranks)
    else:
        middle = (world_size + 1) // 2
        role_ranks = (tuple(range(middle)), tuple(range(middle, world_size)))
    for role_index, role in enumerate(roles):
        eligible = role_ranks[role_index]
        base, remainder = divmod(len(role), len(eligible))
        capacities = {
            rank: base + (position >= len(eligible) - remainder)
            for position, rank in enumerate(eligible)
        }
        role_loads = {rank: 0 for rank in eligible}
        role_counts = {rank: 0 for rank in eligible}
        for task in sorted(role, key=lambda value: (-int(costs[value]), value)):
            rank = min(
                (
                    value
                    for value in eligible
                    if role_counts[value] < capacities[value]
                ),
                key=lambda value: (
                    role_loads[value],
                    role_counts[value],
                    total_loads[value],
                    value,
                ),
            )
            rows[rank].append(task)
            role_loads[rank] += int(costs[task])
            role_counts[rank] += 1
            total_loads[rank] += int(costs[task])
    for task in sorted(held_tasks, key=lambda value: (-int(costs[value]), value)):
        rank = min(range(world_size), key=lambda value: (total_loads[value], value))
        rows[rank].append(task)
        total_loads[rank] += int(costs[task])
    return tuple(tuple(sorted(row)) for row in rows)


@lru_cache(maxsize=32)
def _owner_spreading_phase_offsets(
    orders: tuple[tuple[int, ...], tuple[int, ...]],
    owners: tuple[tuple[int, ...], ...],
    tasks_per_role: int,
) -> tuple[int, int]:
    """Choose fixed role phases that reduce per-step cache-owner stragglers."""

    owner_by_task = {
        task: rank for rank, row in enumerate(owners) for task in row
    }
    period = lcm(
        *(
            len(order) // gcd(len(order), tasks_per_role)
            for order in orders
        )
    )
    best_score: tuple[int, int, int, int, int] | None = None
    best_phases = (0, 0)
    for left_phase in range(len(orders[0])):
        for right_phase in range(len(orders[1])):
            maximum_load = 0
            active_owner_visits = 0
            squared_load = 0
            for step in range(period):
                loads = [0] * len(owners)
                for order, phase in zip(
                    orders, (left_phase, right_phase), strict=True
                ):
                    offset = step * tasks_per_role + phase
                    for index in range(tasks_per_role):
                        task = order[(offset + index) % len(order)]
                        loads[owner_by_task[task]] += 1
                maximum_load = max(maximum_load, max(loads))
                active_owner_visits += sum(value > 0 for value in loads)
                squared_load += sum(value * value for value in loads)
            score = (
                -maximum_load,
                active_owner_visits,
                -squared_load,
                -left_phase,
                -right_phase,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_phases = (left_phase, right_phase)
    return best_phases


def owner_balanced_task_group(
    meta: Sequence[int],
    target: Sequence[int],
    optimizer_step: int,
    *,
    task_owners: Sequence[Sequence[int]],
    tasks_per_role: int,
    seed: int,
) -> tuple[int, ...]:
    """Select role-balanced tasks while spreading each update over cache owners."""

    owners = tuple(tuple(map(int, row)) for row in task_owners)
    owner_by_task = {
        task: rank for rank, row in enumerate(owners) for task in row
    }
    roles = (tuple(map(int, meta)), tuple(map(int, target)))
    if (
        optimizer_step < 0
        or tasks_per_role <= 0
        or not owners
        or len(owner_by_task) != sum(len(row) for row in owners)
        or any(not set(role) <= set(owner_by_task) for role in roles)
    ):
        raise ValueError("shared Writer owner-balanced schedule changed")

    role_orders: list[tuple[int, ...]] = []
    for role_index, role in enumerate(roles):
        by_rank = {
            rank: [task for task in role if owner_by_task[task] == rank]
            for rank in range(len(owners))
        }
        ranks = tuple(rank for rank, values in by_rank.items() if values)
        if not ranks:
            raise ValueError("shared Writer role lost all cache owners")
        for rank, values in by_rank.items():
            random.Random(int(seed) + 1009 * role_index + 9176 * rank).shuffle(values)
        rank_offset = (role_index * tasks_per_role) % len(ranks)
        rank_order = (*ranks[rank_offset:], *ranks[:rank_offset])
        cursors = {rank: 0 for rank in ranks}
        ordered: list[int] = []
        while len(ordered) < len(role):
            for rank in rank_order:
                cursor = cursors[rank]
                values = by_rank[rank]
                if cursor < len(values):
                    ordered.append(values[cursor])
                    cursors[rank] += 1
        role_orders.append(tuple(ordered))

    phases = _owner_spreading_phase_offsets(
        (role_orders[0], role_orders[1]), owners, tasks_per_role
    )
    selected: list[int] = []
    for ordered, phase in zip(role_orders, phases, strict=True):
        offset = (optimizer_step * tasks_per_role + phase) % len(ordered)
        selected.extend(
            ordered[(offset + index) % len(ordered)]
            for index in range(tasks_per_role)
        )
    if len(selected) != len(set(selected)):
        raise RuntimeError("shared Writer owner-balanced task group repeated a task")
    return tuple(selected)


def _split_ids(runtime: Any) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    split = runtime.config["task_split"]
    meta = tuple(map(int, split["gradient_meta"]))
    target = tuple(map(int, split["gradient_target"]))
    held = tuple(
        map(
            int,
            (
                *split["true_task_held_meta"],
                *split["true_task_held_target"],
            ),
        )
    )
    if (
        not meta
        or not target
        or not held
        or len(set((*meta, *target, *held))) != len(meta) + len(target) + len(held)
    ):
        raise ValueError("shared Writer registered task split changed")
    return meta, target, held


def _evaluation_ids(runtime: Any) -> tuple[int, ...]:
    split = runtime.config["task_split"]
    gradient_meta = tuple(
        map(int, split.get("panel_b_gradient_meta", split["gradient_meta"]))
    )
    gradient_target = tuple(
        map(int, split.get("panel_b_gradient_target", split["gradient_target"]))
    )
    values = (
        *gradient_meta,
        *gradient_target,
        *map(int, split["true_task_held_meta"]),
        *map(int, split["true_task_held_target"]),
    )
    meta, target, held = _split_ids(runtime)
    if (
        not set(gradient_meta) <= set(meta)
        or not set(gradient_target) <= set(target)
        or len(values) != len(set(values))
        or not set(values) <= set((*meta, *target, *held))
    ):
        raise ValueError("shared Writer Panel-B diagnostic split changed")
    return tuple(values)


def _video_splits(
    runtime: Any, task_ids: Sequence[int]
) -> tuple[dict[int, VideoSplit], dict[int, int]]:
    video_cell = runtime.config["data"].get("video_split")
    if video_cell is not None:
        if (
            video_cell.get("source")
            != "functional_panel_program_video_demos"
            or int(video_cell.get("fit_pool_max", -1)) not in {2, 3, 4}
            or video_cell.get("held_selection") != "last_sorted"
            or video_cell.get("selection_uses_outcomes") is not False
        ):
            raise ValueError("shared Writer scalable video split changed")
        result: dict[int, VideoSplit] = {}
        costs: dict[int, int] = {}
        for task in map(int, task_ids):
            demos = tuple(sorted(set(runtime.panels[task].program_video_demos)))
            fit_count = min(int(video_cell["fit_pool_max"]), len(demos) - 1)
            if fit_count < 2:
                raise ValueError(
                    f"shared Writer needs two fit videos and one held video for task {task}"
                )
            fit = demos[:fit_count]
            held = demos[-1]
            if held in fit:
                raise ValueError("shared Writer scalable video split overlapped")
            result[task] = (fit, held)
            costs[task] = sum(
                runtime.video_store.frame_counts(task, demo)[1]
                for demo in (*fit, held)
            )
        return result, costs

    split = load_mapping_split(runtime.base, asset_root=runtime.args.asset_root)
    result: dict[int, VideoSplit] = {}
    costs: dict[int, int] = {}
    for task in map(int, task_ids):
        fit = split.fit_by_task.get(task, ())
        held = split.video_held_by_task.get(task, ())
        if fit:
            uses_gradient_mapping = True
            if len(fit) < 2 or len(held) != 1:
                raise ValueError(f"shared Writer video split changed for task {task}")
            selected = (fit[0], fit[1])
            held_condition = held[0]
        else:
            uses_gradient_mapping = False
            task_held = tuple(
                sorted(
                    (
                        value
                        for value in split.task_held
                        if int(value.authority_id) == task
                    ),
                    key=lambda value: int(value.video_demo),
                )
            )
            if len(task_held) < 3:
                raise ValueError(
                    f"shared Writer true task-held videos changed for task {task}"
                )
            selected = (task_held[0], task_held[1])
            held_condition = task_held[2]
        if task not in runtime.panels or any(
            value.role != runtime.panels[task].role
            for value in (*selected, held_condition)
        ):
            raise ValueError(f"shared Writer video role changed for task {task}")
        fit_demos = tuple(int(value.video_demo) for value in selected)
        held_demo = int(held_condition.video_demo)
        if uses_gradient_mapping and not {*fit_demos, held_demo} <= set(
            runtime.panels[task].program_video_demos
        ):
            raise ValueError("shared Writer video escaped its sealed panel")
        result[task] = (fit_demos, held_demo)
        costs[task] = sum(
            int(value.sampled_frames) for value in (*selected, held_condition)
        )
    return result, costs
