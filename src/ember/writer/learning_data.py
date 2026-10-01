"""Task metadata and deterministic cross-episode queries shared by current training.

Task and episode identities remain orchestration metadata, never Writer inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from ember.writer.data import WriterTaskAuthority

@dataclass(frozen=True)
class LearningTask:
    authority: WriterTaskAuthority
    suite: str
    suite_task_id: int
    episode_lengths: tuple[int, ...]

def load_learning_tasks(
    asset_root: Path, task_ids: Sequence[int], *, role: str = "train", protocol_path: str | None = None,
) -> dict[int, LearningTask]:
    """Load task metadata only; training callers retain the fixed train default."""
    if role not in {"train", "validation", "test"}:
        raise ValueError("task metadata requires a registered target split")
    from ember.task_protocol import load_task_authorities
    _, manifest = load_task_authorities(asset_root, protocol_path)
    selected = tuple(map(int, task_ids))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("learning tasks must be explicit and unique")
    rows = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    output = {}
    data_root = asset_root / "data/datasets" / manifest["dataset"]["revision"]
    for task_id in selected:
        row = rows[task_id]
        suite, local = row["suite"], int(row["task_id"])
        if row["split_role"] != role:
            raise ValueError("selected task crosses the fixed target split")
        authority = WriterTaskAuthority(
            task_id, str(row["language"]), data_root / row["hdf5"]["relative_path"], int(row["hdf5"]["bytes"]),
        )
        output[task_id] = LearningTask(
            authority, suite, local, tuple(map(int, row["demonstrations"]["episode_lengths"])),
        )
    return output

EVENT_SCHEMA = "video_teaching_task_mixing_events_v2"

CONDITIONAL_EVENT_SCHEMA = "conditional_compilation_diagnostics_events_v1"

RELATIONAL_EVENT_SCHEMA = "relational_support_causality_events_v1"

TASKS_PER_UPDATE = 12

MAIN_EVENT_QUERIES = 21

TEACHING_EVENT_QUERIES = 7

def query_allocation(config, update_index):
    """Actual query counts; full event/RNG pools remain independently fixed."""
    if config.get("event_schema_version") in {CONDITIONAL_EVENT_SCHEMA, RELATIONAL_EVENT_SCHEMA}:
        tasks_per_update = config.get("tasks_per_update")
        main = config.get("queries_per_task")
        extra = config.get("teaching_queries_per_task")
        if (tasks_per_update != 4 or main != MAIN_EVENT_QUERIES
                or extra != TEACHING_EVENT_QUERIES or type(update_index) is not int
                or update_index < 0):
            raise ValueError("conditional compilation requires four tasks with fixed 21+7 query groups")
        return main, (extra,) * tasks_per_update
    main, teaching = config.get("queries_per_task"), config.get("teaching_query_counts")
    if (type(main) is not int or main not in (7, 21)
            or not isinstance(teaching, list) or len(teaching) != TASKS_PER_UPDATE
            or any(type(count) is not int for count in teaching)
            or teaching not in ([3, 2, 2] * 4, [7] * TASKS_PER_UPDATE)
            or "teaching_queries_per_task" in config):
        raise ValueError("task-mixing query contract requires main 7/21 and an explicit rotating auxiliary allocation")
    return main, tuple(teaching[(position - update_index) % TASKS_PER_UPDATE]
                       for position in range(TASKS_PER_UPDATE))

def _episode_queries(task, lengths, order, *, seed, cursor, count, teacher_demo=None):
    """Walk an episode permutation; skip the teacher without consuming labels."""
    demos, frames = [], []
    while len(demos) < count:
        cycle, offset = divmod(cursor, len(order))
        cursor += 1
        demo = order[offset]
        if demo == teacher_demo:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, task, demo, cycle, 0xF4A]))
        demos.append(demo)
        frames.append(int(rng.integers(lengths[demo] - 1)))
    return demos, frames, cursor
