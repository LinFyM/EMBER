"""Fixed §15 support-distribution fork; shared trainer/model stay unchanged."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

TASK = "conditional_support_diversity_pilot_20261002"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
SPEC_NAME = "conditional_support_diversity_spec.json"
PROTOCOL = "configs/operator_read_write_v1/support_diversity_protocol.json"
MANIFEST = "configs/operator_read_write_v1/support_diversity_manifest.json"
SCHEMA = "conditional_support_diversity_events_v1"
LOCAL_TASKS = (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22,
               23, 24, 26, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45,
               55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
               73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89)
SOURCE_TASKS = tuple(40 + t for t in LOCAL_TASKS)
CHECKPOINTS = (540, 630)
C12_GIT = "85919994aef11c17b49b7d0e70a2c110158bff61"
C12_SPEC = Path("/data1/user/ymdai/projects/EMBER-conditional-read-write-continuation900-formal"
                "/configs/operator_read_write_v1/conditional_read_write_continuation900_spec.json")
C12_CHECKPOINT = (ROOT.parent / "conditional_read_write_continuation900_20261002"
                 / "conditional_read_write/train/attempts/continuation/checkpoints/macro_00000630")


def expected_spec(parent: dict) -> dict:
    from .joint_training import CONDITIONAL_PARENT_GIT, CONDITIONAL_ROOT
    from .data import TASKS

    return {**parent, "task": TASK, "run_root": str(ROOT),
            "design": "docs/designs/conditional_read_write_architecture.md#15",
            "source": {**parent["source"], "data_protocol": PROTOCOL},
            "events": {**parent["events"], "schema_version": SCHEMA,
                       "task_ids": list(TASKS[:24] + SOURCE_TASKS),
                       "parent_schema": parent["events"]["schema_version"],
                       "support_permutation_seed": [20260928, 15, "cycle"],
                       "window": [451, 630], "support_conditions": 240,
                       "support_weight": "240/(71*planned_visits)",
                       "macro_reduction": "sum(weight*mean28_fullFM)/4"},
            "execution": {**parent["execution"], "updates_per_mode": 630,
                          "queries_per_mode": 70560, "checkpoints": list(CHECKPOINTS),
                          "only_selected_checkpoint": 630},
            "evaluation": {**parent["evaluation"], "bank_macro": 630},
            "continuation": {"parent_run_root": str(CONDITIONAL_ROOT), "parent_macro": 450,
                             "parent_training_git": CONDITIONAL_PARENT_GIT,
                             "parent_event_schema": parent["events"]["schema_version"],
                             "sampler_migration": "registered_support_distribution_fork"},
            "budget": {"new_gpu_hours_hard": 12, "peak_new_gib": 64,
                       "expected_wall_hours": [2, 4], "report_gpu_hours": 9}}


def schedule() -> dict[int, tuple[dict, ...]]:
    """Scheduling metadata only; no HDF5, RGB, actions or state read."""
    from .data import TASKS

    source_order = np.concatenate([np.random.default_rng(np.random.SeedSequence(
        [20260928, 15, cycle])).permutation(SOURCE_TASKS) for cycle in range(4)])[:240]
    counts = Counter(map(int, source_order))
    visits = {task: (50 if task in TASKS[24:] else 0) for task in SOURCE_TASKS}
    cursor, result = 0, {}
    for step in range(450, 630):
        visit, slot = divmod(step, 9)
        order = np.random.default_rng(np.random.SeedSequence([20260928, 0, visit])).permutation(TASKS)
        jobs = []
        for original in map(int, order[4 * slot:4 * (slot + 1)]):
            if original < 40:
                task, task_visit, weight, group = original, visit, 1., "target"
            else:
                task = int(source_order[cursor]); cursor += 1
                task_visit, weight, group = visits[task], 240 / (71 * counts[task]), "source"
                visits[task] += 1
            jobs.append({"task": task, "visit": task_visit, "weight": weight,
                         "group": group, "original_task": original})
        if len({job["task"] for job in jobs}) != 4:
            raise ValueError("support fork requires four different tasks per macro")
        result[step] = tuple(jobs)
    if cursor != 240 or Counter(counts.values()) != {3: 44, 4: 27}:
        raise ValueError("support fork fixed 71-task coverage changed")
    return result


def sampler_state(parent_state: dict, next_step: int) -> dict:
    plan = schedule()
    return {"schema_version": SCHEMA, "next_step": next_step,
            "parent_sampler": {**parent_state, "next_step": 450},
            "seed": 20260928, "support_tasks": list(SOURCE_TASKS),
            "support_permutation_seed": [20260928, 15, "cycle"],
            "support_slots": [job for jobs in plan.values() for job in jobs if job["group"] == "source"],
            "query_offset": 1, "queries_per_task": 28,
            "macro_reduction": "sum(weight*mean28_fullFM)/4"}
