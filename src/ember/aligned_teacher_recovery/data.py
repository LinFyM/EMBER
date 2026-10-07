"""One task's fixed without-replacement episode/frame FM query stream."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from torch.utils.data import default_collate

from ember.writer.data import FunctionalQueryDataset
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.learning_data import load_learning_tasks

TASKS = (12, 29, 32, 38)
SEED, UPDATES, QUERIES, BLOCK = 20261008, 480, 112, 28
SCHEMA = "aligned_teacher_recovery_queries_v1"


class TeacherData:
    """Metadata only in scheduling; policy receives its real RGB/state/language."""

    def __init__(self, asset_root: Path, task: int, protocol: str):
        if task not in TASKS:
            raise ValueError("teacher task outside the registered four")
        self.task = task
        self.authority = load_learning_tasks(asset_root, (task,), role="train",
                                              protocol_path=protocol)[task]
        lengths = self.authority.episode_lengths
        if len(lengths) != 50 or min(lengths) < 2:
            raise ValueError("teacher requires the canonical fifty valid demos")
        self.queries = FunctionalQueryDataset((self.authority.authority,),
            demo_indices=tuple(range(50)), action_chunk_size=50, action_start_offset=1)
        self.rows = self.queries.task_episode_rows[task]
        self.next_update = 0
        self._episode_orders, self._frame_orders = {}, {}

    def query(self, n: int) -> dict:
        if type(n) is not int or not 0 <= n < UPDATES * QUERIES:
            raise ValueError("query cursor outside the finite teacher stream")
        cycle, slot = divmod(n, 50)
        if cycle not in self._episode_orders:
            self._episode_orders[cycle] = np.random.default_rng(
                np.random.SeedSequence([SEED, self.task, 0, cycle])).permutation(50)
        demo = int(self._episode_orders[cycle][slot])
        m = self.authority.episode_lengths[demo] - 1
        turn, position = divmod(cycle, m)
        key = demo, turn
        if key not in self._frame_orders:
            self._frame_orders[key] = np.random.default_rng(
                np.random.SeedSequence([SEED, self.task, 1, demo, turn])).permutation(m)
        return {"n": n, "demo": demo, "frame": int(self._frame_orders[key][position])}

    def event(self, update: int) -> dict:
        if type(update) is not int or not 0 <= update < UPDATES:
            raise ValueError("update outside the registered teacher schedule")
        queries = [self.query(update * QUERIES + n) for n in range(QUERIES)]
        blocks = []
        for block in range(4):
            rows = queries[block * BLOCK:(block + 1) * BLOCK]
            seed = task_logical_batch_policy_rng_seed(optimization_seed=7,
                task_id=self.task, task_visit=4 * update + block,
                demo_indices=[row["demo"] for row in rows],
                frame_indices=[row["frame"] for row in rows])
            blocks.append({"block": block, "start": block * BLOCK, "flow_seed": seed,
                           "logical_batch": BLOCK, "query_offset": 1})
        return {"task": self.task, "update": update + 1, "queries": queries, "blocks": blocks}

    def batch(self, event: dict):
        if event["task"] != self.task or len(event["queries"]) != QUERIES:
            raise ValueError("the actual FM batch lost its task or112 samples")
        return default_collate([self.queries[self.rows[row["demo"]][row["frame"]]]
                                for row in event["queries"]])

    def sampler_state(self, cursor: int | None = None) -> dict:
        cursor = self.next_update if cursor is None else cursor
        return {"schema_version": SCHEMA, "task": self.task, "root": SEED,
                "next_update": cursor, "next_query": cursor * QUERIES,
                "updates": UPDATES, "queries_per_update": QUERIES,
                "episode_lengths": list(self.authority.episode_lengths),
                "action_start_offset": 1, "logical_flow_batch": BLOCK,
                "flow_owner": "task_logical_batch_policy_rng_seed/flow_sample"}

    def close(self):
        self.queries.close()
