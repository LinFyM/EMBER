"""Resumable equal-weight36-task events and training-only functional queries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import default_collate

from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore

from .contract import ASSET_ROOT, EVENT_SCHEMA, SEED, TASKS36, condition_seed, training_tasks


@dataclass(frozen=True)
class Event:
    update: int
    position: int
    task_id: int
    teacher_demo: int
    queries28: tuple[tuple[int, int], ...]
    query_states2: tuple[int, int]
    masked_experience: bool
    seed: int
    condition_id: str

    def as_dict(self) -> dict:
        return asdict(self)


def event_for_update(tasks, update: int) -> list[Event]:
    """Four consecutive task slots; visits/bags continue through warm and meta."""
    if type(update) is not int or update < 0 or tuple(tasks) != TASKS36:
        raise ValueError("event needs a nonnegative continuous update and the ordered fixed36 tasks")
    result = []
    for position in range(4):
        cursor = update * 4 + position
        task_id = TASKS36[cursor % len(TASKS36)]
        visit = cursor // len(TASKS36)
        bag, slot = divmod(visit, 50)
        teacher = int(np.random.default_rng(np.random.SeedSequence(
            [SEED, 0x5445, task_id, bag])).permutation(50)[slot])
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 0x5155, task_id, visit]))
        demos = rng.choice([demo for demo in range(50) if demo != teacher], size=7, replace=False)
        queries = []
        for demo in demos:
            available = int(tasks[task_id].episode_lengths[int(demo)]) - 1
            if available < 4:
                raise ValueError("query episode cannot supply four nonempty time intervals")
            for interval in range(4):
                frame = int(rng.integers(available * interval // 4, available * (interval + 1) // 4))
                queries.append((int(demo), frame))
        query_states = tuple(map(int, rng.choice(50, size=2, replace=False)))
        # One uniformly sampled slot per block of eight gives exactly1/8,
        # without prescribing task identity or adaptation/read counts.
        masked_slot = int(np.random.default_rng(np.random.SeedSequence(
            [SEED, 0x4D41, cursor // 8])).integers(8))
        result.append(Event(update, position, task_id, teacher, tuple(queries), query_states,
                            cursor % 8 == masked_slot, condition_seed(update, position, task_id, teacher),
                            f"update{update:06d}_position{position}_task{task_id:03d}_demo{teacher:02d}"))
    return result


class QueryData:
    """Only the fixed train24+support12 own HDF5 labels; teacher reads RGB only."""

    def __init__(self, asset_root: Path = ASSET_ROOT) -> None:
        self.tasks = training_tasks(Path(asset_root))
        if tuple(self.tasks) != TASKS36:
            raise ValueError("query labels must remain inside the ordered fixed36 train authority")
        authorities = tuple(task.authority for task in self.tasks.values())
        self.videos = RawTeacherVideoStore(authorities, frame_stride=5, camera_view="dual")
        self.queries = FunctionalQueryDataset(authorities, demo_indices=tuple(range(50)),
                                               action_chunk_size=50, action_start_offset=1)
        self.rows = self.queries.task_episode_rows
        self.next_update = 0

    def events(self, update: int | None = None) -> list[Event]:
        return event_for_update(self.tasks, self.next_update if update is None else update)

    def state_dict(self) -> dict:
        return {"schema_version": EVENT_SCHEMA, "seed": SEED, "task_order": list(TASKS36),
                "next_update": self.next_update, "tasks_per_update": 4,
                "teacher_pool": list(range(50)), "query_episodes": 7,
                "intervals_per_episode": 4, "action_offset": 1, "masked_fraction": [1, 8]}

    def load_state_dict(self, state: dict) -> None:
        expected = self.state_dict()
        if (type(state.get("next_update")) is not int or state["next_update"] < 0
                or {key: value for key, value in state.items() if key != "next_update"}
                != {key: value for key, value in expected.items() if key != "next_update"}):
            raise ValueError("event identity or continuous update cursor changed")
        self.next_update = state["next_update"]

    def query_batch(self, event: Event, processor) -> dict:
        demos = [demo for demo, _ in event.queries28]
        if (event.task_id not in self.tasks or len(demos) != 28 or len(set(demos)) != 7
                or event.teacher_demo in demos):
            raise ValueError("functional queries must use seven nonteacher episodes of a fixed36 task")
        rows = [self.queries[self.rows[event.task_id][demo][frame]] for demo, frame in event.queries28]
        batch = default_collate(rows)
        # IDs/filenames are orchestration metadata, never policy/Compiler inputs.
        allowed = ("observation.images.camera1", "observation.images.camera2", "observation.state",
                   "action", "action_is_pad", "task")
        return processor.training_batch({key: batch[key] for key in allowed})

    def close(self) -> None:
        self.videos.close()
        self.queries.close()
