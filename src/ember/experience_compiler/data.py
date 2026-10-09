"""Equal-weight event reuse from recorded actual parameter/experience pools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import default_collate

from ember.pi05_source_checkpoint import read_json
from ember.writer.data import FunctionalQueryDataset
from .contract import ASSET_ROOT, TASKS36, training_tasks


@dataclass(frozen=True)
class Event:
    update: int
    position: int
    task_id: int
    teacher_demo: int
    queries28: tuple[tuple[int, int], ...]
    masked_experience: bool
    seed: int
    condition_id: str
    pool: str
    endpoint: int
    incoming: str
    record_path: str
    behavior_version: str

    def as_dict(self):
        return asdict(self)


class QueryData:
    """Only train24+support12 labels; sampling ignores success and queue order."""
    def __init__(self, asset_root=ASSET_ROOT):
        self.tasks = training_tasks(Path(asset_root))
        if tuple(self.tasks) != TASKS36:
            raise ValueError('functional labels changed fixed36 authority')
        self.queries = FunctionalQueryDataset(tuple(t.authority for t in self.tasks.values()),
            demo_indices=tuple(range(50)), action_chunk_size=50, action_start_offset=1)
        self.rows = self.queries.task_episode_rows

    def query_coordinates(self, task_id, teacher_demo, rng):
        """Seven other episodes, one native query in each of four intervals."""
        task = self.tasks[task_id]
        demos = rng.choice([d for d in range(50) if d != teacher_demo], 7, replace=False)
        rows = []
        for demo in demos:
            length = int(task.episode_lengths[int(demo)]) - 1
            for interval in range(4):
                low, high = length * interval // 4, length * (interval + 1) // 4
                rows.append((int(demo), int(rng.integers(low, high))))
        return tuple(rows)

    def raw_query_batch(self, event):
        demos = [demo for demo, _ in event.queries28]
        if (event.task_id not in self.tasks or len(demos) != 28 or len(set(demos)) != 7
                or event.teacher_demo in demos):
            raise ValueError('FM queries require seven nonteacher episodes of an allowed task')
        rows = [self.queries[self.rows[event.task_id][demo][frame]] for demo, frame in event.queries28]
        batch = default_collate(rows)
        allowed = ('observation.images.camera1', 'observation.images.camera2', 'observation.state',
                   'action', 'action_is_pad', 'task')
        return {key: batch[key] for key in allowed}

    def query_batch(self, event, processor):
        return processor.training_batch(self.raw_query_batch(event))

    def close(self):
        self.queries.close()
