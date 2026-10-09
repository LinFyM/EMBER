"""Equal-weight event reuse from recorded actual parameter/experience pools."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import default_collate

from ember.pi05_source_checkpoint import read_json
from ember.writer.data import FunctionalQueryDataset
from .contract import ASSET_ROOT, EVENT_SCHEMA, SEED, TASKS36, condition_seed, training_tasks


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


def collection_conditions(pool: str, *, asset_root=ASSET_ROOT):
    if pool not in {'pool0', 'refresh180'}:
        raise ValueError('only the two registered collection pools exist')
    tasks, result = training_tasks(Path(asset_root)), []
    offset, count = (0, 4) if pool == 'pool0' else (4, 2)
    for task_id, task in tasks.items():
        bag = np.random.default_rng(np.random.SeedSequence([SEED, 0xC011, task_id])).permutation(50)
        for ordinal in range(offset, offset + count):
            demo = int(bag[ordinal])
            result.append(dict(condition_id=f'{pool}_task{task_id:03d}_demo{demo:02d}',
                task_id=task_id, suite=task.suite, suite_task_id=task.suite_task_id,
                language=task.authority.language, teacher_demo=demo,
                seed=condition_seed(task_id, demo, ordinal, domain=0xC011),
                excluded_states=[32, 33, 34], pool=pool))
    return result


def _masked_cursors():
    """One event per eight, each task once per eight equal-weight visits."""
    # Match 36 eight-event blocks to distinct tasks in their actual RR slots.
    assigned = {}
    def place(block, seen):
        for slot in range(8):
            task = (block * 8 + slot) % 36
            if task in seen:
                continue
            seen.add(task)
            if task not in assigned or place(assigned[task], seen):
                assigned[task] = block
                return True
        return False
    for block in range(36):
        if not place(block, set()):
            raise RuntimeError('balanced experience-mask assignment failed')
    return frozenset(next(cursor for cursor in range(block * 8, block * 8 + 8)
                          if cursor % 36 == task) for task, block in assigned.items())


MASKED_CURSORS = _masked_cursors()


def event_for_update(tasks, pools, update):
    if type(update) is not int or not 0 <= update < 360 or tuple(tasks) != TASKS36:
        raise ValueError('event needs update0..359 and the fixed36 task order')
    result = []
    for position in range(4):
        cursor = update * 4 + position
        task_id, visit = TASKS36[cursor % 36], cursor // 36
        pool = 'pool0' if update < 180 or (visit - 20) % 2 == 0 else 'refresh180'
        conditions = pools[pool]['by_task'][task_id]
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 0xE017, cursor]))
        condition = conditions[int(rng.integers(len(conditions)))]
        endpoint = condition['events'][int(rng.integers(len(condition['events'])))]
        teacher = condition['teacher_demo']
        demos = rng.choice([demo for demo in range(50) if demo != teacher], size=7, replace=False)
        queries = []
        for demo in demos:
            available = int(tasks[task_id].episode_lengths[int(demo)]) - 1
            if available < 4:
                raise ValueError('query episode needs four nonempty intervals')
            for interval in range(4):
                queries.append((int(demo), int(rng.integers(available * interval // 4,
                                                           available * (interval + 1) // 4))))
        result.append(Event(update, position, task_id, teacher, tuple(queries),
            cursor % 288 in MASKED_CURSORS, condition_seed(update, position, task_id, teacher),
            condition['condition_id'], pool, int(endpoint['endpoint']), endpoint['incoming'],
            condition['record_path'], endpoint['behavior_version']))
    return result


class QueryData:
    """Only train24+support12 labels; sampling ignores success and queue order."""
    def __init__(self, asset_root=ASSET_ROOT, *, pool_paths=()):
        self.tasks = training_tasks(Path(asset_root))
        if tuple(self.tasks) != TASKS36:
            raise ValueError('functional labels changed fixed36 authority')
        self.queries = FunctionalQueryDataset(tuple(t.authority for t in self.tasks.values()),
            demo_indices=tuple(range(50)), action_chunk_size=50, action_start_offset=1)
        self.rows, self.next_update, self.pools = self.queries.task_episode_rows, 0, {}
        for path in pool_paths:
            self.attach_pool(path)

    def attach_pool(self, path):
        manifest = read_json(Path(path))
        pool = manifest['pool']
        expected = collection_conditions(pool)
        if (manifest.get('schema_version') != EVENT_SCHEMA or not manifest.get('complete')
                or [c['condition_id'] for c in manifest['conditions']] != [c['condition_id'] for c in expected]):
            raise ValueError('collection manifest is incomplete or changed registered conditions')
        by_task = {task: [] for task in TASKS36}
        for condition, authority in zip(manifest['conditions'], expected, strict=True):
            if any(condition[k] != authority[k] for k in authority) or not condition['events']:
                raise ValueError('recorded condition lost authority or actual endpoints')
            by_task[condition['task_id']].append(condition)
        self.pools[pool] = dict(path=str(Path(path).resolve()), version=manifest['version'], by_task=by_task)

    def events(self, update=None):
        return event_for_update(self.tasks, self.pools, self.next_update if update is None else update)

    def state_dict(self):
        return dict(schema_version=EVENT_SCHEMA, seed=SEED, task_order=list(TASKS36),
            next_update=self.next_update, tasks_per_update=4, queries=[7, 4], action_offset=1,
            masked_fraction=[1, 8], pool_versions={k: v['version'] for k, v in self.pools.items()})

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {'next_update', 'pool_versions'}:
            if state.get(key) != expected[key]:
                raise ValueError('logical sampler authority changed')
        if type(state.get('next_update')) is not int or not 0 <= state['next_update'] <= 360:
            raise ValueError('logical cursor changed')
        for pool, version in state['pool_versions'].items():
            if expected['pool_versions'].get(pool) != version:
                raise ValueError('recorded behavior pool version changed')
        if expected['pool_versions'].keys() != state['pool_versions'].keys() and state['next_update'] != 180:
            raise ValueError('refresh can only be attached at the registered180 boundary')
        self.next_update = state['next_update']

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
