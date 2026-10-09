"""World-independent task, actual-incoming and cross-episode query streams."""
from __future__ import annotations

from copy import deepcopy
import random

import numpy as np

from .data import Event


SAMPLER_SCHEMA = 'ember_functional_revision_sampler_v1'
LOGICAL_BATCH = 4
PHASE_UPDATES = 180


def incoming_layer(event):
    """New actual-coordinate identity overrides a filename, without editing it."""
    is_mt = event.get('incoming_is_MT', event['incoming'] == 'MT')
    if type(is_mt) is not bool or (event['incoming'] == 'MT' and not is_mt):
        raise ValueError('actual incoming layer contradicts its MT authority')
    return 'MT' if is_mt else 'nonMT'


def _seed(seed, *coordinates):
    values = np.random.SeedSequence([seed, *coordinates]).generate_state(2)
    return ((int(values[0]) << 32) | int(values[1])) & ((1 << 63) - 1)


class _Bag:
    """Independent shuffled rounds; incomplete rounds survive a checkpoint."""
    def __init__(self, seed, values):
        self.values = tuple(values)
        self.rng, self.remaining = random.Random(seed), []
        if not self.values or len(set(self.values)) != len(self.values):
            raise ValueError('a sampling bag needs distinct available choices')

    def draw(self):
        if not self.remaining:
            self.remaining = list(self.values)
            self.rng.shuffle(self.remaining)
        return self.remaining.pop()

    def state_dict(self):
        return dict(values=self.values, remaining=list(self.remaining), rng=self.rng.getstate())

    def load_state_dict(self, state):
        if (state['values'] != self.values or len(set(state['remaining'])) != len(state['remaining'])
                or not set(state['remaining']).issubset(self.values)):
            raise ValueError('sampling bag choices changed on resume')
        self.remaining = list(state['remaining'])
        self.rng.setstate(state['rng'])


class EventSampler:
    """One task round per nine updates; layers are balanced inside each task.

    Conditions are uniform within an available layer, then endpoints are
    uniform within that condition. A long chain never increases task or
    condition weight. Query randomness uses absolute event coordinates,
    independently of all selection bags and physical rank assignment.
    """
    def __init__(self, manifest, query_lengths, *, query_coordinates, task_ids, seed, phase):
        self.task_ids, self.seed, self.phase = tuple(task_ids), int(seed), int(phase)
        self.query_coordinates = query_coordinates
        if (phase not in (1, 2) or len(self.task_ids) != 36
                or len(set(self.task_ids)) != 36 or not manifest.get('complete')):
            raise ValueError('formal sampling requires one complete pool and the fixed36 tasks')
        self.conditions = sorted(deepcopy(manifest['conditions']), key=lambda row: row['condition_id'])
        if len({row['condition_id'] for row in self.conditions}) != len(self.conditions):
            raise ValueError('duplicate actual condition identity')
        self.query_lengths = {task: {int(d): int(n) for d, n in query_lengths[task].items()}
                              for task in self.task_ids}
        self.by_identity = {row['condition_id']: row for row in self.conditions}
        self.by_layer = {task: {} for task in self.task_ids}
        self.bags, self.cursor = {}, 0
        self.exposures = {task: dict(MT=0, nonMT=0) for task in self.task_ids}
        self._add_bag(('tasks',), self.task_ids)
        for index, condition in enumerate(self.conditions):
            self._index_condition(index, condition)
        for task, layers in self.by_layer.items():
            if not layers:
                raise ValueError(f'task {task} has no actual event; task substitution is forbidden')
            self._add_bag(('layer', task), tuple(sorted(layers)))
            for layer, indices in layers.items():
                self._add_bag(('condition', task, layer), indices)
        self.plan = dict(schema_version=SAMPLER_SCHEMA, phase=phase, seed=seed,
            task_ids=self.task_ids, logical_batch=LOGICAL_BATCH, phase_updates=PHASE_UPDATES,
            conditions=self.conditions, query_lengths=self.query_lengths,
            query_rule='seven_nonteacher_demos_four_intervals_offset1_absolute_event_seed')

    def _add_bag(self, key, values):
        # Domain strings are encoded explicitly, never via salted Python hash.
        domains = {'tasks': 1, 'layer': 2, 'condition': 3, 'endpoint': 4}
        coordinates = [self.phase, domains[key[0]]]
        coordinates.extend(5 if value == 'MT' else 6 if value == 'nonMT' else int(value)
                           for value in key[1:])
        self.bags[key] = _Bag(_seed(self.seed, *coordinates), values)

    def _index_condition(self, index, condition):
        task, demo = int(condition['task_id']), int(condition['teacher_demo'])
        if (task not in self.by_layer or not 0 <= demo < 50 or not condition.get('complete')
                or not condition.get('actual_incoming_parameters') or not condition.get('record_path')):
            raise ValueError('pool condition lacks legal task/teacher/actual-incoming authority')
        lengths = self.query_lengths[task]
        if set(lengths) != set(range(50)) or any(length < 4 for length in lengths.values()):
            raise ValueError('query authority needs all50 episodes with four nonempty intervals')
        events = condition['events']
        if (not events or len({e['endpoint'] for e in events}) != len(events)
                or any(int(e['endpoint']) < 1 or not e['incoming'] or not e['behavior_version'] for e in events)):
            raise ValueError('pool condition lacks distinct recorded actual endpoints')
        for layer in ('MT', 'nonMT'):
            endpoints = [i for i, event in enumerate(events) if incoming_layer(event) == layer]
            if endpoints:
                self.by_layer[task].setdefault(layer, []).append(index)
                self._add_bag(('endpoint', index, layer), endpoints)

    def _queries(self, task, teacher, absolute_position):
        rng = np.random.default_rng(_seed(self.seed, self.phase, 10, absolute_position))
        queries = self.query_coordinates(task, teacher, rng)
        return tuple(queries), _seed(self.seed, self.phase, 11, absolute_position)

    def next_batch(self):
        if self.cursor >= PHASE_UPDATES:
            raise StopIteration('registered180-update phase completed')
        update = (self.phase - 1) * PHASE_UPDATES + self.cursor + 1
        result = []
        for position in range(LOGICAL_BATCH):
            task = self.bags[('tasks',)].draw()
            layer = self.bags[('layer', task)].draw()
            index = self.bags[('condition', task, layer)].draw()
            condition = self.conditions[index]
            endpoint_index = self.bags[('endpoint', index, layer)].draw()
            actual = condition['events'][endpoint_index]
            queries, seed = self._queries(task, int(condition['teacher_demo']),
                                          (update - 1) * LOGICAL_BATCH + position)
            result.append(Event(update, position, task, int(condition['teacher_demo']), queries,
                False, seed, condition['condition_id'], condition.get('pool', 'bootstrap' if self.phase == 1 else 'refresh180'),
                int(actual['endpoint']), actual['incoming'], condition['record_path'], actual['behavior_version']))
            self.exposures[task][layer] += 1
        self.cursor += 1
        return tuple(result)

    def state_dict(self):
        return deepcopy(dict(plan=self.plan, cursor=self.cursor, exposures=self.exposures,
                             bags={key: bag.state_dict() for key, bag in self.bags.items()}))

    def event_record(self, event):
        actual = next(e for e in self.by_identity[event.condition_id]['events'] if e['endpoint'] == event.endpoint)
        return dict(event.as_dict(), actual_incoming_layer=incoming_layer(actual))

    def load_state_dict(self, state):
        if state['plan'] != self.plan or set(state['bags']) != set(self.bags):
            raise ValueError('logical event/query plan changed on resume')
        cursor = int(state['cursor'])
        if (not 0 <= cursor <= PHASE_UPDATES or set(state['exposures']) != set(self.task_ids)
                or sum(sum(counts.values()) for counts in state['exposures'].values()) != cursor * LOGICAL_BATCH):
            raise ValueError('sampler cursor and actual exposure counts disagree')
        for key, bag in self.bags.items():
            bag.load_state_dict(state['bags'][key])
        self.cursor, self.exposures = cursor, deepcopy(state['exposures'])

    def coverage(self):
        return dict(phase=self.phase, updates=self.cursor,
            event_presentations=self.cursor * LOGICAL_BATCH,
            actual_conditions=len(self.conditions), actual_endpoints=sum(len(c['events']) for c in self.conditions),
            nonMT_tasks=[task for task in self.task_ids if 'nonMT' in self.by_layer[task]],
            per_task=deepcopy(self.exposures),
            nonMT_presentations=sum(counts['nonMT'] for counts in self.exposures.values()))
