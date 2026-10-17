"""Fresh equal-task teaching/query stream; report labels never enter conditions."""
from pathlib import Path
import json
import numpy as np
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.learning_data import _episode_queries
from .contract import (ASSETS, TASKS, SCHEMA, PHASE, LIMITS, SOURCE, NORMALIZATION,
    INITIAL_READER, PREVIOUS_ROOT, H_ROOT, OBJECTIVE, FINAL, PERMANENT,
    STATES, IDENTITY_TASKS, OPTIMIZER)


def permutation(task, domain, cycle, count):
    return np.random.default_rng(np.random.SeedSequence([7, int(task), int(domain), int(cycle)])).permutation(count).tolist()


def report_conditions():
    conditions = {int(t): c for t, c in read_json(H_ROOT/'conditions.json').items()}
    if set(conditions) != set(TASKS):
        raise ValueError('only the original36 non-held source conditions are authorized')
    for task, condition in conditions.items():
        if (condition['task'] != task or condition['v'] != 0
                or condition['query_states'] != list(STATES) or condition['query_noise_root'] != 202610151
                or condition['practice_noise_root'] != 202610161 or len(set(condition['practice_states'])) != 4
                or any(s not in range(16, 50) for s in condition['practice_states'])
                or condition['teacher_demo'] not in range(50)
                or condition['original']['task'] != task
                or condition['original']['demo'] != condition['teacher_demo']):
            raise ValueError('report v0/practice/query identity changed')
    return conditions


def registered_panels(conditions=None):
    conditions = report_conditions() if conditions is None else conditions
    return dict(seen=[dict(task=t, init_state_id=i, teacher_demo=conditions[t]['teacher_demo'],
        noise_root=202610151) for t in TASKS for i in STATES])


def prepare(root):
    root = Path(root)
    conditions = report_conditions()
    environment = read_json(H_ROOT/'run_contract.json')['environment']
    tasks = {t['global_task_id']: t for t in environment['tasks']}
    if set(tasks) != set(TASKS) or any(t['horizon'] != 400 for t in tasks.values() if t['suite'] == 'libero_90'):
        raise ValueError('the original audited36-task environment/support400 changed')
    contract = dict(schema=SCHEMA, phase=PHASE, objective=OBJECTIVE, limits=LIMITS,
        tasks=list(TASKS), environment=environment, environments=dict(seen=environment),
        conditions=str(root/'conditions.json'), panels=str(root/'panels.json'),
        source_checkpoint=str(ASSETS/SOURCE), normalization=str(ASSETS/NORMALIZATION),
        frozen_reader=str(INITIAL_READER), source_history_root=str(H_ROOT),
        shared_M=dict(root=str(PREVIOUS_ROOT/'cache/M'), read_only=True, global_capacity=2**30),
        historical_source_practice=dict(episodes=144, control_steps=38937, dummy_steps=1440,
            total_steps=40377, successes=52, replans=7808, source_only=True, read_only=True),
        identity_tasks=list(IDENTITY_TASKS), rank=8, alpha=8, dropout=0., targets=38, seed=7,
        optimizer=dict(OPTIMIZER, clip=1., schedule='constant'), source_frozen=True, MT_loaded=False,
        logical=dict(tasks=4, FM_queries_per_task=28, query_episodes=7, time_strata=4,
            updates=FINAL, conditions=720, FM_queries=20160, cycles=20),
        checkpoints=dict(recovery_only=[90], fixed_terminal=FINAL, permanent=list(PERMANENT), rolling=True),
        shared_updates=FINAL, final_update=FINAL, optimizer_restored=False, eval_rows=144,
        new_practice=0, query_feedback_to_compile=False, interaction_adaptation=True,
        teacher_condition='action-hidden ordered RGB memory only',
        compiler_condition='legal teaching memory plus original source practice000 selected16 decisions',
        reference_parameters_are_inputs=False, held=False, Test=False)
    root.mkdir(parents=True, exist_ok=True)
    for name, value in (('run_contract.json', contract), ('conditions.json', conditions), ('panels.json', registered_panels(conditions))):
        value = json.loads(json.dumps(value)); path = root/name
        if path.exists() and read_json(path) != value:
            raise ValueError('immutable native-operator contract changed: '+name)
        if not path.exists():
            write_json_atomic(path, value)
    return contract


class Stream:
    """IDs/cursors are scheduler metadata, never neural conditioning tokens."""
    def __init__(self, runtime, report_metadata=None):
        self.runtime = runtime
        values = report_conditions() if report_metadata is None else report_metadata
        self.conditions = {int(t): c for t, c in values.items()}
        if set(runtime.tasks) != set(TASKS) or set(self.conditions) != set(TASKS):
            raise ValueError('stream requires exactly the audited36 training tasks')
        self.next_update = 0
        self.visits = {t: 0 for t in TASKS}
        self.query_cursor = {t: 0 for t in TASKS}
        self.data = None

    def _event(self, task, update, ordinal):
        from ember.writer.functional import task_logical_batch_policy_rng_seed
        visit = self.visits[task]; report = self.conditions[task]['teacher_demo']
        pool = [d for d in range(50) if d != report]
        demo = pool[permutation(task, 1, 0, 49)[visit]]
        before = self.query_cursor[task]; lengths = self.runtime.tasks[task].episode_lengths
        demos, _, after = _episode_queries(task, lengths, permutation(task, 3, 0, 50),
            seed=7, cursor=before, count=7, teacher_demo=demo)
        queries = []
        for episode in demos:
            for stratum in range(4):
                low, high = (stratum*(lengths[episode]-1)//4, (stratum+1)*(lengths[episode]-1)//4)
                if high <= low:
                    raise ValueError('each query episode must support four nonempty temporal strata')
                rng = np.random.default_rng(np.random.SeedSequence([7, task, 0xF4A, visit, episode, stratum]))
                queries.append(dict(demo=int(episode), frame=int(rng.integers(low, high)), stratum=stratum))
        seed = task_logical_batch_policy_rng_seed(optimization_seed=7, task_id=task, task_visit=visit,
            demo_indices=[q['demo'] for q in queries], frame_indices=[q['frame'] for q in queries])
        self.visits[task] += 1; self.query_cursor[task] = after
        return dict(task=task, visit=visit, demo=demo, teacher_demo=demo, report_teacher_demo=report,
            update=update, logical_ordinal=ordinal, queries=queries, demos=[q['demo'] for q in queries],
            frames=[q['frame'] for q in queries], query_cursor_before=before, query_cursor_after=after,
            noise_time_root=seed, noise_seed=seed, time_seed=seed, random_batch=28, random_offset=0)

    def next(self, update=None):
        update = self.next_update if update is None else update
        if type(update) is not int or update != self.next_update:
            raise ValueError('logical update must continue the exact saved stream cursor')
        if update >= FINAL:
            raise StopIteration('the unique180-update native-operator window is complete')
        cycle, slot = divmod(update, 9); order = permutation(0, 0, cycle, 36)
        events = [self._event(TASKS[index], update, update*4+j)
            for j, index in enumerate(order[slot*4:slot*4+4])]
        self.next_update += 1
        return events

    def event_for(self, task, visit):
        """Pure metadata for the authorized disposable consumer; no formal cursor change."""
        if task not in TASKS or type(visit) is not int or not 0 <= visit < 20:
            raise ValueError('consumer metadata must stay in the registered36-task/20-visit pool')
        probe = Stream(self.runtime, self.conditions)
        for turn in range(visit+1):
            event = probe._event(task, turn, turn)
        return event

    def state_dict(self):
        return dict(schema=SCHEMA, seed=7, next_update=self.next_update, visits=dict(self.visits),
            query_cursor=dict(self.query_cursor), report_teachers={t: c['teacher_demo'] for t, c in self.conditions.items()},
            query_scheme='7_non_teacher_episodes_x4_temporal_strata_offset1_chunk50_v1',
            noise_time_scheme='task_logical_batch_policy_rng_seed_independent_gaussian_beta_v1')

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ('schema', 'seed', 'report_teachers', 'query_scheme', 'noise_time_scheme'):
            value = {int(k): v for k, v in state[key].items()} if key == 'report_teachers' else state[key]
            if value != expected[key]:
                raise ValueError('resume changed the fixed training stream identity: '+key)
        count = state['next_update']
        if type(count) is not int or not 0 <= count <= FINAL:
            raise ValueError('resume update is outside the finite180 window')
        # Replay metadata only, without opening a dataset or consuming any labels.
        probe = Stream(self.runtime, self.conditions)
        for update in range(count):
            probe.next(update)
        for name in ('visits', 'query_cursor'):
            values = {int(k): int(v) for k, v in state[name].items()}
            if values != getattr(probe, name):
                raise ValueError('resume did not preserve the exact task/video/query cursor: '+name)
            setattr(self, name, values)
        self.next_update = count

    def sample(self, event, indices=None):
        from torch.utils.data import default_collate
        from ember.writer.data import FunctionalQueryDataset
        from ember.writer.function_credit import flow_sample, FlowSample
        if self.data is None:
            self.data = FunctionalQueryDataset(tuple(t.authority for t in self.runtime.tasks.values()),
                demo_indices=tuple(range(50)), action_chunk_size=50, action_start_offset=1)
        rows = self.data.task_episode_rows[event['task']]
        if (len(event['queries']) != 28 or len({q['demo'] for q in event['queries']}) != 7
                or any(q['demo'] == event['demo'] for q in event['queries'])):
            raise ValueError('FM labels must remain seven cross-episode groups of four queries')
        spans = _physical_spans(indices)
        samples = []
        for span in spans:
            batch = default_collate([self.data[rows[event['queries'][i]['demo']][event['queries'][i]['frame']]] for i in span])
            batch = self.runtime.processor.training_batch(batch)
            samples.append(flow_sample(self.runtime.policy, batch, seed=event['noise_time_root'], device=self.runtime.device,
                random_batch=28, offset=span[0]))
        if len(samples) == 1:
            return samples[0]
        return FlowSample(_cat_tree([s.arguments for s in samples]),
            _cat_tree([s.target for s in samples]), samples[0].action_width)

    def close(self):
        if self.data is not None:
            self.data.close()
        self.data = None


def _cat_tree(values):
    import torch
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=0)
    return type(values[0])(_cat_tree([v[i] for v in values]) for i in range(len(values[0])))


def _physical_spans(indices):
    indices = list(range(28)) if indices is None else list(indices)
    if not indices or indices != sorted(set(indices)) or any(type(i) is not int or i not in range(28) for i in indices):
        raise ValueError('physical query indices must be a nonempty ordered subset of the logical28')
    spans = []
    for index in indices:
        if not spans or index != spans[-1][-1]+1:
            spans.append([])
        spans[-1].append(index)
    return spans
