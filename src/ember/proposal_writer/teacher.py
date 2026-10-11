"""One task trajectory: native demoFM and fixed initial-MT proximal units."""
from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import save_file, load_file
from torch.utils.data import default_collate

from ember.pi05_source_checkpoint import write_json_atomic, read_json
from ember.writer.data import FunctionalQueryDataset
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.model import DirectLoRAParameters
from ember.writer.runtime import autocast

from .contract import OPTIMIZER, seed


def tree_slice(value, start, stop):
    if isinstance(value, torch.Tensor):
        return value[start:stop]
    if isinstance(value, (list, tuple)):
        return type(value)(tree_slice(v, start, stop) for v in value)
    raise TypeError('native flow tuple contains unsupported field')


def tree_cat(values):
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values)
    return type(values[0])(tree_cat([v[i] for v in values]) for i in range(len(values[0])))


class QueryStream:
    def __init__(self, runtime, task, teaching_demo, event_ordinal, *, logical_queries=112,
                 query_demo_indices=None):
        self.task, self.demo, self.ordinal = int(task), int(teaching_demo), int(event_ordinal)
        if logical_queries <= 0 or logical_queries % 28:
            raise ValueError('teacher query count must contain complete registered 28-query flow blocks')
        self.logical_queries = int(logical_queries)
        self.query_demos = tuple(i for i in (range(50) if query_demo_indices is None else query_demo_indices)
                                 if i != self.demo)
        if not self.query_demos or len(set(self.query_demos)) != len(self.query_demos):
            raise ValueError('teacher query episode pool must be explicit and unique')
        self.authority = runtime.tasks[task]
        self.data = FunctionalQueryDataset((self.authority.authority,), demo_indices=tuple(range(50)),
                                           action_chunk_size=50, action_start_offset=1)
        self.rows = self.data.task_episode_rows[task]
        self.orders, self.frames = {}, {}

    def query(self, n):
        cycle, slot = divmod(n, len(self.query_demos))
        if cycle not in self.orders:
            self.orders[cycle] = np.random.default_rng(seed(10, self.task, self.ordinal, cycle)).permutation(self.query_demos)
        demo = int(self.orders[cycle][slot])
        length = self.authority.episode_lengths[demo] - 1
        turn, position = divmod(cycle, length)
        key = demo, turn
        if key not in self.frames:
            self.frames[key] = np.random.default_rng(seed(11, self.task, self.ordinal, demo, turn)).permutation(length)
        return dict(demo=demo, frame=int(self.frames[key][position]))

    def sample(self, runtime, update):
        queries = [self.query(update * self.logical_queries + i) for i in range(self.logical_queries)]
        if any(row['demo'] == self.demo for row in queries):
            raise ValueError('teaching episode actions crossed condition wall')
        batch = default_collate([self.data[self.rows[row['demo']][row['frame']]] for row in queries])
        batch = runtime.processor.training_batch(batch)
        samples, blocks = [], []
        for i in range(self.logical_queries // 28):
            rows = queries[28 * i:28 * (i + 1)]
            flow_seed = task_logical_batch_policy_rng_seed(optimization_seed=7, task_id=self.task,
                task_visit=(self.logical_queries // 28) * update + i, demo_indices=[r['demo'] for r in rows],
                frame_indices=[r['frame'] for r in rows])
            samples.append(flow_sample(runtime.policy, {k: v[28 * i:28 * (i + 1)] for k, v in batch.items()},
                seed=flow_seed, device=runtime.device, random_batch=28, offset=0))
            blocks.append(dict(logical_batch=28, flow_seed=flow_seed))
        sample = FlowSample(tree_cat([s.arguments for s in samples]), torch.cat([s.target for s in samples]),
                            samples[0].action_width)
        return sample, dict(update=update + 1, task_id=self.task, teacher_demo=self.demo,
                            event_ordinal=self.ordinal, queries=queries, blocks=blocks, offset=1)

    def close(self):
        self.data.close()


def save_checkpoint(root, model, optimizer, stream, update, event, *, extra=None):
    destination = Path(root) / 'checkpoints' / f'update_{update:08d}'
    destination.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model().items()}, str(destination / 'lora.safetensors'))
    state = dict(schema_version='ember_full36_common_teacher_checkpoint_v4', optimizer=optimizer.state_dict(),
        next_update=update, sampler=dict(task=stream.task, teacher_demo=stream.demo, event_ordinal=stream.ordinal,
                                       logical_queries=stream.logical_queries, query_demos=stream.query_demos,
                                       next_query=update * stream.logical_queries), names=list(model.names),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.random.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()), topology=dict(world_size=1), event=event, extra=extra)
    torch.save(state, destination / 'state.pt')
    write_json_atomic(destination / 'manifest.json', dict(schema_version=state['schema_version'], complete=True,
        next_update=update, event=event, full_optimizer_rng_sampler=True, physical_world_size=1))
    return destination


def fit_teacher(runtime, event, parent, root, *, microbatch, stop, logical_queries,
                query_demo_indices, checkpoint_updates, resume=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model = DirectLoRAParameters({k: v.detach().float() for k, v in parent.items()}).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    stream = QueryStream(runtime, event['task_id'], event['teacher_demo'], event['event_ordinal'],
                         logical_queries=logical_queries, query_demo_indices=query_demo_indices)
    start = 0
    if resume is not None:
        state = torch.load(Path(resume) / 'state.pt', map_location='cpu', weights_only=False)
        if (state['schema_version'] != 'ember_full36_common_teacher_checkpoint_v4'
                or state['event'] != event or tuple(state['names']) != model.names
                or state['sampler']['logical_queries'] != logical_queries
                or tuple(state['sampler']['query_demos']) != stream.query_demos):
            raise ValueError('teacher resume event or complete factor topology changed')
        weights = load_file(str(Path(resume) / 'lora.safetensors'), device=str(runtime.device))
        with torch.no_grad():
            for name, value in zip(model.names, model.values, strict=True):
                value.copy_(weights[name])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        torch.random.set_rng_state(state['rng']['cpu']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
        start = state['next_update']
    scales = runtime.prox_scales
    valid_coordinates = runtime.lora.parameter_count
    owner = NativeFlowPrediction(runtime.policy)
    try:
        for update in range(start, stop):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            sample, query_event = stream.sample(runtime, update)
            losses = dict(demo=0., proximal=0.)
            for first in range(0, logical_queries, microbatch):
                last = min(first + microbatch, logical_queries)
                part = FlowSample(tree_slice(sample.arguments, first, last), sample.target[first:last], sample.action_width)
                with autocast(runtime.device):
                    prediction = torch.func.functional_call(owner, runtime.task_parameters(model()),
                                                             (part,), strict=False)
                    loss = mean_velocity_loss(prediction, part.target, part.action_width) * ((last - first) / logical_queries)
                loss.backward()
                losses['demo'] += float(loss.detach())
            proximal = sum(((value - parent[name].to(value)) / scales[name]).square().sum()
                           for name, value in model().items()) * (1e-3 / valid_coordinates)
            proximal.backward()
            losses['proximal'] = float(proximal.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(runtime.device)
            record = dict(event_id=event['event_id'], update=update + 1, losses=losses,
                          gradient_norm=float(norm), seconds=time.monotonic() - started, query_event=query_event)
            with (root / 'metrics.jsonl').open('a') as f:
                import json
                f.write(json.dumps(record) + '\n')
            if update + 1 in checkpoint_updates:
                save_checkpoint(root, model, optimizer, stream, update + 1, event)
        return model()
    finally:
        stream.close()
