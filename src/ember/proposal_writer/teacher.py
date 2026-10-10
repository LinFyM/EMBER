"""Training-only endpoint optimization with actual recovery/keep labels."""
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
from ember.writer.practice import processed
from ember.writer.runtime import autocast

from .contract import SOURCE, OPTIMIZER, seed


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
    def __init__(self, runtime, task, teaching_demo, event_ordinal):
        self.task, self.demo, self.ordinal = int(task), int(teaching_demo), int(event_ordinal)
        self.authority = runtime.tasks[task]
        self.data = FunctionalQueryDataset((self.authority.authority,), demo_indices=tuple(range(50)),
                                           action_chunk_size=50, action_start_offset=1)
        self.rows = self.data.task_episode_rows[task]
        self.orders, self.frames = {}, {}

    def query(self, n):
        cycle, slot = divmod(n, 49)
        if cycle not in self.orders:
            pool = [i for i in range(50) if i != self.demo]
            self.orders[cycle] = np.random.default_rng(seed(10, self.task, self.ordinal, cycle)).permutation(pool)
        demo = int(self.orders[cycle][slot])
        length = self.authority.episode_lengths[demo] - 1
        turn, position = divmod(cycle, length)
        key = demo, turn
        if key not in self.frames:
            self.frames[key] = np.random.default_rng(seed(11, self.task, self.ordinal, demo, turn)).permutation(length)
        return dict(demo=demo, frame=int(self.frames[key][position]))

    def sample(self, runtime, update):
        queries = [self.query(update * 112 + i) for i in range(112)]
        if any(row['demo'] == self.demo for row in queries):
            raise ValueError('teaching episode actions crossed condition wall')
        batch = default_collate([self.data[self.rows[row['demo']][row['frame']]] for row in queries])
        batch = runtime.processor.training_batch(batch)
        samples, blocks = [], []
        for i in range(4):
            rows = queries[28 * i:28 * (i + 1)]
            flow_seed = task_logical_batch_policy_rng_seed(optimization_seed=7, task_id=self.task,
                task_visit=4 * update + i, demo_indices=[r['demo'] for r in rows],
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


def functional_labels(runtime, runner, event, history, task, expert):
    """Recovery only from this live history; global pools capped at 4/16."""
    rec, keep, validations = [], [], []
    successful = [row for row in history.records if
        history.episodes[row['episode']]['success'] and row['parameter_ref'] == event['parent_ref']]
    failed = [row for row in history.records if not history.episodes[row['episode']]['success']]
    def spaced(rows, cap):
        return [rows[i] for i in np.floor(np.linspace(0, len(rows) - 1, min(cap, len(rows)))).astype(int)] if rows else []
    for row in spaced(successful, 16):
        batch = processed(runtime, history.observations[row['pre']], task['language'])
        noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(row['noise_seed']))
        keep.append(dict(batch={k: v.cpu() for k, v in batch.items()}, noise=noise,
            target=row['normalized_actions'][None], mask=torch.ones(1, 5, 7, dtype=torch.bool),
            source_episode=row['episode'], source_step=row['step'], actual_parent_function=True))
    for ordinal, row in enumerate(spaced(failed, 4)):
        if row.get('snapshot') is None or expert is None:
            continue
        episode = history.episodes[row['episode']]
        batch = processed(runtime, history.observations[row['pre']], task['language'])
        root = seed(12, event['task_id'], event['event_ordinal'], row['episode'], ordinal)
        snapshot = row['snapshot']
        remaining = task['horizon'] - snapshot['episode_control_steps']
        request = dict(task=task, state=expert, parameter_ref=f"recovery_expert_{event['task_id']}",
            state_id=episode['init_state_id'], noise_root=root, episode_id=f"{event['event_id']}_rec{ordinal}",
            remaining=remaining, snapshot=snapshot, snapshots=False, uses_teaching=False)
        result = next(runner.run([request]))
        validations.append(dict(episode=row['episode'], step=row['step'], success=result['row']['success'],
                                row=result['row'], actual_snapshot=True))
        if result['row']['success'] and result['history'].records:
            first = result['history'].records[0]
            noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(first['noise_seed']))
            rec.append(dict(batch={k: v.cpu() for k, v in batch.items()}, noise=noise,
                target=first['normalized_actions'][None], mask=first['executed'][None, :, None].expand(1, 5, 7),
                source_episode=row['episode'], source_step=row['step'], verified_remaining_horizon_success=True))
    return dict(rec=rec, keep=keep, recovery_validations=validations)


def save_checkpoint(root, model, optimizer, stream, update, event, *, extra=None):
    destination = Path(root) / 'checkpoints' / f'update_{update:08d}'
    destination.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model().items()}, str(destination / 'lora.safetensors'))
    state = dict(schema_version='ember_parameter_teacher_checkpoint_v1', optimizer=optimizer.state_dict(),
        next_update=update, sampler=dict(task=stream.task, teacher_demo=stream.demo, event_ordinal=stream.ordinal,
                                       next_query=update * 112), names=list(model.names),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.random.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()), topology=dict(world_size=1), event=event, extra=extra)
    torch.save(state, destination / 'state.pt')
    write_json_atomic(destination / 'manifest.json', dict(schema_version=state['schema_version'], complete=True,
        next_update=update, event=event, full_optimizer_rng_sampler=True, physical_world_size=1))
    return destination


def function_credit(runtime,model,items,*,microbatch=4):
    """Physical batching only; retain the uniform mean of masked per-item losses."""
    total=0.
    for first in range(0,len(items),microbatch):
        part=items[first:first+microbatch]
        batch={k:torch.cat([v['batch'][k] for v in part]).to(runtime.device) for k in part[0]['batch']}
        noise=torch.cat([v['noise'] for v in part]).to(runtime.device)
        pred=runtime.native_actions([model()],batch,noise,
            batch_indices=torch.zeros(len(part),dtype=torch.long,device=runtime.device))
        targets=torch.cat([v['target'] for v in part]).to(runtime.device)
        mask=torch.cat([v['mask'] for v in part]).to(runtime.device)
        error=(pred.float()-targets.float()).square().masked_fill(~mask,0)
        value=(error.flatten(1).sum(1)/mask.flatten(1).sum(1)).sum()/len(items)
        value.backward();total+=float(value.detach())
    return total


def fit_teacher(runtime, event, parent, labels, root, *,microbatch=28,stop=480,resume=None,function_microbatch=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model = DirectLoRAParameters({k: v.detach().float() for k, v in parent.items()}).to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
    stream = QueryStream(runtime, event['task_id'], event['teacher_demo'], event['event_ordinal'])
    start = 0
    if resume is not None:
        state = torch.load(Path(resume) / 'state.pt', map_location='cpu', weights_only=False)
        if state['event'] != event or tuple(state['names']) != model.names:
            raise ValueError('teacher resume event or complete factor topology changed')
        weights = load_file(str(Path(resume) / 'lora.safetensors'), device=str(runtime.device))
        with torch.no_grad():
            for name, value in zip(model.names, model.values, strict=True):
                value.copy_(weights[name])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        torch.random.set_rng_state(state['rng']['cpu']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
        start = state['next_update']
    scales = {name: runtime.generator.layout.scale[first, 0].detach() for name, _, _, _, first, _ in runtime.generator.layout.parts}
    owner = NativeFlowPrediction(runtime.policy)
    try:
        for update in range(start, stop):
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            sample, query_event = stream.sample(runtime, update)
            losses = dict(demo=0., rec=None, keep=None, proximal=0.)
            for first in range(0, 112, microbatch):
                last = min(first + microbatch, 112)
                part = FlowSample(tree_slice(sample.arguments, first, last), sample.target[first:last], sample.action_width)
                with autocast(runtime.device):
                    prediction = torch.func.functional_call(owner, {'policy.' + k: v for k, v in model().items()},
                                                             (part,), strict=False)
                    loss = mean_velocity_loss(prediction, part.target, part.action_width) * ((last - first) / 112)
                loss.backward()
                losses['demo'] += float(loss.detach())
            for kind in ('rec', 'keep'):
                pool = labels[kind]
                if not pool:
                    continue
                rng = np.random.default_rng(seed(13 if kind == 'rec' else 14, event['task_id'], event['event_ordinal'], update))
                chosen = rng.choice(len(pool), min(4, len(pool)), replace=False).tolist()
                losses[kind]=function_credit(runtime,model,[pool[i] for i in chosen],
                    microbatch=function_microbatch)
            proximal = sum(((value - parent[name].to(value)) / scales[name]).square().sum()
                           for name, value in model().items()) * (1e-3 / 10297344)
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
            if update + 1 in (160, 480):
                save_checkpoint(root, model, optimizer, stream, update + 1, event)
        return model()
    finally:
        stream.close()


def endpoint_distribution(selection_rows, parent_rows):
    parent = {row['init_state_id']: int(row['success']) for row in parent_rows}
    legal, preferred, effects = [], [], {}
    for update, rows in selection_rows.items():
        if {row['init_state_id'] for row in rows} != set(parent):
            raise ValueError('teacher preference lost independent paired selection pool')
        gains = sum(int(row['success']) - parent[row['init_state_id']] for row in rows) / len(rows)
        legal.append(int(update)); effects[int(update)] = gains
        if gains > 0:
            preferred.append(int(update))
    probability = {key: .25 / len(legal) + (.75 / len(preferred) if key in preferred else 0)
                   for key in legal} if preferred else {key: 1 / len(legal) for key in legal}
    return dict(legal_nodes=legal, preferred_nodes=preferred, selection_gain=effects,
                probabilities=probability, missing_labels=not legal)
