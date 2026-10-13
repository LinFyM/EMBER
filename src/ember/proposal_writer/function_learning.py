"""One native ten-flow G learner, with an explicit trainable parameter set."""
from __future__ import annotations

from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.runtime import autocast
from ember.writer.practice.execution import NativeVelocity
from .contract import TASKS, OPTIMIZER, seed
from .function_data import FunctionInputs, CHECKPOINT_SCHEMA
from .learning import logical_tasks, synchronize_gradients, memory_peak
from .runtime import finish_phases, phase


def configure_parameters(runtime, arm):
    if arm not in ('F_only', 'full_G'):
        raise ValueError('registered A/F-only or B/full-G parameter set required')
    runtime.generator.train().requires_grad_(arm == 'full_G')
    runtime.generator.endpoint.requires_grad_(True)
    runtime.actor.eval().requires_grad_(False)
    runtime.policy.eval().requires_grad_(False)
    runtime.psi_frozen = arm == 'F_only'
    runtime.feature_cache.clear()
    runtime.feature_cache_size = 0
    return [(n, p) for n, p in runtime.generator.named_parameters() if p.requires_grad]


def import_source(runtime, root, arm, context):
    source = Path(read_json(Path(root)/'run_contract.json')['source_checkpoint'])
    manifest = read_json(source/'manifest.json')
    state = torch.load(source/'state.pt', map_location='cpu', weights_only=False)
    if (manifest['schema_version'] != 'ember_full_writer_frozen_frontend_endpoint_v10'
            or not manifest['complete'] or state['G_sampler']['next_update'] != 38349
            or state['supervision_version'] != 9
            or state['parameter_names']['psi'] != list(dict(runtime.generator.named_parameters()))):
        raise ValueError('complete original v10 G38349/bank9 initialization required')
    runtime.generator.load_state_dict(state['psi'])
    runtime.version = 38349
    owned = configure_parameters(runtime, arm)
    # Fresh logical stream; old Adam and old rank RNG are deliberately not imported.
    value = seed(213, context.rank)
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    optimizer = torch.optim.AdamW([p for _, p in owned], **OPTIMIZER)
    if optimizer.state:
        raise ValueError('both arms must start with empty fresh AdamW state')
    runtime.function_source = dict(checkpoint=str(source), G_cursor=38349, bank_version=9,
        frontend_checkpoint=state['frozen_frontend_checkpoint'], old_Adam_inherited=False,
        old_theta_training_disabled=True, new_logical_cursor=0, initialization_seed=value)
    return optimizer, [n for n, _ in owned]


def gradient_ownership(runtime, arm):
    groups = {}
    for name, parameter in runtime.generator.named_parameters():
        group = name.split('.')[0]
        record = groups.setdefault(group, dict(trainable=0, with_gradient=0, norm_squared=0.))
        record['trainable'] += int(parameter.requires_grad)
        if parameter.grad is not None:
            if not parameter.requires_grad:
                raise ValueError('gradient reached a frozen G component')
            record['with_gradient'] += 1
            record['norm_squared'] += float(parameter.grad.float().square().sum())
    expected = {'endpoint'} if arm == 'F_only' else {'meta', 'reader', 'static', 'endpoint'}
    if ({g for g, r in groups.items() if r['trainable']} != expected
            or any(groups[g]['norm_squared'] <= 0 for g in expected)
            or any(p.grad is not None for p in runtime.policy.parameters())
            or any(p.grad is not None for p in runtime.actor.parameters())):
        raise ValueError('actual native control consumer lost expected G credit/frozen source')
    if arm == 'full_G':
        meta = runtime.generator.meta
        branches = {name: sum(float(p.grad.float().square().sum()) for p, group in zip(meta.values, meta.groups)
                             if group == name and p.grad is not None) for name in ('gemma', 'action')}
        if any(value <= 0 for value in branches.values()):
            raise ValueError('full-G native credit lost a dual-meta branch')
        groups['meta']['branch_norm_squared'] = branches
    return groups


class NativeFunctionTrainer:
    def __init__(self, runtime, context, root, arm, *, microbatch):
        self.runtime, self.context, self.root, self.arm = runtime, context, Path(root), arm
        self.microbatch = int(microbatch)
        if self.microbatch < 1 or not 1 <= context.world_size <= 4:
            raise ValueError('positive physical query batch and B4-compatible topology required')
        self.optimizer, self.names = import_source(runtime, root, arm, context)
        self.inputs = FunctionInputs(runtime, root)

    def generated(self, condition):
        runtime = self.runtime
        parent = self.inputs.state(condition['parent_weights'])
        memory = None
        cache = self.root/'frozen_F_contexts'/f"{condition['condition_id']}.pt"
        if self.arm == 'F_only' and cache.exists():
            saved = torch.load(cache, map_location='cpu', weights_only=False)
            if saved['source'] != runtime.function_source['checkpoint']:
                raise ValueError('F-only memory came from a different frozen frontend')
            memory = saved['memory'].to(runtime.device)
            features, rows, states = None, None, None
        else:
            history = self.inputs.histories.get(condition['H'])
            rows = history.model_rows(runtime)
            self.inputs.histories.remember_features(condition['H'], history)
            states = history.states
            features = runtime.teaching(condition['task'], condition['teacher_demo'])
            if self.arm == 'F_only':
                with torch.no_grad(), autocast(runtime.device):
                    memory = runtime.generator.context(features, rows, states).detach()
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(source=runtime.function_source['checkpoint'], memory=memory.cpu()), cache)
        with autocast(runtime.device):
            return runtime.generator.compile(parent, features, rows, states, memory=memory)

    def condition(self, condition):
        runtime = self.runtime
        self.inputs.current_task = condition['task']
        # Obtain label-side observations before constructing the trainable graph.
        categories = self.inputs.queries(condition)
        available = [(name, rows) for name, rows in categories if rows]
        generated = self.generated(condition)
        teacher = self.inputs.state(self.inputs.targets[condition['T_target_id']]['weights'])
        ordered = [(name, query) for name, rows in available for query in rows]
        chunks = [ordered[first:first+self.microbatch] for first in range(0, len(ordered), self.microbatch)]
        losses, sources = {}, {}
        for index, entries in enumerate(chunks):
            queries = [query for _, query in entries]
            batch, noise = self.inputs.batch(queries)
            owners = torch.zeros(len(noise), device=runtime.device, dtype=torch.long)
            with phase(runtime, 'functional_native_prefix'), autocast(runtime.device):
                velocity = NativeVelocity(runtime.policy, batch, capture_phi=False)
            with phase(runtime, 'native_function_credit'):
                with torch.no_grad():
                    target = runtime.native_actions([teacher], batch, noise, batch_indices=owners,
                        checkpointed=False, prepared_velocity=velocity).detach()
                prediction = runtime.native_actions([generated], batch, noise, batch_indices=owners,
                    checkpointed=True, prepared_velocity=velocity)
                error = (prediction.float()-target.float()).square().flatten(1).mean(1)
                weights = error.new_tensor([q['weight'] for q in queries])
                loss = (error*weights).sum()/len(available)/4
                loss.backward(retain_graph=index+1 < len(chunks))
            for position, (name, query) in enumerate(entries):
                losses[name] = losses.get(name, 0.)+float(error[position].detach())*query['weight']/len(available)
                sources.setdefault(name, []).append(dict(query['source'], noise_seed=query['noise_seed'], weight=query['weight']))
            del prediction, target, velocity, loss, batch, noise
        return dict(condition_id=condition['condition_id'], task=condition['task'],
            task_visit=condition['task_visit'], T_target_id=condition['T_target_id'],
            losses=losses, category_queries={name: len(rows) for name, rows in categories}, query_sources=sources,
            empty_categories=[name for name, rows in categories if not rows])

    def update(self, conditions):
        tick = time.monotonic()
        before = dict(self.runtime.components)
        self.optimizer.zero_grad(set_to_none=True)
        records = [self.condition(c) for i, c in enumerate(conditions) if i % self.context.world_size == self.context.rank]
        synchronize_gradients(self.runtime.generator, self.context)
        groups = gradient_ownership(self.runtime, self.arm)
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.runtime.generator.parameters() if p.requires_grad],
                                           1., error_if_nonfinite=True)
        self.optimizer.step()
        if self.arm == 'full_G':
            self.runtime.mark_psi_update()
        finish_phases(self.runtime)
        torch.cuda.synchronize(self.runtime.device)
        return dict(seconds=time.monotonic()-tick, conditions=records, gradient_groups=groups,
            gradient_norm=float(norm), peak=memory_peak(self.runtime.device),
            microbatch=self.microbatch, frame_chunk=self.runtime.frame_chunk,
            component_totals=dict(self.runtime.components),
            component_delta={k:v-before.get(k,0.) for k,v in self.runtime.components.items()})

    def save(self, update):
        rng = capture_rng(self.context)
        ranks = [None]*self.context.world_size if self.context.is_main else None
        if self.context.world_size > 1:
            dist.gather_object(rng, ranks, dst=0)
        else:
            ranks = [rng]
        if self.context.is_main:
            directory = self.root/'arms'/self.arm/'checkpoints'/f'update_{update:08d}'
            directory.mkdir(parents=True, exist_ok=False)
            topology = dict(world_size=self.context.world_size, logical_condition_batch=4, task_order=list(TASKS))
            state = dict(schema_version=CHECKPOINT_SCHEMA, arm=self.arm, next_update=update,
                psi=self.runtime.generator.state_dict(), optimizer=self.optimizer.state_dict(),
                optimizer_parameter_names=self.names, rank_rng=ranks, topology=topology,
                source=self.runtime.function_source, psi_version=self.runtime.version,
                sampler=dict(next_update=update, task_seed_domain=44, full36_round_updates=9,
                             condition_order='MT_H1 video46..49 then G_H2 video46..49'),
                scheduler=None, scaler=None, frozen_source=True, frozen_targets=True)
            torch.save(state, directory/'state.pt')
            write_json_atomic(directory/'manifest.json', dict(complete=True, schema_version=CHECKPOINT_SCHEMA,
                arm=self.arm, next_update=update, topology=topology, full_G_optimizer_sampler_RNG=True,
                parameter_names=self.names, source=self.runtime.function_source,
                checkpoint_role='sole_scientific_endpoint' if update == 72 else 'recovery_only'))
        if self.context.world_size > 1:
            dist.barrier(device_ids=[self.context.local_rank])

    def resume(self, directory):
        state = torch.load(Path(directory)/'state.pt', map_location='cpu', weights_only=False)
        if (state['schema_version'] != CHECKPOINT_SCHEMA or state['arm'] != self.arm
                or state['optimizer_parameter_names'] != self.names or state['source'] != self.runtime.function_source
                or state['topology']['world_size'] != self.context.world_size):
            raise ValueError('resume requires complete matching scientific stream and rank RNG')
        self.runtime.generator.load_state_dict(state['psi'])
        self.runtime.version = state['psi_version']
        self.optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rank_rng'][self.context.rank], self.context)
        return state['next_update']


def learn(args, runtime, context):
    trainer = NativeFunctionTrainer(runtime, context, args.root, args.arm, microbatch=args.microbatch)
    initialize_deferred_process_group(context, rendezvous_root=args.root/'rendezvous'/args.arm)
    directory = args.root/'arms'/args.arm
    directory.mkdir(parents=True, exist_ok=True)
    if args.profile:
        # Genuine B4 with the four longest G-H2 conditions; no new episodes.
        candidates = sorted((c for c in trainer.inputs.manifest['conditions'] if c['task_visit'] >= 4),
                            key=lambda c: (c['sampled_teacher_frames'], c['H']['records']), reverse=True)
        conditions, selected = [], set()
        for c in candidates:
            if c['task'] not in selected:
                selected.add(c['task']); conditions.append(c)
            if len(conditions) == 4:
                break
        results = []
        for index, microbatch in enumerate((args.microbatch, args.profile_microbatch)):
            trainer.microbatch = microbatch
            torch.cuda.reset_peak_memory_stats(runtime.device)
            record = trainer.update(conditions)
            results.append(record)
            write_json_atomic(directory/'profile'/f'configuration_{index}_rank_{context.rank}.json',
                              dict(record, discarded=True, effective_updates=0))
        write_json_atomic(directory/'profile'/f'complete_rank_{context.rank}.json', dict(complete=True,
            discarded_B4=2, effective_updates=0, source_restored_at_formal_start=True, configurations=results))
        return
    start = trainer.resume(args.resume) if args.resume else 0
    for update in range(start, 72):
        visit = update//9
        conditions = [trainer.inputs.conditions[(task, visit)] for task in logical_tasks(update)]
        record = trainer.update(conditions)
        with (directory/f'metrics_rank_{context.rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(record, update=update+1))+'\n')
        if update+1 in (18, 36, 54, 72):
            trainer.save(update+1)
    write_json_atomic(directory/'complete'/f'train_rank_{context.rank}.json', dict(complete=True,
        next_update=72, all288_conditions=True, arm=args.arm, physical_world_size=context.world_size))
