"""One same-T G learner, with operator-only and operator/native objectives."""
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
from .learning import logical_tasks, synchronize_gradients, memory_peak, OperatorObjective
from .runtime import finish_phases, phase


def configure_parameters(runtime, arm):
    if arm not in ('operator_only', 'operator_plus_native'):
        raise ValueError('registered D/operator-only or C/operator-plus-native arm required')
    runtime.generator.train().requires_grad_(True)
    runtime.generator.endpoint.requires_grad_(True)
    runtime.actor.eval().requires_grad_(False)
    runtime.policy.eval().requires_grad_(False)
    runtime.psi_frozen = False
    runtime.feature_cache.clear()
    runtime.feature_cache_size = 0
    return [(n, p) for n, p in runtime.generator.named_parameters() if p.requires_grad]


def learning_window(root):
    contract = read_json(Path(root)/'run_contract.json')
    window = contract.get('learning_window', dict(start=0, stop=72, condition_offset=0))
    continuation = contract['task'] == 'operator_D_adjacent_stability'
    if (continuation and (window != dict(start=72, stop=144, condition_offset=72)
            or set(contract['function_objectives']) != {'operator_only'})) or (
            not continuation and window != dict(start=0, stop=72, condition_offset=0)):
        raise ValueError('only the registered D72-to144 condition repeat is authorized')
    return window


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
    # Initialize the optimizer container; a continuation restores its complete state before any update.
    value = seed(213, context.rank)
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    optimizer = torch.optim.AdamW([p for _, p in owned], **OPTIMIZER)
    if optimizer.state:
        raise ValueError('both arms must start with empty fresh AdamW state')
    objective = read_json(Path(root)/'run_contract.json')['function_objectives'][arm]
    expected_native = 0. if arm == 'operator_only' else 69.25903014646993
    if objective != dict(operator_weight=1., native_weight=expected_native):
        raise ValueError('registered same-T operator/native objective changed')
    runtime.function_source = dict(checkpoint=str(source), G_cursor=38349, bank_version=9,
        frontend_checkpoint=state['frozen_frontend_checkpoint'], old_Adam_inherited=False,
        old_theta_training_disabled=True, new_logical_cursor=0, initialization_seed=value)
    runtime.function_source['objective'] = objective
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
    expected = {'meta', 'reader', 'static', 'endpoint'}
    if ({g for g, r in groups.items() if r['trainable']} != expected
            or any(groups[g]['norm_squared'] <= 0 for g in expected)
            or any(p.grad is not None for p in runtime.policy.parameters())
            or any(p.grad is not None for p in runtime.actor.parameters())):
        raise ValueError('actual native control consumer lost expected G credit/frozen source')
    if arm in ('operator_only', 'operator_plus_native'):
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
        self.window = learning_window(root)
        self.resume_provenance = None
        self.parent_rank_rng = None
        self.operator = OperatorObjective(runtime, root)
        self.native_weight = runtime.function_source['objective']['native_weight']

    def generated(self, condition):
        return self.inputs.compile(condition)

    def condition(self, condition):
        runtime = self.runtime
        self.inputs.current_task = condition['task']
        # Obtain label-side observations before constructing the trainable graph.
        categories = self.inputs.queries(condition)
        available = [(name, rows) for name, rows in categories if rows]
        generated = self.generated(condition)
        teacher = self.inputs.state(self.inputs.targets[condition['T_target_id']]['weights'])
        operator_loss, target_energy = self.operator(generated, teacher)
        (operator_loss/4).backward(retain_graph=bool(self.native_weight))
        ordered = [(name, query) for name, rows in available for query in rows]
        chunks = ([ordered[first:first+self.microbatch] for first in range(0, len(ordered), self.microbatch)]
                  if self.native_weight else [])
        losses, sources = {}, {name: [dict(q['source'], noise_seed=q['noise_seed'], weight=q['weight'])
                                    for q in rows] for name, rows in available}
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
                loss = self.native_weight*(error*weights).sum()/len(available)/4
                loss.backward(retain_graph=index+1 < len(chunks))
            for position, (name, query) in enumerate(entries):
                losses[name] = losses.get(name, 0.)+float(error[position].detach())*query['weight']/len(available)
            del prediction, target, velocity, loss, batch, noise
        native_loss = sum(losses.values()) if self.native_weight else None
        return dict(condition_id=condition['condition_id'], task=condition['task'],
            task_visit=condition['task_visit'], T_target_id=condition['T_target_id'],
            operator_loss=float(operator_loss.detach()), operator_target_energy=float(target_energy.detach()),
            operator_applications=1, native_loss=native_loss, native_weight=self.native_weight,
            combined_loss=float(operator_loss.detach())+self.native_weight*(native_loss or 0.),
            native_student_queries=len(ordered) if self.native_weight else 0,
            native_target_queries=len(ordered) if self.native_weight else 0,
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
                             condition_order='MT_H1 video46..49 then G_H2 video46..49',
                             condition_offset=self.window['condition_offset'],
                             next_condition_cursor=update-self.window['condition_offset'],
                             learning_window=self.window),
                resume_provenance=self.resume_provenance, parent_rank_rng=self.parent_rank_rng,
                scheduler=None, scaler=None, frozen_source=True, frozen_targets=True)
            torch.save(state, directory/'state.pt')
            write_json_atomic(directory/'manifest.json', dict(complete=True, schema_version=CHECKPOINT_SCHEMA,
                arm=self.arm, next_update=update, topology=topology, full_G_optimizer_sampler_RNG=True,
                parameter_names=self.names, source=self.runtime.function_source,
                resume_provenance=self.resume_provenance,
                checkpoint_role='sole_scientific_endpoint' if update == self.window['stop'] else 'recovery_only'))
        if self.context.world_size > 1:
            dist.barrier(device_ids=[self.context.local_rank])

    def resume(self, directory):
        state = torch.load(Path(directory)/'state.pt', map_location='cpu', weights_only=False)
        manifest = read_json(Path(directory)/'manifest.json')
        contract = read_json(self.root/'run_contract.json')
        old_world = state['topology']['world_size']
        expected_source = dict(self.runtime.function_source, initialization_seed=seed(213, 0))
        if (not manifest['complete'] or manifest['next_update'] != state['next_update']
                or state['schema_version'] != CHECKPOINT_SCHEMA or state['arm'] != self.arm
                or state['optimizer_parameter_names'] != self.names or state['source'] != expected_source
                or state['topology']['logical_condition_batch'] != 4 or state['topology']['task_order'] != list(TASKS)
                or len(state['rank_rng']) != old_world or state['sampler']['task_seed_domain'] != 44
                or state['sampler']['full36_round_updates'] != 9
                or state['sampler']['condition_order'] != 'MT_H1 video46..49 then G_H2 video46..49'
                or state['scheduler'] is not None or state['scaler'] is not None
                or not state['frozen_source'] or not state['frozen_targets']):
            raise ValueError('resume requires the complete matching source, optimizer and logical stream')
        parent = contract.get('continuation_checkpoint')
        from_parent = parent is not None and Path(directory).resolve() == Path(parent).resolve()
        if from_parent:
            if (state['next_update'] != self.window['start'] or state['sampler']['next_update'] != 72
                    or state['sampler'].get('condition_offset', 0) != 0 or self.arm != 'operator_only'
                    or len(state['optimizer']['state']) != 485 or state['psi_version'] != 38421):
                raise ValueError('D continuation must enter from its complete original72 boundary')
        elif (state['sampler'].get('learning_window', dict(start=0, stop=72, condition_offset=0)) != self.window
                or state['sampler']['next_update'] != state['next_update']
                or state['sampler'].get('next_condition_cursor', state['next_update'])
                   != state['next_update']-self.window['condition_offset']):
            raise ValueError('resume cursor belongs to another registered condition window')
        if old_world != self.context.world_size and not (from_parent and contract.get('allow_boundary_topology_migration')):
            raise ValueError('physical topology migration needs the registered complete parent boundary')
        self.runtime.generator.load_state_dict(state['psi'])
        self.runtime.version = state['psi_version']
        self.runtime.function_source = state['source']
        self.optimizer.load_state_dict(state['optimizer'])
        if any(group[key] != value for group in self.optimizer.param_groups for key, value in OPTIMIZER.items()):
            raise ValueError('continuation changed the registered AdamW hyperparameters')
        if self.context.rank < old_world:
            restore_rng(state['rank_rng'][self.context.rank], self.context)
        self.parent_rank_rng = state['rank_rng'] if from_parent else state.get('parent_rank_rng')
        self.resume_provenance = (dict(checkpoint=str(Path(directory).resolve()),
            old_topology=state['topology'], new_topology=dict(world_size=self.context.world_size, logical_condition_batch=4),
            optimizer_restored=True, global_cursor=state['next_update'], condition_cursor=0,
            bitwise_exact_claim=False, rank_rng_sources=[dict(rank=i, parent_rank=i if i < old_world else None,
                initialization_seed=None if i < old_world else seed(213, i)) for i in range(self.context.world_size)])
            if from_parent else state.get('resume_provenance'))
        if self.context.is_main:
            write_json_atomic(self.root/'arms'/self.arm/'resume_consumption.json', dict(complete=True,
                checkpoint=str(directory), next_update=state['next_update'], psi_version=self.runtime.version,
                optimizer_states=len(self.optimizer.state), resume_provenance=self.resume_provenance))
        return state['next_update']


def learn(args, runtime, context):
    trainer = NativeFunctionTrainer(runtime, context, args.root, args.arm, microbatch=args.microbatch)
    initialize_deferred_process_group(context, rendezvous_root=args.root/'rendezvous'/args.arm)
    directory = args.root/'arms'/args.arm
    directory.mkdir(parents=True, exist_ok=True)
    start = trainer.resume(args.resume) if args.resume else 0
    if (trainer.window['start'] and not args.resume) or not trainer.window['start'] <= start <= trainer.window['stop']:
        raise ValueError('registered continuation requires its full checkpoint and global cursor')
    for update in range(start, trainer.window['stop']):
        condition_cursor = update-trainer.window['condition_offset']
        visit = condition_cursor//9
        conditions = [trainer.inputs.conditions[(task, visit)] for task in logical_tasks(condition_cursor)]
        record = trainer.update(conditions)
        with (directory/f'metrics_rank_{context.rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(record, update=update+1, global_update_index=update,
                condition_update_index=condition_cursor))+'\n')
        if update+1 in tuple(trainer.window['start']+i for i in (18,36,54,72)):
            trainer.save(update+1)
    write_json_atomic(directory/'complete'/f'train_rank_{context.rank}.json', dict(complete=True,
        next_update=trainer.window['stop'], all288_conditions=True, arm=args.arm,
        learning_window=trainer.window, physical_world_size=context.world_size))
