"""Complete Writer state with independent endpoint, Actor and baseline owners."""
from pathlib import Path

import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from .contract import TASKS, CHECKPOINT_SCHEMA, G_OBJECTIVE


def endpoint_names(runtime):
    return ['endpoint.'+name for name, _ in runtime.generator.endpoint.named_parameters()]


def checkpoint(root, stage, update, runtime, optimizer, context, *, sampler, baseline=None, baseline_optimizer=None,
               generator_optimizer=None):
    destination = Path(root)/'checkpoints'/stage/f'update_{update:08d}'
    rng = capture_rng(context)
    ranks = [None]*context.world_size if context.is_main else None
    if context.world_size > 1: dist.gather_object(rng, ranks, dst=0)
    else: ranks = [rng]
    baseline = baseline if baseline is not None else runtime.training_baseline
    baseline_optimizer = baseline_optimizer if baseline_optimizer is not None else runtime.training_baseline_optimizer
    F_state = (generator_optimizer.state_dict() if generator_optimizer is not None else
               optimizer.state_dict() if stage=='G' else runtime.generator_optimizer_state)
    theta_state = optimizer.state_dict() if stage!='G' else runtime.actor_optimizer_state
    theta_sampler = sampler if stage!='G' else runtime.component_samplers['theta']
    if context.is_main:
        destination.mkdir(parents=True, exist_ok=False)
        state = dict(schema_version=CHECKPOINT_SCHEMA, stage=stage, next_update=update,
            G_objective=G_OBJECTIVE, generation='deterministic_MT_centered_endpoint', G_gradient_owner='endpoint',
            psi=runtime.generator.state_dict(), theta=runtime.actor.state_dict(),
            optimizer=optimizer.state_dict(), optimizer_owner='endpoint' if stage=='G' else 'theta',
            generator_optimizer=F_state, actor_optimizer=theta_state,
            generator_optimizer_parameter_names=endpoint_names(runtime),
            baseline=baseline.state_dict(), baseline_optimizer=baseline_optimizer.state_dict(),
            sampler=sampler, G_sampler=runtime.G_sampler, theta_sampler=theta_sampler,
            rank_rng=ranks, topology=dict(world_size=context.world_size, task_order=list(TASKS), logical_condition_batch=4),
            psi_version=runtime.version, supervision_version=runtime.supervision_version,
            coordinates=dict(rank=8, valid=runtime.lora.parameter_count, center=str(runtime.root/'initial.safetensors'),
                             fixed_bank_scale=str(runtime.bank_scale_path)),
            fixed_kernel_identity=getattr(runtime,'fixed_kernel_identity',None), objective_migration=None,
            frozen_frontend_checkpoint=runtime.frozen_frontend_checkpoint,
            component_migration=runtime.component_migration, component_rng_sources=runtime.component_rng_sources,
            component_samplers={**runtime.component_samplers,'G':runtime.G_sampler,'theta':theta_sampler},
            parameter_names=dict(psi=list(dict(runtime.generator.named_parameters())),
                theta=list(dict(runtime.actor.named_parameters())), generator_optimizer=endpoint_names(runtime),
                actor_optimizer=list(dict(runtime.actor.named_parameters())), baseline=list(dict(baseline.named_parameters()))))
        torch.save(state, destination/'state.pt')
        write_json_atomic(destination/'manifest.json',dict(schema_version=CHECKPOINT_SCHEMA, stage=stage,
            next_update=update, G_objective=G_OBJECTIVE, full_optimizer_rng_sampler=True, complete=True,
            topology=state['topology'],G_gradient_owner='endpoint',G_next_update=runtime.G_sampler['next_update'],
            supervision_version=runtime.supervision_version, full_G_theta_baseline=True,
            independent_F_theta_optimizers=True, frozen_frontend_checkpoint=runtime.frozen_frontend_checkpoint))
    if context.world_size > 1: dist.barrier(device_ids=[context.local_rank])
    return destination


def load_state(path, runtime):
    path=Path(path).resolve();manifest=read_json(path/'manifest.json')
    state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
    if (state['schema_version']!=CHECKPOINT_SCHEMA or state['G_objective']!=G_OBJECTIVE
            or manifest['schema_version']!=CHECKPOINT_SCHEMA or not manifest['complete']
            or not manifest['full_optimizer_rng_sampler'] or not manifest['full_G_theta_baseline']
            or manifest['next_update']!=state['next_update'] or manifest['stage']!=state['stage']
            or manifest['topology']!=state['topology'] or state['G_gradient_owner']!='endpoint'
            or state['topology']['task_order']!=list(TASKS) or state['topology']['logical_condition_batch']!=4
            or len(state['rank_rng'])!=state['topology']['world_size']
            or state['parameter_names']['psi']!=list(dict(runtime.generator.named_parameters()))
            or state['parameter_names']['theta']!=list(dict(runtime.actor.named_parameters()))
            or state['parameter_names']['generator_optimizer']!=endpoint_names(runtime)
            or len(state['generator_optimizer']['param_groups'][0]['params'])!=len(endpoint_names(runtime))
            or state['G_sampler']['next_update']!=state['psi_version']):
        raise ValueError('full Writer schema, independent owners, samplers or complete topology changed')
    runtime.generator.load_state_dict(state['psi']);runtime.actor.load_state_dict(state['theta'])
    for name in ('generator_optimizer_state','actor_optimizer_state'):
        setattr(runtime,name,state['generator_optimizer' if name.startswith('generator') else 'actor_optimizer'])
    for name in ('G_sampler','supervision_version','frozen_frontend_checkpoint','component_migration',
                 'component_rng_sources','component_samplers'):
        setattr(runtime,name,state[name])
    runtime.generator_optimizer_parameter_names=state['generator_optimizer_parameter_names']
    runtime.version=state['psi_version'];runtime.objective_migration=None
    return state


def resume_checkpoint(path, stage, runtime, optimizer, context, *, baseline=None, baseline_optimizer=None,
                      generator_optimizer=None):
    state=load_state(path,runtime)
    if (state['stage']!=stage or state['topology']['world_size']!=context.world_size
            or state['optimizer_owner']!=('endpoint' if stage=='G' else 'theta')):
        raise ValueError('resume stage/topology/owner changed; use explicit component migration at a complete boundary')
    optimizer.load_state_dict(state['optimizer'])
    if generator_optimizer is not None:generator_optimizer.load_state_dict(state['generator_optimizer'])
    restore_rng(state['rank_rng'][context.rank],context)
    if baseline is not None:
        baseline.load_state_dict(state['baseline'])
        if baseline_optimizer is not None:baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    return state['next_update']
