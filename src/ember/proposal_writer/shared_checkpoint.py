"""Complete Writer state with independent endpoint, Actor and baseline owners."""
from pathlib import Path

import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import read_json, write_json_atomic, capture_rng, restore_rng
from .contract import TASKS, CHECKPOINT_SCHEMA, G_OBJECTIVE, OPERATOR_CHECKPOINT_SCHEMA, OPERATOR_OBJECTIVE, OPTIMIZER


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
            fixed_kernel_identity=getattr(runtime,'fixed_kernel_identity',None), objective_migration=runtime.objective_migration,
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


def objective_identity(path,state,runtime):
    """Accept active FM or the sole registered complete operator-source import."""
    migration=read_json(runtime.root/'run_contract.json').get('objective_migration')
    old_source=state['schema_version']==OPERATOR_CHECKPOINT_SCHEMA
    expected_schema=OPERATOR_CHECKPOINT_SCHEMA if old_source else CHECKPOINT_SCHEMA
    expected_objective=OPERATOR_OBJECTIVE if old_source else G_OBJECTIVE
    # Old v10 is imported only from the one explicitly registered complete
    # RL9/F38349 source, including the frozen parent readout. No shape fallback.
    if old_source and (not migration or path!=Path(migration['source_checkpoint']).resolve()
            or migration['source_schema']!=expected_schema or migration['source_objective']!=expected_objective
            or migration['destination_objective']!=G_OBJECTIVE or state['stage']!='RL' or state['next_update']!=9
            or state['psi_version']!=38349 or state['supervision_version']!=9):
        raise ValueError('old operator checkpoint requires the exact registered objective migration source')
    return old_source,migration,expected_schema,expected_objective


def load_state(path, runtime):
    path=Path(path).resolve();manifest=read_json(path/'manifest.json')
    state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
    old_source,migration,expected_schema,expected_objective=objective_identity(path,state,runtime)
    if (state['schema_version']!=expected_schema or state['G_objective']!=expected_objective
            or manifest['schema_version']!=expected_schema or not manifest['complete']
            or not manifest['full_optimizer_rng_sampler'] or not manifest['full_G_theta_baseline']
            or manifest['next_update']!=state['next_update'] or manifest['stage']!=state['stage']
            or manifest['topology']!=state['topology'] or state['G_gradient_owner']!='endpoint'
            or state['topology']['task_order']!=list(TASKS) or state['topology']['logical_condition_batch']!=4
            or len(state['rank_rng'])!=state['topology']['world_size']
            or state['parameter_names']['psi']!=list(dict(runtime.generator.named_parameters()))
            or state['parameter_names']['theta']!=list(dict(runtime.actor.named_parameters()))
            or state['parameter_names']['generator_optimizer']!=endpoint_names(runtime)
            or len(state['generator_optimizer']['param_groups'][0]['params'])!=len(endpoint_names(runtime))
            or state['G_sampler']['next_update']!=state['psi_version']
            or state['supervision_version']!=9):
        raise ValueError('full Writer schema, independent owners, samplers or complete topology changed')
    runtime.generator.load_state_dict(state['psi']);runtime.actor.load_state_dict(state['theta'])
    for name in ('generator_optimizer_state','actor_optimizer_state'):
        setattr(runtime,name,state['generator_optimizer' if name.startswith('generator') else 'actor_optimizer'])
    for name in ('G_sampler','supervision_version','frozen_frontend_checkpoint','component_migration',
                 'component_rng_sources','component_samplers'):
        setattr(runtime,name,state[name])
    runtime.generator_optimizer_parameter_names=state['generator_optimizer_parameter_names']
    runtime.version=state['psi_version']
    runtime.objective_migration=dict(migration) if old_source else state['objective_migration']
    return state


def initialize_native(runtime, context, root):
    """Controlled complete-state objective migration with frozen theta/baseline."""
    from .learning import Baseline
    run=read_json(Path(root)/'run_contract.json')
    binding=run['objective_migration']
    state=load_state(binding['source_checkpoint'],runtime)
    if context.world_size!=state['topology']['world_size'] or context.world_size!=4:
        raise ValueError('registered source rank RNG must be restored on the complete four-rank topology')
    baseline=Baseline().to(runtime.device)
    baseline.load_state_dict(state['baseline'],strict=True)
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),**OPTIMIZER)
    baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    runtime.training_baseline=baseline.eval().requires_grad_(False)
    runtime.training_baseline_optimizer=baseline_optimizer
    runtime.generator.eval().requires_grad_(False);runtime.actor.eval().requires_grad_(False)
    runtime.policy.eval().requires_grad_(False)
    runtime.G_sampler={**runtime.G_sampler,'native_FM':dict(event_ordinal=20261013,
        query_demos=list(range(46)),logical_queries=28,task_visits={str(t):0 for t in TASKS})}
    restore_rng(state['rank_rng'][context.rank],context)
    write_json_atomic(Path(root)/'code'/f'G_objective_migration_rank_{context.rank:02d}.json',dict(
        complete=True,migration=runtime.objective_migration,F_cursor=runtime.version,
        restored_rng_rank=context.rank,world_size=context.world_size,bank_version=runtime.supervision_version,
        endpoint_parameter_tensors=len(endpoint_names(runtime)),old_contract_exact_resume=False,
        frozen_Actor_baseline=True,full_source_state_restored=True))
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
