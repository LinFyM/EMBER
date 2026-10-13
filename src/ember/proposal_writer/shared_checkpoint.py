"""Full fixed-G Writer checkpoints and explicit selector-distribution migration."""
from pathlib import Path
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import read_json,write_json_atomic,capture_rng,restore_rng
from .contract import TASKS,CHECKPOINT_SCHEMA,OPERATOR_CHECKPOINT_SCHEMA,OPERATOR_OBJECTIVE,OPTIMIZER


def endpoint_names(runtime):
    return ['endpoint.'+name for name,_ in runtime.generator.endpoint.named_parameters()]


def load_state(path,runtime):
    path=Path(path).resolve();run=read_json(runtime.root/'run_contract.json')
    migration=run['learning_migration'];manifest=read_json(path/'manifest.json')
    state=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
    original=path==Path(migration['source_checkpoint']).resolve()
    schema=OPERATOR_CHECKPOINT_SCHEMA if original else CHECKPOINT_SCHEMA
    order=list(TASKS) if original else run['training_tasks']
    owner='endpoint' if original else 'frozen'
    if (state['schema_version']!=schema or manifest['schema_version']!=schema or not manifest['complete']
            or not manifest['full_optimizer_rng_sampler'] or not manifest['full_G_theta_baseline']
            or manifest['stage']!=state['stage'] or manifest['next_update']!=state['next_update']
            or manifest['topology']!=state['topology'] or state['topology']['task_order']!=order
            or state['topology']['logical_condition_batch']!=4 or len(state['rank_rng'])!=state['topology']['world_size']
            or state['G_objective']!=OPERATOR_OBJECTIVE or state['G_gradient_owner']!=owner
            or state['psi_version']!=38349 or state['G_sampler']['next_update']!=38349 or state['supervision_version']!=9
            or state['parameter_names']['psi']!=list(dict(runtime.generator.named_parameters()))
            or state['parameter_names']['theta']!=list(dict(runtime.actor.named_parameters()))
            or state['parameter_names']['generator_optimizer']!=endpoint_names(runtime)
            or state['generator_optimizer_parameter_names']!=endpoint_names(runtime)):
        raise ValueError('complete fixed G38349, explicit task scope, optimizer owners and rank RNG required')
    if original:
        if state['stage']!='RL' or state['next_update']!=9 or state['topology']['world_size']!=4:
            raise ValueError('selector migration accepts only the registered complete v10 RL9')
    elif state['learning_migration']!=migration:
        raise ValueError('selector checkpoint distribution/source migration changed')
    runtime.generator.load_state_dict(state['psi']);runtime.actor.load_state_dict(state['theta'])
    runtime.generator_optimizer_state=state['generator_optimizer'];runtime.actor_optimizer_state=state['actor_optimizer']
    runtime.generator_optimizer_parameter_names=state['generator_optimizer_parameter_names']
    for name in ['G_sampler','supervision_version','frozen_frontend_checkpoint','component_migration',
                 'component_rng_sources','component_samplers']:
        setattr(runtime,name,state[name])
    runtime.version=38349;runtime.learning_migration=migration
    runtime.objective_migration=state.get('objective_migration')
    runtime.source_rank_rng=state['rank_rng'] if original else state['source_rank_rng']
    runtime.source_theta_sampler=state['theta_sampler'] if original else state['source_theta_sampler']
    runtime.physical_rng_migration=state.get('physical_rng_migration')
    runtime.freeze_psi();runtime.policy.eval().requires_grad_(False)
    return state


def initialize_selector(runtime,context,root):
    from .learning import Baseline
    run=read_json(Path(root)/'run_contract.json');state=load_state(run['learning_migration']['source_checkpoint'],runtime)
    if not 1<=context.world_size<=4:raise ValueError('logical B4 supports one through four physical ranks')
    baseline=Baseline().to(runtime.device);baseline.load_state_dict(state['baseline'])
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),**OPTIMIZER)
    baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    optimizer=torch.optim.AdamW(runtime.actor.parameters(),**OPTIMIZER)
    optimizer.load_state_dict(state['actor_optimizer'])
    runtime.training_baseline=baseline.eval().requires_grad_(False)
    runtime.training_baseline_optimizer=baseline_optimizer
    runtime.physical_rng_migration=dict(source_world_size=4,current_world_size=context.world_size,
        restored_source_rank=context.rank,source_unassigned_ranks=list(range(context.world_size,4)),
        original_four_rank_states_retained=True,logical_batch=4,bitwise_exact_claim=False)
    restore_rng(state['rank_rng'][context.rank],context)
    write_json_atomic(Path(root)/'code'/f'selector_migration_{context.device.type}_rank_{context.rank:02d}.json',dict(
        complete=True,learning_migration=runtime.learning_migration,physical_rng_migration=runtime.physical_rng_migration,
        F_cursor=runtime.G_sampler['next_update'],bank_version=runtime.supervision_version,
        theta_Adam_restored=True,baseline_Adam_restored=True,original_contract_exact_resume=False))
    return optimizer,baseline


def checkpoint(root,stage,update,runtime,optimizer,context,*,sampler,baseline=None,baseline_optimizer=None,
               generator_optimizer=None):
    if generator_optimizer is not None:raise ValueError('fixed-G selector has no G optimizer update owner')
    baseline=runtime.training_baseline if baseline is None else baseline
    baseline_optimizer=runtime.training_baseline_optimizer if baseline_optimizer is None else baseline_optimizer
    run=read_json(Path(root)/'run_contract.json');destination=Path(root)/'checkpoints'/stage/f'update_{update:08d}'
    rng=capture_rng(context);ranks=[None]*context.world_size if context.is_main else None
    if context.world_size>1:dist.gather_object(rng,ranks,dst=0)
    else:ranks=[rng]
    if context.is_main:
        destination.mkdir(parents=True,exist_ok=False)
        state=dict(schema_version=CHECKPOINT_SCHEMA,stage=stage,next_update=update,G_objective=OPERATOR_OBJECTIVE,
            generation='deterministic_MT_centered_endpoint',G_gradient_owner='frozen',
            psi=runtime.generator.state_dict(),theta=runtime.actor.state_dict(),optimizer=optimizer.state_dict(),
            optimizer_owner='theta',generator_optimizer=runtime.generator_optimizer_state,
            generator_optimizer_parameter_names=endpoint_names(runtime),actor_optimizer=optimizer.state_dict(),
            baseline=baseline.state_dict(),baseline_optimizer=baseline_optimizer.state_dict(),sampler=sampler,
            G_sampler=runtime.G_sampler,theta_sampler=sampler,rank_rng=ranks,
            topology=dict(world_size=context.world_size,task_order=run['training_tasks'],logical_condition_batch=4),
            psi_version=38349,supervision_version=9,learning_migration=runtime.learning_migration,
            source_rank_rng=runtime.source_rank_rng,source_theta_sampler=runtime.source_theta_sampler,
            physical_rng_migration=runtime.physical_rng_migration,
            coordinates=dict(rank=8,valid=runtime.lora.parameter_count,center=str(runtime.root/'initial.safetensors'),
                fixed_bank_scale=str(runtime.bank_scale_path)),fixed_kernel_identity=getattr(runtime,'fixed_kernel_identity',None),
            objective_migration=runtime.objective_migration,frozen_frontend_checkpoint=runtime.frozen_frontend_checkpoint,
            component_migration=runtime.component_migration,component_rng_sources=runtime.component_rng_sources,
            component_samplers={**runtime.component_samplers,'G':runtime.G_sampler,'theta':sampler},
            parameter_names=dict(psi=list(dict(runtime.generator.named_parameters())),theta=list(dict(runtime.actor.named_parameters())),
                generator_optimizer=endpoint_names(runtime),actor_optimizer=list(dict(runtime.actor.named_parameters())),
                baseline=list(dict(baseline.named_parameters()))))
        torch.save(state,destination/'state.pt')
        write_json_atomic(destination/'manifest.json',dict(schema_version=CHECKPOINT_SCHEMA,stage=stage,next_update=update,
            G_objective=OPERATOR_OBJECTIVE,full_optimizer_rng_sampler=True,complete=True,topology=state['topology'],
            G_gradient_owner='frozen',G_next_update=38349,supervision_version=9,full_G_theta_baseline=True,
            independent_F_theta_optimizers=True,frozen_frontend_checkpoint=runtime.frozen_frontend_checkpoint,
            learning_migration=runtime.learning_migration))
    if context.world_size>1:dist.barrier(device_ids=[context.local_rank])
    return destination


def resume_checkpoint(path,stage,runtime,optimizer,context,*,baseline=None,baseline_optimizer=None,
                      generator_optimizer=None):
    state=load_state(path,runtime)
    if state['stage']!=stage or state['topology']['world_size']!=context.world_size or state['optimizer_owner']!='theta':
        raise ValueError('exact stage resume requires saved physical topology and theta owner')
    optimizer.load_state_dict(state['actor_optimizer']);restore_rng(state['rank_rng'][context.rank],context)
    if baseline is not None:baseline.load_state_dict(state['baseline'])
    if baseline_optimizer is not None:baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    return state['next_update']
