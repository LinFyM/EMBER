"""Full36 common-target G, supervised final use and conditional-continue actor training."""
from __future__ import annotations

from pathlib import Path
import time

import torch

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group

from .contract import TASKS,OPTIMIZER,seed,CHECKPOINT_SCHEMA,SELECTOR_SCHEMA
from .learning import (Baseline,EndpointTrainer,train_endpoint,train_local,on_policy_update,checkpoint,assigned_tasks,
                       resume_checkpoint,memory_peak,metrics,HistoryCache)
from .path import compile_condition,compile_conditions,evaluate_fixed,restore_compilation
from .supervision import (panels,bootstrap_source,query_once,query_conditions,LocalExamples,register_students,
                          readonly_E_query,student_data_complete,query_panel)


def load_version(runtime,directory,*,fixed_psi=True):
    state=torch.load(Path(directory)/'state.pt',map_location='cpu',weights_only=False)
    if state['schema_version']!=CHECKPOINT_SCHEMA:raise ValueError('retired shared model schema; fresh training required')
    runtime.generator.load_state_dict(state['psi']);runtime.actor.load_state_dict(state['theta'])
    runtime.version=state['psi_version']
    runtime.generator_optimizer_state=state['generator_optimizer'];runtime.G_sampler=state['G_sampler']
    runtime.supervision_version=state['supervision_version']
    runtime.fixed_kernel_identity=dict(checkpoint=str(Path(directory).resolve()),psi_version=runtime.version,
        checkpoint_stage=state['stage'],checkpoint_update=state['next_update'])
    if fixed_psi:runtime.freeze_psi()
    return state


def task_spec(contract,task):
    return next(row for row in contract['tasks'] if row['global_task_id']==task)


def fixed_G(args,runtime):
    state=load_version(runtime,args.checkpoint)
    run=read_json(args.root/'run_contract.json')
    if state['stage']!='G' or state['next_update']!=run['fixed_G_update']:
        raise ValueError('registered fixed complete G required; no best-node selection')
    return run


def refresh(args,runtime,runner,environment):
    state=load_version(runtime,args.checkpoint);run=read_json(args.root/'run_contract.json')
    version=run['student_refresh']['version']
    if state['stage']!='G' or state['next_update']!=run['student_refresh']['G_update']:
        raise ValueError('student data only at its registered complete G node')
    panel=panels(args.root)[args.task];task=task_spec(environment,args.task)
    destination=args.root/'student_data'/version/f'task_{args.task:04d}'
    readonly_E=run['student_refresh'].get('readonly_E_query',False)
    if student_data_complete(destination,panel,args.task,require_E=readonly_E):return
    if (destination/'path.json').exists():path=restore_compilation(destination,runtime)
    else:
        history,source=bootstrap_source(args.root,args.task,panel,0,HistoryCache())
        condition=run['student_refresh']['condition_mapping'][str(args.task)]
        path=compile_condition(runtime,runner,task,panel,condition['teacher_demo'],identity=destination.name,
            path_ordinal=condition['path_ordinal'],noise_root=condition['noise_root'],max_rounds=1,uniform=True,student_only=True,
            bootstrap_history=history,bootstrap_provenance=source)
        path.save(destination,retain_U=True)
    if path.completed_rounds!=1:raise ValueError('student refresh needs one real complete G-produced episode')
    actual=path.boundaries[-1]['actual_parameter_ref']
    weight_path=destination/f'{actual}.safetensors'
    history_descriptor=dict(path=str(destination/'history.pt.gz'),records=len(path.history.records),episodes=len(path.history.episodes))
    rewards=query_once(runtime,runner,path,task,panel,0,destination/'query',
        mt_reference_version=run.get('previous_student_version'),mt_reference_root=run.get('previous_root'))
    E=read_json(args.root/'teachers'/f'task_{args.task:04d}'/'E.json')
    reference=readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',E) if readonly_E else None
    write_json_atomic(destination/'parent.json',dict(complete=True,task=args.task,version=version,
        parent=dict(reference=f'{version}_task_{args.task:04d}_{actual}',weights=str(weight_path),histories=[history_descriptor]),
        actual_generated_parameter_ref=actual,common_E_reference=E['reference'],common_E_weights=E['weights'],
        actual_producer_refs=list(path.history.states),fixed_scale_reestimated=False,
        deterministic_endpoint_single_candidate=True,selection_did_not_use_reward_or_query=True,
        kernel_identity=path.kernel_identity,timings=path.timings,query_rewards=rewards,
        query_count=len(query_panel(panel,0)),readonly_E_complete=reference is not None,
        readonly_E_receipt=str(destination/'readonly_E/complete.json') if reference is not None else None,
        warmstart_source='same actual MT/G pair; no additional uniform R5 collection'))


def collect_batch(args,runtime,runner,environment,context,baseline,update,cache):
    """One fixed theta/psi batch shares the persistent official environment workers."""
    tick=time.monotonic();visit=update//9;panel_by_task=panels(args.root);entries=[];missing=[]
    rounds=read_json(args.root/'run_contract.json')['selector_training']['rounds']
    for task_id in assigned_tasks(context,update):
        task=task_spec(environment,task_id);panel=panel_by_task[task_id]
        destination=args.root/'RL_paths'/f'update_{update+1:06d}_task_{task_id:04d}'
        entry=dict(task=task,panel=panel,destination=destination,path=None);entries.append(entry)
        if (destination/'path.json').exists():
            if read_json(destination/'batch_identity.json')['kernel_identity']!=runtime.fixed_kernel_identity:
                raise ValueError('saved path has a different behavior theta or generation kernel')
            entry['path']=restore_compilation(destination,runtime)
        else:
            history,source=bootstrap_source(args.root,task_id,panel,visit,cache)
            missing.append((entry,dict(task=task,panel=panel,demo=panel['policy_videos'][visit%4],identity=destination.name,
                path_ordinal=visit,noise_root=seed(7,task_id,visit),max_rounds=rounds,conditional_continue=True,
                baseline=baseline,bootstrap_history=history,bootstrap_provenance=source)))
    generation_tick=time.monotonic()
    for (entry,_),path in zip(missing,compile_conditions(runtime,runner,[spec for _,spec in missing]) if missing else [],strict=True):
        entry['path']=path;path.save(entry['destination'],retain_U=True)
        write_json_atomic(entry['destination']/'batch_identity.json',dict(theta_next_update=update,
            checkpoint=str(args.checkpoint.resolve()),schema=SELECTOR_SCHEMA,current_fixed_theta=True,
            kernel_identity=runtime.fixed_kernel_identity))
    collection_seconds=time.monotonic()-generation_tick;query_tick=time.monotonic()
    rewards=query_conditions(runtime,runner,[(e['path'],e['task'],e['panel'],visit,e['destination']/'query') for e in entries],
        mt_reference_version=read_json(args.root/'run_contract.json')['student_refresh']['version'])
    return [e['path'] for e in entries],rewards,dict(collection_and_path_IO_seconds=collection_seconds,
        independent_query_and_row_IO_seconds=time.monotonic()-query_tick,total_collection_query_seconds=time.monotonic()-tick)


def rl(args,runtime,runner,environment,context):
    run=read_json(args.root/'run_contract.json');state=load_version(runtime,args.checkpoint)
    if state['stage']!='local' or state['next_update']!=run['fixed_local_update']:
        raise ValueError('RL begins from the registered final-use warmstart')
    if args.stop not in run['updates']['RL'] or args.stop%9:raise ValueError('RL updates must cover complete all36 rounds')
    baseline=Baseline().to(runtime.device);baseline.load_state_dict(state['baseline'])
    optimizer=torch.optim.AdamW(runtime.actor.parameters(),**OPTIMIZER);optimizer.load_state_dict(state['optimizer'])
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),**OPTIMIZER)
    generator_trainer=EndpointTrainer(runtime,context,args.root)
    generator_updates=run['joint_training']['G_updates_per_batch']
    panel_by_task=panels(args.root)
    query_sampling={str(task):dict(query_rotation=len(query_panel(panel,0)),query_pool_size=len(panel['query_states']))
        for task,panel in panel_by_task.items()}
    if not isinstance(generator_updates,int) or generator_updates<1:
        raise ValueError('joint supervised update block must be explicitly registered')
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    start=resume_checkpoint(args.resume,'RL',runtime,optimizer,context,baseline=baseline,
        baseline_optimizer=baseline_optimizer,generator_optimizer=generator_trainer.optimizer) if args.resume else 0
    generator_trainer.refresh_bank(runtime.supervision_version)
    cache=HistoryCache()
    for update in range(start,args.stop):
        tick=time.monotonic();runtime.freeze_psi();runtime.actor.eval();baseline.eval()
        # Cached learning allocations are not used by the next environment phase.
        # EGL cannot reclaim PyTorch's unused reserved blocks itself.
        torch.cuda.empty_cache()
        runtime.fixed_kernel_identity=dict(origin_checkpoint=str(args.checkpoint.resolve()),
            actor_next_update=update,psi_version=runtime.version,G_next_update=runtime.G_sampler['next_update'],
            supervision_version=runtime.supervision_version,schema=SELECTOR_SCHEMA)
        paths,rewards,timings=collect_batch(args,runtime,runner,environment,context,baseline,update,cache)
        runner.release_renderers()
        theta_tick=time.monotonic()
        on_policy_update(runtime,context,args.root,update+1,paths,rewards,optimizer,baseline,baseline_optimizer)
        optimizer.zero_grad(set_to_none=True);baseline_optimizer.zero_grad(set_to_none=True)
        timings['theta_and_baseline_update_seconds']=time.monotonic()-theta_tick;append_tick=time.monotonic()
        # Theta consumes its whole original kernel first; only then may all psi learn.
        supervision_version=register_students(args.root,paths,update+1,context)
        generator_trainer.refresh_bank(supervision_version)
        for path in paths:
            generator_trainer.cache.seed(args.root/'RL_paths'/path.identity/'history.pt.gz',path.history)
            # Prepared native KV belongs to the finished behavior/score consumer.
            # G uses retained true H/raw features and parameters, not these prefixes.
            path.native_response_cache.clear()
        timings['append_and_reload_supervision_seconds']=time.monotonic()-append_tick;psi_tick=time.monotonic()
        for _ in range(generator_updates):
            generator_trainer.step(runtime.G_sampler['next_update'],stage='joint_G')
        runtime.freeze_psi()
        timings['psi_update_block_seconds']=time.monotonic()-psi_tick;checkpoint_tick=time.monotonic()
        checkpoint(args.root,'RL',update+1,runtime,optimizer,context,
            sampler=dict(next_update=update+1,task_order=list(TASKS),logical_condition_batch=4,
                query_sampling=query_sampling,sampling_identity='conditional_continue',
                actual_depth=run['selector_training']['rounds'],supervision_version=supervision_version,
                checkpoint_role='registered_node' if update+1 in run['updates']['RL'] else 'recovery_only',
                alternating_order=['fixed psi/theta collection','theta','append true P/H','psi','resample']),
            baseline=baseline,baseline_optimizer=baseline_optimizer,generator_optimizer=generator_trainer.optimizer)
        timings['checkpoint_IO_seconds']=time.monotonic()-checkpoint_tick
        metrics(args.root,'RL_cycle',dict(update=update+1,full_update_seconds=time.monotonic()-tick,phases=timings,
            path_timings=[path.timings for path in paths],components=dict(runtime.components),
            G_updates_after_theta=generator_updates,psi_version=runtime.version,
            supervision_version=supervision_version),context)
        del paths


def report(args,runtime,runner,environment):
    state=load_version(runtime,args.checkpoint);run=read_json(args.root/'run_contract.json')
    specification=run['training_report']
    if [state['stage'],state['next_update']] not in specification['versions']:
        raise ValueError('report model was not pre-registered')
    version=f"{state['stage']}_{state['next_update']:08d}";task=task_spec(environment,args.task)
    panel=panels(args.root)[args.task];runtime.actor.eval()
    count=specification['conditions_per_task'];ordinals=range(count) if args.condition_ordinal is None else [args.condition_ordinal]
    for ordinal in ordinals:
        destination=args.root/'reports'/version/args.video_arm/f'task_{args.task:04d}_condition_{ordinal:03d}'
        if (destination/'complete.json').exists():continue
        tick=time.monotonic();demo=panel['policy_videos'][ordinal%4];video_task=None
        if args.video_arm=='other':demo=panel['policy_videos'][(ordinal+1)%4]
        if args.video_arm=='wrong':
            video_task=specification['wrong_task_mapping'][str(args.task)];demo=panels(args.root)[video_task]['policy_videos'][ordinal%4]
        if (destination/'path.json').exists():path=restore_compilation(destination,runtime)
        else:
            history,source=bootstrap_source(args.root,args.task,panel,ordinal,HistoryCache())
            path=compile_condition(runtime,runner,task,panel,demo,identity=destination.name,path_ordinal=ordinal,
                noise_root=seed(8,args.task,ordinal),max_rounds=specification['max_rounds'],video_task=video_task,
                bootstrap_history=history,bootstrap_provenance=source,
                external_max_environment_steps=specification['environment_multiplier']*(task['horizon']+10))
            path.save(destination,retain_U=True)
        rows=evaluate_fixed(runtime,runner,path,task,panel['report_states'],seed(80,args.task,ordinal))
        write_json_atomic(destination/'query.json',dict(rows=rows,finite_train_diagnostic=True,
            single_fixed_task_LoRA=True,feedback_did_not_change_selection=True))
        write_json_atomic(destination/'complete.json',dict(complete=True,version=version,condition=path.identity,
            input_arm=args.video_arm,rows=len(rows),environment_steps=path.environment_steps,
            selected=path.selected,end_to_end_seconds=time.monotonic()-tick,path_timings=path.timings))


def run(args,runtime,runner,environment,context):
    if args.command=='G':train_endpoint(runtime,context,args.root,stop=args.stop,resume=args.resume)
    elif args.command=='student-data':refresh(args,runtime,runner,environment)
    elif args.command=='local':
        fixed_G(args,runtime);baseline=Baseline().to(runtime.device)
        train_local(runtime,context,args.root,LocalExamples(args.root,runtime),stop=args.stop,resume=args.resume,baseline=baseline)
    elif args.command=='RL':rl(args,runtime,runner,environment,context)
    elif args.command=='report':report(args,runtime,runner,environment)
