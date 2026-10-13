"""Fixed-G pair collection, final warm and current-kernel selector learning."""
from pathlib import Path
from datetime import timedelta
import os,re,time,copy
import torch
from ember.pi05_source_checkpoint import read_json,write_json_atomic,restore_rng
from ember.pi05_source_setup import initialize_deferred_process_group
from .contract import OPTIMIZER,SELECTOR_SCHEMA
from .learning import (Baseline,train_local,on_policy_update,checkpoint,assigned_tasks,
                       resume_checkpoint,memory_peak,metrics,HistoryCache)
from .path import compile_conditions,restore_compilation
from .supervision import panels,bootstrap_source,query_conditions,LocalExamples,query_panel


def load_version(runtime,directory,*,fixed_psi=True):
    from .shared_checkpoint import load_state
    state=load_state(directory,runtime)
    runtime.fixed_kernel_identity=dict(checkpoint=str(Path(directory).resolve()),psi_version=runtime.version,
        checkpoint_stage=state['stage'],checkpoint_update=state['next_update'],
        frozen_frontend_checkpoint=runtime.frozen_frontend_checkpoint)
    if fixed_psi:runtime.freeze_psi()
    return state


def task_spec(contract,task):
    return next(row for row in contract['tasks'] if row['global_task_id']==task)


def collect_batch(args,runtime,runner,environment,context,baseline,update,cache):
    """One fixed theta/psi batch shares the persistent official environment workers."""
    tick=time.monotonic();runner_before=copy.deepcopy(runner.components)
    visit=0;panel_by_task=panels(args.root);entries=[];missing=[]
    rounds=read_json(args.root/'run_contract.json')['selector_training']['rounds']
    for task_id in assigned_tasks(context,update,args.root):
        task=task_spec(environment,task_id);panel=panel_by_task[task_id]
        destination=args.root/'RL_paths'/f'update_{update+1:06d}_task_{task_id:04d}'
        entry=dict(task=task,panel=panel,destination=destination,path=None);entries.append(entry)
        if (destination/'path.json').exists():
            if read_json(destination/'batch_identity.json')['kernel_identity']!=runtime.fixed_kernel_identity:
                raise ValueError('saved path has a different behavior theta or generation kernel')
            entry['path']=restore_compilation(destination,runtime)
        else:
            history,source=bootstrap_source(args.root,task_id,panel,visit,cache)
            missing.append((entry,dict(task=task,panel=panel,demo=panel['policy_videos'][1],identity=destination.name,
                path_ordinal=1,noise_root=seed(7,task_id,1),max_rounds=rounds,conditional_continue=True,
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
        independent_query_and_row_IO_seconds=time.monotonic()-query_tick,total_collection_query_seconds=time.monotonic()-tick,
        runner={key:value-runner_before[key] for key,value in runner.components.items() if key!='physical_batch_histogram'},
        physical_batch_histogram={key:value-runner_before['physical_batch_histogram'].get(key,0)
            for key,value in runner.components['physical_batch_histogram'].items()})


def wait_for_collectors(root,context,update):
    """Keep uneven environment/query residence outside the NCCL gradient timeout."""
    if context.world_size==1:return 0.
    token=re.sub(r'[^A-Za-z0-9_.-]','_',
        f"{os.environ['TORCHELASTIC_RUN_ID']}-{os.environ['MASTER_PORT']}-RL-{update}")
    destination=Path(root)/'rendezvous'/f'.collectors-ready-{token}'
    destination.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    store=torch.distributed.FileStore(str(destination),context.world_size)
    store.set(f'rank-{context.rank}',b'ready')
    store.wait([f'rank-{rank}' for rank in range(context.world_size)],timedelta(minutes=30))
    return time.monotonic()-started


def rl(args,runtime,runner,environment,context):
    run=read_json(args.root/'run_contract.json');state=load_version(runtime,args.checkpoint)
    if state['stage']!='local' or state['next_update']!=180 or args.stop!=18:
        raise ValueError('selector RL starts only from registered A180 and ends at B18')
    baseline=Baseline().to(runtime.device);baseline.load_state_dict(state['baseline']);baseline.requires_grad_(True)
    optimizer=torch.optim.AdamW(runtime.actor.parameters(),**OPTIMIZER);optimizer.load_state_dict(state['actor_optimizer'])
    baseline_optimizer=torch.optim.AdamW(baseline.parameters(),**OPTIMIZER)
    baseline_optimizer.load_state_dict(state['baseline_optimizer'])
    if state['topology']['world_size']!=context.world_size:
        raise ValueError('RL topology must match complete A180; explicitly migrate at a full boundary')
    runtime.training_baseline=baseline;runtime.training_baseline_optimizer=baseline_optimizer
    restore_rng(state['rank_rng'][context.rank],context)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    start=resume_checkpoint(args.resume,'RL',runtime,optimizer,context,baseline=baseline,
        baseline_optimizer=baseline_optimizer) if args.resume else 0
    cache=HistoryCache()
    for update in range(start,args.stop):
        tick=time.monotonic();runtime.freeze_psi();runtime.actor.eval();baseline.eval();torch.cuda.empty_cache()
        runtime.fixed_kernel_identity=dict(origin_checkpoint=str(args.checkpoint.resolve()),
            actor_next_update=update,psi_version=38349,G_next_update=38349,supervision_version=9,schema=SELECTOR_SCHEMA)
        paths,rewards,timings=collect_batch(args,runtime,runner,environment,context,baseline,update,cache)
        runner.release_renderers();timings['collector_ready_wait_seconds']=wait_for_collectors(args.root,context,update+1)
        theta_tick=time.monotonic()
        on_policy_update(runtime,context,args.root,update+1,paths,rewards,optimizer,baseline,baseline_optimizer)
        optimizer.zero_grad(set_to_none=True);baseline_optimizer.zero_grad(set_to_none=True)
        timings['theta_and_baseline_update_seconds']=time.monotonic()-theta_tick
        for path in paths:path.native_response_cache.clear()
        assert runtime.G_sampler['next_update']==38349 and runtime.supervision_version==9
        checkpoint(args.root,'RL',update+1,runtime,optimizer,context,
            sampler=dict(next_update=update+1,task_order=run['training_tasks'],logical_condition_batch=4,
                G_seen_weight=.5,G_novel_weight=.5,sampling_identity='conditional_continue',
                actual_depth=5,query_visit=0,teaching_pool_ordinal=1,bank_version=9,
                checkpoint_role='registered_node' if update+1==18 else 'recovery_only',
                alternating_order=['fixed G/current theta collection','theta and baseline','resample']),
            baseline=baseline,baseline_optimizer=baseline_optimizer)
        metrics(args.root,'joint_cycle',dict(theta_update=update+1,G_next_update=38349,
            supervision_version=9,G_updated=False,G_bank_appended=False,
            total_cycle_seconds=time.monotonic()-tick,timings=timings,peak=memory_peak(runtime.device)),context)
    write_json_atomic(args.root/'complete'/f'RL_rank_{context.rank:02d}.json',dict(complete=True,next_update=args.stop,
        full72_rounds=1,world_size=context.world_size,rank=context.rank,G_cursor=38349,bank_version=9))


def run(args,runtime,runner,environment,context):
    if args.command=='local':
        from .shared_checkpoint import initialize_selector
        optimizer,baseline=initialize_selector(runtime,context,args.root)
        runtime.actor_optimizer_state=optimizer.state_dict()
        train_local(runtime,context,args.root,LocalExamples(args.root,runtime),stop=args.stop,resume=args.resume,baseline=baseline)
    elif args.command=='RL':rl(args,runtime,runner,environment,context)
    elif args.command=='student-data':
        load_version(runtime,args.checkpoint)
        from .student_practice import pair_group
        pair_group(args,runtime,runner,environment,[args.task])
    else:raise ValueError('unregistered fixed-G selector consumer')
