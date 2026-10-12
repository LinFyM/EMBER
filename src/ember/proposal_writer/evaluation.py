"""Fixed-reference provenance and action-hidden strict validation400 consumption."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import time

import numpy as np

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.expert_manifold.video_schedule import condition_demo_index,task_video_mapping
from ember.pi05_eval_queue import (EvaluationTask,EvaluationShard,build_cost_balanced_shards,
    initialize_queue,claim_next,complete_job,fail_job,publish_json_exclusive,queue_summary)

from .contract import SOURCE,MT_PATH,seed,environment_contract
from .path import compile_conditions,restore_compilation
from .analysis import paired,billing

MT400=Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1/MT/evaluation/correct400/results.json')


def reference_compatibility(previous,asset_root):
    """Executable model/preprocessing provenance, without tensor hashes or score selection."""
    reasons=[];root=Path(asset_root)
    if Path(previous['model']['checkpoint']).resolve()!=(root/SOURCE['checkpoint']).resolve():reasons.append('source checkpoint')
    if Path(previous['adapter']['checkpoint']).resolve()!=MT_PATH.resolve():reasons.append('MT300 checkpoint')
    policy=dict(num_inference_steps=10,replan_steps=5,camera_rotation_degrees=180,model_resolution=224,
                state_dim=8,action_dim=7,precision='bfloat16')
    if any(previous['policy'].get(k)!=v for k,v in policy.items()):reasons.append('official native policy')
    if previous['environment']['dummy_settling_steps']!=10:reasons.append('dummy settling')
    path=Path(previous['normalization']['path'])
    old=read_json(path) if path.exists() else json.loads(subprocess.check_output(
        ['git','show',f"{previous['git']['commit']}:{SOURCE['normalization']}"],cwd=root,text=True))
    if old['stats']!=read_json(root/SOURCE['normalization'])['stats']:reasons.append('frozen normalization')
    return reasons


def prepare_formal(args):
    run=read_json(args.root/'run_contract.json');spec=run['formal_report']
    if str(args.checkpoint.resolve())!=str(Path(spec['checkpoint']).resolve()):raise ValueError('formal checkpoint not preregistered')
    source=read_json(MT400);previous=read_json(MT400.parent/'run_contract.json')
    reasons=reference_compatibility(previous,args.asset_root)
    if reasons:raise ValueError('MT400 reference incompatible: '+', '.join(reasons))
    environment=environment_contract(args.asset_root,root=args.root,role='validation')
    suite_order=('libero_spatial','libero_object','libero_goal','libero_10')
    tasks={t['global_task_id']:t for t in environment['tasks']}
    rows=[]
    for row in source['rows']:
        task=10*suite_order.index(row['suite'])+int(row['task_id']);state=int(row['init_state_id'])
        if task not in tasks:raise ValueError('MT reference crosses validation split')
        reference=condition_demo_index(spec['video_seed'],row['suite'],row['task_id'],state,
            condition='correct',demo_count=50,sampling_mode='without_replacement')
        scene=row['scene_reference'];path=Path(scene['path'])
        if not path.exists() or path.stat().st_size!=scene['bytes']:raise ValueError('registered formal scene unavailable')
        practice=np.random.default_rng(seed(76,task,state)).permutation([i for i in range(50) if i!=state]).tolist()
        rows.append(dict(task=task,init_state_id=state,teacher_demo=reference,video_ordinal=state,
            practice_states=practice,MT_source=str(MT400),MT_row={**row,'global_task_id':task},
            registered_scene=dict(root=str(path.parent)),noise_root=row['policy_seed_root'],
            MT_reference_reused=True,final_state_excluded_from_adaptation=True))
    if len(rows)!=400 or len({(r['task'],r['init_state_id']) for r in rows})!=400:
        raise ValueError('strict validation scope must contain400 unique rows')
    for task in tasks:
        if {r['teacher_demo'] for r in rows if r['task']==task}!=set(range(50)):
            raise ValueError('formal task teaching schedule is not50 without replacement')
    panel=dict(complete=True,checkpoint=str(args.checkpoint.resolve()),specification=spec,rows=rows,
        environment=environment,held_offline_labels_read=False,no_shared_or_task_optimizer=True,
        no_query_feedback_in_H=True,MT400_source=str(MT400),historical_MT_cost_recounted=False)
    file=args.root/'formal_panel.json'
    if file.exists() and read_json(file)!=panel:raise ValueError('formal panel cannot change after registration')
    write_json_atomic(file,panel)
    return panel


def formal_queue(root,*,workers,condition_batch,arm='correct'):
    """Reuse the official dynamic queue with immutable condition shards."""
    root=Path(root);panel=read_json(root/'formal_panel.json');stage,update=panel['specification']['version']
    base=root/'validation'/f'{stage}_{update:08d}'/arm;base.mkdir(parents=True,exist_ok=True)
    plan_path=base/'queue_plan.json'
    if plan_path.exists():
        plan=read_json(plan_path)
        if plan['checkpoint']!=panel['checkpoint']:
            raise ValueError('formal queue checkpoint changed')
        shards=[EvaluationShard(**{**s,'init_state_ids':tuple(s['init_state_ids'])}) for s in plan['shards']]
    else:
        from dataclasses import asdict
        tasks=[EvaluationTask(t['suite'],t['task_id'],t['horizon'],tuple(r['init_state_id'] for r in panel['rows']
            if r['task']==t['global_task_id'])) for t in panel['environment']['tasks']]
        balanced=build_cost_balanced_shards(tasks,env_batch_size=condition_batch,physical_gpu_count=workers)
        parts=[(s,s.init_state_ids[first:first+condition_batch]) for s in balanced
            for first in range(0,len(s.init_state_ids),condition_batch)]
        shards=[EvaluationShard(f'{s.job_id}_{ordinal:03d}',ordinal,s.suite,s.task_id,s.horizon,ids,
            s.horizon*len(ids),s.preferred_gpu) for ordinal,(s,ids) in enumerate(parts)]
        plan=dict(checkpoint=panel['checkpoint'],arm=arm,condition_batch=condition_batch,
            initial_physical_workers=workers,shards=[asdict(s) for s in shards])
        write_json_atomic(plan_path,plan)
    initialize_queue(base/'jobs.sqlite',shards,contract_reference=str(root/'formal_panel.json'))
    return base


def formal_worker(args,runtime,runner):
    """One policy/kernel load serves successive dynamically claimed groups."""
    from .pipeline import load_version
    state=load_version(runtime,args.checkpoint);runtime.actor.eval()
    panel=read_json(args.root/'formal_panel.json');spec=panel['specification']
    if [state['stage'],state['next_update']]!=spec['version'] or str(args.checkpoint.resolve())!=panel['checkpoint']:
        raise ValueError('strict formal version changed')
    stage,update=spec['version'];base=args.root/'validation'/f'{stage}_{update:08d}'/args.video_arm
    tasks={(t['suite'],t['task_id']):t for t in panel['environment']['tasks']}
    queue=base/'jobs.sqlite'
    while (claim:=claim_next(queue,worker_id=args.worker_id)) is not None:
        try:
            task=tasks[claim.task_key]
            entries=[r for r in panel['rows'] if r['task']==task['global_task_id'] and r['init_state_id'] in claim.shard.init_state_ids]
            results=[]
            for first in range(0,len(entries),args.condition_batch):
                results.extend(formal_conditions(args,runtime,runner,panel,entries[first:first+args.condition_batch]))
            relative=Path('shards')/claim.shard.job_id/f'attempt_{claim.attempt:03d}.json'
            size=publish_json_exclusive(base/relative,dict(rows=results,checkpoint=panel['checkpoint'],
                worker_id=args.worker_id,code_reading_conditions_in_one_persistent_runtime=True))
            complete_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,
                rows_path=str(relative),rows_bytes=size,row_count=len(results),successes=sum(bool(r['success']) for r in results))
        except BaseException as error:
            fail_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,error=repr(error))
            raise
    return queue_summary(queue)


def formal_entry(args,panel,row):
    task=next(t for t in panel['environment']['tasks'] if t['global_task_id']==row['task'])
    demo=row['teacher_demo'];video_task=None
    if args.video_arm=='other':demo=(demo+17)%50
    elif args.video_arm=='wrong':
        keys=[(t['suite'],t['task_id']) for t in panel['environment']['tasks']]
        mapping=task_video_mapping(keys,{key:'validation' for key in keys},'cross_suite_wrong')
        video_task=next(r['video_global_task_id'] for r in mapping if r['language_global_task_id']==row['task'])
    stage,update=panel['specification']['version']
    destination=args.root/'validation'/f'{stage}_{update:08d}'/args.video_arm/f"task_{row['task']:04d}_state_{row['init_state_id']:03d}"
    return dict(row=row,task=task,demo=demo,video_task=video_task,destination=destination,path=None,new_steps=0)


def formal_conditions(args,runtime,runner,panel,rows):
    """Independent paths share slots; their final query never contributes to H."""
    tick=time.monotonic();spec=panel['specification'];entries=[formal_entry(args,panel,r) for r in rows]
    missing=[]
    for entry in entries:
        destination=entry['destination']
        if (destination/'complete.json').exists():continue
        if (destination/'path.json').exists():entry['path']=restore_compilation(destination,runtime)
        else:missing.append(entry)
    specifications=[dict(task=e['task'],panel=dict(practice_states=e['row']['practice_states']),demo=e['demo'],
        identity=e['destination'].name,path_ordinal=e['row']['video_ordinal'],noise_root=seed(77,e['row']['task'],e['row']['video_ordinal']),
        max_rounds=spec['max_rounds'],external_max_environment_steps=spec['environment_multiplier']*(e['task']['horizon']+10),
        video_task=e['video_task']) for e in missing]
    for entry,path in zip(missing,compile_conditions(runtime,runner,specifications) if missing else [],strict=True):
        entry['path']=path;entry['new_steps']=path.timings['new_environment_steps']
        path.save(entry['destination'],retain_U=True)
    requests=[];by_identity={}
    for entry in entries:
        path=entry['path']
        if path is None:continue
        row=entry['row']
        if row['init_state_id'] in {r['init_state_id'] for r in path.history.episodes}:
            raise ValueError('condition final initial state participated in adaptation')
        if path.selected=='MT300':
            entry['result']={**row['MT_row'],'fixed_parameter_result_reused_from':row['MT_source']}
        else:
            identity=f'{path.identity}_fixed_final';by_identity[identity]=entry
            requests.append(dict(task=entry['task'],state=path.states[path.selected],parameter_ref=path.selected,
                state_id=row['init_state_id'],noise_root=row['noise_root'],episode_id=identity,remaining=entry['task']['horizon']+10,
                snapshots=False,uses_teaching=False,capture_history=False,capture_hidden=False,registered_scene=row['registered_scene']))
    for output in runner.run(requests):
        entry=by_identity[output['request']['episode_id']];entry['result']=dict(output['row'])
        entry['new_steps']+=output['row']['environment_steps']
        if output['history'].records:
            entry['result']['first_native_normalized_actions']=output['history'].records[0]['normalized_actions'].tolist()
    for entry in entries:
        path=entry['path']
        if path is None:continue
        row=entry['row'];result={**entry['result'],'global_task_id':row['task'],'teacher_demo':entry['demo'],
            'reference_teacher_demo':row['teacher_demo'],'video_ordinal':row['video_ordinal'],'condition':path.identity,
            'reference':path.selected,'arm':'selected','query_excluded_from_H':True,'fixed_unique_LoRA':True,
            'checkpoint':str(args.checkpoint.resolve())}
        write_json_atomic(entry['destination']/'query.json',dict(selected=result,MT=row['MT_row'],MT_reused_from=row['MT_source']))
        write_json_atomic(entry['destination']/'complete.json',dict(complete=True,condition=path.identity,
            environment_steps=path.environment_steps,actual_new_environment_steps=entry['new_steps'],selected=path.selected,
            shared_group_residence_seconds=time.monotonic()-tick,shared_group_conditions=len(rows),
            compilation_timings=path.timings,no_evaluation_gradients=True))
    return [read_json(e['destination']/'query.json')['selected'] for e in entries]


def formal_summary(root,arm='correct'):
    root=Path(root);panel=read_json(root/'formal_panel.json');stage,update=panel['specification']['version']
    base=root/'validation'/f'{stage}_{update:08d}'/arm;selected=[];mt=[]
    for entry in panel['rows']:
        file=base/f"task_{entry['task']:04d}_state_{entry['init_state_id']:03d}"/'query.json'
        data=read_json(file);selected.append(data['selected']);mt.append(data['MT'])
    result=dict(complete=True,checkpoint=panel['checkpoint'],input_arm=arm,raw_rows=selected,MT_raw_rows=mt,
        paired=paired(mt,selected),per_task={},per_suite={},billing=billing(root),strict_single_checkpoint400=True)
    for key in ('global_task_id','suite'):
        for group in sorted({r[key] for r in selected}):
            result['per_task' if key=='global_task_id' else 'per_suite'][str(group)]=paired(
                [r for r in mt if r[key]==group],[r for r in selected if r[key]==group])
    result['breadth']=sum(r['candidate_success']>0 for r in result['per_task'].values())
    write_json_atomic(base/'results.json',result)
    return result


def student_queue(root):
    """One immutable condition per task, using the existing dynamic row queue."""
    from dataclasses import asdict
    from .contract import TASKS
    from .supervision import panels,query_panel
    root=Path(root);run=read_json(root/'run_contract.json');panel=panels(root)
    version=run['student_refresh']['version'];base=root/'student_data'/version/'queue'
    tasks={task['global_task_id']:task for task in run['environment']['tasks']}
    if set(tasks)!=set(TASKS):raise ValueError('student queue requires the registered full36 tasks')
    endpoint_only=run['student_refresh'].get('readout_mode')=='endpoint_query_only'
    if endpoint_only and run['student_refresh'].get('post_generation_practice') is not False:
        raise ValueError('direct endpoint queue must omit new G practice explicitly')
    practice_count=0 if endpoint_only else 1
    shards=[]
    for ordinal,task_id in enumerate(TASKS):
        task=tasks[task_id];states=tuple(query_panel(panel[task_id],0))
        if len(states)!=run['student_refresh']['query_states_per_parameter']:
            raise ValueError('student queue cannot shrink the actual query panel')
        shards.append(EvaluationShard(f'student_{version}_{task_id:04d}',ordinal,task['suite'],task['task_id'],
            task['horizon'],states,task['horizon']*(len(states)+practice_count),0 if task['horizon']==520 else None))
    plan=dict(checkpoint=str(root/'checkpoints/G'/f"update_{run['student_refresh']['G_update']:08d}"),
        logical_tasks=list(TASKS),new_episodes=36*(practice_count+run['student_refresh']['query_states_per_parameter']),
        new_G_practice_episodes=36*practice_count,readout_mode='endpoint_query_only' if endpoint_only else 'practice_then_query',
        shards=[asdict(shard) for shard in shards],condition_mapping=run['student_refresh']['condition_mapping'],
        G_training_code=run['git'],one_complete_LoRA_per_condition=True)
    file=base/'queue_plan.json'
    if file.exists() and read_json(file)!=json.loads(json.dumps(plan)):
        raise ValueError('student condition queue changed after registration')
    write_json_atomic(file,plan)
    initialize_queue(base/'jobs.sqlite',shards,contract_reference=str(root/'run_contract.json'))
    return base


def student_worker(args,runtime,runner,environment):
    """One frozen resident consumes distinct groups; all policy RNG stays per row."""
    from .pipeline import load_version,refresh_group,endpoint_query_group
    from .supervision import query_row
    from .learning import memory_peak
    import torch
    state=load_version(runtime,args.checkpoint);run=read_json(args.root/'run_contract.json')
    endpoint_only=args.command=='endpoint-query'
    if endpoint_only!=(run['student_refresh'].get('readout_mode')=='endpoint_query_only'):
        raise ValueError('resident command must match its registered practice/query consumer')
    consume=endpoint_query_group if endpoint_only else refresh_group
    if state['stage']!='G' or state['next_update']!=run['student_refresh']['G_update']:
        raise ValueError('resident student worker requires the registered final complete G')
    version=run['student_refresh']['version'];base=args.root/'student_data'/version/'queue'
    queue=base/'jobs.sqlite';tasks={(t['suite'],t['task_id']):t for t in environment['tasks']}
    stop=run.get('absolute_deadline_revision',{}).get('target_complete_and_consumed_UTC')
    groups=0;start=time.monotonic()
    while stop is None or datetime.now(timezone.utc)<datetime.fromisoformat(stop):
        claims=[]
        for _ in range(args.condition_batch):
            claim=claim_next(queue,worker_id=args.worker_id)
            if claim is None:break
            claims.append(claim)
        if not claims:break
        task_ids=[tasks[claim.task_key]['global_task_id'] for claim in claims]
        committed=set()
        try:
            tick=time.monotonic();consume(args,runtime,runner,environment,run,task_ids)
            for claim,task_id in zip(claims,task_ids,strict=True):
                destination=args.root/'student_data'/version/f'task_{task_id:04d}'
                parent=read_json(destination/'parent.json');ref=parent['actual_generated_parameter_ref']
                rows=[query_row(destination/'query'/f'{ref}_{state_id:03d}.json') for state_id in claim.shard.init_state_ids]
                relative=Path('shards')/claim.shard.job_id/f'attempt_{claim.attempt:03d}.json'
                size=publish_json_exclusive(base/relative,dict(rows=rows,task=task_id,condition=str(destination),
                    checkpoint=str(args.checkpoint),worker_id=args.worker_id,shared_group_seconds=time.monotonic()-tick))
                complete_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,
                    rows_path=str(relative),rows_bytes=size,row_count=len(rows),successes=sum(bool(r['success']) for r in rows))
                committed.add(claim.shard.job_id)
            groups+=1
            free,total=torch.cuda.mem_get_info(runtime.device)
            write_json_atomic(base/f'{args.worker_id}_physical.json',dict(groups=groups,
                elapsed_seconds=time.monotonic()-start,condition_batch=args.condition_batch,slots=runner.slots,
                completed_task_ids=task_ids,queue=queue_summary(queue),process_GPU_peak=memory_peak(runtime.device),
                device_free_GiB=free/2**30,device_total_GiB=total/2**30,runner_components=runner.components,
                actual_new_environment_steps=runner.environment_steps))
        except BaseException as error:
            for claim in claims:
                if claim.shard.job_id not in committed:
                    fail_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,error=repr(error))
            raise
    return queue_summary(queue)
