"""Fixed-reference provenance and action-hidden strict validation400 consumption."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import time

import numpy as np

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.expert_manifold.video_schedule import condition_demo_index,task_video_mapping

from .contract import SOURCE,MT_PATH,seed,environment_contract
from .path import compile_condition,restore_compilation
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


def formal_condition(args,runtime,runner):
    from .pipeline import load_version
    state=load_version(runtime,args.checkpoint);runtime.actor.eval()
    panel=read_json(args.root/'formal_panel.json');spec=panel['specification']
    if args.condition_ordinal not in range(50):raise ValueError('formal initial state outside registered50')
    if [state['stage'],state['next_update']]!=spec['version'] or str(args.checkpoint.resolve())!=panel['checkpoint']:
        raise ValueError('strict formal version changed')
    row=next(r for r in panel['rows'] if r['task']==args.task and r['init_state_id']==args.condition_ordinal)
    task=next(t for t in panel['environment']['tasks'] if t['global_task_id']==args.task)
    demo=row['teacher_demo'];video_task=None
    if args.video_arm=='other':demo=(demo+17)%50
    elif args.video_arm=='wrong':
        keys=[(t['suite'],t['task_id']) for t in panel['environment']['tasks']]
        mapping=task_video_mapping(keys,{key:'validation' for key in keys},'cross_suite_wrong')
        video_task=next(r['video_global_task_id'] for r in mapping if r['language_global_task_id']==args.task)
    destination=args.root/'validation'/f"{state['stage']}_{state['next_update']:08d}"/args.video_arm/f"task_{args.task:04d}_state_{row['init_state_id']:03d}"
    if (destination/'complete.json').exists():return
    tick=time.monotonic();before_steps=runner.environment_steps
    if (destination/'path.json').exists():path=restore_compilation(destination,runtime)
    else:
        path=compile_condition(runtime,runner,task,dict(practice_states=row['practice_states']),demo,
            identity=destination.name,path_ordinal=row['video_ordinal'],noise_root=seed(77,args.task,row['video_ordinal']),
            max_rounds=spec['max_rounds'],external_max_environment_steps=spec['environment_multiplier']*(task['horizon']+10),
            video_task=video_task)
        path.save(destination,retain_U=True)
    if row['init_state_id'] in {r['init_state_id'] for r in path.history.episodes}:
        raise ValueError('condition final initial state participated in adaptation')
    request=dict(task=task,state=path.states[path.selected],parameter_ref=path.selected,state_id=row['init_state_id'],
        noise_root=row['noise_root'],episode_id=f'{path.identity}_fixed_final',remaining=task['horizon']+10,
        snapshots=False,uses_teaching=False,registered_scene=row['registered_scene'])
    if path.selected=='MT300':
        result=dict(row['MT_row']);result['fixed_parameter_result_reused_from']=row['MT_source']
        first_actions=result.get('first_native_normalized_actions')
    else:
        output=next(runner.run([request]));result=dict(output['row'])
        first_actions=output['history'].records[0]['normalized_actions'].tolist() if output['history'].records else None
    result={**result,'global_task_id':args.task,'teacher_demo':demo,'reference_teacher_demo':row['teacher_demo'],
        'video_ordinal':row['video_ordinal'],'condition':path.identity,'reference':path.selected,'arm':'selected',
        'query_excluded_from_H':True,'fixed_unique_LoRA':True,'checkpoint':str(args.checkpoint.resolve())}
    if first_actions is not None:result['first_native_normalized_actions']=first_actions
    write_json_atomic(destination/'query.json',dict(selected=result,MT=row['MT_row'],MT_reused_from=row['MT_source']))
    write_json_atomic(destination/'complete.json',dict(complete=True,condition=path.identity,environment_steps=path.environment_steps,
        actual_new_environment_steps=runner.environment_steps-before_steps,selected=path.selected,
        end_to_end_seconds=time.monotonic()-tick,compilation_timings=path.timings,no_evaluation_gradients=True))


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
