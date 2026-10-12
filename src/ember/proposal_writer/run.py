"""Canonical full36 teacher preparation and shared Writer consumers."""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.writer.practice import EpisodeRunner

from .contract import (RUN_ROOT, ASSET_ROOT, TASKS, SOURCE, MT_PATH, OPTIMIZER, SCHEMA,
                       SELECTOR_SCHEMA, CHECKPOINT_SCHEMA, G_OBJECTIVE, materialize_panel, seed, parameter_ref)
from .runtime import Runtime
from .teacher import fit_teacher
from .path import save_history


def task_panel(root, task):
    return next(row for row in materialize_panel(root)['tasks'] if row['task_id'] == task)


def frame_chunk_for_rank(default, context):
    """Consume the launcher's physical chunks in CUDA-visible rank order."""
    chunks=os.environ.get('EMBER_META_FRAME_CHUNKS')
    values=[int(value) for value in chunks.split(',')] if chunks is not None else [int(default)]*context.world_size
    if len(values)!=context.world_size or any(value<1 for value in values):
        raise ValueError('physical meta frame chunks must be positive and match the actual world size')
    return values[context.local_rank]


def create_runtime(args, context, environment):
    return Runtime(context.device,asset_root=args.asset_root,root=args.root,
        frame_chunk=frame_chunk_for_rank(args.frame_chunk,context),
        task_ids=tuple(t['global_task_id'] for t in environment['tasks']),task_role=environment['role'])


def save_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))


def endpoint_learning_registration(args, previous, old):
    """Import v7 G864 once, then use v8 operator-objective recovery checkpoints."""
    resume=getattr(args,'resume',None)
    if resume is None or getattr(args,'stop',None)!=1296 or old['fixed_G_update']!=864:
        raise ValueError('operator learning requires complete original v7 G864 and registered exit1296')
    resume=Path(resume).resolve();expected=(previous/'checkpoints/G/update_00000864').resolve()
    manifest=read_json(resume/'manifest.json')
    source_schema='ember_full36_deterministic_endpoint_writer_v7'
    if (resume!=expected or manifest.get('schema_version')!=source_schema or manifest.get('stage')!='G'
            or manifest.get('next_update')!=864 or not manifest.get('complete')
            or not manifest.get('full_optimizer_rng_sampler') or not (resume/'state.pt').is_file()):
        raise ValueError('operator learning requires complete original v7 G864 and registered exit1296')
    migration=dict(checkpoint=str(resume),source_schema=source_schema,destination_schema=CHECKPOINT_SCHEMA,
        source_objective='masked_endpoint_MSE',destination_objective=G_OBJECTIVE,
        update=864,all_shared_parameters_and_Adam_inherited=True,sampler_and_rank_RNG_inherited=True,
        source_root=str(previous),source_code=old['git'],exact_resume=False)
    return dict(updates=dict(G=[1296]),fixed_G_update=1296,G_checkpoint_updates=[1080,1296],
        G_bank_mode='MT_only',G_bank_version=0,checkpoint_schema=CHECKPOINT_SCHEMA,objective_migration=migration,
        shared_initialization='continue complete v7 G864 psi/theta/Adam/sampler/rank RNG; explicit new operator objective; bank0',
        G_initial_resume=dict(checkpoint=str(resume),update=864,source_root=str(previous),source_code=old['git'],
            topology=manifest['topology'],supervision_version=0,recovery_only_updates=[1080],scientific_exit=1296),
        stage_estimate_seconds={'G_1296':11971.063878*2/3},
        architecture_change='none; full deployed s*B@A objective replaces factor MSE',
        bounded_stop='G1296 then all36 deterministic first students and paired MT/G864/newG/E query16; no local/RL/held400')


def freeze_operator_scale(root, previous):
    """Reuse saved exact teacher energies; no G/query data enters the denominator."""
    evidence=previous/'analysis/main_scientific_readout/endpoint864_factor_effects.json'
    analysis=read_json(evidence);lora=read_json(root/'task_lora_contract.json')
    targets=[item['name'] for item in lora['targets']]
    scaling=lora['adapter']['alpha']/lora['adapter']['rank']
    values={name:{} for name in targets}
    for row in analysis['layers']:
        if row['task'] not in TASKS or row['layer'] not in values or row['task'] in values[row['layer']]:
            raise ValueError('operator normalization needs unique exact full36/38 teacher energies')
        values[row['layer']][row['task']]=row['EE']*scaling**2
    if len(targets)!=38 or any(set(v)!=set(TASKS) for v in values.values()):
        raise ValueError('operator normalization is missing a teacher or target')
    D={name:math.fsum(values[name].values())/len(TASKS) for name in targets}
    if any(not math.isfinite(v) or v<=0 for v in D.values()):
        raise ValueError('all38 task-equal operator energies must be positive and finite')
    sources=[dict(task=t,weights=read_json(root/'teachers'/f'task_{t:04d}'/'E.json')['weights'],
        evidence=str(evidence)) for t in TASKS]
    record=dict(objective=G_OBJECTIVE,D_by_target=D,scaling={name:scaling for name in targets},sources=sources,
        energy='mean36 ||s*B_E@A_E||F^2',task_weight=1/36,layer_weight=1/38,
        teacher_energy_source=str(evidence),frozen=True,sample_normalization=False,
        zero_teachers_included=True,labels_never_conditions=True)
    path=root/'operator_scale.json'
    if path.exists() and read_json(path)!=record:raise ValueError('frozen operator denominators changed')
    if not path.exists():write_json_atomic(path,record)
    return path


def prepare(args):
    """Register one bounded Writer exit against the immutable existing bank."""
    if args.reuse_root is None:raise ValueError('endpoint requires the registered previous E/MT/H and query16 root')
    root=args.root;previous=args.reuse_root.resolve();old=read_json(previous/'run_contract.json')
    learning=endpoint_learning_registration(args,previous,old);stop=learning['fixed_G_update']
    if root.resolve()==previous or tuple(old['training_tasks'])!=TASKS:
        raise ValueError('new output root and exact audited all36 scope required')
    if (old['teacher']['selection_per_node'],old['teacher']['report_per_arm'])!=(16,16):
        raise ValueError('previous teacher selection and independent report must be16/16')
    root.mkdir(parents=True,exist_ok=True)
    for cache in ('cuda','triton','huggingface'):(root/'cache'/cache).mkdir(parents=True,exist_ok=True)
    panel=materialize_panel(root,selection_count=16,report_count=16)
    previous_panel={r['task_id']:r for r in read_json(previous/'panel.json')['tasks']}
    links=['initial.safetensors','task_lora_contract.json','bank_scale.json','base/merged_targets.safetensors']
    condition_mapping={}
    for row in panel['tasks']:
        task=row['task_id'];before=previous_panel[task]
        if any(row[k]!=before[k] for k in ('init_order','video_order','bootstrap_state_id','policy_videos',
                'practice_states','query_states','teacher_report_states','report_states')):
            raise ValueError('original initialization/video/bootstrap/practice/query/report mapping changed')
        E=read_json(previous/'teachers'/f'task_{task:04d}'/'E.json')
        H=read_json(previous/'bootstrap'/f'task_{task:04d}'/'complete.json')
        if not E['complete'] or not H['complete'] or H['reference']!='MT300' or not Path(E['weights']).is_file() or not Path(H['history']).is_file():
            raise ValueError('existing E weights and actual MT bootstrap are required')
        links.extend([f'teachers/task_{task:04d}/E.json',f'bootstrap/task_{task:04d}/complete.json'])
        source_path=previous/'student_data'/old['student_refresh']['version']/f'task_{task:04d}'/'path.json'
        original=read_json(source_path)
        if (original['completed_rounds']!=1 or len(original['boundaries'])!=1
                or original['teacher_demo'] not in row['policy_videos']):
            raise ValueError('same actual first-student teaching/practice condition required')
        practice=original['boundaries'][0]['row'];ordinal=original['path_ordinal']
        if (practice['init_state_id']!=row['practice_states'][(ordinal+1)%len(row['practice_states'])]
                or practice['env_seed']!=seed(32,original['noise_root'],1)
                or practice['policy_seed_root']!=practice['env_seed']):
            raise ValueError('previous actual practice mapping differs from the native consumer')
        condition_mapping[str(task)]=dict(teacher_demo=original['teacher_demo'],path_ordinal=ordinal,
            noise_root=original['noise_root'],practice_state_id=practice['init_state_id'],
            practice_noise_root=practice['env_seed'],source_path=str(source_path))
    for name in links:
        source=(previous/name).resolve(strict=True);target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.resolve()!=source:raise ValueError('reused bank asset destination changed')
        else:target.symlink_to(source)
    operator_scale=freeze_operator_scale(root,previous)
    coordinates={**read_json(previous/'coordinates.json'),'schema_version':SCHEMA,
        'initial':str(root/'initial.safetensors'),'reused_from':str(previous/'coordinates.json'),'S_G_not_yet_formed':False}
    if (root/'coordinates.json').exists() and read_json(root/'coordinates.json')!=coordinates:
        raise ValueError('new coordinates changed')
    write_json_atomic(root/'coordinates.json',coordinates)
    contract = dict(schema_version=SCHEMA, stage='G-and-student-readout',
        design='docs/designs/video_guided_proposal_writer_20261010.md',
        source=SOURCE,mt_checkpoint=str(MT_PATH),training_tasks=list(TASKS),panel=str(root/'panel.json'),
        environment=old['environment'],seed=7,meta_seeds=[202610101,202610102],meta_rank=32,meta_alpha=32,
        task_rank=8,task_targets=38,task_factors=76,probe_seed=1729,
        limits=dict(GPU_hours=None,GPU_hours_hard_cap_active=False,added_peak_GiB=16,carry_in_GPU_hours=0.,
            GPU_hours_policy='actual cost retained; no cumulative GPU-hour hard limit'),
        previous_root=str(previous),previous_student_version=old['student_refresh']['version'],
        reused_bank=dict(E='same selected teacher weights',S_G='same task-equal fixed scale',
            H='same actual MT bootstrap; no new teacher trajectories',MT='same frozen merged base and task initial'),
        task_initial=str(root/'initial.safetensors'),coordinates=str(root/'coordinates.json'),
        bank_scale=str(root/'bank_scale.json'),operator_scale=str(operator_scale),updates=dict(G=[stop]),fixed_G_update=stop,
        student_refresh=dict(version=f'G{stop}_operator_effect_first_students',G_update=stop,task_count=36,
            valid_proposal_rule='one deterministic full endpoint; invalid is failure; no success resampling',
            MT_bootstrap_reference_reused=True,query_states_per_parameter=16,readonly_E_query=True,
            readonly_previous_G_query=True,condition_mapping=condition_mapping,
            parameters_and_H_single_source=True,post_refresh_G_only=False),
        generator_parameterization=dict(input='current p=(P-P0)/S_G plus valid mask; true L/V/H context',
            target='complete deployed s*B_E@A_E; supervision only',output='P0+S_G*e_hat',objective=G_OBJECTIVE,
            initial_output='restore complete learned G864; original update0 head was zero',parameter_noise=False,parameter_time=False,
            current_P_H_are_conditions_not_target_center=True,decoder_layers=6,sender_normalization=True),
        logical_batch=dict(teacher_queries=28,G_conditions=4,actor_conditions=4),
        teacher={**old['teacher'],'readonly_reuse':True,'source_root':str(previous)},
        task_parent_history_sampling='equal task; equal actual parent within task; uniform real H within parent',
        optimizer=OPTIMIZER,selector_schema=SELECTOR_SCHEMA,future_budget_model_input=False,
        compilation=dict(bootstrap='same actual MT episode before first G',proposal_attempts=32,
            proposals_per_decision=1,parameter_decoder_calls=1,response_observations=16,
            generation='deterministic_MT_centered_endpoint',native_policy_flow_steps=10),
        selector_training=dict(**{k:v for k,v in old['selector_training'].items() if k not in
            ('query_states_per_parameter','registered_theta_updates','formal_update_window_not_yet_registered')},
            query_states_per_parameter=16,registered_theta_updates=None,formal_update_window_not_yet_registered=True),
        condition_estimate_seconds={'student-data':300.},
        expected_new_episodes=612,expected_new_query_episodes=576,
        query_independent_of_teacher_selection=False,query_teacher_selection_overlap_per_task=2,
        exact_resume='v8 same-objective complete checkpoints only; original v7 G864 is an explicit objective migration',
        git=git_state(),information_wall='no held offline labels, no per-H Q, no rec/keep; query never in H')
    contract.update(learning)
    path=root/'run_contract.json'
    if path.exists() and read_json(path) != contract:
        raise ValueError('new run contract changed; append explicit revision rather than overwrite')
    if not path.exists():write_json_atomic(path,contract)
    write_json_atomic(root/'bank_ready.json',dict(complete=True,task_count=36,initial_parent='MT300 only',
        history_episodes=36,readonly_source=str(previous),bank_scale=str(root/'bank_scale.json'),retrained_teachers=False))
    print(json.dumps(dict(contract=str(path),tasks=len(panel['tasks']),updates=contract['updates'])))


def teacher_root(root, task):
    return Path(root)/'teachers'/f'task_{int(task):04d}'


def node_weights(root, task, update):
    return (Path(root)/'initial.safetensors' if int(update)==0 else
            teacher_root(root,task)/'checkpoints'/f'update_{int(update):08d}'/'lora.safetensors')


def fit(args, runtime):
    contract=read_json(args.root/'run_contract.json');nodes=contract['updates']['teacher']
    if args.stop not in nodes:raise ValueError('unregistered teacher checkpoint')
    panel=task_panel(args.root,args.task);destination=teacher_root(args.root,args.task)
    complete=destination/'checkpoints'/f'update_{args.stop:08d}'/'manifest.json'
    if complete.exists() and read_json(complete)['complete']:return
    event=dict(event_id=f'task_{args.task:04d}_common_teacher_trajectory',task_id=args.task,
        teacher_demo=panel['teacher_videos'][0],event_ordinal=0,schema_version=SCHEMA,
        base_source=SOURCE,MT=str(MT_PATH),task_rank=8,initial=str(args.root/'initial.safetensors'),
        action_pool=contract['teacher']['action_demos'],proximal_target='fixed initialMT',
        labels='native demoFM only; no privileged expert, rec or keep')
    if args.resume is None:
        previous=[u for u in nodes if u<args.stop and (destination/'checkpoints'/f'update_{u:08d}'/'manifest.json').exists()]
        if previous:args.resume=destination/'checkpoints'/f'update_{max(previous):08d}'
    fit_teacher(runtime,event,runtime.mt,destination,microbatch=args.microbatch,stop=args.stop,
        resume=args.resume,logical_queries=contract['logical_batch']['teacher_queries'],
        query_demo_indices=contract['teacher']['action_demos'],checkpoint_updates=tuple(nodes))
    print(json.dumps(dict(task=args.task,trained_to=args.stop,trajectory=event['event_id'])))


def readout(args, runtime, runner, environment):
    contract=read_json(args.root/'run_contract.json')
    if args.stop not in contract['teacher']['readout_updates']:raise ValueError('unregistered readout node')
    panel=task_panel(args.root,args.task)
    task=next(t for t in environment['tasks'] if t['global_task_id']==args.task)
    weights=load_file(str(node_weights(args.root,args.task,args.stop)),device=str(runtime.device))
    reference=parameter_ref(args.task,args.stop)
    destination=teacher_root(args.root,args.task)/'selection'/f'update_{args.stop:08d}'
    destination.mkdir(parents=True,exist_ok=True)
    requests=[];bootstrap=args.root/'bootstrap'/f'task_{args.task:04d}'/'complete.json'
    for state_id in panel['teacher_selection_states']:
        identity=f'task_{args.task:04d}_u{args.stop:08d}_selection_{state_id:03d}'
        existing=destination/f'state_{state_id:03d}.json'
        capture=args.stop==0 and state_id==panel['bootstrap_state_id']
        prior=read_json(existing) if existing.exists() else None
        if prior and (not capture or bootstrap.exists()):continue
        if prior and 'history' in prior:
            write_json_atomic(bootstrap,dict(complete=True,task=args.task,state_id=state_id,
                reference='MT300',history=prior['history'],selection_source=str(existing),
                same_actual_selection_rollout=True,original_episode_evidence=prior))
            continue
        requests.append(dict(task=task,state=weights,parameter_ref=reference,state_id=state_id,
            noise_root=seed(4,args.task,state_id),episode_id=identity+('_bootstrap_recollect' if prior else ''),
            snapshots=False,uses_teaching=False,capture_phi=False,capture_hidden=capture,
            capture_history=capture,bootstrap_only=bool(prior)))
    for output in runner.run(requests):
        state_id=output['request']['state_id']
        row=dict(**output['row'],global_task_id=args.task,checkpoint_update=args.stop,parameter_ref=reference,
            uses=['teacher_selection'],independent_report=False,checkpoint=str(node_weights(args.root,args.task,args.stop)),
            fold=next(i for i,fold in enumerate(panel['selection_folds']) if state_id in fold))
        if output['history'].records:
            row['first_native_normalized_actions']=output['history'].records[0]['normalized_actions'].tolist()
        if output['request']['capture_history']:
            bootstrap.parent.mkdir(parents=True,exist_ok=True)
            history_path=bootstrap.parent/'history.pt.gz';save_history(history_path,output['history'])
            write_json_atomic(bootstrap,dict(complete=True,task=args.task,state_id=state_id,reference='MT300',
                history=str(history_path),selection_source=str(destination/f'state_{state_id:03d}.json'),
                same_actual_selection_rollout=not output['request']['bootstrap_only'],original_episode_evidence=row))
            row.update(history=str(history_path),history_records=len(output['history'].records),
                bootstrap_self_practice=True,same_actual_rollout_history=True)
        if output['request']['bootstrap_only']:continue
        write_json_atomic(destination/f'state_{state_id:03d}.json',row)
    rows=[read_json(destination/f'state_{i:03d}.json') for i in panel['teacher_selection_states']]
    write_json_atomic(destination/'complete.json',dict(complete=True,task=args.task,checkpoint_update=args.stop,
        rows=rows,components=runner.components,environment_steps=runner.environment_steps,
        history_bytes=sum(Path(r['history']).stat().st_size for r in rows if 'history' in r),
        teacher_histories_not_learning_data=True,bootstrap_history=str(bootstrap) if args.stop==0 else None))
    print(json.dumps(dict(task=args.task,node=args.stop,readout_rows=len(rows),
        bootstrap_history_available=bootstrap.exists(),other_selection_histories_captured=False)))


def select_teacher(root, task):
    contract=read_json(Path(root)/'run_contract.json');panel=task_panel(root,task)
    existing=teacher_root(root,task)/'E.json'
    if existing.exists():return read_json(existing)
    nodes=contract['teacher']['readout_updates'];scores={};candidates=[]
    omission=Path(root)/'scientific_omissions'/f'task_{task:04d}.json'
    omitted=set(read_json(omission)['omitted_jobs']) if omission.exists() else set()
    for update in nodes:
        path=teacher_root(root,task)/'selection'/f'update_{update:08d}'/'complete.json'
        if not path.exists() and f'readout_{task:04d}_{update:08d}' in omitted:continue
        record=read_json(path)
        if not record['complete'] or len(record['rows']) != contract['teacher']['selection_per_node']:
            raise ValueError('all registered teacher node readouts required before E choice')
        rows=record['rows'];scores[update]=sum(bool(r['success']) for r in rows)
        candidates.append(dict(update=update,reference=parameter_ref(task,update),
            weights=str(node_weights(root,task,update)),
            rows=rows,selection_successes=scores[update]))
    best=max(scores,key=lambda u:(scores[u],-u))
    result=dict(complete=True,task=task,selected_update=best,reference=parameter_ref(task,best),
        weights=str(node_weights(root,task,best)),evaluated_candidates=candidates,selection_states=panel['teacher_selection_states'],
        selection_noise=f"{contract['teacher']['selection_per_node']}-row teacher selection; no held/formal checkpoint choice",
        bootstrap=str(Path(root)/'bootstrap'/f'task_{task:04d}'/'complete.json'),
        choice_rule='max success; earliest tie including MT',shared_target_for_all_parent_H=True,
        evaluated_nodes=sorted(scores),scientifically_omitted_nodes=[u for u in nodes if u not in scores],
        saturation_proof=str(omission) if omission.exists() else None)
    write_json_atomic(teacher_root(root,task)/'E.json',result)
    return result


def report_references(root, asset_root=ASSET_ROOT):
    """Index compatible fixed MT originals; row priority never depends on success."""
    root=Path(root);current=read_json(root/'run_contract.json')
    source=Path('/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/MT/evaluation/correct144')
    previous=read_json(source/'run_contract.json');results=read_json(source/'results.json')
    from .evaluation import reference_compatibility
    reasons=reference_compatibility(previous,asset_root)
    tasks={t['global_task_id']:t for t in current['environment']['tasks']};panel={r['task_id']:r for r in materialize_panel(root)['tasks']}
    references={};missing=[]
    for index,row in enumerate(results['rows']):
        task=int(row['operator_read_write_lora']['global_task_id']);state=int(row['init_state_id'])
        if task not in tasks or state not in panel[task]['teacher_report_states']:continue
        local_reasons=list(reasons);spec=tasks[task]
        if (row['suite'],row['task_id'],row['language'])!=(spec['suite'],spec['task_id'],spec['language']):local_reasons.append('task specification')
        if row.get('env_seed')!=7 or row.get('policy_seed_root')!=7 or not row.get('policy_noise_seeds'):local_reasons.append('actual environment/policy RNG')
        scene=row.get('scene_reference');registered=previous.get('operator_read_write_scene')
        if not scene or not registered or not Path(scene['path']).is_file():local_reasons.append('actual registered scene')
        if local_reasons:
            missing.append(dict(task=task,state=state,source=str(source/'results.json'),row_index=index,incompatible=local_reasons));continue
        key=f'{task}:{state}'
        if key in references:raise ValueError('duplicate fixed MT reference')
        raw={k:row[k] for k in ('suite','task_id','language','split_role','init_state_id','env_seed','policy_seed_root','policy_noise_seeds','scene_reference','success','steps')}
        raw['environment_steps']=int(row['steps'])+10
        references[key]=dict(task=task,state=state,noise_root=7,registered_scene=registered,MT_row=raw,
            source=str(source/'results.json'),source_contract=str(source/'run_contract.json'),source_row_index=index,
            reused=True,new_environment_steps=0,new_GPU_hours=0,
            equivalence='same source1000+MT300, frozen MT merged base plus B0 rank8; native BF16 reduction differences accepted')
    record=dict(complete=True,report_states_unchanged=True,selection_and_H_unchanged=True,references=references,
        reused_MT_rows=len(references),source_priority=[str(source/'results.json')],incompatible_rows=missing,
        success_not_used_for_reference_choice=True,score_only_no_bootstrap_H=True,
        unmatched_states='new strictly paired E/MT official rollout with registered seed(5,task,state)',
        report_logical_rows=36*2*current['teacher']['report_per_arm'])
    write_json_atomic(root/'teacher_report_references.json',record)
    return record


def teacher_report(args,runtime,runner,environment):
    chosen=select_teacher(args.root,args.task);panel=task_panel(args.root,args.task)
    task=next(t for t in environment['tasks'] if t['global_task_id']==args.task)
    destination=teacher_root(args.root,args.task)/'report';destination.mkdir(parents=True,exist_ok=True)
    index=read_json(args.root/'teacher_report_references.json')['references']
    weights=load_file(chosen['weights'],device=str(runtime.device));requests=[];reused=0;new_episodes=0
    def publish(row,arm,state,reference,provenance):
        output={**row,'global_task_id':args.task,'arm':arm,'checkpoint_update':chosen['selected_update'] if arm=='E' else 0,
            'actual_parameter_ref':reference,'independent_report':True,'uses':['independent_teacher_report'],
            'query_excluded_from_H':True,'checkpoint_selection_use':False,'reference_provenance':provenance}
        write_json_atomic(destination/f'{arm}_{state:03d}.json',output)
    for state in panel['teacher_report_states']:
        saved=lambda arm:(destination/f'{arm}_{state:03d}.json').exists()
        reference=index.get(f'{args.task}:{state}')
        if reference is not None:
            provenance={k:v for k,v in reference.items() if k not in ('MT_row','registered_scene')}
            if not saved('MT'):publish(reference['MT_row'],'MT',state,'MT300',provenance);reused+=1
            if chosen['selected_update']==0:
                if not saved('E'):publish(reference['MT_row'],'E',state,'MT300',dict(**provenance,same_policy_E_MT_alias=True))
                continue
        noise_root=reference['noise_root'] if reference is not None else seed(5,args.task,state)
        aliases=(['E','MT'] if chosen['selected_update']==0 else ['E'])
        if any(not saved(arm) for arm in aliases):
            requests.append(dict(task=task,state=weights,parameter_ref=chosen['reference'],state_id=state,
                noise_root=noise_root,episode_id=f'task_{args.task:04d}_report_E_{state:03d}',snapshots=False,
                uses_teaching=False,aliases=aliases,capture_history=False,capture_hidden=False,
                registered_scene=reference['registered_scene'] if reference else None))
        if chosen['selected_update']!=0 and reference is None and not saved('MT'):
            requests.append(dict(task=task,state=runtime.mt,parameter_ref='MT300',state_id=state,
                noise_root=noise_root,episode_id=f'task_{args.task:04d}_report_MT_{state:03d}',snapshots=False,
                uses_teaching=False,aliases=['MT'],capture_history=False,capture_hidden=False))
    before_steps=runner.environment_steps
    for output in runner.run(requests):
        request=output['request'];state=request['state_id'];new_episodes+=1
        row=dict(output['row'])
        if output['history'].records:
            row['first_native_normalized_actions']=output['history'].records[0]['normalized_actions'].tolist()
        for arm in request['aliases']:
            publish(row,arm,state,request['parameter_ref'],dict(reused=False,new_actual_episode=request['episode_id'],
                same_policy_E_MT_alias=len(request['aliases'])==2,registered_scene=request.get('registered_scene'),
                new_environment_steps=row['environment_steps']))
    rows=[read_json(destination/f'{arm}_{state:03d}.json') for arm in ('E','MT') for state in panel['teacher_report_states']]
    write_json_atomic(destination/'complete.json',dict(complete=True,task=args.task,rows=rows,
        logical_rows=len(rows),actual_new_episodes_this_attempt=new_episodes,actual_new_environment_steps_this_attempt=runner.environment_steps-before_steps,
        historical_MT_rows=sum(r['arm']=='MT' and r['reference_provenance']['reused'] for r in rows),
        report_excluded_from_H_and_gradient=True,components=runner.components))
    print(json.dumps(dict(task=args.task,independent_logical_rows=len(rows),actual_new_episodes_this_attempt=new_episodes)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','report-references','formal-prepare','formal','teacher','readout','teacher-report','G','student-data','endpoint-query','local','RL','report'])
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--asset-root',type=Path,default=ASSET_ROOT)
    parser.add_argument('--task',type=int,default=0)
    parser.add_argument('--stop',type=int)
    parser.add_argument('--reuse-root',type=Path,help='immutable complete previous E/MT/bootstrap/scale root')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--condition-ordinal',type=int)
    parser.add_argument('--worker-id',help='persistent formal dynamic queue worker identity')
    parser.add_argument('--condition-batch',type=int,help='measured physical independent condition group')
    parser.add_argument('--video-arm',choices=['correct','other','wrong'],default='correct')
    parser.add_argument('--microbatch',type=int,default=28)
    parser.add_argument('--frame-chunk',type=int,default=16)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--physical-gpu',type=int)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args);return
    if args.command=='report-references':
        result=report_references(args.root,args.asset_root);print(json.dumps(dict(reused_MT_rows=result['reused_MT_rows'],incompatible_rows=len(result['incompatible_rows']))));return
    if args.command=='formal-prepare':
        from .evaluation import prepare_formal
        result=prepare_formal(args);print(json.dumps(dict(formal_rows=len(result['rows']),held_offline_labels_read=False)));return
    identity=git_state()
    if identity['dirty_paths'] or identity['branch']:
        raise ValueError('formal consumers require a clean pushed detached frozen worktree')
    subprocess.run(['git','merge-base','--is-ancestor',identity['commit'],'origin/main'],check=True)
    write_json_atomic(args.root/'code'/f'{args.command}_{identity["commit"]}.json',identity)
    shared=args.command in ('G','local','RL')
    if shared:
        args.physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))])
        if args.command not in read_json(args.root/'run_contract.json')['updates']:
            raise ValueError('shared learning update window must be registered before formal launch')
    if args.physical_gpu is None:parser.error('GPU consumer requires its actual physical GPU')
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8)
    environment=read_json(args.root/'formal_panel.json')['environment'] if args.command=='formal' else read_json(args.root/'run_contract.json')['environment']
    if args.command=='formal' and (not args.worker_id or not args.condition_batch or args.condition_batch<1):
        parser.error('formal needs its persistent queue worker identity and measured condition batch')
    resident_student=args.command in ('student-data','endpoint-query') and bool(args.worker_id)
    if args.command=='endpoint-query' and not resident_student:parser.error('endpoint-query needs its persistent worker identity')
    if resident_student and (not args.condition_batch or args.condition_batch<1):
        parser.error('resident student needs its physical condition batch')
    if not shared and args.command!='formal' and not resident_student and args.task not in [t['global_task_id'] for t in environment['tasks']]:raise ValueError('task outside registered consumer role')
    runtime=create_runtime(args,context,environment)
    consumer=args.worker_id or (str(args.stop) if shared else f'task{args.task}')
    write_json_atomic(args.root/'code'/f'{args.command}_{consumer}_{identity["commit"]}_rank{context.local_rank:02d}_execution.json',
        dict(frame_chunk=runtime.frame_chunk,local_rank=context.local_rank,world_size=context.world_size,
            physical_gpu=args.physical_gpu,logical_condition_batch=4 if shared else None))
    runner=None
    try:
        if args.command=='teacher':fit(args,runtime);return
        os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
        from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
        prepare_libero_config(args.root/'libero_config')
        configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
        runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
        if args.command=='readout':readout(args,runtime,runner,environment)
        elif args.command=='teacher-report':teacher_report(args,runtime,runner,environment)
        elif args.command=='formal':
            from .evaluation import formal_worker
            formal_worker(args,runtime,runner)
        elif resident_student:
            from .evaluation import student_worker
            student_worker(args,runtime,runner,environment)
        else:
            from .pipeline import run
            run(args,runtime,runner,environment,context)
    finally:
        if runner is not None:runner.close()
        runtime.close()


if __name__=='__main__':main()
