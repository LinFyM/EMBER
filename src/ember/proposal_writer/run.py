"""Canonical full36 teacher preparation and shared Writer consumers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.writer.practice import EpisodeRunner
from ember.lora import identity_lora_state, LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank

from .contract import (RUN_ROOT, ASSET_ROOT, TASKS, SOURCE, MT_PATH, OPTIMIZER, SCHEMA,
                       PRIOR_ROOT, SELECTOR_SCHEMA, materialize_panel, environment_contract, seed, parameter_ref)
from .runtime import Runtime
from .teacher import fit_teacher
from .path import save_history


def task_panel(root, task):
    return next(row for row in materialize_panel(root)['tasks'] if row['task_id'] == task)


def save_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))


def coordinates(root, asset_root):
    """Actual deterministic Λ0 and fixed proximal units, before any teacher update."""
    if (root/'coordinates.json').exists():
        return
    original = load_pi05_lora_contract(asset_root/SOURCE['lora_contract'])
    current = derive_pi05_lora_rank(original, rank=8)
    initial = identity_lora_state(current)
    save_state(root/'initial.safetensors', initial)
    mt = load_file(str(MT_PATH))
    scales, dense_rms = {}, {}
    for target in current.targets:
        a_name, b_name = target.name + LORA_A_SUFFIX, target.name + LORA_B_SUFFIX
        a = float(initial[a_name].square().mean().sqrt())
        dense = mt[b_name].float() @ mt[a_name].float() * (original.alpha/original.rank)
        dense_rms[target.name] = float(dense.square().mean().sqrt())
        scales[a_name], scales[b_name] = a, dense_rms[target.name]/(8**.5*a)
    write_json_atomic(root/'task_lora_contract.json', current.to_dict())
    write_json_atomic(root/'coordinates.json', dict(schema_version=SCHEMA, task_rank=8,
        valid_coordinates=current.parameter_count, initial=str(root/'initial.safetensors'), initial_seed=7,
        base_source=SOURCE, MT=str(MT_PATH), base_rule='FP32(W_source+alpha/rank*B_MT@A_MT) then native dtype',
        S_prox=scales, actual_MT_dense_RMS=dense_rms, S_G_not_yet_formed=True,
        center='shared actual nonzero Kaiming A0/B0, no gauge restart'))



def prepare(args):
    if any(v is None for v in (args.teacher_stop,args.checkpoint_interval,args.selection_count)):
        raise ValueError('formal teacher update/CP/selection values must be explicitly frozen')
    if args.teacher_stop % args.checkpoint_interval:
        raise ValueError('teacher terminal must be a registered complete checkpoint')
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    panel = materialize_panel(root,selection_count=args.selection_count,report_count=args.report_count)
    coordinates(root,args.asset_root)
    environment = environment_contract(args.asset_root,root=root)
    contract = dict(schema_version=SCHEMA, stage='teacher-data',
        design='docs/designs/video_guided_proposal_writer_20261010.md',
        source=SOURCE,mt_checkpoint=str(MT_PATH),training_tasks=list(TASKS),panel=str(root/'panel.json'),
        environment=environment,seed=7,meta_seeds=[202610101,202610102],meta_rank=32,meta_alpha=32,
        task_rank=8,task_targets=38,task_factors=76,probe_seed=1729,
        limits=dict(GPU_hours=None,GPU_hours_hard_cap_active=False,added_peak_GiB=256,
            carry_in_GPU_hours=read_json(PRIOR_ROOT/'analysis/final_billing.json')['GPU_hours']+sum(read_json(Path(x))['GPU_hours'] for x in args.profile_receipts),
                    GPU_hours_policy='Owner removed cumulative GPU-hour launch/resume limits'),
        historical_roots=[str(PRIOR_ROOT)],historical_profiles=args.profile_receipts,
        task_initial=str(root/'initial.safetensors'),coordinates=str(root/'coordinates.json'),
        bank_scale=str(root/'bank_scale.json'),
        updates=dict(teacher=list(range(100,args.teacher_stop+1,100))),
        logical_batch=dict(teacher_queries=28,G_conditions=4,actor_conditions=4),
        teacher=dict(action_demos=list(range(46)),internal_unfitted_demos=list(range(46,50)),
            trajectory='one per task, continuous optimizer/RNG',loss='native demoFM + 1e-3 fixed-initialMT prox',
            proximal_target=str(root/'initial.safetensors'),save_interval=100,
            readout_updates=[0,*range(args.checkpoint_interval,args.teacher_stop+1,args.checkpoint_interval)],
            selection_H_per_node=args.selection_count,
            selection_folds=[args.selection_count//2]*2,report_per_arm=args.report_count,
            choice='maximum registered selection success count, ties earliest including MT',
            independent_report_not_in_H_or_gradient=True),
        task_parent_history_sampling='equal task; equal actual parent within task; uniform real H within parent',
        optimizer=OPTIMIZER,selector_schema=SELECTOR_SCHEMA,future_budget_model_input=False,
        compilation=dict(proposals_per_decision=4,proposal_attempts=32,parameter_euler_steps=16,
            response_observations=16,bootstrap='actual MT episode before first G',
            future_budget_model_input=False),
        selector_training=dict(schema=SELECTOR_SCHEMA,rounds=5,continue_cost=.001,
            collection='complete parent+STOP logits; conditional nonSTOP parent plus practice',
            objective='mean depths1..5 exact STOP/final expectation; conditional score plus direct gradient'),
        deadline=dict(authority_UTC='2026-10-11T06:30:00+00:00',target_UTC='2026-10-11T12:30:00+00:00',
            bank_ready_prediction_UTC=['2026-10-11T12:40:00+00:00','2026-10-11T13:05:00+00:00'],
            revised_from_original_target=True,fast_device_B28_seconds=2.0744460243731737,fast_capture16_seconds=277.30491822306067),
        exact_resume='full teacher optimizer/sampler/RNG; shared checkpoints retain topology and physical-migration identity',
        git=git_state(),information_wall='no held offline labels, no per-H Q, no rec/keep; query never in H')
    path=root/'run_contract.json'
    if path.exists() and read_json(path) != contract:
        raise ValueError('new run contract changed; append explicit revision rather than overwrite')
    if not path.exists():write_json_atomic(path,contract)
    base=root/'base';base.mkdir(exist_ok=True)
    source=PRIOR_ROOT/'base/merged_targets.safetensors'
    if not source.is_file():raise ValueError('canonical frozen merged-MT targets absent')
    if not (base/source.name).exists():(base/source.name).symlink_to(source.resolve())
    print(json.dumps(dict(contract=str(path),tasks=len(panel['tasks']),updates=contract['updates']['teacher'])))


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
    requests=[]
    for state_id in panel['teacher_selection_states']:
        identity=f'task_{args.task:04d}_u{args.stop:08d}_selection_{state_id:03d}'
        if (destination/f'state_{state_id:03d}.json').exists():continue
        requests.append(dict(task=task,state=weights,parameter_ref=reference,state_id=state_id,
            noise_root=seed(4,args.task,state_id),episode_id=identity,snapshots=False,uses_teaching=False))
    for output in runner.run(requests):
        state_id=output['request']['state_id'];history_path=destination/f'state_{state_id:03d}.history.pt.gz'
        save_history(history_path,output['history'])
        row=dict(**output['row'],global_task_id=args.task,checkpoint_update=args.stop,parameter_ref=reference,
            history=str(history_path),history_records=len(output['history'].records),
            uses=['teacher_selection','G_history','selector_supervision'],independent_report=False,
            same_actual_rollout_history=True,checkpoint=str(node_weights(args.root,args.task,args.stop)),
            fold=next(i for i,fold in enumerate(panel['selection_folds']) if state_id in fold))
        if output['history'].records:
            row['first_native_normalized_actions']=output['history'].records[0]['normalized_actions'].tolist()
        write_json_atomic(destination/f'state_{state_id:03d}.json',row)
    rows=[read_json(destination/f'state_{i:03d}.json') for i in panel['teacher_selection_states']]
    write_json_atomic(destination/'complete.json',dict(complete=True,task=args.task,checkpoint_update=args.stop,
        rows=rows,components=runner.components,environment_steps=runner.environment_steps,
        history_bytes=sum(Path(r['history']).stat().st_size for r in rows)))
    print(json.dumps(dict(task=args.task,node=args.stop,readout_rows=len(rows),capture_reused=True)))


def select_teacher(root, task):
    contract=read_json(Path(root)/'run_contract.json');panel=task_panel(root,task)
    existing=teacher_root(root,task)/'E.json'
    if existing.exists():return read_json(existing)
    nodes=contract['teacher']['readout_updates'];scores={};parents=[]
    for update in nodes:
        path=teacher_root(root,task)/'selection'/f'update_{update:08d}'/'complete.json'
        record=read_json(path)
        if not record['complete'] or len(record['rows']) != contract['teacher']['selection_H_per_node']:
            raise ValueError('all registered teacher node readouts required before E choice')
        rows=record['rows'];scores[update]=sum(bool(r['success']) for r in rows)
        parents.append(dict(update=update,reference=parameter_ref(task,update),
            weights=str(node_weights(root,task,update)),histories=[r['history'] for r in rows],
            rows=rows,selection_successes=scores[update]))
    best=max(nodes,key=lambda u:(scores[u],-u))
    result=dict(complete=True,task=task,selected_update=best,reference=parameter_ref(task,best),
        weights=str(node_weights(root,task,best)),parents=parents,selection_states=panel['teacher_selection_states'],
        selection_folds=panel['selection_folds'],selection_noise=f"{contract['teacher']['selection_H_per_node']}-row teacher selection; no held/formal checkpoint choice",
        choice_rule='max success; earliest tie including MT',shared_target_for_all_parent_H=True)
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
                uses_teaching=False,aliases=aliases,registered_scene=reference['registered_scene'] if reference else None))
        if chosen['selected_update']!=0 and reference is None and not saved('MT'):
            requests.append(dict(task=task,state=runtime.mt,parameter_ref='MT300',state_id=state,
                noise_root=noise_root,episode_id=f'task_{args.task:04d}_report_MT_{state:03d}',snapshots=False,
                uses_teaching=False,aliases=['MT']))
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
    parser.add_argument('command',choices=['prepare','report-references','formal-prepare','formal','teacher','readout','teacher-report','G','student-data','local-data','local','RL','report'])
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--asset-root',type=Path,default=ASSET_ROOT)
    parser.add_argument('--task',type=int,default=0)
    parser.add_argument('--stop',type=int)
    parser.add_argument('--teacher-stop',type=int)
    parser.add_argument('--checkpoint-interval',type=int)
    parser.add_argument('--selection-count',type=int)
    parser.add_argument('--report-count',type=int,default=8)
    parser.add_argument('--profile-receipts',nargs='*',default=[])
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--condition-ordinal',type=int)
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
    if not shared and args.task not in [t['global_task_id'] for t in environment['tasks']]:raise ValueError('task outside registered consumer role')
    runtime=Runtime(context.device,asset_root=args.asset_root,root=args.root,frame_chunk=args.frame_chunk,
        task_ids=tuple(t['global_task_id'] for t in environment['tasks']),task_role=environment['role'])
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
            from .evaluation import formal_condition
            formal_condition(args,runtime,runner)
        else:
            from .pipeline import run
            run(args,runtime,runner,environment,context)
    finally:
        if runner is not None:runner.close()
        runtime.close()


if __name__=='__main__':main()
