"""Fixed revision180 validation conditions and label-side seven-arm readout."""
from pathlib import Path
import numpy as np

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.reference.contract import ASSETS,SOURCE,NORMALIZATION,EVALUATION,ROOT as REFERENCE_ROOT
from ember.reference.analysis import paired,pairing_valid
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .analysis import grouped

BASE=REFERENCE_ROOT.parent
ROOT=BASE/'functional_feedback_validation180_seed7'
VIDEO_ROOT=BASE/'video_initial_seed7'
INITIAL_READER=VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
REVISION_ROOT=BASE/'functional_feedback_seed7'
REVISION_CHECKPOINT=REVISION_ROOT/'checkpoints/update_00000180/state.pt'
BASIS=BASE/'control_linear64_seed7/basis/state.pt'
SCHEMA='ember_source_functional_feedback_validation180_r8_seed7_v1'
ARM='functional_feedback_validation180'
TASKS=(3,6,11,16,23,26,31,39)
LIMITS=dict(physical_submissions=33264,control_steps=3426080,dummy_steps=332640,total_steps=3446720,
    engineering_extra_attempts=64,added_peak_GiB=96,shared_optimizer_updates=0,conditions=400,logical_query_episodes=400,
    condition_practice_steps=8192,frozen_H_cache_GiB=16,frozen_M_cache_GiB=1,active_conditions=32)
OLD_ARMS=('Reader900','Reader1800','MT153','G165','D72','D144')
NEW_ARMS=(ARM,)


def condition_id(task,final_init):
    if task not in TASKS or final_init not in range(50):raise ValueError('only the fixed8 validation tasks and50 initial states are authorized')
    return f'task_{task:04d}_state_{final_init:03d}'


def _existing(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'fixed source asset missing: {path}')
    return path


def scene_identity(task,state_id,registered_scene):
    file=_existing(Path(registered_scene['root'])/f'{task["suite"]}_task_{task["task_id"]:02d}_state_{state_id:03d}.npz')
    return dict(path=str(file),bytes=file.stat().st_size)


def _validate_episode(row,task,states,noise):
    expected=dict(global_task_id=task['global_task_id'],suite=task['suite'],task_id=task['task_id'],language=task['language'],
        env_seed=noise,policy_seed_root=noise,settling_steps=10,full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False)
    if any(row.get(k)!=v for k,v in expected.items()) or row['init_state_id'] not in states:
        raise ValueError('actual row lost its registered task/native execution/RNG')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon'] or row.get('model_failure')
            or row.get('budget_truncated') or row.get('failure_type') or (not row['success'] and steps!=task['horizon'])
            or row.get('environment_steps')!=steps+10):raise ValueError('incomplete or engineering-failed row cannot complete a scientific panel')
    stream=[policy_noise_seed(noise,task['suite'],task['task_id'],row['init_state_id'],i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=stream or row.get('first_noise_seed')!=stream[0]:raise ValueError('absolute native policy noise changed')


def _units():
    import torch
    saved=torch.load(INITIAL_READER,map_location='cpu',weights_only=False,mmap=True)
    if saved['next_update']!=900 or saved['phase']!='video_initial' or saved['frozen_basis']!=str(BASIS):
        raise ValueError('fixed units must identify actual Reader900 and unique basis')
    units=saved['code_units'].float()
    if units.shape!=(64,) or not bool(units.isfinite().all()) or bool((units<=0).any()):raise ValueError('code units must be finite positive64')
    return units


def _initial_controller(task,state,teacher,units):
    import torch
    file=_existing(VIDEO_ROOT/'generated/update_00000900/validation'/f'task_{task:04d}'/f'state_{state:03d}.safetensors')
    value=read_json(file.with_suffix('.json'))
    expected=dict(checkpoint=str(INITIAL_READER),basis_checkpoint=str(BASIS),path=str(file),task=task,
        teacher_demo=teacher,init_state_id=state,reader_update=900,phase='video_initial',role='validation',rank=8,targets=38,
        held_offline_labels_read=False,teacher_input='pure_image_prefix_and_frame_indices')
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('cached held Reader900 controller provenance changed')
    y=torch.as_tensor(value['code'],dtype=torch.float32)/units
    if y.shape!=(64,) or not bool(y.isfinite().all()):raise ValueError('cached initial code is not finite64')
    return y.tolist(),str(file),str(file.with_suffix('.json'))


def _conditions(environment):
    from ember.expert_manifold.video_schedule import condition_demo_index
    tasks={t['global_task_id']:t for t in environment['tasks']};units=_units();conditions=[]
    panel=read_json(VIDEO_ROOT/'panels.json')['validation']
    if len(panel)!=400 or {(r['task'],r['init_state_id']) for r in panel}!={(t,s) for t in TASKS for s in range(50)}:
        raise ValueError('validation requires all400 distinct fixed task/video conditions')
    for row in panel:
        task=row['task'];state=row['init_state_id'];spec=tasks[task]
        teacher=condition_demo_index(7,spec['suite'],spec['task_id'],state,condition='correct',demo_count=50,sampling_mode='without_replacement')
        if row['teacher_demo']!=teacher or row['noise_root']!=7:raise ValueError('official seed7 teaching/final schedule changed')
        y,lora,provenance=_initial_controller(task,state,teacher,units)
        eligible=[s for s in range(50) if s!=state]
        practice=np.random.default_rng(np.random.SeedSequence([7,202610162,task,state])).permutation(eligible)[:4].tolist()
        conditions.append(dict(id=condition_id(task,state),task=task,arm=ARM,final_init=state,teacher_demo=teacher,
            video_ordinal=row['video_ordinal'],practice_states=practice,practice_noise_root=202610161,query_states=[state],query_noise_root=7,
            registered_scene=row['registered_scene'],scene_identity=scene_identity(spec,state,row['registered_scene']),
            initial_y=y,initial_LoRA=lora,initial_controller_provenance=provenance,
            teaching=dict(kind='original',held=True,task=task,demo=teacher),retain_raw_H=state==0))
    for task in TASKS:
        if {c['teacher_demo'] for c in conditions if c['task']==task}!=set(range(50)):
            raise ValueError('every validation task must use each legal teacher exactly once')
    return conditions


def prepare(root):
    """Identity-only preparation: no old final outcomes/H or held labels enter adaptation."""
    root=Path(root);file=root/'run_contract.json'
    if file.exists():
        run=read_json(file)
        if run['schema']!=SCHEMA or run['phase']!=ARM or run['limits']!=LIMITS or len(run['conditions'])!=400:
            raise ValueError('existing fixed validation180 contract is incompatible')
        return run
    original=read_json(VIDEO_ROOT/'run_contract.json');environment=dict(original['environment'])
    environment['tasks']=[t for t in environment['tasks'] if t['global_task_id'] in TASKS];environment['role']='validation'
    if {t['global_task_id'] for t in environment['tasks']}!=set(TASKS):raise ValueError('held metadata must be the fixed validation8 authority')
    run=dict(schema=SCHEMA,phase=ARM,environment=environment,conditions=_conditions(environment),limits=LIMITS,seed=7,
        revision_checkpoint=str(_existing(REVISION_CHECKPOINT)),revision_update=180,revision_producer='0b9b8080',
        frozen_basis=str(BASIS),initial_reader_checkpoint=str(INITIAL_READER),source_checkpoint=str(ASSETS/SOURCE),
        normalization=str(ASSETS/NORMALIZATION),evaluation_config=str(ASSETS/EVALUATION),
        LoRA=dict(rank=8,targets=38,alpha=8,dropout=0.,parameter_count=643584),source_frozen=True,MT_loaded=False,
        shared_updates=0,shared_gradients=False,condition_practice_steps=8192,held=True,held_offline_labels_read=False,Test=False,
        selection='one pre-fixed psi180; no checkpoint/loss/partial400 selection',
        raw_H=dict(active_conditions=32,retain_final_init=0,retained_complete_conditions=8,
            others_retire_only_after='frozen/final complete/event facts and audits atomic/no pending or consumers/full close',
            retired_raw_not_complete_H_replay=True),
        teaching_reuse='initial LoRA uses previous actual Reader900 RGB compilation; new frozen full M only when psi is called; cached is not zero teaching')
    root.mkdir(parents=True,exist_ok=True);write_json_atomic(file,run);return run


def initial_query_rows(condition):
    """LABEL SIDE ONLY: freeze with exact initial code and original LoRA file first."""
    if condition['id']!=condition_id(condition['task'],condition['final_init']):raise ValueError('initial query identity changed')
    file=VIDEO_ROOT/'evaluation/update_00000900/readout/validation'/f'task_{condition["task"]:04d}'/f'state_{condition["final_init"]:03d}.json'
    row=read_json(_existing(file))
    expected=dict(global_task_id=condition['task'],init_state_id=condition['final_init'],teacher_demo=condition['teacher_demo'],
        LoRA=condition['initial_LoRA'],reader_update=900,role='validation',env_seed=7,policy_seed_root=7)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('initial query must be its predesignated actual Reader900 row')
    if any(row.get('scene_reference',{}).get(k)!=v for k,v in condition['scene_identity'].items()):raise ValueError('initial query registered scene changed')
    return [row]


def _paired_rows(before,after):
    """Preserved MT contract: scene bytes/identity and actual noise; proprio when present."""
    a=sorted(before,key=lambda r:r['init_state_id']);b=sorted(after,key=lambda r:r['init_state_id'])
    if len(a)!=len(b):return False
    fields=('suite','task_id','init_state_id','env_seed','policy_seed_root')
    for x,y in zip(a,b,strict=True):
        if any(x.get(k)!=y.get(k) for k in fields):return False
        noises=[r.get('first_noise_seed',next(iter(r.get('policy_noise_seeds',[])),None)) for r in (x,y)]
        if noises[0] is None or noises[0]!=noises[1]:return False
        scenes=[r.get('scene_reference') for r in (x,y)]
        if not all(scenes) or any(scenes[0].get(k)!=scenes[1].get(k) for k in ('path','bytes')):return False
        if all('initial_proprio' in r and 'first_noise_seed' in r for r in (x,y)) and not pairing_valid([x],[y]):return False
    return True


def _task_summary(task,spec,arms):
    anchor=next(iter(arms.values()))
    if not anchor or any(not values or not _paired_rows(anchor,values) for values in arms.values()):raise ValueError(f'validation pairing changed for task{task}')
    comparisons={a+'_to_'+b:paired(arms[a],arms[b]) for i,a in enumerate(arms) for b in list(arms)[i+1:]}
    for value in comparisons.values():
        union=sorted(set(value['before_successes'])|set(value['after_successes']))
        value.update(success_set_union=union,jaccard=len(value['retained'])/len(union) if union else 1.)
    return dict(task=task,suite=spec['suite'],group='validation8',identity=False,cases=len(anchor),comparisons=comparisons,
        success_sets={a:sorted(r['init_state_id'] for r in rows if r['success']) for a,rows in arms.items()})


def _incumbent(path,condition,spec):
    panels=path['panels'];best=0;accepted=[]
    if not panels or panels[0]['kind']!='initial' or panels[0]['y']!=condition['initial_y']:raise ValueError('the actual initial Reader900 must be practiced first')
    for index,panel in enumerate(panels):
        rows=panel['rows']
        if panel['ordinal']!=index or len(rows)!=4 or [r['init_state_id'] for r in rows]!=condition['practice_states']:
            raise ValueError('each practiced controller requires the complete registered four cases')
        for row in rows:
            _validate_episode(row,spec,condition['practice_states'],202610161)
            if row.get('behavior_y')!=panel['y'] or row.get('actual_parameter_ref')!=panel['parameter_ref']:
                raise ValueError('practice row must identify its actual complete controller and behavior code')
        if panel['successes']!=sum(r['success'] for r in rows) or panel.get('reused'):raise ValueError('new held initial/candidate practice cannot reuse historical H')
        if not index:continue
        if panel['kind']!='proposal' or panel['proposal_ordinal']!=index-1:raise ValueError('actual deterministic proposal order changed')
        improve=panel['successes']>panels[best]['successes']
        if bool(panel['accepted'])!=improve:raise ValueError('only strict four-case improvement may replace the incumbent')
        if improve:best=index;accepted.append(index)
    return best,accepted


def _trajectory(panels):
    best=-1;result=[]
    for panel in panels:
        score=panel['successes'];outcome='initial' if best<0 else ('strict_improvement' if score>best else 'tie' if score==best else 'worse')
        result.append(dict(ordinal=panel['ordinal'],proposal_ordinal=panel['proposal_ordinal'],kind=panel['kind'],
            accepted=panel['accepted'],outcome=outcome,successes=score,y=panel['y'],parameter_ref=panel['parameter_ref'],
            success_set=sorted(r['init_state_id'] for r in panel['rows'] if r['success']),steps=sum(r['environment_steps'] for r in panel['rows'])))
        best=max(best,score)
    return result


def _condition_costs(path,root=None):
    rows=[r for p in path['panels'] for r in p['rows']];known=sum(r['environment_steps'] for r in rows);charged=path['condition_steps']
    from .contract import reported_practice_cap
    cap=reported_practice_cap(root,path['id']) if root else 8192
    if not known<=8192 or not known<=charged<=cap:raise ValueError('actual unique practice or named physical recovery exceeded its bound')
    query_steps=sum(r['environment_steps'] for r in path['query_rows'])
    return dict(logical_practice_steps=charged,original_method_cap=8192,named_pending_physical_cap=cap,
        physical_excess_over8192=max(0,charged-8192),known_new_practice_steps=known,engineering_or_unknown_additional_charges=charged-known,
        logical_practice_episodes=len(rows),known_new_practice_episodes=len(rows),reused_old_practice_episodes=0,
        logical_query_steps=query_steps,new_query_steps=0 if path['query_reused'] else query_steps,
        new_query_episodes=0 if path['query_reused'] else 1,old_H_episodes=0,new_H_episodes=len(rows),
        known_new_H_archive_bytes=sum(r.get('history_bytes',0) for r in rows),
        normal_scientific_practice_failures=sum(not r['success'] for r in rows),normal_scientific_query_failures=sum(not r['success'] for r in path['query_rows']))


def _proposal_costs(path):
    receipts=[read_json(Path(d['receipt'])) for d in path.get('decisions',[])]
    measures=[m for r in receipts for m in r['feature_cache']]
    teaching=[m for m in measures if m['kind']=='teaching'];history=[m for m in measures if m['kind']=='history']
    return dict(cached_initial_teaching_compilations=1,psi_calls=len(receipts),
        psi_shared_residence_seconds=sum(r['seconds'] for r in receipts),H_events_read=sum(r['H_events'] for r in receipts),
        teaching_memory_reads=len(teaching),teaching_cache_hits=sum(m['hit'] for m in teaching),
        teaching_new_builds=sum(not m['hit'] for m in teaching),teaching_frame_uses=sum(m['frames'] for m in teaching),
        new_teaching_frame_encodings=sum(m['frames'] for m in teaching if not m['hit']),
        history_cache_reads=len(history),history_cache_hits=sum(m['hit'] for m in history),
        history_new_visual_builds=sum(not m['hit'] for m in history),history_observation_uses=sum(m['frames'] for m in history))


def _condition_summary(root,condition,spec):
    file=Path(root)/'conditions'/condition['id']/'path.json';path=read_json(file)
    if any(path.get(k)!=condition[k] for k in ('id','task','arm')) or not path.get('complete') or not path.get('query_complete'):
        raise ValueError('all400 fixed conditions must complete before analysis')
    best,accepted=_incumbent(path,condition,spec);panels=path['panels'];final=panels[best]
    if path['incumbent_index']!=best or path['proposal_cursor']!=len(panels)-1:raise ValueError('final incumbent/actual proposal cursor changed')
    unchanged=final['y']==condition['initial_y'] and path['selected_LoRA']==condition['initial_LoRA']
    if path['query_reused']!=unchanged or len(path['query_rows'])!=1:raise ValueError('exact initial controller reuses only its unique query; every new controller executes one final')
    row=path['query_rows'][0];_validate_episode(row,spec,[condition['final_init']],7)
    if row.get('LoRA')!=path['selected_LoRA']:raise ValueError('final must deploy the actual single selected complete LoRA')
    if any(row.get('scene_reference',{}).get(k)!=v for k,v in condition['scene_identity'].items()):raise ValueError('final registered scene identity changed')
    if unchanged and row!=initial_query_rows(condition)[0]:raise ValueError('unchanged final must preserve its exact original designated row')
    return dict(id=condition['id'],task=condition['task'],arm=ARM,final_init=condition['final_init'],teacher_demo=condition['teacher_demo'],
        path=str(file),initial_y=condition['initial_y'],final_y=final['y'],selected_LoRA=path['selected_LoRA'],initial_retained=unchanged,
        initial_practice_successes=panels[0]['successes'],final_practice_successes=final['successes'],practice_RGL=paired(panels[0]['rows'],final['rows']),
        accepted_updates=len(accepted),accepted_ordinals=accepted,proposals=path['proposal_cursor'],incumbent_index=best,
        unqueried_candidate_ordinals=[p['ordinal'] for p in panels[1:] if p['ordinal']!=best],
        first_effective_revision=accepted[0] if accepted else None,query_reused=unchanged,query_successes=int(row['success']),
        stop_reason=path['stop_reason'],trajectory=_trajectory(panels),decisions=path.get('decisions',[]),
        retained_raw_H_required=condition['retain_raw_H'],raw_H_lifecycle=path.get('raw_H_lifecycle'),
        costs=dict(_condition_costs(path,root),**_proposal_costs(path)),query_rows=[row])


def _historical_rows():
    old=read_json(VIDEO_ROOT/'analysis/results.json');rows={};sources={}
    for name,update in (('Reader900','900'),('Reader1800','1800')):
        rows[name]=old['checkpoints'][update]['raw_rows']['validation'];sources[name]=str(VIDEO_ROOT/'analysis/results.json')+'#checkpoints/'+update
    for name,file in old['validation_references'].items():
        value=read_json(Path(file));rows[name]=value['arms']['operator_only']['validation']['raw_rows'] if name.startswith('D') else value.get('raw_rows',value.get('rows'))
        sources[name]=file
    expected=dict(Reader900=117,Reader1800=111,MT153=153,G165=165,D72=175,D144=168)
    for name,count in expected.items():
        values=rows[name]
        if len(values)!=400 or sum(r['success'] for r in values)!=count:raise ValueError('fixed historical validation arm changed')
    return rows,sources


def _task_rows(raw,task):
    suites=('libero_spatial','libero_object','libero_goal','libero_10')
    return {a:[r for r in values if r.get('global_task_id',10*suites.index(r['suite'])+int(r['task_id']))==task] for a,values in raw.items()}


def _strata(conditions,raw,specs):
    groups=dict(initial_retained=lambda c:c['initial_retained'],changed_parameters=lambda c:not c['initial_retained'],
        initial_zero=lambda c:c['initial_practice_successes']==0,initial_nonzero=lambda c:c['initial_practice_successes']>0,
        selected_first_proposal=lambda c:c['incumbent_index']==1,selected_later_proposal=lambda c:c['incumbent_index']>1)
    groups.update({'stop_'+reason:(lambda c,reason=reason:c['stop_reason']==reason) for reason in sorted({c['stop_reason'] for c in conditions})})
    result={}
    for name,predicate in groups.items():
        keys={(c['task'],c['final_init']) for c in conditions if predicate(c)};task_rows=[]
        for task in TASKS:
            states={s for t,s in keys if t==task}
            if not states:continue
            arms={a:[r for r in values if r['init_state_id'] in states] for a,values in _task_rows(raw,task).items()}
            task_rows.append(_task_summary(task,specs[task],arms))
        result[name]=grouped(task_rows)
    return result


def analyze(root):
    root=Path(root);run=read_json(root/'run_contract.json')
    if run['schema']!=SCHEMA or run['phase']!=ARM or len(run['conditions'])!=400:raise ValueError('wrong fixed validation180 phase')
    specs={t['global_task_id']:t for t in run['environment']['tasks']}
    conditions=[_condition_summary(root,c,specs[c['task']]) for c in run['conditions']]
    raw,sources=_historical_rows();raw[ARM]=[r for c in conditions for r in c['query_rows']]
    if any(len(v)!=400 for v in raw.values()):raise ValueError('the seven-arm readout requires2800 logical rows')
    tasks=[_task_summary(t,specs[t],_task_rows(raw,t)) for t in TASKS];current=billing(root)
    cost_keys=conditions[0]['costs'];costs={k:sum(c['costs'][k] for c in conditions) for k in cost_keys}
    costs.update(cached_initial_video_compilations=400,new_shared_updates=0,actual_GPU_hours=current['GPU_hours_interval'],
        per_condition_GPU_hours=None,GPU_attribution='shared worker residence is not apportioned by environment steps')
    result=dict(complete=True,schema=SCHEMA,phase=ARM,revision_checkpoint=run['revision_checkpoint'],revision_update=180,
        revision_producer=run['revision_producer'],shared_updates=0,strict_single_checkpoint400=True,checkpoint_selection=False,
        arms=grouped(tasks),raw_rows=raw,conditions=conditions,strata=_strata(conditions,raw,specs),
        costs=costs,billing=current,interactions=summary(root),limits=run['limits'],historical_references=sources,
        legacy_MT_pairing='no original MT initial_proprio; registered scene path/bytes plus task/state/env/policy/actual first noise; proprio checked where available',
        actual_interaction_adaptation=True,held=True,held_offline_labels_read=False,Test=False,matched_video_or_H_ablation=False,
        scope='one fixed revision180 full400 validation compilation; no gradients, checkpoint selection, causal controls or automatic successor')
    folder=root/'analysis';folder.mkdir(exist_ok=True);write_json_atomic(folder/'results.json',result);write_json_atomic(root/'completion.json',result)
    lines=['# 固定功能修订180的完整验证迁移','',f"全部400/臂：{result['arms']['total']['successes']}",'',
        '|task|Reader900|Reader1800|MT153|G165|D72|D144|fixed psi180|','|---|---|---|---|---|---|---|---|']
    for row in tasks:lines.append('|'+str(row['task'])+'|'+'|'.join(str(len(row['success_sets'][a])) for a in OLD_ARMS+NEW_ARMS)+'|')
    lines+=['','全部21对R/G/L/churn/Jaccard、task/suite/breadth、初始未动/实际改参数、初始实践零/非零及首/后提案分组见results.json。',
        '400条教学均真实沿已有Reader900初始编译；缓存不等于零教学。新M按实际psi需要读取，完整输入不裁短。',
        '最终每condition仅一个final，未测中间候选不推oracle。自身final初态不进入本condition实践，各condition不共享H/代码/成绩。',
        '旧MT缺initial_proprio，沿登记场景path/bytes和实际noise配对；与Reader有proprio的行仍核验。',
        '8条init0全链raw H长期保留，其余raw按事前生命周期退役后不再声称完整H replay/exact adaptation resume。',
        '这是有真实交互的固定单点迁移检验，无共享held训练、视频/H因果资格或自动后继。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');return result
