"""Frozen video initialization and label-side functional-feedback readout."""
from pathlib import Path

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.reference.contract import TASKS,ROOT as REFERENCE_ROOT
from ember.reference.analysis import paired,pairing_valid
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .analysis import summarize_task,grouped

BASE=REFERENCE_ROOT.parent
HISTORY_ROOT=BASE/'incumbent_compilation_seed7'
ROOT=BASE/'functional_feedback_seed7'
FEEDBACK_ROOT=BASE/'feedback_proposals_seed7'
READOUT_ROOT=BASE/'feedback_readout_seed7'
VIDEO_ROOT=BASE/'video_initial_seed7'
INITIAL_READER=VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
BASIS=BASE/'control_linear64_seed7/basis/state.pt'
SCHEMA='ember_source_functional_feedback_r8_seed7_v1'
LIMITS=dict(physical_submissions=2704,control_steps=318905,dummy_steps=27040,total_steps=320505,
    engineering_extra_attempts=16,added_peak_GiB=64,shared_optimizer_updates=180,conditions=36,logical_query_episodes=144,
    condition_practice_steps=8192,frozen_H_cache_GiB=16)
ARM='functional_feedback'
NEW_ARMS=(ARM,)
OLD_ARMS=('source','reference','Reader900','first_best','full','video','no_video')
STATES=(0,4,8,12)


def condition_id(task,arm=ARM):
    if arm!=ARM or task not in TASKS:raise ValueError('only the fixed36 functional-feedback conditions are authorized')
    return f'task_{task:04d}_v0'


def _existing(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'fixed source asset missing: {path}')
    return path


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
    if row.get('policy_noise_seeds')!=stream or row.get('first_noise_seed')!=stream[0]:raise ValueError('absolute policy noise changed')


def _initial_practice(task,condition):
    panel=read_json(HISTORY_ROOT/'conditions'/condition['id']/'path.json')['panels'][0]
    if panel['kind']!='initial' or panel['ordinal']!=0 or panel['y']!=condition['initial_y']:
        raise ValueError('only the actually practiced video initial panel may enter new runtime H')
    rows=panel['rows'];files=[]
    if len(rows)!=4 or [r['init_state_id'] for r in rows]!=condition['practice_states']:
        raise ValueError('initial reuse needs the whole actual four-state panel in original order')
    for row in rows:
        _validate_episode(row,task,condition['practice_states'],202610161)
        if row['role']!='practice' or row['panel_ordinal']!=0 or row['behavior_y']!=condition['initial_y']:
            raise ValueError('initial behavior must be the actual frozen Reader900 code')
        _existing(row['history_path'])
        file=_existing(HISTORY_ROOT/'episodes'/panel['identity']/f'state_{row["init_state_id"]:03d}.json')
        if read_json(file)!=row:raise ValueError('initial practice row must match its original saved receipt')
        files.append(str(file))
    return rows,files


def _units():
    import torch
    saved=torch.load(INITIAL_READER,map_location='cpu',weights_only=False,mmap=True)
    if saved['next_update']!=900 or saved['phase']!='video_initial' or saved['frozen_basis']!=str(BASIS):
        raise ValueError('fixed units must identify the actual Reader900 and unique basis')
    units=saved['code_units'].float()
    if units.shape!=(64,) or not bool(units.isfinite().all()) or bool((units<=0).any()):raise ValueError('code units must be finite positive64')
    return units


def _video_initial(task,teacher,units):
    import torch
    file=_existing(VIDEO_ROOT/'generated/update_00000900/seen'/f'task_{task:04d}'/'state_000.safetensors')
    value=read_json(file.with_suffix('.json'))
    expected=dict(checkpoint=str(INITIAL_READER),basis_checkpoint=str(BASIS),path=str(file),task=task,
        teacher_demo=teacher,init_state_id=0,reader_update=900,phase='video_initial',role='seen',rank=8,targets=38)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('cached video controller provenance changed')
    z=torch.as_tensor(value['code'],dtype=torch.float32);y=z/units
    if y.shape!=(64,) or not bool(y.isfinite().all()):raise ValueError('cached video code is not a finite64-vector')
    return y.tolist(),str(file),str(file.with_suffix('.json'))


def prepare(root):
    """No old final parameters, later runtime H, query, or reference enters adaptation."""
    root=Path(root);file=root/'run_contract.json'
    if file.exists():
        run=read_json(file)
        if run['schema']!=SCHEMA or run['phase']!='functional_feedback' or run['limits']!=LIMITS or len(run['conditions'])!=36:
            raise ValueError('existing functional-feedback contract is incompatible')
        return run
    old=read_json(HISTORY_ROOT/'run_contract.json');environment=old['environment']
    tasks={t['global_task_id']:t for t in environment['tasks']};units=_units();conditions=[];initial=[]
    starts={c['task']:c for c in old['conditions'] if c['arm']=='video'}
    if set(tasks)!=set(TASKS) or set(starts)!=set(TASKS):raise ValueError('only the original non-held36 are allowed')
    for task in TASKS:
        item=starts[task]
        if item['query_states']!=list(STATES) or item['query_noise_root']!=202610151 or item['practice_noise_root']!=202610161:
            raise ValueError('practice/query panel authority changed')
        y,lora,provenance=_video_initial(task,item['teacher_demo'],units)
        if y!=item['initial_y'] or lora!=item['initial_LoRA']:raise ValueError('actual cached initial Reader900 controller changed')
        rows,paths=_initial_practice(tasks[task],item);initial.extend(rows)
        conditions.append(dict(id=condition_id(task),task=task,arm=ARM,teacher_demo=item['teacher_demo'],
            practice_states=item['practice_states'],practice_noise_root=202610161,query_states=list(STATES),query_noise_root=202610151,
            initial_y=y,initial_LoRA=lora,initial_controller_provenance=provenance,initial_rows=rows,initial_row_paths=paths))
    reused=dict(episodes=len(initial),steps=sum(r['environment_steps'] for r in initial),successes=sum(r['success'] for r in initial),
        early_stops=sum(sum(r['success'] for r in c['initial_rows'])==4 for c in conditions),
        control=sum(r['environment_steps']-10 for r in initial),dummy=10*len(initial))
    if reused!=dict(episodes=144,steps=35207,successes=83,early_stops=10,control=33767,dummy=1440):
        raise ValueError('mandatory Reader900 initial practice reuse changed')
    run=dict(schema=SCHEMA,phase='functional_feedback',environment=environment,conditions=conditions,limits=LIMITS,
        frozen_basis=str(BASIS),initial_reader_checkpoint=str(INITIAL_READER),shared_updates=180,shared_gradients=True,
        seed=7,condition_practice_steps=8192,initial_practice_reuse=reused,training_history_root=str(HISTORY_ROOT),
        optimizer=dict(lr=1e-4,betas=[.9,.95],eps=1e-8,weight_decay=1e-4,clip=1.,schedule='constant'),
        logical=dict(tasks=4,FM_queries_per_task=28,updates=180,experience_conditions=720,FM_queries=20160),
        checkpoints=dict(recovery_only=[90],fixed_terminal=180),held=False,Test=False,
        adaptation_inputs='cached Reader900 initial code/LoRA, legal teaching and actually observed runtime H; old later H is training-only',
        teaching_reuse='cached computation is not zero teaching; frozen full Reader memory is consumed by the learned H revision')
    root.mkdir(parents=True,exist_ok=True);write_json_atomic(file,run);return run


def initial_query_rows(task,arm=ARM):
    """LABEL SIDE ONLY: caller freezes the actual condition before invoking this."""
    condition_id(task,arm);manifest=read_json(READOUT_ROOT/'manifest.json')
    if manifest['schema']!='ember_source_feedback_readout_r8_seed7_v1':raise ValueError('wrong historical query phase')
    selected=[e for e in manifest['mappings'] if e['arm']=='Reader900' and e['task']==task]
    if len(selected)!=4 or {e['state_id'] for e in selected}!=set(STATES):raise ValueError('reuse requires the designated whole Reader900 query panel')
    return [read_json(_existing(next(e['row_path'] for e in selected if e['state_id']==state))) for state in STATES]


def _incumbent(panels,condition,task):
    import torch
    if not panels or panels[0]['kind']!='initial':raise ValueError('initial controller must be actually practiced first')
    if not torch.allclose(torch.tensor(panels[0]['y']),torch.tensor(condition['initial_y']),atol=1e-6,rtol=1e-6):
        raise ValueError('the initial practiced code differs from the registered source/video start')
    incumbent=0;accepted=[]
    for index,panel in enumerate(panels):
        rows=panel['rows']
        if panel['ordinal']!=index or len(rows)!=4 or {r['init_state_id'] for r in rows}!=set(condition['practice_states']):
            raise ValueError('only complete four-case panels may change incumbent')
        for row in rows:_validate_episode(row,task,condition['practice_states'],202610161)
        if panel['successes']!=sum(r['success'] for r in rows):raise ValueError('recorded practice success count changed')
        if not index:continue
        if panel['kind']!='proposal' or panel['proposal_ordinal']!=index-1:raise ValueError('actual proposal cursor changed')
        improve=panel['successes']>panels[incumbent]['successes']
        if bool(panel['accepted'])!=improve:raise ValueError('only strict improvement may replace incumbent; every tie retains it')
        if improve:incumbent=index;accepted.append(index)
    return incumbent,accepted


def _completed_path(root,condition,task):
    import torch
    path_file=Path(root)/'conditions'/condition['id']/'path.json';path=read_json(path_file)
    if any(path.get(k)!=condition[k] for k in ('id','task','arm')) or not path.get('complete') or not path.get('query_complete'):
        raise ValueError('all36 conditions must actually finish before analysis')
    if not torch.allclose(torch.tensor(path['initial_y']),torch.tensor(condition['initial_y']),atol=1e-6,rtol=1e-6):
        raise ValueError('saved path changed its registered initial code')
    incumbent,accepted=_incumbent(path['panels'],condition,task)
    if path['incumbent_index']!=incumbent or path['proposal_cursor']!=len(path['panels'])-1:
        raise ValueError('final incumbent or proposal cursor differs from complete actual panels')
    if path['query_reused']!=(path['panels'][incumbent]['y']==condition['initial_y']):raise ValueError('only unchanged initial controllers may reuse their registered queries')
    rows=path['query_rows']
    if len(rows)!=4 or {r['init_state_id'] for r in rows}!=set(STATES):raise ValueError('each frozen condition requires four independent query cases')
    for row in rows:_validate_episode(row,task,STATES,202610151)
    if path['query_reused']:
        previous=initial_query_rows(condition['task'],condition['arm'])
        if not pairing_valid(previous,rows) or {r['init_state_id']:r['success'] for r in previous}!={r['init_state_id']:r['success'] for r in rows}:
            raise ValueError('unchanged controller must preserve the exact predesignated original four outcomes')
    return path_file,path,accepted


def _trajectory(panels):
    best=-1;result=[]
    for panel in panels:
        score=panel['successes'];outcome='initial' if best<0 else ('strict_improvement' if score>best else 'tie' if score==best else 'worse')
        result.append(dict(ordinal=panel['ordinal'],proposal_ordinal=panel['proposal_ordinal'],kind=panel['kind'],
            accepted=panel['accepted'],outcome=outcome,successes=score,y=panel['y'],parameter_ref=panel['parameter_ref'],
            success_set=sorted(r['init_state_id'] for r in panel['rows'] if r['success']),
            steps=sum(r['environment_steps'] for r in panel['rows']),reused=panel['reused']))
        best=max(best,score)
    return result


def _condition_costs(path):
    panels=path['panels'];rows=[r for p in panels for r in p['rows']];new=[r for p in panels[1:] for r in p['rows']]
    known=sum(r['environment_steps'] for r in rows);reused=sum(r['environment_steps'] for r in panels[0]['rows']);charged=path['condition_steps']
    return dict(logical_practice_steps=charged,reused_old_practice_steps=reused,
        known_new_practice_steps=known-reused,new_charged_practice_steps=charged-reused,
        engineering_or_unknown_additional_charges=charged-known,
        logical_practice_episodes=len(rows),reused_old_practice_episodes=4,known_new_practice_episodes=len(new),
        logical_query_steps=sum(r['environment_steps'] for r in path['query_rows']),
        new_query_steps=0 if path['query_reused'] else sum(r['environment_steps'] for r in path['query_rows']),
        new_query_episodes=0 if path['query_reused'] else 4,old_H_episodes=4,new_H_episodes=len(new),
        known_new_H_archive_bytes=sum(r.get('history_bytes',0) for r in new),
        normal_scientific_practice_failures=sum(not r['success'] for r in rows),
        new_normal_scientific_practice_failures=sum(not r['success'] for r in new),
        normal_scientific_query_failures=sum(not r['success'] for r in path['query_rows']))


def _condition_summary(root,condition,task):
    import torch
    file,path,accepted=_completed_path(root,condition,task);panels=path['panels'];initial=panels[0];final=panels[path['incumbent_index']]
    known_steps=sum(r['environment_steps'] for p in panels for r in p['rows']);charged=path['condition_steps']
    if not known_steps<=charged<=8192:raise ValueError('known and engineering/unknown practice costs must respect the condition8192 bound')
    if not initial['reused'] or any(p.get('reused') for p in panels[1:]):raise ValueError('only the registered video initial practice is reused')
    return dict(id=condition['id'],task=condition['task'],arm=condition['arm'],path=str(file),
        initial_y=condition['initial_y'],final_y=final['y'],initial_LoRA=condition.get('initial_LoRA'),selected_LoRA=path['selected_LoRA'],
        initial_practice_successes=initial['successes'],final_practice_successes=final['successes'],
        practice_RGL=paired(initial['rows'],final['rows']),accepted_updates=len(accepted),accepted_ordinals=accepted,
        proposals=path['proposal_cursor'],incumbent_index=path['incumbent_index'],initial_retained=final['y']==condition['initial_y'],
        unqueried_candidate_ordinals=[p['ordinal'] for p in panels[1:] if p['ordinal']!=path['incumbent_index']],
        first_effective_revision=accepted[0] if accepted else None,
        query_reused=path['query_reused'],query_successes=sum(r['success'] for r in path['query_rows']),
        stop_reason=path['stop_reason'],trajectory=_trajectory(panels),
        repeated_code_decision=path.get('repeated_code'),history_paths=[r['history_path'] for p in panels for r in p['rows']],
        parameter_displacement_L2=float(torch.linalg.vector_norm(torch.tensor(final['y'])-torch.tensor(condition['initial_y']))),
        costs=_condition_costs(path),query_rows=path['query_rows'])


def _arm_costs(conditions,total_gpu_hours):
    keys=next(iter(conditions))['costs'].keys()
    return dict(**{k:sum(c['costs'][k] for c in conditions) for k in keys},
        cached_initial_video_compilations=36,training_experience_conditions=720,training_FM_queries=20160,
        old_training_panels=314,old_training_logical_practice_episodes=1256,
        actual_GPU_hours=None,GPU_hours_attribution='training/compiler/readout jobs are shared; stage receipts retain real costs without step-proportional attribution',
        conservative_nonadditive_GPU_hours_interval=[0,total_gpu_hours],
        initial_retained=sum(c['initial_retained'] for c in conditions),initial_four_success_stops=sum(c['proposals']==0 and c['initial_practice_successes']==4 for c in conditions),
        accepted_updates=sum(c['accepted_updates'] for c in conditions),proposals=sum(c['proposals'] for c in conditions))


def _strata(conditions,tasks):
    strata={}
    for arm in NEW_ARMS:
        members=[c for c in conditions if c['arm']==arm]
        ids=lambda predicate:{c['task'] for c in members if predicate(c)}
        groups=dict(initial_retained=ids(lambda c:c['initial_retained']),accepted_new=ids(lambda c:not c['initial_retained']),
            initial_four_success=ids(lambda c:c['proposals']==0 and c['initial_practice_successes']==4),
            no_accept_after_proposals=ids(lambda c:c['proposals']>0 and not c['accepted_updates']))
        strata[arm]={name:grouped([r for r in tasks if r['task'] in selected]) for name,selected in groups.items()}
        strata[arm]['stop_reasons']={reason:grouped([r for r in tasks if r['task'] in ids(lambda c:c['stop_reason']==reason)]) for reason in sorted({c['stop_reason'] for c in members})}
    return strata


def analyze(root):
    root=Path(root);run=read_json(root/'run_contract.json')
    if run['schema']!=SCHEMA or run['phase']!='functional_feedback' or len(run['conditions'])!=36:raise ValueError('wrong functional-feedback phase/schema')
    specs={t['global_task_id']:t for t in run['environment']['tasks']}
    conditions=[_condition_summary(root,c,specs[c['task']]) for c in run['conditions']]
    old=read_json(HISTORY_ROOT/'completion.json');raw={a:old['raw_rows'][a] for a in OLD_ARMS}
    raw[ARM]=[r for c in conditions for r in c['query_rows']]
    if any(len(v)!=144 for v in raw.values()):raise ValueError('the eight-arm readout requires1152 logical rows')
    identities={r['task']:r['reference_identity'] for r in read_json(READOUT_ROOT/'completion.json')['controllers']};tasks=[]
    for task in TASKS:
        arms={a:[r for r in rows if r['global_task_id']==task] for a,rows in raw.items()}
        tasks.append(summarize_task(task,specs[task],arms,identity=identities[task]))
    current=billing(root);gpu=current['GPU_hours_interval'][1];strata=_strata(conditions,tasks)
    training_file=root/'training_completion.json';training=read_json(training_file)
    if not training['complete'] or training['next_update']!=180:raise ValueError('only the fixed completed180-update learner may produce the terminal readout')
    stream=training['stream']
    if stream['next_update']!=180 or any(stream['new_visits'].get(str(t),stream['new_visits'].get(t))!=20 for t in TASKS):
        raise ValueError('completed shared training must retain all720 actual experience conditions and their cursors')
    result=dict(complete=True,schema=SCHEMA,phase='functional_feedback',shared_updates=180,conditions=conditions,
        arms=grouped(tasks),raw_rows=raw,strata=strata,costs={ARM:_arm_costs(conditions,gpu)},
        training=dict(receipt=str(training_file),**training),billing=current,interactions=summary(root),limits=run['limits'],
        historical_seven_arm_costs=dict(ROOT=str(HISTORY_ROOT),billing=old['billing'],
            earlier_five_arm_costs=old['historical_five_arm_costs'],included_in_current_billing=False),
        held=False,Test=False,checkpoint_selection=False,matched_video_or_history_ablation=False,
        scope='non-held training-task/seen-initial-H complete learned functional revision; old no-video is historical, not a matched psi ablation')
    folder=root/'analysis';folder.mkdir(exist_ok=True);write_json_atomic(folder/'results.json',result);write_json_atomic(root/'completion.json',result)
    lines=['# 冻结控制空间中的经历条件功能修订','',f"全部144/臂：{result['arms']['total']['successes']}",'',
        '|task|source|reference|Reader900|first-best|旧full|旧video|旧no-video|functional feedback|',
        '|---|---|---|---|---|---|---|---|---|']
    for row in tasks:lines.append('|'+str(row['task'])+'|'+'|'.join(str(len(row['success_sets'][a])) for a in OLD_ARMS+NEW_ARMS)+'|')
    lines+=['','全部28对R/G/L/churn/交并/Jaccard、suite/task/target-support/identity与参考70新增保持见results.json。',
        '逐condition初末实践、严格接受/平局/失败、首次有效修订、停止、未query候选和全部不利case保留；未query不声称oracle优势。',
        '144条旧Reader900实践/H、35207步仍逐condition计入8192；旧物理费用独立保留。新参数保留才真实新query。',
        '共享训练720经历条件/20160跨episodeFM query；自身动作只作H输入。教学缓存不是零教学，旧H/新H与训练/编译/query费用分开登记。',
        '本批是训练task/已见初始H完整流程诊断，无held/Test或视频/H因果资格。旧full变化theta和旧no-video59均只是历史方法参照。',
        '固定终点180，不以loss/practice挑点；不自动360、调参或扩量。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');return result
