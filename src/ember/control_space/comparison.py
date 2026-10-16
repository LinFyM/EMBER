"""Frozen initial controllers and label-side seven-arm incumbent readout."""
from pathlib import Path

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.reference.contract import TASKS,ROOT as REFERENCE_ROOT
from ember.reference.analysis import paired,pairing_valid
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .analysis import summarize_task,grouped

BASE=REFERENCE_ROOT.parent
ROOT=BASE/'incumbent_compilation_seed7'
FEEDBACK_ROOT=BASE/'feedback_proposals_seed7'
READOUT_ROOT=BASE/'feedback_readout_seed7'
VIDEO_ROOT=BASE/'video_initial_seed7'
INITIAL_READER=VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
BASIS=BASE/'control_linear64_seed7/basis/state.pt'
SCHEMA='ember_source_incumbent_compilation_r8_seed7_v1'
LIMITS=dict(physical_submissions=5536,control_steps=658087,dummy_steps=55360,total_steps=662567,
    engineering_extra_attempts=16,added_peak_GiB=64,shared_optimizer_updates=0,conditions=72,logical_query_episodes=288,
    condition_practice_steps=8192)
NEW_ARMS=('video','no_video')
OLD_ARMS=('source','reference','Reader900','first_best','full')
STATES=(0,4,8,12)


def condition_id(task,arm):
    if arm not in NEW_ARMS or task not in TASKS:raise ValueError('only the fixed36 task and two arms are authorized')
    return f'{arm}_task_{task:04d}_v0'


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


def _source_initial(task,metadata):
    import torch
    rows=[];files=[]
    for state in metadata['practice_states']:
        file=_existing(FEEDBACK_ROOT/'episodes'/f'task_{task["global_task_id"]:04d}_v0_practice_000'/f'state_{state:03d}.json')
        row=read_json(file);_validate_episode(row,task,metadata['practice_states'],202610161)
        if row.get('role')!='practice' or row.get('panel_ordinal')!=0 or torch.as_tensor(row['behavior_y']).shape!=(64,) or torch.as_tensor(row['behavior_y']).count_nonzero():
            raise ValueError('no-video may only reuse the exact original source-zero practice panel')
        _existing(row['history_path']);rows.append(row);files.append(str(file))
    if len(rows)!=4 or len({r['init_state_id'] for r in rows})!=4:raise ValueError('source initial panel needs all four fixed cases')
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
    """Adaptation manifest: legal metadata, source practice, and video initial code only."""
    root=Path(root);file=root/'run_contract.json'
    if file.exists():
        run=read_json(file)
        if run['schema']!=SCHEMA or run['phase']!='incumbent_compilation' or run['limits']!=LIMITS or len(run['conditions'])!=72:
            raise ValueError('existing incumbent contract is incompatible')
        return run
    environment=read_json(FEEDBACK_ROOT/'run_contract.json')['environment'];tasks={t['global_task_id']:t for t in environment['tasks']}
    metadata=read_json(FEEDBACK_ROOT/'conditions.json');units=_units();conditions=[];source=[]
    if set(tasks)!=set(TASKS) or set(map(int,metadata))!=set(TASKS):raise ValueError('only the original non-held36 are allowed')
    for task in TASKS:
        item=metadata[str(task)]
        if item['query_states']!=list(STATES) or item['query_noise_root']!=202610151 or item['practice_noise_root']!=202610161:
            raise ValueError('practice/query panel authority changed')
        common=dict(task=task,practice_states=item['practice_states'],practice_noise_root=202610161,
            query_states=list(STATES),query_noise_root=202610151)
        rows,paths=_source_initial(tasks[task],item);source.extend(rows)
        y,lora,provenance=_video_initial(task,item['teacher_demo'],units)
        conditions.append(dict(common,id=condition_id(task,'video'),arm='video',teacher_demo=item['teacher_demo'],
            initial_y=y,initial_LoRA=lora,initial_controller_provenance=provenance))
        conditions.append(dict(common,id=condition_id(task,'no_video'),arm='no_video',initial_y=[0.]*64,
            source_initial_rows=rows,source_initial_row_paths=paths))
    reused=dict(episodes=len(source),steps=sum(r['environment_steps'] for r in source),successes=sum(r['success'] for r in source),
        source_early_stops=sum(sum(r['success'] for r in c['source_initial_rows'])==4 for c in conditions if c['arm']=='no_video'),
        control=sum(r['environment_steps']-10 for r in source),dummy=10*len(source))
    if reused!=dict(episodes=144,steps=40377,successes=52,source_early_stops=11,control=38937,dummy=1440):
        raise ValueError('the mandatory original source practice reuse changed')
    run=dict(schema=SCHEMA,phase='incumbent_compilation',environment=environment,conditions=conditions,limits=LIMITS,
        frozen_basis=str(BASIS),initial_reader_checkpoint=str(INITIAL_READER),shared_updates=0,shared_gradients=False,
        sigma=.25,seed=7,condition_practice_steps=8192,source_practice_reuse=reused,
        new_Reader_forwards=0,new_teaching_encodings=0,held=False,Test=False,
        adaptation_inputs='only conditions.json metadata/source panel0 rows/H and video arm cached deterministic Reader900; no old query/ref/full labels',
        teaching_reuse='video arm uses one previously actually compiled legal video per task; cached computation is not zero teaching cost')
    root.mkdir(parents=True,exist_ok=True);write_json_atomic(file,run);return run


def initial_query_rows(task,arm):
    """LABEL SIDE ONLY: caller must freeze the condition before invoking this."""
    condition_id(task,arm);which='Reader900' if arm=='video' else 'source'
    manifest=read_json(READOUT_ROOT/'manifest.json')
    if manifest['schema']!='ember_source_feedback_readout_r8_seed7_v1':raise ValueError('wrong explicit historical query phase')
    selected=[e for e in manifest['mappings'] if e['arm']==which and e['task']==task]
    if len(selected)!=4 or {e['state_id'] for e in selected}!=set(STATES):raise ValueError('initial reuse needs the whole fixed query panel')
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
        raise ValueError('all72 condition paths must actually finish before analysis')
    if not torch.allclose(torch.tensor(path['initial_y']),torch.tensor(condition['initial_y']),atol=1e-6,rtol=1e-6):
        raise ValueError('saved path changed its registered initial code')
    incumbent,accepted=_incumbent(path['panels'],condition,task)
    if path['incumbent_index']!=incumbent or path['proposal_cursor']!=len(path['panels'])-1:
        raise ValueError('final incumbent or proposal cursor differs from complete actual panels')
    if path['query_reused']!=(not accepted):raise ValueError('only unchanged initial controllers may reuse their registered queries')
    rows=path['query_rows']
    if len(rows)!=4 or {r['init_state_id'] for r in rows}!=set(STATES):raise ValueError('each frozen condition requires four independent query cases')
    for row in rows:_validate_episode(row,task,STATES,202610151)
    if path['query_reused']:
        previous=initial_query_rows(condition['task'],condition['arm'])
        if not pairing_valid(previous,rows) or {r['init_state_id']:r['success'] for r in previous}!={r['init_state_id']:r['success'] for r in rows}:
            raise ValueError('unchanged controller must preserve the exact predesignated original four outcomes')
    return path_file,path,accepted


def _condition_summary(root,condition,task):
    import torch
    file,path,accepted=_completed_path(root,condition,task);panels=path['panels'];initial=panels[0];final=panels[path['incumbent_index']]
    known_steps=sum(r['environment_steps'] for p in panels for r in p['rows']);charged=path['condition_steps']
    if not known_steps<=charged<=8192:raise ValueError('known and engineering/unknown practice costs must respect the condition8192 bound')
    reuse=condition['arm']=='no_video'
    if bool(initial['reused'])!=reuse or any(p.get('reused') for p in panels[1:]):raise ValueError('only the registered no-video source initial panel is old work')
    reused_steps=sum(r['environment_steps'] for r in initial['rows']) if reuse else 0
    trajectory=[dict(ordinal=p['ordinal'],proposal_ordinal=p['proposal_ordinal'],kind=p['kind'],accepted=p['accepted'],
        successes=p['successes'],success_set=sorted(r['init_state_id'] for r in p['rows'] if r['success']),
        steps=sum(r['environment_steps'] for r in p['rows']),reused=p['reused']) for p in panels]
    return dict(id=condition['id'],task=condition['task'],arm=condition['arm'],path=str(file),
        initial_y=condition['initial_y'],final_y=final['y'],initial_LoRA=condition.get('initial_LoRA'),selected_LoRA=path['selected_LoRA'],
        initial_practice_successes=initial['successes'],final_practice_successes=final['successes'],
        practice_RGL=paired(initial['rows'],final['rows']),accepted_updates=len(accepted),accepted_ordinals=accepted,
        proposals=path['proposal_cursor'],incumbent_index=path['incumbent_index'],initial_retained=not accepted,
        query_reused=path['query_reused'],query_successes=sum(r['success'] for r in path['query_rows']),
        stop_reason=path['stop_reason'],trajectory=trajectory,
        parameter_displacement_L2=float(torch.linalg.vector_norm(torch.tensor(final['y'])-torch.tensor(condition['initial_y']))),
        costs=dict(logical_practice_steps=charged,reused_old_practice_steps=reused_steps,
            known_new_practice_steps=known_steps-reused_steps,new_charged_practice_steps=charged-reused_steps,
            engineering_or_unknown_additional_charges=charged-known_steps,
            logical_practice_episodes=4*len(panels),reused_old_practice_episodes=4 if reuse else 0,
            known_new_practice_episodes=4*len(panels)-(4 if reuse else 0),
            logical_query_steps=sum(r['environment_steps'] for r in path['query_rows']),
            new_query_steps=0 if path['query_reused'] else sum(r['environment_steps'] for r in path['query_rows']),
            new_query_episodes=0 if path['query_reused'] else 4),query_rows=path['query_rows'])


def _arm_costs(conditions,total_gpu_hours):
    keys=next(iter(conditions))['costs'].keys()
    return dict(**{k:sum(c['costs'][k] for c in conditions) for k in keys},
        cached_video_compilations=sum(c['arm']=='video' for c in conditions),new_Reader_forwards=0,new_teaching_encodings=0,
        actual_GPU_hours=None,GPU_hours_attribution='shared/unattributed: dynamic mixed-arm workers and shared loading; no step-proportional pseudo measurement',
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
    if run['schema']!=SCHEMA or run['phase']!='incumbent_compilation' or len(run['conditions'])!=72:raise ValueError('wrong incumbent phase/schema')
    specs={t['global_task_id']:t for t in run['environment']['tasks']}
    conditions=[_condition_summary(root,c,specs[c['task']]) for c in run['conditions']]
    old=read_json(READOUT_ROOT/'completion.json');raw={a:old['raw_rows'][a] for a in OLD_ARMS}
    raw.update({a:[r for c in conditions if c['arm']==a for r in c['query_rows']] for a in NEW_ARMS})
    if any(len(v)!=144 for v in raw.values()):raise ValueError('the seven-arm readout must have1008 logical rows')
    identities={r['task']:r['reference_identity'] for r in old['controllers']};tasks=[]
    for task in TASKS:
        arms={a:[r for r in rows if r['global_task_id']==task] for a,rows in raw.items()}
        tasks.append(summarize_task(task,specs[task],arms,identity=identities[task]))
    current=billing(root);gpu=current['GPU_hours_interval'][1];strata=_strata(conditions,tasks)
    result=dict(complete=True,schema=SCHEMA,phase='incumbent_compilation',shared_updates=0,conditions=conditions,
        arms=grouped(tasks),raw_rows=raw,strata=strata,
        costs={a:_arm_costs([c for c in conditions if c['arm']==a],gpu) for a in NEW_ARMS},
        billing=current,interactions=summary(root),limits=run['limits'],
        historical_five_arm_costs=dict(ROOT=str(READOUT_ROOT),new_readout_billing=old['new_readout_billing'],
            historical_feedback=old['historical_feedback'],included_in_current_billing=False),
        held=False,Test=False,checkpoint_selection=False,
        scope='two fixed shared-model task-local incumbent methods; actual interaction, not shared training or semantic H learning; finite non-held panel')
    folder=root/'analysis';folder.mkdir(exist_ok=True);write_json_atomic(folder/'results.json',result);write_json_atomic(root/'completion.json',result)
    lines=['# 视频初始控制保留与任务内参数改进','',f"全部144/臂：{result['arms']['total']['successes']}",'',
        '|task|source|reference|Reader900|first-best|旧full|video|no-video|','|---|---|---|---|---|---|---|---|']
    for row in tasks:lines.append('|'+str(row['task'])+'|'+'|'.join(str(len(row['success_sets'][a])) for a in OLD_ARMS+NEW_ARMS)+'|')
    lines+=['','全部21对R/G/L/churn/交并/Jaccard、suite/task/target-support/identity与参考新增70保持见results.json。',
        '逐condition初末实践、接受轨迹/代码、早停、全部成本和query复用分开保留。旧source144/40377步仍占no-video条件预算，物理费留在旧批。',
        'video缓存复用此前真实编译的一条合法教学，不将教学成本记零；当前没有新Reader/teacher编码。',
        '原始H完整保存，当前更新未作语义读取。混臂worker的GPUh全局实记、arm归属留未知，不由steps比例伪称实测。',
        '旧五臂及其全部费用独立保留；无held/Test、共享更新、checkpoint选择或自动后继。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');return result
