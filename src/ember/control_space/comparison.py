"""Immutable five-arm144 diagnostic using only already chosen controllers."""
from pathlib import Path
import time

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.accounting import summary
from ember.reference.analysis import pairing_valid
from ember.proposal_writer.analysis import billing
from .analysis import summarize_task,grouped
from .contract import TASKS,REFERENCE_ROOT,PARENT_ROOT,VIDEO_ROOT,BASIS,INITIAL_READER,condition_root

ROOT=REFERENCE_ROOT.parent/'feedback_readout_seed7'
FEEDBACK_ROOT=REFERENCE_ROOT.parent/'feedback_proposals_seed7'
SCHEMA='ember_source_feedback_readout_r8_seed7_v1'
LIMITS=dict(physical_submissions=140,control_steps=51200,dummy_steps=1400,total_steps=52600,
    engineering_extra_attempts=16,added_peak_GiB=2,optimizer_updates=0,new_query_episodes=124)
STATES=(0,4,8,12)
ARMS=('source','reference','Reader900','first_best','full')
CHANGED=(1,4,5,12,14,19,43)
NEW_FIRST=(4,12,14,43)
SOURCE_FIRST=(1,5,19)


def _existing(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'fixed comparison asset missing: {path}')
    return path


def _paths():
    result={task:read_json(condition_root(FEEDBACK_ROOT,task)/'path.json') for task in TASKS}
    for task,path in result.items():
        if not path.get('complete') or not path.get('query_complete') or len(path['query_rows'])!=4:
            raise ValueError('all36 complete original paths and independent query are required')
        panels=path['panels']
        if not panels or any(len(p['rows'])!=4 or p['successes']!=sum(r['success'] for r in p['rows']) for p in panels):
            raise ValueError('first-best selection cannot consume incomplete practice panels')
        best=max(range(len(panels)),key=lambda i:(panels[i]['successes'],-i))
        if panels[best]['identity']!=path['best_panel']:raise ValueError('original full best changed its fixed earliest-tie rule')
        first=max(range(min(2,len(panels))),key=lambda i:(panels[i]['successes'],-i))
        path['first_panel_index']=first;path['full_panel_index']=best
    changed=tuple(sorted(t for t in TASKS if result[t]['first_panel_index']!=result[t]['full_panel_index']))
    if changed!=CHANGED or tuple(t for t in changed if result[t]['first_panel_index']==1)!=NEW_FIRST:
        raise ValueError('the registered seven changes/four new controllers changed')
    return result


def _row_file(root,arm,task,state):
    return Path(root)/'episodes'/f'{arm}_task_{task:04d}'/f'state_{state:03d}.json'


def _original_row(task,state):
    return FEEDBACK_ROOT/'episodes'/f'task_{task:04d}_v0_query'/f'state_{state:03d}.json'


def _entry(arm,task,state,teacher,path,*,new=False,lora=None,origin=None):
    original={} if new else read_json(path)
    return dict(arm=arm,task=task,state_id=state,teacher_demo=teacher,row_path=str(path),new_episode=new,
        LoRA=str(lora) if lora is not None else original.get('LoRA'),reuse_origin=origin,
        original_parameter_ref=original.get('actual_parameter_ref'),original_success=original.get('success'))


def _assets(root,task,path,selected):
    source=REFERENCE_ROOT/'selection'/f'task_{task:04d}'/'update_00000000'
    reader=_existing(VIDEO_ROOT/'generated/update_00000900/seen'/f'task_{task:04d}'/'state_000.safetensors')
    metadata=read_json(reader.with_suffix('.json'))
    if metadata['teacher_demo']!=path['teacher_demo'] or metadata['reader_update']!=900 or metadata['path']!=str(reader):
        raise ValueError('Reader900 controller must use exactly the original v0 teaching')
    first,full=path['first_panel_index'],path['full_panel_index']
    lora=_existing(path['selected_LoRA']) if first==full else (root/'generated'/f'first_best_task_{task:04d}.safetensors' if first else None)
    point=None
    if task in NEW_FIRST:
        y=path['panels'][1]['y']
        if y!=path['proposals'][0]['y']:raise ValueError('first controller must use the actual sample, never mu')
        point=dict(task=task,y=y,path=str(lora),source_path=str(condition_root(FEEDBACK_ROOT,task)/'path.json'))
    return source,source.parent/f'update_{selected:08d}',reader,lora,point


def _task_entries(root,task,path,selected):
    source,reference,reader,first_lora,point=_assets(root,task,path,selected)
    teacher=path['teacher_demo'];first,full=path['first_panel_index'],path['full_panel_index'];mappings=[]
    for state in STATES:
        mappings.extend([_entry('source',task,state,teacher,_existing(source/f'state_{state:03d}.json'),origin='canonical_source'),
            _entry('reference',task,state,teacher,_existing(reference/f'state_{state:03d}.json'),origin='frozen_selected_reference'),
            _entry('full',task,state,teacher,_existing(_original_row(task,state)),lora=path['selected_LoRA'],origin='original_full_query')])
        old=VIDEO_ROOT/'evaluation/update_00000900/readout/seen'/f'task_{task:04d}'/'state_000.json'
        mappings.append(_entry('Reader900',task,state,teacher,_existing(old) if state==0 else _row_file(root,'Reader900',task,state),
            new=state!=0,lora=reader,origin='original_same_video_init0' if state==0 else None))
        original=_original_row(task,state) if first==full else (source/f'state_{state:03d}.json' if first==0 else _row_file(root,'first_best',task,state))
        mappings.append(_entry('first_best',task,state,teacher,original,new=task in NEW_FIRST,lora=first_lora,
            origin='same_as_full_preserve_actual_source_execution' if first==full else ('canonical_source' if first==0 else None)))
    requests=[]
    for arm,states,lora in [('Reader900',(4,8,12),reader),('first_best',STATES,first_lora)]:
        if arm=='first_best' and task not in NEW_FIRST:continue
        requests.extend(dict(identity=f'{arm}_task_{task:04d}',role='query',task=task,state_id=state,
            noise_root=202610151,parameter_ref=str(lora),LoRA=str(lora),arm=arm,teacher_demo=teacher) for state in states)
    return mappings,requests,point


def _task_cost(task,path,selected,frames):
    full=path['full_panel_index'];prefix=path['panels'][:min(2,len(path['panels']))]
    category='source_early_stop' if not path['proposals'] else ('read_then_source' if full==0 else ('first_proposal_best' if full==1 else 'later_proposal_best'))
    return dict(task=task,path=str(condition_root(FEEDBACK_ROOT,task)/'path.json'),teacher_demo=path['teacher_demo'],
        reference_identity=selected==0,category=category,first_panel=path['first_panel_index'],full_panel=full,meta_update=path['meta_update'],
        first_best=dict(practice_panels=len(prefix),practice_episodes=sum(len(p['rows']) for p in prefix),
            practice_steps=sum(sum(r['environment_steps'] for r in p['rows']) for p in prefix),
            proposals=min(1,len(path['proposals'])),teaching_reads=min(1,len(path['proposals'])),
            frame_equivalent=frames*min(1,len(path['proposals']))),
        full=dict(practice_panels=len(path['panels']),practice_episodes=sum(len(p['rows']) for p in path['panels']),
            practice_steps=path['used_steps'],proposals=len(path['proposals']),teaching_reads=path['teaching_reads'],frame_equivalent=path['frame_equivalent']))


def _build(root):
    environment=read_json(FEEDBACK_ROOT/'run_contract.json')['environment'];paths=_paths()
    bank={r['task']:r for r in read_json(PARENT_ROOT/'bank.json')['tasks']}
    descriptors=read_json(VIDEO_ROOT/'features/manifest.json')['entries']
    frames={(r['task'],r['demo']):r['frames'] for r in descriptors if r['kind']=='original'}
    mappings=[];requests=[];tasks=[];materializations=[]
    for task in TASKS:
        path=paths[task];selected=bank[task]['selected_update']
        rows,work,point=_task_entries(root,task,path,selected);mappings.extend(rows);requests.extend(work)
        if point is not None:materializations.append(point)
        tasks.append(_task_cost(task,path,selected,frames[task,path['teacher_demo']]))
    known=[r for r in mappings if r['arm']=='first_best' and not r['new_episode']]
    if len(mappings)!=720 or len(requests)!=124 or len(known)!=128 or sum(r['original_success'] for r in known)!=68:
        raise ValueError('fixed720/124/known128 success68 accounting changed')
    if {c:sum(t['category']==c for t in tasks) for c in ('source_early_stop','read_then_source','first_proposal_best','later_proposal_best')}!=dict(source_early_stop=11,read_then_source=8,first_proposal_best=10,later_proposal_best=7):
        raise ValueError('original branch composition changed')
    contract=dict(schema=SCHEMA,phase='feedback_readout',environment=environment,limits=LIMITS,requests=requests,
        frozen_basis=str(BASIS),initial_reader_checkpoint=str(INITIAL_READER),feedback_root=str(FEEDBACK_ROOT),
        video_root=str(VIDEO_ROOT),reference_root=str(REFERENCE_ROOT),manifest=str(root/'manifest.json'),
        training_updates=0,new_proposals=0,new_practice=0,new_teaching_reads=0,new_Reader_forwards=0,checkpoint_selection=False)
    return contract,dict(schema=SCHEMA,mappings=mappings,tasks=tasks,materializations=materializations,
        logical_rows=720,new_episodes=124,known_first_best_rows=128,known_first_best_successes=68)


def _materialize(root,entries):
    import torch
    from safetensors.torch import save_file
    from ember.lora import validate_lora_state
    from .model import load_basis
    missing=[e for e in entries if not Path(e['path']).is_file()]
    if not missing:return
    torch.set_num_threads(8);decoder=load_basis(BASIS)
    saved=torch.load(INITIAL_READER,map_location='cpu',weights_only=False,mmap=True)
    if saved['next_update']!=900 or saved['frozen_basis']!=str(BASIS):raise ValueError('code units no longer identify the fixed Reader900')
    units=saved['code_units'].float();started=time.monotonic()
    with torch.no_grad():
        for item in missing:
            state=decoder.decode(units*torch.tensor(item['y'],dtype=torch.float32));validate_lora_state(state,decoder.lora)
            if any(not bool(v.isfinite().all()) for v in state.values()):raise ValueError('nonfinite actual first proposal cannot execute')
            destination=Path(item['path']);destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.partial')
            save_file({k:v.float().contiguous() for k,v in state.items()},str(temp));temp.replace(destination)
    write_json_atomic(root/'materialization.json',dict(complete=True,CPU_only=True,new_controllers=len(missing),
        basis=str(BASIS),code_units=str(INITIAL_READER),seconds=time.monotonic()-started,no_Reader_forward=True,no_new_teaching=True))


def prepare(root):
    root=Path(root);manifest=root/'manifest.json';contract=root/'run_contract.json'
    if manifest.is_file():
        saved=read_json(manifest);run=read_json(contract)
        if saved['schema']!=SCHEMA or run['schema']!=SCHEMA or run['phase']!='feedback_readout' or run['limits']!=LIMITS:
            raise ValueError('existing comparison manifest/contract is incompatible')
        for entry in saved['materializations']:_existing(entry['path'])
        return run
    run,plan=_build(root);root.mkdir(parents=True,exist_ok=True)
    transaction=root/'materialization_plan.json'
    if transaction.is_file() and read_json(transaction)!=plan:raise ValueError('partial comparison transaction changed')
    if not transaction.exists():write_json_atomic(transaction,plan)
    _materialize(root,plan['materializations'])
    if contract.exists() and read_json(contract)!=run:raise ValueError('comparison run contract changed')
    if not contract.exists():write_json_atomic(contract,run)
    write_json_atomic(manifest,plan);return run


def _validate(row,entry,task):
    from ember.pi05_eval_contract import policy_noise_seed
    expected=dict(global_task_id=entry['task'],suite=task['suite'],task_id=task['task_id'],language=task['language'],
        init_state_id=entry['state_id'],env_seed=202610151,policy_seed_root=202610151,settling_steps=10,
        full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('fixed query/reference pairing or official execution changed')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon'] or row.get('model_failure')
            or row.get('budget_truncated') or row.get('failure_type') or (not row['success'] and steps!=task['horizon'])
            or row.get('environment_steps')!=steps+10):raise ValueError('comparison needs124 valid new rows, including normal scientific failures')
    noise=[policy_noise_seed(202610151,task['suite'],task['task_id'],entry['state_id'],i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=noise or row.get('first_noise_seed')!=noise[0]:raise ValueError('native actual noise stream changed')
    if entry['new_episode'] and (row.get('actual_parameter_ref')!=entry['LoRA'] or row.get('LoRA')!=entry['LoRA'] or row.get('history_path')):
        raise ValueError('new readout must use its existing fixed LoRA without collecting H')
    if entry['arm']=='Reader900' and not entry['new_episode'] and (row.get('LoRA')!=entry['LoRA'] or row.get('teacher_demo')!=entry['teacher_demo']):
        raise ValueError('Reader900 reused row must be the exact same video/controller')


def _costs(tasks):
    keys=('practice_panels','practice_episodes','practice_steps','proposals','teaching_reads','frame_equivalent')
    return dict(Reader900=dict(new_practice_steps=0,new_Reader_forwards=0,new_teaching_reads=0,preexisting_video_controllers=36),
        first_best={k:sum(t['first_best'][k] for t in tasks) for k in keys},full={k:sum(t['full'][k] for t in tasks) for k in keys},
        interpretation='first-best cost is a truncation of the actual original path prefix; no counterfactual saving was actually realized; no counterfactual GPUh inferred')


def analyze(root):
    root=Path(root);run=read_json(root/'run_contract.json');manifest=read_json(root/'manifest.json')
    if run['schema']!=SCHEMA or manifest['schema']!=SCHEMA:raise ValueError('explicit comparison phase required')
    specs={t['global_task_id']:t for t in run['environment']['tasks']};raw={a:[] for a in ARMS}
    for entry in manifest['mappings']:
        row=read_json(_existing(entry['row_path']));_validate(row,entry,specs[entry['task']]);raw[entry['arm']].append(row)
    tasks=[]
    for info in manifest['tasks']:
        task=info['task'];arms={a:[r for r in rows if r['global_task_id']==task] for a,rows in raw.items()}
        if any(len(rows)!=4 for rows in arms.values()):raise ValueError('every arm needs exactly144 fixed cases')
        tasks.append(dict(summarize_task(task,specs[task],arms,identity=info['reference_identity']),category=info['category']))
    historical=read_json(FEEDBACK_ROOT/'completion.json')
    result=dict(complete=True,schema=SCHEMA,phase='feedback_readout',arms=grouped(tasks),raw_rows=raw,
        changed_seven=grouped([t for t in tasks if t['task'] in CHANGED]),
        new_first_sixteen=grouped([t for t in tasks if t['task'] in NEW_FIRST]),
        canonical_source_twelve=grouped([t for t in tasks if t['task'] in SOURCE_FIRST]),
        categories={c:grouped([t for t in tasks if t['category']==c]) for c in sorted({t['category'] for t in tasks})},
        controllers=manifest['tasks'],costs=_costs(manifest['tasks']),new_readout_interactions=summary(root),new_readout_billing=billing(root),
        historical_feedback=dict(ROOT=str(FEEDBACK_ROOT),GPU_hours=historical['billing']['GPU_hours'],interactions=historical['interactions'],
            included_in_new_readout_cost=False),limits=LIMITS,checkpoint_selection=False,held=False,Test=False,
        scope='posthoc non-held controllers from different training theta; not a single checkpoint or H/video causal qualification')
    folder=root/'analysis';folder.mkdir(exist_ok=True);write_json_atomic(folder/'results.json',result);write_json_atomic(root/'completion.json',result)
    lines=['# 同教学144-case既有控制器五臂比较','',f"全部144：{result['arms']['total']['successes']}",
        f"七变化任务28行：{result['changed_seven']['total']['successes']}",'',
        '|task|category|source|reference|Reader900|first-best|full|','|---|---|---|---|---|---|---|']
    for row in tasks:lines.append('|'+str(row['task'])+'|'+row['category']+'|'+'|'.join(str(len(row['success_sets'][a])) for a in ARMS)+'|')
    lines+=['','全部10对R/G/L/churn/交并/Jaccard、task/suite/target/support/identity及参考新增保持见results.json。',
        'first-best29任务116行严格复用full原执行，未将其中相同source参数换成其它较高结果。',
        f"原路径成本前缀：{result['costs']}",
        '前缀截断仅为反事实成本描述；旧实际实践、失败和GPUh全保留，不声称本批实际节省。',
        '0训练/新提案/实践/H/教学读取/Reader forward/held/Test或checkpoint选择。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');return result
