"""Design §33/34 registered finite train panels on the existing evaluator."""
from __future__ import annotations

from pathlib import Path

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.task_protocol import load_task_authorities
from ember.writer.materialization import file_record

STUDY = 'operator_learning_limit_diagnosis_20260930'
ROOT = Path('/data1/user/ymdai/ember_runs') / STUDY
TASKS = (0,12,20,32)
TEACHERS = {0:(40,11),12:(25,14),20:(38,42),32:(17,43)}
ARMS = ('parent','S','P','D')
PROJECTED_ROOT = ROOT.parent / 'operator_projected_repair_consumers_20260930'
STATES = (0,1,2,3)
PARENT = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340')
BASE_BANK = PARENT.parents[4] / 'banks/2340/manifest.json'


def panel_root(arm):
    if arm == 'PZ':
        return PROJECTED_ROOT
    if arm not in ARMS:
        raise ValueError('unregistered learning-limit arm')
    return ROOT


def endpoint(arm):
    if arm == 'PZ':
        complete = read_json(PROJECTED_ROOT/'projection/projection.json')
        if (complete.get('status')!='complete' or complete.get('source_root')!=str(ROOT)
            or complete.get('singular_relative_cutoff')!=1e-6):
            raise ValueError('projected repair source/definition changed')
        return PROJECTED_ROOT/'projection/delta_O.safetensors'
    if arm == 'parent':
        return PARENT
    complete = read_json(panel_root(arm)/arm/'completion.json')
    if complete.get('status')!='complete' or complete.get('updates_per_copy')!=64:
        raise ValueError('only complete64 endpoint may enter the learning-limit panel')
    return ROOT/arm/'recovery_64.pt'


def panel_identity(arm, slot):
    result = {'study':panel_root(arm).name,'teacher_slot':slot,'parent_checkpoint':str(PARENT),
        'parent_training_git':'e2afbfd7c997e3f792921600608efa2fa3c1b25a',
        'query_manifest':file_record(ROOT/'query_manifest.json')}
    if arm == 'PZ':
        result['projection'] = file_record(PROJECTED_ROOT/'projection/projection.json')
    return result


def bank_path(arm, slot):
    if slot not in (0,1):
        raise ValueError('unregistered learning-limit arm/teacher slot')
    return panel_root(arm) / arm / 'bank' / f'panel_teacher{slot}.json'


def tasks_for_panel():
    base = read_json(BASE_BANK)
    spec = read_json(Path(base['spec']['path']))
    _, authority = load_task_authorities(Path(base['asset_root']), spec['source']['data_protocol'])
    rows = {int(r['global_task_id']):r for r in authority['tasks']}
    result = []
    for task in TASKS:
        row = rows[task]
        if row['split_role'] != 'train':
            raise ValueError('learning-limit task crossed the train information wall')
        result.append({k:row[k] for k in ('global_task_id','suite','task_id','language','split_role')})
    return result


def register_inputs(arm='parent'):
    root = panel_root(arm)
    rows = tasks_for_panel()
    subset = {'schema_version':'ember_pi05_task_subset_selection_v1',
        'role':'development_train','mode':'screen','state_count':4,'init_state_ids':list(STATES),
        'task_ordinals':[0,6,12,18],'global_task_ids':list(TASKS),
        'tasks':[{k:r[k] for k in ('global_task_id','suite','task_id')} for r in rows],
        'outcome_dependence':False,'validation_use':False,'test_use':False}
    write_json_atomic(root/'launch/train4_subset.json',subset)
    for slot in (0,1):
        full = [{'suite':r['suite'],'task_id':r['task_id'],'init_state_id':0} for r in rows] if slot==0 else []
        capture = {'schema_version':'ember_pi05_registered_trajectory_capture_v1',
            'study_id':root.name,'task_subset_selection':str(root/'launch/train4_subset.json'),
            'full_conditions':full,'mode':'compact',
            'passive_control_trace':'ember_operator_read_write_passive_capture_v1',
            'stage_predicates':True,'training_gradient_use':False,'checkpoint_selection_use':False,
            'validation_use':False,'test_use':False}
        write_json_atomic(root/f'launch/capture_teacher{slot}.json',capture)


def register_bank(arm, slot):
    from .bank import BANK_SCHEMA,KIND
    base = read_json(BASE_BANK)
    root = panel_root(arm)
    rows, conditions = tasks_for_panel(), []
    for row in rows:
        task = row['global_task_id']
        teacher = TEACHERS[task][slot]
        key = f'task{task:03d}_teacher{teacher:02d}'
        factors = root/arm/'bank'/f'{key}.safetensors'
        conditions.append({'condition_id':key,'global_task_id':task,'teacher_demo':teacher,
                           'factors':file_record(factors)})
        row['episodes'] = [{'init_state_id':state,'condition_id':key,
                            'teacher_demo_indices':[teacher],'video_ordinal':slot} for state in STATES]
    result = {'schema_version':BANK_SCHEMA,'kind':KIND,'status':'sealed','mode':arm,
        'condition_layout':'complete38','learning_limit_panel':panel_identity(arm,slot),
        'asset_root':base['asset_root'],'spec':base['spec'],'source':base['source'],
        'shared':base['shared'],'lora':base['lora'],
        'checkpoint':str(endpoint(arm)),
        'scene_root':str(ROOT/'scenes'),'tasks':rows,'conditions':conditions}
    path = bank_path(arm,slot)
    write_json_atomic(path,result)
    return path


def inspect(bank,path,source,task_keys,role,require_formal,task_states):
    from .bank import EVAL_SCHEMA,KIND,source_matches
    from ember.pi05_eval.scene import inspect_registered_scenes
    from ember.pi05_lora import derive_pi05_lora_rank,load_pi05_lora_contract
    from ember.lora import expected_lora_state_shapes
    from safetensors import safe_open
    panel = bank['learning_limit_panel']
    arm,slot = bank['mode'],panel['teacher_slot']
    root = panel_root(arm)
    base = read_json(BASE_BANK)
    expected = tasks_for_panel()
    if (path!=bank_path(arm,slot).resolve() or bank['kind']!=KIND or bank['status']!='sealed'
        or bank['condition_layout']!='complete38' or role!='development_train' or not require_formal
        or panel!=panel_identity(arm,slot)
        or bank['source']!=source or not source_matches(source,base['source'])
        or any(bank[k]!=base[k] for k in ('asset_root','spec','shared','lora'))
        or bank['scene_root']!=str(ROOT/'scenes')
        or set(task_keys)!={(r['suite'],r['task_id']) for r in expected}
        or task_states is None or any(tuple(v)!=STATES for v in task_states.values())):
        raise ValueError('learning-limit registered scope/source changed')
    if bank['checkpoint']!=str(endpoint(arm)):
        raise ValueError('learning-limit endpoint identity changed')
    spec=read_json(Path(base['spec']['path']))
    lora=derive_pi05_lora_rank(load_pi05_lora_contract(Path(base['asset_root'])/spec['source']['lora_contract']),rank=128)
    shapes=expected_lora_state_shapes(lora)
    if len(bank['tasks'])!=4 or len(bank['conditions'])!=4:
        raise ValueError('learning-limit panel lost conditions')
    for row,wanted,condition in zip(bank['tasks'],expected,bank['conditions'],strict=True):
        task=wanted['global_task_id'];teacher=TEACHERS[task][slot];key=f'task{task:03d}_teacher{teacher:02d}'
        episodes=[{'init_state_id':s,'condition_id':key,'teacher_demo_indices':[teacher],'video_ordinal':slot} for s in STATES]
        factors=root/arm/'bank'/f'{key}.safetensors'
        if row!={**wanted,'episodes':episodes} or condition!={'condition_id':key,'global_task_id':task,
                'teacher_demo':teacher,'factors':file_record(factors)}:
            raise ValueError('learning-limit teacher/state/file pairing changed')
        with safe_open(str(factors),framework='pt',device='cpu') as handle:
            if set(handle.keys())!=set(shapes) or any(tuple(handle.get_slice(k).get_shape())!=v for k,v in shapes.items()):
                raise ValueError('learning-limit complete38 rank128 state shape changed')
    inspect_registered_scenes(ROOT/'scenes',bank['tasks'],states=STATES,schema='ember_operator_seen_task_scenes_v1')
    return {**bank,'schema_version':EVAL_SCHEMA,'arm':'correct','manifest':file_record(path),
            'scene_manifest':file_record(ROOT/'scenes/manifest.json')}


def registered_capture(args,tasks,output,path,manifest,subset,bank):
    from .capture import PASSIVE_TAG
    from ember.pi05_assets import Pi05EvaluationError
    slot=bank['learning_limit_panel']['teacher_slot'];arm=bank['mode']
    root=panel_root(arm)
    full=[{'suite':t.suite,'task_id':t.task_id,'init_state_id':0} for t in tasks] if slot==0 else []
    wanted=read_json(root/f'launch/capture_teacher{slot}.json')
    if (manifest!=wanted or path.resolve()!=root/f'launch/capture_teacher{slot}.json'
        or Path(args.static_task_lora_manifest).resolve()!=bank_path(arm,slot)
        or output.resolve()!=root/arm/'evaluation'/f'teacher{slot}'
        or args.role!='development_train' or args.mode!='screen' or len(tasks)!=4
        or manifest['full_conditions']!=full or subset is None
        or subset.get('selection_path')!=str(root/'launch/train4_subset.json')
        or any(tuple(t.init_state_ids)!=STATES for t in tasks)):
        raise Pi05EvaluationError('learning-limit capture scope changed')
    capture={'schema_version':'ember_pi05_registered_trajectory_capture_v1',
        'selection_path':str(path),'selection_bytes':path.stat().st_size,'mode':'compact',
        'full_conditions':full,'trajectory_root':str(output/'trajectories'),
        'passive_trace':{'schema_version':PASSIVE_TAG,'trace_root':str(output/'continuous_traces')},
        'training_gradient_use':False,'checkpoint_selection_use':False,'validation_use':False,'test_use':False}
    stage={'schema_version':'ember_pi05_stage_predicate_capture_v1',
        'capture':'all_rows_post_settling_then_every_executed_control_step',
        'predicate_source':'installed_LIBERO_BDDL_goal_conjunction','full_conditions_only':False,
        'training_gradient_use':False,'checkpoint_selection_use':False,'validation_action_reads':0,
        'validation_reward_reads':0,'held_data_use':False,'claim_boundary':'BDDL predicates are partial progress signals'}
    return capture,stage


def create_scenes(asset_root,gpu_index):
    import numpy as np
    from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
    from ember.pi05_eval.scene import _scene_snapshot,_restore_scene,_assert_scene_pair,scene_path,inspect_registered_scenes
    from ember.pi05_eval_contract import load_evaluation_authorities,inspect_installed_target_tasks
    authorities=load_evaluation_authorities(asset_root/'configs/libero_24_8_8_coverage_v1/evaluation.json',asset_root)
    installed,paths=inspect_installed_target_tasks(authorities,role='development_train',state_count=4,
                                                 libero_config_dir=ROOT/'scene_libero_config')
    selected={r['suite']:r['task_id'] for r in tasks_for_panel()}
    tasks=[t for t in installed if selected.get(t.suite)==t.task_id]
    if len(tasks)!=4:
        raise ValueError('learning-limit installed scene tasks changed')
    pool=PersistentTaskEnvironmentPool({'libero_paths':paths,'environment':authorities.config['environment'],
                                       'parallel':{'envs_per_replica':1}},physical_gpu_id=gpu_index)
    root=ROOT/'scenes';root.mkdir(exist_ok=True);rows=[]
    dummy=np.asarray(authorities.config['environment']['dummy_action'],dtype=np.float64)
    try:
        for task in tasks:
            envs,states=pool.switch(vars(task));env=envs[0]
            names=sorted(env.env.obj_body_id);goals=[list(g) for g in env.env.parsed_problem['goal_state']]
            for state in STATES:
                path=scene_path(root,vars(task),state)
                if not path.exists():
                    env.seed(7);env.reset();obs=env.set_init_state(states[state])
                    for _ in range(10):
                        obs,_,_,_=env.step(dummy)
                    snapshot=_scene_snapshot(env,obs,names,goals,image=True)
                    # Canonical reobservation, no physics update; seal exactly
                    # the RGB/state that all four arms must subsequently restore.
                    obs=_restore_scene(env,snapshot)
                    snapshot=_scene_snapshot(env,obs,names,goals,image=True)
                    _assert_scene_pair(env,_restore_scene(env,snapshot),names,goals,snapshot,image=True)
                    temp=path.with_suffix('.partial.npz')
                    with temp.open('wb') as handle:
                        np.savez_compressed(handle,**snapshot)
                    temp.replace(path)
                rows.append({'suite':task.suite,'task_id':task.task_id,'state':state,
                             'path':str(path),'bytes':path.stat().st_size})
    finally:
        pool.close()
    write_json_atomic(root/'manifest.json',{'schema_version':'ember_operator_seen_task_scenes_v1',
        'seed':7,'dummy_steps':10,'scope':STUDY,'scenes':rows})
    inspect_registered_scenes(root,[vars(t) for t in tasks],states=STATES,schema='ember_operator_seen_task_scenes_v1')
    return root/'manifest.json'
