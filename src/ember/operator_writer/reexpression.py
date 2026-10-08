"""Read-only identities and captures for the sealed §7 finite-panel diagnostic."""
from __future__ import annotations

from pathlib import Path

from ember.lora import expected_lora_state_shapes
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from . import joint_readout, learning_limit

ROOT = Path('/data1/user/ymdai/ember_runs/conditional_A_reexpression_diagnostic_20261002')
CHECKPOINT = ROOT.parent / 'conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900'
ARMS = ('Original', 'Reexpressed')
TASKS, TEACHERS, STATES = joint_readout.TRAIN_TASKS, joint_readout.TEACHERS, (0, 1, 2, 3)
SCENES = learning_limit.ROOT / 'scenes'
FORMULA = 'C=SX(A0X)^+; A_tilde=A0; B_tilde=(B0+M)(I+C); CPU float64 gelsd/default-rcond'


def source_record():
    return joint_readout.a28_source_record(joint_readout.CONDITIONAL_MODE, CHECKPOINT)


def tasks_for_panel():
    # Same current train identities and existing canonical post-dummy scenes.
    return learning_limit.tasks_for_panel()


def bank_path(arm, slot):
    if arm not in ARMS or slot not in (0, 1):
        raise ValueError('unregistered reexpression arm/teacher')
    return ROOT / arm / 'bank' / f'panel_teacher{slot}.json'


def register_inputs():
    rows = tasks_for_panel()
    write_json_atomic(ROOT/'launch/train4_subset.json', {
        'schema_version':'ember_pi05_task_subset_selection_v1',
        'role':'development_train', 'mode':'screen', 'state_count':4,
        'init_state_ids':list(STATES), 'task_ordinals':[0,6,12,18],
        'global_task_ids':list(TASKS),
        'tasks':[{k:r[k] for k in ('global_task_id','suite','task_id')} for r in rows],
        'outcome_dependence':False, 'validation_use':False, 'test_use':False})
    for slot in (0, 1):
        write_json_atomic(ROOT/f'launch/capture_teacher{slot}.json', {
            'schema_version':'ember_pi05_registered_trajectory_capture_v1',
            'study_id':ROOT.name, 'task_subset_selection':str(ROOT/'launch/train4_subset.json'),
            'full_conditions':[dict(suite=r['suite'],task_id=r['task_id'],init_state_id=0) for r in rows],
            'mode':'compact', 'passive_control_trace':'ember_operator_read_write_passive_capture_v1',
            'stage_predicates':True, 'training_gradient_use':False,
            'checkpoint_selection_use':False, 'validation_use':False, 'test_use':False})


def sealed_source_record():
    spec, training, _ = source_record()
    path = ROOT / 'frozen/configs/operator_read_write_v1/conditional_read_write_continuation900_spec.json'
    if read_json(path) != spec:
        raise ValueError('reexpression sealed spec differs from registered source')
    return spec, training, path


def inspect(bank, path, source, task_keys, role, require_formal, task_states):
    from .bank import EVAL_SCHEMA, KIND, _factor_header, source_matches
    from ember.pi05_eval.scene import inspect_registered_scenes
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
    arm, slot = bank['mode'], bank['reexpression_panel']['teacher_slot']
    spec, training, spec_path = sealed_source_record()
    expected = tasks_for_panel()
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank['asset_root'])/spec['source']['lora_contract']), rank=128)
    facts = ((path,bank_path(arm,slot).resolve()),(role,'development_train'),
        (require_formal,True),(bank['kind'],KIND),(bank['status'],'sealed'),
        (bank['condition_layout'],'complete38'),(bank['checkpoint'],str(CHECKPOINT)),
        (bank['reexpression_panel'],dict(study=ROOT.name,arm=arm,teacher_slot=slot,formula=FORMULA)),
        (bank['source'],source),(bank['source'],training['source']),
        (bank['spec'],file_record(spec_path)),(bank['lora'],lora.to_dict()),
        (bank['shared'],file_record(ROOT/'public.safetensors')),
        (bank['training_git'],training['git']['commit']),(bank['scene_root'],str(SCENES)),
        (set(task_keys),{(r['suite'],r['task_id']) for r in expected}))
    if (any(a!=b for a,b in facts) or not source_matches(source,training['source'])
        or task_states is None or any(tuple(v)!=STATES for v in task_states.values())
        or len(bank['tasks'])!=4 or len(bank['conditions'])!=4):
        raise ValueError('reexpression fixed source/panel changed')
    shapes = expected_lora_state_shapes(lora)
    for row, wanted, condition in zip(bank['tasks'],expected,bank['conditions'],strict=True):
        task, teacher = wanted['global_task_id'],TEACHERS[wanted['global_task_id']][slot]
        key = f'task{task:03d}_teacher{teacher:02d}'
        factor = ROOT/arm/'bank'/f'{key}.safetensors'
        episodes=[dict(init_state_id=s,condition_id=key,teacher_demo_indices=[teacher],video_ordinal=slot) for s in STATES]
        if row!={**wanted,'episodes':episodes} or condition!=dict(condition_id=key,
            global_task_id=task,teacher_demo=teacher,factors=file_record(factor)):
            raise ValueError('reexpression teacher/condition pairing changed')
        _factor_header(factor,shapes,metadata=dict(schema_version=bank['schema_version'],mode=arm,condition_id=key))
    inspect_registered_scenes(SCENES,bank['tasks'],states=STATES,schema='ember_operator_seen_task_scenes_v1')
    return {**bank,'schema_version':EVAL_SCHEMA,'arm':'correct','manifest':file_record(path),
            'scene_manifest':file_record(SCENES/'manifest.json')}


def registered_capture(args,tasks,output,path,manifest,subset,bank):
    from .capture import PASSIVE_TAG
    from ember.pi05_assets import Pi05EvaluationError
    arm,slot=bank['mode'],bank['reexpression_panel']['teacher_slot']
    full=[dict(suite=t.suite,task_id=t.task_id,init_state_id=0) for t in tasks]
    wanted=read_json(ROOT/f'launch/capture_teacher{slot}.json')
    if (manifest!=wanted or path.resolve()!=ROOT/f'launch/capture_teacher{slot}.json'
        or Path(args.static_task_lora_manifest).resolve()!=bank_path(arm,slot)
        or output.resolve()!=ROOT/arm/'evaluation'/f'teacher{slot}'
        or args.role!='development_train' or args.mode!='screen' or len(tasks)!=4
        or manifest['full_conditions']!=full or subset is None
        or subset.get('selection_path')!=str(ROOT/'launch/train4_subset.json')
        or any(tuple(t.init_state_ids)!=STATES for t in tasks)):
        raise Pi05EvaluationError('reexpression capture scope changed')
    capture=dict(schema_version='ember_pi05_registered_trajectory_capture_v1',
        selection_path=str(path),selection_bytes=path.stat().st_size,mode='compact',
        full_conditions=full,trajectory_root=str(output/'trajectories'),
        passive_trace=dict(schema_version=PASSIVE_TAG,trace_root=str(output/'continuous_traces')),
        training_gradient_use=False,checkpoint_selection_use=False,validation_use=False,test_use=False)
    stage=dict(schema_version='ember_pi05_stage_predicate_capture_v1',
        capture='all_rows_post_settling_then_every_executed_control_step',
        predicate_source='installed_LIBERO_BDDL_goal_conjunction',full_conditions_only=False,
        training_gradient_use=False,checkpoint_selection_use=False,validation_action_reads=0,
        validation_reward_reads=0,held_data_use=False,claim_boundary='BDDL predicates are partial progress signals')
    return capture,stage
