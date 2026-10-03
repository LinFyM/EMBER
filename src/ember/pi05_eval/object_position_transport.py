"""Temporary, registered task16 world-XY intervention; canonical evaluator owns rollout."""

from __future__ import annotations

import copy
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from ember.pi05_assets import Pi05EvaluationError, prepare_libero_config
from ember.pi05_source_checkpoint import read_json

ROOT = Path('/data1/user/ymdai/ember_runs/object_position_transport_20261004')
DESIGN = Path('docs/designs/object_position_transport_diagnostic.md')
TAG = 'ember_object_position_transport_v1'
STATES = tuple(range(8))
TEACHERS = [47, 33, 28, 46, 32, 1, 24, 43]
NAMES = ('butter_1', 'orange_juice_1')
REFERENCES = {
    'C900': Path('/data1/user/ymdai/ember_runs/conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400'),
    'T2340': Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/evaluation/2340/correct400'),
    'MT300': Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1/MT/evaluation/correct400'),
}


def _reference(model):
    original = read_json(REFERENCES[model] / 'run_contract.json')
    task = next(row for row in original['tasks'] if (row['suite'], row['task_id']) == ('libero_object', 6))
    route = next(row for row in original['adapter']['tasks'] if (row['suite'], row['task_id']) == ('libero_object', 6))
    task = {**task, 'init_state_ids': list(STATES)}
    route = {**route, 'episodes': [row for row in route['episodes'] if row['init_state_id'] in STATES]}
    if (task['horizon'] != 280 or task['language'] != 'pick up the butter and place it in the basket'
            or [row['init_state_id'] for row in route['episodes']] != list(STATES)
            or [row['paired_correct_demos'][0] for row in route['episodes']] != TEACHERS):
        raise Pi05EvaluationError('task16 frozen episode/teacher contract changed')
    adapter = copy.deepcopy(original['adapter'])
    adapter['tasks'] = [route]
    keys = {row['condition_id'] for row in route['episodes']}
    adapter['conditions'] = [row for row in adapter['conditions'] if row['condition_id'] in keys]
    return original, task, adapter


def prepare(model, layout, gpu):
    """Register a subset of an already validated bank, without materialization or a second evaluator."""
    from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, git_state, inspect_tokenizer, load_evaluation_authorities
    from ember.pi05_eval.preparation import shards_from_contract
    from ember.pi05_eval_queue import initialize_queue, publish_json_exclusive

    repo = Path(__file__).resolve().parents[3]
    git = git_state(repo)
    if git['dirty_paths'] or git['branch'] or not git['authority_contains_commit']:
        raise Pi05EvaluationError('diagnostic requires clean pushed detached runtime')
    original, task, adapter = _reference(model)
    output = ROOT / 'evaluation' / model / layout
    if output.exists():
        raise Pi05EvaluationError('registered diagnostic output already exists')
    output.mkdir(parents=True)
    contract = copy.deepcopy(original)
    contract.update(mode='screen', output_dir=str(output), tasks=[task], adapter=adapter,
                    git=git, prepared_unix=time.time(), command=list(sys.argv),
                    contract_reference=f'{RUN_CONTRACT_SCHEMA}:{uuid.uuid4().hex}')
    authorities = load_evaluation_authorities(Path(original['authorities']['config_path']), repo)
    contract['tokenizer'] = inspect_tokenizer(authorities, Path(original['tokenizer']['path']))
    contract['parallel'].update(physical_gpu_ids=[gpu], physical_gpu_count=1,
                                replicas_per_gpu=1, worker_count=1, envs_per_replica=16)
    contract['parallel'].pop('queue_sharding', None)
    paths = prepare_libero_config(output / 'libero_config')
    if paths != original['libero_paths']:
        raise Pi05EvaluationError('canonical LIBERO asset paths changed')
    contract['libero_paths'] = paths
    contract['object_position_transport'] = {
        'schema_version': TAG, 'model': model, 'layout': layout,
        'reference_contract': str(REFERENCES[model] / 'run_contract.json'),
        'design_path': str(repo / DESIGN), 'design_bytes': (repo / DESIGN).stat().st_size,
        'training_gradient_use': False, 'checkpoint_selection_use': False,
        'teacher_actions_read': False, 'new_native_or_writer': False,
    }
    selection = output / 'capture_selection.json'
    full = [{'suite': 'libero_object', 'task_id': 6, 'init_state_id': 0}]
    publish_json_exclusive(selection, {'study': TAG, 'mode': 'compact', 'full_conditions': full})
    contract['diagnostic_occupancy_capture'] = {
        'schema_version': 'ember_pi05_registered_trajectory_capture_v1',
        'mode': 'compact', 'selection_path': str(selection), 'selection_bytes': selection.stat().st_size,
        'trajectory_root': str(output / 'trajectories'), 'full_conditions': full,
        'passive_trace': {'schema_version': TAG, 'trace_root': str(output / 'continuous_traces')},
        'object_position_transport': True,
    }
    contract['diagnostic_stage_predicates'] = {
        'schema_version': 'ember_pi05_stage_predicate_capture_v1',
        'capture': 'all_rows_post_settling_then_every_executed_control_step',
        'full_conditions_only': False,
    }
    contract['diagnostic_task_subset'] = None  # The fixed registered scope is checked by this diagnostic.
    contract.pop('passive_capture_provenance', None)
    validate_contract(contract, repo)
    publish_json_exclusive(output / 'run_contract.json', contract)
    shards = shards_from_contract(contract)
    if len(shards) != 1 or tuple(shards[0].init_state_ids) != STATES:
        raise Pi05EvaluationError('all eight legal cases must pack into one canonical shard')
    initialize_queue(output / 'queue.sqlite3', shards, contract_reference=contract['contract_reference'])
    return {'output': str(output), 'model': model, 'layout': layout, 'rows': 8,
            'source_bank': adapter['manifest'], 'physical_gpu': gpu, 'capacity': 16}


def validate_contract(contract, repo):
    from ember.pi05_eval_contract import inspect_tokenizer, load_evaluation_authorities

    info = contract['object_position_transport']
    model, layout = info['model'], info['layout']
    original, task, adapter = _reference(model)
    authorities = load_evaluation_authorities(Path(original['authorities']['config_path']), repo)
    tokenizer = inspect_tokenizer(authorities, Path(original['tokenizer']['path']))
    expected_output = ROOT / 'evaluation' / model / layout
    capture = contract['diagnostic_occupancy_capture']
    if (layout not in ('original', 'swapped') or info['schema_version'] != TAG
            or Path(contract['output_dir']) != expected_output or contract['mode'] != 'screen'
            or contract['tasks'] != [task] or contract['adapter'] != adapter
            or info['reference_contract'] != str(REFERENCES[model] / 'run_contract.json')
            or info['design_path'] != str(repo / DESIGN)
            or info['design_bytes'] != (repo / DESIGN).stat().st_size
            or any(info[key] is not False for key in ('training_gradient_use', 'checkpoint_selection_use',
                                                      'teacher_actions_read', 'new_native_or_writer'))
            or any(contract[key] != original[key] for key in
                   ('model', 'normalization', 'environment', 'policy', 'rng', 'operator_read_write_scene'))
            or contract['tokenizer'] != tokenizer
            or capture['passive_trace']['schema_version'] != TAG
            or capture['passive_trace']['trace_root'] != str(expected_output / 'continuous_traces')
            or capture['trajectory_root'] != str(expected_output / 'trajectories')
            or capture['full_conditions'] != [{'suite': 'libero_object', 'task_id': 6, 'init_state_id': 0}]):
        raise Pi05EvaluationError('object-position diagnostic provenance or scope changed')


def reinspect_subset(contract, model):
    """Recheck the pinned original bank and only the selected existing factor files."""
    from safetensors import safe_open

    original, _, expected = _reference(contract['object_position_transport']['model'])
    if original['model'] != model:
        raise Pi05EvaluationError('diagnostic subset source changed')
    manifest = expected['manifest']
    path = Path(manifest['path'])
    if not path.is_file() or path.stat().st_size != manifest['bytes']:
        raise Pi05EvaluationError('original bank manifest changed')
    bank = read_json(path)
    selected = {row['condition_id']: row for row in expected['conditions']}
    actual = {row['condition_id']: row for row in bank['conditions'] if row['condition_id'] in selected}
    task = next(row for row in bank['tasks'] if (row['suite'], row['task_id']) == ('libero_object', 6))
    task = {**task, 'episodes': [row for row in task['episodes'] if row['init_state_id'] in STATES]}
    if (actual != selected or expected['tasks'] != [task]
            or any(bank[key] != expected[key] for key in ('shared', 'lora', 'source'))):
        raise Pi05EvaluationError('selected original factors/source changed')
    records = [expected['shared']] + [row['factors'] for row in selected.values() if 'factors' in row]
    for record in records:
        factor = Path(record['path'])
        if not factor.is_file() or factor.stat().st_size != record['bytes']:
            raise Pi05EvaluationError('selected factor file changed')
        with safe_open(factor, framework='pt', device='cpu') as handle:
            keys = handle.keys()
            dtypes = {'F32', 'BF16'} if expected['mode'] == 'MT' else {'F32'}
            if len(keys) not in (38, 76) or any(handle.get_slice(key).get_dtype() not in dtypes for key in keys):
                raise Pi05EvaluationError('selected complete frozen LoRA structure changed')
    return expected


def passive_in(env):
    owner = env.env
    goal = owner.parsed_problem['goal_state']
    if len(goal) != 1 or len(goal[0]) != 3 or str(goal[0][0]).lower() != 'in' or goal[0][1] != NAMES[0]:
        raise Pi05EvaluationError('butter basket native In goal changed')
    return np.asarray([owner._eval_predicate([goal[0][0], name, goal[0][2]]) for name in NAMES], dtype=np.bool_)


def _controller_snapshot(owner):
    return {key: np.asarray(value).copy() for key, value in vars(owner.robots[0].controller).items()
            if isinstance(value, (np.ndarray, float, int)) and np.asarray(value).dtype.kind in 'biuf'}


def _contacts(sim):
    return [{'geom1': int(c.geom1), 'geom2': int(c.geom2), 'distance_m': float(c.dist)}
            for c in sim.data.contact[:sim.data.ncon]]


def apply(env, observation, contract, state):
    """Translate registered free-joint roots by the required world-body delta; never step."""
    from ember.pi05_eval.scene import _capture_image

    owner, info = env.env, contract['object_position_transport']
    sim, model, data = owner.sim, owner.sim.model, owner.sim.data
    registry = owner.obj_body_id
    bodies = [int(registry[name]) for name in NAMES]
    joints, addresses, roots = [], [], []
    for name in NAMES:
        obj = owner.objects_dict[name]
        if len(obj.joints) != 1:
            raise Pi05EvaluationError(f'{name}: expected one real free-joint root')
        joint = int(model.joint_name2id(obj.joints[0]))
        root = int(model.jnt_bodyid[joint])
        if int(model.jnt_type[joint]) != 0 or int(model.body_parentid[root]) != 0:
            raise Pi05EvaluationError(f'{name}: free-joint world frame is not established')
        joints.append(joint); roots.append(root); addresses.append(int(model.jnt_qposadr[joint]))
    before = {name: np.asarray(value).copy() for name, value in {
        'qpos': data.qpos, 'qvel': data.qvel, 'act': data.act,
        'body_xpos': data.body_xpos, 'body_xquat': data.body_xquat,
        'model_body_pos': model.body_pos, 'model_body_quat': model.body_quat,
        'sim_state': env.get_sim_state(), 'objects_world': data.body_xpos[bodies],
        'eef_pos': observation['robot0_eef_pos'], 'eef_quat': observation['robot0_eef_quat'],
        'gripper_qpos': observation['robot0_gripper_qpos'],
    }.items()}
    controller = _controller_snapshot(owner)
    initial_rgb = _capture_image(observation)
    sim_time, timestep = float(data.time), int(owner.timestep)
    old_contacts = _contacts(sim)
    allowed = np.zeros(len(data.qpos), dtype=np.bool_)
    if info['layout'] == 'swapped':
        for index, address in enumerate(addresses):
            delta = before['objects_world'][1-index, :2] - before['objects_world'][index, :2]
            data.qpos[address:address+2] += delta
            allowed[address:address+2] = True
    sim.forward()
    observation = owner._get_observations(force_update=True)
    after = {name: np.asarray(value).copy() for name, value in {
        'qpos': data.qpos, 'qvel': data.qvel, 'act': data.act,
        'body_xpos': data.body_xpos, 'body_xquat': data.body_xquat,
        'model_body_pos': model.body_pos, 'model_body_quat': model.body_quat,
        'sim_state': env.get_sim_state(), 'objects_world': data.body_xpos[bodies],
        'eef_pos': observation['robot0_eef_pos'], 'eef_quat': observation['robot0_eef_quat'],
        'gripper_qpos': observation['robot0_gripper_qpos'],
    }.items()}
    target = before['objects_world'].copy()
    if info['layout'] == 'swapped':
        target[:, :2] = target[::-1, :2]
    affected = np.zeros(model.nbody, dtype=np.bool_)
    for body in range(model.nbody):
        ancestor = body
        while ancestor and ancestor not in roots:
            ancestor = int(model.body_parentid[ancestor])
        affected[body] = ancestor in roots
    other_controller = _controller_snapshot(owner)
    invariants = {
        'only_world_XY': bool(np.allclose(after['objects_world'], target, rtol=0, atol=1e-9)),
        'other_qpos': bool(np.array_equal(before['qpos'][~allowed], after['qpos'][~allowed])),
        'qvel': bool(np.array_equal(before['qvel'], after['qvel'])),
        'act': bool(np.array_equal(before['act'], after['act'])),
        'body_quaternions': bool(np.allclose(before['body_xquat'], after['body_xquat'], rtol=0, atol=1e-9)),
        'other_bodies': bool(np.allclose(before['body_xpos'][~affected], after['body_xpos'][~affected], rtol=0, atol=1e-9)),
        'model_pose': bool(all(np.array_equal(before[k], after[k]) for k in ('model_body_pos', 'model_body_quat'))),
        'no_step': bool(float(data.time) == sim_time and int(owner.timestep) == timestep),
        'controller': bool(controller.keys() == other_controller.keys() and
                           all(np.array_equal(value, other_controller[key]) for key, value in controller.items())),
        'own_state8_inputs': bool(all(np.array_equal(before[key], after[key]) for key in
                                     ('eef_pos', 'eef_quat', 'gripper_qpos'))),
    }
    path = Path(contract['output_dir']) / 'initial_states' / f'state_{state:03d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **{f'before_{key}': value for key, value in before.items()},
                            **{f'after_{key}': value for key, value in after.items()},
                            body_names=np.asarray([model.body_id2name(i) or '' for i in range(model.nbody)]),
                            object_names=np.asarray(NAMES), object_body_ids=np.asarray(bodies),
                            joint_names=np.asarray([model.joint_id2name(j) for j in joints]),
                            qpos_addresses=np.asarray(addresses), before_rgb=initial_rgb,
                            after_rgb=_capture_image(observation), sim_time=np.asarray(sim_time),
                            **{f'controller_{key}': value for key, value in controller.items()})
    receipt = {**info, 'state': state, 'physical_state': {'path': str(path), 'bytes': path.stat().st_size},
               'before_world': before['objects_world'].tolist(), 'after_world': after['objects_world'].tolist(),
               'invariants': invariants, 'contacts_before': old_contacts, 'contacts_after': _contacts(sim),
               'native_passive_In': passive_in(env).tolist(), 'environment_steps_added': 0}
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not all(invariants.values()):
        raise Pi05EvaluationError(f'physical XY intervention invariant failed; preserved {path}')
    return observation, receipt


def validate_row(row, task, contract):
    info = row.get('object_position_transport') or {}
    path = Path(str((info.get('physical_state') or {}).get('path', '')))
    if (info.get('state') != row['init_state_id'] or info.get('layout') != contract['object_position_transport']['layout']
            or info.get('environment_steps_added') != 0 or not all(info.get('invariants', {}).values())
            or not path.is_file() or path.stat().st_size != info['physical_state']['bytes']):
        raise Pi05EvaluationError('physical position intervention row is incomplete')
