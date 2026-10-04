"""Registered197 cases, composed around the single official persistent evaluator."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from safetensors.torch import save_file
import ember.pi05_evaluation as evaluation
from ember.lora import LORA_A_SUFFIX, validate_lora_state
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.scene import validate_scene_row, _capture_image
from ember.pi05_eval.trajectory_capture import validate_passive_trace_row, start_passive_trace
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from .bank import FrozenOperatorAdapter, PreparedOperatorLoRA, episode_evidence
from .role_binding import compile_fixed
from .binding_run import ROOT, TASKS, HELD_TEACHERS, load_native

TRAIN_REFERENCE = ROOT.parent / 'operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/T/evaluation/correct144/run_contract.json'
HELD_REFERENCE = ROOT.parent / 'conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/run_contract.json'
SWAP_TEACHERS = (47, 33, 28, 46, 32, 1, 24, 43)


def cases(arm):
    rows = [dict(case_id=f'{arm}_task{t:03d}_teacher{d}_init{i}', panel='train', task=t,
                 teacher=d, physical_init=i, noise_init=i, layout='original', full=d == 0 and i == 32)
            for t in TASKS for d in (0, 1) for i in (32, 33)]
    if arm != 'parent':
        rows += [dict(case_id=f'{arm}_task016_scene{i}_teacher{d}_noise{k}', panel='crossover', task=16,
                      teacher=d, physical_init=i, noise_init=k, layout='original', full=True)
                 for i, d, k in product((0, 46), (47, 40), (0, 46))]
        rows += [dict(case_id=f'{arm}_task016_init{i}_{layout}', panel='layout', task=16,
                      teacher=d, physical_init=i, noise_init=i, layout=layout, full=True)
                 for layout in ('original', 'swapped') for i, d in enumerate(SWAP_TEACHERS)
                 if (i, d, layout) != (0, 47, 'original')]
    if len(rows) != (32 if arm == 'parent' else 55):
        raise ValueError('fixed endpoint panel or unique overlap changed')
    return rows


def transform(env, observation, case, path):
    """Exact old world-XY free-joint transport, with no environment step/settling."""
    owner = env.env; sim = owner.sim; model = sim.model; data = sim.data
    names = ('butter_1', 'orange_juice_1')
    bodies = [int(owner.obj_body_id[n]) for n in names]
    addresses = []
    for name in names:
        joints = owner.objects_dict[name].joints
        if len(joints) != 1:
            raise ValueError('same free-joint entity transport contract changed')
        joint = int(model.joint_name2id(joints[0])); body = int(model.jnt_bodyid[joint])
        if int(model.jnt_type[joint]) != 0 or int(model.body_parentid[body]) != 0:
            raise ValueError('same world frame free-joint contract changed')
        addresses.append(int(model.jnt_qposadr[joint]))
    controller = owner.robots[0].controller
    control = {k: np.asarray(v).copy() for k, v in vars(controller).items()
               if isinstance(v, (np.ndarray, float, int)) and np.asarray(v).dtype.kind in 'biuf'}
    before = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), act=data.act.copy(),
        body_pos=data.body_xpos.copy(), body_quat=data.body_xquat.copy(), objects=data.body_xpos[bodies].copy(),
        eef_pos=np.asarray(observation['robot0_eef_pos']).copy(), eef_quat=np.asarray(observation['robot0_eef_quat']).copy(),
        gripper=np.asarray(observation['robot0_gripper_qpos']).copy(), rgb=_capture_image(observation))
    clock = float(data.time), int(owner.timestep)
    allowed = np.zeros(len(data.qpos), bool)
    if case['layout'] == 'swapped':
        for i, address in enumerate(addresses):
            data.qpos[address:address + 2] += before['objects'][1 - i, :2] - before['objects'][i, :2]
            allowed[address:address + 2] = True
    sim.forward(); observation = owner._get_observations(force_update=True)
    target = before['objects'].copy()
    if case['layout'] == 'swapped':
        target[:, :2] = target[::-1, :2]
    invariants = dict(world_XY=np.allclose(data.body_xpos[bodies], target, rtol=0, atol=1e-9),
        other_qpos=np.array_equal(data.qpos[~allowed], before['qpos'][~allowed]),
        qvel=np.array_equal(data.qvel, before['qvel']), act=np.array_equal(data.act, before['act']),
        quaternion=np.allclose(data.body_xquat, before['body_quat'], rtol=0, atol=1e-9),
        controller=all(np.array_equal(getattr(controller, k), v) for k, v in control.items()),
        no_step=(float(data.time), int(owner.timestep)) == clock,
        robot=all(np.array_equal(observation[k], before[v]) for k, v in
                  (('robot0_eef_pos', 'eef_pos'), ('robot0_eef_quat', 'eef_quat'), ('robot0_gripper_qpos', 'gripper'))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **{'before_' + k: v for k, v in before.items()},
                            after_qpos=data.qpos.copy(), after_qvel=data.qvel.copy(),
                            after_objects=data.body_xpos[bodies].copy(), after_rgb=_capture_image(observation),
                            object_names=np.asarray(names), after_eef_pos=observation['robot0_eef_pos'])
    receipt = dict(invariants={k: bool(v) for k, v in invariants.items()}, physical_state=file_record(path),
                   transform_source_git=read_json(ROOT.parent / 'object_position_transport_20261004/evaluation/C900/original/run_contract.json')['git']['commit'], layout=case['layout'], environment_steps_added=0)
    write_json_atomic(path.with_suffix('.json'), receipt)
    if not all(invariants.values()):
        raise ValueError(f'exact frozen XY intervention violated its contract: {receipt}')
    return observation, receipt


def passive_in(env):
    owner = env.env
    goal = owner.parsed_problem['goal_state']
    if len(goal) != 1 or goal[0][0].lower() != 'in' or goal[0][1] != 'butter_1':
        raise ValueError('task16 official butter-only In stop changed')
    return np.asarray([owner._eval_predicate([goal[0][0], name, goal[0][2]])
                       for name in ('butter_1', 'orange_juice_1')], dtype=bool)


@contextmanager
def panel_context(adapter, base, selected, task, arm):
    """Scoped condition/noise/capture identity; physical states remain real."""
    start, noise, finish, record = (evaluation.start_fixed_episode, evaluation.make_policy_noise,
                                   evaluation.finish_episode_row, evaluation.record_passive_step)
    local_contracts = {}
    for ordinal, case in enumerate(selected):
        value = deepcopy(base); path = ROOT / 'evaluation' / arm / 'cases' / case['case_id']
        value['output_dir'] = str(path)
        capture = value['diagnostic_occupancy_capture']
        capture.update(mode='full' if case['full'] else 'compact', full_conditions=[], trajectory_root=str(path / 'trajectory'))
        capture['passive_trace']['trace_root'] = str(path / 'continuous')
        value['native_role_binding_compilation'] = dict(study=ROOT.name, arm=arm, case=case)
        local_contracts[ordinal] = value
        write_json_atomic(path / 'run_contract.json', value)

    def start_case(**kwargs):
        ordinal = int(kwargs['init_state_id']); case = selected[ordinal]
        local = local_contracts[ordinal]
        key = f'task{case["task"]:03d}_teacher{case["teacher"]:02d}'
        bank_task = next(t for t in adapter.bank['tasks'] if t['global_task_id'] == case['task'])
        episode = dict(init_state_id=case['physical_init'], condition_id=key, teacher_demo_indices=[case['teacher']], video_ordinal=None)
        prepared = PreparedOperatorLoRA(key, episode_evidence(local['adapter'], bank_task, episode))
        kwargs.update(init_state_id=case['physical_init'], contract=local,
                      capture_level='full' if case['full'] else 'compact',
                      task_adapter=SimpleNamespace(prepare_episode=lambda **unused: prepared))
        slot = start(**kwargs); slot['binding_case'] = case; slot['binding_contract'] = local
        if case['task'] == 16:
            slot['obs'], slot['layout_receipt'] = transform(kwargs['env'], slot['obs'], case,
                Path(local['output_dir']) / 'initial_state.npz')
            start_passive_trace(kwargs['env'], slot, local['diagnostic_occupancy_capture'])
            slot['passive_In'] = [passive_in(kwargs['env'])]
        return slot

    def make_noise(slots, **kwargs):
        copies = [dict(slot, init_state_id=slot['binding_case']['noise_init']) for slot in slots]
        return noise(copies, **kwargs)

    def passive(env, slot, action, capture):
        record(env, slot, action, capture)
        if slot['binding_case']['task'] == 16:
            slot['passive_In'].append(passive_in(env))

    def finish_case(**kwargs):
        slot = kwargs['slot']; case = slot['binding_case']; local = slot['binding_contract']
        kwargs['contract'] = local
        row = finish(**kwargs)
        row['binding_case'] = case
        if case['task'] == 16:
            row['layout_receipt'] = slot['layout_receipt']
            path = Path(local['output_dir']) / 'passive_In.npz'
            np.savez_compressed(path, In=np.asarray(slot['passive_In']))
            row['passive_In'] = file_record(path)
        expected = [evaluation.policy_noise_seed(7, task['suite'], task['task_id'], case['noise_init'], i)
                    for i in range(len(row['policy_noise_seeds']))]
        if row['policy_noise_seeds'] != expected or row['init_state_id'] != case['physical_init']:
            raise ValueError('actual physical/condition/absolute policy stream identities diverged')
        if bool(all(row['stage_predicates']['final_satisfied'])) != row['success']:
            raise ValueError('official complete goal and success mismatch')
        validate_passive_trace_row(row, local, task); validate_scene_row(row, task, local)
        write_json_atomic(Path(local['output_dir']) / 'results.json', row)
        return row

    evaluation.start_fixed_episode, evaluation.make_policy_noise = start_case, make_noise
    evaluation.finish_episode_row, evaluation.record_passive_step = finish_case, passive
    try:
        yield
    finally:
        evaluation.start_fixed_episode, evaluation.make_policy_noise = start, noise
        evaluation.finish_episode_row, evaluation.record_passive_step = finish, record


def evaluate(runtime, bank, arm, identity):
    train, held = read_json(TRAIN_REFERENCE), read_json(HELD_REFERENCE)
    planned = cases(arm); out = ROOT / 'evaluation' / arm
    base = deepcopy(train)
    base.update(adapter=bank, output_dir=str(out), git=identity,
                native_role_binding_compilation=dict(study=ROOT.name, arm=arm))
    base['parallel'].update(envs_per_replica=23, physical_gpu_count=1, physical_gpu_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])])
    runtime.restore_identity()
    adapter = FrozenOperatorAdapter(policy=runtime.policy, source=bank['source'], evaluation_adapter=bank,
        task_keys=tuple((t['suite'], t['task_id']) for t in bank['tasks']), device=runtime.device,
        require_formal=True, reuse_injected=True)
    pool = PersistentTaskEnvironmentPool(base, physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
    rows, timings = [], []
    try:
        for t in (43, 96, *TASKS[:6], *((16,) if arm != 'parent' else ())):
            reference = held if t == 16 else train
            btask = next(x for x in bank['tasks'] if x['global_task_id'] == t)
            task = deepcopy(next(x for x in reference['tasks'] if (x['suite'], x['task_id']) == (btask['suite'], btask['task_id'])))
            selected = [c for c in planned if c['task'] == t]
            task['init_state_ids'] = list(range(len(selected)))
            current = deepcopy(base)
            current['operator_read_write_scene'] = deepcopy(reference['operator_read_write_scene'])
            current['rng'] = deepcopy(reference['rng']); current['policy'] = deepcopy(reference['policy'])
            current['adapter']['scene_manifest'] = reference['adapter']['scene_manifest']
            started = time.monotonic(); envs, initial = pool.switch(task)
            with panel_context(adapter, current, selected, task, arm):
                result = evaluation.rollout_shard(envs=envs, init_states=initial, task=task,
                    state_ids=task['init_state_ids'], contract=current, policy=runtime.policy,
                    preprocess=runtime.processor, postprocess=runtime.processor.unnormalize_action, task_adapter=adapter)
            rows.extend(result)
            timings.append(dict(task=t, cases=len(selected), seconds=time.monotonic() - started, actual_max_batch=adapter.max_inference_batch))
            write_json_atomic(out / 'results.json', dict(rows=rows, timings=timings, source_loads=1,
                Writer_removed=True, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30))
        if len(rows) != len(planned) or sum(r['binding_case']['full'] for r in rows) != (8 if arm == 'parent' else 31):
            raise ValueError('fixed unique train/held/full endpoint panel incomplete')
    finally:
        pool.close(); adapter.close()
    return rows
