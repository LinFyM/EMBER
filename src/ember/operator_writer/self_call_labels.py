"""Temporary §134 own initial-state masks: CPU restoration, zero physics steps."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from ember.pi05_source_checkpoint import read_json, write_json_atomic

ROOT = Path('/data1/user/ymdai/ember_runs/native_role_self_call_20261004')
OLD = ROOT.parent / 'native_role_binding_compilation_20261004'
ASSET = Path('/data1/user/ymdai/projects/EMBER')
TARGETS = {12: 'salad_dressing_1', 13: 'bbq_sauce_1', 14: 'ketchup_1',
           15: 'tomato_sauce_1', 17: 'milk_1', 19: 'orange_juice_1',
           43: 'butter_2', 96: 'butter_1', 16: 'butter_1'}
IMAGE_KEYS = ('observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb')
HELD_REFERENCE = ROOT.parent / 'conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/run_contract.json'


def initial(row):
    data = torch.load(row['trajectory'], map_location='cpu', weights_only=False, mmap=True)
    if data['capture_level'] != 'full' or int(data['replan_steps'][0]) != 0:
        raise ValueError('only the registered full first replan is authorized')
    obs = {k: v.clone() for k, v in data['observations'][0].items()}
    if set(obs) != {*IMAGE_KEYS, 'observation.language.tokens', 'observation.language.attention_mask'}:
        raise ValueError('saved processed observation interface changed')
    if any(obs[k].shape != (1, 3, 256, 256) for k in IMAGE_KEYS):
        raise ValueError('actual dual 256 own observations changed')
    return obs, data['states'][0].clone(), data['action_chunks'][0].clone(), int(data['policy_noise_seeds'][0])


def register():
    rows, scenes, inputs = [], {}, {}
    for arm in ('parent', 'F', 'G', 'R'):
        bank = read_json(OLD / 'banks' / arm / 'manifest.json')
        selected = [x for x in read_json(OLD / 'evaluation' / arm / 'results.json')['rows']
                    if x['occupancy_trajectory']['capture_level'] == 'full']
        if len(selected) != (8 if arm == 'parent' else 31):
            raise ValueError('registered 8/31/31/31 panel changed')
        for source in selected:
            case = source['binding_case']; key = f"task{case['task']:03d}_init{case['physical_init']:02d}_{case['layout']}"
            condition = next(c for c in bank['conditions'] if c['condition_id'] == source['operator_read_write_lora']['condition_id'])
            row = dict(arm=arm, **case, scene_key=key, trajectory=source['occupancy_trajectory']['path'],
                scene=source['scene_reference']['path'], scene_reference=source['scene_reference'],
                contract=str(OLD / 'evaluation' / arm / 'cases' / case['case_id'] / 'run_contract.json'),
                factors=condition['factors']['path'], condition_id=condition['condition_id'],
                binding=(str(OLD / 'banks/R' / (condition['condition_id'] + '_binding.pt')) if arm == 'R' else None),
                old_success=source['success'], old_steps=source['steps'], language=source['language'],
                suite=source['suite'], local_task_id=source['task_id'], env_seed=source['env_seed'],
                policy_seed_root=source['policy_seed_root'])
            obs, state, action, seed = initial(row)
            if seed != source['policy_noise_seeds'][0] or action.shape != (1, 50, 7):
                raise ValueError('actual seed or first full action chunk changed')
            row.update(noise_seed=seed, state8=state.reshape(-1).tolist(),
                       input_shapes={k: list(v.shape) for k, v in obs.items()})
            if key in inputs:
                previous = inputs[key]
                if any(not torch.equal(v, previous[k]) for k, v in obs.items()):
                    raise ValueError(f'same physical input is not actually paired: {key}')
            else:
                inputs[key] = obs
                scenes[key] = row
            rows.append(row)
    if len(rows) != 101 or len(scenes) != 25 or sum(x['arm'] == 'R' for x in rows) != 31:
        raise ValueError('fixed unique 101/25 identity changed')
    result = dict(schema='native_role_self_call_inputs_v1', rows=rows, scenes=list(scenes.values()),
        queries=101, unique_own_states=25, first_replan=0, flow_steps=10, output_slots=50,
        measured_slots=5, processed_inputs_reused=True, extra_preprocessing=False,
        held_teacher_privileged_read=False, model_forwards_so_far=0, environment_steps=0,
        old_bank_provenance=str(OLD / 'analysis/bank_execution_provenance.json'))
    path = ROOT / 'inputs.json'
    if path.exists() and read_json(path) != result:
        raise ValueError('existing registration has a different identity')
    write_json_atomic(path, result)
    return result


def object_groups(model, target):
    import mujoco
    roots = [int(model.jnt_bodyid[j]) for j in range(model.njnt)
             if model.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_FREE)]
    movable = sorted(model.body(b).name.removesuffix('_main') for b in roots
                     if 'robot' not in model.body(b).name)
    containers = [n for n in movable if 'basket' in n or 'tray' in n]
    # Include task recipient fixtures as passive containers, never rho competitors.
    containers += [model.body(b).name.removesuffix('_main') for b in range(1, model.nbody)
                   if ('basket' in model.body(b).name or 'tray' in model.body(b).name)
                   and int(model.body_parentid[b]) == 0]
    containers = sorted(set(containers))
    candidates = [target] + [n for n in movable if n != target and n not in containers]
    if target not in movable or len(candidates) != len(set(candidates)):
        raise ValueError(f'exact target or movable registry changed: {target}, {movable}')
    names = candidates + containers
    groups = []
    for name in names:
        roots = {i for i in range(1, model.nbody)
                 if model.body(i).name == name or model.body(i).name.startswith(name + '_')}
        members = set()
        for i in range(1, model.nbody):
            parent = i
            while parent and parent not in roots:
                parent = int(model.body_parentid[parent])
            if parent in roots:
                members.add(i)
        groups.append(sorted(members))
    return names, groups, len(candidates)


def render_masks(env, row):
    import mujoco
    from ember.pi05_eval.scene import _restore_scene
    scene = dict(np.load(row['scene']))
    observation = _restore_scene(env, scene)
    owner = env.env; sim = owner.sim
    # Restore the original sealed controller goals; never integrate physics.
    controller = owner.robots[0].controller
    controller.goal_pos = scene['controller_goal_pos'].copy()
    controller.goal_ori = scene['controller_goal_ori'].copy()
    if not np.allclose(controller.goal_pos, scene['controller_goal_pos'], atol=1e-8, rtol=0) or \
       not np.allclose(controller.goal_ori, scene['controller_goal_ori'], atol=1e-8, rtol=0):
        raise ValueError('canonical CPU controller restoration differs')
    model = mujoco.MjModel.from_xml_string(sim.model.get_xml())
    if list(scene['model_body_names']) != [model.body(i).name or 'world' for i in range(model.nbody)]:
        # robosuite's world name is normally 'world'; no heuristic body remapping.
        raise ValueError('actual model/body registry differs from the sealed scene')
    model.body_pos[:] = scene['model_body_pos']; model.body_quat[:] = scene['model_body_quat']
    data = mujoco.MjData(model); state = scene['sim_state']
    if model.na or len(state) != 1 + model.nq + model.nv:
        raise ValueError('sealed own simulation state layout changed')
    data.time = float(state[0]); data.qpos[:] = state[1:1 + model.nq]; data.qvel[:] = state[1 + model.nq:]
    mujoco.mj_forward(model, data)
    if row['layout'] == 'swapped':
        ids = [model.body(n + '_main').id for n in ('butter_1', 'orange_juice_1')]
        before = data.xpos[ids].copy()
        for i, body in enumerate(ids):
            joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body]
            if len(joints) != 1 or int(model.jnt_type[joints[0]]) != 0 or int(model.body_parentid[body]) != 0:
                raise ValueError('frozen world-XY free-joint transport contract changed')
            address = int(model.jnt_qposadr[joints[0]])
            data.qpos[address:address + 2] += before[1 - i, :2] - before[i, :2]
        mujoco.mj_forward(model, data)
        expected = before.copy(); expected[:, :2] = before[::-1, :2]
        if not np.allclose(data.xpos[ids], expected, atol=1e-9, rtol=0):
            raise ValueError('actual own world-XY swap differs')
    names, groups, n_candidates = object_groups(model, TARGETS[row['task']])
    renderer = mujoco.Renderer(model, 256, 256)
    option = mujoco.MjvOption(); option.geomgroup[0] = 0
    masks, rendered = [], []
    try:
        for camera in ('agentview', 'robot0_eye_in_hand'):
            renderer.disable_segmentation_rendering()
            renderer.update_scene(data, camera=camera, scene_option=option)
            rendered.append(renderer.render()[:, ::-1].copy())
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=camera, scene_option=option)
            segmentation = renderer.render()[:, ::-1]
            is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            bodies = np.full(is_geom.shape, -1, np.int32)
            bodies[is_geom] = model.geom_bodyid[segmentation[..., 0][is_geom]]
            masks.append(np.stack([np.isin(bodies, g) for g in groups]))
    finally:
        renderer.close()
    masks = np.stack(masks, 1)
    # Exactly the policy's square-image bilinear/align_corners=False mapping.
    resized = torch.nn.functional.interpolate(torch.from_numpy(masks.copy()).float().reshape(-1, 1, 256, 256),
                                              size=(224, 224), mode='bilinear', align_corners=False)
    f = resized.reshape(len(names), 2, 16, 14, 16, 14).mean((3, 5)).reshape(len(names), 512).numpy()
    q = f / np.maximum(f.sum(-1, keepdims=True), 1e-30)
    obs, raw_state, _, _ = initial(row)
    rgb = np.stack([(obs[k][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for k in IMAGE_KEYS])
    rgb_difference = np.stack(rendered).astype(float) - rgb.astype(float)
    path = ROOT / 'labels' / (row['scene_key'] + '.npz')
    np.savez_compressed(path, f=f, q=q, masks=masks, entities=np.asarray(names),
                        candidates=n_candidates, model_body_names=scene['model_body_names'],
                        qpos=data.qpos.copy(), qvel=data.qvel.copy(), time=data.time,
                        controller_goal_pos=scene['controller_goal_pos'], controller_goal_ori=scene['controller_goal_ori'])
    tiles = []
    colors = [(255, 0, 0), (0, 200, 255), (230, 200, 20), (150, 70, 240), (40, 230, 80), (255, 150, 40), (180, 180, 180)]
    for c in range(2):
        view = rgb[c].copy()
        for i, mask in enumerate(masks[:, c]):
            view[mask] = (.55 * view[mask] + .45 * np.asarray(colors[i % len(colors)])).astype(np.uint8)
        tiles.append(np.concatenate((rgb[c], view, rendered[c]), 1))
    image = np.concatenate(tiles, 0)
    cv2.imwrite(str(ROOT / 'analysis' / (row['scene_key'] + '_mask.png')), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return dict(scene_key=row['scene_key'], path=str(path), entities=names, candidates=n_candidates,
        visible=[bool(x.sum()) for x in f], valid_rho=bool(f[0].sum() and np.count_nonzero(f[:n_candidates].sum(-1)) > 1),
        target_index=0, target=TARGETS[row['task']], scene=row['scene'], source_trajectory=row['trajectory'],
        RGB_mean_abs_difference=float(np.abs(rgb_difference).mean()), RGB_RMS_difference=float(np.sqrt((rgb_difference ** 2).mean())),
        CPU_state_restore=True, controller_restored=True, extra_environment_steps=0,
        own_state8=raw_state.reshape(-1).tolist(), renderer='CPU OSMesa visual-geoms, top-down horizontally flipped')


def retained_mask(row):
    """Keep the four valid CPU masks preceding the task-metadata failure."""
    path = ROOT / 'labels' / (row['scene_key'] + '.npz')
    image_path = ROOT / 'analysis' / (row['scene_key'] + '_mask.png')
    if not path.is_file() or not image_path.is_file():
        return None
    data = np.load(path); names = data['entities'].tolist(); f = data['f']
    if names[0] != TARGETS[row['task']] or f.shape != (len(names), 512):
        raise ValueError('retained valid CPU mask identity changed')
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    rendered = np.stack([image[c * 256:(c + 1) * 256, 512:768] for c in range(2)])
    obs, raw_state, _, _ = initial(row)
    rgb = np.stack([(obs[k][0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for k in IMAGE_KEYS])
    difference = rendered.astype(float) - rgb.astype(float); n = int(data['candidates'])
    return dict(scene_key=row['scene_key'], path=str(path), entities=names, candidates=n,
        visible=[bool(x.sum()) for x in f], valid_rho=bool(f[0].sum() and np.count_nonzero(f[:n].sum(-1)) > 1),
        target_index=0, target=TARGETS[row['task']], scene=row['scene'], source_trajectory=row['trajectory'],
        RGB_mean_abs_difference=float(np.abs(difference).mean()), RGB_RMS_difference=float(np.sqrt((difference ** 2).mean())),
        CPU_state_restore=True, controller_restored=True, extra_environment_steps=0,
        own_state8=raw_state.reshape(-1).tolist(), renderer='retained legal OSMesa mask from e34e4c28; no new render')


def build():
    os.environ['MUJOCO_GL'] = 'osmesa'; os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import mujoco
    original_step1 = mujoco.mj_step1
    def kinematics_only(model, data):
        # LIBERO constructor uses step1 to prepare placement kinematics. It does
        # not integrate; guard the actual clock and generalized positions.
        clock, qpos, qvel = float(data.time), data.qpos.copy(), data.qvel.copy()
        original_step1(model, data)
        if float(data.time) != clock or not np.array_equal(data.qpos, qpos) or not np.array_equal(data.qvel, qvel):
            raise RuntimeError('constructor step1 unexpectedly integrated physics')
    mujoco.mj_step1 = kinematics_only
    def forbidden_step(*args, **kwargs):
        raise RuntimeError('§134 CPU label restoration permits zero physics steps')
    for name in ('mj_step', 'mj_step2'):
        setattr(mujoco, name, forbidden_step)
    import ember.pi05_evaluation
    from ember.pi05_assets import prepare_libero_config, configure_libero_runtime_assets
    paths = prepare_libero_config(ROOT / 'cache/libero_config')
    configure_libero_runtime_assets(Path(paths['assets']))
    from libero.libero.envs import OffScreenRenderEnv
    manifest = read_json(ROOT / 'inputs.json'); records = []; env = None; task_id = None; started = time.monotonic()
    try:
        for row in sorted(manifest['scenes'], key=lambda r: (r['task'], r['physical_init'], r['layout'])):
            retained = retained_mask(row)
            if retained is not None:
                records.append(retained); continue
            if row['task'] != task_id:
                if env is not None:
                    env.close(); env = None
                contract = read_json(HELD_REFERENCE if row['task'] == 16 else Path(row['contract']))
                task = next(t for t in contract['tasks'] if (t['suite'], t['task_id']) == (row['suite'], row['local_task_id']))
                bddl = Path(contract['libero_paths']['bddl_files']) / task['problem_folder'] / task['bddl_file']
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
                env.reset(); task_id = row['task']
            records.append(render_masks(env, row))
    finally:
        if env is not None:
            env.close()
    write_json_atomic(ROOT / 'labels/manifest.json', dict(records=records, unique_states=25,
        environment_steps=0, policy_forwards=0, held_teacher_privileged_read=False, seconds=time.monotonic() - started))
    print({'completed_masks': len(records), 'environment_steps': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--registration-only', action='store_true')
    if parser.parse_args().registration_only:
        register()
    else:
        build()
