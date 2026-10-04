"""CPU-only registered §132 frames and visible entity labels; no actions read."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import cv2
import h5py
import mujoco
import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic

ROOT = Path('/data1/user/ymdai/ember_runs/native_role_binding_compilation_20261004')
ASSET = Path('/data1/user/ymdai/projects/EMBER')
ASSETS = ASSET / 'data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
TARGETS = {12: 'salad_dressing_1', 13: 'bbq_sauce_1', 14: 'ketchup_1',
           15: 'tomato_sauce_1', 17: 'milk_1', 19: 'orange_juice_1',
           43: 'butter_2', 96: 'butter_1'}
CAMERAS = ('agentview', 'robot0_eye_in_hand')


def frames(length):
    result = list(range(0, length, 5))
    if result[-1] != length - 1:
        result.append(length - 1)
    return result


def manifest():
    metadata = json.loads((ROOT / 'launch/task_metadata.json').read_text())
    lengths = {r['task']: r['lengths'] for r in metadata}
    steps, readback = [], {}
    def query(task, demo, keys, flow_keys):
        rng = np.random.default_rng(np.random.SeedSequence(keys))
        seed = int(np.random.SeedSequence(flow_keys).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)
        return dict(demo=demo, frame=int(rng.integers(0, lengths[task][demo] - 1)), flow_seed=seed)
    for step in range(64):
        order = np.random.default_rng(np.random.SeedSequence([7, 132, step // 2])).permutation(sorted(TARGETS))
        selected = [int(t) for t in order[(step % 2) * 4:(step % 2 + 1) * 4]]
        events = {str(t): dict(teachers=[0, 1], queries=[query(t, d, [7, 132, t, step, d, 0],
                      [7, 132, t, step, d, 1]) for d in range(2, 30)]) for t in selected}
        steps.append(dict(update=step + 1, tasks=selected, events=events))
    for t in TARGETS:
        readback[str(t)] = [query(t, d, [7, 132, t, d, 2], [7, 132, t, d, 3]) for d in range(30, 50)]
    result = dict(schema='native_role_binding_queries_v1', query_offset=1, seed=7, steps=steps,
                  B20=readback, condition_weight=1 / 8, queries_per_condition=28,
                  main_condition_query_uses_per_arm=14336, teacher_pool=[0, 1], query_pool=[2, 29])
    path = ROOT / 'query_manifest.json'
    if path.exists() and read_json(path) != result:
        raise ValueError('registered query stream already exists with a different identity')
    write_json_atomic(path, result)
    return result


def model_from_demo(demo):
    import importlib.util
    robosuite = Path(importlib.util.find_spec('robosuite').origin).parent
    xml = ET.fromstring(demo.attrs['model_file'])
    for node in xml.findall('./asset/*'):
        name = node.get('file')
        if name:
            if '/robosuite/' in name:
                path = robosuite / name.split('/robosuite/', 1)[1]
            elif '/assets/' in name:
                path = ASSETS / name.split('/assets/', 1)[1]
            else:
                raise ValueError(f'unmapped original XML asset: {name}')
            if not path.is_file():
                raise FileNotFoundError(path)
            node.set('file', str(path))
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))


def registry(model, target):
    # Movable free-joint entities, excluding the task recipient container and robot.
    roots = [int(model.jnt_bodyid[j]) for j in range(model.njnt)
             if model.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_FREE)]
    names = [model.body(b).name.removesuffix('_main') for b in roots]
    names = sorted(n for n in names if not any(s in n for s in ('robot', 'basket', 'tray')))
    if target not in names or len(names) != len(set(names)):
        raise ValueError(f'exact target entity or movable registry missing: {target}, {names}')
    names = [target] + [n for n in names if n != target]
    groups = []
    for name in names:
        body_roots = {i for i in range(1, model.nbody)
                      if model.body(i).name == name or model.body(i).name.startswith(name + '_')}
        members = set(body_roots)
        for i in range(1, model.nbody):
            parent = i
            while parent and parent not in body_roots:
                parent = int(model.body_parentid[parent])
            if parent in body_roots:
                members.add(i)
        groups.append(sorted(members))
    return names, groups


def patch_weights(body_images, groups):
    weights = []
    for group in groups:
        # MuJoCo's top-down renderer is horizontally flipped here: the original
        # robosuite stored RGB is vertically flipped, then policy rotates 180°.
        mask = np.isin(body_images, group).astype(np.float32)
        grids = [cv2.resize(view, (224, 224), interpolation=cv2.INTER_LINEAR)
                 .reshape(16, 14, 16, 14).mean((1, 3)).reshape(256) for view in mask]
        q = np.concatenate(grids)
        if q.sum() > 0:
            q /= q.sum()
        weights.append(q)
    return np.asarray(weights, np.float32)


def episode(job):
    row, demo_id, requested, overlay = job
    task, target = row['task'], TARGETS[row['task']]
    with h5py.File(row['path'], 'r') as handle:
        demo = handle[f'data/demo_{demo_id}']
        model = model_from_demo(demo)
        if model.na or model.opt.timestep != .002 or demo['states'].shape[1] != 1 + model.nq + model.nv:
            raise ValueError('verified post-action 2ms stored-state contract changed')
        names, groups = registry(model, target)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 128, 128)
        renderer.enable_segmentation_rendering()
        option = mujoco.MjvOption(); option.geomgroup[0] = 0
        eef = [i for i in range(model.nsite) if model.site(i).name.endswith('grip_site')]
        if len(eef) != 1:
            raise ValueError('original single EEF site mapping changed')
        retained = ROOT / 'labels/registration' / f'task{task:03d}_demo{demo_id:02d}.npz'
        cached = np.load(retained)['q'] if overlay and requested == [0] and retained.is_file() else None
        results, errors, supported = [], [], []
        try:
            for frame in requested:
                valid = frame + 1 < len(demo['states'])
                supported.append(valid)
                if not valid:
                    results.append(np.zeros((len(names), 512), np.float32)); continue
                state = demo['states'][frame + 1]
                data.time = float(state[0]) - .002
                data.qpos[:] = state[1:1 + model.nq]; data.qvel[:] = state[1 + model.nq:]
                mujoco.mj_integratePos(model, data.qpos, data.qvel, -.002)
                mujoco.mj_forward(model, data)
                errors.append(float(np.linalg.norm(data.site_xpos[eef[0]] - demo['obs/ee_states'][frame, :3])))
                if cached is not None:
                    results.append(cached[len(results)]); continue
                images = []
                for camera in CAMERAS:
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    segmentation = renderer.render()[:, ::-1]
                    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                    image = np.full(is_geom.shape, -1, np.int32)
                    image[is_geom] = model.geom_bodyid[segmentation[..., 0][is_geom]]
                    images.append(image)
                images = np.stack(images)
                results.append(patch_weights(images, groups))
                if overlay and frame == 0:
                    tiles = []
                    for i, camera in enumerate(('agentview_rgb', 'eye_in_hand_rgb')):
                        rgb = np.ascontiguousarray(demo['obs/' + camera][0][::-1, ::-1])
                        mask = np.isin(images[i], groups[0])
                        rgb[mask] = (.55 * rgb[mask] + .45 * np.array([255, 0, 0])).astype(np.uint8)
                        tiles.append(rgb)
                    image_path = ROOT / 'analysis' / f'task{task:03d}_teacher{demo_id}_visible_mask.png'
                    cv2.imwrite(str(image_path), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
        finally:
            renderer.close()
    if errors and max(errors) > 1e-4:
        raise ValueError(f'physical observation alignment failed task{task} demo{demo_id}: {max(errors)}')
    path = ROOT / 'labels' / ('registration' if len(requested) == 1 and requested[0] == 0 and overlay else 'data') / f'task{task:03d}_demo{demo_id:02d}.npz'
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, frames=np.asarray(requested), q=np.asarray(results), supported=np.asarray(supported))
    return dict(task=task, demo=demo_id, path=str(path), entities=names, target_index=0,
                frames=len(requested), visible_target_frames=int(sum(q[0].sum() > 0 for q in results)),
                EEF_alignment_max_m=max(errors, default=0), model_bodies=[[model.body(i).name for i in g] for g in groups],
                environment_steps=0, action_values_read=False, label_state='states[f+1] undo negative2ms')


def build(verify_only=False):
    started = time.monotonic()
    m = manifest()
    metadata = json.loads((ROOT / 'launch/task_metadata.json').read_text())
    requests = {(row['task'], d): set() for row in metadata for d in range(50)}
    for step in m['steps']:
        for key, event in step['events'].items():
            for q in event['queries']:
                requests[int(key), q['demo']].add(q['frame'])
    for row in metadata:
        t = row['task']
        for d in (0, 1):
            requests[t, d].update(frames(row['lengths'][d])[:-1])
        for q in m['B20'][str(t)]:
            requests[t, q['demo']].add(q['frame'])
    jobs = [(r, d, [0] if verify_only else sorted(requests[r['task'], d]), d == 0)
            for r in metadata for d in ([0] if verify_only else range(50)) if requests[r['task'], d]]
    with ProcessPoolExecutor(max_workers=4 if verify_only else 8,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(episode, jobs))
    result = dict(schema='native_role_binding_visible_labels_v1', records=rows,
                  completed='first_frame_consumer_check' if verify_only else 'complete',
                  CPU_renderer='OSMesa', gradients=False, held_privileged_read=False,
                  environment_steps=0, teacher_mask_frame='origin t-1', queries_offset=1,
                  source_segmentation_git='ca87f05c', source_state_git='7bd8a6f5',
                  actual_frames=sum(r['frames'] for r in rows), seconds=time.monotonic() - started)
    write_json_atomic(ROOT / 'labels' / ('first_frame_check.json' if verify_only else 'manifest.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    build(parser.parse_args().verify_only)
