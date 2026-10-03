"""Task-only existing-query kinematics for the registered joint-effect diagnostic."""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robosuite

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.learning_data import load_learning_tasks

ROOT = Path('/data1/user/ymdai/ember_runs/joint_action_effect_credit_20261003')
ASSET = Path('/data1/user/ymdai/projects/EMBER')
OLD = ROOT.parent / 'operator_learning_limit_diagnosis_20260930'
ASSETS = ASSET / 'data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
POINTS = {0: ('body', 'akita_black_bowl_1_main'),
          12: ('body', 'salad_dressing_1_main'),
          20: ('site', 'wooden_cabinet_1_middle_region'),
          32: ('body', 'moka_pot_1_main')}
TEACHERS = {0: (40, 11), 12: (25, 14), 20: (38, 42), 32: (17, 43)}


def model_from_demo(demo):
    xml = ET.fromstring(demo.attrs['model_file'])
    suite_root = Path(robosuite.__file__).parent
    for node in xml.findall('./asset/*'):
        filename = node.get('file')
        if not filename:
            continue
        if '/robosuite/' in filename:
            path = suite_root / filename.split('/robosuite/', 1)[1]
        elif '/assets/' in filename:
            path = ASSETS / filename.split('/assets/', 1)[1]
        else:
            raise ValueError(f'unmapped original XML asset {filename}')
        if not path.is_file():
            raise FileNotFoundError(path)
        node.set('file', str(path))
    return mujoco.MjModel.from_xml_string(ET.tostring(xml).decode())


def recover(demo, task, frames):
    """7bd8a6f5 post-action cache: states[i+1], undo one 2ms position step."""
    model = model_from_demo(demo)
    if model.na or demo['states'].shape[1] != 1 + model.nq + model.nv or model.opt.timestep != .002:
        raise ValueError('old stored-state integration contract changed')
    kind, name = POINTS[task]
    obj = mujoco.mjtObj.mjOBJ_BODY if kind == 'body' else mujoco.mjtObj.mjOBJ_SITE
    point_id = mujoco.mj_name2id(model, obj, name)
    if point_id < 0:
        raise ValueError(f'exact physical point absent: task{task} {kind} {name}')
    eef_ids = [i for i in range(model.nsite)
               if str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)).endswith('grip_site')]
    if len(eef_ids) != 1:
        raise ValueError('original single gripper site mapping changed')
    frames = np.asarray(sorted(set(frames)), dtype=np.int64)
    if not len(frames) or np.any(frames < 0) or np.any(frames + 1 >= len(demo['states'])):
        raise ValueError('point recovery must use actual available post-action states')
    stored = demo['states'][frames + 1]
    expected_eef = demo['obs/ee_states'][frames, :3]
    data = mujoco.MjData(model)
    positions, errors = [], []
    for state, expected in zip(stored, expected_eef, strict=True):
        data.time = float(state[0]) - model.opt.timestep
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
        mujoco.mj_forward(model, data)
        positions.append((data.xpos if kind == 'body' else data.site_xpos)[point_id].copy())
        errors.append(float(np.linalg.norm(data.site_xpos[eef_ids[0]] - expected)))
    if max(errors) > 1e-4 or not np.isfinite(positions).all():
        raise ValueError(f'post-action observation alignment failed task{task}: EEF max={max(errors)}')
    parent = point_id if kind == 'body' else int(model.site_bodyid[point_id])
    return frames, np.asarray(positions, dtype=np.float64), {
        'point_kind': kind, 'point_name': name, 'point_id': point_id,
        'parent_body': mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
        'EEF_alignment_max_m': max(errors), 'EEF_alignment_RMS_m': float(np.sqrt(np.mean(np.square(errors)))),
        'frames': len(frames), 'timestep_s': model.opt.timestep,
        'state_correspondence': 'obs[i] = states[i+1] with one negative 2ms mj_integratePos cache step',
        'environment_steps': 0, 'render': False,
    }


def registered_queries():
    manifest = read_json(OLD / 'query_manifest.json')
    panels = read_json(Path(manifest['panel_source']))
    if manifest['query_offset'] != 1 or len(manifest['steps']) != 64:
        raise ValueError('fixed learning manifest changed')
    queries = {t: set() for t in POINTS}
    for step in manifest['steps']:
        for task in POINTS:
            rows = step[str(task)]['queries']
            if len(rows) != 28:
                raise ValueError('registered logical query batch changed')
            queries[task].update((int(q['demo']), int(q['frame'])) for q in rows)
    for task in POINTS:
        panel = panels[str(task)]
        if tuple(panel['teachers']) != TEACHERS[task] or len(panel['A']) != 28 or len(panel['B']) != 20:
            raise ValueError('A28/B20 panel changed')
        queries[task].update((int(q['demo']), int(q['frame'])) for group in ('A', 'B') for q in panel[group])
        if set(TEACHERS[task]) & {demo for demo, _ in queries[task]}:
            raise ValueError('teacher privileged state entered query labels')
    return manifest, panels, queries


def extract(*, preflight=False):
    started = time.time()
    manifest, panels, queries = registered_queries()
    spec = read_json(ASSET / 'configs/operator_read_write_v1/conditional_read_write_continuation900_spec.json')
    tasks = load_learning_tasks(ASSET, tuple(POINTS), protocol_path=spec['source']['data_protocol'])
    output, records = {}, []
    for task, row in tasks.items():
        requested = queries[task]
        if preflight:
            q = manifest['steps'][0][str(task)]['queries'][0]
            requested = {(int(q['demo']), int(q['frame']))}
        with h5py.File(row.authority.path, 'r') as handle:
            for demo_id in sorted({d for d, _ in requested}):
                demo = handle[f'data/demo_{demo_id}']
                length = len(demo['actions'])
                available = min(length, len(demo['states']) - 1, len(demo['obs/ee_states']))
                bases = sorted(f for d, f in requested if d == demo_id)
                needed = {i for f in bases for i in range(f, min(f + 51, available))}
                frames, positions, info = recover(demo, task, needed)
                output[f'{task}:{demo_id}'] = {'frames': frames, 'positions_m': positions,
                                              'episode_length': length, 'available_positions': available}
                info.update(task=task, demo=demo_id, hdf5=str(row.authority.path),
                            base_queries=len(bases), episode_length=length, available_positions=available,
                            state_dataset=f'data/demo_{demo_id}/states', point_label_scale_m=.1)
                records.append(info)
    ROOT.joinpath('labels').mkdir(exist_ok=True)
    summary = dict(schema='ember_joint_effect_kinematics_v1', scope='preflight' if preflight else 'fixed_A28_B20',
                   source_restore_git='7bd8a6f5', records=records, seconds=time.time()-started,
                   point_table={str(k): list(v) for k, v in POINTS.items()}, new_data=False, GPUh=0)
    if not preflight:
        import torch
        torch.save(output, ROOT / 'labels/query_positions.pt')
    write_json_atomic(ROOT / f'labels/{"preflight" if preflight else "manifest"}.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    extract(preflight=parser.parse_args().preflight)
