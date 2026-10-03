"""Fixed-role labels from existing query points and eight teacher kinematic snapshots."""
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

ROOT = Path('/data1/user/ymdai/ember_runs/role_coordinate_credit_20261004')
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


def extract():
    """CPU only: no action reads, rendering or simulation steps."""
    import torch
    from ember.pi05_eval_contract import git_state
    started = time.time()
    spec = read_json(ASSET/'configs/operator_read_write_v1/conditional_read_write_continuation900_spec.json')
    tasks = load_learning_tasks(ASSET, tuple(POINTS), protocol_path=spec['source']['data_protocol'])
    manifest = read_json(OLD/'query_manifest.json')
    panels = read_json(Path(manifest['panel_source']))
    old_labels = ROOT.parent/'joint_action_effect_credit_20261003/labels/query_positions.pt'
    positions = torch.load(old_labels, map_location='cpu', weights_only=False)
    query_labels, teacher_labels, records, videos = {}, {}, [], []
    requested = {task:set() for task in POINTS}
    for entry in manifest['steps']:
        for task in POINTS:
            requested[task].update((q['demo'],q['frame']) for q in entry[str(task)]['queries'])
    for task in POINTS:
        requested[task].update((q['demo'],q['frame']) for q in panels[str(task)]['B'])
        if set(TEACHERS[task]) & {d for d,f in requested[task]}:
            raise ValueError('teacher/query episode wall violated')
        with h5py.File(tasks[task].authority.path,'r') as file:
            for demo_id in sorted({d for d,f in requested[task]}):
                bases = np.asarray(sorted(f for d,f in requested[task] if d==demo_id),dtype=np.int64)
                old = positions[f'{task}:{demo_id}']
                indices = np.searchsorted(old['frames'],bases)
                if not np.array_equal(old['frames'][indices],bases):
                    raise ValueError('required existing query points absent')
                eef = file[f'data/demo_{demo_id}/obs/ee_states'][bases,:3]
                point = old['positions_m'][indices]
                query_labels[f'{task}:{demo_id}'] = dict(frames=bases,point_m=point,eef_m=eef,
                    relative_m=point-eef,episode_length=old['episode_length'])
            for teacher in TEACHERS[task]:
                demo = file[f'data/demo_{teacher}']
                raw = len(demo['obs/agentview_rgb'])
                sampled = list(range(0,raw,5))
                if sampled[-1]!=raw-1:sampled.append(raw-1)
                frames,point,info = recover(demo,task,sampled[:-1])
                eef = demo['obs/ee_states'][frames,:3]
                teacher_labels[f'{task}:{teacher}'] = dict(frames=frames,point_m=point,eef_m=eef,
                    relative_m=point-eef)
                records.append(dict(task=task,teacher=teacher,hdf5=str(tasks[task].authority.path),
                    raw_frames=raw,sampled_frames=len(sampled),**info))
                videos.append(dict(task=task,teacher=teacher,sampled_frames=len(sampled)))
    val = load_learning_tasks(ASSET,(16,),role='validation',protocol_path=spec['source']['data_protocol'])
    with h5py.File(val[16].authority.path,'r') as file:
        for teacher in (47,33,28,46,32,1,24,43):
            raw=len(file[f'data/demo_{teacher}/obs/agentview_rgb'])
            n=len(range(0,raw,5))+(int((raw-1)%5!=0))
            videos.append(dict(task=16,teacher=teacher,raw_frames=raw,sampled_frames=n,
                privileged_fields_read=False,cache_persisted=False))
    if manifest['query_offset']!=1 or len(manifest['steps'])!=64:
        raise ValueError('registered fixed64 query stream changed')
    torch.save(dict(queries=query_labels,teachers=teacher_labels),ROOT/'labels/role_labels.pt')
    # X/H preserve actual native dtype; c is float32 and d uses the interpreter's dtype.
    frames=sum(v['sampled_frames'] for v in videos if v['task']!=16)
    x_width=sum([1024]*36+[32,1024])
    native_upper=frames*50*(x_width*4+1024*4*3)
    summary=dict(schema='ember_role_coordinate_labels_v1',source_restore_git='7bd8a6f5',
        label_git=git_state(Path(__file__).resolve().parents[3]),records=records,videos=videos,
        query_point_source=str(old_labels),query_point_manifest=str(old_labels.with_name('manifest.json')),
        query_EEF_source='original obs/ee_states[f,:3]',point_scale_m=1,teacher_actions_read=False,
        held_privileged_read=False,environment_steps=0,render=False,seconds=time.time()-started,
        teacher_origin_labels=sum(len(x['frames']) for x in teacher_labels.values()),
        unique_query_geometry_labels=sum(len(x['frames']) for x in query_labels.values()),
        storage_peak_estimate_GiB=dict(fixed_native_upper=native_upper/2**30,
            two_full_writer_and_optimizer=3.5,banks=1.3,trajectories=2.2,code_and_temporary=1.5,
            bound=16,hard=20))
    write_json_atomic(ROOT/'labels/manifest.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    extract()

