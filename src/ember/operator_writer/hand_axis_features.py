"""Temporary passive H/block17 KV consumer for the fixed hand-axis diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .data import FormalData, TASKS
from .native import _NativeFrameCall
from .prefix_change import NativeAttentionCapture, rms
from .specification import CONTINUATION2340_SPEC_PATH, specification

ROOT = Path('/data1/user/ymdai/ember_runs/native_hand_axis_readout_20261006')
PARENT = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340')
ASSETS = Path('/data1/user/ymdai/projects/EMBER')
HELDOUT = (29, 34, 38, 73)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def manifest():
    """Source metadata and RGB lengths only; no teacher pose/action read."""
    data = FormalData(ASSETS, specification(CONTINUATION2340_SPEC_PATH), query_labels=False)
    rows, offset = [], 0
    for role, demos in (('fit', (16, 17)), ('eval', (42, 43))):
        for task in TASKS:
            if role == 'fit' and task in HELDOUT:
                continue
            authority = data.tasks[task].authority
            for demo in demos:
                count, sampled = data.videos.frame_counts(task, demo)
                indices = list(range(0, count, 5))
                if indices[-1] != count - 1:
                    indices.append(count - 1)
                if sampled != len(indices):
                    raise ValueError('canonical RGB stride/final-frame counts disagree')
                rows.append(dict(key=f't{task:03d}_d{demo:02d}', task=task, demo=demo,
                    role=role, nframes=sampled, raw_frames=count, raw_indices=indices,
                    global_start=offset, hdf5=str(authority.path), language=authority.language,
                    suite=data.tasks[task].suite, suite_task_id=data.tasks[task].suite_task_id))
                offset += sampled
    counts = {role: sum(r['nframes'] for r in rows if r['role'] == role) for role in ('fit', 'eval')}
    if len(rows) != 136 or counts != {'fit': 1940, 'eval': 2442} or offset != 4382:
        raise ValueError(f'registered source frame census changed: {len(rows)}, {counts}')
    result = dict(schema='native_hand_axis_frames_v1', clips=rows, frames=offset,
        counts=counts, tasks=list(TASKS), fit_tasks=[t for t in TASKS if t not in HELDOUT],
        heldout_tasks=list(HELDOUT), parent=str(PARENT), asset_root=str(ASSETS),
        dataset_revision='f13aa24a3da8c43c7225569f28c562979fa0e35a',
        extraction=dict(probe_seed=1729, tau=1, stride=5, final_frame=True, cameras='dual',
                        official_rotation=180, model_image=224, H=[50, 1024], KV=[512, 512],
                        block=17, H_dtype='float32', KV_dtype='native', response_difference=False))
    write_json(ROOT / 'manifest.json', result)
    data.close()
    return result


class VisionKV(NativeAttentionCapture):
    """Original actual observer checks; retain only rotated image K and V."""
    def outputs(self):
        _, key, value, *_ = super().outputs()
        if key.shape[1] != 1 or key.shape[2] < 512 or key.shape[-1] != 256:
            raise ValueError('actual dual-image native KV geometry changed')
        return key[:, 0, :512].contiguous(), value[:, 0, :512].contiguous()


def runtime():
    from .run import build_runtime, frozen_git
    git = frozen_git(continuation=True)
    device = torch.device('cuda:0')
    result = build_runtime(ASSETS, specification(CONTINUATION2340_SPEC_PATH), device, 'T')
    result.writer.load_state_dict(load_file(str(PARENT / 'ecp.safetensors'), device='cuda:0'), strict=True)
    result.writer.eval().requires_grad_(False)
    result.policy.eval().requires_grad_(False)
    if len(result.writer.public_state()) != 76 or tuple(result.writer.probe.shape) != (50, 32):
        raise ValueError('frozen complete beta/probe contract changed')
    return result, git


def extract(runtime, condition, chunk):
    from ember.writer.runtime import autocast
    frames, indices, tokens, mask = condition
    public = {'policy.' + k: v for k, v in runtime.writer.public_state().items()}
    outputs = []
    for start in range(0, len(frames), chunk):
        observer = VisionKV(runtime.policy.model.paligemma_with_expert)
        wrapper = _NativeFrameCall(runtime.policy, (), runtime.writer.probe, observer=observer)
        with torch.no_grad(), autocast(runtime.device):
            h, k, v = torch.func.functional_call(wrapper, public,
                (frames[start:start + chunk], tokens, mask), strict=False)
            outputs.append({'H': rms(h).cpu(), 'K': k.cpu(), 'V': v.cpu()})
    result = {key: torch.cat([r[key] for r in outputs]).contiguous() for key in ('H', 'K', 'V')}
    if (result['H'].shape != (len(frames), 50, 1024)
            or any(result[k].shape != (len(frames), 512, 256) for k in ('K', 'V'))
            or any(not torch.isfinite(v).all() for v in result.values())):
        raise ValueError('native retained features nonfinite or wrong shape')
    return result


def profile():
    torch.set_num_threads(6)
    state = json.loads((ROOT / 'manifest.json').read_text())
    selected = max(state['clips'], key=lambda r: r['nframes'])
    model, git = runtime()
    data = FormalData(ASSETS, specification(CONTINUATION2340_SPEC_PATH), query_labels=False)
    condition, _, _ = data.condition(model, selected['task'], selected['demo'])
    profiles = []
    for chunk in (16, 32, 64, 128):
        if profiles and chunk // 2 >= selected['nframes']:
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = extract(model, condition, chunk)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        profiles.append(dict(frame_chunk=chunk, actual_max_frames=min(chunk, len(condition[0])),
            frames=len(condition[0]), seconds=seconds, frames_per_second=len(condition[0])/seconds,
            allocated_peak_bytes=torch.cuda.max_memory_allocated(),
            reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            dtypes={k: str(v.dtype) for k, v in output.items()}))
        del output
    selected_chunk = max(profiles, key=lambda r: r['frames_per_second'])['frame_chunk']
    write_json(ROOT / 'launch/native_profile.json', dict(clip=selected, git=git, profiles=profiles,
        selected_chunk=selected_chunk, selection='highest measured frames/s',
        stopped_larger_reason='full registered longest clip packed or maximum tested128'))
    data.close()
    print(json.dumps(profiles), flush=True)


def worker(chunk, worker_id):
    torch.set_num_threads(6)
    state = json.loads((ROOT / 'manifest.json').read_text())
    model, git = runtime()
    data = FormalData(ASSETS, specification(CONTINUATION2340_SPEC_PATH), query_labels=False)
    feature_dir = ROOT / 'features'
    claims = ROOT / 'launch/claims'
    feature_dir.mkdir(exist_ok=True)
    claims.mkdir(exist_ok=True)
    rows = []
    for row in sorted(state['clips'], key=lambda r: (-r['nframes'], r['key'])):
        destination = feature_dir / (row['key'] + '.safetensors')
        if destination.exists():
            continue
        claim = claims / row['key']
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        started = time.perf_counter()
        condition, raw, count = data.condition(model, row['task'], row['demo'])
        if raw != row['raw_frames'] or count != row['nframes'] or condition[1].cpu().tolist() != row['raw_indices']:
            raise ValueError('actual canonical RGB frame identity disagrees with manifest')
        values = extract(model, condition, chunk)
        temporary = destination.with_suffix('.partial')
        save_file(values, str(temporary), metadata={'clip':row['key'], 'git':git['commit'],
            'parent':str(PARENT), 'one_native_forward_per_frame':'true',
            'no_response_difference':'true', 'K_V_order':'rotated_prefix_first512_then_actual_V'})
        temporary.replace(destination)
        record = dict(key=row['key'], frames=count, seconds=time.perf_counter()-started,
            bytes=destination.stat().st_size, dtypes={k:str(v.dtype) for k,v in values.items()},
            frame_chunk=chunk, worker=worker_id, git=git)
        write_json(feature_dir / (row['key'] + '.json'), record)
        rows.append(record)
        del values, condition
        # Keep the claim after publication: a late concurrent check cannot
        # acquire it between the destination check and directory creation.
    data.close()
    write_json(ROOT / 'launch' / f'extract_worker_{worker_id}.json', dict(rows=rows,
        git=git, allocated_peak_bytes=torch.cuda.max_memory_allocated(),
        reserved_peak_bytes=torch.cuda.max_memory_reserved()))


def launch_workers(chunk):
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    processes = []
    for index, gpu in enumerate(devices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log = (ROOT / 'launch' / f'extract_worker_{index}.log').open('w')
        processes.append((subprocess.Popen([os.sys.executable, '-m',
            'ember.operator_writer.hand_axis_features', 'worker',
            '--chunk', str(chunk), '--worker-id', str(index)], env=env, stdout=log,
            stderr=subprocess.STDOUT), log))
    codes = []
    try:
        for process, log in processes:
            codes.append(process.wait())
            log.close()
    finally:
        for process, log in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    if any(codes):
        raise RuntimeError(f'native worker exit codes {codes}')
    state = json.loads((ROOT / 'manifest.json').read_text())
    records = [json.loads((ROOT / 'features' / (r['key'] + '.json')).read_text()) for r in state['clips']]
    if len(records) != 136 or sum(r['frames'] for r in records) != 4382:
        raise ValueError('incomplete authorized feature pool')
    write_json(ROOT / 'features/completion.json', dict(clips=136, frames=4382,
        bytes=sum(r['bytes'] for r in records), native_dtype_records=records,
        label_read_during_extraction=False, worker_exit_codes=codes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['manifest', 'profile', 'worker', 'extract'])
    parser.add_argument('--chunk', type=int, default=32)
    parser.add_argument('--worker-id', default='0')
    args = parser.parse_args()
    if args.phase == 'manifest':
        manifest()
    elif args.phase == 'profile':
        profile()
    elif args.phase == 'worker':
        worker(args.chunk, args.worker_id)
    else:
        launch_workers(args.chunk)


if __name__ == '__main__':
    main()
