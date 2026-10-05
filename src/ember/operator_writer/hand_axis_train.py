"""Fixed 500-update detached-feature heads; one terminal readout, no policy work."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

from .hand_axis_features import ROOT, PARENT, write_json


def load_manifest():
    return json.loads((ROOT / 'manifest.json').read_text())


def prepare_labels(*, evaluation=False):
    """Only runs after all action-hidden native feature consumers exit."""
    from .hand_axis_probe import axis_labels
    completion = json.loads((ROOT / 'features/completion.json').read_text())
    if completion['frames'] != 4382 or completion['clips'] != 136:
        raise ValueError('labels cannot precede complete native extraction')
    state = load_manifest()
    arrays, sources = {}, []
    if evaluation:
        for arm in ('H', 'KV'):
            done = json.loads((ROOT/'heads'/arm/'completion.json').read_text())
            if done['updates'] != 500:
                raise ValueError('evaluation labels require both fixed500 endpoints')
        with np.load(ROOT/'labels.npz') as store:
            arrays = {key:store[key] for key in store.files}
        sources = json.loads((ROOT/'label_sources.json').read_text())['sources']
    for clip in state['clips']:
        if clip['role'] != ('eval' if evaluation else 'fit'):
            continue
        indices = np.asarray(clip['raw_indices'], dtype=np.int64)
        with h5py.File(clip['hdf5'], 'r') as handle:
            demo = handle[f'data/demo_{clip["demo"]}']
            orientation = np.asarray(demo['obs/ee_ori'][indices], dtype=np.float64)
            commands = np.asarray(demo['actions'][:, 6])
        positive = np.flatnonzero(commands > 0)
        first = int(positive[0]) if len(positive) else None
        before = indices < first if first is not None and first > 0 else np.zeros(len(indices), dtype=bool)
        z = axis_labels(orientation).astype(np.float32)
        if z.shape != (clip['nframes'], 3) or not np.isfinite(z).all():
            raise ValueError('same-obs hand-axis labels invalid')
        arrays[clip['key'] + '_z'] = z
        arrays[clip['key'] + '_pre_positive'] = before
        sources.append(dict(key=clip['key'], task=clip['task'], demo=clip['demo'], hdf5=clip['hdf5'],
            raw_indices=clip['raw_indices'], orientation_field=f'data/demo_{clip["demo"]}/obs/ee_ori',
            axis='world hand/site +Z = Rodrigues(ee_ori) @ [0,0,1]', same_obs_offset=0,
            command_field=f'data/demo_{clip["demo"]}/actions[:,6]', first_positive_command=first,
            command_use='passive grouping only, never native/head input or sampling',
            head_role=clip['role'], pre_positive_count=int(before.sum())))
    np.savez(ROOT / 'labels.npz', **arrays)
    frame_count = sum(len(value) for key,value in arrays.items() if key.endswith('_z'))
    write_json(ROOT / 'label_sources.json', dict(sources=sources, frames=frame_count,
        read_after_native_completion=True, official_validation_test=False,
        evaluation_labels_read_only_after_both500=evaluation,
        privileged_fields_read=['authorized_train.obs/ee_ori', 'authorized_train.actions[:,6]_passive_only']))
    if evaluation:
        return
    rng = np.random.default_rng(20261006)
    events = []
    snapshots = {}
    lookup = {(c['task'], c['demo']):c for c in state['clips'] if c['role'] == 'fit'}
    for update in range(500):
        rows = []
        for task in state['fit_tasks']:
            for _ in range(16):
                clip = lookup[(task, int(rng.choice([16, 17])))]
                frame = int(rng.integers(clip['nframes']))
                rows.append(clip['global_start'] + frame)
        events.append(rows)
        if update + 1 in (250, 500):
            snapshots[str(update + 1)] = rng.bit_generator.state
    np.save(ROOT / 'events.npy', np.asarray(events, dtype=np.int64))
    write_json(ROOT / 'sampler.json', dict(seed=20261006, tasks=state['fit_tasks'],
        event_shape=[500, 512], events_path=str(ROOT/'events.npy'), identical_two_arms=True,
        sampling='each update each task16; uniform clip then uniform sampled frame; replacement',
        snapshots=snapshots))


def labels(*, fit_only=False):
    state = load_manifest()
    with np.load(ROOT / 'labels.npz') as store:
        return {c['key']:dict(z=store[c['key']+'_z'], pre_positive=store[c['key']+'_pre_positive'],
                    source=dict(hdf5=c['hdf5'], orientation_field=f'data/demo_{c["demo"]}/obs/ee_ori',
                        feature=str(ROOT/'features'/(c['key']+'.safetensors')),
                        raw_indices=c['raw_indices'], same_obs_offset=0))
                for c in state['clips'] if not fit_only or c['role'] == 'fit'}


def feature(clip, arm):
    values = load_file(str(ROOT / 'features' / (clip['key'] + '.safetensors')))
    return values['H'] if arm == 'H' else torch.cat((values['K'], values['V']), dim=-1)


def fit_data(arm, device):
    state, target = load_manifest(), labels(fit_only=True)
    clips = [r for r in state['clips'] if r['role'] == 'fit']
    # Fit is the first contiguous manifest block; indices are the saved global IDs.
    if clips[0]['global_start'] != 0 or sum(c['nframes'] for c in clips) != 1940:
        raise ValueError('fixed fit-event indexing changed')
    blocks = [feature(c, arm).to(device=device, dtype=torch.bfloat16).detach() for c in clips]
    x = torch.cat(blocks)
    del blocks
    y = torch.from_numpy(np.concatenate([target[c['key']]['z'] for c in clips])).to(device)
    events = torch.from_numpy(np.load(ROOT/'events.npy')).to(device)
    if events.shape != (500, 512) or events.min() < 0 or events.max() >= len(x):
        raise ValueError('registered event stream invalid')
    return x, y, events


def new_head(arm, device):
    from .hand_axis_probe import make_heads
    heads = make_heads()
    head = heads[arm].to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, betas=(.9, .95),
        eps=1e-8, weight_decay=1e-4)
    return head, optimizer


def head_context():
    from ember.pi05_source_setup import initialize_distributed
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError('these are independent heads, not distributed scientific batches')
    return context


def update(head, optimizer, x, y, event, micro):
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=x.device)
    for start in range(0, 512, micro):
        indices = event[start:start+micro]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            predicted = head(x[indices])
        error = (predicted.float() - y[indices]).square().sum(-1).sum() / 512
        error.backward()
        total += error.detach()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    if not torch.isfinite(total) or not torch.isfinite(norm):
        raise ValueError('head loss/gradient nonfinite')
    optimizer.step()
    return float(total), float(norm)


def profile(arm):
    from .run import frozen_git
    torch.set_num_threads(4)
    context = head_context()
    torch.manual_seed(7)
    device = context.device
    x, y, events = fit_data(arm, device)
    rows = []
    for micro in (128, 512):
        # Exactly two discarded updates per head; each is from the same fresh function.
        head, opt = new_head(arm, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss, norm = update(head, opt, x, y, events[0], micro)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append(dict(microbatch=micro, logical_batch=512, seconds=elapsed,
            frames_per_second=512/elapsed, loss=loss, gradient_norm=norm,
            allocated_peak_bytes=torch.cuda.max_memory_allocated(),
            reserved_peak_bytes=torch.cuda.max_memory_reserved()))
        del head, opt
    chosen = max(rows, key=lambda r:r['frames_per_second'])['microbatch']
    write_json(ROOT/'launch'/f'head_profile_{arm}.json', dict(arm=arm, profiles=rows,
        selected_microbatch=chosen, reset='fresh head/optimizer/seed7 for full500; events cursor0',
        discarded_updates=2, no_evaluation=True, git=frozen_git(continuation=True),
        stopped_larger_reason='physical512 equals complete registered logical batch'))


def save_checkpoint(arm, head, optimizer, update_index, contract):
    sampler = json.loads((ROOT/'sampler.json').read_text())
    path = ROOT / 'heads' / arm / f'update_{update_index:03d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema='hand_axis_probe_checkpoint_v1', head=head.state_dict(),
        optimizer=optimizer.state_dict(), next_update=update_index, arm=arm, contract=contract,
        sampler=dict(seed=20261006, next_update=update_index,
            RNG=sampler['snapshots'][str(update_index)], events_path=str(ROOT/'events.npy')),
        torch_CPU_RNG=torch.get_rng_state(), torch_CUDA_RNG=torch.cuda.get_rng_state_all()), path)


def train(arm):
    from .run import frozen_git
    torch.set_num_threads(4)
    context = head_context()
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = context.device
    micro = json.loads((ROOT/'launch'/f'head_profile_{arm}.json').read_text())['selected_microbatch']
    x, y, events = fit_data(arm, device)
    head, optimizer = new_head(arm, device)
    contract = dict(arm=arm, git=frozen_git(continuation=True), parent=str(PARENT),
        frozen_writer=True, detached_native_features=True, input_dim=1024 if arm=='H' else 512,
        parameters=sum(p.numel() for p in head.parameters()), seed=7, updates=500,
        sampler_seed=20261006, task_equal32=True, logical_batch=512, microbatch=micro,
        topology=dict(world_size=1, device=os.environ['CUDA_VISIBLE_DEVICES'],
                      numa_node=context.numa_node, cpu_affinity=list(context.cpu_affinity)),
        numeric=dict(features_saved='native KV/FP32 H', head='FP32 weights,BF16 autocast,TF32 allowed'),
        optimizer=dict(kind='AdamW', lr=.001, betas=[.9,.95], eps=1e-8, wd=1e-4, clip=1,
                       scheduler=None), loss='mean per-frame squared vector norm', evaluation_update=500)
    write_json(ROOT/'heads'/arm/'contract.json', contract)
    rows = []
    started = time.perf_counter()
    for step in range(500):
        loss, norm = update(head, optimizer, x, y, events[step], micro)
        rows.append(dict(update=step+1, loss=loss, gradient_norm=norm))
        if step+1 in (250, 500):
            save_checkpoint(arm, head, optimizer, step+1, contract)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-started
    del x, y, events
    torch.cuda.empty_cache()
    # Sole endpoint: predict every authorized fit/eval frame once, no new native.
    head.eval()
    state = load_manifest()
    predictions = {}
    evaluation_seconds = time.perf_counter()
    with torch.no_grad():
        for clip in state['clips']:
            values = feature(clip, arm)
            outputs = []
            for offset in range(0, len(values), 512):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs.append(head(values[offset:offset+512].to(device, dtype=torch.bfloat16)).float().cpu())
            prediction = torch.cat(outputs).numpy()
            if prediction.shape != (clip['nframes'], 3) or not np.isfinite(prediction).all():
                raise ValueError('endpoint prediction invalid')
            predictions[clip['key']] = prediction
    np.savez(ROOT/'heads'/arm/'predictions.npz', **predictions)
    write_json(ROOT/'heads'/arm/'training_rows.json', rows)
    write_json(ROOT/'heads'/arm/'completion.json', dict(updates=500, frames_evaluated=4382,
        clips_evaluated=136, endpoint_only=True, train_seconds=training_seconds,
        evaluation_seconds=time.perf_counter()-evaluation_seconds,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(), reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        fresh_after_profile=True, contract=contract))


def analyze():
    from .hand_axis_probe import analyze as score
    state = load_manifest()
    predictions = {}
    for arm in ('H','KV'):
        completion = json.loads((ROOT/'heads'/arm/'completion.json').read_text())
        if completion['updates'] != 500 or completion['frames_evaluated'] != 4382:
            raise ValueError('both fixed endpoints must precede scoring')
        with np.load(ROOT/'heads'/arm/'predictions.npz') as store:
            predictions[arm] = {key:store[key] for key in store.files}
    if json.loads((ROOT/'label_sources.json').read_text())['frames'] == 1940:
        prepare_labels(evaluation=True)
    target = labels()
    result = score(state, target, predictions, ROOT/'analysis')
    rows = []
    for clip in state['clips']:
        for index, raw_index in enumerate(clip['raw_indices']):
            rows.append(dict(key=clip['key'], task=clip['task'], demo=clip['demo'], role=clip['role'],
                sampled_index=index, raw_obs_index=raw_index, z=target[clip['key']]['z'][index].tolist(),
                H=predictions['H'][clip['key']][index].tolist(), KV=predictions['KV'][clip['key']][index].tolist(),
                pre_positive=bool(target[clip['key']]['pre_positive'][index]),
                hdf5=clip['hdf5'], feature=str(ROOT/'features'/(clip['key']+'.safetensors'))))
    with (ROOT/'analysis/per_frame.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row)+'\n')
    write_json(ROOT/'analysis/prediction_completion.json', dict(frames=len(rows), arms=['H','KV'],
        legal_train_only=True, full_source_mapping=True))
    return result


def paired(phase):
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices) != 2:
        raise ValueError('two independent heads require two selected devices for this physical launch')
    processes = []
    for arm, gpu in zip(('H','KV'), devices, strict=True):
        log = (ROOT/'launch'/f'head_{phase}_{arm}.log').open('w')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        processes.append((subprocess.Popen([os.sys.executable,'-m',
            'ember.operator_writer.hand_axis_train',phase,'--arm',arm],
            env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = [p.wait() for p,_ in processes]
    for _, log in processes:
        log.close()
    if any(codes):
        raise RuntimeError(f'head {phase} exit codes {codes}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['labels','profile','train','paired-profile','paired-train','analyze'])
    parser.add_argument('--arm', choices=['H','KV'])
    args = parser.parse_args()
    if args.phase == 'labels':
        prepare_labels()
    elif args.phase == 'analyze':
        analyze()
    elif args.phase.startswith('paired-'):
        paired(args.phase.removeprefix('paired-'))
    elif args.phase == 'profile':
        profile(args.arm)
    else:
        train(args.arm)


if __name__ == '__main__':
    main()
