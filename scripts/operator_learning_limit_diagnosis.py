"""Bounded design §33 diagnostic; retired after this batch, never a trainer."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import default_collate

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer.data import FormalData
from ember.operator_writer.native import read_native_video
from ember.operator_writer.run import CONTINUATION2340_SPEC_PATH, build_runtime, specification
from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import task_logical_batch_policy_rng_seed
from ember.writer.runtime import autocast

ASSET = Path('/data1/user/ymdai/projects/EMBER')
ROOT = Path('/data1/user/ymdai/ember_runs/operator_learning_limit_diagnosis_20260930')
ECP = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340')
PRIOR = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/functional_credit_transport')
READOUT = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/self_conditioned_readout')
TASKS = (0, 12, 20, 32)
TEACHERS = {0: (40, 11), 12: (25, 14), 20: (38, 42), 32: (17, 43)}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_pt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def prior_readout():
    # Exact file import: the retired diagnostic's ambiguous analysis_script
    # imports previously selected the wrong module before any GPU forward.
    spec = importlib.util.spec_from_file_location('registered_parent_readout', READOUT / 'analysis_script_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panels():
    result = json.loads((PRIOR / 'group0/fixed_panels.json').read_text())
    for task in TASKS:
        p = result[str(task)]
        a, b = ({q['demo'] for q in p[k]} for k in ('A', 'B'))
        teachers = set(p['teachers'])
        if (tuple(p['teachers']) != TEACHERS[task] or len(a) != 28 or len(b) != 20
                or a & b or a & teachers or b & teachers or a | b | teachers != set(range(50))):
            raise ValueError('registered 2/28/20 teacher/A/B partition changed')
    return result


def query_manifest(panel, lengths):
    steps = []
    for s in range(64):
        row = {}
        for task in TASKS:
            demos = sorted(q['demo'] for q in panel[str(task)]['A'])
            rng = np.random.default_rng(np.random.SeedSequence([20260930, 33, task, s]))
            frames = [int(rng.integers(lengths[task][d] - 1)) for d in demos]
            queries = [{'demo': d, 'frame': f} for d, f in zip(demos, frames, strict=True)]
            seed = task_logical_batch_policy_rng_seed(optimization_seed=7, task_id=task,
                task_visit=330000+s, demo_indices=demos, frame_indices=frames)
            row[str(task)] = {'queries': queries, 'flow_seed': seed, 'visit': 330000+s}
        steps.append(row)
    return {'schema': 'ember_learning_limit_queries_v1', 'steps': steps,
            'query_offset': 1, 'flow': 'canonical Gaussian/Beta(1.5,1)',
            'optimization_seed': 7, 'panel_source': str(PRIOR / 'group0/fixed_panels.json')}


def prepare_manifest():
    spec = specification(CONTINUATION2340_SPEC_PATH)
    data = FormalData(ASSET, spec, task_ids=TASKS)
    try:
        manifest = query_manifest(panels(), {t: data.tasks[t].episode_lengths for t in TASKS})
        path = ROOT / 'query_manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('frozen new-query manifest changed')
        save_json(path, manifest)
        print('manifest frozen: 64 steps / 4 tasks / 28 queries; three arms and both teachers share', flush=True)
    finally:
        data.close()


def source_setup(device):
    spec = specification(CONTINUATION2340_SPEC_PATH)
    runtime = build_runtime(ASSET, spec, torch.device(device), 'T', evaluation=False)
    runtime.writer.load_state_dict(load_file(str(ECP / 'ecp.safetensors'), device=device), strict=True)
    runtime.policy.eval()
    runtime.writer.eval()
    runtime.restore_identity()
    for parameter in runtime.writer.common.parameters():
        parameter.requires_grad_(False)
    return spec, runtime


def native_cache(runtime, data, output_queues):
    cached, records = {}, []
    for task in TASKS:
        for teacher in TEACHERS[task]:
            condition, raw, sampled = data.condition(runtime, task, teacher)
            runtime.restore_identity()
            with torch.no_grad(), autocast(runtime.device):
                x, h = read_native_video(runtime.policy, runtime.writer.public_state(),
                    runtime.writer.probe, condition, runtime.writer.names,
                    frame_chunk=32, checkpoint_frames=False)
            if set(x) != set(runtime.writer.names) or h.shape[1:] != (50, 1024):
                raise ValueError('complete 38-target native X/H required')
            cached[(task, teacher)] = ({k: v.detach().cpu().share_memory_() for k, v in x.items()},
                                       h.detach().cpu().share_memory_())
            records.append({'task': task, 'teacher': teacher, 'raw_frames': raw,
                            'sampled_frames': sampled, 'X_targets': len(x), 'H_shape': list(h.shape)})
            print('native', task, teacher, sampled, flush=True)
    # Tensor storage is shared RAM, not a persistent native artifact. The S
    # worker remains alive until P/D acknowledge receiving their own GPU copy.
    for q in output_queues:
        q.put(cached)
    save_json(ROOT / 'native_capture.json', {'conditions': records, 'complete': True,
        'parent': str(ECP), 'training_git': 'e2afbfd7c997e3f792921600608efa2fa3c1b25a',
        'native_disk_cache': False, 'frame_chunk': 32})
    return cached


@torch.no_grad()
def latent_z(writer, cached):
    """Read existing P/C/D recurrence in real-arithmetic M=OZ coordinates."""
    x, h = cached
    h = h.float()
    normalized = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
    result = {}
    public = writer.public_state()
    for name, write in zip(writer.names, writer.writes, strict=True):
        address = public[name + LORA_A_SUFFIX].float()
        z = torch.zeros(256, 128, device=h.device)
        for t in range(len(h)-1):
            key = F.normalize(F.linear(x[name][t].float(), address), dim=-1, eps=1e-6).T
            value = (F.gelu(write.p(key.T) + write.c(normalized[t]))
                     * write.d(normalized[t+1]-normalized[t])).T
            z = z + (value - z @ key) @ key.T / 50
        result[name] = z.cpu()
    return result


def raw_batch(data, task, queries, teacher):
    if teacher in {q['demo'] for q in queries}:
        raise ValueError('teacher episode entered action queries')
    return default_collate([data.queries[data.rows[task][q['demo']][q['frame']]] for q in queries])


def credit(runtime, state, batch, seed, weight, backward, microbatch):
    with autocast(runtime.device):
        return paired_functional_credit(runtime.policy, state, runtime.lora, batch,
            seed=seed, device=runtime.device, random_batch=28, offset=0,
            microbatch=microbatch, condition_weight=weight, backward=backward, prefix_steps=None)


def backprop(state, cotangent):
    names = [name for name, value in state.items() if value.requires_grad]
    if len(names) != 38 or any(not name.endswith(LORA_B_SUFFIX) for name in names):
        raise ValueError('trainable output must be all 38 B factors')
    torch.autograd.backward(tuple(state[n] for n in names), tuple(cotangent[n].to(state[n]) for n in names))


def bank_state(out, task, teacher, state):
    path = out / 'bank' / f'task{task:03d}_teacher{teacher:02d}.safetensors'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(temporary))
    temporary.replace(path)


class Learning:
    def __init__(self, arm, runtime, cached, panel, microbatch):
        self.arm, self.runtime, self.panel, self.microbatch = arm, runtime, panel, microbatch
        self.keys = [(t, d) for t in TASKS for d in TEACHERS[t]]
        self.cached = {key: ({n: v.to(runtime.device) for n, v in x.items()}, h.to(runtime.device))
                       for key, (x, h) in cached.items()}
        self.parent = {}
        with torch.no_grad(), autocast(runtime.device):
            for key in self.keys:
                self.parent[key] = {n: v.detach() for n, v in runtime.writer(*self.cached[key]).items()}
        if arm == 'S':
            self.objects = {'all': runtime.writer}
        elif arm == 'P':
            self.objects = {task: copy.deepcopy(runtime.writer) for task in TASKS}
        else:
            self.objects = {key: torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros_like(self.parent[key][n + LORA_B_SUFFIX], dtype=torch.float32))
                for n in runtime.writer.names]) for key in self.keys}
        self.params = {key: tuple(obj.parameters() if arm == 'D' else obj.writes.parameters())
                       for key, obj in self.objects.items()}
        self.optimizers = {key: torch.optim.AdamW(p, lr=1e-4, betas=(.9, .95), eps=1e-8,
                            weight_decay=1e-4) for key, p in self.params.items()}
        self.hist = []

    def compiled(self, key):
        if self.arm == 'D':
            state = dict(self.parent[key])
            for name, value in zip(self.runtime.writer.names, self.objects[key], strict=True):
                state[name+LORA_B_SUFFIX] = self.parent[key][name+LORA_B_SUFFIX].float() + value
            return state
        writer = self.objects['all' if self.arm == 'S' else key[0]]
        with autocast(self.runtime.device):
            return writer(*self.cached[key])

    def checkpoint(self, step, out):
        save_pt(out / f'recovery_{step:02d}.pt', {'step': step, 'arm': self.arm,
            'parameters': {str(k): obj.state_dict() for k, obj in self.objects.items()},
            'optimizers': {str(k): opt.state_dict() for k, opt in self.optimizers.items()},
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(self.runtime.device),
            'numpy_rng': np.random.get_state(), 'manifest': str(ROOT / 'query_manifest.json'),
            'parent': str(ECP), 'diagnostic_only': True})

    def step(self, step, queries, data):
        start = time.monotonic()
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        losses, displacements = {}, {}
        for task in TASKS:
            q = queries[str(task)]
            batch = self.runtime.processor.training_batch(raw_batch(data, task, q['queries'], TEACHERS[task][0]))
            for teacher in TEACHERS[task]:
                key = (task, teacher)
                state = self.compiled(key)
                weight = 1/8 if self.arm == 'S' else 1/2 if self.arm == 'P' else 1.
                result = credit(self.runtime, state, batch, q['flow_seed'], weight, True, self.microbatch)
                backprop(state, result['lora_cotangent'])
                losses[f'{task}_{teacher}'] = result['flow_loss']
                displacements[f'{task}_{teacher}'] = math.sqrt(sum(float(
                    (state[n+LORA_B_SUFFIX].detach().float()-self.parent[key][n+LORA_B_SUFFIX].float())
                    .square().sum()) for n in self.runtime.writer.names))
                del state, result
        norms = {}
        for key, params in self.params.items():
            if any(p.grad is None for p in params):
                raise ValueError('actual write parameter lacks gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
            norms[str(key)] = {'preclip_l2': norm, 'clip_factor': min(1., 1./max(norm, 1e-30))}
            self.optimizers[key].step()
        if not all(math.isfinite(v) for v in losses.values()):
            raise ValueError('nonfinite real FM')
        self.hist.append({'step': step+1, 'A_fm': losses, 'gradient': norms,
            'preupdate_B_delta_l2': displacements,
            'seconds': time.monotonic()-start, 'microbatch': self.microbatch,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30})

    def endpoints(self, data, out):
        reader = prior_readout()
        rows = []
        for key in self.keys:
            task, teacher = key
            state = self.compiled(key)
            validate_lora_state(state, self.runtime.lora)
            parent = self.parent[key]
            delta = math.sqrt(sum(float((state[n+LORA_B_SUFFIX].detach().float()
                -parent[n+LORA_B_SUFFIX].float()).square().sum()) for n in self.runtime.writer.names))
            p = self.panel[str(task)]
            fixed = self.runtime.processor.training_batch(raw_batch(data, task, p['A'], teacher))
            after = credit(self.runtime, state, fixed, p['A_flow_seed'], 1., False, self.microbatch)['flow_loss']
            refpath = PRIOR / f'group0/task{task:03d}_B_query_noise_target.pt'
            ref = torch.load(refpath, map_location='cpu', weights_only=False)
            obs, noise, target, valid = reader.b_batch(data, self.runtime, task, p['B'], ref, torch=torch)
            self.runtime.restore_identity()
            prediction = reader.generated(self.runtime, state, obs, noise, 10, torch)[:, :, :7]
            risk = reader.risk(prediction, target, valid, torch)
            parent_ref = READOUT / 'matched_parent10' / f'group{0 if task in (0,12) else 1}' / f'task{task:03d}_teacher{teacher:02d}_parent10.pt'
            if not parent_ref.exists():
                raise FileNotFoundError(parent_ref)
            save_pt(out / f'task{task:03d}_teacher{teacher:02d}_endpoint.pt', {
                'prediction': prediction, 'target_ref': str(refpath), 'parent_ref': str(parent_ref),
                'queries': p['B'], 'valid': valid, 'B_risk': risk,
                'A_fixed_fm_after': after, 'B_delta_l2': delta})
            bank_state(out, task, teacher, state)
            if self.arm != 'D':
                writer = self.objects['all' if self.arm == 'S' else task]
                save_pt(out / f'task{task:03d}_teacher{teacher:02d}_Z.pt', latent_z(writer, self.cached[key]))
            rows.append({'task': task, 'teacher': teacher, 'B_risk': risk,
                         'A_fixed_fm_after': after, 'B_delta_l2': delta})
            print(self.arm, 'endpoint', key, risk['first5'], flush=True)
        save_json(out / 'endpoint_rows.json', rows)


def worker(arm, device, incoming, outgoing, acknowledgements, microbatch):
    out = ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Physical GPU-local CPU mapping; independent workers never use NCCL.
    physical = int(device.split(':')[1])
    allowed = set(os.sched_getaffinity(0))
    local = set(range(28)) | set(range(56,84)) if physical < 4 else set(range(28,56)) | set(range(84,112))
    os.sched_setaffinity(0, allowed & local)
    torch.set_num_threads(4)
    torch.cuda.set_device(physical)
    np.random.seed(7)
    torch.manual_seed(7)
    data = None
    try:
        print(arm, 'loading source', flush=True)
        spec, runtime = source_setup(device)
        data = FormalData(ASSET, spec, task_ids=TASKS)
        panel = panels()
        cached = native_cache(runtime, data, outgoing) if arm == 'S' else incoming.get()
        learning = Learning(arm, runtime, cached, panel, microbatch)
        if arm == 'S':
            for key in learning.keys:
                task, teacher = key
                state = learning.parent[key]
                bank_state(ROOT / 'parent', task, teacher, state)
                save_pt(ROOT / 'parent' / f'task{task:03d}_teacher{teacher:02d}_Z.pt', latent_z(runtime.writer, learning.cached[key]))
                batch = runtime.processor.training_batch(raw_batch(data, task, panel[str(task)]['A'], teacher))
                before = credit(runtime, state, batch, panel[str(task)]['A_flow_seed'], 1., False, microbatch)['flow_loss']
                save_json(ROOT / 'parent' / f'task{task:03d}_teacher{teacher:02d}_A_before.json', {'A_fixed_fm_before': before})
        else:
            acknowledgements.put(arm)
        manifest = json.loads((ROOT / 'query_manifest.json').read_text())
        for step, queries in enumerate(manifest['steps']):
            learning.step(step, queries, data)
            save_json(out / 'training_steps.json', learning.hist)
            print(arm, 'step', step+1, 'seconds', learning.hist[-1]['seconds'], flush=True)
            if step+1 in (16,32,64):
                learning.checkpoint(step+1, out)
        learning.endpoints(data, out)
        if arm == 'S':
            for _ in range(2):
                acknowledgements.get()
        save_json(out / 'completion.json', {'status': 'complete', 'arm': arm,
            'updates_per_copy': 64, 'optimizer_calls': 64*len(learning.objects),
            'queries_per_condition': 64*28, 'conditions': 8, 'source_frozen': True})
    except BaseException:
        save_json(out / 'failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        if data is not None:
            data.close()
        save_json(out / 'process.json', {'start_epoch': started, 'end_epoch': time.time(),
            'GPU_hours': (time.time()-started)/3600, 'physical_device': device})


def launch(devices, microbatch, budget):
    import multiprocessing as mp
    from multiprocessing.connection import wait
    context = mp.get_context('spawn')
    queues = [context.Queue() for _ in range(2)]
    ack = context.Queue()
    processes = [context.Process(target=worker,
        args=(arm, device, None if arm == 'S' else queues[i-1], queues if arm == 'S' else [], ack, microbatch))
        for i, (arm, device) in enumerate(zip(('S','P','D'), devices, strict=True))]
    started = time.time()
    starts, exits = {}, {}
    for process in processes:
        process.start()
        starts[process.pid] = time.time()
    alive = list(processes)
    done_cost = 0.
    try:
        while alive:
            remaining = budget*3600-done_cost-(time.time()-started)*len(alive)
            ready = wait([p.sentinel for p in alive], timeout=max(0., remaining/len(alive)))
            if not ready:
                raise TimeoutError('learning allocation reached reserved GPU-hour bound')
            for process in list(alive):
                if process.sentinel in ready:
                    process.join()
                    exits[process.pid] = time.time()
                    done_cost += time.time()-started
                    alive.remove(process)
                    if process.exitcode:
                        raise RuntimeError(f'learning worker {process.pid} exit {process.exitcode}')
        save_json(ROOT / 'learning_completion.json', {'status': 'complete',
            'process_GPU_hours': sum(json.loads((ROOT/a/'process.json').read_text())['GPU_hours'] for a in ('S','P','D'))})
    finally:
        for process in alive:
            if process.is_alive():
                process.terminate()
            process.join()
            exits[process.pid] = time.time()
        save_json(ROOT / 'learning_allocation.json', {'processes': [
            {'pid': p.pid, 'start_epoch': starts[p.pid], 'end_epoch': exits[p.pid],
             'exit_code': p.exitcode, 'GPU_hours': (exits[p.pid]-starts[p.pid])/3600}
            for p in processes], 'GPU_hours': sum((exits[p.pid]-starts[p.pid])/3600 for p in processes)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-manifest', action='store_true')
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2')
    parser.add_argument('--microbatch', type=int, default=14)
    parser.add_argument('--learning-budget-gpuh', type=float, default=2.7)
    args = parser.parse_args()
    if args.prepare_manifest:
        prepare_manifest()
    else:
        if not (ROOT/'query_manifest.json').exists():
            raise ValueError('freeze complete query manifest before any learning')
        launch(args.devices.split(','), args.microbatch, args.learning_budget_gpuh)


if __name__ == '__main__':
    main()
