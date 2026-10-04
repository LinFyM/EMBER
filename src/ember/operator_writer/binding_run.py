"""Fixed native cache and bounded F/G/R64 lifecycle using canonical owners."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import ember.pi05_evaluation  # public import owner before leaf modules
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from ember.writer.topology import bind_current_process_to_cuda_numa
from .data import FormalData
from .run import build_runtime
from .native import read_native_video
from .role_binding import RoleBinding, freeze_heads, native_keys, compile_fixed, q_layer
from .binding_credit import step

ROOT = Path('/data1/user/ymdai/ember_runs/native_role_binding_compilation_20261004')
ASSET = Path('/data1/user/ymdai/projects/EMBER')
TASKS = (12, 13, 14, 15, 17, 19, 43, 96)
HELD_TEACHERS = (47, 40, 33, 28, 46, 32, 1, 24, 43)
PARENT = ROOT.parent / 'conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors'


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def initialize():
    identity = git_state(Path(__file__).resolve().parents[3])
    if identity['branch'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('all actual GPU consumers require clean pushed detached code')
    os.environ['EMBER_LIBERO_ASSETS_ROOT'] = str(ASSET / 'data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6')
    prepare_libero_config(ROOT / 'cache/libero_config')
    torch.cuda.set_device(0); torch.set_num_threads(8)
    affinity = bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7); torch.cuda.manual_seed(7); np.random.seed(7); random.seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    spec = read_json(ROOT / 'launch/parent_spec.json')
    runtime = build_runtime(ASSET, spec, torch.device('cuda:0'), 'conditional_read_write', evaluation=True)
    runtime.writer.load_state_dict(load_file(str(PARENT), device='cuda:0'), strict=True)
    runtime.policy.eval(); runtime.writer.train()
    return runtime, spec, identity, affinity


@torch.no_grad()
def create_native(runtime, spec, identity):
    if (ROOT / 'native/manifest.json').exists():
        raise ValueError('native cache already complete; do not repeat the registered reads')
    records = []
    for tasks, role in ((TASKS, 'train'), ((16,), 'validation')):
        data = FormalData(ASSET, spec, query_labels=False, task_ids=tasks, role=role)
        try:
            for task in tasks:
                for teacher in ((0, 1) if task != 16 else HELD_TEACHERS):
                    path = ROOT / 'native' / f'task{task:03d}_teacher{teacher:02d}.pt'
                    if path.exists():
                        raise ValueError(f'partial prior native artifact requires explicit identity recovery: {path}')
                    condition, raw, sampled = data.condition(runtime, task, teacher)
                    runtime.restore_identity(); started = time.monotonic()
                    with native_keys(runtime.policy) as chunks, autocast(runtime.device):
                        x, h = read_native_video(runtime.policy, runtime.writer.public_state(), runtime.writer.probe,
                            condition, runtime.writer.names, frame_chunk=128, checkpoint_frames=False)
                        c, d = runtime.writer.interpreter(h, condition[1])
                    k = torch.stack([torch.cat(v) for v in chunks], 1)
                    if k.shape != (sampled, 18, 512, 256) or set(x) != set(runtime.writer.names):
                        raise ValueError('same-native full38 X and actual prefix pre-RoPE K capture incomplete')
                    fixed = dict(X={n: v.cpu().contiguous() for n, v in x.items()}, H=h.cpu(), c=c.cpu(),
                                 d=d.cpu(), K=k.contiguous(), frame_indices=condition[1].cpu())
                    torch.save(fixed, path)
                    row = dict(task=task, teacher=teacher, raw_frames=raw, sampled_frames=sampled,
                        native_reading_git=identity['commit'], source_training_git='b8ea00e9fbb86742ef076bac9dd35c5314cd5aed',
                        public_training_git='85919994aef11c17b49b7d0e70a2c110158bff61', frame_chunk=128,
                        frame_chunk_exhausts_actual_video=sampled <= 128, source_once_resident=True,
                        teacher_fields_read=['obs/agentview_rgb', 'obs/eye_in_hand_rgb'], public_native_reads=1,
                        seconds=time.monotonic() - started, reserved_peak_GiB=torch.cuda.max_memory_reserved() / 2**30,
                        fields=['X', 'H', 'c', 'd', 'K_pre_RoPE', 'frame_indices'], prefix_suffix_visibility='native prefix cannot see suffix')
                    write_json_atomic(path.with_suffix('.json'), row)
                    records.append(dict(**row, artifact=file_record(path)))
                    del fixed, chunks, x, h, c, d, k, condition
        finally:
            data.close()
    write_json_atomic(ROOT / 'native/manifest.json', dict(schema='native_role_binding_fixed_native_v1', records=records,
        native_reads=25, shared_all_arms=True, derived_A_S_key_deltaZ_Value_M_alpha_R_cached=False))


def fixed_to_gpu(value, device):
    return {k: fixed_to_gpu(v, device) for k, v in value.items()} if isinstance(value, dict) else value.to(device)


def load_native(runtime, held=False):
    return {(t, d): fixed_to_gpu(torch.load(ROOT / 'native' / f'task{t:03d}_teacher{d:02d}.pt',
                map_location='cpu', weights_only=True, mmap=True), runtime.device)
            for t in ((16,) if held else TASKS) for d in (HELD_TEACHERS if held else (0, 1))}


def load_labels():
    record = read_json(ROOT / 'labels/manifest.json')
    if record['completed'] != 'complete' or record['held_privileged_read']:
        raise ValueError('complete train-only role labels required before learning')
    return {(r['task'], r['demo']): dict(np.load(r['path'])) for r in record['records']}


def optimizer(parameters):
    return torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)


def checkpoint(runtime, binding, opt, contract):
    path = ROOT / contract['arm'] / 'checkpoint64'; path.mkdir(exist_ok=False)
    save_file({k: v.cpu().contiguous() for k, v in runtime.writer.state_dict().items()}, str(path / 'writer.safetensors'))
    if binding is not None:
        save_file({k: v.cpu().contiguous() for k, v in binding.state_dict().items()}, str(path / 'binding.safetensors'))
    torch.save(dict(schema=contract['schema'], optimizer=opt.state_dict(), scheduler=None, scaler=None,
        cursor=64, sampler=dict(manifest=str(ROOT / 'query_manifest.json'), next_update=64),
        RNG=rng_state(), topology=contract['topology'], contract=contract), path / 'trainer_state.pt')
    write_json_atomic(path / 'checkpoint_manifest.json', dict(complete=True, next_update=64,
        schema=contract['schema'], files={p.name: file_record(p) for p in path.iterdir() if p.is_file()}))


def learn(runtime, spec, identity, affinity, arm):
    out = ROOT / arm; out.mkdir(exist_ok=True)
    if (out / 'checkpoint64').exists():
        raise ValueError('fixed64 already produced; do not duplicate training')
    parameters = freeze_heads(runtime.writer)
    binding = RoleBinding(runtime.device) if arm == 'R' else None
    parameters += tuple(binding.parameters()) if binding is not None else ()
    data = FormalData(ASSET, spec, query_labels=True, task_ids=TASKS)
    cached, labels = load_native(runtime), load_labels()
    manifest = read_json(ROOT / 'query_manifest.json')
    contract = dict(schema='native_role_binding_training_v1', study=ROOT.name, arm=arm, git=identity,
        parent=file_record(PARENT), parent_training_git='85919994aef11c17b49b7d0e70a2c110158bff61',
        source_training_git='b8ea00e9fbb86742ef076bac9dd35c5314cd5aed', updates=64,
        query_manifest=file_record(ROOT / 'query_manifest.json'), labels=file_record(ROOT / 'labels/manifest.json'),
        native=file_record(ROOT / 'native/manifest.json'), condition_weight=1 / 8, query_offset=1,
        main_queries=14336, additional_tau1_queries=14336 if arm != 'F' else 0,
        own_Q_weight=.1 if arm != 'F' else 0, teacher_selection_weight=.1 if arm == 'R' else 0,
        teacher_selection_replay='once per condition', query_hidden_live=True, replay='complete76 same-version cotangent',
        frozen=['source', 'common_A0_B0', 'native', 'four_layer_interpreter'],
        trainable_names=[n for n, p in runtime.writer.named_parameters() if p.requires_grad],
        binding_parameters=20381696 if binding is not None else 0,
        optimizer=dict(kind='AdamW', lr=1e-4, betas=[.9, .95], eps=1e-8, weight_decay=1e-4, clip=1, scheduler=None),
        topology=dict(host=socket.gethostname(), world_size=1, physical_device=os.environ['CUDA_VISIBLE_DEVICES'], affinity=affinity))
    initial = [p.detach().clone() for p in parameters]
    initial_rng = rng_state()
    profiles = []
    try:
        for micro in (14, 28):
            with torch.no_grad():
                for p, before in zip(parameters, initial, strict=True):
                    p.copy_(before)
            restore_rng(initial_rng); opt = optimizer(parameters)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                result = step(runtime, cached, data, labels, manifest['steps'][0], arm, binding, opt, parameters, micro)
            except torch.cuda.OutOfMemoryError as error:
                opt.zero_grad(set_to_none=True)
                result = dict(microbatch=micro, failed=True, error=str(error), seconds=time.monotonic() - started,
                              reserved_peak_GiB=torch.cuda.max_memory_reserved() / 2**30)
            profiles.append(result)
            del opt
            write_json_atomic(out / 'profile.json', dict(discarded_updates=len(profiles), registered_update=1, profiles=profiles))
        usable = [r for r in profiles if not r.get('failed')]
        if not usable:
            raise RuntimeError('both bounded real update profiles failed; no unregistered scan')
        selected = min(usable, key=lambda r: r['seconds'])
        with torch.no_grad():
            for p, before in zip(parameters, initial, strict=True):
                p.copy_(before)
        del initial
        restore_rng(initial_rng); opt = optimizer(parameters)
        contract.update(microbatch=selected['microbatch'], expected_training_seconds=64 * selected['seconds'],
            profile=profiles, profiles_restored_all_parameters_optimizer_RNG=True,
            packing='two teacher conditions share the same real prefix; micro28 exhausts28 paired queries; independent arms parallel')
        write_json_atomic(out / 'training_contract.json', contract)
        print(json.dumps(dict(event='profile_complete', arm=arm, seconds_per_update=selected['seconds'],
            expected_training_seconds=contract['expected_training_seconds'], selected_microbatch=selected['microbatch'],
            reserved_peak_GiB=selected['reserved_peak_GiB'])), flush=True)
        if input().strip() != 'RUN64':
            raise RuntimeError('launcher withheld64 because the actual budget projection failed')
        with (out / 'metrics.jsonl').open('x') as log:
            for entry in manifest['steps']:
                result = step(runtime, cached, data, labels, entry, arm, binding, opt, parameters, selected['microbatch'])
                log.write(json.dumps(dict(update=entry['update'], **result)) + '\n'); log.flush()
        checkpoint(runtime, binding, opt, contract)
        print(json.dumps(dict(event='checkpoint64_complete', arm=arm)), flush=True)
    finally:
        data.close()
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('native', 'learn'), required=True)
    parser.add_argument('--arm', choices=('F', 'G', 'R'))
    args = parser.parse_args()
    runtime, spec, identity, affinity = initialize()
    if args.stage == 'native':
        create_native(runtime, spec, identity)
    elif args.arm is None:
        raise ValueError('registered learning arm required')
    else:
        learn(runtime, spec, identity, affinity, args.arm)
    write_json_atomic(ROOT / ('native/consumer_completion.json' if args.stage == 'native' else args.arm + '/learning_completion.json'),
                      dict(status='complete', git=identity, stage=args.stage, arm=args.arm))


if __name__ == '__main__':
    main()
