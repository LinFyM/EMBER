#!/usr/bin/env python3
"""Registered 16-case fixed video-source learning decomposition on the canonical rollout consumer."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.lora import (LoRATarget, validate_lora_state, task_lora_state_dict,
                        copy_task_lora_state_, LORA_A_SUFFIX, LORA_B_SUFFIX)
from ember.operator_writer.model import OperatorReadWrite
from ember.operator_writer.native import read_native_video
from ember.operator_writer.data import FormalData
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.writer.runtime import autocast
from torch.nn import functional as F
from ember.operator_writer.bank import (FrozenOperatorAdapter, PreparedOperatorLoRA,
                                        episode_evidence, inspect_bank)
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
from ember.pi05_eval_contract import (git_state, git_state_is_clean_pushed_or_frozen_authority,
                                     policy_noise_seed)
from ember.pi05_evaluation import rollout_shard, _validate_episode_row
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.topology import bind_current_process_to_cuda_numa

ROOT = Path('/data1/user/ymdai/ember_runs/task32_learned_video_segment_20261002')
OLD = ROOT.parent / 'operator_learning_limit_diagnosis_20260930'
STUDY = ROOT.name
SOURCES = ((17, 'S', 0), (43, 'S', 1))
CUTS = {17: 80, 43: 95}
ARMS = ('E', 'L')
STATE_IDS = tuple(i for _ in SOURCES for _ in ARMS for i in range(4))


def key(teacher, arm):
    return f'task032_teacher{teacher}_{arm}'


def case_id(teacher, arm, state):
    return f'teacher{teacher}_{arm}_init{state}'


def load_originals():
    banks, contracts, results = {}, {}, {}
    for teacher, learned, slot in SOURCES:
        contracts[teacher] = read_json(OLD / f'parent/evaluation/teacher{slot}/run_contract.json')
        contract = contracts[teacher]
        task_keys = tuple((r['suite'], r['task_id']) for r in contract['tasks'])
        states = {(r['suite'], r['task_id']): tuple(r['init_state_ids']) for r in contract['tasks']}
        for arm in ('parent', learned):
            banks[(teacher, arm)] = inspect_bank(
                manifest_path=OLD / f'{arm}/bank/panel_teacher{slot}.json',
                source=contract['adapter']['source'], task_keys=task_keys,
                evaluation_role='development_train', require_formal=True,
                task_init_state_ids=states)
            result = read_json(OLD / f'{arm}/evaluation/teacher{slot}/results.json')
            rows = [r for r in result['rows'] if (r['suite'], r['task_id']) == ('libero_10', 2)]
            if (len(rows) != 4 or {r['init_state_id'] for r in rows} != set(range(4))
                    or any(r['split_role'] != 'train' or
                           r['operator_read_write_lora']['teacher_demo'] != teacher for r in rows)):
                raise ValueError('original task32 rows/source pairing changed')
            for row in rows:
                expected = [policy_noise_seed(7, 'libero_10', 2, row['init_state_id'], i)
                            for i in range(len(row['policy_noise_seeds']))]
                if row['policy_noise_seeds'] != expected or row['env_seed'] != 7:
                    raise ValueError('original root7 policy clock changed')
            results[(teacher, arm)] = {r['init_state_id']: r for r in rows}
    first = banks[(17, 'parent')]
    if (any(any(bank[k] != first[k] for k in ('source', 'shared', 'lora', 'spec', 'scene_root'))
            for bank in banks.values()) or
            any(contracts[17][k] != contracts[43][k] for k in
                ('policy', 'environment', 'rng', 'normalization', 'operator_read_write_scene'))):
        raise ValueError('original source/normalization/public/scene contract differs')
    return banks, contracts, results


def segment_delta(parent, learned, address, x, h, frame_indices, cut):
    """Actual TargetWrite Value maps and ordered coverage; arrival-index partition."""
    if (len(x) != len(h) or len(frame_indices) != len(h)
            or any(int(b) <= int(a) for a, b in zip(frame_indices[:-1], frame_indices[1:]))):
        raise ValueError('native transition clock changed')
    with torch.autocast(device_type=x.device.type, enabled=False):
        normalized = h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + 1e-6)
        keys = F.normalize(F.linear(x.float(), address.float()), dim=-1, eps=1e-6).transpose(1, 2)
        memories = [torch.zeros(parent.o.out_features, address.shape[0], device=x.device)
                    for _ in range(4)]
        for t in range(len(h)-1):
            k = keys[t]
            change = normalized[t+1]-normalized[t]
            values = [w.o(F.gelu(w.p(k.T)+w.c(normalized[t]))*w.d(change)).T
                      for w in (parent, learned)]
            dv = values[1]-values[0]
            early = int(frame_indices[t+1]) <= cut
            additions = (*values, dv if early else torch.zeros_like(dv),
                         torch.zeros_like(dv) if early else dv)
            memories = [m+(v-m@k)@k.T/50 for m, v in zip(memories, additions, strict=True)]
        if any(not torch.isfinite(t).all() for t in (*memories, keys)):
            raise ValueError('nonfinite segment recurrence')
    return memories, keys


def group_error(pairs):
    numerator = sum(float((a.float()-b.float()).square().sum()) for a, b in pairs)
    denominator = sum(float(b.float().square().sum()) for _, b in pairs)
    return dict(L2_error=numerator**.5, relative_L2=(numerator/max(denominator, 1e-30))**.5,
                max_abs_error=max(float((a.float()-b.float()).abs().max()) for a, b in pairs))


def reconstruct_condition(adapter, parent, learned, data, reader, teacher, parameter_sources):
    started = time.monotonic()
    condition, raw_frames, sampled = data.condition(reader, 32, teacher)
    expected = (250, 51) if teacher == 17 else (240, 49)
    if (raw_frames, sampled) != expected or CUTS[teacher] not in condition[1].tolist():
        raise ValueError('fixed video/cut changed')
    copy_task_lora_state_(adapter.policy, adapter.identity, adapter.lora)
    with autocast(reader.device):
        x, h = read_native_video(adapter.policy, parent.public_state(), parent.probe,
            condition, parent.names, frame_chunk=sampled, checkpoint_frames=False)
    torch.cuda.synchronize()
    native_seconds = time.monotonic()-started
    retained = {arm: load_file(str(OLD/f'{arm}/bank/task032_teacher{teacher}.safetensors'), device='cuda:0')
                for arm in ('parent', 'S')}
    if len(x) != 38 or h.shape != (sampled, 50, 1024):
        raise ValueError('public native lost full38/50 suffix')
    generated = {arm: {} for arm in ARMS}
    all_keys, recon_parent, recon_S, closure = {}, [], [], []
    for name, wp, ws in zip(parent.names, parent.writes, learned.writes, strict=True):
        a, b = name+LORA_A_SUFFIX, name+LORA_B_SUFFIX
        address, base = parent.public_state()[a], parent.public_state()[b]
        if any(not torch.equal(retained[r][a], address) for r in retained):
            raise ValueError('retained complete A differs from public parent')
        (mp, ms, early, late), k = segment_delta(wp, ws, address, x[name], h, condition[1], CUTS[teacher])
        all_keys[name] = k.cpu()
        recon_parent.append((base+mp, retained['parent'][b]))
        recon_S.append((base+ms, retained['S'][b]))
        closure.append((early+late, ms-mp))
        for arm, delta in zip(ARMS, (early, late), strict=True):
            generated[arm][a] = retained['parent'][a].cpu()
            generated[arm][b] = (retained['parent'][b]+delta).cpu()
    errors = dict(parent=group_error(recon_parent), S=group_error(recon_S),
                  E_plus_L_equals_S_minus_parent=group_error(closure))
    record = dict(teacher=teacher, raw_frames=raw_frames, sampled_frames=sampled,
        cut=CUTS[teacher], frame_indices=condition[1].tolist(),
        early_transitions=sum(int(i)<=CUTS[teacher] for i in condition[1][1:]),
        late_transitions=sum(int(i)>CUTS[teacher] for i in condition[1][1:]),
        hdf5=str(data.tasks[32].authority.path), fields_read=['obs/agentview_rgb','obs/eye_in_hand_rgb'],
        public_native_reads=1, frame_chunk=sampled, native_seconds=native_seconds,
        sampled_frames_per_second=sampled/native_seconds, errors=errors,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    write_json_atomic(ROOT/f'native/teacher{teacher}_reconstruction.json', record)
    if (errors['E_plus_L_equals_S_minus_parent']['relative_L2'] > .01
            or max(errors[r]['relative_L2'] for r in ('parent', 'S')) > .1):
        raise ValueError('material reconstruction/closure deviation; scientific validity stop, no numerical sweep')
    torch.save(dict(H=h.cpu(), K=all_keys, frame_indices=condition[1].cpu(),
        parameter_sources=parameter_sources,
        provenance=record), ROOT/f'native/task032_teacher{teacher}_H_K.pt')
    conditions = []
    for arm in ARMS:
        validate_lora_state(generated[arm], adapter.lora)
        path = ROOT/'bank'/f'{key(teacher,arm)}.safetensors'
        if path.exists():
            raise ValueError('refusing to overwrite segment bank')
        save_file(generated[arm], str(path))
        conditions.append(dict(condition_id=key(teacher,arm), global_task_id=32,
                                       teacher_demo=teacher, factors=file_record(path)))
    del x, h, all_keys, retained, recon_parent, recon_S, closure, generated
    return record, conditions


def construct(adapter, tokenizer_path, banks, originals, identity):
    """One source load; exactly two public-native reads and no query labels/optimizer."""
    pbank = banks[(17, 'parent')]
    checkpoint = Path(pbank['checkpoint'])/'ecp.safetensors'
    recovery = OLD/'S/recovery_64.pt'
    parent = OperatorReadWrite(adapter.lora, task_lora_state_dict(adapter.policy, clone=True), 'T').to('cuda:0')
    parent.load_state_dict(load_file(str(checkpoint)), strict=True)
    endpoint = torch.load(recovery, map_location='cpu', mmap=True, weights_only=False)
    if endpoint['step'] != 64 or endpoint['arm'] != 'S' or endpoint['parent'] != pbank['checkpoint']:
        raise ValueError('shared S64 endpoint/parent changed')
    learned = copy.deepcopy(parent)
    learned.load_state_dict(endpoint['parameters']['all'], strict=True)
    fixed = ['probe', *[n for n in parent.state_dict() if n.startswith('common.')]]
    if any(not torch.equal(parent.state_dict()[n], learned.state_dict()[n]) for n in fixed):
        raise ValueError('S changed public A/B0/probe')
    del endpoint
    parent.eval().requires_grad_(False)
    learned.eval().requires_grad_(False)
    bank = copy.deepcopy(pbank)
    bank.pop('learning_limit_panel')
    task = copy.deepcopy(next(t for t in bank['tasks'] if t['global_task_id'] == 32))
    task['episodes'] = [dict(init_state_id=i, condition_id=key(17, 'E'),
                            teacher_demo_indices=[17], video_ordinal=0) for i in range(4)]
    bank.update(mode='learned_video_segment', condition_factors='complete_A0_plus_S_B0_plus_M',
                conditions=[], tasks=[task], learned_video_segment=dict(study=STUDY,
                    cuts=CUTS, partition='arrival frame_indices[t+1]<=cut', construction_git=identity,
                    parent_training_git='e2afbfd7c997e3f792921600608efa2fa3c1b25a',
                    learning_git='092a0ae83080f85ddff0e4129c92fb42aa527c0c',
                    original_reading_git='5313257c8070a792bbec0c6345c98376ed5cd1ea'))
    spec = read_json(Path(pbank['spec']['path']))
    data = FormalData(Path(pbank['asset_root']), spec, query_labels=False, task_ids=(32,))
    tokens = Pi05TeacherPrefixTokenizer(tokenizer_path, 200, 'cuda:0')
    reader = SimpleNamespace(device=torch.device('cuda:0'), tokenizer=tokens)
    records = []
    try:
        for teacher, _, slot in SOURCES:
            record, conditions = reconstruct_condition(adapter, parent, learned, data, reader, teacher,
                dict(parent=file_record(checkpoint), S64=file_record(recovery)))
            records.append(record)
            bank['conditions'].extend(conditions)
            write_json_atomic(ROOT/'construction_readback.json', dict(status='constructing', records=records))
    finally:
        data.close()
    bank_path = ROOT/'bank/manifest.json'
    write_json_atomic(bank_path, bank)
    bank['manifest'] = file_record(bank_path)
    write_json_atomic(ROOT/'prepared_bank.json', bank)
    write_json_atomic(ROOT/'construction_readback.json', dict(status='complete', records=records,
        native_reads=2, source_loading_count=1, no_optimizer_created=True, no_query_labels_read=True,
        parameter_sources=dict(parent=file_record(checkpoint), shared_S64=file_record(recovery)),
        same_fixed_features=True, complete38_rank128=True, construction_git=identity))
    write_json_atomic(ROOT/'inputs.json', dict(original_rows={f'{t}_{a}': list(v.values())
        for (t,a),v in originals.items()}, original_banks={f'{t}_{a}': b['manifest'] for (t,a),b in banks.items()},
        no_held_reads=True, no_new_actions_or_FM=True, no_optimizer=True,
        A='retained_parent', B='retained_parent_complete_B_plus_segment_DeltaM_no_extra_B0'))
    return bank


def build_cases(bank, contracts):
    contract = copy.deepcopy(contracts[17])
    contract.update(git=git_state(Path(__file__).resolve().parents[1]), output_dir=str(ROOT),
                    role='development_train', mode='screen', analysis_only=True, adapter=bank)
    task = copy.deepcopy(next(t for t in contract['tasks'] if (t['suite'], t['task_id']) == ('libero_10', 2)))
    task['init_state_ids'] = list(STATE_IDS)
    contract['tasks'] = [task]
    contract['parallel'].update(envs_per_replica=16, physical_gpu_count=1,
        physical_gpu_ids=[int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))],
        replicas_per_gpu=1, worker_count=1)
    ids = [case_id(t, a, i) for t, _, _ in SOURCES for a in ARMS for i in range(4)]
    contract['frozen_video_segment_cases'] = dict(study=STUDY, case_ids=ids, physical_init_states=4,
                                       no_initial_or_mid_state_intervention=True)
    cases = []
    for teacher, learned, slot in SOURCES:
        for arm in ARMS:
            for i in range(4):
                identifier = case_id(teacher, arm, i)
                evidence = dict(case_id=identifier, teacher=teacher, arm=arm, learned_arm=learned,
                    init_state_id=i, condition_id=key(teacher, arm),
                    full_capture=teacher == 43 and i in (2, 3))
                per_case = copy.deepcopy(contract)
                output = ROOT / 'cases' / identifier
                per_case['output_dir'] = str(output)
                capture = per_case['diagnostic_occupancy_capture']
                capture.update(mode='compact', full_conditions=[dict(suite='libero_10', task_id=2,
                    init_state_id=i)] if evidence['full_capture'] else [], trajectory_root=str(output / 'trajectories'))
                capture['passive_trace']['trace_root'] = str(output / 'continuous_traces')
                per_task = copy.deepcopy(bank['tasks'][0])
                per_task['episodes'] = [dict(init_state_id=j, condition_id=key(teacher, arm),
                    teacher_demo_indices=[teacher], video_ordinal=slot) for j in range(4)]
                per_case['adapter']['tasks'] = [per_task]
                episode = per_task['episodes'][i]
                prepared = PreparedOperatorLoRA(key(teacher, arm), episode_evidence(bank, per_task, episode))
                cases.append(dict(evidence=evidence, contract=per_case, prepared_adapter=prepared))
    return contract, task, cases


def run():
    started = time.monotonic()
    prepare_libero_config(ROOT / 'libero_config')
    banks, contracts, originals = load_originals()
    bank = copy.deepcopy(banks[(17, 'parent')])
    bank['tasks'] = [next(t for t in bank['tasks'] if t['global_task_id'] == 32)]
    contract = copy.deepcopy(contracts[17])
    contract['git'] = git_state(Path(__file__).resolve().parents[1])
    if not git_state_is_clean_pushed_or_frozen_authority(contract['git']):
        raise ValueError('actual consumer requires clean pushed frozen code')
    write_json_atomic(ROOT / 'run_contract.json', contract)
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    model, normalization, tokenizer = validate_worker_assets(contract)
    policy, preprocess, postprocess = load_policy(model, normalization['stats'], tokenizer, contract['policy'])
    adapter = FrozenOperatorAdapter(policy=policy, source=bank['source'], evaluation_adapter=bank,
        task_keys=(('libero_10', 2),), device=torch.device('cuda:0'), require_formal=True)
    bank = construct(adapter, tokenizer, banks, originals, contract['git'])
    contract, task, cases = build_cases(bank, contracts)
    write_json_atomic(ROOT / 'run_contract.json', contract)
    for case in cases:
        write_json_atomic(Path(case['contract']['output_dir']) / 'run_contract.json', case['contract'])
    adapter.bank = bank
    adapter.conditions = {r['condition_id']: r for r in bank['conditions']}
    adapter.tasks = {(r['suite'], r['task_id']): r for r in bank['tasks']}
    prediction = adapter.predict_action_chunk
    forward_records = []

    def measured(prepared, batch, **kwargs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        result = prediction(prepared, batch, **kwargs)
        b.record()
        b.synchronize()
        forward_records.append(dict(batch=len(prepared), seconds=a.elapsed_time(b) / 1000,
            reserved_GiB=torch.cuda.memory_reserved() / 2**30,
            allocated_GiB=torch.cuda.memory_allocated() / 2**30))
        return result

    adapter.predict_action_chunk = measured
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        envs, init_states = pool.switch(task)
        rollout_started = time.monotonic()
        rows = rollout_shard(envs=envs, init_states=init_states, task=task, state_ids=STATE_IDS,
            contract=contract, policy=policy, preprocess=preprocess, postprocess=postprocess,
            task_adapter=adapter, episode_contexts=cases)
        for case in cases:
            row = next(r for r in rows if r['frozen_video_segment_case']['case_id'] == case['evidence']['case_id'])
            shard = EvaluationShard(job_id=case['evidence']['case_id'], ordinal=0, suite='libero_10', task_id=2,
                init_state_ids=(row['init_state_id'],), horizon=520, estimated_cost=520,
                preferred_gpu=None)
            _validate_episode_row(row, contract=case['contract'], shard=shard, task=task)
        write_json_atomic(ROOT / 'results.json', dict(rows=rows, run_seconds=time.monotonic()-started,
            rollout_seconds=time.monotonic()-rollout_started, physical_batch=16, source_loading_count=1,
            generated_chunks=sum(len(r['policy_noise_seeds']) for r in rows), forward_records=forward_records,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, affinity=affinity,
            consumer_git=contract['git']))
    finally:
        pool.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(6)
    run()
