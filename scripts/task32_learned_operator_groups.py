#!/usr/bin/env python3
"""Registered 16-case frozen bank selection on the canonical rollout consumer."""
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

from ember.lora import LoRATarget, validate_lora_state
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

ROOT = Path('/data1/user/ymdai/ember_runs/task32_learned_operator_groups_20261002')
OLD = ROOT.parent / 'operator_learning_limit_diagnosis_20260930'
STUDY = ROOT.name
SOURCES = ((17, 'D', 0), (43, 'S', 1))
ARMS = ('Q', 'R')
STATE_IDS = tuple(i for _ in SOURCES for _ in ARMS for i in range(4))


def key(teacher, arm):
    return f'task032_teacher{teacher}_{arm}'


def case_id(teacher, arm, state):
    return f'teacher{teacher}_{arm}_init{state}'


def target_groups(lora):
    names = [target['name'] for target in lora['targets']]
    q = [name for name in names if name.endswith('.q_proj')]
    r = [name for name in names if name not in q]
    if (len(q) != 18 or len(r) != 20 or len(set(names)) != 38
            or sum(name.endswith('.v_proj') for name in r) != 18
            or {'model.action_in_proj', 'model.action_out_proj'} !=
               {name for name in r if not name.endswith('.v_proj')}):
        raise ValueError('registered full38 Q18/R20 target groups changed')
    return {'Q': q, 'R': r}


def select_complete(parent, learned, lora, arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    topology = SimpleNamespace(rank=lora['adapter']['rank'],
                               targets=tuple(LoRATarget(**t) for t in lora['targets']))
    for state in (parent, learned):
        validate_lora_state(state, topology)
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError('nonfinite original complete factor')
    selected = set(target_groups(lora)[arm])
    result = {}
    for target in lora['targets']:
        name = target['name']
        a, b = (f'{name}.lora_{factor}.default.weight' for factor in ('A', 'B'))
        if not torch.equal(parent[a], learned[a]):
            raise ValueError(f'fixed parent A changed at {name}')
        result[a] = parent[a]
        result[b] = learned[b] if name in selected else parent[b]
    validate_lora_state(result, topology)
    return result


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


def prepare():
    identity = git_state(Path(__file__).resolve().parents[1])
    if not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('bank construction requires clean pushed frozen code')
    banks, contracts, originals = load_originals()
    parent = banks[(17, 'parent')]
    bank = copy.deepcopy(parent)
    bank.pop('learning_limit_panel')
    bank.update(mode='learned_operator_groups',
                condition_factors='complete_A0_plus_S_B0_plus_M', conditions=[], tasks=[],
                learned_operator_groups=dict(study=STUDY, targets=target_groups(bank['lora']),
                    construction_git=identity, parent_training_git='e2afbfd7c997e3f792921600608efa2fa3c1b25a',
                    learning_git='092a0ae83080f85ddff0e4129c92fb42aa527c0c',
                    original_reading_git='5313257c8070a792bbec0c6345c98376ed5cd1ea'))
    task = copy.deepcopy(next(t for t in parent['tasks'] if t['global_task_id'] == 32))
    task['episodes'] = []
    provenance = []
    (ROOT / 'bank').mkdir(exist_ok=True)
    for teacher, learned, slot in SOURCES:
        p, a = (OLD / f'{arm}/bank/task032_teacher{teacher}.safetensors' for arm in ('parent', learned))
        parent_state, learned_state = load_file(str(p)), load_file(str(a))
        for arm in ARMS:
            path = ROOT / 'bank' / f'{key(teacher, arm)}.safetensors'
            if path.exists():
                raise ValueError('refusing to overwrite a derived bank')
            save_file(select_complete(parent_state, learned_state, bank['lora'], arm), str(path))
            bank['conditions'].append(dict(condition_id=key(teacher, arm), global_task_id=32,
                                           teacher_demo=teacher, factors=file_record(path)))
            provenance.append(dict(condition_id=key(teacher, arm), teacher=teacher, arm=arm,
                learned_arm=learned, parent_factors=file_record(p), learned_factors=file_record(a),
                parent_bank=banks[(teacher, 'parent')]['manifest'],
                learned_bank=banks[(teacher, learned)]['manifest'],
                A='parent', B='direct_existing_complete_factor_selection_no_additional_B0'))
    task['episodes'] = [dict(init_state_id=i, condition_id=key(17, 'Q'),
                            teacher_demo_indices=[17], video_ordinal=0) for i in range(4)]
    bank['tasks'] = [task]
    bank_path = ROOT / 'bank/manifest.json'
    write_json_atomic(bank_path, bank)
    bank['manifest'] = file_record(bank_path)
    write_json_atomic(ROOT / 'prepared_bank.json', bank)
    write_json_atomic(ROOT / 'inputs.json', dict(provenance=provenance,
        original_rows={f'{t}_{a}': list(rows.values()) for (t, a), rows in originals.items()},
        original_contracts={str(t): file_record(OLD / f'parent/evaluation/teacher{s}/run_contract.json')
                            for t, _, s in SOURCES},
        no_held_reads=True, no_new_actions_or_FM=True, no_Writer_native_or_optimizer=True))
    return bank, contracts


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
    contract['frozen_lora_cases'] = dict(study=STUDY, case_ids=ids, physical_init_states=4,
                                       no_initial_or_mid_state_intervention=True)
    cases = []
    for teacher, learned, slot in SOURCES:
        for arm in ARMS:
            for i in range(4):
                identifier = case_id(teacher, arm, i)
                evidence = dict(case_id=identifier, teacher=teacher, arm=arm, learned_arm=learned,
                    init_state_id=i, condition_id=key(teacher, arm),
                    full_capture=(teacher, i) in ((17, 0), (43, 3)))
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
    bank = read_json(ROOT / 'prepared_bank.json')
    _, contracts, _ = load_originals()
    contract, task, cases = build_cases(bank, contracts)
    if not git_state_is_clean_pushed_or_frozen_authority(contract['git']):
        raise ValueError('actual consumer requires clean pushed frozen code')
    write_json_atomic(ROOT / 'run_contract.json', contract)
    for case in cases:
        write_json_atomic(Path(case['contract']['output_dir']) / 'run_contract.json', case['contract'])
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
            row = next(r for r in rows if r['frozen_lora_case']['case_id'] == case['evidence']['case_id'])
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
    parser.add_argument('phase', choices=('prepare', 'run'))
    args = parser.parse_args()
    torch.set_num_threads(6)
    if args.phase == 'prepare':
        prepare()
    else:
        run()
