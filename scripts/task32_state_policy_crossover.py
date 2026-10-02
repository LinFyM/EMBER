#!/usr/bin/env python3
"""One registered four-case call to the canonical frozen PI05 rollout consumer."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ember.operator_writer.bank import (
    FrozenOperatorAdapter, PreparedOperatorLoRA, episode_evidence, inspect_bank,
)
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority, policy_noise_seed
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.topology import bind_current_process_to_cuda_numa

ROOT = Path('/data1/user/ymdai/ember_runs/task32_state_policy_crossover_20261002')
ORIGINAL = ROOT.parent / 'conditional_A_reexpression_diagnostic_20261002/Original'
PAIRS = ((17, 17), (17, 43), (43, 17), (43, 43))
CUT = 180


def case_id(i, j):
    return f'prefix{i}_policy{j}'


def load_inputs():
    banks, contracts, rows, traces, compact = {}, {}, {}, {}, {}
    for slot, teacher in enumerate((17, 43)):
        panel = ORIGINAL / f'bank/panel_teacher{slot}.json'
        contract = read_json(ORIGINAL / f'evaluation/teacher{slot}/run_contract.json')
        source = contract['adapter']['source']
        keys = tuple((r['suite'], r['task_id']) for r in contract['tasks'])
        states = {(r['suite'], r['task_id']): tuple(r['init_state_ids']) for r in contract['tasks']}
        banks[teacher] = inspect_bank(manifest_path=panel, source=source, task_keys=keys,
                                     evaluation_role='development_train', require_formal=True,
                                     task_init_state_ids=states)
        contracts[teacher] = contract
        row = next(r for r in read_json(ORIGINAL / f'evaluation/teacher{slot}/results.json')['rows']
                   if (r['suite'], r['task_id'], r['init_state_id']) == ('libero_10', 2, 2))
        if row['operator_read_write_lora']['teacher_demo'] != teacher:
            raise ValueError('original task32 teacher identity changed')
        rows[teacher] = row
        with np.load(row['continuous_control_trace']['trace']['path'], allow_pickle=False) as value:
            traces[teacher] = {k: value[k] for k in value.files}
        compact[teacher] = torch.load(row['occupancy_trajectory']['path'], map_location='cpu', weights_only=False)
        if ('observations' in compact[teacher] or compact[teacher]['capture_level'] != 'compact'
                or compact[teacher]['replan_steps'][36] != CUT):
            raise ValueError('original state2 compact/RGB/anchor provenance changed')
    common = tuple(rows[43]['policy_noise_seeds'])
    expected = tuple(policy_noise_seed(7, 'libero_10', 2, 2, k) for k in range(104))
    if (common != expected or tuple(rows[17]['policy_noise_seeds']) != common[:56]
            or rows[17]['success'] is not True or rows[43]['success'] is not False
            or rows[17]['scene_reference'] != rows[43]['scene_reference']):
        raise ValueError('original state/seed/positive-negative pairing changed')
    for teacher in (17, 43):
        trace, capture = traces[teacher], compact[teacher]
        physical = np.concatenate([a.numpy() for a in capture['executed_action_prefixes'][:36]])
        if not np.array_equal(physical, trace['actions'][:CUT]):
            raise ValueError('archived physical actions disagree between raw originals')
    return banks, contracts, rows, traces, compact, common


def build_cases(banks, original, rows, traces, compact, noise):
    contract = copy.deepcopy(original[17])
    contract.update(git=git_state(Path(__file__).resolve().parents[1]), output_dir=str(ROOT),
                    role='development_train', mode='screen', analysis_only=True)
    if not git_state_is_clean_pushed_or_frozen_authority(contract['git']):
        raise ValueError('crossover consumer requires a clean pushed detached source')
    contract['frozen_case_matrix'] = dict(schema_version='ember_saved_action_case_matrix_v1',
        case_ids=[case_id(i, j) for i, j in PAIRS], cut_control_steps=CUT,
        task=['libero_10', 2], init_state_id=2, policy_noise_source=file_record(
            ORIGINAL / 'evaluation/teacher1/results.json'))
    capture = contract['diagnostic_occupancy_capture']
    capture.update(mode='compact', external_prefix_commands=True,
                   full_conditions=[dict(suite='libero_10', task_id=2, init_state_id=2)])
    contract['diagnostic_stage_predicates']['full_conditions_only'] = False
    contract['parallel'].update(envs_per_replica=4, physical_gpu_count=1, worker_count=1)
    task = next(r for r in contract['tasks'] if (r['suite'], r['task_id']) == ('libero_10', 2))
    task['init_state_ids'] = [2, 2, 2, 2]
    contract['tasks'] = [task]
    cases = []
    for i, j in PAIRS:
        selected = next(r for r in banks[j]['tasks'] if (r['suite'], r['task_id']) == ('libero_10', 2))
        episode = next(r for r in selected['episodes'] if r['init_state_id'] == 2)
        evidence = dict(case_id=case_id(i, j), prefix_teacher=i, policy_teacher=j,
            cut_control_steps=CUT, follower_start_replan_index=36,
            original_prefix_trace=rows[i]['continuous_control_trace']['trace'],
            original_prefix_compact=rows[i]['occupancy_trajectory'],
            policy_bank=banks[j]['manifest'],
            policy_factors=next(c['factors'] for c in banks[j]['conditions']
                                if c['condition_id'] == episode['condition_id']),
            initial_scene=rows[i]['scene_reference'],
            prefix_RGB='newly_captured_during_replay; original_compact_has_no_RGB',
            prefix_normalized_chunks='archived_original_only; no_new_policy_forward')
        per_case = copy.deepcopy(contract)
        output = ROOT / 'cases' / evidence['case_id']
        per_case['output_dir'] = str(output)
        per_case['diagnostic_occupancy_capture']['trajectory_root'] = str(output / 'trajectories')
        per_case['diagnostic_occupancy_capture']['passive_trace']['trace_root'] = str(output / 'continuous_traces')
        cases.append(dict(evidence=evidence, contract=per_case,
            prepared_adapter=PreparedOperatorLoRA(episode['condition_id'], episode_evidence(banks[j], selected, episode)),
            physical_prefix=traces[i]['actions'][:CUT].copy(),
            archived_chunks=compact[i]['action_chunks'][:36], noise_seeds=noise))
    write_json_atomic(ROOT / 'run_contract.json', contract)
    write_json_atomic(ROOT / 'inputs.json', dict(originals={str(k): rows[k] for k in rows},
        original_contracts={str(k): file_record(ORIGINAL / f'evaluation/teacher{s}/run_contract.json')
                            for s, k in enumerate((17, 43))}, cases=[c['evidence'] for c in cases],
        no_held_reads=True, no_optimizer=True, no_Writer_or_native_compilation=True))
    return contract, task, cases


def run():
    banks, old, rows, traces, compact, noise = load_inputs()
    contract, task, cases = build_cases(banks, old, rows, traces, compact, noise)
    os.environ['EMBER_LIBERO_ASSETS_ROOT'] = contract['libero_paths']['assets']
    if prepare_libero_config(ROOT/'libero_config') != contract['libero_paths']:
        raise ValueError('registered LIBERO runtime paths changed')
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, normalization, tokenizer = validate_worker_assets(contract)
    policy, preprocess, postprocess = load_policy(model, normalization['stats'], tokenizer, contract['policy'])
    adapter = FrozenOperatorAdapter(policy=policy, source=banks[17]['source'],
        evaluation_adapter=banks[17], task_keys=tuple((r['suite'], r['task_id']) for r in banks[17]['tasks']),
        device=torch.device('cuda:0'), require_formal=True)
    selected43 = next(c for c in banks[43]['conditions'] if c['condition_id'] == 'task032_teacher43')
    adapter.conditions[selected43['condition_id']] = selected43
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        envs, init_states = pool.switch(task)
        result = rollout_shard(envs=envs, init_states=init_states, task=task, state_ids=(2, 2, 2, 2),
            contract=contract, policy=policy, preprocess=preprocess, postprocess=postprocess,
            task_adapter=adapter, episode_contexts=cases)
        for row in result:
            write_json_atomic(ROOT / 'cases' / row['frozen_case']['case_id'] / 'row.json', row)
        receipt = dict(rows=result, run_seconds=time.monotonic() - started, physical_batch=4,
            peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), cpu_affinity=affinity,
            execution_git=contract['git'])
        write_json_atomic(ROOT / 'results.json', receipt)
        print(json.dumps(dict(event='four_cases_finished', **{k:v for k,v in receipt.items() if k != 'rows'})), flush=True)
    finally:
        pool.close()
        adapter.close()


def channel_summary(array):
    return {name:dict(mean=array[:, sl].mean(axis=0).tolist(),
                      L2=float(np.linalg.norm(array[:, sl])), RMS=float(np.sqrt(np.mean(array[:, sl]**2))))
            for name, sl in [('translation3', slice(0, 3)), ('rotation3', slice(3, 6)), ('gripper1', slice(6, 7))]}


def decomposition(table):
    a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    dw, ds = ((b-a)+(d-c))/2, ((c-a)+(d-b))/2
    return dict(D=d-a, D_W=dw, D_s=ds, I=d-c-b+a), float(np.max(np.abs((d-a)-dw-ds)))


def read_case(row, old_traces, compact, noise):
    e = row['frozen_case']; i, j = e['prefix_teacher'], e['policy_teacher']
    full = torch.load(row['occupancy_trajectory']['path'], map_location='cpu', weights_only=False)
    with np.load(row['continuous_control_trace']['trace']['path'], allow_pickle=False) as raw:
        trace = {k:raw[k] for k in raw.files}
    if (full['capture_level'] != 'full' or full['replan_steps'][36] != CUT
            or full['command_kinds'][:36] != ('external_saved_action',)*36
            or any(c is not None for c in full['action_chunks'][:36])
            or len(full['observations']) != len(full['action_chunks'])
            or tuple(row['policy_noise_seeds']) != noise[:len(row['policy_noise_seeds'])]
            or row['operator_read_write_lora']['teacher_demo'] != j
            or not np.array_equal(trace['actions'][:CUT], old_traces[i]['actions'][:CUT])):
        raise ValueError('actual replay/full capture/LoRA/noise clock changed')
    images = {k:v.numpy() for k,v in full['observations'][36].items() if 'images' in k}
    if len(images) != 2 or any(v.shape != (1,3,256,256) for v in images.values()):
        raise ValueError('crossover full capture lacks actual dual256 RGB')
    for chunk in full['action_chunks'][36:]:
        if chunk.shape != (1,50,7) or not torch.isfinite(chunk).all():
            raise ValueError('crossover generated full50 action capture incomplete')
    f, physical = full['action_chunks'][36][0].numpy(), full['executed_action_prefixes'][36].numpy()
    anchor = {k:trace[k][CUT] for k in ('eef_pos','eef_quat','gripper_qpos','body_positions','predicates')}
    error = {k:float(np.max(np.abs(v.astype(float)-old_traces[i][k][CUT].astype(float))))
             for k,v in anchor.items()}
    error['raw_state8'] = float((full['states'][36] - compact[i]['states'][36]).abs().max())
    anchor['raw_state8'] = full['states'][36].numpy()
    anchor.update(images)
    index = [str(s) for s in trace['body_names']].index('moka_pot_1')
    lift = trace['body_positions'][:,index,2] - trace['body_positions'][0,index,2]
    raised, closing = np.flatnonzero(lift >= .03), np.flatnonzero(trace['actions'][CUT:,6] > 0)
    report = dict(case_id=e['case_id'],prefix_teacher=i,policy_teacher=j,
        success=row['success'],steps=row['steps'],first_lift3cm_step=int(raised[0]) if len(raised) else None,
        max_lift_cm=float(lift.max()*100),first_close_action_index=int(closing[0]+CUT) if len(closing) else None,
        final_predicates=row['stage_predicates']['final_satisfied'],
        predicate_transitions=row['stage_predicates']['transitions'],anchor_original_errors=error,
        first_physical_action=physical[0].tolist(),first_normalized_action=f[0].tolist(),
        full_trajectory=row['occupancy_trajectory'],continuous=row['continuous_control_trace']['trace'])
    return f, physical, report, anchor


def readback():
    _, _, originals, old_traces, compact, noise = load_inputs()
    rows = read_json(ROOT / 'results.json')['rows']
    if len(rows) != 4 or {r['frozen_case']['case_id'] for r in rows} != {case_id(i,j) for i,j in PAIRS}:
        raise ValueError('four crossover cases incomplete')
    first, physical, reports, anchors = np.zeros((2,2,50,7)), np.zeros((2,2,5,7)), [], {}
    for row in rows:
        i, j = row['frozen_case']['prefix_teacher'], row['frozen_case']['policy_teacher']
        p, q = (17,43).index(i), (17,43).index(j)
        first[p,q], physical[p,q], report, anchors[(i,j)] = read_case(row, old_traces, compact, noise)
        reports.append(report)
    pair_errors = {str(i):{k:float(np.max(np.abs(anchors[(i,17)][k].astype(float)-anchors[(i,43)][k].astype(float))))
                         for k in anchors[(i,17)]} for i in (17,43)}
    arrays, stats = {}, {}
    for name, table in [('normalized_full50',first),('physical_first5',physical)]:
        terms, residual = decomposition(table)
        arrays[name+'_F'] = table
        for key, value in terms.items():arrays[name+'_'+key] = value
        stats[name] = dict(terms={k:channel_summary(v) for k,v in terms.items()}, algebra_max_error=residual,
                          policy_effect_s17=channel_summary(table[0,1]-table[0,0]),
                          policy_effect_s43=channel_summary(table[1,1]-table[1,0]))
    np.savez_compressed(ROOT / 'first_actions_and_decomposition.npz', **arrays)
    diag = {(r['prefix_teacher'],r['policy_teacher']):r['success'] for r in reports}
    diagonals = diag[(17,17)] is True and diag[(43,43)] is False
    paired = all(max(v.values()) <= 1e-6 for v in pair_errors.values())
    result = dict(status='complete',cases=sorted(reports,key=lambda r:(r['prefix_teacher'],r['policy_teacher'])),
        diagonal_positive_negative_reproduced=diagonals,same_prefix_pair_valid=paired,
        identification='case contrast available' if diagonals and paired else '本次无法识别预定对比',
        anchor_same_prefix_errors=pair_errors,first_actions=stats,
        raw_first_actions=str(ROOT / 'first_actions_and_decomposition.npz'),
        original_state2_RGB_available=False,no_new_training_or_held_reads=True)
    write_json_atomic(ROOT / 'readback.json', result)
    write_json_atomic(ROOT / 'completion.json', dict(status='complete',rows=4,full=4,readback=str(ROOT/'readback.json'),
        identification=result['identification'],additional_probe_or_training_authorized=False))
    print(json.dumps(dict(event='readback_complete',diagonals=diagonals,paired=paired)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run','readback'))
    args = parser.parse_args()
    {'run':run, 'readback':readback}[args.command]()
