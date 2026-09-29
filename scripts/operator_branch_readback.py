"""Pair §28 B20 branches against saved microbatch-10 parents and old references."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/operator_branch_intervention_20260929')
PRIOR = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/functional_credit_transport')
MATCHED = Path('/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929/self_conditioned_readout/matched_parent10')
TASKS = (0, 12, 20, 32)
ARMS = ('no_q_write', 'no_v_write', 'no_io_write')
METRICS = ('first5', 'full50', 'valid_future', 'first5_motion6', 'first5_gripper1',
           'full50_motion6', 'full50_gripper1', 'valid_motion6', 'valid_gripper1')


def scores(prediction, target, valid):
    pred, goal = prediction.float(), target.float()
    if pred.shape != goal.shape or pred.shape != (20, 50, 7) or valid.shape != (20,):
        raise ValueError('B20 generated prediction/target/valid shape changed')
    error = (pred - goal).square()
    mask = torch.arange(50)[None, :] < valid[:, None]
    sign_error = torch.sign(pred[:, :, 6]) != torch.sign(goal[:, :, 6])
    return {
        'first5': float(error[:, :5].mean()),
        'full50': float(error.mean()),
        'valid_future': float(error[mask].mean()),
        'first5_motion6': float(error[:, :5, :6].mean()),
        'first5_gripper1': float(error[:, :5, 6].mean()),
        'full50_motion6': float(error[:, :, :6].mean()),
        'full50_gripper1': float(error[:, :, 6].mean()),
        'valid_motion6': float(error[:, :, :6][mask].mean()),
        'valid_gripper1': float(error[:, :, 6][mask].mean()),
        'sign_error_first5': int(sign_error[:, :5].sum()),
        'sign_error_full50': int(sign_error.sum()),
        'sign_error_valid': int(sign_error[mask].sum()),
        'per_query_first5': error[:, :5].mean((1, 2)).tolist(),
        'per_query_motion6': error[:, :5, :6].mean((1, 2)).tolist(),
        'per_query_gripper1': error[:, :5, 6].mean(1).tolist(),
        'per_query_sign_error_first5': sign_error[:, :5].sum(1).tolist(),
    }


def delta(current, parent):
    result = {key: current[key] - parent[key] for key in METRICS}
    result.update({key: current[key] - parent[key]
                   for key in ('sign_error_first5', 'sign_error_full50', 'sign_error_valid')})
    for key in ('per_query_first5', 'per_query_motion6', 'per_query_gripper1',
                'per_query_sign_error_first5'):
        result[key] = [a-b for a,b in zip(current[key], parent[key], strict=True)]
    return result


def main():
    receipts = [json.loads((ROOT / f'launch/jobs/group{group}.result.json').read_text())
                for group in (0, 1)]
    completions = [json.loads((ROOT / f'group{group}/completion.json').read_text())
                   for group in (0, 1)]
    if any(row['exit_code'] != 0 for row in receipts) or any(
            row['status'] != 'complete' or row['new_paths'] != 12 for row in completions):
        raise ValueError('both complete 12-path group exits required')
    rows = [row for group in (0, 1)
            for row in json.loads((ROOT / f'group{group}/rows.json').read_text())]
    if (len(rows) != 24 or len({(r['task'], r['teacher'], r['arm']) for r in rows}) != 24
            or {r['arm'] for r in rows} != set(ARMS)):
        raise ValueError('fixed 24-path panel incomplete or duplicated')
    parent_rows = [row for group in (0, 1)
                   for row in json.loads((MATCHED / f'group{group}/rows.json').read_text())]
    parents = {(row['task'], row['teacher']): row for row in parent_rows}
    if len(parents) != 8:
        raise ValueError('eight matched microbatch10 parents missing')
    prior = json.loads((PRIOR / 'readback.json').read_text())
    old_reference = {(row['task'], row['condition'], row.get('teacher')): row['risk']
                     for row in prior['parent_and_reference_B']}
    out_rows = []
    for row in rows:
        task, teacher, arm = row['task'], row['teacher'], row['arm']
        if task not in TASKS or arm not in ARMS:
            raise ValueError('unexpected task or arm')
        raw = torch.load(row['raw'], map_location='cpu', weights_only=False)
        parent_row = parents[(task, teacher)]
        parent = torch.load(parent_row['raw'], map_location='cpu', weights_only=False)
        reference = torch.load(row['B_query_noise_target'], map_location='cpu', weights_only=False)
        if (raw['task'] != task or raw['teacher'] != teacher or raw['arm'] != arm
                or raw['microbatch'] != 10 or parent['microbatch'] != 10
                or raw['queries'] != row['queries'] or row['queries'] != reference['queries']
                or parent['B_query_noise_target'] != row['B_query_noise_target']
                or raw['prediction50x7'].shape != (20, 50, 7)
                or not torch.equal(raw['prediction50x7'], raw['full_latent50x32'][:, :, :7])
                or not torch.isfinite(raw['prediction50x7']).all()):
            raise ValueError('saved B20 prediction/query/latent/parent pairing changed')
        actual, baseline = (scores(item['prediction50x7'], reference['target'],
                                   reference['valid_future_lengths'])
                            for item in (raw, parent))
        for key in ('first5', 'full50', 'valid_future', 'first5_motion6', 'first5_gripper1'):
            if abs(actual[key] - raw['risk'][key]) > 1e-8 or abs(baseline[key] - parent['risk'][key]) > 1e-8:
                raise ValueError('recomputed score differs from actual saved prediction')
        out_rows.append({
            'task': task, 'teacher': teacher, 'arm': arm,
            'queries': reference['queries'], 'query_ref': row['B_query_noise_target'],
            'candidate_raw': row['raw'], 'matched_parent_raw': parent_row['raw'],
            'removed_M_targets': raw['removed_M_targets'],
            'candidate': actual, 'parent10': baseline, 'candidate_minus_parent10': delta(actual, baseline),
        })
    by_task = {}
    by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in out_rows if row['arm'] == arm]
        by_arm[arm] = {
            'mean_delta': {key: sum(r['candidate_minus_parent10'][key] for r in arm_rows)/8
                           for key in (*METRICS, 'sign_error_first5', 'sign_error_valid')},
            'better_conditions_first5': sum(r['candidate_minus_parent10']['first5'] < 0 for r in arm_rows),
            'worse_conditions_first5': sum(r['candidate_minus_parent10']['first5'] > 0 for r in arm_rows),
        }
        for task in TASKS:
            pair = [r for r in arm_rows if r['task'] == task]
            if len(pair) != 2:
                raise ValueError('teacher pair incomplete')
            by_task[f'{arm}/task{task:03d}'] = {
                'teachers': [r['teacher'] for r in pair],
                'teacher_first5_deltas': [r['candidate_minus_parent10']['first5'] for r in pair],
                'mean_delta': {key: sum(r['candidate_minus_parent10'][key] for r in pair)/2
                               for key in (*METRICS, 'sign_error_first5', 'sign_error_valid')},
            }
    gpu_hours = sum(row['gpu_hours'] for row in receipts)
    if gpu_hours > .75:
        raise ValueError('new GPU-hour hard limit exceeded')
    output = {
        'task': 'operator_branch_intervention_20260929',
        'boundary': '24 frozen B20 branches, no new video compilation, learning, environment or held',
        'comparison': 'same teacher/query/noise, candidate microbatch10 vs matched parent microbatch10',
        'gripper_sign': 'sign of normalized action channel 6 equals evaluator sign under q01=-1,q99=+1',
        'rows': out_rows, 'by_task': by_task, 'by_arm': by_arm,
        'old_batch5_reference': {f'task{task:03d}': {
            'public_beta': old_reference[(task, 'public_beta', None)],
            'MT300': old_reference[(task, 'MT300', None)],
            'parent_by_teacher': {str(teacher): old_reference[(task, 'parent_B', teacher)]
                                  for teacher in [r['teacher'] for r in out_rows if r['task'] == task and r['arm'] == ARMS[0]]},
        } for task in TASKS},
        'group_completions': completions, 'exit_receipts': receipts,
        'new_gpu_hours': gpu_hours,
    }
    (ROOT / 'readback.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'gpu_hours': gpu_hours, 'by_arm': by_arm, 'by_task': by_task}, indent=2))


if __name__ == '__main__':
    main()
