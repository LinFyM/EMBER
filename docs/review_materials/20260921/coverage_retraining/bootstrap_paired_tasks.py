"""Task-cluster bootstrap for one complete paired 8-task, 50-state panel.

Resample the eight tasks with replacement, retaining all fifty paired state rows
inside each sampled task. Candidate minus reference, in percentage points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(source: Path, *, seed: int, draws: int) -> dict:
    payload = json.loads(source.read_text())
    comparison = payload['comparison']
    tasks = comparison['per_task']
    assert comparison['rows'] == 400 and len(tasks) == 8 and all(task['rows'] == 50 for task in tasks)
    assert sum(task['reference_successes'] for task in tasks) == comparison['reference_successes']
    assert sum(task['candidate_successes'] for task in tasks) == comparison['candidate_successes']
    delta = np.array([task['candidate_successes'] - task['reference_successes'] for task in tasks], dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(tasks), size=(draws, len(tasks)))
    samples = delta[indices].sum(axis=1) / 400 * 100
    lo, hi = np.quantile(samples, [.025, .975])
    return {'schema_version': 'ember_coverage_paired_task_cluster_bootstrap_v1',
            'comparison': str(source.resolve()), 'task_count': 8, 'paired_states_per_task': 50,
            'draws': draws, 'seed': seed, 'resampling_unit': 'task; all 50 paired states retained',
            'estimand': 'candidate_minus_reference_success_rate_percentage_points',
            'point_estimate': (comparison['candidate_successes'] - comparison['reference_successes']) / 4,
            'percentile_95_interval': [float(lo), float(hi)],
            'per_task_success_difference': [int(x) for x in delta]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--draws', type=int, default=20000)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.comparison, seed=args.seed, draws=args.draws), indent=2) + '\n')
