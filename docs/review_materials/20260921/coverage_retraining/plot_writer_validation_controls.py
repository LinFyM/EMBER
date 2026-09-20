"""Recreate the frozen Writer Validation video-control figure from formal results."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get('EMBER_COVERAGE_STUDY', '/data0/user/ymdai/ember_runs/coverage_retraining_20260920'))
arms = ['correct', 'same_task_other', 'cross_suite_wrong', 'shuffled', 'reversed']
labels = ['Correct', 'Other', 'Wrong', 'Shuffled', 'Reversed']
colors = ['#0072B2', '#56B4E9', '#D55E00', '#CC79A7', '#9A6324']
scores = []
for arm in arms:
    suffix = '' if arm == 'correct' else ('other' if arm == 'same_task_other' else arm)
    panel = json.loads((STUDY / f'evaluation/writer_00001000{"_" + suffix if suffix else ""}/results.json').read_text())
    assert panel['overall']['episodes'] == 400 and panel['role'] == 'validation'
    scores.append(panel['overall']['successes'])
assert scores == [117, 119, 107, 86, 62]
effects, lower, upper = [], [], []
for arm in arms[1:]:
    suffix = 'other' if arm == 'same_task_other' else arm
    evidence = json.loads((HERE / f'writer_1000_correct_vs_{suffix}_validation_bootstrap.json').read_text())
    value = evidence['point_estimate']
    lo, hi = evidence['percentile_95_interval']
    effects.append(value); lower.append(value - lo); upper.append(hi - value)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), gridspec_kw={'width_ratios': [1, 1.18]})
fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.76, wspace=.26)
fig.text(.075, .94, 'Selected Writer: Validation video controls', fontsize=17, weight='bold')
fig.text(.075, .875, 'Step 1000 · frozen checkpoint · eight tasks × 50 paired initial states per arm', fontsize=10)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.7)
    ax.set_axisbelow(True)
left, right = axes
x = np.arange(5)
left.bar(x, scores, color=colors, width=.62)
left.set_xticks(x, labels, rotation=20, ha='right')
left.set_ylim(0, 137)
left.set_ylabel('Successful rollouts / 400')
left.set_title('Complete formal panels', loc='left')
for i, score in enumerate(scores):
    left.text(i, score + 2, str(score), ha='center', fontsize=10)
rx = np.arange(4)
right.axhline(0, color='#444444', linewidth=1, linestyle='--')
right.errorbar(rx, effects, yerr=np.array([lower, upper]), fmt='o', color='#0072B2',
               ecolor='#0072B2', capsize=5, markersize=6, linewidth=1.6)
right.set_xticks(rx, labels[1:], rotation=20, ha='right')
right.set_ylabel('Control − correct (percentage points)')
right.set_xlim(-.3, 3.6)
right.set_ylim(-28, 7)
right.set_title('Paired task-cluster contrasts', loc='left')
for i, value in enumerate(effects):
    right.annotate(f'{value:+.2f}', (i, value), xytext=(7, 5), textcoords='offset points', fontsize=9)
fig.text(.075, .085, 'Error bars: 95% percentile intervals from 20,000 resamples of the eight tasks; all 50 paired states stay within each task.', fontsize=9)
fig.text(.075, .045, 'Wrong-video interval crosses zero; reversed-video decline is clear on this Validation panel. One Writer training seed.', fontsize=9)
for suffix in ('svg', 'png'):
    fig.savefig(HERE / f'writer_validation_controls.{suffix}', dpi=180, facecolor='white')
plt.close(fig)
