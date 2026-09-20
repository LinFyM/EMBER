"""Recreate complete frozen Writer and MT-BC correct400 Validation curves."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get('EMBER_COVERAGE_STUDY', '/data0/user/ymdai/ember_runs/coverage_retraining_20260920'))
source = json.loads((STUDY / 'evaluation/source_validation/results.json').read_text())
assert source['overall']['episodes'] == 400
source_score = source['overall']['successes']
series = []
for method, expected, selected, stopped in (('writer', 9, 1000, 1800), ('mtbc', 10, 300, 500)):
    rows = list(csv.DictReader((HERE / f'{method}_validation_nodes.csv').open()))
    assert len(rows) == expected and all(int(row['validation_rows']) == 400 for row in rows)
    steps = [int(row['optimizer_updates']) for row in rows]
    scores = [int(row['correct_successes']) for row in rows]
    assert [int(row['optimizer_updates']) for row in rows if row['selected'] == 'True'] == [selected]
    assert [int(row['optimizer_updates']) for row in rows if row['stopping_node'] == 'True'] == [stopped]
    series.append((method, steps, scores, selected, stopped))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.75, wspace=.22)
fig.text(.075, .94, 'Complete correct400 Validation curves', fontsize=17, weight='bold')
fig.text(.075, .875, 'New 24/8/8 + 12 protocol · fixed 8 tasks × 50 matched initial states and video ordinals', fontsize=10)
for ax, (method, steps, scores, selected, stopped) in zip(axes, series):
    color = '#0072B2' if method == 'writer' else '#D55E00'
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.7)
    ax.set_axisbelow(True)
    ax.axhline(source_score, color='#555555', linestyle='--', linewidth=1.1)
    ax.plot(steps, scores, color=color, marker='o', markersize=5.5, linewidth=1.5)
    score = scores[steps.index(selected)]
    ax.scatter([selected], [score], marker='*', s=210, color='#E69F00', zorder=5)
    ax.axvline(stopped, color='#555555', linestyle=':', linewidth=1)
    ax.annotate(f'Selected {score}', (selected, score), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    ax.annotate(f'Stop {scores[-1]}', (stopped, scores[-1]), xytext=(-7, -18), textcoords='offset points', ha='right', fontsize=9)
    ax.set_title('EMBER Writer' if method == 'writer' else 'MT-BC shared adapter', loc='left', fontsize=12)
    ax.set_xticks(steps[::2] + ([stopped] if stopped not in steps[::2] else []))
    ax.set_xlim(min(steps)-.07*stopped, stopped*1.05)
    ax.set_ylim(40, 175)
    ax.set_xlabel('Optimizer update (method-specific semantics)')
    ax.set_ylabel('Successful rollouts / 400')
fig.text(.075, .08, 'Source: 51/400. Writer update samples 4 of 36 tasks; MT-BC update covers all 36. Horizontal update counts are not equal exposure.', fontsize=9)
fig.text(.075, .045, 'Both models use one fresh training seed. Selection and stopping use only complete correct400 nodes under the registered rules.', fontsize=9)
for suffix in ('svg', 'png'):
    fig.savefig(HERE / f'training_validation_curves.{suffix}', dpi=180, facecolor='white')
plt.close(fig)
