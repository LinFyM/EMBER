"""Recreate the frozen Writer correct400 Validation curve from verified node CSV."""
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
rows = list(csv.DictReader((HERE / 'writer_validation_nodes.csv').open()))
assert len(rows) == 9
steps = [int(row['optimizer_updates']) for row in rows]
scores = [int(row['correct_successes']) for row in rows]
assert steps == list(range(200, 1801, 200))
assert all(int(row['validation_rows']) == 400 for row in rows)
assert [int(row['optimizer_updates']) for row in rows if row['selected'] == 'True'] == [1000]
assert [int(row['optimizer_updates']) for row in rows if row['stopping_node'] == 'True'] == [1800]
source = json.loads((STUDY / 'evaluation/source_validation/results.json').read_text())
assert source['overall']['episodes'] == 400
source_score = source['overall']['successes']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, ax = plt.subplots(figsize=(10.5, 5.4))
fig.subplots_adjust(left=.09, right=.97, bottom=.25, top=.76)
fig.text(.09, .94, 'EMBER Writer: complete Validation nodes', fontsize=17, weight='bold')
fig.text(.09, .875, 'New 24/8/8 + 12 protocol · correct K=1 · 400 matched rollouts per checkpoint', fontsize=10.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#dddddd', linewidth=.7)
ax.set_axisbelow(True)
ax.axhline(source_score, color='#4d4d4d', linestyle='--', linewidth=1.3, label=f'Frozen Source: {source_score}/400')
ax.plot(steps, scores, color='#0072B2', marker='o', markersize=6, linewidth=1.7, label='Writer correct400')
ax.scatter([1000], [117], marker='*', s=230, color='#E69F00', zorder=5, label='Selected: step 1000')
ax.axvline(1800, color='#a0522d', linestyle=':', linewidth=1.2)
ax.annotate('Selected 117', (1000, 117), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
ax.annotate('Stop 92', (1800, 92), xytext=(-9, -23), textcoords='offset points', ha='right', fontsize=10)
ax.set_xlim(120, 1880)
ax.set_ylim(40, 135)
ax.set_xticks(steps)
ax.set_xlabel('Registered optimizer update', labelpad=10)
ax.set_ylabel('Successful rollouts / 400')
ax.legend(loc='upper right', frameon=False, fontsize=9)
fig.text(.09, .065, 'The registered sustained-decline rule stopped training at step 1800; selection uses the single highest complete correct400 node.', fontsize=9.5)
for suffix in ('svg', 'png'):
    fig.savefig(HERE / f'writer_validation_curve.{suffix}', dpi=180, facecolor='white')
plt.close(fig)
