"""Recreate the discrete checkpoint figure from the adjacent paired_summary.json."""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / 'paired_summary.json').read_text())
assert summary['complete'] and summary['steps'] == [300, 600, 900, 1200]
rows = [summary['matched']['validation'][str(step)]['comparison'] for step in summary['steps']]
assert all(row['rows'] == 400 for row in rows)
reference = np.array([row['reference_successes'] for row in rows])
candidate = np.array([row['candidate_successes'] for row in rows])
delta = 100 * (candidate - reference) / 400
interval = 100 * np.array([row['task_cluster_bootstrap_rate_difference_95ci'] for row in rows])
x = np.arange(4)
# Two restrained roots; open square versus filled circle also works in grayscale.
blue, orange, ink, grid = '#0072B2', '#E69F00', '#252525', '#dddddd'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'text.color': ink, 'axes.labelcolor': ink,
                     'xtick.color': ink, 'ytick.color': ink, 'axes.edgecolor': ink})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
fig.subplots_adjust(left=.075, right=.975, bottom=.23, top=.74, wspace=.31)
fig.text(.075, .94, 'Matched A vs learned frame-set', fontsize=19, weight='bold')
fig.text(.075, .885, 'Validation8: 400 paired task-state-video conditions per checkpoint; one training seed', fontsize=11)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=grid, linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xticks(x, summary['steps'])
    ax.set_xlim(-.45, 3.45)
    ax.set_xlabel('Registered optimizer update', labelpad=9)
    for label in ax.get_xticklabels()[2:]:
        label.set_fontweight('bold')
left, right = axes
left.set_title('Observed checkpoint success', loc='left', fontsize=13, pad=36)
left.plot(x-.09, reference, linestyle='none', marker='s', markersize=7,
          markerfacecolor='white', markeredgewidth=1.6, color=blue, label='A (historical)')
left.plot(x+.09, candidate, linestyle='none', marker='o', markersize=7,
          color=orange, label='Frame-set (fresh)')
left.legend(loc='lower left', bbox_to_anchor=(-.02, 1.01), frameon=False,
            ncol=2, fontsize=10, handletextpad=.5, columnspacing=1.1)
left.set_ylabel('Successful rollouts / 400')
left.set_ylim(0, math.ceil((max(reference.max(), candidate.max())+25)/20)*20)
for i, (a, b) in enumerate(zip(reference, candidate)):
    left.annotate(str(a), (i-.09, a), xytext=(-5, 9), textcoords='offset points', ha='center', fontsize=10)
    left.annotate(str(b), (i+.09, b), xytext=(5, -17), textcoords='offset points', ha='center', fontsize=10)
right.set_title('Frame-set minus A', loc='left', fontsize=13, pad=36)
right.text(0, 1.045, '95% task-cluster bootstrap intervals', transform=right.transAxes, fontsize=10)
right.axhline(0, color=ink, linewidth=1, linestyle='--')
right.errorbar(x, delta, yerr=np.stack((delta-interval[:, 0], interval[:, 1]-delta)),
               fmt='o', color=orange, ecolor=orange, capsize=5, markersize=7, linewidth=1.6)
right.set_ylabel('Success-rate difference (percentage points)')
right.set_ylim(math.floor((interval.min()-3)/5)*5, math.ceil((interval.max()+3)/5)*5)
for i, value in enumerate(delta):
    right.annotate(f'{value:+.2f}', (i, value), xytext=(9, 5), textcoords='offset points', fontsize=10)
fig.text(.075, .09, 'Primary comparison: 900 and 1200. Fixed checkpoint observations; no interpolation or peak selection.', fontsize=10)
fig.text(.075, .045, '8 task clusters; 20,000 bootstrap draws (seed 20260915). An interval containing zero does not establish equivalence.', fontsize=9.5)
fig.savefig(ROOT / 'checkpoint_comparison.png', dpi=180, facecolor='white',
            metadata={'Description': 'Source: paired_summary.json; matched A/frame-set validation results.'})
plt.close(fig)
