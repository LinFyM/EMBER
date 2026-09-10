"""Recount the exported outcomes and verify key paired comparisons (stdlib only)."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def load_rows(panel):
    return [json.loads(line) for name in panel['rows_files']
            for line in (ROOT / name).read_text().splitlines()]


def paired(left, right):
    key = lambda row: (row['suite'], row['task_id'], row['init_state_id'])
    a, b = ({key(row): row for row in rows} for rows in (left, right))
    assert a.keys() == b.keys()
    for k, row in a.items():
        other = b[k]
        for field in ('language', 'split_role', 'env_seed', 'policy_seed_root'):
            assert row[field] == other[field], (k, field)
        n = min(len(row['policy_noise_seeds']), len(other['policy_noise_seeds']))
        assert row['policy_noise_seeds'][:n] == other['policy_noise_seeds'][:n]
        if 'horizon_writer_lora' in row:
            for field in ('teacher_demo_indices', 'teacher_videos'):
                assert row['horizon_writer_lora'][field] == other['horizon_writer_lora'][field]
        else:
            for field in ('source_task', 'teacher_demo', 'camera_view'):
                assert row['diagnostic_control'][field] == other['diagnostic_control'][field]
    x, y = ({key(r) for r in rows if r['success']} for rows in (left, right))
    return {'retained': len(x & y), 'gained': len(y - x), 'lost': len(x - y)}


if __name__ == '__main__':
    index = read(ROOT / 'index.json')
    panels = read(ROOT / 'panel_summary.json')['panels']
    entries = {p['panel_id']: p for p in panels}
    rows = {}
    for panel in panels:
        values = load_rows(panel)
        assert len(values) == panel['rows']
        assert sum(v['success'] for v in values) == panel['successes']
        counts = collections.Counter((r['suite'], r['task_id']) for r in values)
        wins = collections.Counter((r['suite'], r['task_id']) for r in values if r['success'])
        assert len(counts) == len(panel['per_task'])
        assert sum(wins[k] > 0 for k in counts) == panel['breadth']
        for task in panel['per_task']:
            k = task['suite'], task['task_id']
            assert (counts[k], wins[k]) == (task['rows'], task['successes'])
        rows[panel['panel_id']] = values
    assert len(panels) == index['outcome_panels']
    assert sum(map(len, rows.values())) == index['outcome_rows']
    comparisons = {}
    for seed in (7, 11):
        for step in (200, 400):
            suffix = f'validation_correct_step{step}_J0'
            comparisons[f'init{seed}_step{step}'] = paired(
                rows[f'learning/all{seed}/{suffix}'], rows[f'learning/off{seed}/{suffix}'])
    for arm in ('correct', 'same_task_other', 'wrong_same_suite', 'wrong_cross_suite',
                'shuffled', 'reversed', 'static_first', 'static_middle', 'static_last'):
        comparisons[f'video_{arm}'] = paired(
            rows[f'diagnostics/video_controls/step400/{arm}'],
            rows[f'diagnostics/best_model_video_controls/step400/{arm}'])
    for entry in index['entries']:
        assert all((ROOT / name).is_file() for name in entry['exports'])
    print(json.dumps({'verified_panels': len(panels), 'verified_rows': index['outcome_rows'],
                      'paired_comparisons': comparisons}, indent=2))
