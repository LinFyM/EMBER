"""Recount published evidence and key paired contrasts; no ML dependencies."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def load_rows(root, names):
    return [json.loads(line) for name in names for line in (root / name).read_text().splitlines()]


def paired(before, after, *, same_video=False):
    def keyed(rows):
        result = {(r['suite'], r['task_id'], r['init_state_id']): r for r in rows}
        assert len(result) == len(rows)
        return result

    left, right = keyed(before), keyed(after)
    assert left.keys() == right.keys()
    for key, row in left.items():
        other = right[key]
        for field in ('language', 'env_seed', 'policy_seed_root'):
            assert row[field] == other[field], (key, field)
        length = min(len(row['policy_noise_seeds']), len(other['policy_noise_seeds']))
        assert length > 0
        assert row['policy_noise_seeds'][:length] == other['policy_noise_seeds'][:length]
        if same_video:
            for field in ('teacher_demo_indices', 'teacher_videos'):
                assert row['horizon_writer_lora'][field] == other['horizon_writer_lora'][field]
    a = {k for k, r in left.items() if r['success']}
    b = {k for k, r in right.items() if r['success']}
    return {'reference_successes': len(a), 'candidate_successes': len(b),
            'retained': len(a & b), 'gained': len(b - a), 'lost': len(a - b)}


def matches(actual, expected):
    for field, value in actual.items():
        assert value == expected[field], (field, value, expected[field])


def main():
    index = read(ROOT / 'index.json')
    summaries = read(ROOT / 'panel_summary.json')['panels']
    rows = {}
    for panel in summaries:
        values = load_rows(ROOT, panel['rows_files'])
        assert len(values) == panel['rows']
        assert all(type(r['success']) is bool for r in values)
        assert len({(r['suite'], r['task_id'], r['init_state_id']) for r in values}) == len(values)
        assert sum(r['success'] for r in values) == panel['successes']
        counts = collections.Counter((r['suite'], r['task_id']) for r in values)
        wins = collections.Counter((r['suite'], r['task_id']) for r in values if r['success'])
        assert len(counts) == len(panel['per_task'])
        assert sum(wins[k] > 0 for k in counts) == panel['breadth']
        for task in panel['per_task']:
            key = task['suite'], task['task_id']
            assert (counts[key], wins[key]) == (task['rows'], task['successes'])
        for suite, expected in panel['per_suite'].items():
            assert sum(v for (s, _), v in counts.items() if s == suite) == expected['rows']
            assert sum(v for (s, _), v in wins.items() if s == suite) == expected['successes']
        if panel['role'] == 'validation8_50videos':
            assert len(counts) == 8 and set(counts.values()) == {50}
            for key in counts:
                videos = [r['horizon_writer_lora']['teacher_demo_indices'][0] for r in values
                          if (r['suite'], r['task_id']) == key]
                assert sorted(videos) == list(range(50))
        rows[panel['panel_id']] = values
    assert len(summaries) == index['outcome_panels'] == 45
    assert sum(map(len, rows.values())) == index['outcome_rows'] == 7216
    assert sum(p['rows'] for p in summaries if p['reused_reference_rows']) == 192
    for entry in index['entries']:
        assert all((ROOT / name).is_file() for name in entry['exports'])

    # Reuse previously published references without making duplicate copies.
    old = ROOT.parent / '20260911'
    old_panels = read(old / 'panel_summary.json')['panels']
    old_rows = {p['panel_id']: load_rows(old, p['rows_files']) for p in old_panels
                if p['panel_id'].startswith('diagnostics/best_model_video_controls/step400/')
                or p['panel_id'] == 'learning/off7/validation_correct_step400_J0'}
    source_root = ROOT.parent / '20260907'
    source_entry = next(e for e in read(source_root / 'index.json')['files'] if e['source'] ==
                        'runs/analysis/pi05_ecp_prw_meta73_equal_exposure_20260906/source_strict400/results.json')
    source_val = load_rows(source_root, [p for p in source_entry['exports'] if p.endswith('.jsonl')])
    assert len(source_val) == 400 and sum(r['success'] for r in source_val) == 47

    comparisons = 0
    mechanism = {'source': rows['references/source_train96']}
    for name, value in rows.items():
        if name.startswith('mechanism/') and not name.startswith('mechanism/nochange'):
            _, model, arm = name.split('/')
            mechanism[f'{model}_{arm}'] = value
    for name, value in old_rows.items():
        if name.startswith('diagnostics/'):
            mechanism['off400_' + name.rsplit('/', 1)[1]] = value
    recorded = read(ROOT / 'analysis/mechanism/behavior_summary.json')
    assert {name: sum(r['success'] for r in value) for name, value in mechanism.items()} == recorded['counts']
    for contrast in recorded['contrasts']:
        actual = paired(mechanism[contrast['reference']], mechanism[contrast['candidate']])
        assert [actual['reference_successes'], actual['candidate_successes']] == contrast['counts']
        for field in ('retained', 'gained', 'lost'):
            assert actual[field] == contrast[field]
        comparisons += 1

    for step in (100, 200):
        recorded = read(ROOT / f'analysis/nochange/step{step}_summary.json')
        current = rows[f'mechanism/nochange{step}/correct']
        for arm, expected in recorded['train'].items():
            if arm == 'source':
                reference = rows['references/source_train96']
            elif arm == 'old_C_same_step':
                reference = rows[f'mechanism/c{step}/correct']
            elif arm == 'previous_node':
                reference = rows[f'mechanism/nochange{step - 100}/correct']
            else:
                reference = rows[f'mechanism/nochange{step}/{arm}']
            matches(paired(reference, current), expected)
            comparisons += 1
        if step == 200:
            current = rows['learning/nochange/validation_correct_step200_J0']
            matches(paired(source_val, current), recorded['validation']['source'])
            matches(paired(rows['learning/c/validation_correct_step200_J0'], current, same_video=True),
                    recorded['validation']['old_C_same_step'])
            comparisons += 2
        else:
            assert recorded['status'] == 'pending' and 'validation' not in recorded

    paths = read(ROOT / 'analysis/paths/behavior_summary.json')['results']
    for arm, expected in paths.items():
        actual = paired(rows['paths/c200/full_correct'], rows[f'paths/c200/{arm}'])
        assert actual['candidate_successes'] == expected['successes']
        assert actual['retained'] == expected['retained']
        assert actual['gained'] == expected['gained_by_substitution']
        assert actual['lost'] == expected['lost_by_substitution']
        comparisons += 1
    old_off = old_rows['learning/off7/validation_correct_step400_J0']
    assert paired(old_off, rows['learning/r/validation_correct_step400_J0'], same_video=True) == {
        'reference_successes': 126, 'candidate_successes': 85, 'retained': 69, 'gained': 16, 'lost': 57}
    assert paired(old_off, rows['learning/off_continuation/validation_correct_step600_J0'], same_video=True) == {
        'reference_successes': 126, 'candidate_successes': 54, 'retained': 43, 'gained': 11, 'lost': 83}
    comparisons += 2
    pause = read(ROOT / 'analysis/nochange/owner_pause.json')
    assert pause['checkpoints_preserved'] and not pause['resume_launched']
    assert not pause['validation100_qualification']
    print(json.dumps({'verified_panels': len(summaries), 'verified_rows': index['outcome_rows'],
                      'reused_reference_rows': 192, 'verified_paired_comparisons': comparisons,
                      'scope': 'Recount, teacher coverage, execution pairing and success sets; no ML execution; bootstrap intervals retained from source analyses'}, indent=2))


if __name__ == '__main__':
    main()
