"""Actual400 queue/scene/video/RNG identities survive a physical worker increase."""
from pathlib import Path

from ember.pi05_eval.preparation import shards_from_contract
from ember.pi05_eval.recovery import migrated_worker_contract
from ember.pi05_eval.launcher_evidence import _validate_start
from ember.pi05_source_checkpoint import read_json


def test_actual_clock450_two_to_three_workers_preserves_existing_queue_and_science():
    root = Path('/data1/user/ymdai/ember_runs/operator_change_clock_continuation450_20260930')
    old = read_json(root / 'T_change_clock/evaluation/450/correct400/run_contract.json')
    new = migrated_worker_contract(old, 3, {'commit': 'new-reader', 'branch': '', 'dirty_paths': []})
    assert shards_from_contract(old) == shards_from_contract(new)
    assert {k: v for k, v in old.items() if k not in ('parallel', 'git')} == {
        k: v for k, v in new.items() if k not in ('parallel', 'git')}
    assert new['parallel']['worker_count'] == 9
    prior = new['parallel']['prior_worker_topologies'][0]
    _validate_start({'contract_reference': old['contract_reference'], 'worker_ids': prior['worker_ids']},
                    new, {'worker_ids': [f'{gpu}-r{replica}' for gpu in (0, 1, 2) for replica in range(3)]})
    again = migrated_worker_contract(new, 2, {'commit': 'another-reader', 'branch': '', 'dirty_paths': []})
    assert shards_from_contract(again) == shards_from_contract(old)
