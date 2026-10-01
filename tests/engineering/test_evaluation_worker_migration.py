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


def test_retired_interventions_are_readable_but_cannot_restart_workers(tmp_path):
    import json
    import pytest
    from ember.pi05_assets import Pi05EvaluationError
    from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, load_run_contract
    from ember.pi05_eval.recovery import validate_resume_inputs
    from ember.pi05_eval.worker_setup import validate_worker_assets

    contract = {"schema_version": RUN_CONTRACT_SCHEMA,
        "contract_reference": f"{RUN_CONTRACT_SCHEMA}:archived", "output_dir": str(tmp_path),
        "content_hash_policy": "disabled_by_owner", "adapter": None,
        "frozen_prefix_intervention": {"anchor": "recorded"}}
    path = tmp_path / "run_contract.json"
    path.write_text(json.dumps(contract))
    assert load_run_contract(path)["frozen_prefix_intervention"] == {"anchor": "recorded"}
    for consumer in (validate_resume_inputs, validate_worker_assets):
        with pytest.raises(Pi05EvaluationError, match="recorded frozen runtime"):
            consumer(contract)
