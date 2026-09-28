"""CPU evidence for the sole active 900→1350 operator continuation contract."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.operator_writer import bank as operator_bank
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.run import (CONTINUATION_SPEC_PATH, CONTINUATION1350_SPEC_PATH,
                                       complete_checkpoint, specification, train, validate_attempt)
from ember.pi05_source_checkpoint import read_json, write_json_atomic

ASSET = Path('/data1/user/ymdai/projects/EMBER')
OLD = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900')
GIT = {'commit': 'new-pushed-frozen', 'branch': '', 'dirty_paths': [],
       'pushed_ref': 'origin/codex/demonstration-transfer'}


def _contract(mode, spec, sampler):
    parent = OLD / mode / 'train/attempts/continuation'
    original = read_json(parent / 'run_contract.json')
    checkpoint = parent / 'checkpoints/macro_00000900'
    return {**original, 'git': GIT, 'spec': str(CONTINUATION1350_SPEC_PATH),
            'events': spec['events'], 'sampler': sampler,
            'continuation': spec['continuation'], 'parent_checkpoint': str(checkpoint),
            'stop_after_macro': None}


def _fake_checkpoint(path, macro, mode, sampler):
    path.mkdir(parents=True)
    files = {name: {'bytes': 1} for name in ('ecp.safetensors', 'rank_00_state.pt',
                                           'rank_01_state.pt')}
    for name in files:
        (path / name).write_bytes(b'x')
    torch.save({'schema_version': 'ember_ecp_checkpoint_v1',
                'stage': 'operator_read_write_learning', 'next_macro': macro,
                'metrics_rows': macro, 'optimizer': {'param_groups': [{'lr': 1e-5}]},
                'scheduler': {'last_epoch': macro}, 'scaler': None,
                'training_state': {'updates': macro, 'mode': mode},
                'sampler_state': sampler | {'next_step': macro}}, path / 'trainer_state.pt')
    files['trainer_state.pt'] = {'bytes': (path / 'trainer_state.pt').stat().st_size}
    write_json_atomic(path / 'checkpoint_manifest.json', {
        'stage': 'operator_read_write_learning',
        'run_contract_schema': 'ember_operator_read_write_formal_run_v1',
        'next_macro': macro, 'world_size': 2, 'files': files})
    assert complete_checkpoint(path)


def test_real_900_migration_preserves_prefix_and_third_teacher_round(tmp_path):
    old_spec, new_spec = specification(CONTINUATION_SPEC_PATH), specification(CONTINUATION1350_SPEC_PATH)
    assert new_spec['execution']['checkpoints'] == [990, 1080, 1170, 1260, 1350]
    assert new_spec['budget']['peak_new_gib'] == 52
    old = FormalData(ASSET, old_spec, query_labels=False)
    new = FormalData(ASSET, new_spec, query_labels=False)
    try:
        for step in (0, 269, 449, 899):
            assert new.tasks_for_step(step) == old.tasks_for_step(step)
            for task in old.tasks_for_step(step):
                assert new.event(step, task) == old.event(step, task)
        for task in TASKS:
            observed = []
            for visit in range(100, 150):
                step = next(step for step in range(visit * 9, visit * 9 + 9)
                            if task in new.tasks_for_step(step))
                event = new.event(step, task)
                assert event['visit'] == visit and event['query_offset'] == 1
                assert len(event['queries']) == 28
                assert all(row['demo'] != event['teacher_demo'] for row in event['queries'])
                observed.append(event['teacher_demo'])
            expected = np.random.default_rng(np.random.SeedSequence(
                [20260928, 1, task, 2])).permutation(50).tolist()
            assert observed == expected and len(set(observed)) == 50
        for mode in ('T', 'U'):
            checkpoint = OLD / mode / 'train/attempts/continuation/checkpoints/macro_00000900'
            trainer = torch.load(checkpoint / 'trainer_state.pt', map_location='meta', mmap=True,
                                 weights_only=True)
            assert trainer['next_macro'] == trainer['metrics_rows'] == 900
            assert trainer['scheduler']['last_epoch'] == 900
            assert trainer['optimizer']['param_groups']
            migration = new.restore(trainer['sampler_state'], migrate_continuation_900=True)
            assert migration['from_schema'].endswith('_v3') and migration['to_schema'].endswith('_v4')
            assert migration['cursor'] == new.next_step == 900
            with pytest.raises(ValueError, match='migration source'):
                new.restore(trainer['sampler_state'] | {'next_step': 899},
                            migrate_continuation_900=True)
            prefix = [json.loads(line) for line in
                      (checkpoint.parent.parent / 'metrics.jsonl').read_text().splitlines()]
            assert len(prefix) == 900 and [r['update'] for r in prefix] == list(range(1, 901))
            spec = {**new_spec, 'run_root': str(tmp_path)}
            contract = _contract(mode, spec, {k: v for k, v in new.sampler_state().items()
                                              if k != 'next_step'})
            args = SimpleNamespace(mode=mode, resume=checkpoint)
            validate_attempt(spec, args, contract, tmp_path / mode / 'train/attempts/first')
            with pytest.raises(ValueError):
                validate_attempt(spec, args, contract | {'source': {'checkpoint': '/wrong'}},
                                 tmp_path / mode / 'train/attempts/rejected')
        with pytest.raises(ValueError, match='continuation1800 spec'):
            train(old_spec, SimpleNamespace())
        invalid = SimpleNamespace(mode='T', attempt='first', resume=OLD / 'T/train/attempts/continuation/checkpoints/macro_00000900',
                                  microbatch=28, frame_chunk=8, stop_after_macro=910)
        with pytest.raises(ValueError, match='continuation1800 spec'):
            train(new_spec, invalid)
    finally:
        old.close(); new.close()


def test_latest_new_ecp_and_selected_bank_source_reject_old_or_wrong_identity(tmp_path, monkeypatch):
    spec = {**specification(CONTINUATION1350_SPEC_PATH), 'run_root': str(tmp_path)}
    data = FormalData(ASSET, spec, query_labels=False)
    try:
        sampler = {k: v for k, v in data.sampler_state().items() if k != 'next_step'}
        parent = OLD / 'T/train/attempts/continuation/checkpoints/macro_00000900'
        contract = _contract('T', spec, sampler)
        first = tmp_path / 'T/train/attempts/first'
        checkpoint = first / 'checkpoints/macro_00001080'
        _fake_checkpoint(checkpoint, 1080, 'T', sampler)
        write_json_atomic(first / 'run_contract.json', contract)
        write_json_atomic(first / 'resume_provenance.json', {'checkpoint': str(parent)})
        old_lines = (parent.parent.parent / 'metrics.jsonl').read_text().splitlines()
        assert len(old_lines) == 900
        (first / 'metrics.jsonl').write_text('\n'.join(old_lines + [
            json.dumps({'update': step}) for step in range(901, 1081)]) + '\n')
        with pytest.raises(ValueError, match='latest complete'):
            validate_attempt(spec, SimpleNamespace(mode='T', resume=parent), contract,
                             tmp_path / 'T/train/attempts/replay900')
        current = contract | {'parent_checkpoint': str(checkpoint)}
        validate_attempt(spec, SimpleNamespace(mode='T', resume=checkpoint), current,
                         tmp_path / 'T/train/attempts/second')
        with pytest.raises(ValueError):
            validate_attempt(spec, SimpleNamespace(mode='U', resume=checkpoint), current,
                             tmp_path / 'U/train/attempts/wrong_arm')
        monkeypatch.setattr(operator_bank, 'frozen_git', lambda **_: GIT)
        assert operator_bank.inspect_training_source(spec, checkpoint, 'T') == contract
        with pytest.raises(ValueError, match='same-arm'):
            operator_bank.inspect_training_source(spec, checkpoint, 'U')
        bad = deepcopy(contract); bad['git'] = {'commit': 'wrong'}
        write_json_atomic(first / 'run_contract.json', bad)
        with pytest.raises(ValueError, match='source/ECP'):
            operator_bank.inspect_training_source(spec, checkpoint, 'T')
        write_json_atomic(first / 'run_contract.json', contract)
        bad = deepcopy(contract); bad['source'] = {'checkpoint': '/wrong'}
        write_json_atomic(first / 'run_contract.json', bad)
        with pytest.raises(ValueError, match='source/ECP'):
            operator_bank.inspect_training_source(spec, checkpoint, 'T')
    finally:
        data.close()
