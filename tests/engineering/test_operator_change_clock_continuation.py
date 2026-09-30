"""Real sealed-parent and event-stream checks for the bounded clock continuation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.operator_writer import change_clock
from ember.operator_writer import bank as operator_bank
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.run import (CHANGE_CLOCK_SPEC_PATH, CHANGE_CLOCK_CONTINUATION_SPEC_PATH,
                                       CONTINUATION_SPEC_PATH, audit, specification,
                                       validate_attempt, validate_train_request)
from ember.operator_writer.scope import capture_expectations
from ember.pi05_source_checkpoint import read_json
from ember.pi05_eval.preparation import _registered_trajectory_capture

ASSET = Path('/data1/user/ymdai/projects/EMBER')


def test_actual_270_parent_migration_keeps_events_and_completes_50_teachers(tmp_path):
    spec = specification(CHANGE_CLOCK_CONTINUATION_SPEC_PATH) | {'run_root': str(tmp_path)}
    old = FormalData(ASSET, specification(CHANGE_CLOCK_SPEC_PATH), query_labels=False)
    new = FormalData(ASSET, spec, query_labels=False)
    reference = FormalData(ASSET, specification(CONTINUATION_SPEC_PATH), query_labels=False)
    try:
        parent = change_clock.PARENT_CHECKPOINT
        trainer = torch.load(parent / 'trainer_state.pt', map_location='meta', mmap=True, weights_only=True)
        assert trainer['next_macro'] == trainer['metrics_rows'] == trainer['scheduler']['last_epoch'] == 270
        assert trainer['optimizer']['param_groups'] and trainer['scaler'] is None
        migration = new.restore(trainer['sampler_state'], migrate_sealed_270=True)
        assert migration['historical_teacher_pool_meaning'] == 'visits_0_to_29_not_demo_ids'
        assert new.next_step == 270
        for step in range(450):
            assert new.tasks_for_step(step) == reference.tasks_for_step(step)
            for task in new.tasks_for_step(step):
                event = new.event(step, task)
                assert event == reference.event(step, task)
                if step < 270:
                    assert event == old.event(step, task)
        summary = audit(spec, ASSET)
        assert summary['queries_per_mode'] == 50400
        assert len(summary['teacher_order']) == len(TASKS) == 36
        assert all(len(rows) == len(set(rows)) == 50 for rows in summary['teacher_order'].values())
        new.restore(new.sampler_state() | {'next_step': 360})
        with pytest.raises(ValueError):
            new.restore(trainer['sampler_state'] | {'next_step': 269}, migrate_sealed_270=True)
        original = read_json(parent.parent.parent / 'run_contract.json')
        for world in (1, 2, 3, 4):
            contract = {**original, 'git': {'commit': 'new-frozen'},
                        'spec': str(CHANGE_CLOCK_CONTINUATION_SPEC_PATH),
                        'events': spec['events'], 'continuation': spec['continuation'],
                        'sampler': {k: v for k, v in new.sampler_state().items() if k != 'next_step'},
                        'topology': {'world_size': world}}
            args = SimpleNamespace(mode=change_clock.MODE, resume=parent, attempt='resume270',
                                   pilot_arm=None, microbatch=28, frame_chunk=8, stop_after_macro=None)
            validate_train_request(spec, args)
            validate_attempt(spec, args, contract, tmp_path / f'world{world}')
            changed = deepcopy(contract)
            changed['operator']['erase_rule'] = 'constant'
            with pytest.raises(ValueError):
                validate_attempt(spec, args, changed, tmp_path / f'bad{world}')
    finally:
        for data in (old, new, reference):
            data.close()


def test_450_capture_uses_new_study_and_same_full_compact_geometry():
    spec = specification(CHANGE_CLOCK_CONTINUATION_SPEC_PATH)
    tasks = [SimpleNamespace(suite=('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')[i // 10],
                             task_id=i % 10) for i in spec['evaluation']['task_ids']]
    path = change_clock.CONTINUATION_ROOT / change_clock.MODE / 'banks/450/manifest.json'
    capture = capture_expectations({'mode': change_clock.MODE}, path, tasks)
    assert capture['study'] == change_clock.CONTINUATION_TASK
    assert capture['output'] == change_clock.CONTINUATION_ROOT / change_clock.MODE / 'evaluation/450/correct400'
    assert read_json(capture['capture'])['full_conditions'] == capture['full']
    assert len(capture['full']) == 8 and capture['states'] == tuple(range(50))


def test_official_reader_and_materializer_share_450_source_identity(monkeypatch):
    spec = specification(CHANGE_CLOCK_CONTINUATION_SPEC_PATH)
    git = {'commit': 'clean-pushed-frozen'}
    monkeypatch.setattr(operator_bank, 'frozen_git', lambda **_: git)
    identities = [operator_bank._continuation_source_identity(spec, 450, sealed)
                  for sealed in (False, True)]
    assert identities[0] == identities[1]
    assert identities[0][1] == change_clock.CONTINUATION_TRAINING_SPEC_PATH
    assert identities[0][-1] == change_clock.CONTINUATION_TRAINING_GIT
    assert identities[0][-1] != git
    with pytest.raises(ValueError):
        operator_bank._continuation_source_identity(spec, 360, True)
    changed = deepcopy(spec)
    changed['operator']['erase_rule'] = 'constant'
    with pytest.raises(ValueError, match='actual training specs differ'):
        operator_bank._continuation_source_identity(changed, 450, True)


def test_real_evaluator_capture_entry_routes_450_without_train_subset(tmp_path):
    spec = specification(CHANGE_CLOCK_CONTINUATION_SPEC_PATH)
    tasks = [SimpleNamespace(suite=('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')[i // 10],
                             task_id=i % 10, init_state_ids=tuple(range(50)))
             for i in spec['evaluation']['task_ids']]
    bank_path = tmp_path / 'T_change_clock/banks/450/manifest.json'
    bank_path.parent.mkdir(parents=True)
    import json
    bank_path.write_text(json.dumps({'kind': operator_bank.KIND, 'mode': change_clock.MODE}))
    capture_path = CHANGE_CLOCK_SPEC_PATH.with_name(change_clock.CONTINUATION_CAPTURE_NAME)
    output = tmp_path / 'T_change_clock/evaluation/450/correct400'
    args = SimpleNamespace(trajectory_capture_selection=capture_path,
                           static_task_lora_manifest=bank_path, role='validation', mode='formal')
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, ASSET)
    assert capture['selection_path'] == str(capture_path)
    assert len(capture['full_conditions']) == 8
    assert capture['passive_trace']['trace_root'] == str(output / 'continuous_traces')
    assert stage['validation_action_reads'] == stage['validation_reward_reads'] == 0
