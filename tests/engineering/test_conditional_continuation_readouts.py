"""§14 actual capture/source entrypoints; endpoint fixtures are CPU scope checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.operator_writer import bank, joint_readout as readout, joint_training as study
from ember.operator_writer import run, specification as specs
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_source_checkpoint import read_json, write_json_atomic


@pytest.mark.parametrize('mode,macro', [
    (study.CONDITIONAL_MODE, 900), (readout.CONDITIONAL_SEEN_MODE, 900),
    (study.CONDITIONAL_MODE, 810)])
def test_actual_continuation_capture_entry_and_bounded_scope(tmp_path, monkeypatch, mode, macro):
    monkeypatch.setattr(study, 'CONDITIONAL_CONTINUATION_ROOT', tmp_path)
    checkpoint = tmp_path / study.CONDITIONAL_MODE / f'train/attempts/continuation/checkpoints/macro_{macro:08d}'
    seen = mode == readout.CONDITIONAL_SEEN_MODE
    registered_path = readout.capture_path(mode, checkpoint)
    registered = read_json(registered_path)
    states = readout.scope.STATES if seen else tuple(range(50))
    tasks = [SimpleNamespace(suite=row['suite'], task_id=row['task_id'], init_state_ids=states)
             for row in registered['full_conditions']]
    path = readout.bank_path(mode, checkpoint)
    path.parent.mkdir(parents=True)
    write_json_atomic(path, {'kind': bank.KIND, 'mode': mode, 'joint_public_study': True,
                            'checkpoint': str(checkpoint)})
    args = SimpleNamespace(role=readout.scope.ROLE if seen else 'validation', mode='formal',
                           static_task_lora_manifest=path, trajectory_capture_selection=registered_path)
    output = readout.evaluation_path(mode, checkpoint)
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert path == tmp_path / mode / f'banks/{macro}/manifest.json'
    assert output == tmp_path / mode / f'evaluation/{macro}' / ('correct144' if seen else 'correct400')
    assert registered['study_id'] == study.CONDITIONAL_CONTINUATION_TASK
    assert len(capture['full_conditions']) == (36 if seen else 8)
    assert capture['passive_trace'] and not stage['full_conditions_only']
    fresh = read_json(readout.capture_path(mode))
    assert {**registered, 'study_id': fresh['study_id']} == fresh
    with pytest.raises(Pi05EvaluationError, match='scope changed'):
        _registered_trajectory_capture(args, tasks, output.with_name('public144'), None, readout.REPO)
    with pytest.raises(ValueError, match='complete900'):
        readout.bank_path(readout.CONDITIONAL_SEEN_MODE, checkpoint.with_name('macro_00000810'))
    with pytest.raises(ValueError, match='complete900'):
        readout.bank_path(mode, checkpoint.with_name('macro_00000720'))
    with pytest.raises(ValueError, match='full900'):
        readout.a28_source_record(study.CONDITIONAL_MODE, checkpoint.with_name('macro_00000810'))


def test_real_parent_reader_identity_and_unchanged_teacher_scene_geometry(monkeypatch):
    monkeypatch.setattr(run, 'frozen_git', lambda **kw: {'commit': 'cpu-reader-only', 'branch': '',
                        'dirty_paths': [], 'pushed_ref': 'origin/main'})
    fresh, training, path = readout.source_record(study.CONDITIONAL_MODE, study.CONDITIONAL_PARENT_CHECKPOINT)
    assert training['git']['commit'] == study.CONDITIONAL_PARENT_GIT
    assert training['spec'] == str(study.CONDITIONAL_PARENT_SPEC_PATH)
    assert path == specs.CONDITIONAL_SPEC_PATH
    continuation = specs.specification(specs.CONDITIONAL_CONTINUATION_SPEC_PATH)
    asset = Path('/data1/user/ymdai/projects/EMBER')
    for mode in readout.CONDITIONAL_MODES:
        assert readout._geometry(mode, continuation, asset) == readout._geometry(mode, fresh, asset)


def test_900_source_uses_registered_continuation_spec_and_retains_training_identity(monkeypatch):
    checkpoint = study.CONDITIONAL_CONTINUATION_ROOT / study.CONDITIONAL_MODE / (
        'train/attempts/cpu_reader_fixture/checkpoints/macro_00000900')
    actual_spec = specs.specification(specs.CONDITIONAL_CONTINUATION_SPEC_PATH)
    training = {'git': {'commit': 'actual-training-fixture'},
                'spec': '/data1/training-fixture/configs/operator_read_write_v1/conditional_read_write_continuation900_spec.json'}
    calls = []
    def inspect(spec, endpoint):
        calls.append((spec, endpoint))
        return training
    monkeypatch.setattr(study, 'inspect_source', inspect)
    spec, original, reading_spec = readout.a28_source_record(study.CONDITIONAL_MODE, checkpoint)
    assert spec == actual_spec and calls == [(actual_spec, checkpoint)]
    assert original is training and reading_spec == specs.CONDITIONAL_CONTINUATION_SPEC_PATH
    assert type(original['spec']) is str and original['spec'] != str(reading_spec)


def _complete900_fixture(root):
    # Tiny files meet the actual header/size consumer. They are not trained weights.
    checkpoint = root / study.CONDITIONAL_MODE / 'train/attempts/finish900/checkpoints/macro_00000900'
    checkpoint.mkdir(parents=True)
    names = ('ecp.safetensors', 'trainer_state.pt', 'rank_00_state.pt')
    for name in names:
        (checkpoint / name).write_bytes(b'CPU endpoint scope fixture')
    write_json_atomic(checkpoint / 'checkpoint_manifest.json', {
        'stage': run.STAGE, 'run_contract_schema': run.SCHEMA, 'next_macro': 900, 'world_size': 1,
        'files': {name: {'bytes': (checkpoint / name).stat().st_size} for name in names}})
    capture = read_json(readout.capture_path(study.CONDITIONAL_MODE, checkpoint))
    rows = [{'suite': task['suite'], 'task_id': task['task_id'], 'init_state_id': state,
             'success': index * 50 + state < 154}
            for index, task in enumerate(capture['full_conditions']) for state in range(50)]
    results_path = readout.evaluation_path(study.CONDITIONAL_MODE, checkpoint) / 'results.json'
    result = {'mode': 'formal', 'role': 'validation', 'arm': 'correct', 'rows': rows,
              'adapter': {'checkpoint': str(checkpoint), 'mode': study.CONDITIONAL_MODE},
              'overall': {'episodes': 400, 'successes': 154},
              'launcher': {'return_codes': {'worker': 0}, 'queue': {'completed_rows': 400}}}
    write_json_atomic(results_path, result)
    trigger = {'status': 'validated', 'successes': 154, 'checkpoint': str(checkpoint),
               'results': str(results_path)}
    return checkpoint, results_path, result, trigger


def test_810_requires_actual_valid_complete900_and_all400_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'CONDITIONAL_CONTINUATION_ROOT', tmp_path)
    earlier = tmp_path / study.CONDITIONAL_MODE / 'train/attempts/at810/checkpoints/macro_00000810'
    with pytest.raises(ValueError, match='validated complete900'):
        readout.source_record(study.CONDITIONAL_MODE, earlier)
    main, result_path, result, trigger = _complete900_fixture(tmp_path)
    path = tmp_path / 'launch/900_branch_trigger.json'
    write_json_atomic(path, trigger)
    readout._conditional_adjacent_trigger(earlier)  # Main900 is in a different resume attempt.
    for changed in ({**trigger, 'successes': 153}, {**trigger, 'status': 'pending'},
                    {**trigger, 'results': str(result_path.with_name('screen80.json'))}):
        write_json_atomic(path, changed)
        with pytest.raises(ValueError, match='validated complete900'):
            readout._conditional_adjacent_trigger(earlier)
    write_json_atomic(path, trigger)
    for defect in ('partial', 'duplicate', 'score', 'failed', 'checkpoint'):
        bad = deepcopy(result)
        if defect == 'partial': bad['rows'].pop()
        elif defect == 'duplicate': bad['rows'][-1] = bad['rows'][0]
        elif defect == 'score': bad['overall']['successes'] = 155
        elif defect == 'failed': bad['launcher']['return_codes']['worker'] = 1
        else: bad['adapter']['checkpoint'] = str(earlier)
        write_json_atomic(result_path, bad)
        with pytest.raises(ValueError, match='validated complete900'):
            readout._conditional_adjacent_trigger(earlier)
    write_json_atomic(result_path, result)
    (main / 'rank_00_state.pt').unlink()
    with pytest.raises(ValueError, match='validated complete900'):
        readout._conditional_adjacent_trigger(earlier)
