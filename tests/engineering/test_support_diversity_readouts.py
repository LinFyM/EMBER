"""§15 fixed630 arm routing; synthetic endpoints are consumer scope fixtures."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.operator_writer import bank, joint_readout as readout, run, support_diversity as study
from ember.operator_writer import specification as specs
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_source_checkpoint import read_json, write_json_atomic


@pytest.mark.parametrize('arm', ['C12', 'D71'])
@pytest.mark.parametrize('mode', readout.CONDITIONAL_MODES)
def test_actual_capture_entry_requires_correct_arm630_and_original_scope(tmp_path, monkeypatch, arm, mode):
    monkeypatch.setattr(study, 'ROOT', tmp_path / 'R3')
    control = tmp_path / 'R2' / readout.CONDITIONAL_MODE / 'train/attempts/continuation/checkpoints/macro_00000630'
    monkeypatch.setattr(study, 'C12_CHECKPOINT', control)
    checkpoint = control if arm == 'C12' else (
        study.ROOT / readout.CONDITIONAL_MODE / 'train/attempts/diversity/checkpoints/macro_00000630')
    path = readout.bank_path(mode, checkpoint, arm=arm)
    capture_path = readout.capture_path(mode, checkpoint, arm=arm)
    registered = read_json(capture_path)
    states = readout.scope.STATES if mode == readout.CONDITIONAL_SEEN_MODE else tuple(range(50))
    tasks = [SimpleNamespace(suite=r['suite'], task_id=r['task_id'], init_state_ids=states)
             for r in registered['full_conditions']]
    manifest = {'kind': bank.KIND, 'mode': mode, 'joint_public_study': True,
                'checkpoint': str(checkpoint), 'support_diversity_arm': arm}
    write_json_atomic(path, manifest)
    args = SimpleNamespace(role=readout.scope.ROLE if len(states) == 4 else 'validation', mode='formal',
                           static_task_lora_manifest=path, trajectory_capture_selection=capture_path)
    output = readout.evaluation_path(mode, checkpoint, arm=arm)
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert path == study.ROOT / arm / mode / 'banks/630/manifest.json'
    assert output == study.ROOT / arm / mode / 'evaluation/630' / (
        'correct144' if len(states) == 4 else 'correct400')
    assert len(capture['full_conditions']) == (36 if len(states) == 4 else 8)
    assert capture['passive_trace'] and not stage['full_conditions_only']
    assert registered['study_id'] == study.TASK
    fresh = read_json(readout.capture_path(mode))
    assert {**registered, 'study_id': fresh['study_id']} == fresh
    with pytest.raises(Pi05EvaluationError, match='scope changed'):
        _registered_trajectory_capture(args, tasks, output.with_name('public144'), None, readout.REPO)
    write_json_atomic(path, {**manifest, 'support_diversity_arm': 'D71' if arm == 'C12' else 'C12'})
    with pytest.raises(Pi05EvaluationError, match='registered actual630'):
        _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    for macro in (450, 540, 720, 810, 900):
        with pytest.raises(ValueError, match='fixed630'):
            readout.bank_path(mode, checkpoint.with_name(f'macro_{macro:08d}'), arm=arm)
    with pytest.raises(ValueError):
        readout.bank_path(mode, checkpoint)
    with pytest.raises(ValueError, match='fixed630'):
        readout.bank_path('context', checkpoint, arm=arm)
    with pytest.raises(ValueError, match='full C12 or D71'):
        readout.a28_source_record(readout.CONDITIONAL_SEEN_MODE, checkpoint, arm=arm)


def test_real_C12_source_and_original400_seen144_teacher_scenes(monkeypatch):
    monkeypatch.setattr(run, 'frozen_git', lambda **kw: {'commit': 'cpu-reader-only', 'branch': '',
                        'dirty_paths': [], 'pushed_ref': 'origin/main'})
    spec, training, reading_path = readout.source_record(
        readout.CONDITIONAL_MODE, study.C12_CHECKPOINT, arm='C12')
    assert training['git']['commit'] == study.C12_GIT
    assert type(training['spec']) is str and training['spec'] != str(reading_path)
    assert reading_path == specs.SUPPORT_DIVERSITY_SPEC_PATH
    assert training['spec'].endswith('/conditional_read_write_continuation900_spec.json')
    fresh = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    asset = Path('/data1/user/ymdai/projects/EMBER')
    for mode in readout.CONDITIONAL_MODES:
        assert readout._geometry(mode, spec, asset) == readout._geometry(mode, fresh, asset)
    seen, conditions, scenes, states = readout._geometry(readout.CONDITIONAL_SEEN_MODE, spec, asset)
    assert len(seen) == 36 and len(conditions) == 144 and states == (32, 33, 34, 35)
    from ember.task_protocol import load_task_authorities
    _, manifest = load_task_authorities(asset, spec['source']['data_protocol'])
    assert sum(row['split_role'] == 'train' for row in manifest['tasks']) == 95


def test_registered_D71_source_keeps_actual_training_separate_from_reader(monkeypatch):
    checkpoint = study.ROOT / readout.CONDITIONAL_MODE / 'train/attempts/diversity/checkpoints/macro_00000630'
    spec = specs.specification(specs.SUPPORT_DIVERSITY_SPEC_PATH)
    training = {'git': {'commit': 'D71-training-fixture'}, 'spec': '/data1/D71-frozen/training-spec.json'}
    calls = []
    def inspect(requested_spec, requested_checkpoint):
        calls.append((requested_spec, requested_checkpoint))
        return training
    from ember.operator_writer import joint_training
    monkeypatch.setattr(joint_training, 'inspect_source', inspect)
    actual, original, reader = readout.a28_source_record(readout.CONDITIONAL_MODE, checkpoint, arm='D71')
    assert actual == spec and original is training and reader == specs.SUPPORT_DIVERSITY_SPEC_PATH
    assert calls == [(spec, checkpoint)]


@pytest.mark.parametrize('arm', ['C12', 'D71'])
def test_episode_arm_and_actual_source_provenance_are_retained(arm):
    row = {'init_state_id': 0, 'condition_id': 'task03-demo1', 'teacher_demo_indices': [1], 'video_ordinal': 0}
    task = {'global_task_id': 3, 'suite': 'libero_spatial', 'task_id': 3, 'episodes': [row]}
    manifest = {'mode': readout.CONDITIONAL_MODE, 'shared': {'path': 'public-A0-provenance'},
                'checkpoint': f'/data1/{arm}/macro_00000630', 'scene_manifest': {'path': 'same-scene'},
                'native_reading': readout._self_read_evidence(readout.CONDITIONAL_MODE),
                'support_diversity_arm': arm, 'training_git': 'actual-trained-Git',
                'training_spec': '/data1/actual-training-spec', 'materialization_git': {'commit': 'reader-Git'},
                'tasks': [task]}
    evidence = bank.episode_evidence(manifest, task, row)
    assert evidence['support_diversity_arm'] == arm and evidence['training_git'] == 'actual-trained-Git'
    assert evidence['training_spec'] == manifest['training_spec']
    assert evidence['materialization_git'] == manifest['materialization_git']
    assert bank.validate_episode(manifest, evidence, suite=task['suite'], task_id=3, init_state_id=0)
    evidence['support_diversity_arm'] = 'D71' if arm == 'C12' else 'C12'
    assert not bank.validate_episode(manifest, evidence, suite=task['suite'], task_id=3, init_state_id=0)
