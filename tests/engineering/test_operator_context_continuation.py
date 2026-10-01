"""Real parent450, original second teacher cycle and official consumer routing."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.operator_writer import joint_training as study, joint_readout as readout, run
from ember.operator_writer.data import FormalData, TASKS
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.source_sft.control import clamped_lr_multiplier

ASSET = Path('/data1/user/ymdai/projects/EMBER')


def test_real_parent_sampler_absolute_lr_and_original_second_teacher_cycle():
    spec = run.specification(run.CONTEXT_CONTINUATION_SPEC_PATH)
    parent = torch.load(study.CONTEXT_PARENT_CHECKPOINT / 'trainer_state.pt',
                        map_location='meta', mmap=True, weights_only=True)
    candidate = FormalData(ASSET, spec, query_labels=False)
    original = FormalData(ASSET, run.specification(run.CONTINUATION_SPEC_PATH), query_labels=False)
    teachers = defaultdict(list)
    try:
        assert candidate.restore(parent['sampler_state']) is None
        assert candidate.next_step == parent['next_macro'] == parent['scheduler']['last_epoch'] == 450
        lr = spec['optimization']['lr'] * clamped_lr_multiplier(
            450, warmup=150, decay=1200, peak=.0003, floor=1e-5)
        assert parent['optimizer']['param_groups'][0]['lr'] == pytest.approx(lr)
        assert lr == pytest.approx(.00024540602126951636)
        for step in range(450, 900):
            assert candidate.tasks_for_step(step) == original.tasks_for_step(step)
            for task in candidate.tasks_for_step(step):
                assert candidate.event(step, task) == original.event(step, task)
                event = candidate.event(step, task)
                teachers[task].append(event['teacher_demo'])
                assert event['visit'] >= 50 and event['update'] == step + 1
        assert set(teachers) == set(TASKS)
        assert all(len(v) == len(set(v)) == 50 for v in teachers.values())
        assert candidate.checkpoints == (540, 630, 720, 810, 900)
        with pytest.raises(ValueError, match='cursor changed'):
            candidate.restore({**parent['sampler_state'], 'next_step': 451})
    finally:
        candidate.close(); original.close()


def test_real_parent_source_consumer_and_resume_contract(tmp_path, monkeypatch):
    # Caller Git bookkeeping is the only fixture; actual sealed parent code,
    # run/ECP/optimizer/sampler headers and completion are all read directly.
    monkeypatch.setattr(run, 'frozen_git', lambda **kw: {'commit': 'cpu-caller-only'})
    spec = run.specification(run.CONTEXT_CONTINUATION_SPEC_PATH)
    old = study.inspect_source(read_json(study.CONTEXT_PARENT_SPEC_PATH), study.CONTEXT_PARENT_CHECKPOINT)
    assert old['git']['commit'] == study.CONTEXT_PARENT_GIT
    contract = {**old, 'continuation': spec['continuation'], 'spec': str(run.CONTEXT_CONTINUATION_SPEC_PATH),
                'parent_checkpoint': str(study.CONTEXT_PARENT_CHECKPOINT)}
    args = SimpleNamespace(mode='context', attempt='cpu_contract_unused', resume=study.CONTEXT_PARENT_CHECKPOINT,
                           pilot_arm=None, microbatch=28, frame_chunk=8, stop_after_macro=None)
    run.validate_train_request(spec, args)
    # The real batch is now completed. Isolate the hypothetical new attempt;
    # source450 remains real and no historical attempt/checkpoint is modified.
    monkeypatch.setattr(study, 'CONTEXT_CONTINUATION_ROOT', tmp_path)
    output = study.CONTEXT_CONTINUATION_ROOT / 'context/train/attempts/cpu_contract_unused'
    run.validate_attempt(spec, args, contract, output)
    bad = deepcopy(contract); bad['optimizer']['lr'] *= 2
    with pytest.raises(ValueError, match='optimizer changed'):
        run.validate_attempt(spec, args, bad, output)
    args.resume = None
    with pytest.raises(ValueError, match='registered fresh identity'):
        run.validate_train_request(spec, args)


@pytest.mark.parametrize('mode,macro', [('context',900),('context_public',900),('context',810)])
def test_real_capture_prepare_route_new_root_and_checkpoint(tmp_path, monkeypatch, mode, macro):
    # A scope fixture exercises the actual prepare entry without claiming a
    # future bank or GPU consumer exists. It remains wholly in tmp_path.
    from ember.operator_writer import bank
    monkeypatch.setattr(study, 'CONTEXT_CONTINUATION_ROOT', tmp_path)
    checkpoint = tmp_path / f'context/train/attempts/continuation/checkpoints/macro_{macro:08d}'
    path = readout.bank_path(mode, checkpoint)
    capture_path = readout.capture_path(mode, checkpoint)
    registered = read_json(capture_path)
    public = mode in readout.PUBLIC_MODES
    states = readout.scope.STATES if public else tuple(range(50))
    tasks = [SimpleNamespace(suite=row['suite'],task_id=row['task_id'],init_state_ids=states)
             for row in registered['full_conditions']]
    path.parent.mkdir(parents=True)
    write_json_atomic(path, {'kind':bank.KIND, 'mode':mode,'joint_public_study':True,
                            'checkpoint':str(checkpoint)})
    args = SimpleNamespace(role=readout.scope.ROLE if public else 'validation',mode='formal',
                           static_task_lora_manifest=path,trajectory_capture_selection=capture_path)
    output = readout.evaluation_path(mode, checkpoint)
    capture, stage = _registered_trajectory_capture(args,tasks,output,None,readout.REPO)
    assert registered['study_id'] == study.CONTEXT_CONTINUATION_TASK
    assert len(capture['full_conditions']) == (36 if public else 8)
    assert capture['passive_trace'] and not stage['full_conditions_only']
    if macro == 810:
        with pytest.raises(ValueError,match='validated complete900'):
            readout.source_record(mode,checkpoint)
        with pytest.raises(ValueError,match='registered main/conditional endpoint'):
            readout.bank_path('context_public',checkpoint)


def test_saved810_source_follows_completed900_resume_lineage(tmp_path):
    attempts = tmp_path / 'context/train/attempts'
    earlier, terminal = attempts/'at810', attempts/'finish900'
    earlier.mkdir(parents=True);terminal.mkdir()
    checkpoint = earlier/'checkpoints/macro_00000810'
    write_json_atomic(earlier/'run_contract.json', {'parent_checkpoint':str(study.CONTEXT_PARENT_CHECKPOINT)})
    write_json_atomic(terminal/'run_contract.json', {'parent_checkpoint':str(checkpoint)})
    write_json_atomic(terminal/'completion.json', {'updates':900})
    assert study._completed_metrics_source(tmp_path,'context',checkpoint,900) == terminal
    write_json_atomic(terminal/'run_contract.json', {'parent_checkpoint':str(checkpoint.with_name('macro_00000720'))})
    with pytest.raises(ValueError,match='actual resume boundary'):
        study._completed_metrics_source(tmp_path,'context',checkpoint,900)


def test_seen144_actual900_source_teacher_scope_and_capture_consumer(tmp_path, monkeypatch):
    from ember.operator_writer import bank
    from ember.pi05_eval.scene import inspect_registered_scenes

    # Only current dirty CPU caller identity is substituted. Parent source,
    # complete ECP, original spec and the registered task/video/scene rows are real.
    monkeypatch.setattr(run, 'frozen_git', lambda **kw: {'commit': 'cpu-caller-only'})
    spec, training, spec_path = readout.source_record(readout.SEEN_MODE, readout.SEEN_CHECKPOINT)
    assert training['git']['commit'] == readout.SEEN_TRAINING_GIT
    assert spec_path == run.CONTEXT_CONTINUATION_SPEC_PATH
    rows, conditions, scenes, states = readout._geometry(readout.SEEN_MODE, spec, ASSET)
    assert tuple(row['global_task_id'] for row in rows) == TASKS
    assert len(conditions) == 144 and states == (32, 33, 34, 35)
    assert conditions == readout._geometry('context_public', spec, ASSET)[1]
    inspect_registered_scenes(scenes, rows, states=states,
                              schema='ember_operator_seen_task_scenes_v1')
    data = FormalData(ASSET, spec, query_labels=False, task_ids=TASKS, role='train')
    data.close()
    with pytest.raises(ValueError, match='fixed actual Context900'):
        readout.source_record(readout.SEEN_MODE, readout.SEEN_CHECKPOINT.with_name('macro_00000810'))

    monkeypatch.setattr(readout, 'SEEN_ROOT', tmp_path)
    path = readout.bank_path(readout.SEEN_MODE, readout.SEEN_CHECKPOINT)
    path.parent.mkdir(parents=True)
    write_json_atomic(path, {'kind': bank.KIND, 'mode': readout.SEEN_MODE,
                            'joint_public_study': True, 'checkpoint': str(readout.SEEN_CHECKPOINT)})
    tasks = [SimpleNamespace(suite=row['suite'], task_id=row['task_id'], init_state_ids=states)
             for row in rows]
    args = SimpleNamespace(role=readout.scope.ROLE, mode='formal', static_task_lora_manifest=path,
                           trajectory_capture_selection=readout.capture_path(readout.SEEN_MODE))
    output = readout.evaluation_path(readout.SEEN_MODE, readout.SEEN_CHECKPOINT)
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert read_json(readout.capture_path(readout.SEEN_MODE))['study_id'] == readout.SEEN_TASK
    assert len(capture['full_conditions']) == 36
    assert all(row['init_state_id'] == 32 for row in capture['full_conditions'])
    assert capture['passive_trace'] and not stage['full_conditions_only']
    args.role = 'validation'
    with pytest.raises(RuntimeError, match='capture scope changed'):
        _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
