"""The fresh joint run uses the original first-round T event and LR trajectory."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.operator_writer import joint_training
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.run import (CONTINUATION_SPEC_PATH, JOINT_SPEC_PATH, CONTEXT_SPEC_PATH, audit,
                                       specification, validate_attempt, validate_train_request)

ASSET = Path('/data1/user/ymdai/projects/EMBER')


@pytest.mark.parametrize("spec_path", [JOINT_SPEC_PATH, CONTEXT_SPEC_PATH])
def test_original_450_events_full_teacher_coverage_and_fresh_scope(tmp_path, spec_path):
    spec = specification(spec_path)
    root, mode, joint = joint_training.settings(spec)
    reference = FormalData(ASSET, specification(CONTINUATION_SPEC_PATH), query_labels=False)
    candidate = FormalData(ASSET, spec, query_labels=False)
    try:
        for step in range(450):
            assert reference.tasks_for_step(step) == candidate.tasks_for_step(step)
            for task in candidate.tasks_for_step(step):
                assert reference.event(step, task) == candidate.event(step, task)
        summary = audit(spec, ASSET)
        assert summary['queries_per_mode'] == 50400
        assert set(summary['teacher_order']) == set(TASKS)
        assert all(len(rows) == len(set(rows)) == 50 for rows in summary['teacher_order'].values())
        args = SimpleNamespace(mode=mode, attempt='fresh', resume=None, pilot_arm=None,
                               microbatch=28, frame_chunk=8, stop_after_macro=None)
        validate_train_request(spec, args)
        args.resume = Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928'
                           '/continuation900/T/train/attempts/continuation/checkpoints/macro_00000450')
        with pytest.raises(ValueError, match='fresh identity'):
            validate_train_request(spec, args)
        args.attempt = 'foreign'
        contract = {'loss_variant': joint_training.LOSS, 'joint': joint}
        with pytest.raises(ValueError, match='owned same-loss'):
            validate_attempt(spec, args, contract, root/mode/'train/attempts/foreign')
    finally:
        candidate.close()
        reference.close()
