"""Sealed source identity and actual finite-panel capture contracts; CPU only."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from ember.operator_writer import reexpression as owner
from ember.pi05_assets import Pi05EvaluationError


def test_bank_reader_keeps_sealed_spec_identity_after_relocation(monkeypatch,tmp_path):
    import json
    spec={'source':{'checkpoint':'fixed-source'},'task_panel':[0,12,20,32]}
    training={'checkpoint':'fixed-writer'}
    monkeypatch.setattr(owner,'ROOT',tmp_path/'original-run')
    monkeypatch.setattr(owner,'source_record',lambda:(spec,training,tmp_path/'new-reader/spec.json'))
    sealed=owner.ROOT/'frozen/configs/operator_read_write_v1/conditional_read_write_continuation900_spec.json'
    sealed.parent.mkdir(parents=True)
    sealed.write_text(json.dumps(spec))
    assert owner.sealed_source_record()==(spec,training,sealed)
    sealed.write_text(json.dumps({**spec,'task_panel':[0,12,20]}))
    with pytest.raises(ValueError,match='sealed spec differs'):
        owner.sealed_source_record()


def test_actual_capture_routes_both_teachers_full_state0(monkeypatch,tmp_path):
    from ember.operator_writer import capture
    from ember.pi05_eval.preparation import _registered_trajectory_capture
    monkeypatch.setattr(owner,'ROOT',tmp_path/owner.ROOT.name)
    owner.register_inputs()
    rows=owner.tasks_for_panel()
    tasks=[SimpleNamespace(suite=r['suite'],task_id=r['task_id'],init_state_ids=owner.STATES) for r in rows]
    original=capture.read_json
    for arm in owner.ARMS:
        for slot in (0,1):
            bank=owner.bank_path(arm,slot)
            monkeypatch.setattr(capture,'read_json',lambda p:{'mode':arm,'reexpression_panel':{'teacher_slot':slot}} if p==bank else original(p))
            args=SimpleNamespace(static_task_lora_manifest=bank,role='development_train',mode='screen',
                trajectory_capture_selection=owner.ROOT/f'launch/capture_teacher{slot}.json',
                occupancy_capture_selection=None,capture_stage_predicates=False)
            output=owner.ROOT/arm/'evaluation'/f'teacher{slot}'
            result,stage=_registered_trajectory_capture(args,tasks,output,
                {'selection_path':str(owner.ROOT/'launch/train4_subset.json')},Path(__file__).resolve().parents[2])
            assert len(result['full_conditions'])==4 and not stage['full_conditions_only']
            assert result['passive_trace']['trace_root']==str(output/'continuous_traces')
            args.role='validation'
            with pytest.raises(Pi05EvaluationError,match="scope changed"):
                _registered_trajectory_capture(args,tasks,output,
                    {'selection_path':str(owner.ROOT/'launch/train4_subset.json')},Path(__file__).resolve().parents[2])
