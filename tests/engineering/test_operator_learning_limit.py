"""Actual consumers for the bounded §33 manifest and algebraic readback."""
from pathlib import Path
from types import SimpleNamespace

import torch





def test_actual_capture_entry_routes_both_teacher_slots(monkeypatch):
    from ember.operator_writer import capture,learning_limit as owner
    from ember.pi05_eval.preparation import _registered_trajectory_capture
    original=capture.read_json
    tasks=[SimpleNamespace(suite=r['suite'],task_id=r['task_id'],init_state_ids=(0,1,2,3))
           for r in owner.tasks_for_panel()]
    for slot in (0,1):
        bank=owner.bank_path('parent',slot)
        monkeypatch.setattr(capture,'read_json',lambda p: {'mode':'parent',
            'learning_limit_panel':{'teacher_slot':slot}} if p==bank else original(p))
        args=SimpleNamespace(static_task_lora_manifest=bank,role='development_train',mode='screen',
            trajectory_capture_selection=owner.ROOT/f'launch/capture_teacher{slot}.json',
            occupancy_capture_selection=None,capture_stage_predicates=False)
        output=owner.ROOT/'parent/evaluation'/f'teacher{slot}'
        result,stage=_registered_trajectory_capture(args,tasks,output,{'selection_path':str(owner.ROOT/'launch/train4_subset.json')},Path(__file__).resolve().parents[2])
        assert len(result['full_conditions'])==(4 if slot==0 else 0)
        assert result['passive_trace']['schema_version']==capture.PASSIVE_TAG
        assert stage['full_conditions_only'] is False


def test_projected_panel_actual_capture_entry_reuses_original_states(monkeypatch,tmp_path):
    from ember.operator_writer import capture,learning_limit as owner
    from ember.pi05_eval.preparation import _registered_trajectory_capture
    root=tmp_path/'operator_projected_repair_consumers_20260930'
    monkeypatch.setattr(owner,'PROJECTED_ROOT',root)
    owner.register_inputs('PZ')
    original=capture.read_json
    tasks=[SimpleNamespace(suite=r['suite'],task_id=r['task_id'],init_state_ids=(0,1,2,3))
           for r in owner.tasks_for_panel()]
    for slot in (0,1):
        bank=owner.bank_path('PZ',slot)
        monkeypatch.setattr(capture,'read_json',lambda p: {'mode':'PZ',
            'learning_limit_panel':{'teacher_slot':slot}} if p==bank else original(p))
        args=SimpleNamespace(static_task_lora_manifest=bank,role='development_train',mode='screen',
            trajectory_capture_selection=root/f'launch/capture_teacher{slot}.json',
            occupancy_capture_selection=None,capture_stage_predicates=False)
        result,stage=_registered_trajectory_capture(args,tasks,root/'PZ/evaluation'/f'teacher{slot}',
            {'selection_path':str(root/'launch/train4_subset.json')},Path(__file__).resolve().parents[2])
        assert len(result['full_conditions'])==(4 if slot==0 else 0)
        assert result['trajectory_root']==str(root/'PZ/evaluation'/f'teacher{slot}'/'trajectories')
        assert result['passive_trace']['schema_version']==capture.PASSIVE_TAG
        assert stage['full_conditions_only'] is False
