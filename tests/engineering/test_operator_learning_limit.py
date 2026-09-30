"""Actual consumers for the bounded §33 manifest and algebraic readback."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from ember.lora import LORA_A_SUFFIX
from ember.operator_writer.model import TargetWrite


def script():
    path = Path(__file__).resolve().parents[2] / 'scripts/operator_learning_limit_diagnosis.py'
    spec = importlib.util.spec_from_file_location('learning_limit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_manifest_actual_seed_api_and_partition():
    module = script()
    panels = module.panels()
    lengths = {t: [200]*50 for t in module.TASKS}
    manifest = module.query_manifest(panels, lengths)
    assert manifest == module.query_manifest(panels, lengths)
    assert len(manifest['steps']) == 64
    for step, tasks in enumerate(manifest['steps']):
        for task in module.TASKS:
            row = tasks[str(task)]
            queries = row['queries']
            assert [q['demo'] for q in queries] == sorted(q['demo'] for q in panels[str(task)]['A'])
            assert not {q['demo'] for q in queries} & {q['demo'] for q in panels[str(task)]['B']}
            assert row['visit'] == 330000+step
            assert all(0 <= q['frame'] <= 198 for q in queries)
    assert manifest['steps'][0] != manifest['steps'][1]
    assert callable(module.prior_readout().b_batch)


def test_z_readback_through_actual_targetwrite():
    module = script()
    torch.manual_seed(5)
    write = TargetWrite(32, 16)
    torch.nn.init.normal_(write.o.weight, std=.02)
    address = torch.randn(128,32)
    x = torch.randn(4,50,32)
    h = torch.randn(4,50,1024)
    writer = SimpleNamespace(names=('site',), writes=(write,),
        public_state=lambda: {'site'+LORA_A_SUFFIX: address})
    z = module.latent_z(writer, ({'site':x}, h))['site']
    assert z.shape == (256,128)
    assert torch.allclose(write.o.weight @ z, write(address,x,h), atol=2e-5, rtol=2e-4)


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
