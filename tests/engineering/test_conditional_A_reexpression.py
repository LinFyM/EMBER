"""Numerical-rank and actual finite-panel consumer contracts; CPU only."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from ember.operator_writer import reexpression as owner
from ember.pi05_assets import Pi05EvaluationError
from ember.operator_writer.functional_readout import masked_risk


def test_default_rank_deficient_projection_preserves_unexcited_B():
    a0=torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,0.,0.]])
    s=torch.tensor([[2.,0.,0.],[0.,3.,0.],[4.,0.,0.]])
    b=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[2.,3.,0.]])
    at,bt,c,stats=owner.reexpress_target(a0,s,b,x)
    assert stats['address_rank']==2
    torch.testing.assert_close(at,a0)
    torch.testing.assert_close(x@(a0+s).T@b.T,x@at.T@bt.T)
    # I retains the original B on the address direction not excited by A0 X.
    null=torch.tensor([1.,0.,-1.],dtype=torch.float64)
    torch.testing.assert_close(bt.double()@null,b.double()@null)
    assert c.dtype==torch.float64


def test_action_in_full_column_rank_and_nonrepresentable_hidden():
    generator=torch.Generator().manual_seed(3)
    a0=torch.randn(128,32,generator=generator)
    s=torch.randn(128,32,generator=generator)
    b=torch.randn(7,128,generator=generator)
    x=torch.randn(60,32,generator=generator)
    at,bt,c,stats=owner.reexpress_target(a0,s,b,x)
    assert stats['address_rank']==32
    torch.testing.assert_close(x@(a0+s).T@b.T,x@at.T@bt.T,rtol=2e-5,atol=2e-4)
    # Teacher-support equality does not imply equality outside that support.
    a=torch.tensor([[1.,0.]]);edit=torch.tensor([[0.,2.]])
    at,bt,_,_=owner.reexpress_target(a,edit,torch.ones(1,1),torch.tensor([[1.,0.],[2.,0.]]))
    h=torch.tensor([[0.,1.]])
    assert not torch.allclose(h@(a+edit).T,h@at.T@bt.T)


def test_full38_and_nonfinite_are_enforced():
    with pytest.raises(ValueError,match='all38'):
        owner.reexpress_compilation({},dict(x={}),('Q8','V8','out'))
    with pytest.raises(ValueError,match='nonfinite'):
        owner.reexpress_target(torch.ones(1,2),torch.full((1,2),float('nan')),torch.ones(1,1),torch.ones(2,2))


def test_mask_keeps_full50_and_excludes_padded_future():
    prediction=torch.zeros(2,50,7);target=torch.zeros_like(prediction)
    prediction[:,3:]=10
    valid=torch.arange(50)[None].expand(2,-1)<3
    scores=masked_risk(prediction,target,valid)
    assert scores['full50']>90 and scores['valid_future']==0 and scores['valid_first5']==0


def test_passive_hidden_summary_uses_original_inputs_and_all_ten_steps():
    from ember.operator_writer.functional_readout import LocalEffects
    from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX
    policy=torch.nn.Module();policy.target=torch.nn.Linear(2,2,bias=False)
    a=torch.tensor([[1.,0.]]);b=torch.tensor([[1.],[2.]])
    original={'target'+LORA_A_SUFFIX:a,'target'+LORA_B_SUFFIX:b}
    edited={'target'+LORA_A_SUFFIX:a,'target'+LORA_B_SUFFIX:2*b}
    effect=LocalEffects(policy,original,edited,{'target':a},'cpu',14)
    h=torch.ones(14,50,2)
    with effect.capture('FM'):
        for _ in range(2):policy.target(h)
    with effect.capture('generation'):
        for _ in range(20):policy.target(h)
    records=effect.summary()
    assert [r['calls'] for r in records]==[2,20]
    assert all(r['relative_E_to_original']==1 for r in records)
    assert records[1]['E_squared']==[10*v for v in records[0]['E_squared']]


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
