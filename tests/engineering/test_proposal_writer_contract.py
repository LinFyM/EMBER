"""Retained numerical-coordinate and complete-score protocol regressions."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ember.pi05_lora import load_pi05_lora_contract
from ember.lora import expected_lora_state_shapes
from ember.proposal_writer.model import FactorLayout
from ember.proposal_writer.path import score_log_probability
from ember.proposal_writer.contract import pilot_panel
from ember.writer.practice.execution import action_chunk


def test_complete_original_coordinate_roundtrip():
    root=Path(__file__).resolve().parents[2]
    contract=load_pi05_lora_contract(root/'configs/pi05_lora_rank128_aligned.json')
    generator=torch.Generator().manual_seed(11)
    state={k:torch.randn(shape,generator=generator) for k,shape in expected_lora_state_shapes(contract).items()}
    layout=FactorLayout(contract,state);blocks=layout.pack(state)
    # Arbitrary padding cannot change any original A row/B column.
    blocks[~layout.valid]=123.
    restored=layout.unpack(blocks)
    assert layout.count==10297344 and len(blocks)==161024
    assert all(torch.equal(restored[k],v) for k,v in state.items())


def test_condition_sum_and_forced_zero_credit():
    logits=torch.tensor([.2,.6],requires_grad=True)
    score=score_log_probability(logits,0)+score_log_probability(logits,1)
    forced=score_log_probability(logits[:1],0,forced=True)
    (score+forced).backward()
    torch.testing.assert_close(logits.grad,torch.ones(2)-2*logits.detach().softmax(0))
    assert float(forced.detach())==0


def test_roles_are_disjoint_full_fifty_panels():
    for row in pilot_panel()['tasks']:
        roles=['selection_states','audit_states','practice_states','local_query_states','rl_query_states','report_states']
        states=[s for role in roles for s in row[role]]
        assert len(states)==50 and set(states)==set(range(50))
        assert len(set(row['video_order']))==50


def test_full_native_latent_and_same_call_hidden_times():
    class Velocity(nn.Module):
        def __init__(self):
            super().__init__();self.policy=SimpleNamespace(model=SimpleNamespace(action_out_proj=nn.Identity()))
            self.calls=[]
        def forward(self,value,time):
            assert value.shape==(2,50,32)
            self.calls.append(time)
            self.policy.model.action_out_proj(torch.full((2,50,1024),time))
            return torch.ones_like(value)
    velocity=Velocity();actions,hidden=action_chunk(velocity,torch.ones(2,50,32))
    assert len(velocity.calls)==10 and actions.shape==(2,50,7) and hidden.shape==(2,2,50,1024)
    torch.testing.assert_close(actions,torch.zeros_like(actions),atol=1e-6,rtol=0)
    assert float(hidden[0,0,0,0])==1. and abs(float(hidden[0,1,0,0])-.1)<.001


def test_frozen_response_cache_keeps_fresh_actor_gradient():
    from ember.proposal_writer.path import decision_logits
    from ember.writer.practice import History
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(2.));self.calls=0
        def forward(self,kind,features,rows,states,candidates,responses,budget):
            self.calls+=1
            return self.weight*responses['MT300'][0,:1]
    class Runtime:
        def __init__(self):
            self.device=torch.device('cpu');self.actor=Reader();self.calls=0
            self.tasks={12:SimpleNamespace(authority=SimpleNamespace(language='real target language'))}
        def observation_features(self,observations):return {}
        def responses(self,states,history,language,*,parent=None):
            self.calls+=1;return {'MT300':torch.ones(1,70)}
    runtime=Runtime();state={'factor':torch.ones(1)}
    path=SimpleNamespace(task=12,history=History(states={'MT300':state}),states={'MT300':state})
    decision=dict(kind='final',candidates=['MT300'],records=0,episodes=0,budget=[1.,1.,0.,0.])
    first=decision_logits(runtime,path,decision,{})
    first.sum().backward()
    assert float(runtime.actor.weight.grad)==1
    with torch.no_grad():runtime.actor.weight.add_(1.)
    runtime.actor.zero_grad(set_to_none=True)
    second=decision_logits(runtime,path,decision,{})
    second.sum().backward()
    assert runtime.calls==1 and runtime.actor.calls==2
    assert float(second.detach()[0])==3 and float(runtime.actor.weight.grad)==1


def test_candidate_retirement_requires_consumption_and_preserves_facts(tmp_path):
    import pytest
    from ember.proposal_writer.path import Compilation,retire_consumed_parameters,load_history
    from ember.writer.practice import History
    from ember.pi05_source_checkpoint import write_json_atomic
    names=['layer.lora_A.default.weight','layer.lora_B.default.weight']
    mt={k:torch.ones(2,2) for k in names}
    states={'MT300':mt,'G_000':{k:v+1 for k,v in mt.items()},'G_001':{k:v+2 for k,v in mt.items()}}
    raw=torch.arange(24,dtype=torch.uint8).reshape(2,3,2,2)
    hidden=torch.ones(2,50,3)
    h=History(records=[dict(parameter_ref='G_000',hidden=hidden,actions=torch.ones(5,7))],
        observations={'seen':dict(images=raw)},states=states,image_features={'seen':torch.ones(2,3)})
    path=Compilation(12,42,'owned_test',0,7,h,states,practiced=list(states),valid=list(states),
        selected='G_001',kernel_identity={'checkpoint':'retained_fixed_psi'})
    path.save(tmp_path)
    proof=tmp_path/'consumer.json';write_json_atomic(proof,{'complete':False})
    with pytest.raises(ValueError,match='complete consumer'):retire_consumed_parameters(tmp_path,proof,mt)
    assert (tmp_path/'G_000.safetensors').exists()
    write_json_atomic(proof,{'complete':True})
    result=retire_consumed_parameters(tmp_path,proof,mt)
    saved=load_history(tmp_path/'history.pt.gz')
    assert not (tmp_path/'G_000.safetensors').exists() and (tmp_path/'G_001.safetensors').exists()
    assert torch.equal(saved.observations['seen']['images'],raw) and torch.equal(saved.records[0]['hidden'],hidden)
    assert set(saved.states)=={'MT300'} and not saved.image_features
    assert result['deleted_parameter_bytes']>0 and result['full_compilation_replay_available'] is False
