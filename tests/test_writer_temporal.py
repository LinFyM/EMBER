"""Verify the A-matched removal of frame order without removing content."""
import pytest
import torch
from torch import nn
from ember.writer.temporal import (FrameSetProcedureEncoder, LanguageSemanticCore,
    RoPEContentBlock, SlotNormalizedCoreProcedureCompiler)

@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(previous)

def components():
    core = LanguageSemanticCore(width=16, heads=4, blocks=2, frame_attention_initial_lambda=.05)
    procedure = FrameSetProcedureEncoder(width=16, heads=4, blocks=2)
    compiler = SlotNormalizedCoreProcedureCompiler(width=16, heads=4, initialization_seed=8)
    with torch.no_grad():
        compiler.modulation.weight.normal_(std=.1)
    return core, procedure, compiler

@pytest.mark.parametrize('permutation', [[3,0,4,1,2], [4,3,2,1,0]])
def test_content_equivariance_and_nonzero_slot_invariance(permutation):
    core, procedure, compiler = components()
    q, evidence = torch.randn(1,3,16), torch.randn(1,5,3,16)
    response = torch.randn(1,5,16,requires_grad=True)
    valid, tokens = torch.ones(1,5,dtype=torch.bool), torch.ones(1,3,dtype=torch.bool)
    positions = torch.tensor([[0,5,10,15,18]])
    c, _ = core(q,evidence,valid,tokens)
    p = procedure(response,positions,valid)
    expected = compiler(c,tokens,p,positions,valid)
    changed_c, _ = core(q,evidence[:,permutation],valid,tokens)
    changed_p = procedure(response[:,permutation],positions,valid)
    torch.testing.assert_close(changed_c,c,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(changed_p,p[:,permutation],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(compiler(changed_c,tokens,changed_p,positions,valid),expected,atol=3e-6,rtol=3e-5)
    different = response.detach().clone()
    different[:,2] += torch.randn(16)
    other = compiler(c,tokens,procedure(different,positions,valid),positions,valid)
    assert not torch.allclose(other[0],expected[0])
    sum((value*torch.randn_like(value)).sum() for value in expected).backward()
    assert bool((response.grad.norm(dim=-1)>0).all())

def test_frame_indices_do_not_address_procedure_or_reader():
    _, procedure, compiler = components()
    x, c = torch.randn(1,5,16), torch.randn(1,3,16)
    valid, tokens = torch.ones(1,5,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool)
    first, second = torch.tensor([[0,5,10,15,18]]),torch.tensor([[0,91,142,245,699]])
    p = procedure(x,first,valid)
    torch.testing.assert_close(procedure(x,second,valid),p)
    torch.testing.assert_close(compiler(c,tokens,p,first,valid),compiler(c,tokens,p,second,valid))
    assert all(not b.causal and not b.rotary for b in procedure.blocks)
    assert not compiler.procedure_reader.attention.rotary_keys

def test_language_positions_are_retained():
    core,_,_ = components()
    q,evidence = torch.randn(1,4,16),torch.randn(1,3,4,16)
    valid,tokens = torch.ones(1,3,dtype=torch.bool),torch.ones(1,4,dtype=torch.bool)
    permutation=[2,0,3,1]
    expected,_ = core(q,evidence,valid,tokens)
    observed,_ = core(q[:,permutation],evidence[:,:,permutation],valid,tokens)
    assert all(b.rotary for b in core.blocks)
    assert not torch.allclose(observed,expected[:,permutation],atol=1e-5,rtol=1e-5)

def test_padding_has_no_content_or_gradient():
    _,procedure,_ = components()
    x=torch.randn(2,5,16,requires_grad=True)
    valid=torch.tensor([[True,True,True,False,False],[True]*5])
    positions=torch.tensor([[0,5,9,0,0],[0,5,10,15,19]])
    observed=procedure(x,positions,valid)
    torch.testing.assert_close(observed[:1,:3],procedure(x[:1,:3],positions[:1,:3],valid[:1,:3]))
    assert observed[~valid].count_nonzero()==0
    observed.square().sum().backward()
    assert x.grad[~valid].count_nonzero()==0

def test_same_parameter_count_and_initialization_rng():
    torch.manual_seed(7)
    original=nn.ModuleList(RoPEContentBlock(width=16,heads=4,causal=True) for _ in range(2))
    expected_rng=torch.get_rng_state()
    torch.manual_seed(7)
    frame_set=FrameSetProcedureEncoder(width=16,heads=4,blocks=2)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert sum(p.numel() for p in original.parameters())==sum(p.numel() for p in frame_set.parameters())

class EvidenceEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale=nn.Parameter(torch.randn(256)*.1)
    def forward(self, policy, frames, ids, tokens, masks, spans, *, frame_parallel_group=None):
        self.group=frame_parallel_group
        signal=frames.float().mean(dim=tuple(range(1,frames.ndim)))/255
        axis=torch.linspace(.1,2.,256)
        evidence=(signal[:,None,None]+axis[None,None]+torch.arange(3)[None,:,None]*.2).sin()*self.scale
        interactions=(signal[:,None]*axis).cos()*self.scale
        query=(tokens[:,:3,None].float()+axis).sin()
        return query,evidence,interactions,torch.ones(tokens.shape[0],3,dtype=torch.bool)

@pytest.mark.parametrize('batch',[1,2])
def test_complete_writer_nonidentity_set_contract(monkeypatch,batch):
    from pathlib import Path
    from types import SimpleNamespace
    from ember.lora import identity_lora_state,validate_lora_state
    from ember.pi05_lora import load_pi05_lora_contract
    from ember.writer import model as module
    monkeypatch.setattr(module,'Pi05LanguageAxialEncoder',EvidenceEncoder)
    contract=load_pi05_lora_contract(Path(__file__).resolve().parents[1]/'configs/pi05_lora_v1.json')
    template=identity_lora_state(contract)
    backbone=SimpleNamespace(layers=range(18))
    model=module.CompleteLoRAWriter(module.build_lora_tensor_specs(template),template_state=template,
        paligemma_model=backbone,expert_model=backbone,image_width=2048,expert_width=1024,
        program_width=256,text_meta_lora_rank=4,vl_meta_lora_rank=4,action_meta_lora_rank=4,
        patch_grounding_heads=8,max_frames_per_encoder_call=8,action_horizon=50,padded_action_dim=32,
        semantic_core_heads=8,semantic_core_blocks=2,frame_attention_initial_lambda=.05,
        procedure_heads=8,procedure_blocks=2,fusion_heads=8,factor_hidden_width=216,
        initialization_seed=7,activation_checkpointing=True)
    assert sum(p.numel() for p in model.factor_heads.parameters())==1_838_592
    frames=torch.arange(batch*3,dtype=torch.uint8)[:,None,None,None].expand(-1,3,4,4)*37
    indices=torch.tensor([0,5,8]*batch)
    tokens=torch.arange(1,4).expand(batch,-1)
    masks=torch.ones_like(tokens,dtype=torch.bool)
    args=(frames,indices,torch.arange(batch+1)*3,tokens,masks,masks)
    state=model(*args,policy=None)
    assert len(state)==76
    validate_lora_state(state if batch==1 else {k:v[0] for k,v in state.items()},contract)
    expected=template if batch==1 else {k:v[None].expand(batch,-1,-1) for k,v in template.items()}
    torch.testing.assert_close(state,expected)
    with torch.no_grad():
        for head in model.factor_heads.values():
            head.network[-1].weight.normal_(std=.01)
        model.compiler.modulation.weight.normal_(std=.01)
    learned=model(*args,policy=None,frame_parallel_group='forwarded-group')
    assert model.semantic_encoder.group=='forwarded-group'
    permutation=torch.tensor([2,0,1])[None]+torch.arange(batch)[:,None]*3
    permuted=model(frames[permutation.flatten()],*args[1:],policy=None)
    torch.testing.assert_close(permuted,learned,atol=3e-6,rtol=3e-5)
    assert any(not torch.allclose(learned[k],expected[k]) for k in learned)
    sum(value.square().mean() for value in learned.values()).backward()
    assert model.semantic_encoder.scale.grad.norm()>0
    assert model.compiler.modulation.weight.grad.norm()>0
