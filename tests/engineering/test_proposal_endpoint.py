"""Signal-scale invariants, retired checkpoint rejection and actual chunk consumer."""
from types import SimpleNamespace

import pytest
import torch

from ember.proposal_writer.model import BlockLayer


def test_endpoint_messages_do_not_amplify_accumulated_sender_scale():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    layer=BlockLayer(normalize_messages=True)
    with torch.no_grad():
        for name,parameter in layer.named_parameters():
            if 'norm' not in name:parameter.zero_()
        blocks=torch.randn(6,256)
        ranks=torch.zeros(1,2,256);targets=torch.zeros(1,256)
        group=torch.tensor([0,0,0,1,1,1]);memory=torch.empty(0,256)
        _,r,t=layer(blocks,ranks,targets,group,memory)
        _,large_r,large_t=layer(blocks*1000,ranks,targets,group,memory)
    # The local residual keeps amplitude; cross-level messages keep normalized signal.
    torch.testing.assert_close(r,large_r,atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(t,large_t,atol=3e-5,rtol=3e-5)
    assert float(r.square().mean().sqrt())<3
    assert float(t.square().mean().sqrt())<1.1


def test_sender_role_preserves_initialization_draws_and_parameter_schema():
    torch.manual_seed(7);static=BlockLayer();after_static=torch.rand(3)
    torch.manual_seed(7);endpoint=BlockLayer(normalize_messages=True);after_endpoint=torch.rand(3)
    assert list(static.state_dict())==list(endpoint.state_dict())
    assert torch.equal(after_static,after_endpoint)
    assert not static.normalize_messages and endpoint.normalize_messages


def test_same_shape_previous_writer_cannot_be_loaded(tmp_path):
    from ember.proposal_writer.pipeline import load_version
    torch.save({'schema_version':'ember_full36_sender_normalized_flow_writer_v6'},tmp_path/'state.pt')
    with pytest.raises(ValueError,match='fresh training required'):
        load_version(object(),tmp_path)


@pytest.mark.parametrize('rank,expected',[(0,16),(1,8)])
def test_runtime_creation_consumes_rank_chunks(monkeypatch,rank,expected):
    from ember.proposal_writer import run
    monkeypatch.setenv('EMBER_META_FRAME_CHUNKS','16,8')
    seen=[]
    monkeypatch.setattr(run,'Runtime',lambda device,**kwargs:seen.append(kwargs) or SimpleNamespace(**kwargs))
    args=SimpleNamespace(asset_root='assets',root='output',frame_chunk=16)
    context=SimpleNamespace(device='cpu',world_size=2,local_rank=rank)
    runtime=run.create_runtime(args,context,dict(tasks=[dict(global_task_id=13)],role='train'))
    assert runtime.frame_chunk==expected and seen[0]['frame_chunk']==expected


@pytest.mark.parametrize('chunks',['16','16,0','16,8,4'])
def test_invalid_chunk_topology_is_rejected(monkeypatch,chunks):
    from ember.proposal_writer.run import frame_chunk_for_rank
    monkeypatch.setenv('EMBER_META_FRAME_CHUNKS',chunks)
    with pytest.raises(ValueError,match='world size'):
        frame_chunk_for_rank(16,SimpleNamespace(world_size=2,local_rank=0))


def small_layout():
    """Actual 38×rank8 grouping with short blocks for a bounded decoder regression."""
    layout=torch.nn.Module()
    n=38*8
    layout.factor_rank=8
    layout.register_buffer('target',torch.arange(38).repeat_interleave(8))
    layout.register_buffer('rank',torch.arange(8).repeat(38))
    layout.register_buffer('side',torch.zeros(n,dtype=torch.long))
    layout.register_buffer('coordinate',torch.zeros(n,dtype=torch.long))
    layout.register_buffer('group',torch.arange(n))
    mask=torch.ones(n,64,dtype=torch.bool);mask[:,-1]=False
    layout.register_buffer('valid',mask)
    layout.register_buffer('center',torch.randn(n,64)*mask)
    layout.register_buffer('scale',torch.full((n,1),.3))
    layout.pack=lambda state:state['complete']
    layout.unpack=lambda value:{'complete':value}
    return layout


def test_zero_head_returns_MT_center_for_a_changed_parent_without_RNG_consumption():
    from ember.proposal_writer.model import Generator
    torch.set_num_threads(2);torch.manual_seed(7)
    layout=small_layout();generator=Generator(layout,torch.nn.Identity()).eval()
    parent={'complete':layout.center+2*layout.valid}
    memory=torch.randn(3,256)
    before=torch.random.get_rng_state()
    first=generator.generate(parent,None,None,None,memory=memory)
    second=generator.generate(parent,None,None,None,memory=memory)
    assert torch.equal(before,torch.random.get_rng_state())
    torch.testing.assert_close(first['complete'],layout.center)
    torch.testing.assert_close(first['complete'],second['complete'])
    assert generator.endpoint.input.in_features==128
    assert all(layer.normalize_messages for layer in generator.endpoint.layers)
    assert not any(layer.normalize_messages for layer in generator.static.layers)
    assert not hasattr(generator,'field')


def test_endpoint_zero_head_opens_upstream_gradient_after_first_update_and_masks_padding():
    from ember.proposal_writer.model import FactorEncoder
    torch.set_num_threads(2);torch.manual_seed(7)
    layout=small_layout();decoder=FactorEncoder(layout,6,endpoint=True)
    parent=torch.randn_like(layout.center);memory=torch.randn(4,256,requires_grad=True)
    target=torch.randn_like(parent)*layout.valid
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=1e-4)
    first=decoder(parent,memory=memory)
    assert torch.count_nonzero(first)==0
    loss=(first[layout.valid]-target[layout.valid]).square().mean();loss.backward()
    assert decoder.output.weight.grad.norm()>0 and memory.grad.norm()==0
    optimizer.step();optimizer.zero_grad();memory.grad=None
    second=decoder(parent,memory=memory)
    assert torch.count_nonzero(second[~layout.valid])==0
    (second[layout.valid]-target[layout.valid]).square().mean().backward()
    assert memory.grad.norm()>0 and decoder.input.weight.grad.norm()>0


def test_only_one_deterministic_candidate_is_generated_and_invalid_output_is_not_retried(monkeypatch):
    from ember.proposal_writer import path as owner
    from ember.writer.practice import History
    calls=[]
    history=History(records=[{'parameter_ref':'MT300'}],episodes=[{}])
    history.model_rows=lambda runtime:[]
    compiled=owner.Compilation(13,47,'task_0013',0,7,history,{'MT300':{'factor':torch.ones(1)}})
    generator=SimpleNamespace(context=lambda *args:torch.empty(0,256),
        generate=lambda *args,**kwargs:calls.append(kwargs) or {'factor':torch.ones(1)})
    runtime=SimpleNamespace(device=torch.device('cpu'),components={},generator=generator,lora=None)
    monkeypatch.setattr(owner,'validate_lora_state',lambda *args:None)
    assert owner.generate_candidates(runtime,compiled,'MT300',{})==['G_000']
    assert len(calls)==1 and len(compiled.attempts)==1
    assert compiled.attempts[0]['decoder_calls']==1 and 'noise_seed' not in compiled.attempts[0]
    generator.generate=lambda *args,**kwargs:calls.append(kwargs) or {'factor':torch.tensor([float('nan')])}
    with pytest.raises(ValueError,match='invalid deterministic endpoint'):
        owner.generate_candidates(runtime,compiled,'MT300',{})
    assert len(calls)==2 and len(compiled.attempts)==2
