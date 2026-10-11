"""Signal-scale invariants, retired checkpoint rejection and actual chunk consumer."""
from types import SimpleNamespace

import pytest
import torch

from ember.proposal_writer.model import BlockLayer


def test_flow_messages_do_not_amplify_accumulated_sender_scale():
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
    torch.manual_seed(7);flow=BlockLayer(normalize_messages=True);after_flow=torch.rand(3)
    assert list(static.state_dict())==list(flow.state_dict())
    assert torch.equal(after_static,after_flow)
    assert not static.normalize_messages and flow.normalize_messages


def test_same_shape_previous_writer_cannot_be_loaded(tmp_path):
    from ember.proposal_writer.pipeline import load_version
    torch.save({'schema_version':'ember_full36_student_joint_conditional_continue_writer_v5'},tmp_path/'state.pt')
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
