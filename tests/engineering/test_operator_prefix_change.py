"""Literal native-prefix response, credit and original-T inclusion oracles."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.gemma import modeling_gemma
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, make_att_2d_masks

from ember.lora import LoRATarget, LORA_B_SUFFIX, identity_lora_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.operator_writer import prefix_change as pc, native, joint_training, joint_readout, scope
from ember.operator_writer.model import OperatorReadWrite
from ember.operator_writer.specification import specification, PREFIX_CHANGE_SPEC_PATH, JOINT_SPEC_PATH
from ember.writer.materialization import planned_episodes


def test_rotated_native_whole_prefix_swap_current_suffix_boundary_and_credit():
    torch.set_num_threads(4)
    torch.manual_seed(41)
    n, prefix, suffix, width = 5, 6, 50, 256
    attention = SimpleNamespace(num_key_value_groups=8, training=False, scaling=width**-.5)
    projection = nn.Linear(8*width, 1024, bias=True).requires_grad_(False)
    q = torch.randn(n, 8, suffix, width, requires_grad=True)
    kp, vp = torch.randn(n, 1, prefix, width), torch.randn(n, 1, prefix, width)
    ks = torch.randn(n, 1, suffix, width, requires_grad=True)
    vs = torch.randn_like(ks, requires_grad=True)
    # Use the installed native RoPE operation and true token coordinates.
    pos = torch.arange(prefix+suffix).float()
    angle = pos[:, None] * torch.linspace(.001, .03, width)[None]
    full_q = torch.cat((torch.zeros(n, 8, prefix, width), q), 2)
    q_rot, key_rot = modeling_gemma.apply_rotary_pos_emb(
        full_q, torch.cat((kp, ks), 2), angle.cos()[None], angle.sin()[None])
    q, kp, ks = q_rot[:, :, -50:], key_rot[:, :, :-50].detach(), key_rot[:, :, -50:]
    padding = torch.ones(1, prefix+suffix, dtype=torch.bool); padding[:, 4:6] = False
    blocks = torch.zeros_like(padding); blocks[:, prefix] = True
    mask = PI05Pytorch._prepare_attention_masks_4d(None, make_att_2d_masks(padding, blocks))[:, :, -50:]
    def actual(q, k, v):
        result, _ = modeling_gemma.eager_attention_forward(attention, q, k, v, mask, attention.scaling)
        return projection(result.flatten(2))
    u = actual(q, torch.cat((kp, ks), 2), torch.cat((vp, vs), 2))
    fields = (q, kp, vp, ks, vs, u)
    chunks = [tuple(x[a:b] for x in fields)+(mask,) for a,b in ((0,2),(2,4),(4,5))]
    observed = pc.response_difference(chunks, attention, projection, frame_chunk=2)
    expected = pc.rms(actual(q[:-1], torch.cat((kp[1:], ks[:-1]), 2),
                             torch.cat((vp[1:], vs[:-1]), 2))) - pc.rms(u[:-1])
    assert observed.shape == (4,50,1024)  # Both boundaries and the last real frame.
    torch.testing.assert_close(observed, expected, rtol=4e-5, atol=5e-6)
    gradients = torch.autograd.grad((observed*torch.randn_like(observed)).sum(), (q, ks, vs))
    assert all(torch.isfinite(g).all() and g[:-1].norm()>0 for g in gradients)
    assert not kp.requires_grad and not vp.requires_grad
    altered = list(fields); altered[4] = vs+torch.randn_like(vs)*.2
    changed = pc.response_difference([tuple(altered)+(mask,)], attention, projection, frame_chunk=5)
    assert (changed-observed).norm()>0


def writer(mode):
    contract=derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(__file__).resolve().parents[2]/"configs/pi05_lora_v1.json"),rank=128)
    contract=replace(contract,targets=tuple(LoRATarget(t.name,4,3) for t in contract.targets))
    return OperatorReadWrite(contract,identity_lora_state(contract),mode)


def test_zero_E_exact_original_initialization_inclusion_and_live_value_credit():
    old=writer("T"); before=torch.random.get_rng_state(); new=writer(pc.MODE)
    assert torch.equal(before,torch.random.get_rng_state())
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==9961472
    for i in (0,37):
        for group in ("p","c","d","o"):
            torch.testing.assert_close(getattr(old.writes[i],group).weight,getattr(new.writes[i],group).weight)
    assert all(w.e.bias is None and w.e.weight.count_nonzero()==0 for w in new.writes)
    with torch.no_grad():
        old.writes[0].o.weight.normal_(std=.02)
    missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
    assert len(missing)==38 and not unexpected
    h=torch.randn(3,50,1024,requires_grad=True)
    x={name:torch.randn(3,50,4,requires_grad=True) for name in new.names}
    dp=torch.randn(2,50,1024,requires_grad=True)
    expected=old(x,h); actual=new(x,h,prefix_change=dp)
    first=new.names[0]+LORA_B_SUFFIX
    torch.testing.assert_close(actual[first],expected[first])
    with torch.no_grad(): new.writes[0].e.weight.normal_(std=.03)
    state=new(x,h,prefix_change=dp)
    state[first].square().sum().backward()
    for value in (h,x[new.names[0]],dp,new.writes[0].e.weight,new.writes[0].d.weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.norm()>0
    with torch.no_grad(): new(x,h,prefix_change=dp,capture_prefix_stats=True)
    stats=new.last_prefix_statistics
    assert stats["dP_norm"].shape==(2,50) and len(stats["targets"])==38


def test_unique_full_only_fresh_spec_readouts_and_eight_real_seen_mappings():
    spec=specification(PREFIX_CHANGE_SPEC_PATH)
    reference=specification(JOINT_SPEC_PATH)
    assert spec["events"]==reference["events"] and spec["optimization"]==reference["optimization"]
    assert joint_training.settings(spec)==(pc.ROOT,pc.MODE,pc.JOINT)
    assert spec["execution"]["checkpoints"]==[90,180,270,360,450]
    assert spec["joint"]["loss_variant"]=="full"
    count=0
    for task in scope.registration()["global_task_ids"]:
        for episode in planned_episodes(scope.selection(),task):
            condition={"global_task_id":task,"condition_id":episode["condition_id"]}
            selected=pc.passive_condition(condition,pc.SEEN_MODE)
            assert selected==(task in (0,12,20,32) and episode["init_state_id"] in (32,33))
            count+=selected
    assert count==8
    cp=pc.ROOT/pc.MODE/"train/attempts/fresh/checkpoints/macro_00000270"
    assert joint_readout.bank_path(pc.MODE,cp)==pc.ROOT/pc.MODE/"banks/270/manifest.json"
    assert joint_readout._seen_geometry(pc.SEEN_MODE)
    assert joint_readout.PUBLIC_SCENES.name=="scenes" and "scene_canonical144" in str(joint_readout.PUBLIC_SCENES)
