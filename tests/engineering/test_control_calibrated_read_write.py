"""Control consumption, credit, short-gap and feature-wall regression oracles."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from ember.lora import LoRATarget, identity_lora_state
from ember.operator_writer import control_calibration as control
from ember.operator_writer.bare_native import validate
from ember.operator_writer.conditional_read_write import ConditionalTarget
from ember.operator_writer.model import OperatorReadWrite
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract


def bare():
    return {"H0": torch.randn(3, 50, 1024), "mu0": torch.randn(3, 5, 7),
            "frame_indices": torch.tensor([0, 5, 8])}


def test_fresh_old_initialization_and_independent_Gamma_identity():
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(repo / "configs/pi05_lora_v1.json"), rank=128)
    contract = replace(contract, targets=tuple(LoRATarget(t.name, 4, 3) for t in contract.targets))
    template = identity_lora_state(contract)
    original = OperatorReadWrite(contract, template, "conditional_read_write")
    writer = OperatorReadWrite(contract, template, control.MODE)
    # One aggregate reconstruction of the complete inherited initialization.
    old = original.state_dict()
    assert sum((writer.state_dict()[k] - v).float().square().sum() for k, v in old.items()) == 0
    assert len(writer.conditional_targets) == 38
    assert all(unit.ua.bias is None and unit.ub.bias is None and not unit.ua.weight.any()
               and not unit.ub.weight.any() for unit in writer.conditional_targets)
    features = bare()
    q, valid = writer.gamma.controls(features, features["frame_indices"])
    torch.testing.assert_close(q[0], features["mu0"][0])
    assert valid.tolist() == [True, False] and not q[1].any()
    target = torch.zeros(1, 5, 7)
    loss = control.auxiliary_loss(q, valid, target)
    loss.backward()
    assert writer.gamma.w2.weight.grad.norm() > 0
    assert all(p.grad is None for n, p in writer.named_parameters() if not n.startswith("gamma."))


def test_q_consumption_zero_dynamic_and_main_plus_aux_replay():
    torch.manual_seed(71)
    gamma = control.ControlCalibration()
    unit = ConditionalTarget(4, 3)
    unit.ua, unit.ub = nn.Linear(35, 256, bias=False), nn.Linear(35, 256, bias=False)
    with torch.no_grad():
        gamma.w2.weight.normal_(std=.01)
        unit.a_out.weight.normal_(std=.02)
        unit.b_out.weight.normal_(std=.02)
    a0 = torch.randn(128, 4, requires_grad=True)
    b0 = torch.randn(3, 128, requires_grad=True)
    features = bare()
    rawx, rawh = torch.randn(3, 50, 4), torch.randn(3, 50, 1024)
    target = torch.randn(1, 5, 7)
    ca, cb = torch.randn_like(a0), torch.randn_like(b0)

    def compile_pair():
        native = (b0 @ a0).square().mean().tanh()
        x, h = rawx + native * .1, rawh + native * .1
        c, d = h * .7, torch.cat((torch.zeros_like(h[:1]), h[1:] - h[:-1]))
        q, valid = gamma.controls(features, features["frame_indices"])
        a, b, s, m = unit(a0, b0, x, h, c, d, q.flatten(1))
        return a, b, control.auxiliary_loss(q, valid, target), (x, h, c, d, q)

    params = (gamma.w2.weight, unit.ua.weight, unit.ub.weight, a0, b0)
    a, b, aux, _ = compile_pair()
    direct = torch.autograd.grad((a * ca).sum() + (b * cb).sum() + aux, params)
    a, b, aux, args = compile_pair()
    replay = torch.autograd.grad((a, b, aux), params, grad_outputs=(ca, cb, torch.ones_like(aux)))
    for actual, expected in zip(replay, direct, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        assert actual.norm() > 0
    x, h, c, d, q = args
    zero = unit(a0, b0, x, h, c, torch.zeros_like(d), q.flatten(1))
    assert not zero[2].any() and not zero[3].any()
    actual = unit(a0, b0, x, h, c, d, q.flatten(1))
    changed = unit(a0, b0, x, h, c, d, torch.zeros_like(q).flatten(1))
    assert (actual[0] - changed[0]).norm() > 0 and (actual[1] - changed[1]).norm() > 0


def test_action_free_cache_and_mean_loss_not_target_or_rank_repeated():
    f = bare()
    value = {**f, "schema": "ember_bare_native_control_features_v1", "source": {}, "reading_git": {}}
    validate(value, f["frame_indices"])
    with pytest.raises(ValueError):
        validate({**value, "target": torch.zeros(2, 5, 7)}, f["frame_indices"])
    gamma = control.ControlCalibration()
    q, valid = gamma.controls(f, f["frame_indices"])
    target = torch.zeros(1, 5, 7)
    one = control.auxiliary_loss(q, valid, target)
    duplicated = control.auxiliary_loss(torch.cat((q, q)), torch.cat((valid, valid)), target.repeat(2, 1, 1))
    torch.testing.assert_close(one, duplicated)
    torch.testing.assert_close(one, q[0].square().mean() / 4)
