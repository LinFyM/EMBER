"""CPU oracles for the bounded shared-address operator, without loading PI05."""

import torch

from ember.lora import identity_lora_state
from ember.operator_writer.model import TargetWrite
from ember.operator_writer.native import _capture_inputs
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract


def test_zero_change_cannot_write_and_later_credit_reaches_shared_address():
    torch.manual_seed(23)
    unit = TargetWrite(4, 3)
    address = torch.randn(128, 4, requires_grad=True)
    feature = torch.randn(3, 50, 4, requires_grad=True)
    hidden = torch.randn(3, 50, 1024, requires_grad=True)
    assert torch.count_nonzero(unit(address, feature, hidden)) == 0
    first_loss = unit(address, feature, hidden).square().sum() + unit(address, feature, hidden).sum()
    first_loss.backward()
    assert unit.o.weight.grad is not None and unit.o.weight.grad.norm() > 0
    assert address.grad is not None and torch.count_nonzero(address.grad) == 0
    assert all(torch.count_nonzero(layer.weight.grad) == 0
               for layer in (unit.p, unit.c, unit.d))

    with torch.no_grad():
        unit.o.weight.fill_(0.01)
    unit.zero_grad(set_to_none=True)
    address.grad = feature.grad = hidden.grad = None
    written = unit(address, feature, hidden)
    common_b = torch.randn_like(written, requires_grad=True)
    final_b = common_b + written
    query = torch.randn(4)
    output = final_b @ (address @ query)
    assert torch.allclose(output, (common_b @ address + written @ address) @ query)
    output.square().sum().backward()
    assert all(value.grad is not None and value.grad.norm() > 0
               for value in (common_b, address, feature, hidden))
    assert all(layer.weight.grad is not None and layer.weight.grad.norm() > 0
               for layer in (unit.p, unit.c, unit.d, unit.o))

    constant = hidden.detach()[0:1].expand(3, -1, -1)
    assert torch.count_nonzero(unit(address.detach(), feature.detach(), constant)) == 0


def test_scoped_actual_projection_capture_and_functional_weight_gradient():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source = torch.nn.Linear(4, 4, bias=False)
            self.projection = torch.nn.Linear(4, 3, bias=False)

        def forward(self, value):
            return self.projection(self.source(value))

    owner = Tiny()
    for parameter in owner.parameters():
        parameter.requires_grad_(False)
    input_value = torch.randn(2, 4, requires_grad=True)
    substituted = torch.randn(4, 4, requires_grad=True)
    with _capture_inputs(owner, ("projection",)) as captured:
        output = torch.func.functional_call(
            owner, {"source.weight": substituted}, (input_value,), strict=False)
    assert captured["projection"].shape == (2, 4)
    (output.square().sum() + captured["projection"].square().sum()).backward()
    assert substituted.grad is not None and substituted.grad.norm() > 0
    assert input_value.grad is not None and input_value.grad.norm() > 0
    assert owner.source.weight.grad is None
    assert owner.projection.weight.grad is None
    assert not owner.projection._forward_pre_hooks


def test_T_U_share_exact_initial_public_function_but_U_has_independent_key():
    from pathlib import Path

    from ember.operator_writer.model import OperatorReadWrite

    repo = Path(__file__).resolve().parents[2]
    contract = derive_pi05_lora_rank(
        load_pi05_lora_contract(repo / "configs/pi05_lora_v1.json"), rank=128)
    template = identity_lora_state(contract)
    t = OperatorReadWrite(contract, template, "T")
    u = OperatorReadWrite(contract, template, "U")
    assert t.separate_keys is None and len(u.separate_keys) == 38
    assert all(torch.equal(t.public_state()[key], u.public_state()[key]) for key in template)
    assert all(torch.equal(t.writes[i].state_dict()[key], u.writes[i].state_dict()[key])
               for i in range(38) for key in t.writes[i].state_dict())
    for i, name in enumerate(u.names):
        assert torch.equal(u.separate_keys[i], template[name + ".lora_A.default.weight"])
        assert u.separate_keys[i].data_ptr() != u.public_state()[name + ".lora_A.default.weight"].data_ptr()
