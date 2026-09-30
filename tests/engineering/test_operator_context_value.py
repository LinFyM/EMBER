"""CPU oracles for §36's literal temporal Value extension and original-T inclusion."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget, identity_lora_state
from ember.operator_writer.model import OperatorReadWrite, TargetWrite
from ember.operator_writer.value_context import ValueContext
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract


def small_writer(mode):
    path = Path(__file__).resolve().parents[2] / "configs/pi05_lora_v1.json"
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(path), rank=128)
    contract = replace(contract, targets=tuple(LoRATarget(target.name, 4, 3) for target in contract.targets))
    return OperatorReadWrite(contract, identity_lora_state(contract), mode)


def test_parameter_increment_initialization_rng_and_zero_U_original_T_inclusion():
    torch.manual_seed(31)
    previous = small_writer("T")
    before = torch.random.get_rng_state()
    candidate = small_writer("context")
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in previous.parameters()) == 3014656
    assert candidate.separate_keys is None
    assert all(write.u is not None and write.u.bias is None and write.u.weight.count_nonzero() == 0
               for write in candidate.writes)
    assert all(layer.bias is None for layer in (candidate.value_context.wh, candidate.value_context.q,
               candidate.value_context.k, candidate.value_context.v, candidate.value_context.out))
    # Representative beginning/end weights protect the original initialization sequence.
    for index in (0, 37):
        for name in ("p", "c", "d", "o"):
            torch.testing.assert_close(getattr(candidate.writes[index], name).weight,
                                       getattr(previous.writes[index], name).weight)
    torch.testing.assert_close(candidate.probe, previous.probe)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected_wh = torch.nn.Linear(1024, 256, bias=False)
    torch.testing.assert_close(candidate.value_context.wh.weight, expected_wh.weight)
    with torch.no_grad():
        for write in previous.writes:
            write.o.weight.normal_(std=.02)
        previous.writes[0].c.weight.normal_(std=.02)
        previous.common.values[1].add_(.01)
    missing, unexpected = candidate.load_state_dict(previous.state_dict(), strict=False)
    assert len(missing) == 43 and not unexpected
    hidden = torch.randn(3, 50, 1024)
    inputs = {name: torch.randn(3, 50, 4) for name in previous.names}
    with torch.no_grad():
        expected, observed = previous(inputs, hidden), candidate(inputs, hidden, [0, 5, 16])
    for name in (previous.names[0], previous.names[-1]):
        torch.testing.assert_close(observed[name + LORA_B_SUFFIX], expected[name + LORA_B_SUFFIX], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(observed[name + LORA_A_SUFFIX], expected[name + LORA_A_SUFFIX])
    with pytest.raises(ValueError, match="frame indices"):
        candidate(inputs, hidden)


def literal_attention(module, hidden, indices):
    hbar = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + 1e-6)
    r = module.wh(hbar)
    def heads(layer):
        return layer(r).reshape(len(hidden), 50, 4, 64).permute(1, 2, 0, 3)
    q, k, v = heads(module.q), heads(module.k), heads(module.v)
    frequencies = 10000. ** (-torch.arange(32).float() / 32)
    phases = torch.as_tensor(indices).float()[:, None] / 5 * frequencies[None]
    rotations = torch.polar(torch.ones_like(phases), phases)[None, None]
    def rotate(value):
        paired = torch.complex(value[..., 0::2], value[..., 1::2]) * rotations
        return torch.view_as_real(paired).flatten(-2)
    attention = torch.softmax(rotate(q) @ rotate(k).transpose(-1, -2) / 8, dim=-1)
    return module.out((attention @ v).permute(2, 0, 1, 3).reshape(len(hidden), 50, 256))


def test_literal_real_position_rope_last_frame_and_independent_horizons():
    torch.manual_seed(17)
    module = ValueContext()
    hidden = torch.randn(4, 50, 1024)
    indices = [0, 5, 10, 16]  # Real last position is 3.2 rather than the ordinal 3.
    actual = module(hidden, indices)
    torch.testing.assert_close(actual, literal_attention(module, hidden, indices), rtol=2e-5, atol=2e-6)
    assert (actual - module(hidden, [0, 5, 10, 15])).norm() > 1e-5
    altered = hidden.clone()
    altered[-1, 17] += torch.randn(1024)
    changed = module(altered, indices)
    assert (changed[:-1, 17] - actual[:-1, 17]).norm() > 1e-4
    torch.testing.assert_close(changed[:, 18], actual[:, 18])
    torch.testing.assert_close(changed[:, 0], actual[:, 0])
    with pytest.raises(ValueError, match="integer"):
        module(hidden, [0, 5, 10, 15.5])
    with pytest.raises(ValueError, match="integer"):
        module(hidden, [0, 5, 5, 16])
    with pytest.raises(ValueError, match="actual N-frame"):
        module(hidden[None], indices)


def test_open_U_O_preserves_context_and_native_gradient_paths():
    torch.manual_seed(41)
    writer = small_writer("context")
    with torch.no_grad():
        writer.writes[0].o.weight.normal_(std=.03)
        writer.writes[0].u.weight.normal_(std=.05)
    hidden = torch.randn(4, 50, 1024, requires_grad=True)
    inputs = {name: torch.randn(4, 50, 4, requires_grad=True) for name in writer.names}
    contexts = []
    handle = writer.value_context.register_forward_hook(lambda module, arguments, result: contexts.append(result))
    try:
        state = writer(inputs, hidden, torch.tensor([0, 5, 10, 16]))
    finally:
        handle.remove()
    first = writer.names[0]
    loss = (state[first + LORA_B_SUFFIX] @ (state[first + LORA_A_SUFFIX] @ torch.randn(4))).square().sum()
    context_credit = torch.autograd.grad(loss, contexts[0], retain_graph=True)[0]
    native_context_credit = torch.autograd.grad(contexts[0], hidden, grad_outputs=context_credit, retain_graph=True)[0]
    assert torch.isfinite(context_credit).all() and context_credit.norm() > 0
    assert torch.isfinite(native_context_credit).all() and native_context_credit.norm() > 0
    loss.backward()
    for value in (hidden, inputs[first], writer.writes[0].u.weight, writer.writes[0].o.weight,
                  state[first + LORA_A_SUFFIX], writer.public_state()[first + LORA_B_SUFFIX]):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.norm() > 0
    for parameter in writer.value_context.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0


def test_no_virtual_terminal_write_and_zero_change_even_with_open_context():
    write = TargetWrite(4, 3)
    write.u = torch.nn.Linear(256, 256, bias=False)
    torch.nn.init.normal_(write.o.weight, std=.03)
    address, hidden = torch.randn(128, 4), torch.randn(1, 50, 1024)
    assert write(address, torch.randn(1, 50, 4), hidden, torch.randn(1, 50, 256)).count_nonzero() == 0
    assert write(address, torch.randn(3, 50, 4), hidden.expand(3, -1, -1),
                 torch.randn(3, 50, 256)).count_nonzero() == 0
