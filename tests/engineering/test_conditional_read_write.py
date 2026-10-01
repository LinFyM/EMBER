"""§13 causal/zero-preserving computation and full two-sided credit oracles."""
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget, identity_lora_state
from ember.operator_writer.conditional_read_write import CausalInterpreter, ConditionalTarget, delta_memory
from ember.operator_writer.model import OperatorReadWrite
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract


def test_causal_frame_blocks_and_zero_dynamic_propagation():
    torch.set_num_threads(6)
    torch.manual_seed(19)
    model = CausalInterpreter().eval()
    h = torch.randn(3, 50, 1024)
    with torch.no_grad():
        c, d = model(h, torch.tensor([0, 5, 11]))
        future = h.clone(); future[-1].normal_(mean=2, std=3)
        c2, d2 = model(future, torch.tensor([0, 5, 11]))
        torch.testing.assert_close(c[:2], c2[:2], rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(d[:2], d2[:2], rtol=3e-5, atol=3e-6)
        same = h[:1].expand(3, -1, -1)
        _, zero = model(same, torch.tensor([0, 5, 11]))
        assert torch.count_nonzero(zero) == 0
        h[2] = h[1]
        _, transferred = model(h, torch.tensor([0, 5, 11]))
        assert transferred[2].norm() > 0  # Past dynamics reach a locally static frame.


def test_target_origin_address_final_A_and_complete_composite_credit():
    torch.manual_seed(13)
    unit = ConditionalTarget(4, 3).double()
    with torch.no_grad():
        unit.a_out.weight.normal_(std=.03); unit.b_out.weight.normal_(std=.03)
    a0 = torch.randn(128, 4, dtype=torch.float64, requires_grad=True)
    b0 = torch.randn(3, 128, dtype=torch.float64, requires_grad=True)
    rawx = torch.randn(3, 50, 4, dtype=torch.float64)
    rawh = torch.randn(3, 50, 1024, dtype=torch.float64)
    # A toy native dependency makes β's indirect and direct paths independently testable.
    def compile_state():
        native = (b0 @ a0).square().mean().tanh()
        x = rawx + .1 * native
        h = rawh + .1 * native
        c, d = h * .7, torch.cat((torch.zeros_like(h[:1]), h[1:] - h[:-1]))
        # Production computation permits BF16/FP32. Exercise its FP32 memory directly.
        return checkpoint(unit, a0.float(), b0.float(), x.float(), h.float(), c.float(), d.float(),
                          use_reentrant=False, preserve_rng_state=False)
    unit = unit.float()
    a, b, s, m = compile_state()
    outputs = (a, b)
    ca, cb = torch.randn_like(a), torch.randn_like(b)
    objective = (a * ca).sum() + (b * cb).sum()
    parameters = (a0, b0, unit.a_out.weight, unit.b_out.weight)
    direct = torch.autograd.grad(objective, parameters)
    replay = compile_state()
    vjp = torch.autograd.grad(replay[:2], parameters, grad_outputs=(ca, cb))
    for actual, wanted in zip(vjp, direct, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=3e-6)
        assert actual.norm() > 0
    native = (b0 @ a0).square().mean().tanh()
    x, h = (rawx + .1 * native).float(), (rawh + .1 * native).float()
    c, d = h * .7, torch.cat((torch.zeros_like(h[:1]), h[1:] - h[:-1]))
    xi = F.normalize(x[:-1], dim=-1, eps=1e-6)
    context = torch.cat((c[1:], h[1:]), -1)
    va = unit.a_out(F.gelu(unit.a_x(xi) + unit.a_z(F.linear(xi, a0.float()))
                           + unit.a_context(context)) * unit.a_dynamic(d[1:]))
    oracle_s = delta_memory(va, xi, 128)
    oracle_a = a0.float() + oracle_s
    key = F.normalize(F.linear(x[:-1], oracle_a), dim=-1, eps=1e-6)
    vb = unit.b_out(F.gelu(unit.b_key(key) + unit.b_delta(F.linear(x[:-1], oracle_s))
                           + unit.b_context(context)) * unit.b_dynamic(d[1:]))
    oracle_m = delta_memory(vb, key, 3)
    torch.testing.assert_close(s, oracle_s); torch.testing.assert_close(m, oracle_m)
    indirect = torch.autograd.grad((oracle_m * cb).sum(), oracle_s, retain_graph=True)[0]
    assert indirect.norm() > 0  # Final A affects M through key and raw delta-z.
    zero = unit(a0.float(), b0.float(), x, h, c, torch.zeros_like(d))
    assert torch.count_nonzero(zero[2]) == torch.count_nonzero(zero[3]) == 0


def test_actual_38_target_identity_and_single_complete_pair(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    contract = derive_pi05_lora_rank(load_pi05_lora_contract(root / 'configs/pi05_lora_v1.json'), rank=128)
    small = replace(contract, targets=tuple(LoRATarget(t.name, 4, 3) for t in contract.targets))
    writer = OperatorReadWrite(small, identity_lora_state(small), 'conditional_read_write').eval()
    common = writer.public_state()
    from ember.operator_writer import run
    x = {name: torch.randn(2, 50, 4) for name in writer.names}
    h = torch.randn(2, 50, 1024)
    monkeypatch.setattr(run, 'read_native_video', lambda *args, **kwargs: (x, h))
    monkeypatch.setattr(run.Runtime, 'restore_identity', lambda self: None)
    runtime = run.Runtime(None, writer, None, None, small, {}, torch.device('cpu'), {})
    with torch.no_grad():
        result, evidence = runtime.compile((None, [0, 7], None, None), capture_mechanism=True)
    assert len(evidence['passes']) == 1 and evidence['mechanism']['c'].shape == h.shape
    with torch.no_grad():
        result = writer({name: torch.randn(2, 50, 4) for name in writer.names},
                        torch.randn(2, 50, 1024), [0, 7], capture_mechanism=True)
    assert len(result) == 76 and len(writer.last_mechanism['targets']) == 38
    for name in writer.names:
        torch.testing.assert_close(result[name + LORA_A_SUFFIX], common[name + LORA_A_SUFFIX])
        assert torch.count_nonzero(result[name + LORA_B_SUFFIX]) == 0
    assert writer.value_context is None and len(writer.writes) == 0
