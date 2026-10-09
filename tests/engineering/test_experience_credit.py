"""Paired edit pressure and complete-factor VJPs, independent of chunks/ranks."""
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace

import pytest
import torch

from ember.experience_compiler import credit
from ember.batched_lora import BatchedLoRAInference
from ember.lora import LoRATarget, inject_task_lora, task_lora_state_dict
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.function_credit import FlowSample


def test_edit_pressure_stops_incoming_and_uses_global_four_event_weight():
    incoming = torch.tensor([1., 2., 4.], requires_grad=True)
    outgoing = torch.tensor([2., 1., 4.], requires_grad=True)
    objective, regression = credit.edit_objective(outgoing, incoming)
    assert objective.item() == pytest.approx((7 / 3 + .2 / 3) / 4)
    objective.backward()
    torch.testing.assert_close(regression, torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(outgoing.grad, torch.tensor([1.2, 1., 1.]) / 12)
    assert incoming.grad is None


def test_heterogeneous_native_credit_keeps_queries_noise_prefix_and_complete_vjp(monkeypatch):
    torch.manual_seed(17)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(5, 7, bias=False)
        def forward(self, x):
            return torch.nn.functional.pad(self.proj(x), (0, 25))
    contract = replace(load_pi05_lora_contract(Path(__file__).resolve().parents[2] / 'configs/pi05_lora_v1.json'),
        targets=(LoRATarget('proj', 5, 7),), rank=2, alpha=2, dropout=0., identity_seed=7)
    policy = inject_task_lora(Policy(), contract)
    identity = task_lora_state_dict(policy, clone=True)
    incoming = [{k: torch.randn_like(v) for k, v in identity.items()} for _ in range(2)]
    outgoing = [{k: torch.randn_like(v) for k, v in identity.items()} for _ in range(2)]
    batches = [dict(action=torch.zeros(28, 50, 7), x=torch.randn(28, 50, 5),
                    target=torch.randn(28, 50, 32)) for _ in range(2)]
    prepared, draws = [], []
    class Owner(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.policy = p
        def prepare(self, sample):
            marker = object()
            prepared.append((len(sample.target), marker))
            return marker
        def forward(self, sample, cache):
            assert cache is prepared[-1][1]
            return self.policy(sample.arguments[0])
    def sample(p, batch, **kwargs):
        draws.append(kwargs)
        return FlowSample((batch['x'],), batch['target'], 7)
    monkeypatch.setattr(credit, 'NativeFlowPrediction', Owner)
    monkeypatch.setattr(credit, 'flow_sample', sample)
    execution = BatchedLoRAInference(policy, contract)
    runtime = SimpleNamespace(policy=policy, execution=execution, device=torch.device('cpu'))
    full = credit.fm_credit(runtime, incoming, outgoing, batches, seeds=[71, 93], microbatch=56)
    assert [size for size, _ in prepared] == [56]
    assert [(d['seed'], d['random_batch'], d['offset']) for d in draws] == [(71, 28, 0), (93, 28, 0)]
    for microbatch in (7, 14, 28):
        chunked = credit.fm_credit(runtime, incoming, outgoing, batches, seeds=[71, 93], microbatch=microbatch)
        for expected, actual in zip(full, chunked):
            assert expected['weighted_loss'] == pytest.approx(actual['weighted_loss'], rel=2e-6)
            for key in expected['cotangent']:
                torch.testing.assert_close(expected['cotangent'][key], actual['cotangent'][key])
    for index in range(2):
        leaves = {k: v.clone().requires_grad_() for k, v in outgoing[index].items()}
        before = torch.func.functional_call(policy, incoming[index], (batches[index]['x'],), strict=False)
        after = torch.func.functional_call(policy, leaves, (batches[index]['x'],), strict=False)
        ell_in = (before[..., :7] - batches[index]['target'][..., :7]).square().mean((1, 2))
        ell_out = (after[..., :7] - batches[index]['target'][..., :7]).square().mean((1, 2))
        objective, _ = credit.edit_objective(ell_out, ell_in)
        expected = torch.autograd.grad(objective, tuple(leaves.values()))
        for key, gradient in zip(leaves, expected):
            torch.testing.assert_close(full[index]['cotangent'][key], gradient)
    execution.close()
