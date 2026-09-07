"""Analytic full-integrator VJP oracle, independent of large policy assets."""

from types import SimpleNamespace

import torch
from torch import nn

from ember.lora import LoRATarget, SmolVLALoRAContract, task_lora_state_dict
from ember.writer.flow import flow_actions, flow_mean_lora_gradient
from ember.writer.functional import prepare_frozen_writer_policy


class LinearFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(32) * 0.2)

    def denoise_step(self, padding, cache, x, time):
        # Cross-horizon dependency makes early cropping to 5 steps detectable.
        return self.proj(x) + 0.03 * x.mean(1, keepdim=True)


def fixture(monkeypatch):
    policy = nn.Module()
    policy.model = LinearFlow()
    contract = SmolVLALoRAContract((LoRATarget("model.proj", 32, 32),), 2, 2, 0.0, 19)
    prepare_frozen_writer_policy(policy, contract)
    state = {name: value.double().requires_grad_(True) for name, value in task_lora_state_dict(policy, clone=True).items()}
    with torch.no_grad():
        for value in state.values():
            value.uniform_(-0.1, 0.1)
    policy.double()
    monkeypatch.setattr("ember.writer.flow.prepare_execution_policy_prefix", lambda *args: SimpleNamespace(padding=None))
    monkeypatch.setattr("ember.writer.flow.prepare_prefix_kv_cache", lambda *args: None)
    return policy, state, contract


def test_all_ten_flow_steps_and_horizon_remain_connected(monkeypatch):
    torch.manual_seed(4)
    policy, state, contract = fixture(monkeypatch)
    noise = torch.randn(2, 50, 32, dtype=torch.float64)
    score = torch.randn(2, 35, dtype=torch.float64)
    a, b = state.values()
    full_weight = policy.model.proj.base_layer.weight + b @ a
    oracle = noise
    for _ in range(10):
        oracle = oracle - 0.1 * (oracle @ full_weight.T + 0.03 * oracle.mean(1, keepdim=True))
    expected = torch.autograd.grad(oracle[:, :5, :7], tuple(state.values()),
                                   grad_outputs=score.reshape(2, 5, 7))
    actual = flow_actions(policy, state, contract, {}, noise)
    torch.testing.assert_close(actual, oracle)
    gradients = flow_mean_lora_gradient(policy, state, contract, {}, noise, score)
    for observed, reference in zip(gradients.values(), expected, strict=True):
        torch.testing.assert_close(observed, reference)
    with torch.no_grad():
        inference = flow_actions(policy, state, contract, {}, noise)
    torch.testing.assert_close(inference, actual)
    assert all(parameter.grad is None for parameter in policy.parameters())
    # A last-step-only derivative is a materially different estimator.
    prefix = noise
    with torch.no_grad():
        for _ in range(9):
            prefix = prefix - 0.1 * (prefix @ full_weight.T + 0.03 * prefix.mean(1, keepdim=True))
    last = prefix - 0.1 * (prefix @ (policy.model.proj.base_layer.weight + b @ a).T + 0.03 * prefix.mean(1, keepdim=True))
    truncated = torch.autograd.grad(last[:, :5, :7], tuple(state.values()), grad_outputs=score.reshape(2, 5, 7))
    assert sum(float((x - y).norm()) for x, y in zip(expected, truncated, strict=True)) > 1


def test_checkpoint_replay_uses_functional_adapter_not_physical_identity(monkeypatch):
    policy, state, contract = fixture(monkeypatch)
    noise = torch.randn(1, 50, 32, dtype=torch.float64)
    results = []
    for checkpointed in (False, True):
        output = flow_actions(policy, state, contract, {}, noise, checkpoint_steps=checkpointed)
        results.append(torch.autograd.grad(output.square().mean(), tuple(state.values())))
    for dense, replay in zip(*results, strict=True):
        torch.testing.assert_close(dense, replay)
