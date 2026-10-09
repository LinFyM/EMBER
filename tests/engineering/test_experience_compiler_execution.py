"""Native capture must preserve actions/credit and retain complete practice facts."""
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from ember.experience_compiler import execution
from ember.experience_compiler import interaction
from ember.pi05_eval_contract import policy_noise_seed


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = SimpleNamespace(model=SimpleNamespace(action_out_proj=nn.Linear(1024, 32)))
        self.inputs = []

    def forward(self, z, tau):
        hidden = torch.nn.functional.pad(z, (0, 992)) + tau
        self.inputs.append(hidden.clone())
        return self.policy.model.action_out_proj(hidden)


@pytest.mark.parametrize("sde_seed", [None, 91])
def test_optional_capture_preserves_ten_flow_calls_actions_and_sde_credit(sde_seed):
    velocity = Velocity()
    noise = torch.randn(2, 50, 32)
    with torch.no_grad():
        actions, hidden, path = execution.action_chunk(velocity, noise, sde_seed=sde_seed)
    assert len(velocity.inputs) == 10 and hidden.shape == (2, 2, 50, 1024)
    torch.testing.assert_close(hidden[:, 0], velocity.inputs[0].to(torch.bfloat16))
    torch.testing.assert_close(hidden[:, 1], velocity.inputs[-1].to(torch.bfloat16))
    velocity.inputs.clear()
    fast_actions, absent, fast_path = execution.action_chunk(velocity, noise,
        sde_seed=sde_seed, capture_hidden=False)
    assert absent is None and len(velocity.inputs) == 10
    torch.testing.assert_close(actions, fast_actions, rtol=0, atol=0)
    assert len(path) == len(fast_path) == (0 if sde_seed is None else 10)
    for original, fast in zip(path, fast_path):
        assert original.step == fast.step and original.tau == fast.tau
        torch.testing.assert_close(execution.score_cotangent(original, 1, replans=10, retained=10),
                                   execution.score_cotangent(fast, 1, replans=10, retained=10), rtol=0, atol=0)
    assert not velocity.policy.model.action_out_proj._forward_pre_hooks


def test_prefix_computation_is_kept_when_unused_phi_is_not_copied(monkeypatch):
    embeddings = torch.randn(1, 520, 8)
    padding = torch.ones(1, 520, dtype=torch.bool)
    model = SimpleNamespace(embed_prefix=lambda *_: (embeddings, padding, None))
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=32), model=model,
                             _preprocess_images=lambda batch: ([], []))
    calls = []
    def prepare(policy, prefix):
        calls.append(prefix)
        return prefix
    monkeypatch.setattr(execution, "prepare_prefix_kv_cache", prepare)
    batch = {'observation.language.tokens': None, 'observation.language.attention_mask': None}
    practice = execution.NativeVelocity(policy, batch)
    final = execution.NativeVelocity(policy, batch, capture_phi=False)
    assert practice.phi.shape == (1, 512, 8) and final.phi is None and len(calls) == 2
    assert calls[0].embeddings is embeddings and calls[1].embeddings is embeddings


def test_final_batch_reuses_adapter_until_size_changes_and_keeps_slot_rng(monkeypatch):
    activations, calls = [], []
    @contextmanager
    def activate(states):
        activations.append(len(states))
        yield
    class Environment:
        def __init__(self, stop):
            self.stop, self.steps = stop, 0
        def step(self, action):
            self.steps += 1
            return {}, 0., self.steps == self.stop, {}
    envs = [Environment(5), Environment(100), Environment(11)]
    class Processor:
        def __call__(self, value):
            return {'input': torch.zeros(1, 2)}
        def unnormalize_action(self, value):
            return value
    runner = object.__new__(interaction.Runner)
    runner.runtime = SimpleNamespace(device=torch.device('cpu'), policy=None, processor=Processor(),
                                     execution=SimpleNamespace(activate=activate))
    runner.pool = SimpleNamespace(switch=lambda task: (envs, None))
    runner.contract = {'environment': {'dummy_settling_steps': 10, 'dummy_action': [0.] * 7,
                                      'horizons': {'libero_spatial': 15}}}
    runner.total_environment_steps, runner.started = 0, 0
    def start(**kwargs):
        return dict(obs={}, steps=0, policy_noise_seeds=[], init_state_id=kwargs['init_state_id'], replan_index=0)
    def chunk(velocity, noise, *, capture_hidden):
        assert capture_hidden is False
        calls.append(len(noise))
        return torch.ones(len(noise), 50, 7), None, []
    monkeypatch.setattr(interaction, 'start_fixed_episode', start)
    monkeypatch.setattr(interaction, 'finish_episode_row', lambda **kwargs: dict(kwargs['slot']))
    monkeypatch.setattr(interaction, 'libero_policy_input', lambda *args: {})
    monkeypatch.setattr(interaction, 'NativeVelocity', lambda *args, capture_phi: None if not capture_phi else pytest.fail())
    monkeypatch.setattr(interaction, 'action_chunk', chunk)
    task = {'suite': 'libero_spatial', 'task_id': 0, 'language': 'exact task'}
    rows = runner.final_many(task, [32, 33, 34], {'complete_adapter': torch.zeros(1)}, noise_root=7)
    assert calls == [3, 2, 2] and activations == [3, 2]
    assert runner.total_environment_steps == 30 + 5 + 15 + 11
    for row in rows:
        assert row['policy_noise_seeds'] == [policy_noise_seed(7, 'libero_spatial', 0,
            row['init_state_id'], index) for index in range(row['replan_index'])]
