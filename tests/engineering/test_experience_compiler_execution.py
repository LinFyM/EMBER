"""Native ten-step evidence and independent budget/success-driven slots."""
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch import nn

from ember.experience_compiler import execution, interaction
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


def test_optional_capture_keeps_native_ten_flow_calls_and_selects_actual_slots():
    velocity = Velocity()
    noise = torch.randn(3, 50, 32)
    actions, hidden = execution.action_chunk(velocity, noise, capture_indices=[0, 2])
    assert len(velocity.inputs) == 10 and hidden.shape == (2, 2, 50, 1024)
    torch.testing.assert_close(hidden[:, 0], velocity.inputs[0][[0, 2]].to(torch.bfloat16))
    torch.testing.assert_close(hidden[:, 1], velocity.inputs[-1][[0, 2]].to(torch.bfloat16))
    velocity.inputs.clear()
    fast, absent = execution.action_chunk(velocity, noise, capture_hidden=False)
    assert absent is None and len(velocity.inputs) == 10
    torch.testing.assert_close(actions, fast)
    assert not velocity.policy.model.action_out_proj._forward_pre_hooks


def test_native_prefix_keeps_full_batch_when_only_practice_phi_is_copied(monkeypatch):
    embeddings, padding = torch.randn(3, 520, 8), torch.ones(3, 520, dtype=torch.bool)
    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=32),
        model=SimpleNamespace(embed_prefix=lambda *_: (embeddings, padding, None)),
        _preprocess_images=lambda batch: ([], []))
    calls = []
    monkeypatch.setattr(execution, 'prepare_prefix_kv_cache', lambda policy, prefix: calls.append(prefix) or prefix)
    batch = {'observation.language.tokens': None, 'observation.language.attention_mask': None}
    practice = execution.NativeVelocity(policy, batch, capture_indices=[1])
    final = execution.NativeVelocity(policy, batch, capture_phi=False)
    assert practice.phi.shape == (1, 512, 8) and final.phi is None
    assert all(call.embeddings is embeddings for call in calls)


class FakeSlots:
    """CPU bookkeeping only; each slot has its own state and action counter."""
    def __init__(self, contract, physical_gpu, count):
        self.pending, self.messages, self.slots, self.closed = {}, {}, {}, False

    def submit(self, index, kind, **value):
        assert index not in self.pending
        self.pending[index], self.messages[index] = kind, value

    def receive(self, block=False):
        result = []
        for index in list(self.pending):
            kind, value = self.pending.pop(index), self.messages.pop(index)
            if kind == 'start':
                settle = min(10, value.get('remaining') or 10)
                limit = value.get('remaining')
                limit = 15 if limit is None else min(15, limit - settle)
                self.slots[index] = dict(state_id=value['state_id'], root=value['noise_root'], task=value['task'],
                                         steps=0, limit=limit, settle=settle, seeds=[])
                row = dict(kind=kind, settling_steps=settle, state_id=value['state_id'], controls=limit)
            else:
                slot = self.slots[index]
                n = min(5, slot['limit'] - slot['steps'])
                # Task0 succeeds in its first actual action group; task1 fails.
                slot['steps'] += n
                slot['seeds'].append(value['noise_seed'])
                row = dict(kind=kind, executed=np.ones((n, 7), dtype=np.float32),
                           steps=slot['steps'], done=slot['task']['task_id'] == 0, reward=0.)
            slot = self.slots[index]
            done = bool(row.get('done', False))
            ended = done or slot['steps'] >= slot['limit']
            row.update(raw={'images': torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
                            'proprio': torch.full((8,), float(index))}, operation_seconds=0., episode_ended=ended)
            if ended:
                row['row'] = dict(success=done, environment_steps=slot['settle'] + slot['steps'],
                    steps=slot['steps'], init_state_id=slot['state_id'], policy_noise_seeds=slot['seeds'],
                    suite=slot['task']['suite'], task_id=slot['task']['task_id'])
            result.append((index, row))
        return result

    def close(self):
        assert not self.pending
        self.closed = True


def fake_runtime(monkeypatch):
    batches, edits, incoming_states = [], [], []
    @contextmanager
    def activate(states):
        incoming_states.append([float(s['factor']) for s in states])
        yield
    processor = SimpleNamespace(unnormalize_action=lambda x: x)
    class Velocity:
        def __init__(self, policy, batch, capture_phi, capture_indices):
            batches.append(len(batch['input']))
            self.phi = torch.zeros(len(capture_indices or []), 512, 4) if capture_phi else None
    def chunk(velocity, noise, capture_hidden, capture_indices):
        return torch.ones(len(noise), 50, 7), (torch.zeros(len(capture_indices), 2, 50, 1024)
                                              if capture_hidden else None)
    runtime = SimpleNamespace(device=torch.device('cpu'), policy=None, mt={'factor': torch.tensor(10.)},
        processor=processor, execution=SimpleNamespace(activate=activate),
        io=SimpleNamespace(submit=lambda *args, **kw: None), features=SimpleNamespace(put_many=lambda *args: None),
        observation_features=lambda raw: {k: torch.zeros(512, 4) for k in raw},
        teacher=lambda *args: {'indices': torch.tensor([0, 5, 9])}, last_teacher_cost={})
    def edit(incoming, teacher, evidence):
        edits.append((float(incoming['factor']), len(evidence.get('feedback', [])),
                      evidence.get('feedback', torch.empty(0, 4)).clone()))
        return {'factor': incoming['factor'] + 1}
    runtime.edit = edit
    monkeypatch.setattr(interaction, 'processed', lambda *args: {'input': torch.ones(1, 1)})
    monkeypatch.setattr(interaction, 'EnvironmentSlots', FakeSlots)
    monkeypatch.setattr(interaction, 'NativeVelocity', Velocity)
    monkeypatch.setattr(interaction, 'action_chunk', chunk)
    return runtime, batches, edits, incoming_states


def request(task_id, kind='adapt', budget=35):
    task = dict(suite='libero_spatial', task_id=task_id, language=f'exact {task_id}')
    c = dict(task_id=task_id, teacher_demo=0, condition_id=f'c{task_id}', seed=42 + task_id,
             excluded_states=[32, 33, 34])
    return dict(kind=kind, condition=c, task=task, behavior_version='profile', step_budget=budget)


def test_success_is_edited_before_freeze_and_budget_counts_settling_per_slot(monkeypatch):
    runtime, batches, edits, states = fake_runtime(monkeypatch)
    runner = interaction.Runner(runtime, {}, 0, slot_batch=2)
    results = list(runner.run([request(0), request(1)]))
    try:
        by_task = {x['request']['condition']['task_id']: x['chain'] for x in results}
        success, failed = by_task[0], by_task[1]
        assert success.metrics['actual_J'] == 1 and success.states[-1]['factor'] == 11
        assert success.metrics['environment_steps'] == 15 and edits[0][2][-1, 1] == 1
        # Task1: 10 settling+15 control, then10 settling-only, no invented edit.
        assert failed.metrics['environment_steps'] == 35 and failed.metrics['actual_J'] == 1
        assert failed.endpoints == [3] and len(failed.episodes) == 2
        assert runner.total_environment_steps == 50 and 2 in batches
        assert states[0] == [10., 10.] and failed.states[-1]['factor'] == 11
        assert not set(e['init_state_id'] for e in failed.episodes) & {32, 33, 34}
    finally:
        runner.close()


def test_final_different_loras_keep_independent_noise_and_skip_all_fact_reading(monkeypatch):
    runtime, batches, edits, states = fake_runtime(monkeypatch)
    runner = interaction.Runner(runtime, {}, 0, slot_batch=2)
    requests = [dict(kind='final', task=request(i)['task'], state_id=32+i,
                     state={'factor': torch.tensor(20.+i)}, noise_root=7) for i in (0, 1)]
    rows = [x['row'] for x in runner.run(requests)]
    try:
        assert not edits and states[0] == [20., 21.] and runner.total_environment_steps == 40
        for row in rows:
            assert row['policy_noise_seeds'] == [policy_noise_seed(7, row['suite'], row['task_id'],
                row['init_state_id'], index) for index in range(len(row['policy_noise_seeds']))]
    finally:
        runner.close()


def test_saved_slot_hidden_does_not_serialize_other_conditions(monkeypatch, tmp_path):
    from ember.experience_compiler.storage import save_condition

    runtime, batches, edits, states = fake_runtime(monkeypatch)
    captured = []
    def chunk(velocity, noise, capture_hidden, capture_indices):
        hidden = torch.stack([torch.full((2, 50, 1024), float(i + 1), dtype=torch.bfloat16)
                              for i in range(len(capture_indices))])
        captured.append(hidden)
        return torch.ones(len(noise), 50, 7), hidden
    monkeypatch.setattr(interaction, 'action_chunk', chunk)
    runner = interaction.Runner(runtime, {}, 0, slot_batch=2)
    try:
        results = list(runner.run([request(0), request(1)]))
        first = next(r for r in results if r['request']['condition']['task_id'] == 0)
        chain = first['chain']
        save_condition(tmp_path / 'condition', first['request']['condition'], chain)
        saved = torch.load(tmp_path / 'condition/experience.pt', map_location='cpu', weights_only=False)
        original, restored = captured[0][0], saved['records'][0]['hidden']
        assert captured[0].shape == (2, 2, 50, 1024)
        assert original.untyped_storage().nbytes() == 2 * original.nbytes
        assert restored.shape == original.shape and restored.dtype == original.dtype
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        assert restored.untyped_storage().nbytes() == restored.nbytes
        assert (tmp_path / 'condition/experience.pt').stat().st_size < original.nbytes + 20000
    finally:
        runner.close()


def test_null_replay_starts_actual_MT_and_never_uses_real_intermediate_parameters():
    from ember.experience_compiler.runtime import Runtime
    runtime = Runtime.__new__(Runtime)
    runtime.mt = {'a': torch.tensor(2.)}
    seen = []
    def edit(state, teacher, evidence):
        seen.append(float(state['a']))
        assert evidence == {}
        return {'a': state['a'] + 3}
    runtime.edit = edit
    chain = interaction.Chain(states=[{'a': torch.tensor(99.)}], endpoints=[5, 10, 11])
    assert Runtime.null_replay(runtime, {}, chain)['a'] == 11 and seen == [2, 5, 8]
