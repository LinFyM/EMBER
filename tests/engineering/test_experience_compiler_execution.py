"""Native ten-step evidence and independent budget/success-driven slots."""
from types import SimpleNamespace
from contextlib import contextmanager
import json
from safetensors.torch import save_file

import numpy as np
import pytest
import torch
from torch import nn

from ember.experience_compiler import execution, interaction, edit_readouts as readouts
from ember.writer import function_credit
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


def test_final_raw_capture_preserves_noise_commands_and_skips_facts(monkeypatch, tmp_path):
    """Capture adds evidence to the same final actor, including an early prefix."""
    runtime, batches, edits, states = fake_runtime(monkeypatch)
    calls = []
    def chunk(velocity, noise, capture_hidden, capture_indices):
        assert not capture_hidden and capture_indices is None
        calls.append(noise.clone())
        return noise[..., :7].clone(), None
    monkeypatch.setattr(interaction, 'action_chunk', chunk)
    runner = interaction.Runner(runtime, {}, 0, slot_batch=2)
    requests = [dict(kind='final', task=request(i)['task'], state_id=32+i,
                     state={'factor': torch.tensor(20.+i)}, noise_root=7, capture=True) for i in (0, 1)]
    try:
        results = list(runner.run(requests))
        assert not edits and states[0] == [20., 21.]
        first = next(r for r in results if r['request']['task']['task_id'] == 0)
        record = first['trajectory'][0]
        seed = policy_noise_seed(7, 'libero_spatial', 0, 32, 0)
        expected = torch.randn((50, 32), generator=torch.Generator().manual_seed(seed))
        assert record['noise_seed'] == seed and record['step'] == 0
        torch.testing.assert_close(record['noise'], expected)
        torch.testing.assert_close(record['normalized_actions'], expected[:, :7])
        torch.testing.assert_close(record['commands'], expected[:, :7])
        assert record['raw']['images'].dtype == torch.uint8 and record['raw']['proprio'].shape == (8,)
        assert record['executed'].tolist() == [True] * 5 and 'hidden' not in record
        torch.save(first['trajectory'], tmp_path / 'raw.pt')
        restored = torch.load(tmp_path / 'raw.pt', weights_only=False)[0]
        for name in ('noise', 'normalized_actions', 'commands'):
            torch.testing.assert_close(restored[name], record[name])
            assert restored[name].untyped_storage().nbytes() == restored[name].nbytes
    finally:
        runner.close()


def test_final_capture_records_only_the_executed_terminal_prefix(monkeypatch):
    runtime, *_ = fake_runtime(monkeypatch)
    runner = interaction.Runner(runtime, {}, 0)
    slot = runner._new_slot(dict(kind='final', task=request(0)['task'], state_id=32, capture=True))
    slot.raw = {'images': torch.zeros(2, 3, 2, 2, dtype=torch.uint8), 'proprio': torch.zeros(8)}
    slot.pending = dict(raw=slot.raw, noise=torch.zeros(50, 32), noise_seed=10,
                        normalized_actions=torch.ones(50, 7), commands=torch.ones(50, 7), step=20)
    actual = np.arange(14, dtype=np.float32).reshape(2, 7)
    runner._record(slot, dict(raw=slot.raw, executed=actual, reward=1., done=True))
    row = slot.trajectory[0]
    assert row['executed'].tolist() == [True, True, False, False, False]
    torch.testing.assert_close(row['actions'][:2], torch.from_numpy(actual))
    assert row['actions'][2:].count_nonzero() == 0 and row['done']


# Frozen same-incoming native function and FM consumers.
class TinyCore:
    config = SimpleNamespace(time_sampling_beta_alpha=1.5, time_sampling_beta_beta=1.,
                             time_sampling_scale=.999, time_sampling_offset=.001)

    def __init__(self):
        self.values, self.calls = None, []

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def sample_time(self, size, device):
        return torch.rand(size, device=device)

    def embed_prefix(self, images, masks, tokens, token_masks):
        values = images[0].mean((1, 2, 3))
        return values[:, None, None].expand(-1, 2, 4), torch.ones(len(values), 2, dtype=torch.bool), None

    def denoise_step(self, padding, cache, noisy, time):
        assert not torch.is_grad_enabled()
        self.calls.append(len(noisy))
        return .03 * noisy + self.values[:, None, None] + cache[:, :1, :1] + time[:, None, None]


class TinyPolicy(torch.nn.Module):
    config = SimpleNamespace(chunk_size=50, max_action_dim=32,
                             output_features={'action': SimpleNamespace(shape=(7,))})

    def __init__(self):
        super().__init__()
        self.model = TinyCore()

    def _preprocess_images(self, batch):
        images = [batch[key] for key in ('observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb')]
        return images, [torch.ones(len(image), dtype=torch.bool) for image in images]

    def prepare_action(self, batch):
        return torch.nn.functional.pad(batch['action'], (0, 25))


class TinyProcessor:
    def __call__(self, raw):
        assert 'action' not in raw
        return {**{key: value.unsqueeze(0) for key, value in raw.items() if key.startswith('observation.images.')},
                'observation.language.tokens': torch.ones(1, 3, dtype=torch.long),
                'observation.language.attention_mask': torch.ones(1, 3, dtype=torch.bool)}

    def training_batch(self, raw):
        return {'observation.images.base_0_rgb': raw['observation.images.camera1'].float() / 255,
                'observation.images.left_wrist_0_rgb': raw['observation.images.camera2'].float() / 255,
                'observation.language.tokens': torch.ones(28, 3, dtype=torch.long),
                'observation.language.attention_mask': torch.ones(28, 3, dtype=torch.bool),
                'action': raw['action'] / 2}

    def unnormalize_action(self, value):
        return value * 2 + 3


class TinyExecution:
    def __init__(self, policy):
        self.policy, self.assignments = policy, []

    @contextmanager
    def activate(self, states, *, batch_indices):
        assert not torch.is_grad_enabled()
        self.assignments.append(batch_indices.tolist())
        self.policy.model.values = torch.tensor([float(state['weight']) for state in states])[batch_indices]
        try:
            yield
        finally:
            self.policy.model.values = None


class TinyQueries:
    instances = []

    def __init__(self, _asset_root):
        self.closed = False
        self.tasks = {0: SimpleNamespace(episode_lengths=tuple(20 + demo for demo in range(50)))}
        self.instances.append(self)

    def raw_query_batch(self, event):
        self.event = event
        return {'observation.images.camera1': torch.zeros(28, 3, 256, 256, dtype=torch.uint8),
                'observation.images.camera2': torch.zeros(28, 3, 256, 256, dtype=torch.uint8),
                'observation.state': torch.zeros(28, 8), 'task': ['exact language'] * 28,
                'action': torch.ones(28, 50, 7) * 2,
                'action_is_pad': torch.arange(50).expand(28, -1) >= 3}

    def close(self):
        self.closed = True


def raw(value=0):
    return dict(images=torch.full((2, 3, 256, 256), value, dtype=torch.uint8), proprio=torch.zeros(8))


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(readouts, 'QueryData', TinyQueries)
    monkeypatch.setattr(execution, 'prepare_prefix_kv_cache', lambda _policy, prefix: prefix.embeddings)
    monkeypatch.setattr(function_credit, 'prepare_prefix_kv_cache',
                        lambda _policy, prefix, **_kwargs: prefix.embeddings)
    parent, root = tmp_path / 'parent', tmp_path / 'diagnostic'
    parent.mkdir()
    condition = dict(condition_id='train_task000_demo29', task_id=0, suite='libero_spatial', suite_task_id=0,
        teacher_demo=29, language='exact language', final_state_ids=[32, 33, 34], actual_J=2,
        source_record=str(parent / 'record.json'), source_experience=str(parent / 'experience.pt'),
        last_event=dict(episode=1, endpoint=13, incoming='incoming_001.safetensors', behavior_version='lambda001'))
    weights = {'I': parent / 'incoming_001.safetensors', 'E+': parent / 'end.safetensors',
               'E0': root / 'materialized' / condition['condition_id'] / 'E0.safetensors'}
    for index, path in enumerate(weights.values(), 1):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({'weight': torch.tensor(float(index))}, path)
    condition['weights'] = {arm: str(path) for arm, path in weights.items()}
    source = {**condition, 'complete': True, 'events': [dict(episode=0), condition['last_event']],
              'metrics': {'actual_J': 2}, 'MT_reference': str(parent / 'unused_mt.safetensors')}
    (parent / 'record.json').write_text(json.dumps(source))
    (root / 'run_contract.json').write_text(json.dumps(dict(asset_root=str(tmp_path), checkpoint='chi360',
                                                          conditions=[condition])))
    records, observations = [], {}
    for episode, count in enumerate((5, 8)):
        for index in range(count):
            key = f'{episode}:{index}'
            observations[key] = raw(index)
            records.append(dict(pre=key, episode=episode, step=index * 5, actions=torch.ones(5, 7),
                executed=torch.tensor([True, True, False, False, False]), feedback=torch.zeros(4),
                behavior_version='lambda001' if episode else 'MT'))
    torch.save(dict(records=records, observations=observations, endpoints=[5, 13],
                    episodes=[{'policy_noise_seeds': list(range(500, 505))},
                              {'policy_noise_seeds': list(range(600, 608))}]), parent / 'experience.pt')
    for state in (32, 33, 34):
        for arm, count in (('I', 7), ('E+', 9)):
            path = root / 'evaluation' / 'traces' / condition['condition_id'] / arm / f'state{state:02d}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            trace = []
            for index in range(count):
                seed = state * 100 + index
                norm = torch.ones(50, 7) * (1 if arm == 'I' else 2)
                trace.append(dict(raw=raw(0 if arm == 'I' else 10), noise_seed=seed,
                    noise=torch.randn((50, 32), generator=torch.Generator().manual_seed(seed)),
                    normalized_actions=norm, commands=norm * 2 + 3, actions=torch.ones(5, 7),
                    executed=torch.ones(5, dtype=torch.bool), step=index * 5, reward=0., done=False))
            torch.save(dict(condition_id=condition['condition_id'], arm=arm, init_state_id=state,
                            row={}, records=trace, complete=True), path)
    policy = TinyPolicy()
    runtime = SimpleNamespace(policy=policy, processor=TinyProcessor(), device=torch.device('cpu'),
                              asset_root=tmp_path, execution=TinyExecution(policy))
    return runtime, root, condition


def test_full_readout_preserves_pairing_units_missing_legacy_tail_and_padding(case):
    runtime, root, condition = case
    rng = torch.get_rng_state().clone()
    result = readouts.read_condition(runtime, root, condition, microbatch=17)
    assert torch.equal(torch.get_rng_state(), rng)
    assert result['counts']['extra_ten_step_predictions'] == 72
    assert result['counts']['fm_predictions'] == 84
    assert result['counts']['fm_prefix_batches'] == 2
    assert len(runtime.policy.model.calls) == 10 * 5 + 3 * 2
    assert any(len(set(indices)) > 1 for indices in runtime.execution.assignments)
    payload = torch.load(result['predictions'], weights_only=False)
    assert [point['decision_index'] for point in payload['cross'][:6]] == [0, 1, 2, 3, 4, 6]
    assert [point['record_index'] for point in payload['practice']] == [5, 6, 7, 9, 10, 12]
    point = payload['cross'][0]
    assert torch.equal(point['predictions']['I@I']['commands'], torch.ones(50, 7) * 5)
    for unit in ('normalized_actions', 'commands'):
        parts = point['differences']
        torch.testing.assert_close(parts['total'][unit], parts['parameter_on_I'][unit] + parts['state_with_E+'][unit])
        torch.testing.assert_close(parts['total'][unit], parts['parameter_on_E+'][unit] + parts['state_with_I'][unit])
    practice = payload['practice'][0]
    assert not practice['original_full50_saved'] and practice['noise_seed'] == 600
    assert torch.equal(practice['noise'], torch.randn((50, 32), generator=torch.Generator().manual_seed(600)))
    assert practice['original']['executed'].sum() == 2
    fm = payload['fm']
    assert fm['action_is_pad'].sum() == 28 * 47
    assert len(set(demo for demo, _ in fm['queries28'])) == 7
    assert all(demo != 29 for demo, _ in fm['queries28'])
    for arm in readouts.ARMS:
        expected = (fm['predictions'][arm][..., :7] - fm['target'][..., :7]).square().mean((1, 2))
        torch.testing.assert_close(expected, torch.tensor(fm['metrics'][arm]['per_query_loss']))
        assert expected.mean().item() == pytest.approx(fm['metrics'][arm]['full50_loss'])
    for label, left, right in (('E+minusI', 'E+', 'I'), ('E0minusI', 'E0', 'I'), ('E+minusE0', 'E+', 'E0')):
        delta = ((fm['predictions'][left][..., :7] - fm['target'][..., :7]).square()
                 - (fm['predictions'][right][..., :7] - fm['target'][..., :7]).square())
        assert fm['paired'][label]['full50']['mean'] == pytest.approx(delta.mean().item())
        assert fm['paired'][label]['front5']['mean'] == pytest.approx(delta[:, :5].mean().item())
    assert TinyQueries.instances[-1].closed
    calls = len(runtime.policy.model.calls)
    repeated = readouts.read_condition(runtime, root, condition, microbatch=28)
    assert repeated['counts'] == result['counts'] and len(runtime.policy.model.calls) == calls


@pytest.mark.parametrize('mismatch', ['noise', 'seed'])
def test_common_prefix_rejects_actual_noise_or_seed_mismatch(case, mismatch):
    runtime, root, condition = case
    path = root / 'evaluation' / 'traces' / condition['condition_id'] / 'E+' / 'state32.pt'
    trace = torch.load(path, weights_only=False)
    if mismatch == 'noise':
        trace['records'][0]['noise'][0, 0] += 1
    else:
        trace['records'][0]['noise_seed'] += 1
    torch.save(trace, path)
    with pytest.raises(ValueError, match='policy-noise pairing'):
        readouts.read_condition(runtime, root, condition, microbatch=28)
    assert not runtime.policy.model.calls


def test_completed_readout_requires_matching_prediction_identity(case):
    runtime, root, condition = case
    result = readouts.read_condition(runtime, root, condition, microbatch=28)
    path = root / 'readouts' / condition['condition_id'] / 'result.json'
    saved = json.loads(path.read_text())
    saved['identity']['checkpoint'] = 'other-checkpoint'
    path.write_text(json.dumps(saved))
    calls = len(runtime.policy.model.calls)
    with pytest.raises(ValueError, match='frozen condition'):
        readouts.read_condition(runtime, root, condition, microbatch=28)
    assert len(runtime.policy.model.calls) == calls and result['complete']


def test_fm_uses_one_logical_sample_independent_of_physical_chunk(case):
    runtime, root, condition = case
    states = [readouts.load_file(condition['weights'][arm]) for arm in readouts.ARMS]
    with torch.no_grad():
        left = readouts._fm(runtime, runtime.asset_root, condition, states, 7)
        right = readouts._fm(runtime, runtime.asset_root, condition, states, 28)
    for key in ('noise', 'time', 'target'):
        assert torch.equal(left[key], right[key])
    assert left['queries28'] == right['queries28']
    for arm in readouts.ARMS:
        torch.testing.assert_close(left['predictions'][arm], right['predictions'][arm])


def test_fixed_fm_sampling_uses_declared_seed_and_endpoint_intervals():
    tasks = {20: SimpleNamespace(episode_lengths=tuple(13 + demo for demo in range(50)))}
    event = readouts._fm_queries(tasks, {'task_id': 20, 'teacher_demo': 15})
    rng = np.random.default_rng(np.random.SeedSequence([20261009, 0xED17, 20, 15]))
    expected = []
    for demo in map(int, rng.choice([d for d in range(50) if d != 15], 7, replace=False)):
        available = tasks[20].episode_lengths[demo] - 1
        for interval in range(4):
            expected.append((demo, int(rng.integers(available * interval // 4, available * (interval + 1) // 4))))
    assert event.queries28 == tuple(expected)
