"""CPU checks of the actual readout/ten-step/FM consumers with a tiny policy."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from ember.experience_compiler import edit_readouts as readouts
from ember.experience_compiler import execution
from ember.writer import function_credit


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
