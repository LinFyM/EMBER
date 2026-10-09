"""Outer-objective invariants; real native derivatives are checked by the run."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ember.experience_compiler import credit, learning
from ember.writer.function_credit import FlowSample


class NativeMock:
    """Small deterministic consumer for batching, weights and score oracles."""
    def __init__(self):
        self.device = torch.device('cpu')
        self.policy = SimpleNamespace(runtime=self)
        self.execution = SimpleNamespace(activate=self.activate)
        self.calls = []
        self.compiler = nn.Linear(1, 1, bias=False)

    @contextmanager
    def activate(self, states, *, batch_indices):
        self.states, self.assignment = states, batch_indices
        yield

    @staticmethod
    def factor_response(states, assignment):
        a = torch.stack([state['A'].reshape(()) for state in states])
        b = torch.stack([state['B'].reshape(()) for state in states])
        return (a * b)[assignment]

    def native_actions(self, states, batch, noise, *, batch_indices, checkpointed=True):
        self.calls.append(dict(batch=len(noise), checkpointed=checkpointed))
        pressure = self.factor_response(states, batch_indices) + batch['input'].reshape(-1)
        coordinate = torch.arange(1, 33, dtype=noise.dtype) / 32
        z = noise
        for _ in range(10):
            z = z - .1 * (.02 * z + pressure[:, None, None] * coordinate)
        return z[:, :5, :7]


class FMPrediction:
    def __init__(self, policy):
        self.runtime = policy.runtime

    def prepare(self, sample):
        return sample.arguments[0]

    def __call__(self, sample, prepared):
        runtime = self.runtime
        response = runtime.factor_response(runtime.states, runtime.assignment)
        return prepared * response[:, None, None]


@pytest.fixture
def runtime(monkeypatch):
    result = NativeMock()
    monkeypatch.setattr(credit, 'NativeFlowPrediction', FMPrediction)
    monkeypatch.setattr(credit, 'flow_sample', lambda policy, batch, **kw:
                        FlowSample((batch['input'],), batch['target'], 7))
    monkeypatch.setattr(credit, 'processed', lambda runtime, raw, language:
                        {'input': raw['input'].reshape(1, 1)})
    return result


def states(count, *, incoming=False):
    return [{'A': torch.tensor(.1 if incoming else .4 + index / 10, requires_grad=True),
             'B': torch.tensor(.3 + index / 20, requires_grad=True)} for index in range(count)]


def fm_batches(count):
    generator = torch.Generator().manual_seed(14)
    return [dict(action=torch.zeros(28, 50, 7), input=torch.randn(28, 50, 32, generator=generator),
                 target=torch.randn(28, 50, 32, generator=generator)) for _ in range(count)]


def assert_gradients(rows, expected, outgoing):
    position = 0
    for row, state in zip(rows, outgoing, strict=True):
        for name in state:
            torch.testing.assert_close(row['cotangent'][name], expected[position], rtol=2e-5, atol=2e-6)
            position += 1
    assert all(value.grad is None for state in outgoing for value in state.values())


@pytest.mark.parametrize('microbatch', [1, 17, 1000])
def test_fm_six_events_matches_dense_mean_without_regression_keep(runtime, microbatch):
    incoming, outgoing, batches = states(6, incoming=True), states(6), fm_batches(6)
    loss = sum(((batch['input'][..., :7] * state['A'] * state['B']
                 - batch['target'][..., :7]).square().mean()) / 6
               for state, batch in zip(outgoing, batches, strict=True))
    expected = torch.autograd.grad(loss, [value for state in outgoing for value in state.values()])
    rows = credit.fm_credit(runtime, incoming, outgoing, batches, seeds=list(range(6)), microbatch=microbatch)
    assert_gradients(rows, expected, outgoing)
    assert sum(row['weighted_loss'] for row in rows) == pytest.approx(float(loss.detach()), rel=1e-6)
    assert all(row['queries'] == 28 and 'keep' not in row for row in rows)
    assert all(value.grad is None for state in incoming for value in state.values())


def test_explicit_global_weight_is_not_renormalized_to_local_shard(runtime):
    incoming, outgoing, batches = states(2, incoming=True), states(2), fm_batches(2)
    local = credit.fm_credit(runtime, incoming, outgoing, batches, seeds=[2, 3], microbatch=5)
    global_shard = credit.fm_credit(runtime, incoming, outgoing, batches, seeds=[2, 3],
                                   microbatch=13, condition_weight=1 / 8)
    for normal, shard in zip(local, global_shard, strict=True):
        for name in normal['cotangent']:
            torch.testing.assert_close(shard['cotangent'][name], normal['cotangent'][name] / 4)


@pytest.mark.parametrize('microbatch', [1, 7, 64])
def test_keep_averages_success_points_and35_coordinates_independently(runtime, microbatch):
    incoming, outgoing = states(3, incoming=True), states(3)
    supports = [{}, dict(batch={'input': torch.tensor([[1.]])}, noise=torch.zeros(1, 50, 32)),
                dict(batch={'input': torch.arange(16).reshape(16, 1).float()}, noise=torch.zeros(16, 50, 32))]
    objective = 0
    for index, support in enumerate(supports):
        if not support:
            continue
        assignment = torch.full((len(support['noise']),), index, dtype=torch.long)
        with torch.no_grad():
            before = runtime.native_actions(incoming, support['batch'], support['noise'], batch_indices=assignment)
        after = runtime.native_actions(outgoing, support['batch'], support['noise'], batch_indices=assignment)
        objective = objective + (after - before).square().mean() / 3
    expected = torch.autograd.grad(objective, [value for state in outgoing for value in state.values()])
    rows = credit.keep_credit(runtime, incoming, outgoing, supports, microbatch=microbatch)
    assert_gradients(rows, expected, outgoing)
    assert [row['points'] for row in rows] == [0, 1, 16]
    assert rows[0]['keep_loss'] == 0
    assert sum(row['weighted_loss'] for row in rows) == pytest.approx(float(objective.detach()), rel=1e-5)


def query_episodes(count, *, zero=False):
    generator = torch.Generator().manual_seed(83)
    events = []
    for index in range(count):
        queries = []
        for query, total in enumerate((3, 30)):
            records = []
            for replan in range(min(16, total)):
                executed = torch.arange(5) < (1 + replan % 5)
                records.append(dict(raw={'input': torch.tensor((replan + index) / 50)},
                    noise=torch.randn(50, 32, generator=generator),
                    gaussian_actions=torch.randn(5, 7, generator=generator).requires_grad_(), executed=executed))
            queries.append(dict(success=False if zero else query == 0,
                baseline_success=False if zero else query == 1,
                total_replans=total, records=records, language='exact task'))
        events.append(queries)
    return events


def dense_pg_loss(runtime, outgoing, episodes):
    # Independent log-density oracle, rather than the production score formula.
    objective = 0
    for index, queries in enumerate(episodes):
        for episode in queries:
            advantage = int(episode['success']) - int(episode['baseline_success'])
            for record in episode['records']:
                batch = {'input': record['raw']['input'].reshape(1, 1)}
                mean = runtime.native_actions(outgoing, batch, record['noise'][None],
                                              batch_indices=torch.tensor([index]))[0]
                log_density = -(record['gaussian_actions'].detach() - mean).square() / (2 * .1**2)
                log_density = (log_density * record['executed'][:, None]).sum()
                objective = objective - (advantage * episode['total_replans']
                    / len(episode['records']) / 2 / len(episodes)) * log_density
    return objective


@pytest.mark.parametrize('microbatch', [1, 5, 1000])
def test_pg_matches_masked_log_density_and_reservoir_compensation(runtime, microbatch):
    outgoing, episodes = states(3), query_episodes(3)
    objective = dense_pg_loss(runtime, outgoing, episodes)
    expected = torch.autograd.grad(objective, [value for state in outgoing for value in state.values()])
    rows = credit.pg_credit(runtime, outgoing, episodes, microbatch=microbatch)
    assert_gradients(rows, expected, outgoing)
    assert all(row['replans'] == 33 and row['reservoir_records'] == 19 for row in rows)
    assert all(row['mean_return'] == .5 and row['mean_baseline_return'] == .5 for row in rows)
    assert all(not row['zero_advantage'] and row['queries'] == 2 for row in rows)
    assert all(record['gaussian_actions'].grad is None for event in episodes
               for episode in event for record in episode['records'])


def test_no_return_difference_reports_zero_pg_instead_of_fake_labels(runtime):
    rows = credit.pg_credit(runtime, states(2), query_episodes(2, zero=True), microbatch=8)
    assert runtime.calls  # The mean consumer is still exercised on the recorded version.
    assert all(row['zero_advantage'] and row['mean_return'] == 0 for row in rows)
    assert all(torch.count_nonzero(value) == 0 for row in rows for value in row['cotangent'].values())


@pytest.mark.parametrize('violation', ['reservoir', 'mask'])
def test_pg_rejects_lost_reservoir_or_nonprefix_execution(runtime, violation):
    episodes = query_episodes(1)
    if violation == 'reservoir':
        episodes[0][1]['records'].pop()
    else:
        episodes[0][0]['records'][0]['executed'] = torch.tensor([False, True, False, False, False])
    with pytest.raises(ValueError):
        credit.pg_credit(runtime, states(1), episodes, microbatch=4)


def test_rl_update_has_fresh_optimizer_and_never_calls_fm(runtime, monkeypatch):
    supervised, supervised_scheduler = learning.fresh_optimizer(runtime.compiler, stage='supervised')
    runtime.compiler(torch.ones(1, 1)).square().sum().backward()
    learning.finish_update(runtime, supervised, supervised_scheduler)
    assert supervised.state
    reinforcement, reinforcement_scheduler = learning.fresh_optimizer(runtime.compiler, stage='reinforcement')
    assert not reinforcement.state and reinforcement_scheduler.last_epoch == 0
    assert reinforcement is not supervised and reinforcement.param_groups[0]['lr'] == 3e-5
    assert reinforcement.param_groups[0]['weight_decay'] == 0
    monkeypatch.setattr(learning, 'fm_credit', lambda *a, **kw: pytest.fail('RL called FM'))
    collected = []
    runtime.backward_revision = lambda incoming, teacher, experience, support, value, **kw: collected.append(value)
    incoming, outgoing = states(1, incoming=True), states(1)
    prepared = [dict(incoming=incoming[0], outgoing=outgoing[0], teacher={}, experience={}, support={})]
    result = learning.reinforcement_backward(runtime, prepared, query_episodes(1), microbatch=4)
    assert len(collected) == 1 and set(collected[0]) == set(outgoing[0])
    assert result['records'][0]['stage'] == 'reinforcement'
    assert set(result['records'][0]['components']) == {'PG', 'keep'}


def test_supervised_backward_combines_fm_and_independent_keep_before_revision(runtime):
    incoming, outgoing, batches = states(1, incoming=True), states(1), fm_batches(1)
    keep_support = dict(batch={'input': torch.zeros(2, 1)}, noise=torch.zeros(2, 50, 32))
    fm = credit.fm_credit(runtime, incoming, outgoing, batches, seeds=[1], microbatch=7)[0]
    keep = credit.keep_credit(runtime, incoming, outgoing, [keep_support], microbatch=7)[0]
    collected = []
    runtime.backward_revision = lambda incoming, teacher, experience, support, value, **kw: collected.append(value)
    prepared = [dict(incoming=incoming[0], outgoing=outgoing[0], batch=batches[0], seed=1,
                     teacher={}, experience={}, support={}, keep_support=keep_support)]
    result = learning.supervised_backward(runtime, prepared, microbatch=7, revision_microbatch=2)
    for name in outgoing[0]:
        torch.testing.assert_close(collected[0][name], fm['cotangent'][name] + keep['cotangent'][name])
    assert result['records'][0]['stage'] == 'supervised'
    assert set(result['records'][0]['components']) == {'FM', 'keep'}
