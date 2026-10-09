"""Native FM, successful-function preservation and real-query score credit."""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch

from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample
from ember.writer.runtime import autocast
from .interaction import processed


def per_query_loss(prediction, sample):
    if prediction.shape != sample.target.shape or prediction.shape[1:] != (50, 32):
        raise ValueError('FM retains native 50x32 latent and first-seven action consumer')
    return (prediction[..., :7].float() - sample.target[..., :7].float()).square().mean((1, 2))


def _map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, Mapping):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, function) for item in value)
    return value


def _join(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    if isinstance(first, Mapping):
        if any(value.keys() != first.keys() for value in values):
            raise ValueError('native query batches must retain the same fields')
        return {key: _join([value[key] for value in values]) for key in first}
    if isinstance(first, (tuple, list)):
        if any(len(value) != len(first) for value in values):
            raise ValueError('native query arguments changed structure')
        return type(first)(_join([value[index] for value in values]) for index in range(len(first)))
    raise TypeError('native batched arguments must contain tensors')


def _weights(count, condition_weight):
    if count < 1:
        raise ValueError('credit needs at least one logical event')
    if condition_weight is None:
        values = [1. / count] * count
    elif isinstance(condition_weight, Real):
        values = [float(condition_weight)] * count
    else:
        values = list(map(float, condition_weight))
    if len(values) != count or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('explicit weights must cover each unchanged logical event')
    return values


class _FactorCredit:
    """One detached full-LoRA leaf set, shared by physical query chunks."""
    def __init__(self, runtime, states):
        self.leaves = [{key: value.detach().to(runtime.device).requires_grad_()
                        for key, value in state.items()} for state in states]
        if any(not state or state.keys() != states[0].keys() for state in states):
            raise ValueError('credit needs the same complete factor mapping per event')
        self.flat = [value for state in self.leaves for value in state.values()]
        self.credits = [{} for _ in states]

    def backward(self, output, active, *, grad_outputs=None):
        gradients = torch.autograd.grad(output, self.flat, grad_outputs=grad_outputs, allow_unused=True)
        offset = 0
        for index, state in enumerate(self.leaves):
            for name in state:
                gradient = gradients[offset]
                offset += 1
                if gradient is None:
                    if index in active:
                        raise RuntimeError(f'active native query did not consume factor {name}')
                    continue
                value = gradient.detach().float()
                if name in self.credits[index]:
                    self.credits[index][name].add_(value)
                else:
                    self.credits[index][name] = value

    def result(self):
        return [{name: (self.credits[index][name] if name in self.credits[index]
                       else torch.zeros_like(value, dtype=torch.float32))
                 for name, value in state.items()} for index, state in enumerate(self.leaves)]


def fm_credit(runtime, incoming, outgoing, batches, *, seeds, microbatch, condition_weight=None):
    """All28 cross-episode queries/event; physical chunks never set task weight.

    Incoming losses are paired reporting evidence only. The actual objective
    is the outgoing FM mean, with no former FM-regression preservation term.
    Explicit event weights support a shard of a globally normalized update.
    """
    count = len(outgoing)
    weights = _weights(count, condition_weight)
    if len(incoming) != count or len(batches) != count or len(seeds) != count:
        raise ValueError('FM needs paired incoming/outgoing, queries and seeds per event')
    if microbatch < 1 or any(len(batch['action']) != 28 for batch in batches):
        raise ValueError('each event needs seven nonteacher episodes x four intervals')
    samples = [flow_sample(runtime.policy, _map_tensors(batch, torch.Tensor.detach), seed=seed,
                          device=runtime.device, random_batch=28, offset=0)
               for batch, seed in zip(batches, seeds, strict=True)]
    sample = FlowSample(_join([item.arguments for item in samples]),
                        torch.cat([item.target for item in samples]), 7)
    owner, factors = NativeFlowPrediction(runtime.policy), _FactorCredit(runtime, outgoing)
    losses = [[0., 0.] for _ in range(count)]
    for start in range(0, 28 * count, microbatch):
        end = min(start + microbatch, 28 * count)
        chunk = FlowSample(_map_tensors(sample.arguments, lambda value: value[start:end]),
                           sample.target[start:end], 7)
        assignment = torch.arange(start, end, device=runtime.device) // 28
        active = set(range(start // 28, (end - 1) // 28 + 1))
        with autocast(runtime.device):
            prepared = owner.prepare(chunk)
        with torch.no_grad(), runtime.execution.activate(incoming, batch_indices=assignment), autocast(runtime.device):
            before = per_query_loss(owner(chunk, prepared), chunk)
        with runtime.execution.activate(factors.leaves, batch_indices=assignment), autocast(runtime.device):
            after = per_query_loss(owner(chunk, prepared), chunk)
            coefficients = torch.as_tensor(weights, device=after.device)[assignment] / 28
            factors.backward((after * coefficients).sum(), active)
        for index in active:
            mask = assignment == index
            losses[index][0] += float(before[mask].sum()) / 28
            losses[index][1] += float(after.detach()[mask].sum()) / 28
    return [dict(cotangent=credit, incoming_loss=loss[0], outgoing_loss=loss[1],
                 paired_difference=loss[1] - loss[0], queries=28, condition_weight=weight,
                 weighted_loss=weight * loss[1])
            for credit, loss, weight in zip(factors.result(), losses, weights, strict=True)]


def _support_queries(supports):
    batches, noises, owners, counts = [], [], [], []
    for index, support in enumerate(supports):
        noise = support.get('noise') if support else None
        count = 0 if noise is None else len(noise)
        if count and (count > 16 or noise.shape[1:] != (50, 32)):
            raise ValueError('keep needs at most16 real successful states and their native noise')
        counts.append(count)
        if count:
            batches.append(_map_tensors(support['batch'], torch.Tensor.detach))
            noises.append(noise.detach())
            owners.extend([index] * count)
    return batches, noises, owners, counts


def keep_credit(runtime, incoming, outgoing, supports, *, microbatch, condition_weight=None):
    """Mean35-coordinate ten-step function MSE on actual successful states.

    Selection and success provenance belong to the real-experience owner;
    an empty support is exactly zero, without an invented action target.
    """
    count = len(outgoing)
    weights = _weights(count, condition_weight)
    if len(incoming) != count or len(supports) != count or microbatch < 1:
        raise ValueError('keep must retain paired events and a positive physical microbatch')
    factors = _FactorCredit(runtime, outgoing)
    batches, noises, owners, counts = _support_queries(supports)
    losses = [0.] * count
    if owners:
        batch, noise = _join(batches), torch.cat(noises).to(runtime.device)
        for start in range(0, len(owners), microbatch):
            end = min(start + microbatch, len(owners))
            sliced = _map_tensors(batch, lambda value: value[start:end])
            assignment = torch.tensor(owners[start:end], device=runtime.device, dtype=torch.long)
            with torch.no_grad(), autocast(runtime.device):
                before = runtime.native_actions(incoming, sliced, noise[start:end],
                                                batch_indices=assignment, checkpointed=False)
            with autocast(runtime.device):
                after = runtime.native_actions(factors.leaves, sliced, noise[start:end],
                                               batch_indices=assignment, checkpointed=True)
                if before.shape != after.shape or after.shape[1:] != (5, 7):
                    raise ValueError('keep must compare actual normalized first5x7 actions')
                loss = (after.float() - before.float()).square().mean((1, 2))
                coefficients = loss.new_tensor([weights[index] / counts[index] for index in owners[start:end]])
                factors.backward((loss * coefficients).sum(), set(owners[start:end]))
            for index in set(owners[start:end]):
                losses[index] += float(loss.detach()[assignment == index].sum()) / counts[index]
    return [dict(cotangent=credit, keep_loss=loss, points=points, condition_weight=weight,
                 weighted_loss=weight * loss)
            for credit, loss, points, weight in zip(factors.result(), losses, counts, weights, strict=True)]


def _pg_queries(runtime, episodes, weights):
    batches, noises, actions, masks, owners, coefficients, summaries = [], [], [], [], [], [], []
    for index, queries in enumerate(episodes):
        if len(queries) != 2:
            raise ValueError('real return credit averages exactly two registered query states per event')
        summary = dict(query_returns=[], baseline_returns=[], advantages=[], replans=0, reservoir_records=0)
        for episode in queries:
            returned, baseline = episode['success'], episode['baseline_success']
            if type(returned) is not bool or type(baseline) is not bool:
                raise ValueError('PG uses actual boolean success returns only')
            total, records = episode['total_replans'], episode['records']
            if type(total) is not int or total < 0 or len(records) != min(16, total):
                raise ValueError('uniform reservoir must retain min(16,N) of the actual N replans')
            advantage = float(returned) - float(baseline)
            summary['query_returns'].append(int(returned))
            summary['baseline_returns'].append(int(baseline))
            summary['advantages'].append(advantage)
            summary['replans'] += total
            summary['reservoir_records'] += len(records)
            for record in records:
                y, noise, executed = record['gaussian_actions'], record['noise'], record['executed']
                if y.shape != (5, 7) or noise.shape != (50, 32) or executed.shape != (5,) or executed.dtype != torch.bool:
                    raise ValueError('PG needs raw Gaussian y, native noise and actual five-action execution mask')
                if bool((executed[1:] & ~executed[:-1]).any()):
                    raise ValueError('actual execution mask must retain a prefix')
                batches.append(processed(runtime, record['raw'], episode['language']))
                noises.append(noise.detach())
                actions.append(y.detach())
                masks.append(executed.detach())
                owners.append(index)
                coefficients.append(weights[index] * advantage * total / len(records) / 2)
        summaries.append(summary)
    return batches, noises, actions, masks, owners, coefficients, summaries


def pg_credit(runtime, outgoing, episodes, *, microbatch, condition_weight=None):
    """Negative-return score VJP through the real complete-ten-step mean.

    Baseline policy/exploration RNG independence is established at collection.
    The stored untransformed y and current score are stopped; only mu receives
    the cotangent. There is no FM here, no action-count mean and no noise-density
    derivative. N/K compensation and two-query averaging precede chunking.
    """
    count = len(outgoing)
    weights = _weights(count, condition_weight)
    if len(episodes) != count or microbatch < 1:
        raise ValueError('PG needs query pairs per logical event and a positive microbatch')
    factors = _FactorCredit(runtime, outgoing)
    batches, noises, actions, masks, owners, coefficients, summaries = _pg_queries(runtime, episodes, weights)
    if owners:
        batch = _join(batches)
        noise, y = torch.stack(noises).to(runtime.device), torch.stack(actions).to(runtime.device)
        executed = torch.stack(masks).to(runtime.device)
        for start in range(0, len(owners), microbatch):
            end = min(start + microbatch, len(owners))
            assignment = torch.tensor(owners[start:end], device=runtime.device, dtype=torch.long)
            sliced = _map_tensors(batch, lambda value: value[start:end])
            with autocast(runtime.device):
                mu = runtime.native_actions(factors.leaves, sliced, noise[start:end],
                                            batch_indices=assignment, checkpointed=True)
                if mu.shape != y[start:end].shape:
                    raise ValueError('PG mean must be the actual normalized first5x7 native actions')
                score = (y[start:end].float() - mu.detach().float()) / .1**2
                score = score * executed[start:end, :, None]
                coefficient = score.new_tensor(coefficients[start:end])[:, None, None]
                factors.backward(mu.float(), set(owners[start:end]), grad_outputs=-(score * coefficient).detach())
    return [dict(cotangent=credit, queries=2, condition_weight=weight, gaussian_std=.1,
                 mean_return=sum(summary['query_returns']) / 2,
                 mean_baseline_return=sum(summary['baseline_returns']) / 2,
                 mean_advantage=sum(summary['advantages']) / 2,
                 zero_advantage=not any(summary['advantages']), **summary)
            for credit, summary, weight in zip(factors.result(), summaries, weights, strict=True)]
