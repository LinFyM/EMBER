"""Paired incoming/outgoing native FM credit for one observed edit event."""
from __future__ import annotations

import torch

from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample
from ember.writer.runtime import autocast


def per_query_loss(prediction, sample):
    if prediction.shape != sample.target.shape or prediction.shape[1:] != (50, 32):
        raise ValueError('FM retains native 50x32 latent and first-seven action consumer')
    return (prediction[..., :7].float() - sample.target[..., :7].float()).square().mean((1, 2))


def edit_objective(outgoing, incoming, *, condition_weight=.25):
    if outgoing.shape != incoming.shape or outgoing.ndim != 1:
        raise ValueError('keep compares the same per-query incoming/outgoing losses')
    regression = torch.relu(outgoing - incoming.detach())
    return condition_weight * (outgoing + .2 * regression).mean(), regression


def _map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    return type(value)(_map_tensors(item, function) for item in value)


def _join(values):
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values)
    return type(values[0])(_join([v[index] for v in values]) for index in range(len(values[0])))


def fm_credit(runtime, incoming, outgoing, batches, *, seeds, microbatch):
    """Batch different complete LoRAs and queries without changing event weights.

    All28 noise/time samples are drawn per logical event before chunking. The
    same prepared native prefix is used by incoming/outgoing. Indexed hooks
    retain the factor VJP while sharing each event's factors across its queries.
    """
    count = len(incoming)
    if not 1 <= count <= 4 or len(outgoing) != count or len(batches) != count or len(seeds) != count:
        raise ValueError('FM physical rank must own one through four logical events')
    if microbatch < 1 or any(len(b['action']) != 28 for b in batches):
        raise ValueError('each event needs seven nonteacher episodes x four intervals')
    samples = [flow_sample(runtime.policy, batch, seed=seed, device=runtime.device,
                          random_batch=28, offset=0) for batch, seed in zip(batches, seeds, strict=True)]
    sample = FlowSample(_join([s.arguments for s in samples]), torch.cat([s.target for s in samples]), 7)
    owner = NativeFlowPrediction(runtime.policy)
    leaves = [{k: v.detach().to(runtime.device).requires_grad_() for k, v in state.items()} for state in outgoing]
    flattened = [value for state in leaves for value in state.values()]
    credits = [{} for _ in range(count)]
    losses, keep = [[0., 0.] for _ in range(count)], [0.] * count
    for start in range(0, 28 * count, microbatch):
        end = min(start + microbatch, 28 * count)
        chunk = FlowSample(_map_tensors(sample.arguments, lambda v: v[start:end]), sample.target[start:end], 7)
        assignment = torch.arange(start, end, device=runtime.device) // 28
        with autocast(runtime.device):
            prepared = owner.prepare(chunk)
        with torch.no_grad(), runtime.execution.activate(incoming, batch_indices=assignment), autocast(runtime.device):
            before = per_query_loss(owner(chunk, prepared), chunk)
        with runtime.execution.activate(leaves, batch_indices=assignment), autocast(runtime.device):
            after = per_query_loss(owner(chunk, prepared), chunk)
            regression = torch.relu(after - before.detach())
            objective = (after + .2 * regression).sum() / 28 / 4
            gradients = torch.autograd.grad(objective, flattened)
        offset = 0
        for index, state in enumerate(leaves):
            for name in state:
                value = gradients[offset].detach().float()
                offset += 1
                if name in credits[index]:
                    credits[index][name].add_(value)
                else:
                    credits[index][name] = value
            mask = assignment == index
            losses[index][0] += float(before[mask].sum()) / 28
            losses[index][1] += float(after.detach()[mask].sum()) / 28
            keep[index] += float(regression.detach()[mask].sum()) / 28
    return [dict(cotangent=credit, incoming_loss=loss[0], outgoing_loss=loss[1], keep=penalty,
                 weighted_loss=loss[1] + .2 * penalty, paired_difference=loss[1] - loss[0], queries=28)
            for credit, loss, penalty in zip(credits, losses, keep, strict=True)]


def surrogate(outgoing, credit):
    """Exact shared-module VJP carrier; its numerical value is not an FM loss."""
    if outgoing.keys() != credit.keys():
        raise ValueError('shared edit VJP lost complete factor coverage')
    return sum((outgoing[name].float() * value).sum() for name, value in credit.items())
