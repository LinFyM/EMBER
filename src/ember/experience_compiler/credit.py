"""Per-query FM/keep credit and globally normalized real SDE terminal credit."""
from __future__ import annotations

import torch

from ember.writer.function_credit import NativeFlowPrediction, flow_sample
from ember.writer.runtime import autocast

from .execution import NativeVelocity, score_cotangent
from .interaction import processed


def add_credit(destination, names, gradients, weight=1.):
    for name, value in zip(names, gradients, strict=True):
        contribution = value.detach().float() * weight
        if name in destination:
            destination[name].add_(contribution)
        else:
            destination[name] = contribution


def per_query_loss(prediction, sample):
    if prediction.shape != sample.target.shape or prediction.shape[1:] != (50, 32):
        raise ValueError('FM retains native 50x32 latent with canonical first-seven consumer')
    return (prediction[..., :7].float() - sample.target[..., :7].float()).square().mean((1, 2))


def stage_weights(count):
    if count < 1:
        raise ValueError('compilation must have an initial complete state')
    return [1.] if count == 1 else [1 / (2 * (count - 1))] * (count - 1) + [.5]


def fm_credit(runtime, states, batch, *, seed, microbatch):
    """One noise/time/query sample shared by every actual compilation stage."""
    count = len(batch['action'])
    if count != 28 or microbatch < 1:
        raise ValueError('one condition requires seven episodes x four interval queries')
    owner = NativeFlowPrediction(runtime.policy)
    credits = [{} for _ in states]
    losses, keep = [0.] * len(states), 0.
    alphas, revisions = stage_weights(len(states)), len(states) - 1
    for start in range(0, count, microbatch):
        end = min(count, start + microbatch)
        sliced = {k: v[start:end] if isinstance(v, torch.Tensor) and v.ndim and len(v) == count else v
                  for k, v in batch.items()}
        sample = flow_sample(runtime.policy, sliced, seed=seed, device=runtime.device,
                             random_batch=count, offset=start)
        with autocast(runtime.device):
            prepared = owner.prepare(sample)
        previous = None
        for j, state in enumerate(states):
            leaves = {k: v.detach().to(runtime.device).requires_grad_() for k, v in state.items()}
            with autocast(runtime.device):
                prediction = torch.func.functional_call(owner, {'policy.' + k: v for k, v in leaves.items()},
                                                         (sample, prepared), strict=False)
                ell = per_query_loss(prediction, sample)
                regression = torch.zeros_like(ell) if previous is None else torch.relu(ell - previous)
                objective = (alphas[j] * ell + (0.2 / revisions * regression if revisions else 0.)).sum() / count / 4
                gradients = torch.autograd.grad(objective, tuple(leaves.values()))
            add_credit(credits[j], leaves, gradients)
            losses[j] += float(ell.detach().sum()) / count
            if previous is not None:
                keep += float(regression.detach().sum()) / count / revisions
            previous = ell.detach()
    return dict(cotangents=credits, stage_losses=losses, keep=keep,
                weighted_loss=sum(a * b for a, b in zip(alphas, losses)) + .2 * keep)


def pg_credit(runtime, final_state, task, queries, advantages, *, microbatch=16):
    if len(queries) != 2 or len(advantages) != 2:
        raise ValueError('each condition has exactly two independent SDE query labels')
    credit = {k: torch.zeros_like(v, dtype=torch.float32, device=runtime.device)
              for k, v in final_state.items()}
    decisions, transitions = 0, 0
    for query, advantage in zip(queries, advantages, strict=True):
        records, replans = query['reservoir'], query['row']['replans']
        if replans < 1 or len(records) != min(16, replans):
            raise ValueError('query reservoir lost actual decision count')
        points = [(record['raw'], transition) for record in records for transition in record['transitions']]
        decisions += len(records)
        transitions += len(points)
        if float(advantage) == 0:
            continue  # An exactly zero score coefficient has exactly zero credit.
        for start in range(0, len(points), microbatch):
            block = points[start:start + microbatch]
            rows = [processed(runtime, raw, task['language']) for raw, _ in block]
            batch = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
            with autocast(runtime.device):
                owner = NativeVelocity(runtime.policy, batch, capture_phi=False)
            leaves = {k: v.detach().to(runtime.device).requires_grad_() for k, v in final_state.items()}
            cotangent = torch.cat([score_cotangent(t, advantage, replans=replans, retained=len(records))
                                   for _, t in block]).to(runtime.device)
            z = torch.cat([t.z for _, t in block]).to(runtime.device)
            tau = torch.tensor([t.tau for _, t in block], device=runtime.device)
            with autocast(runtime.device):
                prediction = torch.func.functional_call(owner, {'policy.' + k: v for k, v in leaves.items()},
                    (z, tau), strict=False)
                gradients = torch.autograd.grad(prediction, tuple(leaves.values()), grad_outputs=cotangent)
            add_credit(credit, leaves, gradients)
    return dict(cotangent=credit, reservoir_decisions=decisions, transitions=transitions,
                advantages=list(map(float, advantages)), returns=[int(q['row']['success']) for q in queries])


def surrogate(states, fm, pg=None, beta=0.):
    """Shared Compiler VJP; this scalar is a cotangent carrier, not an FM loss."""
    result = None
    for j, state in enumerate(states):
        for name, cotangent in fm[j].items():
            value = (state[name].float() * cotangent).sum()
            result = value if result is None else result + value
    if pg is not None and beta:
        for name, cotangent in pg.items():
            value = beta * (states[-1][name].float() * cotangent).sum()
            result = value if result is None else result + value
    return result


def pg_surrogate(final, credit):
    return sum((final[name].float() * value).sum() for name, value in credit.items())
