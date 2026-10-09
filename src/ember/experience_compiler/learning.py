"""Event semi-gradient updates for the single functional revision Compiler."""
from __future__ import annotations

import time

import torch

from .credit import fm_credit, keep_credit, pg_credit


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = name.split('.')[0]
            groups[group] = groups.get(group, 0.) + float(parameter.grad.float().square().sum())
    return {key: value**.5 for key, value in groups.items()}


def fresh_optimizer(model, *, stage):
    """RL always starts with a new AdamW/scheduler, without supervised moments."""
    if stage not in {'supervised', 'reinforcement'}:
        raise ValueError('the registered shared stages are supervised and reinforcement')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    return optimizer, scheduler


def _backward_events(runtime, prepared, components, *, microbatch, stage):
    records, credits = [], []
    for index, item in enumerate(prepared):
        credit, metrics = {}, {}
        for component, rows in components.items():
            row = rows[index]
            if row['cotangent'].keys() != item['outgoing'].keys():
                raise ValueError('outer credit must cover the complete outgoing factors')
            for name, value in row['cotangent'].items():
                if name in credit:
                    credit[name].add_(value)
                else:
                    credit[name] = value
            metrics[component] = {key: value for key, value in row.items() if key != 'cotangent'}
        credits.append(credit)
        records.append(dict(stage=stage, components=metrics))
    started = time.monotonic()
    runtime.backward_revisions(prepared, credits, microbatch=microbatch)
    seconds = time.monotonic() - started
    for record in records:
        record['group_revision_backward_seconds'] = seconds
        record['group_events'] = len(prepared)
    return records


def supervised_backward(runtime, prepared, *, microbatch, revision_microbatch=None, condition_weight=None):
    """Accumulate shared FM+keep gradients; caller owns zero/reduction/step.

    Prepared events retain their actual incoming, complete experience, chosen
    functional support and uniformly selected successful-state keep_support.
    No gradient goes through behavior, incoming or preceding editing events.
    """
    started = time.monotonic()
    fm = fm_credit(runtime, [item['incoming'] for item in prepared],
        [item['outgoing'] for item in prepared], [item['batch'] for item in prepared],
        seeds=[item['seed'] for item in prepared], microbatch=microbatch, condition_weight=condition_weight)
    fm_seconds = time.monotonic() - started
    started = time.monotonic()
    keep = keep_credit(runtime, [item['incoming'] for item in prepared],
        [item['outgoing'] for item in prepared], [item.get('keep_support', {}) for item in prepared],
        microbatch=microbatch, condition_weight=condition_weight)
    keep_seconds = time.monotonic() - started
    records = _backward_events(runtime, prepared, dict(FM=fm, keep=keep),
        microbatch=microbatch if revision_microbatch is None else revision_microbatch, stage='supervised')
    return dict(records=records, FM_seconds=fm_seconds, keep_seconds=keep_seconds)


def reinforcement_backward(runtime, prepared, episodes, *, microbatch,
                           revision_microbatch=None, condition_weight=None):
    """Accumulate negative real-return score+keep, with no FM or trajectory fit.

    Episodes were collected once with this exact outgoing/phi version. Any
    consumer replay precedes the optimizer step; it is not another PG round.
    """
    started = time.monotonic()
    pg = pg_credit(runtime, [item['outgoing'] for item in prepared], episodes,
                   microbatch=microbatch, condition_weight=condition_weight)
    pg_seconds = time.monotonic() - started
    started = time.monotonic()
    keep = keep_credit(runtime, [item['incoming'] for item in prepared],
        [item['outgoing'] for item in prepared], [item.get('keep_support', {}) for item in prepared],
        microbatch=microbatch, condition_weight=condition_weight)
    keep_seconds = time.monotonic() - started
    records = _backward_events(runtime, prepared, dict(PG=pg, keep=keep),
        microbatch=microbatch if revision_microbatch is None else revision_microbatch, stage='reinforcement')
    return dict(records=records, PG_seconds=pg_seconds, keep_seconds=keep_seconds)


def finish_update(runtime, optimizer, scheduler, *, world_size=1):
    """Reduce already globally weighted event credits, clip1 and step once."""
    parameters = tuple(runtime.compiler.parameters())
    if world_size > 1:
        from ember.writer.replay import sum_writer_gradients
        sum_writer_gradients(parameters, world_size=world_size, bucket_bytes=64 * 1024**2)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    groups = gradient_groups(runtime.compiler)
    started = time.monotonic()
    optimizer.step()
    scheduler.step()
    return dict(combined_gradient_norm=float(norm), gradient_groups=groups,
                optimizer_seconds=time.monotonic() - started, learning_rate=scheduler.get_last_lr())
