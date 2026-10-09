"""Actual native derivative checks and bounded complete learning consumers."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.writer.runtime import autocast
from .contract import SEED, condition_seed
from .data import Event, QueryData
from .execution import factor_jvp, factor_vjp, slice_batch
from .interaction import Chain, Runner


def measure(runtime, label, function):
    from .run import check_budget
    check_budget(runtime.profile_root)
    if time.monotonic() >= runtime.profile_deadline:
        raise RuntimeError('bounded compute observation line reached')
    gc.collect()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.monotonic()
    try:
        value = function()
        torch.cuda.synchronize(runtime.device)
        row = dict(label=label, valid=True, seconds=time.monotonic() - started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(runtime.device) / 1024**3)
    except torch.cuda.OutOfMemoryError:
        runtime.compiler.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        row, value = dict(label=label, valid=False, OOM=True, seconds=time.monotonic() - started), None
    append_jsonl(runtime.profile_root / 'components.jsonl', row)
    print(row, flush=True)
    return row, value


def load_original(runtime, condition):
    """Use exactly the original event's incoming and full cumulative real E."""
    started = time.monotonic()
    metadata = read_json(condition['source_record'])
    event = next(e for e in metadata['events'] if e['endpoint'] == condition['endpoint'])
    if (event['behavior_version'] != condition['behavior_version'] or
            event['incoming'] != condition['original_incoming']):
        raise ValueError('original parameter/behavior event provenance changed')
    chain = Chain.from_record(torch.load(condition['source_experience'], map_location='cpu', weights_only=False))
    incoming = load_file(condition['incoming'], device=str(runtime.device))
    support = chain.support(runtime, condition['language'], endpoint=condition['endpoint'])
    experience = chain.experience(condition['endpoint'], runtime=runtime)
    teacher = runtime.teacher(condition['task_id'], condition['teacher_demo'])
    keep = chain.support(runtime, condition['language'], endpoint=condition['endpoint'], successful_only=True)
    return dict(condition=condition, incoming=incoming, experience=experience, teacher=teacher,
                support=support, keep_support=keep, io_seconds=time.monotonic() - started,
                teacher_cost=dict(runtime.last_teacher_cost))


def verify_native(runtime, contract):
    item = load_original(runtime, contract['conditions'][0])
    incoming, support = item['incoming'], item['support']
    # One actual point is sufficient for a numerical AD/binding oracle; the
    # complete M<=16 edit and credit consumers are measured separately.
    support = dict(indices=support['indices'][:1], batch=slice_batch(support['batch'], 0, 1),
                   noise=support['noise'][:1])
    generator = torch.Generator(device=runtime.device).manual_seed(SEED)
    directions = {key: torch.randn(value.shape, generator=generator, device=runtime.device) * runtime.compiler.rms[i]
                  for i, (key, value) in enumerate((k, incoming[k]) for k in runtime.compiler.keys)}
    row, tangent = measure(runtime, 'actual_76factor_tenstep_JVP',
        lambda: factor_jvp(runtime, incoming, support, directions, microbatch=1))
    if not row['valid']:
        raise RuntimeError('actual ten-step JVP did not fit')
    q = tangent / tangent.float().square().mean().sqrt().clamp_min(1e-12)
    row, credit = measure(runtime, 'actual_76factor_checkpointed_tenstep_VJP',
        lambda: factor_vjp(runtime, incoming, support, q, microbatch=1))
    if not row['valid']:
        raise RuntimeError('actual ten-step VJP did not fit')
    lhs = float((tangent.double() * q.double()).sum())
    rhs = sum(float((directions[key].double() * value.double()).sum()) for key, value in credit.items())
    relative = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    eps = .01
    def response(sign):
        state = {k: v + sign * eps * directions[k] for k, v in incoming.items()}
        with torch.no_grad():
            return runtime.native_actions([state], support['batch'], support['noise'])
    row, finite = measure(runtime, 'actual_bounded_directional_difference',
        lambda: (response(1) - response(-1)) / (2 * eps))
    error = float((finite - tangent).float().square().mean().sqrt() /
                  tangent.float().square().mean().sqrt().clamp_min(1e-12))
    # Only one directional scalar checks binding against uncheckpointed AD;
    # no per-tensor equality or bitwise scan is part of this contract.
    def ordinary_vjp():
        leaves = {k: v.detach().requires_grad_() for k, v in incoming.items()}
        output = runtime.native_actions([leaves], support['batch'], support['noise'], checkpointed=False)
        gradients = torch.autograd.grad(output, tuple(leaves.values()), q.to(output))
        return sum(float((directions[k].double() * g.double()).sum())
                   for k, g in zip(leaves, gradients, strict=True))
    row, uncheckpointed = measure(runtime, 'actual_uncheckpointed_binding_direction', ordinary_vjp)
    binding = None if not row['valid'] else abs(uncheckpointed - rhs) / max(abs(rhs), 1e-12)
    result = dict(task_id=0, teacher_demo=29, actual_noise_seed=item['support']['noise_seeds'][0],
        full_factor_count=len(credit), native_latent=[50, 32], native_flow_steps=10, response=[5, 7],
        lhs=lhs, rhs=rhs, adjoint_relative_error=relative, finite_direction_relative_RMS=error,
        checkpoint_binding_relative_error=binding, finite_difference_only_for_numerical_oracle=True,
        code_git=runtime.profile_git)
    write_json_atomic(runtime.profile_root / 'native_derivatives.json', result)
    if len(credit) != 76 or relative > .1 or binding is None or binding > .05:
        raise RuntimeError(f'actual native derivative/binding check failed: {result}')
    # BF16 finite differencing has quantization noise; retain its measured
    # error for interpretation instead of silently using FD as a gradient.
    return result


def event_queries(data, condition):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0xF00, condition['position']]))
    demos = rng.choice([d for d in range(50) if d != condition['teacher_demo']], 7, replace=False)
    queries = []
    for demo in demos:
        length = int(data.tasks[condition['task_id']].episode_lengths[int(demo)]) - 1
        for interval in range(4):
            queries.append((int(demo), int(rng.integers(length * interval // 4, length * (interval + 1) // 4))))
    return Event(0, condition['position'], condition['task_id'], condition['teacher_demo'], tuple(queries),
        False, condition_seed(condition['position'], domain=0xF00), condition['condition_id'],
        'legal_original_bootstrap_disposable', condition['endpoint'], condition['original_incoming'],
        str(Path(condition['source_record']).parent), condition['behavior_version'])


def profile(runtime, contract, args):
    raise RuntimeError('complete consumer profile must be integrated before launch')
