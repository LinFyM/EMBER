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
        row = dict(label=label, valid=True, code_git=runtime.profile_git, unix=time.time(), seconds=time.monotonic() - started,
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
    metadata = read_json(Path(condition['source_record']))
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
    fp32 = None
    if error > .2:
        row, fp32 = measure(runtime, 'actual_FP32_bounded_directional_oracle',
                            lambda: numerical_fp32(runtime, incoming, support, directions))
        if fp32 is None or fp32['finite_direction_relative_RMS'] > .1:
            raise RuntimeError(f'continuous native directional check failed: {fp32}')
    result = dict(task_id=0, teacher_demo=29, actual_noise_seed=item['support']['noise_seeds'][0],
        full_factor_count=len(credit), native_latent=[50, 32], native_flow_steps=10, response=[5, 7],
        lhs=lhs, rhs=rhs, adjoint_relative_error=relative, finite_direction_relative_RMS=error,
        checkpoint_binding_relative_error=binding, finite_difference_only_for_numerical_oracle=True,
        FP32_numerical_oracle=fp32, code_git=runtime.profile_git)
    write_json_atomic(runtime.profile_root / f'native_derivatives_{runtime.profile_git[:8]}.json', result)
    write_json_atomic(runtime.profile_root / 'native_derivatives.json', result)
    if len(credit) != 76 or relative > .1 or binding is None or binding > .05:
        raise RuntimeError(f'actual native derivative/binding check failed: {result}')
    # BF16 finite differencing has quantization noise; retain its measured
    # error for interpretation instead of silently using FD as a gradient.
    return result


def numerical_fp32(runtime, incoming, support, direction):
    """One floating-point oracle for a measured BF16 small-perturbation mismatch.

    Same weights, inputs and direction; temporarily promote frozen arithmetic.
    This is not a production gradient or changed learning/policy consumer.
    """
    from .execution import native_actions
    original = runtime.native_actions
    parameters = [(parameter, parameter.dtype) for parameter in runtime.policy.parameters()]
    buffers = [(name, value.dtype) for name, value in runtime.policy.named_buffers()]
    precision = torch.get_float32_matmul_precision()
    def precise(states, batch, noise, **kwargs):
        with torch.autocast('cuda', enabled=False):
            return native_actions(runtime, states, batch, noise, **kwargs)
    try:
        torch.set_float32_matmul_precision('highest')
        runtime.policy.float()
        runtime.native_actions = precise
        tangent = factor_jvp(runtime, incoming, support, direction, microbatch=1)
        epsilon = .01
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                state = {key: value + sign * epsilon * direction[key] for key, value in incoming.items()}
                values.append(runtime.native_actions([state], support['batch'], support['noise']))
        finite = (values[0] - values[1]) / (2 * epsilon)
        error = float((finite - tangent).square().mean().sqrt() / tangent.square().mean().sqrt().clamp_min(1e-12))
        return dict(finite_direction_relative_RMS=error, epsilon=epsilon, arithmetic='FP32',
                    same_actual_input_and_noise=True, no_optimizer=True)
    finally:
        runtime.native_actions = original
        for parameter, dtype in parameters:
            parameter.data = parameter.data.to(dtype=dtype)
        for name, dtype in buffers:
            parent, _, field = name.rpartition('.')
            module = runtime.policy.get_submodule(parent) if parent else runtime.policy
            module._buffers[field] = module._buffers[field].to(dtype=dtype)
        torch.set_float32_matmul_precision(precision)


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



def _compile(runtime, prepared, label):
    def run():
        outgoing = runtime.edit_many(prepared)
        for item, state in zip(prepared, outgoing, strict=True):
            item['outgoing'] = state
        return dict(runtime.last_revision_cost)
    row, costs = measure(runtime, label, run)
    if not row['valid']:
        raise RuntimeError('complete four-condition edit did not fit')
    return dict(measurement=row, group=costs)


def _best(rows, label):
    valid = [r for r in rows if r['valid']]
    if not valid:
        raise RuntimeError(f'no usable physical configuration: {label}')
    return min(valid, key=lambda r: r['seconds'])['microbatch']


def profile_physical(runtime, prepared, args):
    from .credit import fm_credit
    from .learning import gradient_groups
    rows = []
    for chunk in (4, 8, 16, 32, 64):
        runtime.support_microbatch = chunk
        row, _ = measure(runtime, f'four_complete_edits_support_micro{chunk}', lambda: runtime.edit_many(prepared))
        rows.append(dict(row, microbatch=chunk))
        if not row['valid']:
            break
    support_chunk = _best(rows, 'support')
    runtime.support_microbatch = support_chunk
    compiled = _compile(runtime, prepared, 'initial_four_complete_edits')
    fm_rows, latest = [], None
    for chunk in (28, 56, 112):
        row, values = measure(runtime, f'outer_FM_112queries_micro{chunk}', lambda:
            fm_credit(runtime, [p['incoming'] for p in prepared], [p['outgoing'] for p in prepared],
                [p['batch'] for p in prepared], seeds=[p['seed'] for p in prepared], microbatch=chunk))
        fm_rows.append(dict(row, microbatch=chunk))
        if values is not None:
            latest = values
        if not row['valid']:
            break
    if latest is None:
        raise RuntimeError('full four-event FM credit did not fit')
    fm_chunk = _best(fm_rows, 'FM')
    jvp_rows = []
    credits = [row['cotangent'] for row in latest]
    for chunk in (4, 8, 16, 32, 64):
        runtime.compiler.zero_grad(set_to_none=True)
        row, _ = measure(runtime, f'real_FM_shared_adjoint_micro{chunk}', lambda:
            runtime.backward_revisions(prepared, credits, microbatch=chunk))
        row.update(microbatch=chunk, gradient_groups=gradient_groups(runtime.compiler))
        jvp_rows.append(row)
        if not row['valid']:
            break
    jvp_chunk = _best(jvp_rows, 'shared adjoint')
    runtime.compiler.zero_grad(set_to_none=True)
    return dict(support_microbatch=support_chunk, FM_microbatch=fm_chunk,
        adjoint_microbatch=jvp_chunk, support=rows, FM=fm_rows, adjoint=jvp_rows, initial_compile=compiled,
        enlargement_boundary='per-event M<=16, four-event supports and exactly112 FM queries; useful physical sizes measured')


def _save_stage(runtime, optimizer, scheduler, update, stage):
    path = runtime.profile_root / 'implementation_state.pt'
    partial = path.with_suffix('.partial')
    torch.save(dict(model=runtime.compiler.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), FM_updates=update, stage=stage,
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
        disposable_implementation_validation=True, code_git=runtime.profile_git), partial)
    partial.replace(path)


def collect_queries(runtime, contract, prepared, args):
    """Exactly sixteen new episodes; completed episodes survive engineering exit."""
    root = runtime.profile_root / 'queries'
    root.mkdir(exist_ok=True)
    if (root / 'partial').exists():
        raise RuntimeError('preserved partial queries require a scope decision; do not repeat episodes')
    tasks = {task['global_task_id']: task for task in contract['environment_contract']['tasks']}
    requests, results = [], {}
    for item in prepared:
        condition = item['condition']
        for query, state in enumerate(condition['query_state_ids']):
            roots = condition['roots'][query]
            for arm in ('outgoing', 'incoming'):
                identity = f"{condition['condition_id']}_query{query}_{arm}"
                path = root / f'{identity}.pt'
                if path.exists():
                    results[identity] = torch.load(path, weights_only=False, map_location='cpu')
                    continue
                requests.append(dict(identity=identity, task=tasks[condition['task_id']],
                    state=item[arm], state_id=state, arm=arm, capture=True, reservoir=16,
                    reservoir_seed=roots[f'{arm}_reservoir'], exploration_sigma=.1,
                    exploration_root=roots[f'{arm}_exploration'], noise_root=roots[f'{arm}_policy'],
                    environment_root=roots['environment'], query=query, condition_id=condition['condition_id']))
    runner = Runner(runtime, contract['environment_contract'], args.physical_gpu,
                    slot_batch=min(args.slot_batch, 16))
    started = time.monotonic()
    try:
        for result in runner.final_requests(requests):
            request = result['request']
            saved = dict(records=result['trajectory'], total_replans=result['total_replans'],
                success=bool(result['row']['success']), row=result['row'], language=request['task']['language'],
                state_id=request['state_id'], arm=request['arm'], query=request['query'],
                condition_id=request['condition_id'], identity=request['identity'],
                phi_version='disposable_FM2', code_git=runtime.profile_git)
            torch.save(saved, root / f"{request['identity']}.pt")
            results[request['identity']] = saved
    except BaseException:
        runner.preserve_partial(root / 'partial')
        raise
    finally:
        runner.close()
        write_json_atomic(root / 'execution.json', dict(new_environment_steps=runner.total_environment_steps,
            seconds=time.monotonic() - started, components=runner.components, episode_count=len(results)))
    if len(results) != 16 or sum(r['row']['environment_steps'] for r in results.values()) > 5440:
        raise ValueError('actual queries exceeded their registered episode/environment scope')
    episodes = []
    for item in prepared:
        queries = []
        for query in range(2):
            prefix = f"{item['condition']['condition_id']}_query{query}_"
            outgoing, incoming = results[prefix + 'outgoing'], results[prefix + 'incoming']
            queries.append(dict(**outgoing, baseline_success=incoming['success']))
        episodes.append(queries)
    return episodes, dict(total_environment_steps=sum(r['row']['environment_steps'] for r in results.values()),
        actual_query_episodes=16, seconds=time.monotonic() - started,
        returns={key: int(value['success']) for key, value in results.items()}, components=runner.components)


def profile(runtime, contract, args):
    from .learning import fresh_optimizer, finish_update, supervised_backward, reinforcement_backward
    prepared, data = [], QueryData()
    root = runtime.profile_root
    try:
        for condition in contract['conditions']:
            row, item = measure(runtime, f"original_input_task{condition['task_id']}",
                                lambda: load_original(runtime, condition))
            if item is None:
                raise RuntimeError('actual original condition failed to load')
            event = event_queries(data, condition)
            item.update(batch=data.query_batch(event, runtime.processor), seed=event.seed, event=event.as_dict())
            prepared.append(item)
        write_json_atomic(root / 'event_stream.json', dict(events=[item['event'] for item in prepared],
            bootstrap_origin='original_chi360_actual_event_not_new_model_experience'))
        optimizer, scheduler = fresh_optimizer(runtime.compiler, stage='supervised')
        state_path = root / 'implementation_state.pt'
        update, resumed = 0, None
        if state_path.exists():
            resumed = torch.load(state_path, map_location=runtime.device, weights_only=False)
            runtime.compiler.load_state_dict(resumed['model'])
            update = resumed['FM_updates']
            if resumed['stage'] == 'reinforcement_complete':
                return dict(FM_updates=2, PG_updates=1, already_computed=True,
                            query=read_json(root / 'query_summary.json'), no_formal_learning=True)
            optimizer.load_state_dict(resumed['optimizer'])
            scheduler.load_state_dict(resumed['scheduler'])
        if update == 0:
            physical = profile_physical(runtime, prepared, args)
            write_json_atomic(root / 'physical_selection.json', physical)
        else:
            physical = read_json(root / 'physical_selection.json')
        runtime.support_microbatch = physical['support_microbatch']
        while update < 2:
            compile_cost = _compile(runtime, prepared, f'FM{update + 1}_four_complete_edits')
            optimizer.zero_grad(set_to_none=True)
            row, consumer = measure(runtime, f'FM{update + 1}_complete_shared_update', lambda:
                supervised_backward(runtime, prepared, microbatch=physical['FM_microbatch'],
                                    revision_microbatch=physical['adjoint_microbatch']))
            if consumer is None:
                raise RuntimeError('full supervised consumer failed to fit selected physical batch')
            step = finish_update(runtime, optimizer, scheduler)
            update += 1
            _save_stage(runtime, optimizer, scheduler, update, 'supervised')
            append_jsonl(root / 'updates.jsonl', dict(stage='FM', update=update, consumer=consumer,
                step=step, measurement=row, compile=compile_cost))
        compiled = _compile(runtime, prepared, 'FM2_frozen_query_edit_four_conditions')
        episodes, query = collect_queries(runtime, contract, prepared, args)
        write_json_atomic(root / 'query_summary.json', query)
        # New RL moments, no FM; all PG replays occur before this sole step.
        optimizer, scheduler = fresh_optimizer(runtime.compiler, stage='reinforcement')
        pg_rows, chosen = [], None
        for chunk in (4, 8, 16, 32, 64):
            optimizer.zero_grad(set_to_none=True)
            row, consumer = measure(runtime, f'real_PG_keep_shared_micro{chunk}', lambda:
                reinforcement_backward(runtime, prepared, episodes, microbatch=chunk,
                                       revision_microbatch=physical['adjoint_microbatch']))
            pg_rows.append(dict(row, microbatch=chunk, adjoint=dict(runtime.last_adjoint_cost)))
            if consumer is not None:
                chosen = consumer
            if not row['valid']:
                break
        pg_chunk = _best(pg_rows, 'actual score/keep')
        optimizer.zero_grad(set_to_none=True)
        row, consumer = measure(runtime, f'PG1_complete_shared_update_micro{pg_chunk}', lambda:
            reinforcement_backward(runtime, prepared, episodes, microbatch=pg_chunk,
                                   revision_microbatch=physical['adjoint_microbatch']))
        if consumer is None:
            raise RuntimeError('complete actual PG+keep consumer did not fit')
        step = finish_update(runtime, optimizer, scheduler)
        _save_stage(runtime, optimizer, scheduler, 2, 'reinforcement_complete')
        append_jsonl(root / 'updates.jsonl', dict(stage='PG', update=1, consumer=consumer,
            step=step, measurement=row, compile=compiled, physical_profiles=pg_rows,
            adjoint=dict(runtime.last_adjoint_cost)))
        return dict(FM_updates=2, PG_updates=1, disposable=True, query=query,
                    physical=physical, PG_physical=pg_rows, no_formal_learning=True)
    finally:
        data.close()
