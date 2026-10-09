"""Bounded actual-condition/LoRA/environment and shared-FM execution profile."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors.torch import load_file

from ember.operator_writer.native import read_frozen_teacher_features
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization_workers import _configure_device
from ember.writer.runtime import autocast
from ember.writer.function_credit import NativeFlowPrediction, flow_sample

from .contract import condition_seed, learning_environment
from .credit import fm_credit, surrogate
from .data import Event, QueryData, collection_conditions
from .interaction import Runner
from .learning import gradient_groups
from .runtime import Runtime
from .storage import save_condition


ENV_LIMIT = 4096
OLD_PANEL = Path('/data1/user/ymdai/ember_runs/experience_conditioned_compiler_20261009/evaluation/meta54')


def measure(runtime, label, function):
    if time.monotonic() >= runtime.profile_deadline:
        raise RuntimeError('actual profile2GPUh boundary reached')
    runtime.compiler.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.monotonic()
    try:
        result = function()
        torch.cuda.synchronize(runtime.device)
        record = dict(label=label, valid=True, seconds=time.monotonic() - started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(runtime.device) / 1024**3,
            gradient_groups=gradient_groups(runtime.compiler))
    except torch.cuda.OutOfMemoryError:
        runtime.compiler.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        record, result = dict(label=label, valid=False, OOM=True, seconds=time.monotonic() - started), None
    print(record, flush=True)
    return record, result


def best(rows, category):
    candidates = [r for r in rows if r.get('category') == category and r['valid']]
    if not candidates:
        raise RuntimeError(f'no valid actual consumer for {category}')
    if category == 'slot':
        return max(candidates, key=lambda row: row['environment_steps_per_second'])['value']
    return min(candidates, key=lambda row: row['seconds'])['value']


def profile_event(data, condition, endpoint, position):
    """Profile-only query metadata; never masquerades as a retained behavior pool."""
    task = data.tasks[condition['task_id']]
    rng = np.random.default_rng(np.random.SeedSequence([20261009, 0xF00, position]))
    demos = rng.choice([d for d in range(50) if d != condition['teacher_demo']], 7, replace=False)
    queries = []
    for demo in demos:
        length = task.episode_lengths[int(demo)] - 1
        for interval in range(4):
            queries.append((int(demo), int(rng.integers(length * interval // 4, length * (interval + 1) // 4))))
    return Event(0, position, condition['task_id'], condition['teacher_demo'], tuple(queries), False,
        condition_seed(position, domain=0xF00), condition['condition_id'], 'disposable_profile',
        endpoint, 'profile_actual_incoming', '', 'archived_complete_factors_actual_new_practice')


def replay(runtime, prepared, credits):
    for item, credit in zip(prepared, credits, strict=True):
        outgoing = runtime.edit(item['incoming'], item['teacher'], item['experience'])
        surrogate(outgoing, credit['cotangent']).backward()


def native_consumer_check(runtime, prepared):
    """One actual nonheld query checks indexed hooks against the native owner."""
    item = prepared[1]
    batch = {k: v[:1] if isinstance(v, torch.Tensor) and v.ndim and len(v) == 28 else v
             for k, v in item['batch'].items()}
    owner = NativeFlowPrediction(runtime.policy)
    with torch.no_grad(), autocast(runtime.device):
        sample = flow_sample(runtime.policy, batch, seed=item['event'].seed, device=runtime.device,
                             random_batch=28, offset=0)
        prefix = owner.prepare(sample)
        assignment = torch.tensor([1], device=runtime.device)
        with runtime.execution.activate([runtime.mt, item['incoming']], batch_indices=assignment):
            actual = owner(sample, prefix)
        reference = torch.func.functional_call(owner, {'policy.' + k: v.to(runtime.device)
            for k, v in item['incoming'].items()}, (sample, prefix), strict=False)
    difference = (actual[..., :7].float() - reference[..., :7].float()).square().mean().sqrt()
    relative = float(difference / reference[..., :7].float().square().mean().sqrt().clamp_min(1e-6))
    if relative > .02:
        raise ValueError(f'indexed actual native FM diverged from its functional consumer: {relative}')
    return dict(relative_velocity_RMS=relative, actual_nonheld_task=item['event'].task_id,
                query_count=1, allowance='ordinary physical BF16/reduction differences')


def _profile_shared(runtime, data, actual, rows, output):
    prepared, descriptors = [], []
    for position, result in enumerate(actual[:4]):
        c, chain = result['request']['condition'], result['chain']
        event = profile_event(data, c, chain.endpoints[-1], position)
        teacher = runtime.teacher(c['task_id'], c['teacher_demo'])
        incoming, experience = chain.states[0], chain.experience(chain.endpoints[-1])
        with torch.no_grad():
            outgoing = runtime.edit(incoming, teacher, experience)
        prepared.append(dict(incoming=incoming, outgoing=outgoing, teacher=teacher,
                             experience=experience, batch=data.query_batch(event, runtime.processor), event=event))
        descriptors.append(dict(event=event.as_dict(), chain_path=result['request']['record_path'],
            incoming_path=result['request'].get('source', 'MT')))
    write_json_atomic(output / 'FM_events.json', dict(events=descriptors, actual_incoming=True,
        optimizer_updates=0, retained_training_pool=False))
    rows.append(dict(label='actual_indexed_native_FM_consumer', valid=True, **native_consumer_check(runtime, prepared)))
    latest = None
    for size, micro in ((1, 14), (1, 28), (4, 28), (4, 56), (4, 112)):
        group = prepared[:size]
        record, credits = measure(runtime, f'paired_FM_{size}events_micro{micro}', lambda:
            fm_credit(runtime, [p['incoming'] for p in group], [p['outgoing'] for p in group],
                [p['batch'] for p in group], seeds=[p['event'].seed for p in group], microbatch=micro))
        record.update(category=f'FM_events{size}', value=micro, queries=28 * size)
        if credits is not None:
            record.update(incoming_losses=[c['incoming_loss'] for c in credits],
                          outgoing_losses=[c['outgoing_loss'] for c in credits])
            latest = credits if size == 4 else latest
            if any(abs(c['paired_difference']) > max(.002, .02 * c['incoming_loss']) for c in credits):
                raise ValueError('fresh zero edit changed the actual paired native FM consumer')
        rows.append(record)
    if latest is None:
        raise RuntimeError('full logical four-event physical FM profile did not fit')
    record, _ = measure(runtime, 'shared_full_edit_VJP_four_actual_events', lambda: replay(runtime, prepared, latest))
    rows.append(record)
    if record['gradient_groups'].get('update_out', 0.) == 0:
        raise ValueError('zero edit failed to receive native FM credit at its live output projection')
    # At the fresh zero edit, shared decoder derivatives cancel; this is not a
    # claim that every upstream module receives nonzero gradients on update1.
    for chunk in (16384, 65536, 131072):
        runtime.compiler.decoder.chunk_rows = chunk
        record, _ = measure(runtime, f'decoder_rows{chunk}', lambda: replay(runtime, prepared, latest))
        rows.append(dict(record, category='decoder', value=chunk))
    runtime.compiler.decoder.chunk_rows = best(rows, 'decoder')
    for chunk in (16, 64, 256):
        runtime.compiler.encoder.chunk = chunk
        record, _ = measure(runtime, f'experience_chunk{chunk}', lambda: replay(runtime, prepared, latest))
        rows.append(dict(record, category='experience', value=chunk))
    runtime.compiler.encoder.chunk = best(rows, 'experience')
    return prepared, latest


def _profile_teaching(runtime, data, prepared, credits, rows):
    # Only allowed metadata/video RGB; no held labels or query action condition.
    _, task_id, demo = max((length, task_id, demo) for task_id, task in data.tasks.items()
                          for demo, length in enumerate(task.episode_lengths))
    tasks, store = runtime._store('train')
    raw = store.load(task_id, demo)
    tokens, mask, _ = runtime.tokenizer([tasks[task_id].authority.language])
    for chunk in (16, 64, 128):
        def native():
            with autocast(runtime.device):
                return read_frozen_teacher_features(runtime.policy, runtime.mt, runtime.probe,
                    (torch.from_numpy(raw.frames).to(runtime.device), torch.from_numpy(raw.frame_indices), tokens, mask),
                    frame_chunk=chunk)
        record, features = measure(runtime, f'native_teacher_frames{chunk}', native)
        rows.append(dict(record, category='native_frames', value=chunk, frames=len(raw.frames)))
        del features
    for chunk in (16, 64, 128):
        runtime.compiler.reader.chunk = chunk
        record, _ = measure(runtime, f'learned_teacher_frames{chunk}', lambda: replay(runtime, prepared, credits))
        rows.append(dict(record, category='learned_frames', value=chunk))
    runtime.compiler.reader.chunk = best(rows, 'learned_frames')
    runtime.native_frame_chunk = best(rows, 'native_frames')
    return dict(task_id=task_id, teacher_demo=demo, sampled_frames=len(raw.frames), raw_frames=raw.raw_frame_count)


def _archived_profile_items(data, conditions):
    archived = read_json(OLD_PANEL / 'results.json')['conditions']
    # Complete existing nonheld LoRAs are allowed for physical profiling;
    # their old incomplete intermediate chains are never new training data.
    items = []
    for old in archived:
        c = old['condition']
        if c['task_id'] not in data.tasks:
            raise ValueError('batch profile crossed the nonheld allowlist')
        authority = next(x for x in conditions if x['task_id'] == c['task_id'])
        state = load_file(str(OLD_PANEL / old['artifact_root'] / old['weights']['end']))
        items.append(dict(condition={**authority, 'condition_id': f"profile_{c['condition_id']}"}, state=state,
                          source=str(OLD_PANEL / old['artifact_root'] / old['weights']['end'])))
    return items


def _profile_control(runtime, data, args, rows, runners, all_runners):
    all_steps, actual = 0, []
    contract = learning_environment(asset_root=args.asset_root)
    tasks = {t['global_task_id']: t for t in contract['tasks']}
    conditions = collection_conditions('pool0', asset_root=args.asset_root)
    items = _archived_profile_items(data, conditions)
    eight = [items[index] for index in range(0, 16, 2)]
    for B, rep in ((4, 0), (8, 0), (16, 0), (16, 1)):
        used = items if B == 16 else eight
        budget = 60 if B == 16 else 70
        if all_steps + len(used) * budget + 1024 > ENV_LIMIT:
            raise RuntimeError('profile preallocation would exceed4096 actual steps')
        runner = next((r for r in runners if r.slot_batch == B), None)
        if runner is None:
            for previous in runners:
                previous.close()
            runners.clear()
            runner = Runner(runtime, contract, args.physical_gpu, slot_batch=B)
            runners.append(runner)
            all_runners.append(runner)
        before = runner.total_environment_steps
        before_components = deepcopy(runner.components)
        def execute():
            requests = [dict(kind='fixed', **item, task=tasks[item['condition']['task_id']],
                role='train', fixed_behavior_version='profile_archived_complete_actual_factor',
                step_budget=budget) for item in used]
            result = list(runner.run(requests))
            runtime.io.flush()
            return result
        record, result = measure(runtime, f'actual_practice_B{B}_rep{rep}', execute)
        delta = runner.total_environment_steps - before
        all_steps += delta
        components = {k: v - before_components[k] for k, v in runner.components.items() if k != 'physical_batch_histogram'}
        record.update(category='slot', value=B, rep=rep, environment_steps=delta,
            conditions=len(used), controls=sum(e['steps'] for x in result for e in x['chain'].episodes),
            components=components, actual_batch_histogram=dict(runner.components['physical_batch_histogram']),
            native_actor_reads=0, real_independent_LoRAs=True, official_flow_steps=10, action_prefix=5,
            semantics='disposable short actual practice budget, no formal qualification')
        record['environment_steps_per_second'] = delta / record['seconds']
        rows.append(record)
        if not actual:
            actual = result[:4]
        for index, item in enumerate(result):
            path = args.output / f'B{B}_rep{rep}' / item['request']['condition']['condition_id']
            path.mkdir(parents=True, exist_ok=True)
            torch.save(item['chain'].to_record(), path / 'experience.pt')
            item['request']['record_path'] = str(path.resolve())
        write_json_atomic(args.output / 'measurements.json', dict(measurements=rows, environment_steps=all_steps))
    for runner in runners:
        runner.close()
    runners.clear()
    full = Runner(runtime, contract, args.physical_gpu, slot_batch=1)
    runners.append(full)
    all_runners.append(full)
    _, longest_task, longest_demo = max((length, task_id, demo) for task_id, task in data.tasks.items()
                                        for demo, length in enumerate(task.episode_lengths))
    condition = dict(next(c for c in conditions if c['task_id'] == longest_task),
        teacher_demo=longest_demo, condition_id=f'profile_long_task{longest_task}_demo{longest_demo}',
        pool='disposable_profile')
    record, completed = measure(runtime, 'complete_new_condition_MT_start',
        lambda: list(full.adapt_many([condition], behavior_version='profile_fresh_chi')))
    all_steps += full.total_environment_steps
    chain = completed[0]['chain']
    rows.append(dict(record, category='adaptation', value=1, environment_steps=full.total_environment_steps,
                     actual_J=chain.metrics['actual_J'], compilation=chain.metrics))
    saved = save_condition(args.output / 'actual_complete_condition', condition, chain)
    completed[0]['request']['record_path'] = saved['record_path']
    rows.append(dict(label='actual_condition_uncompressed_save', valid=True,
                     seconds=saved['serialization_seconds']))
    if all_steps > ENV_LIMIT:
        raise RuntimeError('profile exceeded4096 actual environment steps')
    return completed + actual[:3], all_steps


def profile(args):
    from .run import frozen_git, cost_summary, check_budget
    from ember.pi05_source_contract import append_jsonl
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'profile.json').exists():
        raise ValueError('completed profile cannot be overwritten')
    device = torch.device(args.device)
    _configure_device(device, args.cpu_threads)
    started, tick = time.time(), time.monotonic()
    initial_cost = cost_summary(args.run_root, phases={'profile'})['GPU_hours']
    env_ledger = args.run_root / 'profile_environment_costs.jsonl'
    previous_steps = sum(json.loads(row)['environment_steps'] for row in env_ledger.read_text().splitlines()) if env_ledger.exists() else 0
    if previous_steps + 4064 > ENV_LIMIT or initial_cost >= 2:
        raise RuntimeError('remaining actual profile budget cannot fund the declared full measurement; retain prior evidence')
    runtime = data = None
    runners, all_runners, rows, actual, all_steps = [], [], [], [], 0
    telemetry_handle = (args.output / 'gpu_telemetry.csv').open('w')
    telemetry = subprocess.Popen(['nvidia-smi', '-i', str(args.physical_gpu),
        '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits',
        '-lms', '500'], stdout=telemetry_handle, stderr=subprocess.DEVNULL)
    try:
        runtime = Runtime(args.asset_root, device, frame_chunk=args.frame_chunk, decoder_chunk=args.decoder_chunk,
            native_frame_chunk=args.native_frame_chunk, experience_chunk=args.experience_chunk,
            cache_root=args.run_root / 'frozen_features')
        runtime.profile_deadline = tick + (2 - initial_cost) * 3600
        data = QueryData(args.asset_root)
        actual, all_steps = _profile_control(runtime, data, args, rows, runners, all_runners)
        prepared, credits = _profile_shared(runtime, data, actual, rows, args.output)
        longest = _profile_teaching(runtime, data, prepared, credits, rows)
        selection = dict(slot_batch=best(rows, 'slot'), microbatch=best(rows, 'FM_events4'),
            microbatch_one_event=best(rows, 'FM_events1'), decoder_chunk=best(rows, 'decoder'),
            experience_chunk=best(rows, 'experience'), frame_chunk=best(rows, 'learned_frames'),
            native_frame_chunk=best(rows, 'native_frames'))
        cost = check_budget(args.run_root, storage=True)
        profile_cost = cost_summary(args.run_root, phases={'profile'})['GPU_hours']
        if profile_cost > 2:
            raise RuntimeError('disposable profile exceeded2 GPUh')
        write_json_atomic(args.output / 'profile.json', dict(git=frozen_git(), measurements=rows,
            selection=selection, longest_allowed_video=longest, environment_steps=all_steps,
            profile_GPUh=profile_cost, cost=cost, optimizer_updates=0, complete=True,
            archived_parameters_physical_profile_only=True, frozen_cache_only=True,
            limits=dict(GPUh=2, environment_steps=ENV_LIMIT),
            unmeasured=dict(distributed_rank_speed='requires measured physical-rank update probe',
                slot_above16='full condition plus B4/8 and two sustained16 runs consume4064 worst-case steps'),
            wall_seconds=time.time() - started))
    finally:
        for runner in runners:
            runner.close()
        write_json_atomic(args.output / 'execution_receipt.json', dict(environment_steps=sum(
            r.total_environment_steps for r in all_runners), started_unix=started,
            finished_unix=time.time(), complete=(args.output / 'profile.json').exists(),
            optimizer_updates=0, semantics='all confirmed actual environment work including failures'))
        append_jsonl(env_ledger, dict(pid=os.getpid(), output=str(args.output),
            environment_steps=sum(r.total_environment_steps for r in all_runners),
            complete=(args.output / 'profile.json').exists()))
        if runtime is not None:
            runtime.close()
        if data is not None:
            data.close()
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_handle.close()
