"""Disposable native/full-loop profiling on fixed non-held teaching authority."""
from __future__ import annotations

from dataclasses import replace
import gc
import gzip
import time

import torch

from ember.operator_writer.native import read_frozen_teacher_features
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.runtime import autocast
from ember.writer.materialization_workers import _configure_device

from .credit import fm_credit, pg_credit, surrogate
from .data import QueryData
from .interaction import Chain, Runner, cpu_state
from .learning import gradient_groups
from .runtime import Runtime
from .contract import query_seed


def measure(runtime, label, function):
    runtime.compiler.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.monotonic()
    try:
        result = function()
        torch.cuda.synchronize(runtime.device)
        record = dict(label=label, seconds=time.monotonic() - started,
            peak_allocated_GiB=torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved(runtime.device) / 1024**3,
            gradient_groups=gradient_groups(runtime.compiler), valid=True)
    except torch.cuda.OutOfMemoryError:
        runtime.compiler.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        record, result = dict(label=label, valid=False, OOM=True, seconds=time.monotonic() - started), None
    print(record, flush=True)
    return record, result


def initial(runtime, teacher):
    with torch.no_grad(), autocast(runtime.device):
        q = runtime.compiler.initial(teacher)
        return Chain(states=[cpu_state(runtime.compiler.decode(q))])


def replay_backward(runtime, teacher, chain, credit):
    surrogate(runtime.replay(teacher, chain), credit['cotangents']).backward()


def best(rows, category):
    groups = {}
    for row in rows:
        if row['category'] == category:
            groups.setdefault(row['value'], []).append(row)
    # Keep a physical size eligible only if every profiled workload fit it.
    eligible = [max(group, key=lambda r: r['seconds']) for group in groups.values()
                if all(r['valid'] for r in group)]
    return min(eligible, key=lambda r: r['seconds'])


def profile_initial(runtime, teacher, event, batch):
    chain = initial(runtime, teacher)
    equality = fm_credit(runtime, [cpu_state(runtime.mt), chain.states[0]], batch,
                          seed=event.seed, microbatch=7)
    init_behavior = dict(MT_FM=equality['stage_losses'][0], decoded_FM=equality['stage_losses'][1],
                         difference=equality['stage_losses'][1] - equality['stage_losses'][0])
    if not abs(init_behavior['difference']) < max(.002, .02 * init_behavior['MT_FM']):
        raise ValueError(f'complete MT initialization changed its actual native FM consumer: {init_behavior}')
    optimizer = torch.optim.AdamW(runtime.compiler.parameters(), lr=3e-5, weight_decay=0.)
    warm = []
    for _ in range(2):
        current = initial(runtime, teacher)
        credit = fm_credit(runtime, current.states, batch, seed=event.seed, microbatch=7)
        optimizer.zero_grad(set_to_none=True)
        replay_backward(runtime, teacher, current, credit)
        warm.append(gradient_groups(runtime.compiler))
        torch.nn.utils.clip_grad_norm_(runtime.compiler.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    if warm[-1].get('reader', 0.) == 0 or warm[-1].get('initializer', 0.) == 0:
        raise ValueError('shared teaching/initial-Q route stayed functionally disconnected after head learning')
    return init_behavior, warm


def profile_fm(runtime, event, teacher, batch):
    chain = initial(runtime, teacher)
    rows, latest = [], None
    for micro in (7, 14, 28):
        record, credit = measure(runtime, f'FM_micro{micro}',
            lambda micro=micro: fm_credit(runtime, chain.states, batch, seed=event.seed, microbatch=micro))
        record.update(category='FM', value=micro, samples=28)
        rows.append(record)
        if credit is not None:
            latest = credit
    if latest is None:
        raise RuntimeError('no valid full native FM batch')
    return rows


def profile_teaching(runtime, data, rows):
    # Longest eligible teaching clip: legal RGB length metadata only.
    events = [e for u in range(9) for e in data.events(u)]
    by_task = {e.task_id: e for e in events}
    length, task_id, demo = max((length, task_id, demo)
        for task_id, task in data.tasks.items() for demo, length in enumerate(task.episode_lengths)
        if demo not in {d for d, _ in by_task[task_id].queries28})
    long_event = replace(by_task[task_id], teacher_demo=demo)
    long_teacher = runtime.teacher(task_id, demo)
    raw = data.videos.load(task_id, demo)
    tokens, mask, _ = runtime.tokenizer([data.tasks[task_id].authority.language])
    for chunk in dict.fromkeys((8, 16, 32, 64, 128, len(raw.frames))):
        def native_read():
            with autocast(runtime.device):
                return read_frozen_teacher_features(runtime.policy, runtime.mt, runtime.probe,
                    (torch.from_numpy(raw.frames).to(runtime.device), torch.from_numpy(raw.frame_indices), tokens, mask),
                    frame_chunk=chunk)
        record, features = measure(runtime, f'native_frame_chunk{chunk}', native_read)
        record.update(category='native_frames', value=chunk, frames=len(raw.frames))
        rows.append(record)
        del features
    long_chain = initial(runtime, long_teacher)
    long_batch = data.query_batch(long_event, runtime.processor)
    long_credit = fm_credit(runtime, long_chain.states, long_batch, seed=long_event.seed,
                            microbatch=best(rows, 'FM')['value'])
    for chunk in dict.fromkeys((8, 16, 32, 64, 128, len(raw.frames))):
        runtime.compiler.reader.chunk = chunk
        record, _ = measure(runtime, f'learned_frame_chunk{chunk}',
                             lambda: replay_backward(runtime, long_teacher, long_chain, long_credit))
        record.update(category='learned_frames', value=chunk)
        rows.append(record)
    runtime.compiler.reader.chunk = best(rows, 'learned_frames')['value']
    total_rows = len(runtime.compiler.decoder.coordinate_embeddings)
    for chunk in (4096, 16384, 65536, total_rows):
        runtime.compiler.decoder.chunk_rows = chunk
        record, _ = measure(runtime, f'decoder_rows{chunk}',
                             lambda: replay_backward(runtime, long_teacher, long_chain, long_credit))
        record.update(category='decoder', value=chunk)
        rows.append(record)
    runtime.compiler.decoder.chunk_rows = best(rows, 'decoder')['value']
    runtime.native_frame_chunk = best(rows, 'native_frames')['value']
    return long_event, long_teacher, dict(task_id=task_id, demo=demo, raw_frames=length, sampled_frames=len(raw.frames))


def profile_actual(runtime, runner, event, teacher, batch, rows, destination):
    # Actual open-ended practice: the implementation imposes no J menu.
    task = next(t for t in runner.contract['tasks'] if t['global_task_id'] == event.task_id)
    record, chain = measure(runtime, 'actual_adaptation',
        lambda: runner.adapt(task, teacher, event.seed, (*event.query_states2, 32, 33, 34)))
    rows.append(dict(record, category='adaptation', compilation=chain.metrics))
    queries = []
    retain_profile_chain(destination, chain, queries)
    for i, state_id in enumerate(event.query_states2):
        record, query = measure(runtime, 'actual_SDE_query',
            lambda state_id=state_id, i=i: runner.query(task, state_id, chain.states[-1], query_seed(event.seed, i)))
        rows.append(dict(record, category='SDE_query', environment_steps=query['row']['environment_steps'],
                         success=query['row']['success']))
        queries.append(query)
        retain_profile_chain(destination, chain, queries)
    returns = [int(q['row']['success']) for q in queries]
    if any(returns):
        for micro in (4, 8, 16, 32):
            record, _ = measure(runtime, f'PG_micro{micro}', lambda micro=micro:
                pg_credit(runtime, chain.states[-1], task, queries, returns, microbatch=micro))
            rows.append(dict(record, category='PG', value=micro))
    credit = fm_credit(runtime, chain.states, batch, seed=event.seed, microbatch=best(rows, 'FM')['value'])
    if chain.endpoints:
        for chunk in dict.fromkeys((16, 32, 64, 128, len(chain.records))):
            runtime.compiler.encoder.chunk = chunk
            record, _ = measure(runtime, f'experience_chunk{chunk}',
                lambda: replay_backward(runtime, teacher, chain, credit))
            rows.append(dict(record, category='experience', value=chunk))
        runtime.compiler.encoder.chunk = best(rows, 'experience')['value']
    record, _ = measure(runtime, 'actual_all_stage_replay', lambda: replay_backward(runtime, teacher, chain, credit))
    rows.append(dict(record, category='all_stage_replay'))
    record, finals = measure(runtime, 'actual_final_batch3',
                             lambda: runner.final_many(task, [32, 33, 34], chain.states[-1]))
    rows.append(dict(record, category='final_batch3', rows=finals))
    record, final = measure(runtime, 'actual_final_single',
                            lambda: runner.final(task, 32, chain.states[-1]))
    rows.append(dict(record, category='final_single', environment_steps=final['environment_steps'],
                     success=final['success']))
    return chain, queries, returns


def retain_profile_chain(target, chain, queries):
    target.mkdir(parents=True, exist_ok=True)
    with gzip.open(target / 'experience.pt.gz', 'wb', compresslevel=1) as handle:
        torch.save(chain.to_record(), handle)
    with gzip.open(target / 'query.pt.gz', 'wb', compresslevel=1) as handle:
        torch.save(queries, handle)


def profile(args):
    from .contract import learning_environment
    from .run import frozen_git, check_budget

    device = torch.device(args.device)
    _configure_device(device, args.cpu_threads)
    runtime = Runtime(args.asset_root, device, frame_chunk=args.frame_chunk, decoder_chunk=args.decoder_chunk,
                      native_frame_chunk=args.native_frame_chunk, experience_chunk=args.experience_chunk)
    data = QueryData(args.asset_root)
    env_contract = learning_environment(asset_root=args.asset_root)
    env_contract['parallel'] = {'envs_per_replica': 3}
    runner = Runner(runtime, env_contract, args.physical_gpu)
    event = data.events(135)[0]
    teacher = runtime.teacher(event.task_id, event.teacher_demo)
    batch = data.query_batch(event, runtime.processor)
    init_behavior, warm = profile_initial(runtime, teacher, event, batch)
    rows = profile_fm(runtime, event, teacher, batch)
    long_event, long_teacher, longest = profile_teaching(runtime, data, rows)
    chain, queries, returns = profile_actual(runtime, runner, event, teacher, batch, rows, args.output / 'short_condition')
    long_batch = data.query_batch(long_event, runtime.processor)
    long_chain, long_queries, long_returns = profile_actual(runtime, runner, long_event, long_teacher, long_batch, rows, args.output / 'long_condition')
    if runner.total_environment_steps > 12000:
        raise RuntimeError('disposable profile exceeded its real environment-step budget')
    pg_observed = any(returns + long_returns)
    selection = dict(microbatch=best(rows, 'FM')['value'],
        pg_microbatch=best(rows, 'PG')['value'] if pg_observed else best(rows, 'FM')['value'],
        frame_chunk=best(rows, 'learned_frames')['value'], native_frame_chunk=best(rows, 'native_frames')['value'],
        decoder_chunk=best(rows, 'decoder')['value'], experience_chunk=runtime.compiler.encoder.chunk)
    write_json_atomic(args.output / 'profile.json', dict(git=frozen_git(), initialization=init_behavior,
        warm_gradient_groups=warm, longest_video=longest, measurements=rows, selection=selection,
        query_returns=returns + long_returns, PG_timing_observed=pg_observed,
        environment_steps=runner.total_environment_steps, cost=check_budget(args.run_root, storage=True),
        complete=True, disposable=True))
    runner.close()
    runtime.close()
    data.close()
