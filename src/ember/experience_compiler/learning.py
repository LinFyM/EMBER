"""Fresh shared FM editing from explicit actual incoming/experience events."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import time

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_source_checkpoint import barrier, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.replay import sum_writer_gradients

from .contract import SCHEMA, STAGE
from .credit import fm_credit, surrogate
from .storage import load_event


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = name.split('.')[0]
            groups[group] = groups.get(group, 0.) + float(parameter.grad.float().square().sum())
    return {key: value**.5 for key, value in groups.items()}


def prefetch_event(data, event):
    """One CPU reader overlaps the current event; it never runs policy forward."""
    tick = time.monotonic()
    record = torch.load(Path(event.record_path) / 'experience.pt', map_location='cpu', weights_only=False)
    raw_batch = data.raw_query_batch(event)
    return record, raw_batch, time.monotonic() - tick


def prepare_edit(runtime, event, prefetched):
    tick = time.monotonic()
    raw_record, raw_batch, io_seconds = prefetched
    incoming, experience, authority = load_event(runtime, event, loaded=raw_record)
    load_seconds = time.monotonic() - tick
    teacher = runtime.teacher(event.task_id, event.teacher_demo)
    teacher_cost = dict(runtime.last_teacher_cost)
    tick = time.monotonic()
    with torch.no_grad():
        outgoing = runtime.edit(incoming, teacher, experience)
    edit_seconds = time.monotonic() - tick
    batch = runtime.processor.training_batch(raw_batch)
    record = dict(event=event.as_dict(), behavior_actor_uses_teaching=authority['behavior_actor_uses_teaching'],
        learning_neural_reads=1, differentiable_replay_reads=1, full_video_equivalent_reads=2.,
        read_frames=2 * len(teacher['indices']), teacher_feature_cost=teacher_cost,
        prefetch_io_seconds=io_seconds, event_load_seconds=load_seconds, edit_forward_seconds=edit_seconds)
    return dict(incoming=incoming, outgoing=outgoing, experience=experience,
                teacher=teacher, batch=batch, event=event, record=record)


def group_backward(runtime, prepared, *, microbatch):
    tick = time.monotonic()
    credits = fm_credit(runtime, [p['incoming'] for p in prepared], [p['outgoing'] for p in prepared],
        [p['batch'] for p in prepared], seeds=[p['event'].seed for p in prepared], microbatch=microbatch)
    fm_seconds = time.monotonic() - tick
    records = []
    for item, credit in zip(prepared, credits, strict=True):
        item.pop('batch')
        item.pop('outgoing')
        tick = time.monotonic()
        replay = runtime.edit(item['incoming'], item['teacher'], item['experience'])
        surrogate(replay, credit['cotangent']).backward()
        record = item['record']
        record.update({k: v for k, v in credit.items() if k != 'cotangent'})
        record.update(shared_replay_vjp_seconds=time.monotonic() - tick,
                      FM_physical_microbatch=microbatch, FM_physical_group_events=len(prepared),
                      FM_group_vjp_seconds=fm_seconds)
        records.append(record)
        del replay
    return records


def update(runtime, data, optimizer, scheduler, context, index, args, prefetch):
    parameters = tuple(runtime.compiler.parameters())
    optimizer.zero_grad(set_to_none=True)
    events = [e for e in data.events(index) if e.position % context.world_size == context.rank]
    started = time.monotonic()
    futures = [prefetch.submit(prefetch_event, data, event) for event in events]
    prepared = [prepare_edit(runtime, event, future.result())
                for event, future in zip(events, futures, strict=True)]
    records = group_backward(runtime, prepared, microbatch=args.microbatch)
    del prepared
    local_seconds = time.monotonic() - started
    tick = time.monotonic()
    sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 1024**2)
    collective_seconds = time.monotonic() - tick
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    groups = gradient_groups(runtime.compiler)
    optimizer.step()
    scheduler.step()
    local = dict(records=records, rank=context.rank, local_work_seconds=local_seconds,
                 collective_seconds=collective_seconds)
    gathered = [None] * context.world_size if context.is_main else None
    if context.world_size > 1:
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    if context.is_main:
        all_records = sorted([r for row in gathered for r in row['records']], key=lambda r: r['event']['position'])
        if len(all_records) != 4:
            raise ValueError('physical topology changed the logical four-event update')
        append_jsonl(args.output / 'metrics.jsonl', dict(update=index + 1,
            phase='pool0' if index < 180 else 'pool0_refresh_half',
            records=all_records, rank_timing=[{k: v for k, v in row.items() if k != 'records'} for row in gathered],
            wall_seconds=time.monotonic() - started, combined_gradient_norm=float(norm), gradient_groups=groups))


def save_training(args, runtime, data, optimizer, scheduler, context, macro):
    data.next_update = macro
    return save_ecp_checkpoint(output_dir=args.output, macro=macro, stage=STAGE, context=context,
        model=runtime.compiler, optimizer=optimizer, scheduler=scheduler, run_contract_schema=SCHEMA,
        metrics_rows=macro, sampler_state=data.state_dict(),
        training_state=dict(pool_versions={k: v['version'] for k, v in data.pools.items()},
                            update_objective='paired FM_out + .2 ReLU(FM_out - stopgrad FM_in)',
                            logical_events=4, FM_queries_per_event=28))


def train(args):
    from .data import QueryData
    from .runtime import Runtime
    from .run import register_launch, check_budget

    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 4 or args.stop_update not in {180, 360}:
        raise ValueError('logical batch4 uses ranks1..4 and registered180/360 nodes')
    seed_everything(20261009, context)
    runtime = Runtime(args.asset_root, context.device, frame_chunk=args.frame_chunk,
        decoder_chunk=args.decoder_chunk, experience_chunk=args.experience_chunk,
        native_frame_chunk=args.native_frame_chunk, cache_root=args.run_root / 'frozen_features')
    paths = [args.run_root / 'pools/pool0/manifest.json']
    if args.stop_update == 360:
        paths.append(args.run_root / 'pools/refresh180/manifest.json')
    data = QueryData(args.asset_root, pool_paths=paths)
    optimizer = torch.optim.AdamW(runtime.compiler.parameters(), lr=3e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    initialize_deferred_process_group(context, rendezvous_root=args.output, collective_timeout=timedelta(minutes=45))
    index, topology_resume = 0, None
    try:
        if args.resume:
            state = {}
            index, rows = load_ecp_checkpoint(checkpoint=args.resume, stage=STAGE, context=context,
                model=runtime.compiler, optimizer=optimizer, scheduler=scheduler, run_contract_schema=SCHEMA,
                restored_state=state, allow_world_size_change=args.allow_topology_change)
            data.load_state_dict(state['sampler_state'])
            topology_resume = state.get('topology_resume')
            if rows != index or data.next_update != index or scheduler.last_epoch != index:
                raise ValueError('logical sampler/optimizer/scheduler cursor changed')
        elif args.stop_update != 180:
            raise ValueError('stage360 must preserve the valid180 checkpoint')
        if context.is_main:
            register_launch(args, runtime, context, data, index, topology_resume=topology_resume)
        barrier(context)
        if not args.resume:
            save_training(args, runtime, data, optimizer, scheduler, context, 0)
        stopped, checkpoint, macro = False, args.resume, index
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='FM-prefetch') as prefetch:
            for update_index in range(index, args.stop_update):
                check_budget(args.run_root)
                update(runtime, data, optimizer, scheduler, context, update_index, args, prefetch)
                macro = update_index + 1
                # An early scientific stop saves this just-completed update.
                signal = torch.tensor(int(context.is_main and
                    (args.run_root / 'early_stop_requested.json').is_file()), device=context.device)
                if context.world_size > 1:
                    dist.broadcast(signal, src=0)
                stopped = bool(signal.item())
                if macro % 45 == 0 or stopped:
                    checkpoint = save_training(args, runtime, data, optimizer, scheduler, context, macro)
                    if context.is_main:
                        print(dict(checkpoint_ready=str(checkpoint), update=macro), flush=True)
                if stopped:
                    break
        if context.is_main:
            write_json_atomic(args.output / f'completion_{args.stop_update}.json',
                dict(updates=macro, checkpoint=str(checkpoint), requested_stop_update=args.stop_update,
                     training_complete=macro == 360, stage_complete=macro == args.stop_update,
                     scientific_early_stop=stopped, topology_resume=topology_resume))
    finally:
        data.close()
        runtime.close()
        if context.world_size > 1:
            dist.destroy_process_group()
