"""Bounded 451..630, main-FM-only P/Q update owner using canonical ECP.

Retire together with relation_input_compilation_20261007. Physical ranks change
placement only; all four tasks retain their original quarter weight.
"""
from __future__ import annotations

from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.operator_writer.data import FormalData
from ember.operator_writer.run import frozen_git, gather, optimizer_for
from ember.pi05_source_checkpoint import read_json, restore_rng, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed, initialize_deferred_process_group, seed_everything
from ember.writer.replay import sum_writer_gradients
from ember.writer.task_execution import condition_assignment
from ember.relation_input_compilation import (PARENT, build_runtime, freeze_upstream,
                                             load_G_weights, make_labels, one_condition)

SCHEMA = 'ember_relation_input_compilation_v1'
STAGE = 'relation_input_compilation_learning'


def gradient_norm(module):
    terms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(terms).norm()) if terms else 0.


def frozen_snapshot(writer, optimizer):
    return [(name, p, optimizer.state[p].get('step', torch.tensor(0.)).clone())
            for name, p in writer.named_parameters() if not p.requires_grad]


def check_frozen(snapshot, optimizer):
    if any(p.grad is not None or not torch.equal(step, optimizer.state[p].get('step', torch.tensor(0.)))
           for _name, p, step in snapshot):
        raise ValueError('frozen common/Phi received gradient or an optimizer step')


def restore_parent(runtime, optimizer, scheduler, data, context):
    """Explicit intervention fork: original full G Adam IDs, clock and RNG."""
    manifest = read_json(PARENT / 'checkpoint_manifest.json')
    if (manifest['stage'] != 'relation_grounded_writer_learning' or manifest['next_macro'] != 450
            or manifest['world_size'] != 4):
        raise ValueError('parent G450 ECP identity changed')
    runtime.writer.load_state_dict(load_G_weights(PARENT, device=runtime.device), strict=True)
    saved = torch.load(PARENT / 'trainer_state.pt', map_location='cpu', weights_only=False)
    if saved['next_macro'] != 450 or set(saved['optimizer']) != {'G', 'F'}:
        raise ValueError('parent optimizer/cursor lost G owner')
    optimizer.load_state_dict(saved['optimizer']['G'])
    scheduler.load_state_dict(saved['scheduler']['G'])
    data.restore(saved['sampler_state'])
    if scheduler.last_epoch != 450 or data.next_step != 450:
        raise ValueError('parent absolute scheduler/sampler clocks changed')
    if context.rank < 4:
        state = torch.load(PARENT / f'rank_{context.rank:02d}_state.pt', map_location='cpu', weights_only=False)
        if state['rank'] != context.rank or state['next_macro'] != 450 or state['world_size'] != 4:
            raise ValueError('parent rank RNG identity changed')
        restore_rng(state['rng'], context)
    return {'checkpoint_world_size': 4, 'current_world_size': context.world_size,
            'checkpoint_rng_ranks': list(range(min(4, context.world_size))),
            'fresh_seeded_ranks': list(range(4, context.world_size)),
            'intervention_not_original_G_F_exact_resume': True}


def update(runtime, data, labels, context, optimizer, scheduler, active, frozen, step, args):
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
    costs = {i: data.videos.frame_counts(e['task'], e['teacher_demo'])[1] - 1 for i, e in enumerate(jobs)}
    assigned = condition_assignment(tuple(range(4)), costs=costs, world_size=context.world_size)
    rows, failure = [], None
    try:
        for i in assigned[context.rank]:
            rows.append({'job': i, **one_condition(runtime, data, labels, jobs[i],
                        microbatch=args.microbatch, frame_chunk=args.frame_chunk)})
    except Exception:
        failure = traceback.format_exc()
    failures = gather(failure, context.world_size)
    if any(failures):
        if context.is_main:
            append_jsonl(Path(args.output) / 'failures.jsonl', {'absolute_update': step + 1,
                         'at': time.time(), 'rank_failures': failures})
        raise RuntimeError('actual condition consumer failed\n' + '\n'.join(f for f in failures if f))
    # Frozen parameters never enter the reducer: its missing-gradient fill must
    # not manufacture zero gradients that would cause AdamW decay/moment steps.
    sum_writer_gradients(active, world_size=context.world_size, bucket_bytes=64 * 1024**2)
    norm = float(torch.nn.utils.clip_grad_norm_(active, 1.))
    if not torch.isfinite(torch.tensor(norm)):
        raise ValueError('nonfinite main FM gradient')
    groups = {name: gradient_norm(getattr(runtime.writer, name))
              for name in ('common', 'phi', 'omega', 'read', 'conditional_targets')}
    check_frozen(frozen, optimizer)
    lr_used = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    check_frozen(frozen, optimizer)
    merged = [r for part in gather(rows, context.world_size) for r in part]
    if len(merged) != 4 or {r['task'] for r in merged} != set(data.tasks_for_step(step)):
        raise ValueError('logical four-condition macro changed')
    return {'absolute_update': step + 1, 'update': step + 1 - 450, 'queries': 112,
            'jobs': sorted(merged, key=lambda r: r['job']), 'grad_norm_before_clip': norm,
            'gradient_groups_after_clip': groups, 'frozen_gradients_none': True,
            'frozen_optimizer_steps_unchanged': True, 'lr_used': lr_used,
            'lr_next': optimizer.param_groups[0]['lr'], 'scheduler_last_epoch': scheduler.last_epoch,
            'world_size': context.world_size, 'microbatch': args.microbatch,
            'frame_chunk': args.frame_chunk, 'seconds': time.perf_counter() - started,
            'peak_reserved_bytes': max(gather(torch.cuda.max_memory_reserved(runtime.device), context.world_size))}


def save_checkpoint(runtime, optimizer, scheduler, data, context, args, contract, macro):
    return save_ecp_checkpoint(output_dir=Path(args.output), macro=macro, stage=STAGE, context=context,
        model=runtime.writer, optimizer=optimizer, scheduler=scheduler, run_contract_schema=SCHEMA,
        metrics_rows=macro, sampler_state=data.sampler_state(),
        training_state={'source': runtime.source, 'topology': contract['topology'],
                        'arm': args.arm, 'parent': str(PARENT), 'absolute_macro': macro,
                        'matched_intervention': contract['diagnostic'], 'parameter_order': contract['parameter_order']})


def profile(runtime, optimizer, scheduler, data, labels, context, args, active, output):
    """At most two first legal macros; reload G/Adam/clock/cursor/all rank RNG."""
    records = []
    for chunk in (64, 128)[:args.profile]:
        args.frame_chunk = chunk
        torch.cuda.reset_peak_memory_stats(runtime.device)
        frozen = frozen_snapshot(runtime.writer, optimizer)
        try:
            row = update(runtime, data, labels, context, optimizer, scheduler, active, frozen, 450, args)
            row['status'] = 'complete'
        except RuntimeError as error:
            row = {'status': 'failed', 'error': str(error), 'frame_chunk': args.frame_chunk,
                   'peak_reserved_bytes': max(gather(torch.cuda.max_memory_reserved(runtime.device), context.world_size))}
        records.append(row)
        runtime.writer.zero_grad(set_to_none=True)
        runtime.writer.last_prediction = None
        torch.cuda.empty_cache()
        restore_parent(runtime, optimizer, scheduler, data, context)
    successful = [r for r in records if r['status'] == 'complete']
    if not successful:
        raise RuntimeError('both authorized first-event physical profiles failed')
    chosen = min(successful, key=lambda row: row['seconds'])
    args.microbatch, args.frame_chunk = chosen['microbatch'], chosen['frame_chunk']
    if context.is_main:
        write_json_atomic(output / 'profile.json', {'discarded_updates': len(records), 'records': records,
            'chosen': {'microbatch': args.microbatch, 'frame_chunk': args.frame_chunk},
            'reset': 'full_G_and_original_Adam_G_scheduler_sampler_all_rank_RNG',
            'stop_amplification': 'logical query batch28 exhausted; frame128 tested against64 on identical macro451'})


def restore_state(runtime, optimizer, scheduler, data, context, args):
    if not args.resume:
        return 450, restore_parent(runtime, optimizer, scheduler, data, context)
    saved_state = {}
    macro, _rows = load_ecp_checkpoint(checkpoint=args.resume, stage=STAGE, context=context,
        model=runtime.writer, optimizer=optimizer, scheduler=scheduler, run_contract_schema=SCHEMA,
        restored_state=saved_state, allow_world_size_change=args.allow_topology_change)
    if macro not in (540, 630) or saved_state['training_state']['arm'] != args.arm:
        raise ValueError('unregistered P/Q recovery checkpoint')
    data.restore(saved_state['sampler_state'])
    return macro, saved_state.get('topology_resume', {'world_size': context.world_size})


def training_contract(runtime, optimizer, all_parameters, active, context, migration, git, spec, args):
    order = [{'index': i, 'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype),
              'trainable': p.requires_grad} for i, (name, p) in enumerate(runtime.writer.named_parameters())]
    if len(all_parameters) != len(order) or len(optimizer.param_groups[0]['params']) != len(order):
        raise ValueError('Adam IDs no longer cover original full G registration')
    return {'schema_version': SCHEMA, 'stage': STAGE, 'git': git, 'spec': str(Path(args.spec).resolve()),
            'arm': args.arm, 'parent': str(PARENT), 'source': runtime.source, 'source_trainable': 0,
            'lora': runtime.lora.to_dict(), 'events': spec['events'], 'optimization': spec['optimization'],
            'model': spec['model'], 'diagnostic': spec['diagnostic'], 'parameter_order': order,
            'topology': {'world_size': context.world_size, 'node': args.node, 'migration': migration},
            'microbatch': args.microbatch, 'frame_chunk': args.frame_chunk,
            'parameter_counts': {'all': sum(p.numel() for p in all_parameters),
                                 'trainable': sum(p.numel() for p in active)},
            'information_wall': 'P frozen RGB Phi; Q existing train teacher GT; no F/live/query GT/held/Test'}


def train(spec, args):
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size not in range(1, 7) or args.profile not in (0, 1, 2):
        raise ValueError('unregistered physical topology/profile count')
    seed_everything(7, context)
    torch.set_num_threads(args.cpu_threads)
    git = frozen_git(continuation=True)
    output = Path(args.output).resolve()
    if output != Path(spec['run_root']) / 'train' / args.arm:
        raise ValueError('unregistered P/Q output path')
    runtime = build_runtime(args.asset_root, spec, context.device, args.arm, freeze=False)
    optimizer, scheduler, all_parameters = optimizer_for(runtime.writer, spec)
    data = FormalData(args.asset_root, spec)
    labels = make_labels(data, args.asset_root, spec) if args.arm == 'Q' else None
    initialize_deferred_process_group(context, rendezvous_root=Path(spec['run_root']) / f'rendezvous_{args.arm}')
    macro, migration = restore_state(runtime, optimizer, scheduler, data, context, args)
    active = freeze_upstream(runtime)
    runtime.writer.train()
    runtime.writer.common.eval()
    runtime.writer.phi.eval()
    contract = training_contract(runtime, optimizer, all_parameters, active, context, migration, git, spec, args)
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'run_contract.json').exists() and not args.resume:
            raise ValueError('P/Q attempt already started; explicit recovery required')
        write_json_atomic(output / ('run_contract.json' if not args.resume else f'resume_contract_{macro}.json'), contract)
    try:
        if args.profile and not args.resume:
            profile(runtime, optimizer, scheduler, data, labels, context, args, active, output)
            contract.update(microbatch=args.microbatch, frame_chunk=args.frame_chunk)
            if context.is_main:
                write_json_atomic(output / 'run_contract.json', contract)
        frozen = frozen_snapshot(runtime.writer, optimizer)
        started = time.perf_counter()
        for step in range(macro, 630):
            if args.deadline_epoch and time.time() >= args.deadline_epoch:
                raise RuntimeError('hard wall deadline before fixed window completed')
            row = update(runtime, data, labels, context, optimizer, scheduler, active, frozen, step, args)
            data.next_step = step + 1
            if context.is_main:
                append_jsonl(output / 'metrics.jsonl', row)
            if step + 1 in (540, 630):
                save_checkpoint(runtime, optimizer, scheduler, data, context, args, contract, step + 1)
        if context.is_main:
            write_json_atomic(output / 'completion.json', {'status': 'complete', 'arm': args.arm,
                'updates': 180, 'absolute_macro': 630, 'queries': 20160, 'conditions': 720,
                'source_trainable': 0, 'git': git, 'seconds_after_profile': time.perf_counter() - started})
    finally:
        data.close()
        if labels is not None:
            labels.close()
        if dist.is_initialized():
            dist.destroy_process_group()
