"""Formal FM+own-function learning, physical sharding and complete resume."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import random
import socket
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from ember.pi05_source_checkpoint import (git_state, read_json, source_reference_matches,
                                          write_json_atomic)
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.replay import sum_writer_gradients
from .contract import MT_PATH, SCHEMA, TASKS36
from .data import QueryData
from .interaction import Chain
from .learning import finish_update, fresh_optimizer, gradient_groups, supervised_backward
from .sampling import EventSampler, LOGICAL_BATCH, PHASE_UPDATES, incoming_layer
from .storage import load_event


CHECKPOINT_SCHEMA = 'ember_functional_revision_learning_checkpoint_v1'


def _gather(value, context):
    if context.world_size == 1:
        return [value]
    output = [None] * context.world_size
    dist.all_gather_object(output, value)
    return output


def _collective_call(context, function):
    result, error = None, None
    try:
        result = function()
    except Exception:
        error = traceback.format_exc()
    failures = [error for error in _gather(error, context) if error]
    if failures:
        raise RuntimeError('functional learning rank failure:\n' + '\n'.join(failures))
    return result


def _seed_rank(seed, context):
    value = int(seed) + context.rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if context.device.type == 'cuda':
        torch.cuda.manual_seed(value)
    return value


def _rank_rng(context):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(context.device) if context.device.type == 'cuda' else None)


def _restore_rng(value, context):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch_cpu'])
    if context.device.type == 'cuda':
        torch.cuda.set_rng_state(value['torch_cuda'], context.device)


def _topology(context):
    local = dict(rank=context.rank, local_rank=context.local_rank, hostname=socket.gethostname(),
        device=str(context.device), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        numa_node=context.numa_node, cpu_affinity=list(context.cpu_affinity or ()))
    if context.device.type == 'cuda':
        local['gpu_uuid'] = str(torch.cuda.get_device_properties(context.device).uuid)
    return dict(world_size=context.world_size, ranks=_gather(local, context),
                nccl_p2p_disable=os.environ.get('NCCL_P2P_DISABLE'))


def _identity(runtime, contract):
    return dict(compiler_schema=SCHEMA, run_contract_schema=contract['schema_version'],
        source=runtime.source, mt_reference=str(MT_PATH), lora=runtime.lora.to_dict(),
        trainable_shapes={name: list(p.shape) for name, p in runtime.compiler.named_parameters() if p.requires_grad})


def prepare_events(runtime, data, events):
    """Recompute the complete context and current F/J on each real incoming/E."""
    prepared = []
    for event in events:
        loaded = torch.load(Path(event.record_path) / 'experience.pt', map_location='cpu', weights_only=False)
        incoming, experience, record = load_event(runtime, event, loaded=loaded)
        chain = Chain.from_record(loaded)
        support = chain.support(runtime, record['language'], endpoint=event.endpoint)
        if support is None or event.masked_experience or not experience:
            raise ValueError('formal events require the complete real E and at least one support decision')
        keep = chain.support(runtime, record['language'], endpoint=event.endpoint, successful_only=True)
        actual = next(e for e in record['events'] if e['endpoint'] == event.endpoint)
        prepared.append(dict(event=dict(event.as_dict(), actual_incoming_layer=incoming_layer(actual)), condition=record,
            incoming={name: value.detach() for name, value in incoming.items()},
            experience={name: value.detach() for name, value in experience.items()},
            teacher=runtime.teacher(event.task_id, event.teacher_demo), support=support, keep_support=keep,
            batch=data.query_batch(event, runtime.processor), seed=event.seed,
            teacher_cost=dict(runtime.last_teacher_cost)))
    return prepared


def save_checkpoint(runtime, optimizer, scheduler, sampler, context, *, root, contract,
                    topology, code_git, previous_phases):
    """Publish one model plus optimizer, logical cursor and every rank's RNG."""
    update = (sampler.phase - 1) * PHASE_UPDATES + sampler.cursor
    final = Path(root) / 'training' / 'checkpoints' / f'step_{update:08d}'
    rank_rng = _gather(_rank_rng(context), context)

    def publish():
        if not context.is_main:
            return
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name('.' + final.name + '.partial')
        if final.exists() or partial.exists():
            raise ValueError(f'checkpoint already exists; preserve its state: {final}')
        partial.mkdir()
        save_file({name: value.detach().cpu().contiguous() for name, value in runtime.compiler.state_dict().items()},
                  str(partial / 'ecp.safetensors'))
        torch.save(dict(schema_version=CHECKPOINT_SCHEMA, macro_update=update, phase=sampler.phase,
            model_identity=_identity(runtime, contract), run_contract=contract,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=None,
            sampler=sampler.state_dict(), previous_phases=previous_phases, rank_rng=rank_rng,
            topology=topology, code_git=code_git, logical_batch=LOGICAL_BATCH), partial / 'state.pt')
        manifest = dict(schema_version=CHECKPOINT_SCHEMA, compiler_schema=SCHEMA,
            macro_update=update, phase=sampler.phase, complete=True, logical_batch=LOGICAL_BATCH,
            condition_weight=1 / LOGICAL_BATCH, world_size=context.world_size, topology=topology,
            source=runtime.source, code_git=code_git, scaler=None,
            files={name: dict(bytes=(partial / name).stat().st_size) for name in ('ecp.safetensors', 'state.pt')},
            coverage=sampler.coverage(), full_training_resume=True)
        write_json_atomic(partial / 'manifest.json', manifest)
        partial.replace(final)
        write_json_atomic(final.parent.parent / 'latest_checkpoint.json', dict(path=str(final), macro_update=update))

    _collective_call(context, publish)
    return final


def _checkpoint_bundle(runtime, checkpoint, contract):
    """Validate the complete publication before applying any training state."""
    manifest = read_json(checkpoint / 'manifest.json')
    state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    update, old_phase = int(manifest['macro_update']), int(manifest['phase'])
    manifest_header = dict(schema_version=CHECKPOINT_SCHEMA, compiler_schema=SCHEMA,
                          complete=True, logical_batch=LOGICAL_BATCH, condition_weight=.25)
    state_header = dict(schema_version=CHECKPOINT_SCHEMA, macro_update=update, phase=old_phase,
                       scaler=None, run_contract=contract, logical_batch=LOGICAL_BATCH, topology=manifest['topology'])
    if (any(manifest.get(key) != value for key, value in manifest_header.items())
            or any(state.get(key) != value for key, value in state_header.items())
            or len(state['rank_rng']) != manifest['world_size']
            or set(manifest['files']) != {'ecp.safetensors', 'state.pt'}):
        raise ValueError('complete checkpoint scientific/schema authority changed')
    for name, metadata in manifest['files'].items():
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size != metadata['bytes']:
            raise ValueError(f'incomplete checkpoint asset: {name}')
    recorded, current = dict(state['model_identity']), _identity(runtime, contract)
    recorded_source, current_source = recorded.pop('source'), current.pop('source')
    if recorded != current or not source_reference_matches(recorded_source, current_source):
        raise ValueError('source, complete-LoRA or shared-model identity changed')
    if (state['sampler']['plan']['phase'] != old_phase
            or (old_phase - 1) * PHASE_UPDATES + state['sampler']['cursor'] != update):
        raise ValueError('checkpoint optimizer/sampler cursor disagreement')
    return manifest, state


def restore_checkpoint(runtime, optimizer, scheduler, sampler, context, checkpoint, *, contract, topology):
    """Restore logical work unchanged; explicitly record physical migration."""
    checkpoint = Path(checkpoint)
    manifest, state = _checkpoint_bundle(runtime, checkpoint, contract)
    update, old_phase = int(manifest['macro_update']), int(manifest['phase'])
    previous = deepcopy(state['previous_phases'])
    if old_phase == sampler.phase:
        sampler.load_state_dict(state['sampler'])
    elif old_phase == 1 and sampler.phase == 2 and update == PHASE_UPDATES:
        previous.append(state['sampler'])
    else:
        raise ValueError('phase migration is allowed only at the complete180 boundary')
    runtime.load_checkpoint(checkpoint)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if scheduler.last_epoch != update or any(
            (group['lr'], group['weight_decay'], group['eps'], tuple(group['betas']))
            != (3e-5, 0., 1e-8, (.9, .999)) for group in optimizer.param_groups):
        raise ValueError('constant scheduler and optimizer update cursor disagree')
    old_world = manifest['world_size']
    if context.rank < old_world:
        _restore_rng(state['rank_rng'][context.rank], context)
    else:
        _seed_rank(contract['seed'], context)
    migration = dict(checkpoint=str(checkpoint), macro_update=update,
        old_topology=state['topology'], new_topology=topology,
        restored_rank_rng=list(range(min(old_world, context.world_size))),
        fresh_rank_rng={rank: int(contract['seed']) + rank for rank in range(old_world, context.world_size)},
        physical_topology_changed=state['topology'] != topology,
        logical_event_query_stream_unchanged=True, bitwise_exact_claim=False,
        parent_code_git=state['code_git'])
    return previous, migration


def _synchronize(runtime):
    if runtime.device.type == 'cuda':
        torch.cuda.synchronize(runtime.device)


def _batch_backward(runtime, data, events, args, context):
    _synchronize(runtime)
    started = time.monotonic()
    prepared = prepare_events(runtime, data, events)
    _synchronize(runtime)
    prepare_seconds = time.monotonic() - started
    compile_cost, consumer = {}, {}
    if prepared:
        outgoing = runtime.edit_many(prepared)
        for item, state in zip(prepared, outgoing, strict=True):
            item['outgoing'] = state
        compile_cost = dict(runtime.last_revision_cost)
        consumer = supervised_backward(runtime, prepared, microbatch=args.fm_microbatch,
            revision_microbatch=args.adjoint_microbatch, condition_weight=1 / LOGICAL_BATCH)
    _synchronize(runtime)
    return dict(rank=context.rank, events=[item['event'] for item in prepared], prepare_seconds=prepare_seconds,
        local_seconds=time.monotonic() - started, compile=compile_cost, consumer=consumer,
        adjoint=dict(runtime.last_adjoint_cost) if prepared else {},
        teacher_costs=[item['teacher_cost'] for item in prepared])


def _pool_sampler(root, data, phase, seed):
    path = Path(root) / 'pools' / ('bootstrap' if phase == 1 else 'refresh180') / 'manifest.json'
    manifest = read_json(path)
    lengths = {task: {demo: len(rows) for demo, rows in data.rows[task].items()} for task in TASKS36}
    sampler = EventSampler(manifest, lengths, query_coordinates=data.query_coordinates,
                           task_ids=TASKS36, seed=seed, phase=phase)
    coverage = sampler.coverage()
    if phase == 1:
        expected_missing = {0, 2, 5, 7, 14, 20, 21, 28, 35, 42, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101}
        counts = [sum(incoming_layer(e) == layer for c in manifest['conditions'] for e in c['events'])
                  for layer in ('MT', 'nonMT')]
        if (coverage['actual_conditions'] != 216 or coverage['actual_endpoints'] != 362
                or counts != [314, 48] or set(TASKS36) - set(coverage['nonMT_tasks']) != expected_missing
                or any(len({c['teacher_demo'] for c in manifest['conditions'] if c['task_id'] == task}) != 6
                       for task in TASKS36)):
            raise ValueError('bootstrap changed registered216/362 and actual16-task recursive coverage')
    elif (len(manifest['conditions']) != 72
          or any(len({c['teacher_demo'] for c in manifest['conditions'] if c['task_id'] == task}) != 2
                 for task in TASKS36)):
        raise ValueError('phase2 requires exactly the new72 actual conditions')
    return sampler


def _profile(runtime, data, sampler, args, context, output):
    if args.resume_checkpoint is not None or sampler.phase != 1:
        raise ValueError('physical profiles replay only the same fresh first formal batch')
    runtime.profile_root = output  # Existing Runtime timing synchronizes component boundaries.
    rng = _rank_rng(context)
    events = sampler.next_batch()
    local = events[context.rank::context.world_size]
    runtime.compiler.zero_grad(set_to_none=True)
    if runtime.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.monotonic()
    metrics = _collective_call(context, lambda: _batch_backward(runtime, data, local, args, context))
    reduction_start = time.monotonic()
    parameters = tuple(runtime.compiler.parameters())
    sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 1024**2)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    _synchronize(runtime)
    metrics.update(reduction_clip_seconds=time.monotonic() - reduction_start,
        combined_gradient_norm=float(norm), gradient_groups=gradient_groups(runtime.compiler),
        seconds=time.monotonic() - started)
    if runtime.device.type == 'cuda':
        metrics.update(peak_allocated_GiB=torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
                       peak_reserved_GiB=torch.cuda.max_memory_reserved(runtime.device) / 1024**3)
    result = dict(profile_only=True, optimizer_updates=0, logical_events=[sampler.event_record(e) for e in events],
                  topology=_topology(context), ranks=_gather(metrics, context))
    if context.is_main:
        write_json_atomic(output / 'profile.json', result)
    runtime.compiler.zero_grad(set_to_none=True)
    _restore_rng(rng, context)
    return result


def run_training(runtime, args, context):
    """Parent CLI owns launch/resources; this consumer owns formal updates."""
    phase, root = int(args.phase), Path(args.run_root)
    if phase not in (1, 2) or args.stop_update != phase * PHASE_UPDATES:
        raise ValueError('registered FM phases end exactly at180/360, without extra updates')
    contract = read_json(root / 'run_contract.json')
    if args.fm_microbatch < 1 or args.adjoint_microbatch < 1:
        raise ValueError('physical microbatches must be positive')
    if phase == 2 and args.resume_checkpoint is None:
        raise ValueError('phase2 must retain the complete180 optimizer and logical stream')
    data = QueryData(runtime.asset_root)
    try:
        sampler = _pool_sampler(root, data, phase, int(contract['seed']))
        _seed_rank(contract['seed'], context)
        runtime.compiler.train()
        initialize_deferred_process_group(context, rendezvous_root=root / 'training')
        attempt = _gather(getattr(args, 'attempt', None) or f'phase{phase}_{time.time_ns()}', context)[0]
        output = root / 'training' / 'attempts' / attempt
        output.mkdir(parents=True, exist_ok=True)
        topology, code_git = _topology(context), git_state()
        if args.profile_only:
            return _profile(runtime, data, sampler, args, context, output)
        optimizer, scheduler = fresh_optimizer(runtime.compiler, stage='supervised')
        previous, migration = [], None
        if args.resume_checkpoint is not None:
            previous, migration = _collective_call(context, lambda: restore_checkpoint(runtime,
                optimizer, scheduler, sampler, context, args.resume_checkpoint, contract=contract, topology=topology))
        if context.is_main:
            write_json_atomic(output / 'execution_contract.json', dict(run_contract=str(root / 'run_contract.json'),
                phase=phase, topology=topology, code_git=code_git, resume=migration,
                logical_batch=LOGICAL_BATCH, event_weight=.25, gradient_reduction='SUM',
                microbatch=args.fm_microbatch, adjoint_microbatch=args.adjoint_microbatch,
                support_microbatch=runtime.support_microbatch, source=runtime.source, coverage=sampler.coverage()))
        checkpoint = str(args.resume_checkpoint) if args.resume_checkpoint is not None else None
        while sampler.cursor < PHASE_UPDATES:
            events = sampler.next_batch()
            update = events[0].update
            optimizer.zero_grad(set_to_none=True)
            started = time.monotonic()
            local = events[context.rank::context.world_size]
            metrics = _collective_call(context, lambda: _batch_backward(runtime, data, local, args, context))
            step = finish_update(runtime, optimizer, scheduler, world_size=context.world_size)
            _synchronize(runtime)
            ranks = _gather(metrics, context)
            if context.is_main:
                append_jsonl(output / 'metrics.jsonl', dict(update=update, phase=phase,
                    logical_events=[sampler.event_record(e) for e in events], ranks=ranks, step=step,
                    seconds=time.monotonic() - started))
            if update % 90 == 0:
                checkpoint = str(save_checkpoint(runtime, optimizer, scheduler, sampler, context,
                    root=root, contract=contract, topology=topology, code_git=code_git, previous_phases=previous))
        result = dict(phase=phase, macro_update=args.stop_update, checkpoint=checkpoint,
                      output=str(output), coverage=sampler.coverage(), code_git=code_git)
        if context.is_main:
            write_json_atomic(output / 'completion.json', dict(complete=True, **result))
        return result
    finally:
        data.close()
