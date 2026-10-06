"""Fixed logical450 joint G/F updates using the existing events and exact recovery owner."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.operator_writer.data import FormalData
from ember.operator_writer.run import frozen_git, gather, optimizer_for
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed, initialize_deferred_process_group, seed_everything
from ember.writer.replay import sum_writer_gradients
from ember.writer.task_execution import condition_assignment

from .credit import one_condition
from .feedback import FeedbackFunction
from .runtime import build_runtime

SCHEMA = 'ember_relation_grounded_learning_v1'
STAGE = 'relation_grounded_writer_learning'


class PairedState:
    """Keep two genuinely independent optimizer/scheduler states in one ECP payload."""
    def __init__(self, g, f):
        self.g, self.f = g, f

    def state_dict(self):
        return {'G': self.g.state_dict(), 'F': self.f.state_dict()}

    def load_state_dict(self, state):
        if set(state) != {'G', 'F'}:
            raise ValueError('checkpoint lost one independent learning state')
        self.g.load_state_dict(state['G'])
        self.f.load_state_dict(state['F'])


def gradient_norm(module):
    terms = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(terms).norm()) if terms else 0.


def gradient_groups(runtime, feedback):
    writer = runtime.writer
    return {'G_public': gradient_norm(writer.common), 'G_Phi': gradient_norm(writer.phi),
            'G_Omega': gradient_norm(writer.omega), 'G_read': gradient_norm(writer.read),
            'G_compiler': gradient_norm(writer.conditional_targets), 'F': gradient_norm(feedback),
            'F_encoder': gradient_norm(feedback.encoder), 'F_out': gradient_norm(feedback.out)}


def train(spec, args):
    from .labels import LabelStore
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size not in range(1, 7):
        raise ValueError('physical topology requires1..6 ranks on one node')
    seed_everything(7, context)
    torch.set_num_threads(args.cpu_threads)
    git = frozen_git(continuation=True)
    output = Path(args.output).resolve()
    runtime = build_runtime(args.asset_root, spec, context.device)
    feedback = FeedbackFunction().to(context.device)
    models = torch.nn.ModuleDict({'G': runtime.writer, 'F': feedback})
    runtime.writer.train()
    feedback.train()
    data = FormalData(args.asset_root, spec)
    labels = LabelStore(data, args.asset_root, cache_root=Path(spec['run_root']) / 'labels')
    semantics = read_json(Path(spec['run_root']) / 'labels/semantic_vectors.json')
    labels.set_semantics(semantics['vectors'])
    opt_g, sched_g, params_g = optimizer_for(runtime.writer, spec)
    opt_f, sched_f, params_f = optimizer_for(feedback, spec)
    optimizers, schedulers = PairedState(opt_g, opt_f), PairedState(sched_g, sched_f)
    parameters = (*params_g, *params_f)
    contract = {'schema_version': SCHEMA, 'stage': STAGE, 'git': git,
                'spec': str(Path(args.spec).resolve()), 'source': runtime.source,
                'source_trainable': 0, 'lora': runtime.lora.to_dict(), 'events': spec['events'],
                'optimization': spec['optimization'], 'model': spec['model'],
                'topology': {'world_size': context.world_size, 'node': args.node},
                'microbatch': args.microbatch, 'frame_chunk': args.frame_chunk,
                'initialization': 'fresh_identity20260721_Gmodule7_Ffork20261006',
                'parameter_counts': {name: sum(p.numel() for p in module.parameters())
                                     for name, module in models.items()},
                'information_wall': spec['information_wall']}
    if context.is_main:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'run_contract.json').exists():
            raise ValueError('new attempt already has a run contract')
        write_json_atomic(output / 'run_contract.json', contract)
    initialize_deferred_process_group(context, rendezvous_root=Path(spec['run_root']) / 'rendezvous')
    updates = 0
    if args.resume:
        updates, rows = load_ecp_checkpoint(checkpoint=args.resume, stage=STAGE, context=context,
            model=models, optimizer=optimizers, scheduler=schedulers, run_contract_schema=SCHEMA,
            allow_world_size_change=args.allow_topology_change)
        saved = torch.load(args.resume / 'trainer_state.pt', map_location='cpu', weights_only=False)
        data.restore(saved['sampler_state'])
        if rows != updates or sched_g.last_epoch != updates or sched_f.last_epoch != updates:
            raise ValueError('G/F scheduler and logical cursor changed')
    else:
        save_checkpoint(output, 0, context, models, optimizers, schedulers, data, contract)
        if args.profile:
            profile(runtime, feedback, data, labels, context, args, output, models, optimizers, schedulers)
            contract.update(microbatch=args.microbatch, frame_chunk=args.frame_chunk)
            if context.is_main:
                write_json_atomic(output / 'run_contract.json', contract)
    started = time.perf_counter()
    try:
        for step in range(updates, 450):
            if args.deadline_epoch and time.time() >= args.deadline_epoch:
                raise RuntimeError('hard wall deadline reached before completing the fixed window')
            record = update(runtime, feedback, data, labels, context, opt_g, opt_f,
                            sched_g, sched_f, parameters, step, args.microbatch, args.frame_chunk)
            data.next_step = step + 1
            if context.is_main:
                append_jsonl(output / 'metrics.jsonl', record)
            if (step + 1) % 90 == 0:
                save_checkpoint(output, step + 1, context, models, optimizers, schedulers, data, contract)
        if context.is_main:
            write_json_atomic(output / 'completion.json', {'status': 'complete', 'updates': 450,
                'queries': 50400, 'source_trainable': 0, 'git': git,
                'seconds_after_profile': time.perf_counter() - started})
    finally:
        data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def update(runtime, feedback, data, labels, context, opt_g, opt_f, sched_g, sched_f,
           parameters, step, microbatch, frame_chunk):
    started = time.perf_counter()
    opt_g.zero_grad(set_to_none=True)
    opt_f.zero_grad(set_to_none=True)
    jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
    costs = [data.videos.frame_counts(e['task'], e['teacher_demo'])[1] for e in jobs]
    assigned = condition_assignment(tuple(range(4)), costs=dict(enumerate(costs)), world_size=context.world_size)
    local, failure = [], None
    try:
        for index in assigned[context.rank]:
            local.append({'job': index, **one_condition(runtime, feedback, data, labels, jobs[index],
                microbatch=microbatch, frame_chunk=frame_chunk, update=step + 1)})
    except Exception:
        failure = traceback.format_exc()
    failures = gather(failure, context.world_size)
    if any(failures):
        if context.is_main:
            append_jsonl(Path(data.spec['run_root']) / 'engineering/update_failures.jsonl',
                         {'update': step + 1, 'at': time.time(), 'failures': failures})
        raise RuntimeError('a rank failed actual condition execution\n' + '\n'.join(f for f in failures if f))
    sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 1024**2)
    norms = {'G': float(torch.nn.utils.clip_grad_norm_(runtime.writer.parameters(), 1.)),
             'F': float(torch.nn.utils.clip_grad_norm_(feedback.parameters(), 1.))}
    if any(not torch.isfinite(torch.tensor(norm)) for norm in norms.values()):
        raise ValueError('nonfinite independent G/F gradient norm')
    groups = gradient_groups(runtime, feedback)
    opt_g.step()
    opt_f.step()
    sched_g.step()
    sched_f.step()
    rows = [row for part in gather(local, context.world_size) for row in part]
    if len(rows) != 4 or {row['task'] for row in rows} != set(data.tasks_for_step(step)):
        raise ValueError('joint update lost the four equally weighted conditions')
    return {'update': step + 1, 'queries': 112, 'jobs': sorted(rows, key=lambda row: row['job']),
            'grad_norm_before_clip': norms, 'gradient_groups': groups,
            'lr': {'G': opt_g.param_groups[0]['lr'], 'F': opt_f.param_groups[0]['lr']},
            'world_size': context.world_size, 'microbatch': microbatch, 'frame_chunk': frame_chunk,
            'seconds': time.perf_counter() - started,
            'peak_reserved_bytes': max(gather(torch.cuda.max_memory_reserved(runtime.device), context.world_size))}


def save_checkpoint(output, macro, context, models, optimizers, schedulers, data, contract):
    return save_ecp_checkpoint(output_dir=output, macro=macro, stage=STAGE, context=context,
        model=models, optimizer=optimizers, scheduler=schedulers, run_contract_schema=SCHEMA,
        metrics_rows=macro, sampler_state=data.sampler_state(),
        training_state={'source': contract['source'], 'topology': contract['topology'],
                        'model': contract['model'], 'two_independent_optimizers': True})


def profile(runtime, feedback, data, labels, context, args, output, models, optimizers, schedulers):
    """At most3 first legal macro updates, then restore all states/RNG/cursor."""
    configurations = [(28, 16), (28, 32), (28, 64)][:args.profile]
    records = []
    for index, (microbatch, frame_chunk) in enumerate(configurations):
        torch.cuda.reset_peak_memory_stats(runtime.device)
        started = time.perf_counter()
        try:
            record = update(runtime, feedback, data, labels, context,
                optimizers.g, optimizers.f, schedulers.g, schedulers.f,
                tuple(models.parameters()), 0, microbatch, frame_chunk)
            record.update(status='complete', queries_per_second=112 / record['seconds'],
                          conditions_per_second=4 / record['seconds'])
        except RuntimeError as error:
            record = {'status': 'failed', 'error': str(error), 'microbatch': microbatch,
                      'frame_chunk': frame_chunk, 'seconds': time.perf_counter() - started,
                      'peak_reserved_bytes': max(gather(torch.cuda.max_memory_reserved(runtime.device), context.world_size))}
        records.append(record)
        models.zero_grad(set_to_none=True)
        runtime.writer.last_prediction = None
        torch.cuda.empty_cache()
        load_ecp_checkpoint(checkpoint=output / 'checkpoints/macro_00000000', stage=STAGE,
            context=context, model=models, optimizer=optimizers, scheduler=schedulers, run_contract_schema=SCHEMA)
        data.next_step = 0
    successful = [record for record in records if record['status'] == 'complete']
    if not successful:
        raise RuntimeError('all first-event physical throughput profiles failed')
    chosen = min(successful, key=lambda record: record['seconds'])
    configuration = [chosen['microbatch'], chosen['frame_chunk']]
    if context.world_size > 1:
        dist.broadcast_object_list(configuration, src=0)
    args.microbatch, args.frame_chunk = configuration
    if context.is_main:
        write_json_atomic(output / 'profile.json', {'discarded_updates': len(records), 'records': records,
            'chosen': {'microbatch': args.microbatch, 'frame_chunk': args.frame_chunk},
            'restored': 'complete_G_F_optimizers_schedulers_rank_RNG_and_logical_cursor',
            'stop_amplification': 'frame64 measured or failed; logical query batch28 exhausted'})
