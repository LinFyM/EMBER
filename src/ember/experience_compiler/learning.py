"""Continuous fresh FM warm -> shared experience/FM/SDE meta learning."""
from __future__ import annotations

import gzip
from pathlib import Path
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from safetensors.torch import save_file

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_deferred_process_group, initialize_distributed, seed_everything
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast

from .credit import fm_credit, pg_credit, pg_surrogate, surrogate
from .interaction import Chain, Runner, cpu_state
from .runtime import Runtime


class ValueBaseline(nn.Module):
    """Only detached current Q summary and actual pre-query proprioception."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261009)
            self.network = nn.Sequential(nn.LayerNorm(264), nn.Linear(264, 64), nn.SiLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features.detach()).flatten()


def baseline_features(chain, queries, device):
    return torch.stack([torch.cat((chain.q_context,
        torch.tensor(query['row']['initial_proprio'], dtype=torch.float32))) for query in queries]).to(device)


def gradient_norm(parameters):
    return sum(p.grad.float().square().sum() for p in parameters if p.grad is not None).sqrt()


def gradient_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            group = name.split('.')[0]
            groups[group] = groups.get(group, 0.) + float(parameter.grad.float().square().sum())
    return {key: value**.5 for key, value in groups.items()}


def save_condition(output, event, chain, queries, *, meta_index, credit):
    target = output / 'conditions' / event.condition_id
    target.mkdir(parents=True, exist_ok=False)
    with gzip.open(target / 'experience.pt.gz', 'wb', compresslevel=1) as handle:
        torch.save(chain.to_record(), handle)
    with gzip.open(target / 'query.pt.gz', 'wb', compresslevel=1) as handle:
        torch.save(queries, handle)
    save_file({k: v.contiguous() for k, v in chain.states[-1].items()}, str(target / 'adapted.safetensors'))
    write_json_atomic(target / 'record.json', dict(event=event.as_dict(), meta_index=meta_index,
        compilation=chain.metrics, fm={k: v for k, v in credit['fm'].items() if k != 'cotangents'},
        pg={k: v for k, v in credit['pg'].items() if k != 'cotangent'},
        shared_parameter_version=event.update, complete=True))


def one_condition(runtime, data, runner, baseline, event, args):
    teacher = runtime.teacher(event.task_id, event.teacher_demo)
    queries, pg, features = [], None, None
    if event.update < 128:
        with torch.no_grad(), autocast(runtime.device):
            q = runtime.compiler.initial(teacher)
            chain = Chain(states=[cpu_state(runtime.compiler.decode(q))])
    else:
        task = next(t for t in runner.contract['tasks'] if t['global_task_id'] == event.task_id)
        chain = runner.adapt(task, teacher, event.seed, event.query_states2, masked=event.masked_experience)
        for position, state_id in enumerate(event.query_states2):
            query_seed = int(__import__('numpy').random.SeedSequence([event.seed, position, 0xA5DE]).generate_state(1)[0])
            queries.append(runner.query(task, state_id, chain.states[-1], query_seed))
        features = baseline_features(chain, queries, runtime.device)
        with torch.no_grad():
            predictions = baseline(features)
        advantages = [float(q['row']['success']) - float(v) for q, v in zip(queries, predictions)]
        pg = pg_credit(runtime, chain.states[-1], task, queries, advantages, microbatch=args.pg_microbatch)
        pg['baseline_predictions'] = predictions.tolist()
    batch = data.query_batch(event, runtime.processor)
    fm = fm_credit(runtime, chain.states, batch, seed=event.seed, microbatch=args.microbatch)
    replay = runtime.replay(teacher, chain, masked=event.masked_experience)
    return dict(chain=chain, queries=queries, fm=fm, pg=pg, replay=replay, baseline_features=features)


def accumulate(parameters, gradients, destination):
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            destination[parameter].add_(gradient.detach())


def condition_backward(result, parameters, beta, fm_buffer, pg_buffer):
    states, fm, pg = result['replay'], result['fm']['cotangents'], result['pg']
    if pg is None or beta is not None:
        surrogate(states, fm, None if pg is None else pg['cotangent'], beta or 0.).backward()
    else:
        gradients = torch.autograd.grad(surrogate(states, fm), parameters, allow_unused=True, retain_graph=True)
        accumulate(parameters, gradients, fm_buffer)
        gradients = torch.autograd.grad(pg_surrogate(states[-1], pg['cotangent']), parameters, allow_unused=True)
        accumulate(parameters, gradients, pg_buffer)


def calibrate_and_reduce(parameters, context, beta, fm_buffer, pg_buffer):
    stats = {}
    if fm_buffer is not None:
        for p in parameters:
            p.grad = fm_buffer[p]
        sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 1024**2)
        fm_norm = float(gradient_norm(parameters))
        fm_global = [p.grad.clone() for p in parameters]
        for p in parameters:
            p.grad = pg_buffer[p]
        sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 1024**2)
        pg_norm = float(gradient_norm(parameters))
        if fm_norm > 0 and pg_norm > 0:
            beta = min(1., .25 * fm_norm / pg_norm)
        for p, fm in zip(parameters, fm_global):
            p.grad = fm + (beta or 0.) * p.grad
        stats.update(fm_chi_gradient_norm=fm_norm, pg_chi_gradient_norm=pg_norm)
    else:
        sum_writer_gradients(parameters, world_size=context.world_size, bucket_bytes=64 * 1024**2)
    return beta, stats


def update(runtime, data, runner, baseline, value_optimizer, optimizer, scheduler, context, index, beta, args):
    parameters = tuple(runtime.compiler.parameters())
    optimizer.zero_grad(set_to_none=True)
    calibration = index >= 128 and beta is None
    fm_buffer = {p: torch.zeros_like(p) for p in parameters} if calibration else None
    pg_buffer = {p: torch.zeros_like(p) for p in parameters} if calibration else None
    records, value_batches = [], []
    started = time.monotonic()
    for event in data.events(index):
        if event.position % context.world_size != context.rank:
            continue
        result = one_condition(runtime, data, runner, baseline, event, args)
        condition_backward(result, parameters, beta, fm_buffer, pg_buffer)
        record = dict(event=event.as_dict(), fm_loss=result['fm']['weighted_loss'],
                      stage_losses=result['fm']['stage_losses'], keep=result['fm']['keep'])
        if index >= 128:
            save_condition(args.output, event, result['chain'], result['queries'], meta_index=index - 127, credit=result)
            record.update(compilation=result['chain'].metrics,
                          pg={k: v for k, v in result['pg'].items() if k != 'cotangent'})
            targets = torch.tensor(result['pg']['returns'], dtype=torch.float32, device=context.device)
            value_batches.append((result['baseline_features'].detach(), targets))
        records.append(record)
        del result
    beta, stats = calibrate_and_reduce(parameters, context, beta, fm_buffer, pg_buffer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    stats.update(beta=beta, combined_gradient_norm=float(norm), gradient_groups=gradient_groups(runtime.compiler))
    optimizer.step()
    scheduler.step()
    # Every advantage above used the old baseline, before this batch's labels.
    value_optimizer.zero_grad(set_to_none=True)
    for features, targets in value_batches:
        ((baseline(features) - targets).square().sum() / 8).backward()
    if index >= 128:
        sum_writer_gradients(tuple(baseline.parameters()), world_size=context.world_size)
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1., error_if_nonfinite=True)
        value_optimizer.step()
    gathered = [None] * context.world_size if context.is_main else None
    if context.world_size > 1:
        dist.gather_object(records, gathered, dst=0)
    else:
        gathered = [records]
    if context.is_main:
        all_records = sorted([r for rows in gathered for r in rows], key=lambda r: r['event']['position'])
        if len(all_records) != 4:
            raise ValueError('physical topology changed the four-condition update')
        append_jsonl(args.output / 'metrics.jsonl', dict(update=index + 1, phase='warm' if index < 128 else 'meta',
            meta_update=max(0, index - 127), wall_seconds=time.monotonic() - started, records=all_records, **stats))
    return beta


def save_training(args, runtime, data, baseline, value_optimizer, optimizer, scheduler, context, macro, beta):
    from .contract import SCHEMA, STAGE
    data.next_update = macro
    return save_ecp_checkpoint(output_dir=args.output, macro=macro, stage=STAGE, context=context,
        model=runtime.compiler, optimizer=optimizer, scheduler=scheduler, run_contract_schema=SCHEMA,
        metrics_rows=macro, sampler_state=data.state_dict(),
        training_state=dict(beta=beta, value=baseline.state_dict(), value_optimizer=value_optimizer.state_dict()))


def train(args):
    from .contract import SCHEMA, STAGE, learning_environment
    from .data import QueryData
    from .run import register_launch, check_budget, cost_interval

    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if not 1 <= context.world_size <= 4:
        raise ValueError('this four-condition batch uses one through four useful training ranks')
    seed_everything(20261009, context)
    runtime = Runtime(args.asset_root, context.device, frame_chunk=args.frame_chunk,
                      decoder_chunk=args.decoder_chunk, experience_chunk=args.experience_chunk,
                      native_frame_chunk=args.native_frame_chunk)
    data = QueryData(args.asset_root)
    runner = Runner(runtime, learning_environment(asset_root=args.asset_root), args.physical_gpus[context.local_rank])
    baseline = ValueBaseline().to(context.device)
    optimizer = torch.optim.AdamW(runtime.compiler.parameters(), lr=3e-5, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    value_optimizer = torch.optim.AdamW(baseline.parameters(), lr=3e-5, weight_decay=0.)
    initialize_deferred_process_group(context, rendezvous_root=args.output, collective_timeout=timedelta(minutes=45))
    index, beta = 0, None
    with cost_interval(args.output.parent, 'train', args.physical_gpus, rank=context.rank):
        if args.resume:
            state = {}
            index, rows = load_ecp_checkpoint(checkpoint=args.resume, stage=STAGE, context=context,
                model=runtime.compiler, optimizer=optimizer, scheduler=scheduler, run_contract_schema=SCHEMA,
                restored_state=state, allow_world_size_change=args.allow_topology_change)
            data.load_state_dict(state['sampler_state'])
            if rows != index or data.next_update != index or scheduler.last_epoch != index:
                raise ValueError('continuous sampler/optimizer/scheduler cursor changed')
            beta = state['training_state']['beta']
            baseline.load_state_dict(state['training_state']['value'])
            value_optimizer.load_state_dict(state['training_state']['value_optimizer'])
        if context.is_main:
            register_launch(args, runtime, context, data, index)
        barrier(context)
        if not args.resume:
            save_training(args, runtime, data, baseline, value_optimizer, optimizer, scheduler, context, 0, beta)
        for update_index in range(index, 182):
            check_budget(args.output.parent)
            beta = update(runtime, data, runner, baseline, value_optimizer, optimizer, scheduler,
                          context, update_index, beta, args)
            macro = update_index + 1
            if macro in (32, 64, 96, 128) or macro > 128 and (macro - 128) % 9 == 0:
                checkpoint = save_training(args, runtime, data, baseline, value_optimizer, optimizer, scheduler,
                                           context, macro, beta)
                if context.is_main:
                    print({'checkpoint_ready': str(checkpoint), 'meta_update': max(0, macro - 128)}, flush=True)
        if context.is_main:
            write_json_atomic(args.output / 'completion.json', dict(updates=182, warm=128, meta=54,
                beta=beta, checkpoint=str(checkpoint), training_complete=True))
    runner.close()
    data.close()
    runtime.close()
    if context.world_size > 1:
        dist.destroy_process_group()
