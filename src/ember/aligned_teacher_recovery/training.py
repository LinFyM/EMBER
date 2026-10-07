"""Finite direct A/B learning using canonical native FM and complete ECP owners."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import os
import socket
import time
import traceback

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import checkpoint_macro, load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_eval_contract import (git_state, git_state_is_clean_pushed_or_frozen_authority,
    inspect_source_checkpoint, load_evaluation_authorities)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json, source_reference_matches, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (initialize_distributed, initialize_deferred_process_group,
                                    load_policy, seed_everything)
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import DirectLoRAParameters
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast
from .data import TeacherData, BLOCK, QUERIES

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/data1/user/ymdai/ember_runs/aligned_teacher_recovery_20261008')
MT = Path('/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc/checkpoints/step_00000300/lora.safetensors')
SCHEMA = 'aligned_teacher_recovery_training_v1'


def gather(value, context):
    if context.world_size == 1:
        return [value]
    rows = [None] * context.world_size
    dist.all_gather_object(rows, value)
    return rows


def tree_slice(value, first, last):
    if isinstance(value, torch.Tensor):
        return value[first:last]
    return type(value)(tree_slice(item, first, last) for item in value)


def tree_concat(values):
    if isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=0)
    return type(values[0])(tree_concat(items) for items in zip(*values, strict=True))


def actual_flow(policy, batch, event, device):
    """Generate four original28-query random panels before physical redivision."""
    samples = []
    for block in event['blocks']:
        first = block['start']
        part = {key: value[first:first + BLOCK] if isinstance(value, torch.Tensor)
                and value.ndim and len(value) == QUERIES else value for key, value in batch.items()}
        samples.append(flow_sample(policy, part, seed=block['flow_seed'], device=device,
                                   random_batch=BLOCK, offset=0, noise_endpoint=False))
    return FlowSample(tree_concat([sample.arguments for sample in samples]),
                      torch.cat([sample.target for sample in samples]), samples[0].action_width)


@dataclass
class Session:
    context: object
    model: DirectLoRAParameters
    policy: object
    processor: object
    data: TeacherData
    optimizer: object
    output: Path
    contract: dict
    microbatch: int


def prepare(args):
    git = git_state(REPO)
    if git['branch'] or not git_state_is_clean_pushed_or_frozen_authority(git):
        raise ValueError('GPU consumers require clean pushed detached source')
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if os.environ.get('NCCL_P2P_DISABLE') != '1':
        raise ValueError('explicit BCI NCCL contract required')
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    asset = args.asset_root
    spec = read_json(asset / 'configs/operator_read_write_v1/learning_spec.json')['source']
    authorities = load_evaluation_authorities(asset / spec['evaluation_config'], asset)
    path = asset / spec['checkpoint']
    source = inspect_source_checkpoint(authorities, path.parent.parent, path, evaluation_mode='formal')
    policy = load_policy(Path(source['model_path']), authorities.source_base_config, context.device).to(context.device)
    lora = load_pi05_lora_contract(asset / 'configs/pi05_lora_rank128_aligned.json')
    prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    weights = {name: value.float() for name, value in load_file(str(MT), device='cpu').items()}
    validate_lora_state(weights, lora)
    copy_task_lora_state_(policy, weights, lora)
    model = DirectLoRAParameters(weights).to(context.device).train()
    if any(p.requires_grad for p in policy.parameters()) or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('source freeze or FP32 A/B changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
    processor = Pi05LiberoProcessor(read_json(asset / spec['normalization'])['stats'],
                                    asset / spec['tokenizer'], 200, str(context.device))
    data = TeacherData(asset, args.task, spec['data_protocol'])
    group = 'training' if args.kind == 'formal' else 'engineering'
    output = ROOT / group / f'task_{args.task:04d}' / 'attempts' / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    physical = {'rank': context.rank, 'numa_node': context.numa_node,
                'cpu_affinity': list(context.cpu_affinity or ()),
                'gpu_uuid': str(torch.cuda.get_device_properties(context.local_rank).uuid)}
    topology = {'host': socket.gethostname(), 'world_size': context.world_size,
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'ranks': gather(physical, context), 'microbatch': args.microbatch,
                'query_partition': 'contiguous floor(112*rank/world) intervals; summed global mean'}
    contract = {'schema_version': SCHEMA, 'task': args.task, 'kind': args.kind, 'source': source,
        'mt_weights': str(MT), 'lora': lora.to_dict(), 'names': list(model.names), 'training_git': git,
        'topology': topology, 'scheduler': None, 'source_trainable': 0,
        'learning_dtype': 'float32', 'optimizer': {'name': 'AdamW', 'lr': 1e-4, 'betas': [.9, .95],
            'eps': 1e-8, 'weight_decay': 1e-4, 'clip': 1.0}, 'sampler': data.sampler_state(),
        'loss': 'canonical NativeFlowPrediction/full50x7 mean_velocity_loss, no padding mask',
        'parameters': sum(p.numel() for p in model.parameters()),
        'information_wall': 'train-only own dualRGB+8state+exactL; actions only FM targets; no Writer/video input'}
    if context.is_main:
        if (output / 'run_contract.json').exists():
            raise ValueError('attempt already registered; never overwrite provenance')
        write_json_atomic(output / 'run_contract.json', contract)
    return Session(context, model, policy, processor, data, optimizer, output, contract, args.microbatch)


def one_update(session, update, *, optimize=True):
    started = time.monotonic()
    c = session.context
    event = session.data.event(update)
    sample = actual_flow(session.policy, session.processor.training_batch(session.data.batch(event)), event, c.device)
    if c.is_main:
        path = session.output / 'flow_events'
        path.mkdir(exist_ok=True)
        target = path / f'update_{update + 1:04d}'
        if not target.with_suffix('.pt').exists():
            torch.save({'event': event, 'noise': sample.arguments[-2].detach().cpu(),
                        'time': sample.arguments[-1].detach().cpu()}, target.with_suffix('.pt'))
    session.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(c.device)
    owner = NativeFlowPrediction(session.policy)
    state = {'policy.' + name: value for name, value in session.model().items()}
    start, end = QUERIES * c.rank // c.world_size, QUERIES * (c.rank + 1) // c.world_size
    loss_sum, error = 0., None
    try:
        for first in range(start, end, session.microbatch):
            last = min(first + session.microbatch, end)
            part = FlowSample(tree_slice(sample.arguments, first, last), sample.target[first:last], sample.action_width)
            with autocast(c.device):
                prepared = owner.prepare(part)
                prediction = torch.func.functional_call(owner, state, (part, prepared), strict=False)
                loss = mean_velocity_loss(prediction, part.target, part.action_width)
                (loss * ((last - first) / QUERIES)).backward()
            loss_sum += float(loss.detach()) * (last - first) / QUERIES
            del prediction, loss, prepared, part
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, c) if value]
    if failures:
        raise RuntimeError('actual teacher FM failed: ' + '\n'.join(failures))
    parameters = tuple(session.model.parameters())
    sum_writer_gradients(parameters, world_size=c.world_size, bucket_bytes=64 * 2**20)
    groups = {}
    for group in ('lora_A', 'lora_B'):
        terms = [p.grad.detach().float().square().sum() for name, p in zip(session.model.names, parameters, strict=True)
                 if group in name and p.grad is not None]
        groups[group] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
    if any(p.grad is None for p in parameters) or not all(value > 0 for value in groups.values()):
        raise ValueError('complete A/B native FM credit missing')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    if optimize:
        session.optimizer.step()
        session.data.next_update = update + 1
    torch.cuda.synchronize(c.device)
    losses = gather(loss_sum, c)
    memory = gather({'rank': c.rank, 'allocated_GiB': torch.cuda.max_memory_allocated(c.device) / 2**30,
        'reserved_GiB': torch.cuda.max_memory_reserved(c.device) / 2**30}, c)
    record = {'task': session.data.task, 'update': update + 1, 'queries': QUERIES, 'event': event,
        'flow_loss': sum(losses), 'gradient_groups': groups, 'gradient_norm_before_clip': norm,
        'memory': memory, 'seconds': time.monotonic() - started, 'optimizer_applied': optimize}
    if c.is_main:
        append_jsonl(session.output / 'metrics.jsonl', record)
        print(json.dumps({k: record[k] for k in ('task', 'update', 'flow_loss', 'seconds', 'memory')}), flush=True)
    return record


def save(session, macro):
    return save_ecp_checkpoint(output_dir=session.output, macro=macro,
        stage=f'aligned_task_{session.data.task}', context=session.context, model=session.model,
        optimizer=session.optimizer, scheduler=None, run_contract_schema=SCHEMA, metrics_rows=macro,
        sampler_state=session.data.sampler_state(), training_state={'origin': session.contract,
            'topology': session.contract['topology']})


def restore_or_origin(session, args):
    if args.resume is None:
        if args.kind == 'formal':
            save(session, 0)
        return 0
    macro = checkpoint_macro(args.resume)
    trainer = torch.load(args.resume / 'trainer_state.pt', map_location='cpu', weights_only=False)
    origin = trainer['training_state']['origin']
    if (origin['task'] != args.task or origin['kind'] != args.kind or origin['mt_weights'] != str(MT)
            or not source_reference_matches(origin['source'], session.contract['source'])
            or origin['names'] != session.contract['names']):
        raise ValueError('complete checkpoint source/model/task origin changed')
    restored = {}
    step, rows = load_ecp_checkpoint(checkpoint=args.resume, stage=f'aligned_task_{args.task}',
        context=session.context, model=session.model, optimizer=session.optimizer, scheduler=None,
        run_contract_schema=SCHEMA, expected_sampler_state=session.data.sampler_state(macro),
        restored_state=restored, allow_world_size_change=True)
    session.data.next_update = step
    if session.context.is_main:
        prefix = (args.resume.parent.parent / 'metrics.jsonl').read_text().splitlines()[:rows]
        if len(prefix) != rows:
            raise ValueError('resume lost its retained metrics prefix')
        (session.output / 'metrics.jsonl').write_text('\n'.join(prefix) + ('\n' if prefix else ''))
        write_json_atomic(session.output / 'resume_provenance.json', {'checkpoint': str(args.resume),
            'previous_origin': origin, 'new_reading_git': session.contract['training_git'],
            'previous_topology': trainer['training_state']['topology'],
            'current_topology': session.contract['topology'], 'restored': restored,
            'new_ranks_rng': 'seed_everything(7,current context); retained ranks restore saved RNG',
            'logical_queries_and_four_block_flow_stream_unchanged': True})
    return step


def profile(s, args):
    if args.task != 38 or args.resume is not None or s.context.world_size != 1:
        raise ValueError('profile is only the first task38 batch on one physical consumer')
    profiles = []
    for micro in (28, 56, 112):
        s.microbatch = micro
        try:
            result = one_update(s, 0, optimize=False)
            profiles.append({'microbatch': micro, 'passed': True, 'actual': result})
        except Exception as error:
            profiles.append({'microbatch': micro, 'passed': False, 'error': str(error)})
        s.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    if s.context.is_main:
        write_json_atomic(s.output / 'profiles.json', profiles)


def finish(s, args, macro, started):
    if not s.context.is_main:
        return
    dtypes = {str(value.dtype) for state in s.optimizer.state.values()
              for key, value in state.items() if key != 'step' and isinstance(value, torch.Tensor)}
    steps = {int(state['step']) for state in s.optimizer.state.values()}
    if dtypes != {'torch.float32'} or steps != {macro}:
        raise ValueError('FP32 optimizer state/update clocks changed')
    write_json_atomic(s.output / 'completion.json', {'status': 'complete' if macro == 480 or args.kind == 'smoke'
        else 'complete_checkpoint_boundary', 'task': args.task, 'kind': args.kind,
        'completed_updates': macro, 'queries': macro * QUERIES, 'scheduler': None,
        'optimizer_dtypes': sorted(dtypes), 'source_trainable': 0,
        'seconds_after_setup': time.monotonic() - started,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'git': s.contract['training_git']})


def train(s, args, started):
    macro = restore_or_origin(s, args)
    allowed = (320, 480) if args.kind == 'formal' else (2,) if args.resume is None else (4,)
    if args.stop_after not in allowed or args.kind == 'smoke' and (args.task != 38 or args.resume and macro != 2):
        raise ValueError('registered formal/smoke endpoint changed')
    for step in range(macro, args.stop_after):
        one_update(s, step)
        macro = step + 1
        if macro in (160, 320, 480) or args.kind == 'smoke' and macro == args.stop_after:
            save(s, macro)
    finish(s, args, macro, started)


def run(args):
    s = prepare(args)
    started = time.monotonic()
    try:
        if args.kind == 'profile':
            profile(s, args)
        else:
            train(s, args, started)
    except Exception:
        if s.context.is_main:
            write_json_atomic(s.output / 'failure.json', {'error': traceback.format_exc()})
        raise
    finally:
        s.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()
