"""Frozen conditional addresses and B-only same-version credit for §116."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import default_collate

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.function_credit import paired_condition_credit
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from .conditional_read_write import delta_memory
from .effect_labels import ROOT, TEACHERS
from .native import read_native_video

PARENT = ROOT.parent / 'conditional_A_reexpression_diagnostic_20261002/Original'
B_GROUPS = ('b_key', 'b_delta', 'b_context', 'b_dynamic', 'b_out')


def label_batch(positions, task, queries):
    targets, masks = [], []
    for q in queries:
        item = positions[f'{task}:{q["demo"]}']
        by_frame = {int(i): p for i, p in zip(item['frames'], item['positions_m'], strict=True)}
        f = int(q['frame'])
        valid = np.asarray([f in by_frame and f+h in by_frame for h in range(1, 51)])
        r = np.zeros((50, 3), dtype=np.float32)
        for h in np.flatnonzero(valid):
            r[h] = (by_frame[f+int(h)+1] - by_frame[f]) / .1
        targets.append(r); masks.append(valid)
    return torch.from_numpy(np.stack(targets)), torch.from_numpy(np.stack(masks))


def raw_batch(data, task, queries):
    if set(TEACHERS[task]) & {q['demo'] for q in queries}:
        raise ValueError('teacher entered query labels')
    return default_collate([data.queries[data.rows[task][q['demo']][q['frame']]] for q in queries])


def freeze_except_B(writer):
    writer.requires_grad_(False)
    for unit in writer.conditional_targets:
        for group in B_GROUPS:
            getattr(unit, group).requires_grad_(True)
    selected = tuple(p for p in writer.parameters() if p.requires_grad)
    names = [n for n, p in writer.named_parameters() if p.requires_grad]
    if len(names) != 38 * 5 or any(not any(f'.{g}.' in n for g in B_GROUPS) for n in names):
        raise ValueError('only all38 B Value parameter sets may learn')
    return selected


@torch.no_grad()
def native_cache(runtime, data, *, create, frame_chunk=128):
    records, cached = [], {}
    for task, teachers in TEACHERS.items():
        for teacher in teachers:
            key = f'task{task:03d}_teacher{teacher:02d}'
            path = ROOT / 'native' / f'{key}.pt'
            if create:
                condition, raw, sampled = data.condition(runtime, task, teacher)
                started = time.monotonic()
                runtime.restore_identity()
                with autocast(runtime.device):
                    x, h = read_native_video(runtime.policy, runtime.writer.public_state(), runtime.writer.probe,
                        condition, runtime.writer.names, frame_chunk=frame_chunk, checkpoint_frames=False)
                    state = runtime.writer(x, h, frame_indices=condition[1], capture_mechanism=True)
                    c, d = runtime.writer.last_mechanism['c'], runtime.writer.last_mechanism['d']
                    fixed = dict(H=h, frame_indices=condition[1], context=torch.cat((c[1:], h[1:].float()), -1),
                                 dynamic=d[1:], targets={})
                    for name in runtime.writer.names:
                        origin = x[name][:-1].float()
                        a = state[name+LORA_A_SUFFIX]
                        s = runtime.writer.last_mechanism['targets'][name]['S']
                        fixed['targets'][name] = dict(A=a,
                            K=F.normalize(F.linear(origin, a).float(), dim=-1, eps=1e-6),
                            delta_z=F.linear(origin, s))
                path.parent.mkdir(exist_ok=True)
                def cpu(item):
                    return {k: cpu(v) if isinstance(v, dict) else v.cpu().contiguous() for k, v in item.items()}
                torch.save(cpu(fixed), path)
                rec = dict(task=task, teacher=teacher, path=str(path), raw_frames=raw, sampled_frames=sampled,
                    seconds=time.monotonic()-started, frame_chunk=frame_chunk,
                    fields_read=['obs/agentview_rgb', 'obs/eye_in_hand_rgb'], public_native_reads=1,
                    parent_public_frozen=True, no_teacher_privileged_inputs=True)
                write_json_atomic(path.with_suffix('.json'), rec)
                del x, h, state, fixed
                runtime.writer.last_mechanism = None
            item = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if set(item['targets']) != set(runtime.writer.names) or item['H'].shape[1:] != (50, 1024):
                raise ValueError('fixed native cache requires all38 actual sites')
            def gpu(value):
                return {k: gpu(v) if isinstance(v, dict) else v.to(runtime.device) for k, v in value.items()}
            cached[(task, teacher)] = gpu(item)
            records.append(file_record(path))
    if create:
        write_json_atomic(ROOT/'native/manifest.json', dict(records=records, native_reads=8,
            same_cache_both_arms=True, fields=['H','frame_indices','context','dynamic','A','K','delta_z'],
            B_Value_or_M_cached=False))
    return cached


def target_B(writer, unit, name, fixed):
    target = fixed['targets'][name]
    values = unit.b_values(target['K'], target['delta_z'], fixed['context'], fixed['dynamic'])
    return writer.public_state()[name+LORA_B_SUFFIX] + delta_memory(values, target['K'], unit.b_out.out_features)


@torch.no_grad()
def compile_B(runtime, fixed):
    with autocast(runtime.device):
        state = {}
        for name, unit in zip(runtime.writer.names, runtime.writer.conditional_targets, strict=True):
            state[name+LORA_A_SUFFIX] = fixed['targets'][name]['A']
            state[name+LORA_B_SUFFIX] = target_B(runtime.writer, unit, name, fixed)
    validate_lora_state(state, runtime.lora)
    return state


def step(runtime, cached, data, positions, entry, arm, optimizer, parameters, micro):
    started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    rows = []
    for task, teachers in TEACHERS.items():
        event = entry[str(task)]
        batch = runtime.processor.training_batch(raw_batch(data, task, event['queries']))
        labels, valid = label_batch(positions, task, event['queries'])
        states = [compile_B(runtime, cached[(task, teacher)]) for teacher in teachers]
        runtime.restore_identity()
        with autocast(runtime.device):
            credits = paired_condition_credit(runtime.policy, *states, runtime.lora, batch,
                seed=event['flow_seed'], device=runtime.device, random_batch=28, offset=0,
                microbatch=micro, condition_weight=1/8,
                **(dict(effect_target=labels, effect_valid=valid) if arm == 'J' else {}))
        for teacher,credit in zip(teachers,credits,strict=True):
            with autocast(runtime.device):
                # The same parameter version is replayed target by target; A stays frozen.
                for name, unit in zip(runtime.writer.names, runtime.writer.conditional_targets, strict=True):
                    b = target_B(runtime.writer, unit, name, cached[(task, teacher)])
                    b.backward(credit['lora_cotangent'][name+LORA_B_SUFFIX].to(b))
            rows.append(dict(task=task, teacher=teacher, weight=1/8, flow_seed=event['flow_seed'],
                queries=event['queries'], action_FM=credit['action_flow_loss'], effect_FM=credit['effect_flow_loss'],
                valid_effect_positions=int(valid.sum()), missing_effect_positions=int((~valid).sum())))
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    group_norms = {g: float(torch.stack([getattr(u,g).weight.grad.float().norm()
        for u in runtime.writer.conditional_targets]).norm()) for g in B_GROUPS}
    if any(p.grad is not None for p in runtime.writer.parameters() if not p.requires_grad):
        raise ValueError('fixed parent paths acquired gradients')
    optimizer.step(); torch.cuda.synchronize()
    return dict(rows=rows, seconds=time.monotonic()-started, microbatch=micro,
        unclipped_gradient_norm=float(norm), B_gradient_groups=group_norms,
        allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
