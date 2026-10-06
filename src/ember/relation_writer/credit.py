"""Same-sample full G/F FM and stop-gradient distillation through native G credit."""
from __future__ import annotations

import time

import torch

from ember.lora import validate_lora_state
from ember.operator_writer.credit import native_credit
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.runtime import autocast

from .runtime import source_query, to_device


def _slice_batch(batch, start, stop, total):
    return {key: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim
            and len(value) == total else value for key, value in batch.items()}


def _add(cotangent, leaves, gradients, weight):
    for (name, _leaf), gradient in zip(leaves.items(), gradients, strict=True):
        value = gradient.detach().float().mul(weight)
        if name in cotangent:
            cotangent[name].add_(value)
        else:
            cotangent[name] = value


def functional_feedback_credit(runtime, feedback, batch, teacher, current, indices,
                               state, event, *, microbatch, update):
    """One physical FM sample feeds source/F/G; retain no policy query graph."""
    owner = NativeFlowPrediction(runtime.policy)
    cotangent, metrics = {}, {'G_FM': 0., 'F_FM': 0., 'KD': 0., 'source_calls': 0, 'G_calls': 0}
    coefficient = .25 * min(update / 90., 1.)
    total = len(current['p'])
    if total != 28 or any(p.requires_grad for p in runtime.policy.parameters()):
        raise ValueError('feedback FM must keep 28 cross-episode queries and frozen source')
    validate_lora_state(state, runtime.lora)
    for start in range(0, total, microbatch):
        stop = min(start + microbatch, total)
        weight = (stop - start) / total
        sliced = _slice_batch(batch, start, stop, total)
        query = {key: value[start:stop] for key, value in current.items()}
        with autocast(runtime.device):
            sample = flow_sample(runtime.policy, sliced, seed=event['flow_seed'], device=runtime.device,
                                 random_batch=28, offset=start)
            prepared = owner.prepare(sample)
            h0, v0 = source_query(runtime.policy, owner, sample, prepared)
            v_f = feedback(teacher, query, h0, v0, indices)
            f_loss = mean_velocity_loss(v_f, sample.target, sample.action_width)
            (f_loss * .25 * weight).backward()
            target_f = v_f.detach()
            del v_f
            leaves = {name: value.detach().requires_grad_(True) for name, value in state.items()}
            prediction = torch.func.functional_call(owner,
                {'policy.' + name: value for name, value in leaves.items()},
                (sample, prepared), strict=False)
            fm = mean_velocity_loss(prediction, sample.target, sample.action_width)
            kd = mean_velocity_loss(prediction, target_f, sample.action_width)
            gradients = torch.autograd.grad(fm + coefficient * kd, tuple(leaves.values()))
            _add(cotangent, leaves, gradients, .25 * weight)
        for key, loss in (('G_FM', fm), ('F_FM', f_loss), ('KD', kd)):
            metrics[key] += float(loss.detach()) * weight
        metrics['source_calls'] += 1
        metrics['G_calls'] += 1
        del prediction, gradients, leaves, prepared, sample, target_f
    return cotangent, {**metrics, 'lambda': coefficient, 'same_z_tau_y': True,
                       'source_adapter_bypass': True, 'F_credit': 'LF_only', 'KD_F_stopgrad': True}


def one_condition(runtime, feedback, data, labels, event, *, microbatch, frame_chunk, update):
    from .relation_loss import relation_loss
    started = time.perf_counter()
    condition, raw, sampled = data.condition(runtime, event['task'], event['teacher_demo'])
    indices = condition[1]
    teacher = to_device(labels.teacher(event['task'], event['teacher_demo'], indices.cpu().numpy()), runtime.device)
    current = to_device(labels.queries(event['task'], event['queries']), runtime.device)
    with torch.no_grad():
        state, _ = runtime.compile(condition, frame_chunk=frame_chunk)
    compiled_at = time.perf_counter()
    batch = runtime.processor.training_batch(data.batch(event))
    cotangent, metric = functional_feedback_credit(runtime, feedback, batch, teacher, current,
        indices, state, event, microbatch=microbatch, update=update)
    functional_at = time.perf_counter()
    with torch.enable_grad():
        replay, native = runtime.compile(condition, frame_chunk=frame_chunk, retain_native=True)
        with autocast(runtime.device):
            loss, relation_metrics = relation_loss(runtime.writer.last_prediction, teacher)
            auxiliary = .1 * .25 * loss
        if set(replay) != set(cotangent) or any(not torch.isfinite(value).all() for value in cotangent.values()):
            raise ValueError('complete G FM/KD cotangent invalid')
        torch.autograd.backward((*replay.values(), auxiliary),
            (*[cotangent[name].to(replay[name]) for name in replay], torch.ones_like(auxiliary)))
    torch.cuda.synchronize(runtime.device)
    result = {'task': event['task'], 'teacher_demo': event['teacher_demo'], 'visit': event['visit'],
            'queries': len(event['queries']), 'query_demos': [q['demo'] for q in event['queries']],
            'query_frames': [q['frame'] for q in event['queries']], 'flow_seed': event['flow_seed'],
            'raw_frames': raw, 'sampled_frames': sampled, 'teacher_geometry_frames': int(teacher['valid'].sum()),
            'relation_loss': float(loss.detach()), 'relation': relation_metrics, 'relation_copies': 1,
            'condition_weight': .25, **metric, 'native_credit': native_credit(native),
            'compile_seconds': compiled_at - started, 'functional_seconds': functional_at - compiled_at,
            'replay_seconds': time.perf_counter() - functional_at,
            'seconds': time.perf_counter() - started}
    runtime.writer.last_prediction = None
    return result
