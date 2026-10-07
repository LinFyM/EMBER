"""Complete query and teaching credit via finite C/K/V cotangent replay."""
from __future__ import annotations

from contextlib import nullcontext
import time
import torch

from ember.operator_writer.native import read_native_video
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.runtime import autocast


def _flat(values):
    return tuple(value for pair in values for value in pair)


def _norm(values):
    terms = [value.detach().float().square().sum() for value in values if value is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def native_memory(runtime, condition, *, frame_chunk: int, backward: bool):
    model = runtime.controller
    with torch.set_grad_enabled(backward), autocast(runtime.device):
        x, h = read_native_video(runtime.policy, model.public_state(), model.probe,
                                 condition, model.names, frame_chunk=frame_chunk,
                                 checkpoint_frames=backward)
        del x
        return model.encode(h, condition[1])


def _query_credit(runtime, batch, event, microbatch, memory, leaves, cotangents, check_initial):
    model, device = runtime.controller, runtime.device
    prepared_leaves = tuple(zip(leaves[::2], leaves[1::2], strict=True)) if leaves else None
    owner = NativeFlowPrediction(runtime.policy)
    state = {"policy." + name: value for name, value in model.public_state().items()}
    total = len(event["queries"])
    if total != 28 or not 0 < microbatch <= total:
        raise ValueError("native-video condition must keep all 28 independent queries")
    loss_value = 0.
    baseline_max = None
    hook_counts = None
    for first in range(0, total, microbatch):
        last = min(first + microbatch, total)
        sliced = {name: value[first:last] if isinstance(value, torch.Tensor) and value.ndim and len(value) == total
                  else value for name, value in batch.items()}
        sample = flow_sample(runtime.policy, sliced, seed=event["flow_seed"], device=device,
                             random_batch=28, offset=first, noise_endpoint=False)
        with autocast(device):
            prefix = owner.prepare(sample)
            scope = (model.execution_scope(runtime.policy, memory, key_values=prepared_leaves)
                     if model.arm == "V" else nullcontext())
            with scope as hooks:
                prediction = torch.func.functional_call(owner, state, (sample, prefix), strict=False)
                loss = mean_velocity_loss(prediction, sample.target, sample.action_width)
                if check_initial and first == 0:
                    # Same actual FM query, beta and frozen prefix. Reader is zero only at this origin.
                    initial_prediction = prediction.detach()
                (loss * ((last - first) / 28) * .25).backward()
                if hooks is not None:
                    hook_counts = list(hooks.calls)
                    if hook_counts != [1] * 18:
                        raise ValueError("actual query did not consume every post-block Reader exactly once")
            if check_initial and first == 0:
                with torch.no_grad():
                    baseline = torch.func.functional_call(owner, state, (sample, prefix), strict=False)
                baseline_max = float((initial_prediction.float() - baseline.float()).abs().max())
                if baseline_max > 1e-3:
                    raise ValueError(f"zero-output Reader changed the MT initial FM function: {baseline_max}")
        loss_value += float(loss.detach()) * (last - first) / 28
        for accumulated, leaf in zip(cotangents, leaves, strict=True):
            if leaf.grad is None:
                raise ValueError("Reader lost an actual K/V query credit path")
            accumulated.add_(leaf.grad.float())
            leaf.grad = None
        del prediction, loss, prefix, sample
    return loss_value, baseline_max, hook_counts


def _replay_credit(runtime, memory, condition, cotangents, frame_chunk):
    # Restore the K/V parameter path once, then the entire legal teaching path.
    model, device = runtime.controller, runtime.device
    c_leaf = memory.detach().requires_grad_()
    with autocast(device):
        projected = _flat(model.prepare_memory(c_leaf))
    torch.autograd.backward(projected, tuple(value.to(output.dtype) for value, output
                                            in zip(cotangents, projected, strict=True)))
    if c_leaf.grad is None:
        raise ValueError("Reader K/V projections detached the encoder credit")
    memory_credit_norm = _norm((c_leaf.grad,))
    common = tuple(model.common.values)
    before = tuple(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in common)
    replay = native_memory(runtime, condition, frame_chunk=frame_chunk, backward=True)
    replay.backward(c_leaf.grad.to(replay.dtype))
    teacher_beta_norm = _norm(p.grad - old if p.grad is not None else None
                              for p, old in zip(common, before, strict=True))
    return memory_credit_norm, teacher_beta_norm


def condition_credit(runtime, data, event, *, microbatch: int, frame_chunk: int,
                     check_initial: bool = False) -> dict:
    """One globally weighted task; no optimizer mutation until all replay ends."""
    model, device = runtime.controller, runtime.device
    start_time = time.monotonic()
    condition, memory = None, None
    leaves, cotangents = (), ()
    raw_count = sampled_count = 0
    if model.arm == "V":
        condition, raw_count, sampled_count = data.condition(runtime, event["task"], event["teacher_demo"])
        with torch.no_grad(), autocast(device):
            memory = native_memory(runtime, condition, frame_chunk=frame_chunk, backward=False)
            projected = model.prepare_memory(memory)
        leaves = tuple(value.detach().requires_grad_() for value in _flat(projected))
        cotangents = tuple(torch.zeros_like(value, dtype=torch.float32) for value in leaves)
        del projected
    batch = runtime.processor.training_batch(data.batch(event))
    loss_value, baseline_max, hook_counts = _query_credit(runtime, batch, event, microbatch,
                                                        memory, leaves, cotangents, check_initial)
    memory_credit_norm = teacher_beta_norm = 0.
    if model.arm == "V":
        memory_credit_norm, teacher_beta_norm = _replay_credit(runtime, memory, condition, cotangents, frame_chunk)
    return {"event": event, "flow_loss": loss_value, "raw_teacher_frames": raw_count,
            "sampled_teacher_frames": sampled_count, "memory_tokens": 0 if memory is None else memory.shape[1],
            "kv_cotangent_norm": _norm(cotangents), "memory_cotangent_norm": memory_credit_norm,
            "teacher_beta_credit_norm": teacher_beta_norm, "initial_MT_max_difference": baseline_max,
            "reader_hook_counts": hook_counts, "condition_seconds": time.monotonic() - start_time}
