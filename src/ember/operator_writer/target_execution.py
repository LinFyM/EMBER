"""Extra physical ranks execute target VJPs for the same four conditions.

Native/interpreter graphs stay with their condition owners. Remote targets return
both complete factors and cotangents of the actual X/H/c/d boundary. Public and
target parameter gradients join the canonical global SUM before one Adam step.
"""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.writer.runtime import autocast


def target_partition(writer, *, owners: int, world: int) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
    """Balance target arithmetic, assigning one physical share to each helper."""
    if not 1 <= owners <= world <= 6:
        raise ValueError("target execution supports the registered physical topology")
    bins, loads = [[] for _ in range(world - owners + 1)], [0] * (world - owners + 1)
    weights = [owners] + [1] * (world - owners)
    costs = {i: unit.a_x.in_features + unit.b_out.out_features + 6144
             for i, unit in enumerate(writer.conditional_targets)}
    for index in sorted(costs, key=lambda i: (-costs[i], i)):
        selected = min(range(len(bins)), key=lambda b: (loads[b] / weights[b], b))
        bins[selected].append(index)
        loads[selected] += costs[index]
    return (tuple(sorted(bins[0])),
            {owners + i: tuple(sorted(group)) for i, group in enumerate(bins[1:])})


def compile_targets(writer, indices, x, h, c, d, q=None):
    """Use the canonical target computation and its existing activation replay."""
    common, result = writer.public_state(), {}
    for index in indices:
        name, unit = writer.names[index], writer.conditional_targets[index]
        inputs = (common[name + LORA_A_SUFFIX], common[name + LORA_B_SUFFIX], x[name], h, c, d)
        if q is not None:
            inputs = (*inputs, q)
        a, b, _, _ = (checkpoint(unit, *inputs, use_reentrant=False, preserve_rng_state=False)
                      if torch.is_grad_enabled() else unit(*inputs))
        result[name + LORA_A_SUFFIX], result[name + LORA_B_SUFFIX] = a, b
    return result


_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


def send_tensors(tensors, peer):
    """Start same-node transfers; callers compute their local heads concurrently."""
    tensors = [value.detach().contiguous() for value in tensors]
    header = torch.tensor([[value.ndim, _DTYPES.index(value.dtype),
                            *value.shape, *([0] * (4 - value.ndim))] for value in tensors],
                          dtype=torch.int64, device=tensors[0].device)
    values = [header, *tensors]
    # Match header and payload batches on both peers. Mixing batched sends with
    # unbatched NCCL receives creates different lazy communicators and hangs.
    pending = dist.batch_isend_irecv([dist.P2POp(dist.isend, header, peer)])
    pending += dist.batch_isend_irecv([dist.P2POp(dist.isend, value, peer) for value in tensors])
    return values, pending  # Retain storage until every transfer finishes.


def finish_sends(pending):
    for work in pending[1]:
        work.wait()


def receive_tensors(count, peer, device):
    header = torch.empty((count, 6), dtype=torch.int64, device=device)
    for work in dist.batch_isend_irecv([dist.P2POp(dist.irecv, header, peer)]):
        work.wait()
    tensors = []
    for row in header.cpu().tolist():
        ndim, dtype, *shape = row
        if not 1 <= ndim <= 4 or not 0 <= dtype < len(_DTYPES) or min(shape[:ndim]) <= 0:
            raise ValueError("target transfer shape or dtype changed")
        value = torch.empty(shape[:ndim], dtype=_DTYPES[dtype], device=device)
        tensors.append(value)
    for work in dist.batch_isend_irecv([dist.P2POp(dist.irecv, value, peer) for value in tensors]):
        work.wait()
    return tensors


class _RemoteTargets(torch.autograd.Function):
    @staticmethod
    def forward(ctx, call, *inputs):
        ctx.call = call
        finish_sends(call.pop("pending"))
        values = receive_tensors(len(call["names"]), call["peer"], inputs[0].device)
        call["seconds"] += time.perf_counter() - call["wait_started"]
        return tuple(values)

    @staticmethod
    def backward(ctx, *cotangents):
        call = ctx.call
        started = time.perf_counter()
        finish_sends(send_tensors(cotangents, call["peer"]))
        gradients = receive_tensors(call["inputs"], call["peer"], cotangents[0].device)
        call["seconds"] += time.perf_counter() - started
        return (None, *gradients)


class TargetShardClient:
    """One condition owner overlaps native/local targets with remote target work."""
    def __init__(self, writer, *, owners: int, world: int):
        self.local, self.remote = target_partition(writer, owners=owners, world=world)
        self.calls = []

    def __call__(self, writer, x, h, c, d, q=None):
        calls = []
        for peer, indices in self.remote.items():
            inputs = (h, c, d, *((q,) if q is not None else ()), *(x[writer.names[i]] for i in indices))
            names = tuple(writer.names[i] + suffix for i in indices
                          for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX))
            call = {"peer": peer, "names": names, "inputs": len(inputs),
                    "pending": send_tensors(inputs, peer), "seconds": 0.}
            calls.append((call, inputs))
        result = compile_targets(writer, self.local, x, h, c, d, q)
        for call, inputs in calls:
            call["wait_started"] = time.perf_counter()
            result.update(zip(call["names"], _RemoteTargets.apply(call, *inputs), strict=True))
            self.calls.append(call)
        return result

    def record(self):
        return {"strategy": "automatic_target_shards", "local_targets": len(self.local),
                "helper_targets": {str(peer): len(indices) for peer, indices in self.remote.items()},
                "remote_wait_seconds": sum(call["seconds"] for call in self.calls),
                "credit": "same-version complete X/H/c/d/q and public/target parameter VJP"}


def serve_target_shards(runtime, *, owners: int, world: int):
    """Two real compiles per condition, then return the exact replay cotangents."""
    writer, device = runtime.writer, runtime.device
    _, shards = target_partition(writer, owners=owners, world=world)
    indices = shards[dist.get_rank()]
    for replay in (False, True):
        for owner in range(owners):
            inputs = receive_tensors(3 + int(getattr(writer, "gamma", None) is not None) + len(indices), owner, device)
            if replay:
                inputs = [value.requires_grad_() for value in inputs]
            h, c, d, *xs = inputs
            q = xs.pop(0) if getattr(writer, "gamma", None) is not None else None
            x = {writer.names[i]: value for i, value in zip(indices, xs, strict=True)}
            with torch.set_grad_enabled(replay), autocast(device):
                state = compile_targets(writer, indices, x, h, c, d, q)
            finish_sends(send_tensors(tuple(state.values()), owner))
            if replay:
                cotangents = receive_tensors(len(state), owner, device)
                torch.autograd.backward(tuple(state.values()), tuple(cotangents))
                finish_sends(send_tensors([value.grad if value.grad is not None else torch.zeros_like(value)
                                          for value in inputs], owner))
            del state, inputs, x, h, c, d, xs
