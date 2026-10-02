"""Task-only §102 computation; removed after the registered live/stop batch."""
from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, mean_velocity_loss


def rms(value):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)


def recurrence(write, keys, h):
    """Original full M recurrence plus its pre-O content on the same fixed K/H."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        normalized = rms(h)
        # Independent per-frame linear maps are packed; ordered coverage remains sequential.
        u = (F.gelu(write.p(keys[:-1].transpose(1, 2)) + write.c(normalized[:-1]))
             * write.d(normalized[1:] - normalized[:-1])).transpose(1, 2)
        values = write.o(u.transpose(1, 2)).transpose(1, 2)
        m = h.new_zeros((write.o.out_features, keys.shape[1]), dtype=torch.float32)
        z = h.new_zeros((write.o.in_features, keys.shape[1]), dtype=torch.float32)
        for k, v, content in zip(keys[:-1], values, u, strict=True):
            m = m + (v - m @ k) @ k.T / 50
            z = z + (content - z @ k) @ k.T / 50
        return m, z


def compile_content(writer, features, *, checkpoint_targets=True):
    h, keys = features['H'], features['K']
    if set(keys) != set(writer.names) or h.shape[1:] != (50, 1024):
        raise ValueError('fixed native content lost full38/50/1024')
    public, state, memories = writer.public_state(), {}, []
    for name, write in zip(writer.names, writer.writes, strict=True):
        fn = partial(recurrence, write)
        m, z = (checkpoint(fn, keys[name], h, use_reentrant=False)
                if checkpoint_targets and torch.is_grad_enabled() else fn(keys[name], h))
        state[name + LORA_A_SUFFIX] = public[name + LORA_A_SUFFIX]
        state[name + LORA_B_SUFFIX] = public[name + LORA_B_SUFFIX] + m
        memories.append(z)
    return state, torch.stack(memories)


class CreditReader(nn.Module):
    """Exact four-head gamma; coordinate identity is key-only."""

    def __init__(self, *, targets=38, rank=128, hidden=1024, width=256, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('attention head width must divide memory width')
        self.heads, self.width = heads, width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.q = nn.Linear(hidden, width, bias=False)
            self.k = nn.Linear(width, width, bias=False)
            self.v = nn.Linear(width, width, bias=False)
            self.o = nn.Linear(width, 7, bias=False)
            nn.init.zeros_(self.o.weight)
            self.target = nn.Parameter(torch.randn(targets, width) * .02)
        positions = torch.arange(rank).float()[:, None]
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        rank_code = torch.empty(rank, width)
        rank_code[:, 0::2] = torch.sin(positions * frequency)
        rank_code[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer('rank_code', rank_code)

    def memory(self, z):
        if z.ndim == 3:
            z = z.unsqueeze(0)
        tokens = rms(z.transpose(-1, -2))
        key = self.k(tokens) + self.target[None, :, None] + self.rank_code[None, None]
        value = self.v(tokens)
        key = key.to(value.dtype)
        batch = z.shape[0]
        key = key.flatten(1, 2).reshape(batch, -1, self.heads, self.width//self.heads).transpose(1, 2)
        value = value.flatten(1, 2).reshape(batch, -1, self.heads, self.width//self.heads).transpose(1, 2)
        return key, value

    def read(self, h, memory):
        q = self.q(rms(h)).reshape(len(h), h.shape[1], self.heads, self.width//self.heads).transpose(1, 2)
        k, v = memory
        # Native efficient attention is the same scaled softmax, without dropout.
        content = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.o(content.transpose(1, 2).reshape(len(h), h.shape[1], self.width))

    def forward(self, h, z):
        return self.read(h, self.memory(z))


class ExecutionHidden(NativeFlowPrediction):
    """Capture action_out input during the one canonical real FM suffix."""

    def forward(self, sample, prepared=None):
        captured = []
        handle = self.policy.model.action_out_proj.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0]))
        try:
            velocity = super().forward(sample, prepared)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('FM must have exactly one action_out invocation')
        return velocity, captured[0]


def coupled_credit(runtime, state, z, gamma, batch, *, seed, microbatch, live, backward=True):
    """Weighted leaf VJP for full LoRA AND Z, gamma on the same parameter version."""
    total = next(len(v) for v in batch.values() if isinstance(v, torch.Tensor) and v.ndim)
    owner, cotangent, z_cotangent = ExecutionHidden(runtime.policy), {}, torch.zeros_like(z)
    losses, predictions, targets = [0., 0.], [[], []], []
    for begin in range(0, total, microbatch):
        end = min(begin+microbatch, total)
        sliced = {k: v[begin:end] if isinstance(v, torch.Tensor) and v.ndim and len(v) == total else v
                  for k, v in batch.items()}
        sample = flow_sample(runtime.policy, sliced, seed=seed, device=runtime.device,
                             random_batch=total, offset=begin)
        prepared = owner.prepare(sample)
        leaves = {k: v.detach().requires_grad_(backward) for k, v in state.items()}
        z_leaf = z.detach().requires_grad_(backward)
        with torch.set_grad_enabled(backward):
            v, h = torch.func.functional_call(owner, {'policy.'+k: v for k, v in leaves.items()},
                                              (sample, prepared), strict=False)
            r = gamma(h if live else h.detach(), z_leaf)
            lf = mean_velocity_loss(v, sample.target, 7)
            lr = mean_velocity_loss(v.detach()[..., :7] + r, sample.target[..., :7], 7)
            weight = (end-begin)/total
            if backward:
                ((lf+lr) * weight / 8).backward()
                for name, leaf in leaves.items():
                    if leaf.grad is None:
                        raise RuntimeError('full LoRA FM leaf lost its gradient')
                    if name not in cotangent:
                        cotangent[name] = leaf.grad.detach().float().clone()
                    else:
                        cotangent[name].add_(leaf.grad.detach().float())
                if z_leaf.grad is None:
                    raise RuntimeError('auxiliary Z credit was dropped')
                z_cotangent.add_(z_leaf.grad)
        losses[0] += float(lf.detach()) * weight
        losses[1] += float(lr.detach()) * weight
        if not backward:
            predictions[0].append(v.detach().float().cpu()[..., :7])
            predictions[1].append((v.detach()[..., :7] + r).float().cpu())
            targets.append(sample.target.detach().float().cpu()[..., :7])
    result = dict(L_F=losses[0], L_R=losses[1], lora_cotangent=cotangent, Z_cotangent=z_cotangent)
    if not backward:
        result.update(student=torch.cat(predictions[0]), reader=torch.cat(predictions[1]), target=torch.cat(targets))
    return result


def writer_vjp(writer, state, z, credit):
    names = [n for n, v in state.items() if v.requires_grad]
    if len(names) != 38 or any(not n.endswith(LORA_B_SUFFIX) for n in names):
        raise ValueError('exactly the 38 B factors must carry Writer FM credit')
    torch.autograd.backward([*[state[n] for n in names], z],
                            [*[credit['lora_cotangent'][n].to(state[n]) for n in names],
                             credit['Z_cotangent'].to(z)])


@contextmanager
def residual_output(policy, gamma, z):
    """Diagnostic only: add R to real7 velocity, retain native pad and sampler."""
    memory = gamma.memory(z)
    def add(module, inputs, output):
        residual = gamma.read(inputs[0], memory)
        return torch.cat((output[..., :7] + residual.to(output), output[..., 7:]), dim=-1)
    handle = policy.model.action_out_proj.register_forward_hook(add)
    try:
        yield
    finally:
        handle.remove()
