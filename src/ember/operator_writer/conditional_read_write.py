"""Causal dynamic interpretation and final-A-associated two-sided compilation.

Owns the new §13 computation, under the canonical OperatorReadWrite runtime.
Historical registered models remain readable through their frozen contracts.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def rms_zero(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)


def position_code(indices: torch.Tensor, width: int) -> torch.Tensor:
    frequency = torch.exp(torch.arange(0, width, 2, device=indices.device).float()
                          * (-math.log(10000.0) / width))
    phase = indices.float()[:, None] * frequency[None]
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


class InterpretationLayer(nn.Module):
    """Same causal Q/K selection, independent context/dynamic Value streams."""
    def __init__(self) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(1024)
        self.q = nn.Linear(1024, 1024)
        self.k = nn.Linear(1024, 1024)
        self.context_v = nn.Linear(1024, 1024)
        self.context_o = nn.Linear(1024, 1024)
        self.ffn_norm = nn.LayerNorm(1024)
        self.context_ffn = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
        self.dynamic_v = nn.Linear(1024, 1024, bias=False)
        self.dynamic_o = nn.Linear(1024, 1024, bias=False)
        self.dynamic_in = nn.Linear(1024, 4096, bias=False)
        self.dynamic_out = nn.Linear(4096, 1024, bias=False)
        self.dynamic_gate = nn.Linear(1024, 4096)

    def forward(self, c: torch.Tensor, d: torch.Tensor, mask: torch.Tensor):
        shape = c.shape
        def heads(value):
            return value.flatten(0, 1).reshape(-1, 16, 64).transpose(0, 1)[None]
        def merge(value):
            return value[0].transpose(0, 1).reshape(shape)
        normal = self.context_norm(c)
        q, k = heads(self.q(normal)), heads(self.k(normal))
        # Both use exactly the same alpha definition. SDPA avoids retaining alpha.
        c = c + self.context_o(merge(F.scaled_dot_product_attention(
            q, k, heads(self.context_v(normal)), attn_mask=mask, dropout_p=0.0)))
        c = c + self.context_ffn(self.ffn_norm(c))
        d = d + self.dynamic_o(merge(F.scaled_dot_product_attention(
            q, k, heads(self.dynamic_v(rms_zero(d))), attn_mask=mask, dropout_p=0.0)))
        d = d + self.dynamic_out(F.gelu(self.dynamic_in(rms_zero(d)))
                                * torch.sigmoid(self.dynamic_gate(c)))
        return c, d


class CausalInterpreter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(InterpretationLayer() for _ in range(4))

    def forward(self, h: torch.Tensor, frame_indices: torch.Tensor):
        if h.ndim != 3 or h.shape[1:] != (50, 1024) or len(h) < 2:
            raise ValueError("conditional interpreter requires the complete real N×50×1024 grid")
        indices = torch.as_tensor(frame_indices, device=h.device)
        if indices.shape != (len(h),) or not bool((indices[1:] > indices[:-1]).all()):
            raise ValueError("conditional interpreter requires increasing real frame indices")
        normal = rms_zero(h.float())
        time = position_code(indices, 512)[:, None].expand(-1, 50, -1)
        horizon = position_code(torch.arange(50, device=h.device), 512)[None].expand(len(h), -1, -1)
        c = normal + torch.cat((time, horizon), -1)
        d = torch.cat((torch.zeros_like(normal[:1]), normal[1:] - normal[:-1]))
        frames = torch.arange(len(h), device=h.device).repeat_interleave(50)
        # All fifty tokens within the arrival frame are mutually visible.
        mask = frames[:, None] >= frames[None, :]
        for layer in self.layers:
            if torch.is_grad_enabled():
                c, d = checkpoint(layer, c, d, mask, use_reentrant=False, preserve_rng_state=False)
            else:
                c, d = layer(c, d, mask)
        return c, d


def delta_memory(values: torch.Tensor, keys: torch.Tensor, out_width: int) -> torch.Tensor:
    """FP32 ordered writes: keys/values are transition×50×channel."""
    memory = torch.zeros((out_width, keys.shape[-1]), device=keys.device, dtype=torch.float32)
    for t in range(keys.shape[0]):
        k = keys[t].float().transpose(0, 1)
        value = values[t].float().transpose(0, 1)
        memory = memory + (value - memory @ k) @ k.transpose(0, 1) / 50.0
    return memory


class ConditionalTarget(nn.Module):
    """Predict response edits in actual X and compile B under the final A."""
    def __init__(self, in_width: int, out_width: int) -> None:
        super().__init__()
        self.a_x = nn.Linear(in_width, 256, bias=False)
        self.a_z = nn.Linear(128, 256, bias=False)
        self.a_context = nn.Linear(2048, 256, bias=False)
        self.a_dynamic = nn.Linear(1024, 256, bias=False)
        self.a_out = nn.Linear(256, 128, bias=False)
        self.b_key = nn.Linear(128, 256, bias=False)
        self.b_delta = nn.Linear(128, 256, bias=False)
        self.b_context = nn.Linear(2048, 256, bias=False)
        self.b_dynamic = nn.Linear(1024, 256, bias=False)
        self.b_out = nn.Linear(256, out_width, bias=False)
        nn.init.zeros_(self.a_out.weight)
        nn.init.zeros_(self.b_out.weight)

    def forward(self, a0: torch.Tensor, b0: torch.Tensor, x: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor, d: torch.Tensor, q=None):
        # Origin address X[t-1], arrival context/dynamic c[t],d[t], real last transition.
        origin = x[:-1].float()
        xi = F.normalize(origin, dim=-1, eps=1e-6)
        context = torch.cat((c[1:], h[1:].float()), -1)
        dynamic = d[1:]
        z0 = F.linear(xi, a0)
        ga = (1 + self.ua(q.float())[:, None, :]) if q is not None else 1
        gb = (1 + self.ub(q.float())[:, None, :]) if q is not None else 1
        va = self.a_out(F.gelu(self.a_x(xi) + self.a_z(z0) + self.a_context(context))
                        * self.a_dynamic(dynamic) * ga)
        s = delta_memory(va, xi, 128)
        a = a0 + s
        # These dependencies remain differentiable, including raw delta-z amplitude.
        z = F.linear(origin, a)
        key = F.normalize(z.float(), dim=-1, eps=1e-6)
        delta_z = F.linear(origin, s)
        vb = self.b_out(F.gelu(self.b_key(key) + self.b_delta(delta_z) + self.b_context(context))
                        * self.b_dynamic(dynamic) * gb)
        m = delta_memory(vb, key, self.b_out.out_features)
        return a, b0 + m, s, m
