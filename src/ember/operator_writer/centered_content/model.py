"""Frozen learned locator and fresh content × dynamic outer-product Q factors."""
from contextlib import contextmanager
import re
import torch
from torch import nn
from safetensors import safe_open
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.writer.runtime import autocast
from ..conditional_read_write import rms_zero
from .common import LOCATOR

GROUPS = ('a_x', 'a_z', 'a_context', 'a_dynamic', 'a_out',
          'b_key', 'b_delta', 'b_context', 'b_dynamic', 'b_out')


class FrozenLocator(nn.Module):
    def __init__(self, device):
        super().__init__()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            self.Pq = nn.Linear(1024, 256, bias=False)
            self.Pk = nn.Linear(256, 256, bias=False)
        # Exactly two tensors; no old R control or optimizer state is loaded.
        with safe_open(str(LOCATOR), framework='pt', device='cpu') as f, torch.no_grad():
            for name in ('Pq', 'Pk'):
                getattr(self, name).weight.copy_(f.get_tensor(name + '.weight'))
        self.to(device).requires_grad_(False)
        if sum(p.numel() for p in self.parameters()) != 327680:
            raise ValueError('frozen locator parameter count changed')

    def forward(self, c, k):
        q = self.Pq(rms_zero(c[1:].float()))
        keys = self.Pk(rms_zero(k[:-1, 0].float()))
        return (torch.einsum('tjd,tpd->tjp', q.float(), keys.float()) / 16).softmax(-1)


class ContentBinding(nn.Module):
    def __init__(self, device):
        super().__init__()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
            torch.manual_seed(7)
            self.P = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in range(18))
            self.D = nn.ModuleList(nn.Linear(1024, 1024, bias=False) for _ in range(18))
            with torch.no_grad():
                for p, d in zip(self.P, self.D, strict=True):
                    p.weight.copy_(torch.eye(256)); d.weight.zero_()
        self.to(device)
        if sum(p.numel() for p in self.parameters()) != 20054016:
            raise ValueError('registered fresh P/D parameter count changed')

    def layer(self, fixed, layer):
        selected = fixed['p'][layer]
        u = self.P[layer](selected)
        r = self.D[layer](rms_zero(fixed['d'][1:].float())).reshape(-1, 50, 8, 128)
        return (torch.einsum('tjd,tjhr->hdr', u.float(), r.float()) /
                (selected.shape[0] * 50)).reshape(2048, 128)


def freeze_heads(writer):
    writer.requires_grad_(False)
    for unit in writer.conditional_targets:
        for group in GROUPS:
            getattr(unit, group).weight.requires_grad_(True)
    return tuple(p for p in writer.parameters() if p.requires_grad)


def q_layer(name):
    match = re.search(r'gemma_expert\.model\.layers\.(\d+)\.self_attn\.q_proj$', name)
    return int(match[1]) if match else None


def target(runtime, index, fixed, binding):
    writer = runtime.writer; name = writer.names[index]; common = writer.public_state()
    a, b, _, _ = writer.conditional_targets[index](common[name + LORA_A_SUFFIX], common[name + LORA_B_SUFFIX],
                                               fixed['X'][name], fixed['H'], fixed['c'], fixed['d'])
    layer = q_layer(name)
    if layer is not None:
        b = b + binding.layer(fixed, layer)
    return a, b


@torch.no_grad()
def compile_fixed(runtime, fixed, binding, *, components=False):
    state, residual = {}, {}
    with autocast(runtime.device):
        for i, name in enumerate(runtime.writer.names):
            a, b = target(runtime, i, fixed, binding)
            state[name + LORA_A_SUFFIX], state[name + LORA_B_SUFFIX] = a, b
            layer = q_layer(name)
            if components and layer is not None:
                residual[str(layer)] = binding.layer(fixed, layer).float().cpu()
    validate_lora_state(state, runtime.lora)
    return state, residual
