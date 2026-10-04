"""Bounded §132 role-content × causal-dynamic Q compilation; retired after readback."""
from __future__ import annotations

from contextlib import contextmanager
import re
import torch
from torch import nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.writer.runtime import autocast
from .conditional_read_write import rms_zero

GROUPS = ('a_x', 'a_z', 'a_context', 'a_dynamic', 'a_out',
          'b_key', 'b_delta', 'b_context', 'b_dynamic', 'b_out')


class RoleBinding(nn.Module):
    """A single complete rank128 LoRA; no execution-time auxiliary path."""

    def __init__(self, device):
        super().__init__()
        with torch.random.fork_rng(devices=([device.index or 0] if device.type == 'cuda' else [])):
            torch.manual_seed(7)
            self.Pq = nn.Linear(1024, 256, bias=False)
            self.Pk = nn.Linear(256, 256, bias=False)
            self.P = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in range(18))
            self.D = nn.ModuleList(nn.Linear(1024, 8 * 128, bias=False) for _ in range(18))
            with torch.no_grad():
                for p, d in zip(self.P, self.D, strict=True):
                    p.weight.copy_(torch.eye(256))
                    d.weight.zero_()
        self.to(device)
        if sum(p.numel() for p in self.parameters()) != 20381696:
            raise ValueError('registered outer-product parameterization changed')

    def alpha(self, fixed):
        q = self.Pq(rms_zero(fixed['c'][1:].float()))
        k = self.Pk(rms_zero(fixed['K'][:-1, 0].float()))
        return (torch.einsum('tjd,tpd->tjp', q.float(), k.float()) / 16).softmax(-1)

    def layer(self, fixed, alpha, layer):
        selected = torch.einsum('tjp,tpd->tjd', alpha.float(),
                                rms_zero(fixed['K'][:-1, layer].float()))
        u = self.P[layer](selected)
        r = self.D[layer](rms_zero(fixed['d'][1:].float())).reshape(-1, 50, 8, 128)
        # Each head owns a contiguous 256-row block of the native q_proj.
        return (torch.einsum('tjd,tjhr->hdr', u.float(), r.float()) /
                (selected.shape[0] * 50)).reshape(2048, 128)

    def teacher_loss(self, alpha, q):
        a = alpha.float().mean(1).clamp_min(1e-30)
        q = q.float()
        numerator = (q * (q.clamp_min(1e-30).log() - a.log())).sum(-1)
        chance = (q * (q.clamp_min(1e-30).log() + torch.log(q.new_tensor(512.)))).sum(-1)
        valid = (q.sum(-1) > 0) & (chance > 1e-12)
        return torch.where(valid, numerator / chance.clamp_min(1e-12), 0).mean()


def freeze_heads(writer):
    writer.requires_grad_(False)
    for unit in writer.conditional_targets:
        for group in GROUPS:
            getattr(unit, group).weight.requires_grad_(True)
    return tuple(p for p in writer.parameters() if p.requires_grad)


def q_layer(name):
    match = re.search(r'gemma_expert\.model\.layers\.(\d+)\.self_attn\.q_proj$', name)
    return int(match[1]) if match else None


def target(runtime, index, fixed, binding=None, alpha=None):
    writer = runtime.writer
    name = writer.names[index]
    common = writer.public_state()
    a, b, _, _ = writer.conditional_targets[index](
        common[name + LORA_A_SUFFIX], common[name + LORA_B_SUFFIX],
        fixed['X'][name], fixed['H'], fixed['c'], fixed['d'])
    layer = q_layer(name)
    if binding is not None and layer is not None:
        b = b + binding.layer(fixed, alpha, layer)
    return a, b


@torch.no_grad()
def compile_fixed(runtime, fixed, binding=None):
    state, components = {}, {}
    with autocast(runtime.device):
        alpha = binding.alpha(fixed) if binding is not None else None
        for i, name in enumerate(runtime.writer.names):
            a, b = target(runtime, i, fixed, binding, alpha)
            state[name + LORA_A_SUFFIX], state[name + LORA_B_SUFFIX] = a, b
            layer = q_layer(name)
            if binding is not None and layer is not None:
                components[str(layer)] = binding.layer(fixed, alpha, layer).cpu()
    validate_lora_state(state, runtime.lora)
    return state, (None if alpha is None else dict(alpha_frame_mean=alpha.mean(1).cpu(), R=components))


@contextmanager
def native_keys(policy):
    """Capture the real teacher prefix in the same canonical native forward."""
    layers = policy.model.paligemma_with_expert.paligemma.model.language_model.layers
    if len(layers) != 18:
        raise ValueError('teacher prefix depth changed')
    chunks, handles = [[] for _ in layers], []
    for i, layer in enumerate(layers):
        def capture(module, args, output, i=i):
            if output.shape[-1] != 256 or output.shape[1] < 512:
                raise ValueError('teacher actual single-KV-head image prefix changed')
            chunks[i].append(output[:, :512].detach().cpu())
        handles.append(layer.self_attn.k_proj.register_forward_hook(capture))
    try:
        yield chunks
    finally:
        for handle in handles:
            handle.remove()


class RoleObserver:
    """Live actual rotated Q/image-K density, leaving the attention output native."""

    def __init__(self, policy, patch_weights, *, all_slots=False):
        layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
        if len(layers) != 18 or any(l.self_attn.num_key_value_groups != 8 for l in layers):
            raise ValueError('actual eighteen-layer/eight-head single-KV GQA changed')
        self.layers = {id(layer.self_attn): i for i, layer in enumerate(layers)}
        self.q = patch_weights.float()
        self.slots = 50 if all_slots else 5
        self.scores, self.mass = {}, {}

    def attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        layer = self.layers.get(id(module))
        if layer is not None:
            if query.shape[-2:] != (50, 256) or key.shape[1] != 1:
                raise ValueError('actual own Q/key topology changed')
            # Official own prefix includes two visible 256-token cameras, then
            # the masked third camera. Labels cover the first two only.
            logits = (query[:, :, :self.slots].float() @ key.float().transpose(-1, -2)) * scaling
            if mask is not None:
                logits = logits + mask[:, :, :self.slots]
            image = logits[..., :512]
            logq = self.q.clamp_min(1e-30).log()
            density = torch.logsumexp(image[..., None, :] + logq[:, None, None], -1)
            self.scores[layer] = density
            self.mass[layer] = (torch.logsumexp(image, -1) - torch.logsumexp(logits, -1)).exp()
        return self.base(module, query, key, value, mask, scaling, dropout=dropout, **kwargs)

    def loss(self):
        if set(self.scores) != set(range(18)):
            raise ValueError('real own role forward did not visit all eighteen layers')
        scores = torch.stack([self.scores[i] for i in range(18)], 1).mean((1, 2, 3))
        visible = self.q.sum(-1) > 0
        count = visible.sum(-1)
        scores = scores.masked_fill(~visible, -torch.inf)
        valid = visible[:, 0] & (count > 1)
        safe = torch.where(valid[:, None], scores, torch.zeros_like(scores))
        loss = -safe.log_softmax(-1)[:, 0] / count.clamp_min(2).float().log()
        return torch.where(valid, loss, 0).mean()

    @contextmanager
    def capture(self):
        from transformers.models.gemma import modeling_gemma
        self.base = modeling_gemma.eager_attention_forward
        modeling_gemma.eager_attention_forward = self.attention
        try:
            yield self
        finally:
            modeling_gemma.eager_attention_forward = self.base
