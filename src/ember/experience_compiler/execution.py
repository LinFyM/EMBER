"""Actual native ODE/SDE execution, with same-call action hidden evidence.

SDE math is the frozen denoising_return_writer_20261006 NativeVelocity/sample_sde
kernel. Native prefix preparation belongs to ecp.policy_effects; no retired
training/evaluation entrypoint is restored.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

from ember.ecp.policy_effects import ExecutionPolicyPrefix, prepare_prefix_kv_cache


class NativeVelocity(nn.Module):
    def __init__(self, policy, batch):
        super().__init__()
        self.policy = policy
        if (policy.config.chunk_size, policy.config.max_action_dim) != (50, 32):
            raise ValueError('execution requires full native 50x32 flow latent')
        with torch.no_grad():
            images, masks = policy._preprocess_images(batch)
            embeddings, padding, _ = policy.model.embed_prefix(images, masks,
                batch['observation.language.tokens'], batch['observation.language.attention_mask'])
            self.phi = embeddings[:, :512].detach().cpu()
            self.cache = prepare_prefix_kv_cache(policy, ExecutionPolicyPrefix(embeddings, padding))
        self.register_buffer('padding', padding, persistent=False)

    def forward(self, z, tau):
        time = torch.as_tensor(tau, device=z.device, dtype=torch.float32).expand(len(z))
        return self.policy.model.denoise_step(self.padding, self.cache, z, time)


@dataclass
class Transition:
    step: int
    tau: float
    z: torch.Tensor
    m: torch.Tensor
    z_next: torch.Tensor

    def cpu(self):
        return Transition(self.step, self.tau, self.z.detach().cpu(),
                          self.m.detach().cpu(), self.z_next.detach().cpu())


@torch.no_grad()
def action_chunk(velocity, noise, *, sde_seed=None):
    """Ten actual native calls; retain H at the first and last real times."""
    hidden, transitions = [], []
    calls = 0

    def capture(_module, args):
        nonlocal calls
        if calls in (0, 9):
            hidden.append(args[0].detach().cpu().to(torch.bfloat16))
        calls += 1

    handle = velocity.policy.model.action_out_proj.register_forward_pre_hook(capture)
    generator = None if sde_seed is None else torch.Generator().manual_seed(int(sde_seed))
    z = noise
    try:
        for step in range(10):
            tau = 1 - step * .1
            v = velocity(z, tau)
            if generator is None:
                z = z - .1 * v
            else:
                m = z - .1 * ((1 + .5 * (1 - tau)) * v + .5 * z)
                epsilon = torch.randn(z.shape, generator=generator, device='cpu').to(z.device)
                nxt = m + (.1 * tau)**.5 * epsilon
                transitions.append(Transition(step, tau, z, m, nxt))
                z = nxt
    finally:
        handle.remove()
    if calls != 10 or len(hidden) != 2 or hidden[0].shape[1:] != (50, 1024):
        raise ValueError('actual native action-hidden capture lost real flow times')
    return z[:, :, :7], torch.stack(hidden, 1), transitions


def score_cotangent(transition, advantage, *, replans, retained):
    """4-condition x 2-query average and unbiased decision/transition sampling."""
    if replans < retained or not 0 < retained <= 16 or not .09 < transition.tau <= 1:
        raise ValueError('invalid actual query reservoir or SDE time')
    weight = .25 * .5 * replans / retained * 10 / 2
    return float(advantage) * weight * (1 + .5 * (1 - transition.tau)) / transition.tau * (
        transition.z_next - transition.m)
