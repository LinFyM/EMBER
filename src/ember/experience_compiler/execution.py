"""Ten-step native flow execution and same-call actual hidden evidence."""
from __future__ import annotations

import torch
from torch import nn

from ember.ecp.policy_effects import ExecutionPolicyPrefix, prepare_prefix_kv_cache


class NativeVelocity(nn.Module):
    def __init__(self, policy, batch, *, capture_phi=True, capture_indices=None):
        super().__init__()
        self.policy = policy
        if (policy.config.chunk_size, policy.config.max_action_dim) != (50, 32):
            raise ValueError('execution requires full native 50x32 flow latent')
        with torch.no_grad():
            images, masks = policy._preprocess_images(batch)
            embeddings, padding, _ = policy.model.embed_prefix(images, masks,
                batch['observation.language.tokens'], batch['observation.language.attention_mask'])
            selected = embeddings[:, :512]
            if capture_indices is not None:
                selected = selected[capture_indices]
            self.phi = selected.detach().cpu() if capture_phi else None
            self.cache = prepare_prefix_kv_cache(policy, ExecutionPolicyPrefix(embeddings, padding))
        self.register_buffer('padding', padding, persistent=False)

    def forward(self, z, tau):
        time = torch.as_tensor(tau, device=z.device, dtype=torch.float32).expand(len(z))
        return self.policy.model.denoise_step(self.padding, self.cache, z, time)


@torch.no_grad()
def action_chunk(velocity, noise, *, capture_hidden=True, capture_indices=None):
    """Ten native calls; only practice consumes same-call hidden evidence."""
    hidden = []
    calls = 0

    def capture(_module, args):
        nonlocal calls
        if calls in (0, 9):
            value = args[0] if capture_indices is None else args[0][capture_indices]
            hidden.append(value.detach())
        calls += 1

    handle = (velocity.policy.model.action_out_proj.register_forward_pre_hook(capture)
              if capture_hidden else None)
    z = noise
    try:
        for step in range(10):
            tau = 1 - step * .1
            v = velocity(z, tau)
            z = z - .1 * v
    finally:
        if handle is not None:
            handle.remove()
    evidence = None
    if capture_hidden:
        if calls != 10 or len(hidden) != 2 or hidden[0].shape[1:] != (50, 1024):
            raise ValueError('actual native action-hidden capture lost real flow times')
        # Defer the necessary practice transfer until the ten flow calls finish.
        evidence = torch.stack(hidden, 1).to(torch.bfloat16).cpu()
    return z[:, :, :7], evidence
