"""Actual own-Q training loss and passive native-precision first-plan measures."""
from contextlib import contextmanager
import torch

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


class TenFlowObserver(RoleObserver):
    """FP32 measurement only; original mixed-precision attention returned unchanged."""
    def __init__(self, policy, q, f=None):
        super().__init__(policy, q)
        self.policy, self.f, self.steps, self.entity_mass = policy, f, [], {}

    def attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        layer = self.layers.get(id(module))
        if layer is not None:
            if query.shape[1:] != (8, 50, 256) or key.shape[1] != 1 or scaling != 1 / 16:
                raise ValueError('actual18/eight-head/single-KV scaling changed')
            with torch.autocast('cuda', enabled=False):
                logits = (query[:, :, :5].float() @ key.float().transpose(-1, -2)) * scaling
                if mask is not None:
                    logits = logits + mask[:, :, :5].float()
                image = logits[..., :512]
                self.scores[layer] = torch.logsumexp(image[..., None, :] + self.q.clamp_min(1e-30).log()[:, None, None], -1).cpu()
                pi = logits.softmax(-1)[..., :512]
                self.mass[layer] = pi.sum(-1).cpu()
                if self.f is not None:
                    self.entity_mass[layer] = torch.einsum('bhip,bep->bhie', pi, self.f.float()).cpu()
        return self.base(module, query, key, value, mask, scaling, dropout=dropout, **kwargs)

    @contextmanager
    def capture(self):
        original = self.policy.model.denoise_step
        def denoise(*args, **kwargs):
            self.scores.clear(); self.mass.clear(); self.entity_mass.clear()
            value = original(*args, **kwargs)
            if set(self.scores) != set(range(18)):
                raise ValueError('real ten-step consumer missed a layer')
            row = dict(role_scores=torch.stack([self.scores[l] for l in range(18)], 1),
                       image_mass=torch.stack([self.mass[l] for l in range(18)], 1))
            if self.f is not None:
                row['entity_mass'] = torch.stack([self.entity_mass[l] for l in range(18)], 1)
            self.steps.append(row)
            return value
        self.policy.model.denoise_step = denoise
        try:
            with super().capture():
                yield self
        finally:
            self.policy.model.denoise_step = original

    def arrays(self):
        if len(self.steps) != 10:
            raise ValueError('exact official10-flow measure incomplete')
        return {k: torch.stack([s[k] for s in self.steps], 1) for k in self.steps[0]}
