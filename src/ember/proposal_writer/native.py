"""Read-only-source dual meta contexts, contextual teaching and native execution."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ember.operator_writer.native import _teacher_attention_kernel


class MetaReader(nn.Module):
    """Own rank32 factors outside the physical policy/task-export namespace.

    The physical task adapter remains A/nonzero,B/zero. Hooks add only reading
    deltas within an explicit context; execution hooks must be inactive there.
    Whole-frame checkpoint replay re-enters this context with explicit factors.
    """
    def __init__(self, policy, execution, *, frame_chunk=8):
        super().__init__()
        object.__setattr__(self, 'policy', policy)
        object.__setattr__(self, 'execution', execution)
        self.frame_chunk = frame_chunk
        self.active = ContextVar(f'ember_proposal_meta_{id(self)}', default=None)
        self.lock = RLock()
        self.values = nn.ParameterList()
        self.names, self.groups, self.handles = [], [], []
        for group, stem, init_seed, width in (
            ('gemma', 'paligemma.model.language_model', 202610101, 2048),
            ('action', 'gemma_expert.model', 202610102, 1024)):
            generator = torch.Generator().manual_seed(init_seed)
            for layer in range(18):
                for projection, output_width in [('q_proj', 2048), ('v_proj', 256)]:
                    name = f'model.paligemma_with_expert.{stem}.layers.{layer}.self_attn.{projection}'
                    module = policy.get_submodule(name)
                    if tuple(module.weight.shape) != (output_width, width):
                        raise ValueError(f'installed dual-meta projection shape changed: {name}')
                    a = torch.empty((32, width), dtype=torch.float32)
                    nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
                    index = len(self.values)
                    self.values.extend([nn.Parameter(a), nn.Parameter(torch.zeros(output_width, 32))])
                    self.names.extend([name + '.meta_A', name + '.meta_B'])
                    self.groups.extend([group, group])
                    self.handles.append(module.register_forward_hook(self._hook(index)))
        self.register_buffer('probe', torch.randn(50, 32, generator=torch.Generator().manual_seed(1729)))

    def _hook(self, index):
        def add(_module, inputs, output):
            weights = self.active.get()
            if weights is None:
                return output
            a, b = weights[index:index + 2]
            delta = torch.nn.functional.linear(torch.nn.functional.linear(inputs[0].to(a.dtype), a), b)
            return (output + delta).to(output.dtype)
        return add

    @contextmanager
    def activate(self, weights):
        with self.lock:
            if self.active.get() is not None or self.execution._active_state is not None:
                raise ValueError('meta read and native task execution contexts overlap')
            token = self.active.set(weights)
            try:
                yield
            finally:
                self.active.reset(token)

    @torch.no_grad()
    def encode_raw(self, condition, *, chunk=16):
        """Cacheable frozen vision/token input only; no contextual Gemma output."""
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
        frames, _, tokens, _ = condition
        if frames.ndim != 5 or frames.shape[1:3] != (2, 3) or frames.dtype != torch.uint8:
            raise ValueError('dual-meta requires real synchronized action-hidden RGB')
        bridge = self.policy.model.paligemma_with_expert
        raw = []
        for start in range(0,len(frames),chunk):
            selected=frames[start:start+chunk]
            pixels = selected.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
            images = (resize_with_pad_torch(pixels, 224, 224) * 2 - 1).permute(0, 3, 1, 2)
            raw.append(bridge.embed_image(images).reshape(len(selected), 512, 2048))
        return torch.cat(raw), bridge.embed_language_tokens(tokens)

    def _frame(self, raw, language, token_mask, *weights):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        core, count = self.policy.model, len(raw)
        bridge = core.paligemma_with_expert
        language = language.expand(count,-1,-1)
        prefix = torch.cat((raw, language), 1)
        pad = torch.cat((torch.ones(count, 512, dtype=torch.bool, device=raw.device),
                         token_mask.expand(count, -1)), 1)
        with self.activate(weights):
            # PI05 embeds noise and time only; no teacher state, including no zero state.
            suffix, suffix_pad, suffix_attention, time_features = core.embed_suffix(
                self.probe[None].expand(count, -1, -1), torch.ones(count, device=raw.device))
            full_pad = torch.cat((pad, suffix_pad), 1)
            attention = torch.cat((torch.zeros_like(pad), suffix_attention), 1)
            mask = core._prepare_attention_masks_4d(make_att_2d_masks(full_pad, attention))
            positions = torch.cumsum(full_pad, 1) - 1
            dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            with _teacher_attention_kernel(mask):
                (z, h), _ = bridge(attention_mask=mask, position_ids=positions, past_key_values=None,
                    inputs_embeds=[prefix.to(dtype), suffix.to(dtype)], use_cache=False,
                    adarms_cond=[None, time_features])
        if z.shape[1:] != (512 + language.shape[1], 2048) or h.shape[1:] != (50, 1024):
            raise ValueError('contextual post-norm dual stream output changed')
        return z[:, :512], h, language[:1]

    def forward(self, condition, *, checkpoint_frames=True, raw_prefix=None):
        frames, indices, tokens, mask = condition
        if len(frames) != len(indices) or len(frames) < 2:
            raise ValueError('ordered teaching frame/index correspondence changed')
        outputs = []
        raw, language = self.encode_raw(condition) if raw_prefix is None else raw_prefix
        if raw.shape != (len(frames),512,2048) or raw.requires_grad or language.requires_grad:
            raise ValueError('raw cache must contain only actual frozen source embeddings')
        for start in range(0, len(frames), self.frame_chunk):
            args = (raw[start:start + self.frame_chunk], language, mask, *self.values)
            outputs.append(checkpoint(self._frame, *args, use_reentrant=False, preserve_rng_state=False)
                           if checkpoint_frames and torch.is_grad_enabled() else self._frame(*args))
        return dict(z=torch.cat([v[0] for v in outputs]), h=torch.cat([v[1] for v in outputs]),
                    language=outputs[0][2][0], language_mask=mask[0], indices=indices)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
