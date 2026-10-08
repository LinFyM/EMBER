"""Differentiable state-free PI05 teacher read under the same public LoRA."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Iterator

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX


_TEACHER_ATTENTION = ContextVar("ember_teacher_attention", default=False)
_TEACHER_PADDING = ContextVar("ember_teacher_padding_queries", default=None)
_TEACHER_EAGER = ContextVar("ember_teacher_original_attention", default=None)
_TEACHER_MLP = ContextVar("ember_teacher_original_mlp", default=None)
_TEACHER_OBSERVER = ContextVar("ember_teacher_attention_observer", default=None)
_ATTENTION_SCOPE_LOCK = RLock()


def _padding_query_rows(mask):
    if mask is None:
        return None
    # All frames of this teacher have the same token mask. Finite all-masked
    # rows need eager backward: fused SDPA's saved logsumexp loses log(T) there.
    padded = (mask[0, 0] < 0).all(-1)
    return (~padded).nonzero().flatten(), padded.nonzero().flatten()


def _teacher_sdpa(module, query, key, value, attention_mask, scaling, dropout=0.0,
                  *, _eager=None, _query_rows=None, **kwargs):
    """SDPA for visible queries; installed eager preserves finite padding-row VJP."""
    from transformers.models.gemma import modeling_gemma

    repeat = modeling_gemma.repeat_kv
    keys, values = repeat(key, module.num_key_value_groups), repeat(value, module.num_key_value_groups)
    rows = _query_rows if _query_rows is not None else _padding_query_rows(attention_mask)

    def sdpa(q, mask):
        return torch.nn.functional.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask,
            dropout_p=dropout if module.training else 0.0, is_causal=False, scale=scaling)

    if rows is None or rows[1].numel() == 0:
        output = sdpa(query, attention_mask)
    else:
        visible, padded = rows
        output = query.new_zeros((*query.shape[:-1], value.shape[-1]))
        if visible.numel():
            output = output.index_copy(2, visible, sdpa(
                query.index_select(2, visible), attention_mask.index_select(-2, visible)))
        eager = _eager or modeling_gemma.eager_attention_forward
        padded_output, _ = eager(module, query.index_select(2, padded), key, value,
                                 attention_mask.index_select(-2, padded), scaling, dropout=dropout, **kwargs)
        output = output.index_copy(2, padded, padded_output.transpose(1, 2))
    return output.transpose(1, 2).contiguous(), None


@contextmanager
def _teacher_attention_kernel(attention_mask=None, observer=None):
    """Teacher-only SDPA and frozen-MLP replay inside the full frame call."""
    from transformers.models.gemma import modeling_gemma

    # Native callers cannot race the binding; unrelated threads retain eager.
    with _ATTENTION_SCOPE_LOCK:
        original = modeling_gemma.eager_attention_forward
        eager = _TEACHER_EAGER.get() or original
        original_mlp = modeling_gemma.GemmaMLP.forward
        mlp = _TEACHER_MLP.get() or original_mlp
        rows = _padding_query_rows(attention_mask)
        variables = (_TEACHER_ATTENTION, _TEACHER_PADDING, _TEACHER_EAGER, _TEACHER_MLP, _TEACHER_OBSERVER)
        tokens = tuple(variable.set(value) for variable, value in
                       zip(variables, (True, rows, eager, mlp, observer), strict=True))

        def dispatch(*args, **kwargs):
            if not _TEACHER_ATTENTION.get():
                return eager(*args, **kwargs)
            observed = _TEACHER_OBSERVER.get()
            if observed is not None:
                observed(*args, **kwargs)
            return _teacher_sdpa(*args, _eager=eager, _query_rows=_TEACHER_PADDING.get(), **kwargs)

        def mlp_dispatch(module, hidden):
            if not _TEACHER_ATTENTION.get() or not torch.is_grad_enabled() or not hidden.requires_grad:
                return mlp(module, hidden)
            # Only this source-frozen, non-LoRA submodule can safely replay after
            # functional_call has restored physical β. Its input VJP stays live.
            if any(value.requires_grad or ".lora_" in name
                   for name, value in module.named_parameters()):
                raise ValueError("teacher MLP replay requires frozen non-LoRA parameters")
            return checkpoint(lambda x: mlp(module, x), hidden,
                              use_reentrant=False, preserve_rng_state=False)

        modeling_gemma.eager_attention_forward = dispatch
        modeling_gemma.GemmaMLP.forward = mlp_dispatch
        try:
            yield
        finally:
            modeling_gemma.eager_attention_forward = original
            modeling_gemma.GemmaMLP.forward = original_mlp
            for variable, token in zip(variables, tokens, strict=True):
                variable.reset(token)


@contextmanager
def _capture_inputs(policy: nn.Module, names: tuple[str, ...]) -> Iterator[dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    try:
        for name in names:
            module = policy.get_submodule(name)

            def hook(_module, arguments, *, selected=name):
                if selected in captured or not arguments or not isinstance(arguments[0], torch.Tensor):
                    raise ValueError("native target was absent, repeated or changed type")
                captured[selected] = arguments[0]

            handles.append(module.register_forward_pre_hook(hook))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class _NativeFrameCall(nn.Module):
    """Wrap the policy so torch.func substitutes β in suffix and all 38 hooks."""

    def __init__(self, policy: nn.Module, names: tuple[str, ...], probe: torch.Tensor,
                 prefix_change: bool = False, include_image_prefix: bool = False) -> None:
        super().__init__()
        self.policy = policy
        self.names = names
        self.probe = probe
        self.prefix_change = prefix_change
        self.include_image_prefix = include_image_prefix

    def forward(self, frames: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, resize_with_pad_torch

        if frames.ndim != 5 or frames.shape[1:3] != (2, 3) or frames.dtype != torch.uint8:
            raise ValueError("teacher needs synchronized actual dual RGB")
        core = self.policy.model
        bridge = core.paligemma_with_expert
        observer = None
        if self.prefix_change:
            from .prefix_change import NativeAttentionCapture
            observer = NativeAttentionCapture(bridge)
        count = len(frames)
        pixels = frames.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
        images = (resize_with_pad_torch(pixels, 224, 224) * 2 - 1).permute(0, 3, 1, 2)
        with torch.no_grad():
            image_tokens = bridge.embed_image(images)
            language_tokens = bridge.embed_language_tokens(tokens.expand(count, -1))
        if image_tokens.shape[1:] != (256, 2048):
            raise ValueError("native dual-camera patches changed")
        prefix = torch.cat((image_tokens.reshape(count, 512, 2048), language_tokens), dim=1)
        padding = torch.cat((torch.ones((count, 512), dtype=torch.bool, device=frames.device),
                             token_mask.expand(count, -1)), dim=1)
        attention = torch.zeros_like(padding)
        noise = self.probe.to(frames.device)[None].expand(count, -1, -1)
        time = torch.ones(count, dtype=torch.float32, device=frames.device)
        with _capture_inputs(self.policy, self.names) as captured:
            suffix, suffix_pad, suffix_attention, adarms = core.embed_suffix(noise, time)
            full_padding = torch.cat((padding, suffix_pad), dim=1)
            full_attention = torch.cat((attention, suffix_attention), dim=1)
            mask = core._prepare_attention_masks_4d(make_att_2d_masks(full_padding, full_attention))
            positions = torch.cumsum(full_padding, dim=1) - 1
            dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            handle = observer.projection.register_forward_hook(observer.projection_call) if observer else None
            try:
                with _teacher_attention_kernel(mask, observer.attention_call if observer else None):
                    (_, hidden), _ = bridge.forward(
                        attention_mask=mask, position_ids=positions, past_key_values=None,
                        inputs_embeds=[prefix.to(dtype), suffix.to(dtype)], use_cache=False,
                        adarms_cond=[None, adarms],
                    )
            finally:
                if handle is not None:
                    handle.remove()
            # The actual action_out projection is part of the 38-target native read.
            core.action_out_proj(hidden.float())
        if set(captured) != set(self.names) or hidden.shape != (count, 50, 1024):
            raise ValueError("native full suffix/target capture is incomplete")
        return (hidden, *(captured[name] for name in self.names),
                *(observer.outputs() if observer else ()),
                *((image_tokens.reshape(count, 512, 2048),) if self.include_image_prefix else ()))


@torch.no_grad()
def read_frozen_teacher_features(policy, mt_state, probe, condition, *, frame_chunk=8):
    """Return actual legal RGB prefix and MT-probe H, without retaining unused X."""
    frames, indices, tokens, token_mask = condition
    if frame_chunk < 1 or len(frames) != len(indices):
        raise ValueError("frozen teacher frame chunk or position identity changed")
    wrapper = _NativeFrameCall(policy, (), probe, include_image_prefix=True)
    state = {"policy." + name: value for name, value in mt_state.items()}
    outputs = [tuple(value.cpu() for value in torch.func.functional_call(wrapper, state,
               (frames[start:start + frame_chunk], tokens, token_mask), strict=False))
               for start in range(0, len(frames), frame_chunk)]
    return torch.cat([row[1] for row in outputs]), torch.cat([row[0] for row in outputs])


def read_native_video(policy: nn.Module, common: dict[str, torch.Tensor], probe: torch.Tensor,
                      condition: tuple, names: tuple[str, ...], *, frame_chunk: int = 8,
                      checkpoint_frames: bool = True, prefix_change: bool = False):
    """No cached/detached β features: every compile recomputes the legal video."""
    frames, indices, tokens, token_mask = condition
    if (len(frames) != len(indices) or len(frames) < 2 or int(indices[-1]) < int(indices[0])
            or tokens.shape[0] != 1 or token_mask.shape != tokens.shape[:2]
            or frame_chunk < 1 or set(common) != {name + suffix for name in names
                                                   for suffix in (LORA_A_SUFFIX, ".lora_B.default.weight")}):
        raise ValueError("native legal-video/complete-public-state contract changed")
    wrapper = _NativeFrameCall(policy, names, probe, prefix_change)
    keys, values = tuple(common), tuple(common.values())
    outputs = []
    for start in range(0, len(frames), frame_chunk):
        subset = frames[start:start + frame_chunk]

        def call(*weights, batch=subset):
            state = {"policy." + name: value for name, value in zip(keys, weights, strict=True)}
            return torch.func.functional_call(wrapper, state, (batch, tokens, token_mask), strict=False)

        if checkpoint_frames and torch.is_grad_enabled():
            output = checkpoint(call, *values, use_reentrant=False, preserve_rng_state=False)
        else:
            output = call(*values)
        outputs.append(output)
    h = torch.cat([item[0] for item in outputs], dim=0)
    x = {name: torch.cat([item[index + 1] for item in outputs], dim=0)
         for index, name in enumerate(names)}
    if prefix_change:
        from .prefix_change import response_difference
        bridge = policy.model.paligemma_with_expert
        difference = response_difference([item[len(names) + 1:] for item in outputs],
            bridge.paligemma.model.language_model.layers[17].self_attn,
            bridge.gemma_expert.model.layers[17].self_attn.o_proj, frame_chunk=frame_chunk)
        return x, h, difference
    return x, h
