"""Official frozen PI0.5 image/language prefix and KV-cache preparation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ExecutionPolicyPrefix:
    """Official image/language prefix embeddings for real rollout observations."""

    embeddings: torch.Tensor
    padding: torch.Tensor


def _autocast(device: torch.device, *, native_precision: bool = False):
    if native_precision:
        return torch.autocast(device.type, enabled=False)
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def prepare_execution_policy_prefix(
    policy: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *, native_precision: bool = False,
) -> ExecutionPolicyPrefix:
    """Embed the exact prefix used by ``PI05Policy.predict_action_chunk``."""

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    images, image_masks = policy._preprocess_images(batch)
    with _autocast(tokens.device, native_precision=native_precision):
        embeddings, padding, _ = policy.model.embed_prefix(
            images, image_masks, tokens, masks
        )
    return ExecutionPolicyPrefix(
        embeddings=embeddings.detach(),
        padding=padding.detach(),
    )


def prepare_prefix_kv_cache(
    policy: torch.nn.Module,
    prefix: ExecutionPolicyPrefix,
    *, native_precision: bool = False,
) -> Any:
    """Cache the frozen official image/language prefix independently of Action Meta."""

    return prepare_prefix_features_and_cache(policy, prefix, native_precision=native_precision)[1]


def prepare_prefix_features_and_cache(
    policy: torch.nn.Module, prefix: ExecutionPolicyPrefix,
    *, native_precision: bool = False, track_grad: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Return same-forward Z/KV; teacher Meta replay may request their full graph."""

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    core = policy.model
    attention = torch.zeros_like(prefix.padding)
    mask = core._prepare_attention_masks_4d(
        make_att_2d_masks(prefix.padding, attention)
    )
    positions = torch.cumsum(prefix.padding, dim=1) - 1
    bridge = core.paligemma_with_expert
    bridge.paligemma.model.language_model.config._attn_implementation = "eager"
    with torch.set_grad_enabled(track_grad), _autocast(prefix.embeddings.device, native_precision=native_precision):
        outputs, cache = bridge.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix.embeddings, None],
            use_cache=True,
        )
    features = outputs[0]
    if features.shape[:2] != prefix.padding.shape:
        raise ValueError("native prefix evidence and KV positions differ")
    return features if track_grad else features.detach(), cache
