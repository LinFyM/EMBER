"""One-pass frozen PI0.5 full-response capture.

The capture keeps teacher-frame time, Action Expert depth/horizon, and the
antithetic probe axis separate.  It never receives robot state or actions.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    NativeFactorError,
)
from ember.ecp.observer import ActionLayerStateCapture
from ember.ecp.policy_effects import ExecutionPolicyPrefix


@dataclass(frozen=True)
class FrozenPolicyResponseChunk:
    """One contiguous frozen frame chunk with every scientific axis intact."""

    start_frame: int
    patch_states: torch.Tensor
    language_states: torch.Tensor
    language_mask: torch.Tensor
    layer_states: torch.Tensor
    flow_velocity: torch.Tensor
    suffix_noise: torch.Tensor

    @property
    def frame_count(self) -> int:
        return int(self.patch_states.shape[0])

@dataclass(frozen=True)
class FrozenPolicyResponseVideo:
    """Cacheable deployment-visible evidence for one independently ordered video."""

    patch_states: torch.Tensor
    language_states: torch.Tensor
    language_mask: torch.Tensor
    layer_states: torch.Tensor
    flow_velocity: torch.Tensor
    suffix_noise: torch.Tensor
    frame_positions: torch.Tensor

    @property
    def frame_count(self) -> int:
        return int(self.patch_states.shape[0])

    @property
    def tensor_bytes(self) -> int:
        """Resident bytes for the frozen evidence, without allocator overhead."""

        tensors = (
            self.patch_states,
            self.language_states,
            self.language_mask,
            self.layer_states,
            self.flow_velocity,
            self.suffix_noise,
            self.frame_positions,
        )
        return sum(value.numel() * value.element_size() for value in tensors)

    def frame_slice(self, stop: int) -> "FrozenPolicyResponseVideo":
        """Return a prefix-only causal view; no future tensor remains reachable."""

        if not 0 < stop <= self.frame_count:
            raise ValueError("policy-response causal prefix is outside the video")
        return replace(
            self,
            patch_states=self.patch_states[:stop],
            language_states=self.language_states[:stop],
            language_mask=self.language_mask[:stop],
            layer_states=self.layer_states[:stop],
            flow_velocity=self.flow_velocity[:stop],
            frame_positions=self.frame_positions[:stop],
        )

    def to(self, *args: object, **kwargs: object) -> "FrozenPolicyResponseVideo":
        tensor = lambda value: value.to(*args, **kwargs)
        patch_states = tensor(self.patch_states)
        return FrozenPolicyResponseVideo(
            patch_states=patch_states,
            language_states=tensor(self.language_states),
            language_mask=self.language_mask.to(device=patch_states.device),
            layer_states=tensor(self.layer_states),
            flow_velocity=tensor(self.flow_velocity),
            suffix_noise=tensor(self.suffix_noise),
            frame_positions=tensor(self.frame_positions),
        )


def _capture_autocast(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.no_grad()
def capture_policy_response_chunk(
    *,
    policy: torch.nn.Module,
    owners: Sequence[TargetOwner],
    prefix: ExecutionPolicyPrefix,
    fixed_probe: torch.Tensor,
    start_frame: int,
    image_tokens: int = 256,
) -> FrozenPolicyResponseChunk:
    """Capture prefix states, every Action boundary, and velocity in one forward."""

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    frames = int(prefix.embeddings.shape[0])
    if (
        frames <= 0
        or start_frame < 0
        or prefix.padding.shape != prefix.embeddings.shape[:2]
        or fixed_probe.shape != (ACTION_HORIZON, 32)
        or image_tokens <= 0
        or prefix.embeddings.shape[1] <= image_tokens
        or any(parameter.requires_grad for parameter in policy.parameters())
    ):
        raise NativeFactorError("policy-response capture crossed the frozen input wall")

    core = policy.model
    bridge = core.paligemma_with_expert
    expert = bridge.gemma_expert.model
    repeated_embeddings = prefix.embeddings.repeat_interleave(G1_PROBE_COUNT, dim=0)
    repeated_padding = prefix.padding.repeat_interleave(G1_PROBE_COUNT, dim=0)
    probes = torch.stack((fixed_probe, -fixed_probe), dim=0)
    probes = probes[None].expand(frames, -1, -1, -1).reshape(
        frames * G1_PROBE_COUNT, ACTION_HORIZON, 32
    )
    flow_time = torch.ones(frames * G1_PROBE_COUNT, device=probes.device)
    target_dtype = expert.layers[0].self_attn.q_proj.weight.dtype

    with (
        ActionLayerStateCapture(expert, detach=True) as layer_capture,
        _capture_autocast(prefix.embeddings.device),
    ):
        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
            probes, flow_time
        )
        padding = torch.cat((repeated_padding, suffix_padding), dim=1)
        attention = torch.cat(
            (torch.zeros_like(repeated_padding), suffix_attention), dim=1
        )
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(padding, attention)
        )
        positions = torch.cumsum(padding, dim=1) - 1
        (prefix_hidden, suffix_hidden), _ = bridge.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[
                repeated_embeddings.to(target_dtype),
                suffix.to(target_dtype),
            ],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        flow_velocity = core.action_out_proj(suffix_hidden)
        layer_states = layer_capture.stacked()

    if layer_states.shape[1:3] != (19, ACTION_HORIZON):
        raise NativeFactorError("policy-response Action Expert topology changed")
    prefix_width = int(prefix_hidden.shape[-1])
    prefix_hidden = prefix_hidden.detach().reshape(
        frames, G1_PROBE_COUNT, prefix_hidden.shape[1], prefix_width
    )
    # Prefix tokens cannot attend the suffix under the native mask.  Averaging
    # only removes duplicate numerical copies; the probe axis is retained on
    # every Action-side response.
    frozen_prefix = prefix_hidden.float().mean(1).to(prefix_hidden.dtype)
    language_mask = prefix.padding[:, image_tokens:].detach()
    return FrozenPolicyResponseChunk(
        start_frame=start_frame,
        patch_states=frozen_prefix[:, :image_tokens].detach(),
        language_states=frozen_prefix[:, image_tokens:].detach().masked_fill(
            ~language_mask[:, :, None], 0.0
        ),
        language_mask=language_mask,
        layer_states=layer_states.detach().reshape(
            frames,
            G1_PROBE_COUNT,
            19,
            ACTION_HORIZON,
            layer_states.shape[-1],
        ),
        flow_velocity=flow_velocity.detach().reshape(
            frames, G1_PROBE_COUNT, ACTION_HORIZON, 32
        ),
        suffix_noise=torch.stack((fixed_probe, -fixed_probe), dim=0).detach(),
    )


def merge_policy_response_chunks(
    chunks: Sequence[FrozenPolicyResponseChunk],
    *,
    frame_positions: torch.Tensor | None = None,
) -> FrozenPolicyResponseVideo:
    """Merge contiguous capture chunks without changing any candidate axis."""

    values = tuple(chunks)
    if not values:
        raise ValueError("policy-response video has no chunks")
    next_frame = 0
    for chunk in values:
        if (
            chunk.start_frame != next_frame
            or chunk.frame_count <= 0
            or not torch.equal(chunk.suffix_noise, values[0].suffix_noise)
        ):
            raise ValueError("policy-response chunks changed ordering or ownership")
        next_frame += chunk.frame_count
    frames = next_frame
    if frame_positions is None:
        frame_positions = torch.linspace(
            0.0,
            1.0,
            frames,
            device=values[0].patch_states.device,
            dtype=torch.float32,
        )
    if frame_positions.shape != (frames,):
        raise ValueError("policy-response frame positions changed")
    return FrozenPolicyResponseVideo(
        patch_states=torch.cat(tuple(chunk.patch_states for chunk in values)),
        language_states=torch.cat(tuple(chunk.language_states for chunk in values)),
        language_mask=torch.cat(tuple(chunk.language_mask for chunk in values)),
        layer_states=torch.cat(tuple(chunk.layer_states for chunk in values)),
        flow_velocity=torch.cat(tuple(chunk.flow_velocity for chunk in values)),
        suffix_noise=values[0].suffix_noise,
        frame_positions=frame_positions,
    )
