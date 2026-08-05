"""Frozen PI05 policy-layer traces for the canonical K4 Writer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video trace interface changes."""


def temporal_trace_tokens(
    frame_values: torch.Tensor,
    video_offsets: torch.Tensor,
    *,
    terms: int,
) -> torch.Tensor:
    """Project each video's policy-layer traces onto orthonormal DCT-II routes."""

    if (
        frame_values.ndim != 3
        or frame_values.shape[0] <= 0
        or terms <= 0
        or video_offsets.ndim != 1
        or video_offsets.dtype != torch.long
        or video_offsets.numel() < 2
    ):
        raise VideoProgramError("invalid temporal policy-trace batch")
    offsets = tuple(int(value) for value in video_offsets.detach().cpu().tolist())
    if (
        offsets[0] != 0
        or offsets[-1] != frame_values.shape[0]
        or any(right - left < terms for left, right in zip(offsets, offsets[1:]))
    ):
        raise VideoProgramError("temporal policy-trace offsets changed")
    rows = []
    for left, right in zip(offsets, offsets[1:]):
        selected = frame_values[left:right].to(torch.float32)
        count = selected.shape[0]
        time = torch.arange(count, device=selected.device, dtype=torch.float32)
        frequency = torch.arange(terms, device=selected.device, dtype=torch.float32)
        basis = torch.cos(
            math.pi * frequency[:, None] * (time[None] + 0.5) / float(count)
        )
        basis[0].mul_(math.sqrt(0.5))
        basis = basis * math.sqrt(2.0 / float(count))
        pooled = torch.einsum("tf,fgh->gth", basis, selected)
        rows.append(F.normalize(pooled, dim=-1, eps=1e-12))
    result = torch.stack(rows)
    if not bool(torch.isfinite(result).all()):
        raise VideoProgramError("temporal policy trace became non-finite")
    return result


class Pi05FrozenConditionDescriptor(torch.nn.Module):
    """Extract baseline-subtracted video innovations at all PI05 policy layers."""

    NATIVE_IMAGE_TOKENS = 256
    EXPERT_LAYERS = 18
    POLICY_GROUPS = 20
    TRACE_WIDTH = 1024
    TEMPORAL_TERMS = 16

    def __init__(
        self,
        *,
        image_width: int,
        expert_width: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            image_width != 2048
            or expert_width != self.TRACE_WIDTH
            or max_frames_per_encoder_call <= 0
            or action_horizon != 50
            or padded_action_dim != 32
            or initialization_seed < 0
        ):
            raise VideoProgramError("invalid frozen policy-trace topology")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.max_frames_per_encoder_call = int(max_frames_per_encoder_call)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        self.register_buffer(
            "fixed_suffix_noise",
            torch.randn(
                action_horizon,
                padded_action_dim,
                dtype=torch.float32,
                generator=generator,
            ),
            persistent=True,
        )

    @staticmethod
    def _prepare_images(frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frames.shape[1] != 3
            or frames.dtype != torch.uint8
        ):
            raise VideoProgramError("teacher frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    @torch.no_grad()
    def _encode_layer_traces(
        self,
        core: torch.nn.Module,
        frames: torch.Tensor | None,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import (
            compute_layer_complete,
            layernorm_forward,
            make_att_2d_masks,
        )

        bridge = core.paligemma_with_expert
        paligemma_layers = bridge.paligemma.model.language_model.layers
        expert_layers = bridge.gemma_expert.model.layers
        if len(paligemma_layers) != self.EXPERT_LAYERS or len(expert_layers) != self.EXPERT_LAYERS:
            raise VideoProgramError("PI05 action-expert depth changed")
        batch = language_tokens.shape[0]
        text_tokens = bridge.embed_language_tokens(language_tokens)
        if frames is None:
            image_tokens = text_tokens.new_zeros(
                batch, self.NATIVE_IMAGE_TOKENS, self.image_width
            )
        else:
            if frames.shape[0] != batch:
                raise VideoProgramError("frame-language trace batch changed")
            image_tokens = bridge.embed_image(self._prepare_images(frames))
        if image_tokens.shape != (batch, self.NATIVE_IMAGE_TOKENS, self.image_width):
            raise VideoProgramError("PI05 image-token topology changed")
        prefix = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    batch,
                    self.NATIVE_IMAGE_TOKENS,
                    dtype=torch.bool,
                    device=language_tokens.device,
                ),
                language_mask,
            ),
            dim=1,
        )
        suffix_noise = self.fixed_suffix_noise[None].expand(batch, -1, -1)
        timestep = torch.ones(batch, dtype=torch.float32, device=language_tokens.device)
        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
            suffix_noise, timestep
        )
        padding = torch.cat((prefix_padding, suffix_padding), dim=1)
        attention = torch.cat((torch.zeros_like(prefix_padding), suffix_attention), dim=1)
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(padding, attention)
        )
        positions = torch.cumsum(padding, dim=1) - 1
        target_dtype = paligemma_layers[0].self_attn.q_proj.weight.dtype
        hidden = [prefix.to(target_dtype), suffix.to(target_dtype)]
        adarms_cond = [None, adarms]
        traces = [hidden[1].to(torch.float32).mean(dim=1)]
        rotary = bridge.paligemma.model.language_model.rotary_emb
        for paligemma_layer, expert_layer in zip(
            paligemma_layers, expert_layers, strict=True
        ):
            normalized, _ = layernorm_forward(
                expert_layer.input_layernorm,
                hidden[1],
                adarms,
            )
            traces.append(normalized.to(torch.float32).mean(dim=1))
            hidden = compute_layer_complete(
                hidden,
                mask,
                positions,
                adarms_cond,
                layers=(paligemma_layer, expert_layer),
                rotary_emb=rotary,
            )
        final, _ = layernorm_forward(
            bridge.gemma_expert.model.norm,
            hidden[1],
            adarms,
        )
        traces.append(final.to(torch.float32).mean(dim=1))
        result = torch.stack(traces, dim=1)
        if result.shape != (batch, self.POLICY_GROUPS, self.TRACE_WIDTH):
            raise VideoProgramError("frozen policy-layer trace layout changed")
        return result

    @torch.no_grad()
    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_video_ids: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> torch.Tensor:
        videos = language_tokens.shape[0]
        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frame_video_ids.shape != (frames.shape[0],)
            or frame_video_ids.dtype != torch.long
            or language_tokens.ndim != 2
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
            or bool((task_span_mask & ~language_mask).any())
            or not bool(task_span_mask.any(dim=1).all())
            or int(frame_video_ids.min()) < 0
            or int(frame_video_ids.max()) >= videos
        ):
            raise VideoProgramError("invalid frozen policy-trace batch")
        counts = torch.bincount(frame_video_ids, minlength=videos)
        expected_ids = torch.repeat_interleave(
            torch.arange(videos, device=frames.device), counts
        )
        expected_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=frames.device),
                counts.cumsum(dim=0),
            )
        )
        if (
            bool((counts < self.TEMPORAL_TERMS).any())
            or not torch.equal(frame_video_ids, expected_ids)
            or not torch.equal(video_offsets.to(frames.device), expected_offsets)
        ):
            raise VideoProgramError("policy-trace videos must be contiguous and long enough")
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise VideoProgramError("PI05 source policy topology changed")
        baseline = self._encode_layer_traces(
            core,
            None,
            language_tokens,
            language_mask,
        )
        frame_rows = []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            selected = frame_video_ids[start:stop]
            actual = self._encode_layer_traces(
                core,
                frames[start:stop],
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
            )
            frame_rows.append(actual - baseline.index_select(0, selected))
        frame_innovation = torch.cat(frame_rows, dim=0)
        video_traces = temporal_trace_tokens(
            frame_innovation,
            video_offsets,
            terms=self.TEMPORAL_TERMS,
        )
        if video_traces.shape != (
            videos,
            self.POLICY_GROUPS,
            self.TEMPORAL_TERMS,
            self.TRACE_WIDTH,
        ):
            raise VideoProgramError("frozen video policy traces changed shape")
        return video_traces
