"""Frozen task and policy-aware video descriptors for the canonical Writer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video semantic interface changes."""


def temporal_video_tokens(
    frame_values: torch.Tensor,
    video_offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed four-term temporal kernel and retain its four value tokens."""

    if (
        frame_values.ndim != 2
        or frame_values.shape[0] <= 0
        or video_offsets.ndim != 1
        or video_offsets.dtype != torch.long
        or video_offsets.numel() < 2
    ):
        raise VideoProgramError("invalid temporal condition descriptor batch")
    offsets = tuple(int(value) for value in video_offsets.detach().cpu().tolist())
    if (
        offsets[0] != 0
        or offsets[-1] != frame_values.shape[0]
        or any(right <= left for left, right in zip(offsets, offsets[1:]))
    ):
        raise VideoProgramError("temporal condition offsets changed")
    rows = []
    for left, right in zip(offsets, offsets[1:]):
        selected = frame_values[left:right].to(torch.float32)
        count = selected.shape[0]
        tau = (
            torch.zeros(1, device=selected.device, dtype=torch.float32)
            if count == 1
            else torch.linspace(
                -1.0,
                1.0,
                count,
                device=selected.device,
                dtype=torch.float32,
            )
        )
        basis = torch.stack(
            (
                torch.ones_like(tau),
                tau,
                torch.cos(math.pi * tau),
                torch.sin(math.pi * tau),
            ),
            dim=0,
        )
        denominator = basis.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        pooled = (basis @ selected) / denominator
        rows.append(F.normalize(pooled, dim=-1, eps=1e-12))
    result = torch.stack(rows)
    if not bool(torch.isfinite(result).all()):
        raise VideoProgramError("temporal condition descriptor became non-finite")
    return result


def temporal_video_descriptor(
    frame_values: torch.Tensor,
    video_offsets: torch.Tensor,
) -> torch.Tensor:
    """Compatibility view of the four temporal tokens as one flat descriptor."""

    tokens = temporal_video_tokens(frame_values, video_offsets)
    return F.normalize(tokens.flatten(1), dim=-1, eps=1e-12)


class Pi05FrozenConditionDescriptor(torch.nn.Module):
    """Extract fixed task and policy-aware video descriptors without Writer grads."""

    NATIVE_IMAGE_TOKENS = 256
    FRAME_DESCRIPTOR_WIDTH = 128
    TEMPORAL_TERMS = 4

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
            min(
                image_width,
                expert_width,
                max_frames_per_encoder_call,
                action_horizon,
                padded_action_dim,
            )
            <= 0
            or action_horizon != 50
            or padded_action_dim != 32
            or initialization_seed < 0
        ):
            raise VideoProgramError("invalid frozen condition descriptor topology")
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
        projection = torch.randn(
            self.FRAME_DESCRIPTOR_WIDTH,
            image_width + expert_width,
            dtype=torch.float32,
            generator=generator,
        ) / math.sqrt(float(image_width + expert_width))
        self.register_buffer("frame_projection", projection, persistent=True)

    @property
    def task_descriptor_width(self) -> int:
        return self.image_width

    @property
    def video_descriptor_width(self) -> int:
        return self.FRAME_DESCRIPTOR_WIDTH * self.TEMPORAL_TERMS

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

    @staticmethod
    def _pack_hidden(
        hidden: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum_task_tokens: int,
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or task_span_mask.shape != hidden.shape[:2]
            or task_span_mask.dtype != torch.bool
            or maximum_task_tokens <= 0
            or int(task_span_mask.sum(dim=1).max()) > maximum_task_tokens
        ):
            raise VideoProgramError("task-token hidden packing changed")
        ordinal = (task_span_mask.to(torch.long).cumsum(dim=1) - 1).clamp_min(0)
        packed = hidden.new_zeros(
            hidden.shape[0], maximum_task_tokens, hidden.shape[-1]
        )
        return packed.scatter_add(
            1,
            ordinal[..., None].expand(-1, -1, hidden.shape[-1]),
            hidden * task_span_mask[..., None],
        )

    @torch.no_grad()
    def _encode_text(
        self,
        core: torch.nn.Module,
        language_tokens: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        batch = language_tokens.shape[0]
        text_tokens = torch.zeros(
            batch,
            maximum_task_tokens + 1,
            dtype=language_tokens.dtype,
            device=language_tokens.device,
        )
        text_padding = torch.zeros_like(text_tokens, dtype=torch.bool)
        text_tokens[:, 0] = language_tokens[:, 0]
        text_padding[:, 0] = True
        for row in range(batch):
            selected = language_tokens[row, task_span_mask[row]]
            text_tokens[row, 1 : selected.numel() + 1] = selected
            text_padding[row, 1 : selected.numel() + 1] = True
        text_embeds = bridge.embed_language_tokens(text_tokens)
        text_attention = torch.zeros_like(text_padding)
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(text_padding, text_attention)
        )
        positions = torch.cumsum(text_padding, dim=1) - 1
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        (text_hidden, suffix_hidden), _ = bridge.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[text_embeds.to(target_dtype), None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        if (
            suffix_hidden is not None
            or text_hidden.shape
            != (batch, maximum_task_tokens + 1, self.image_width)
        ):
            raise VideoProgramError("frozen text descriptor layout changed")
        packed = text_hidden[:, 1:].to(torch.float32)
        active = text_padding[:, 1:, None]
        descriptor = packed.masked_fill(~active, 0.0).sum(dim=1)
        descriptor = descriptor / active.sum(dim=1).clamp_min(1)
        return packed, descriptor

    @torch.no_grad()
    def _encode_frames(
        self,
        core: torch.nn.Module,
        frames: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        images = self._prepare_images(frames)
        image_tokens = bridge.embed_image(images)
        text_tokens = bridge.embed_language_tokens(language_tokens)
        prefix = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    image_tokens.shape[:2],
                    dtype=torch.bool,
                    device=frames.device,
                ),
                language_mask,
            ),
            dim=1,
        )
        prefix_attention = torch.zeros_like(prefix_padding)
        suffix_noise = self.fixed_suffix_noise[None].expand(frames.shape[0], -1, -1)
        timestep = torch.ones(
            frames.shape[0], dtype=torch.float32, device=frames.device
        )
        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
            suffix_noise, timestep
        )
        padding = torch.cat((prefix_padding, suffix_padding), dim=1)
        attention = torch.cat((prefix_attention, suffix_attention), dim=1)
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(padding, attention)
        )
        positions = torch.cumsum(padding, dim=1) - 1
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        (prefix_hidden, suffix_hidden), _ = bridge.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix.to(target_dtype), suffix.to(target_dtype)],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        if (
            prefix_hidden.shape[:2] != prefix.shape[:2]
            or prefix_hidden.shape[-1] != self.image_width
            or suffix_hidden.shape
            != (frames.shape[0], self.action_horizon, self.expert_width)
        ):
            raise VideoProgramError("frozen frame descriptor layout changed")
        packed_language = self._pack_hidden(
            prefix_hidden[:, self.NATIVE_IMAGE_TOKENS :],
            task_span_mask,
            maximum_task_tokens,
        ).to(torch.float32)
        action = suffix_hidden.to(torch.float32).mean(dim=1)
        return packed_language, action

    @torch.no_grad()
    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditions = language_tokens.shape[0]
        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frame_condition_ids.ndim != 1
            or frame_condition_ids.shape[0] != frames.shape[0]
            or frame_condition_ids.dtype != torch.long
            or language_tokens.ndim != 2
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
            or bool((task_span_mask & ~language_mask).any())
            or not bool(task_span_mask.any(dim=1).all())
            or int(frame_condition_ids.min()) < 0
            or int(frame_condition_ids.max()) >= conditions
        ):
            raise VideoProgramError("invalid frozen condition batch")
        counts = torch.bincount(frame_condition_ids, minlength=conditions)
        expected = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device), counts
        )
        if bool((counts <= 0).any()) or not torch.equal(frame_condition_ids, expected):
            raise VideoProgramError("condition frames must be contiguous")
        expected_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=frames.device),
                counts.cumsum(dim=0),
            )
        )
        if not torch.equal(video_offsets.to(frames.device), expected_offsets):
            raise VideoProgramError("condition video offsets changed")
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise VideoProgramError("PI05 source policy topology changed")
        task_counts = task_span_mask.sum(dim=1)
        maximum_task_tokens = int(task_counts.max())
        text_hidden, task_descriptor = self._encode_text(
            core, language_tokens, task_span_mask, maximum_task_tokens
        )
        frame_rows = []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            selected = frame_condition_ids[start:stop]
            joint, action = self._encode_frames(
                core,
                frames[start:stop],
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                task_span_mask.index_select(0, selected),
                maximum_task_tokens,
            )
            active = (
                torch.arange(maximum_task_tokens, device=frames.device)[None]
                < task_counts.index_select(0, selected)[:, None]
            )
            innovation = (joint - text_hidden.index_select(0, selected)).masked_fill(
                ~active[..., None], 0.0
            )
            innovation = innovation.sum(dim=1) / active.sum(dim=1, keepdim=True)
            combined = torch.cat((innovation, action), dim=-1)
            frame_rows.append(F.linear(combined, self.frame_projection))
        frame_descriptor = torch.cat(frame_rows, dim=0)
        video_descriptor = temporal_video_tokens(frame_descriptor, video_offsets)
        if (
            task_descriptor.shape != (conditions, self.task_descriptor_width)
            or video_descriptor.shape
            != (
                conditions,
                self.TEMPORAL_TERMS,
                self.FRAME_DESCRIPTOR_WIDTH,
            )
        ):
            raise VideoProgramError("frozen condition descriptors changed shape")
        return task_descriptor, video_descriptor
