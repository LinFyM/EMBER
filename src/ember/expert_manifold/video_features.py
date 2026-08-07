"""Frozen PI0.5 high-level video innovations with no language-only LoRA path."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ember.expert_manifold.contract import ExpertManifoldError


NATIVE_IMAGE_TOKENS = 256


def phase_resample(value: torch.Tensor, slots: int) -> torch.Tensor:
    """Linearly align one ordered variable-length video to fixed phase slots."""

    if value.ndim != 2 or value.shape[0] < 2 or slots <= 1:
        raise ExpertManifoldError("video innovation needs at least two frames and phases")
    source_dtype = value.dtype
    aligned = F.interpolate(
        value.to(torch.float32).transpose(0, 1)[None],
        size=slots,
        mode="linear",
        align_corners=True,
    )
    return aligned[0].transpose(0, 1).to(source_dtype)


class FrozenPi05VideoInnovationEncoder(torch.nn.Module):
    """Subtract matched zero-image prompt hidden from each video-frame hidden."""

    def __init__(
        self,
        *,
        image_width: int,
        feature_width: int,
        phase_slots: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            image_width != 2048
            or feature_width != image_width
            or phase_slots <= 1
            or max_frames_per_encoder_call <= 0
            or action_horizon != 50
            or padded_action_dim != 32
            or initialization_seed < 0
        ):
            raise ExpertManifoldError("invalid frozen video-innovation topology")
        self.image_width = int(image_width)
        self.feature_width = int(feature_width)
        self.phase_slots = int(phase_slots)
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
            raise ExpertManifoldError("teacher frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    @torch.no_grad()
    def _encode(
        self,
        core: torch.nn.Module,
        frames: torch.Tensor | None,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        batch = language_tokens.shape[0]
        text_tokens = bridge.embed_language_tokens(language_tokens)
        if frames is None:
            image_tokens = text_tokens.new_zeros(
                batch, NATIVE_IMAGE_TOKENS, self.image_width
            )
        else:
            if frames.shape[0] != batch:
                raise ExpertManifoldError("frame-language video batch changed")
            image_tokens = bridge.embed_image(self._prepare_images(frames))
        if image_tokens.shape != (batch, NATIVE_IMAGE_TOKENS, self.image_width):
            raise ExpertManifoldError("PI0.5 image-token topology changed")
        prefix = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    batch,
                    NATIVE_IMAGE_TOKENS,
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
        attention = torch.cat(
            (torch.zeros_like(prefix_padding), suffix_attention), dim=1
        )
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(padding, attention)
        )
        positions = torch.cumsum(padding, dim=1) - 1
        dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        (prefix_hidden, suffix_hidden), _ = bridge.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        language_hidden = prefix_hidden[:, NATIVE_IMAGE_TOKENS:].to(torch.float32)
        count = task_span_mask.sum(dim=1, keepdim=True).to(torch.float32)
        grounded = language_hidden.masked_fill(~task_span_mask[..., None], 0.0).sum(dim=1)
        grounded = grounded / count
        if (
            suffix_hidden is None
            or grounded.shape != (batch, self.feature_width)
            or not bool(torch.isfinite(grounded).all())
        ):
            raise ExpertManifoldError("PI0.5 high-level video hidden changed")
        return grounded

    @torch.inference_mode()
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
        valid = (
            frames.ndim == 4
            and frames.shape[0] > 0
            and frame_video_ids.shape == (frames.shape[0],)
            and frame_video_ids.dtype == torch.long
            and language_tokens.ndim == 2
            and language_mask.shape == language_tokens.shape
            and language_mask.dtype == torch.bool
            and task_span_mask.shape == language_tokens.shape
            and task_span_mask.dtype == torch.bool
            and not bool((task_span_mask & ~language_mask).any())
            and bool(task_span_mask.any(dim=1).all())
            and int(frame_video_ids.min()) >= 0
            and int(frame_video_ids.max()) < videos
        )
        if not valid:
            raise ExpertManifoldError("invalid frozen video-innovation batch")
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
            bool((counts < 2).any())
            or not torch.equal(frame_video_ids, expected_ids)
            or not torch.equal(video_offsets.to(frames.device), expected_offsets)
        ):
            raise ExpertManifoldError("video innovations must be contiguous")
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise ExpertManifoldError("PI0.5 source policy topology changed")
        baseline = self._encode(
            core, None, language_tokens, language_mask, task_span_mask
        )
        frame_rows = []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            selected = frame_video_ids[start:stop]
            actual = self._encode(
                core,
                frames[start:stop],
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                task_span_mask.index_select(0, selected),
            )
            frame_rows.append(actual - baseline.index_select(0, selected))
        innovation = torch.cat(frame_rows)
        result = torch.stack(
            [
                phase_resample(innovation[left:right], self.phase_slots)
                for left, right in zip(
                    expected_offsets[:-1].tolist(),
                    expected_offsets[1:].tolist(),
                    strict=True,
                )
            ]
        )
        if result.shape != (videos, self.phase_slots, self.feature_width):
            raise ExpertManifoldError("phase-aligned video innovation changed shape")
        return result
