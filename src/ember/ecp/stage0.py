"""Stage 0 video orchestration and training-only action grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ember.ecp.contracts import TargetOwner
from ember.ecp.events import (
    EventConditionedHorizonBinding,
    OrderedEventSegmenter,
    TaskGroundedTransitionMatcher,
)
from ember.ecp.observer import ECPNativeObserver


@dataclass(frozen=True)
class ECPStage0Output:
    process: torch.Tensor
    presence: torch.Tensor
    uncertainty: torch.Tensor
    assignment: torch.Tensor
    state_posterior: torch.Tensor
    confidence: torch.Tensor
    frame_mask: torch.Tensor
    program_summary: torch.Tensor
    frame_action_predictions: torch.Tensor
    event_action_predictions: torch.Tensor


@dataclass(frozen=True)
class ECPVideoEncoderOutput:
    process: torch.Tensor
    presence: torch.Tensor
    uncertainty: torch.Tensor
    assignment: torch.Tensor
    state_posterior: torch.Tensor
    confidence: torch.Tensor
    frame_mask: torch.Tensor
    program_summary: torch.Tensor
    frame_owner_evidence: torch.Tensor
    patch_states: torch.Tensor
    language_summary: torch.Tensor
    scene_transition: torch.Tensor


class ECPVideoEncoder(torch.nn.Module):
    """Encode independently ordered videos through one shared native PI0.5 graph."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        program_width: int = 128,
        event_slots: int = 8,
        presence_threshold_fraction: float = 0.08,
        max_frames_per_call: int = 8,
        fixed_probe_seed: int = 20260821,
    ) -> None:
        super().__init__()
        self.max_frames_per_call = max_frames_per_call
        self.observer = ECPNativeObserver(
            owners,
            prefix_width=prefix_width,
            expert_width=expert_width,
            program_width=program_width,
        )
        self.matcher = TaskGroundedTransitionMatcher(width=program_width)
        self.binding = EventConditionedHorizonBinding(
            width=program_width, owners=len(owners)
        )
        self.segmenter = OrderedEventSegmenter(
            width=program_width,
            event_slots=event_slots,
            presence_threshold_fraction=presence_threshold_fraction,
        )
        generator = torch.Generator(device="cpu").manual_seed(fixed_probe_seed)
        self.register_buffer(
            "fixed_suffix_noise",
            torch.randn(50, 32, generator=generator),
            persistent=True,
        )

    @staticmethod
    def _prepare_images(frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        value = frames.to(torch.float32).div(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    @staticmethod
    def _pad_videos(
        value: torch.Tensor, offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundaries = offsets.detach().cpu().tolist()
        lengths = [stop - start for start, stop in zip(boundaries, boundaries[1:])]
        maximum = max(lengths)
        padded = value.new_zeros(len(lengths), maximum, *value.shape[1:])
        mask = torch.zeros(
            len(lengths), maximum, dtype=torch.bool, device=value.device
        )
        for row, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            length = stop - start
            padded[row, :length] = value[start:stop]
            mask[row, :length] = True
        return padded, mask

    def _native_frames(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        suffix_noise: torch.Tensor | None,
        action_meta_lora: Any | None,
        install_action_meta_lora: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core = policy.model
        bridge = core.paligemma_with_expert
        language_embeddings = self.embed_language_conditions(policy, language_tokens)
        patches = []
        languages = []
        lattices = []
        probe = self.fixed_suffix_noise if suffix_noise is None else suffix_noise
        if probe.shape != self.fixed_suffix_noise.shape:
            raise ValueError("ECP observer probe must have shape [50, 32]")
        for start in range(0, frames.shape[0], self.max_frames_per_call):
            stop = min(start + self.max_frames_per_call, frames.shape[0])
            condition_ids = frame_condition_ids[start:stop]
            prefix, prefix_padding = self.prepare_frame_prefix(
                policy=policy,
                frames=frames[start:stop],
                frame_condition_ids=condition_ids,
                language_embeddings=language_embeddings,
                language_mask=language_mask,
            )
            adapter_context = (
                action_meta_lora.installed(bridge.gemma_expert.model)
                if action_meta_lora is not None and install_action_meta_lora
                else None
            )
            observed = self.observer(
                core,
                prefix,
                prefix_padding,
                probe[None].expand(stop - start, -1, -1),
                torch.ones(stop - start, device=frames.device),
                track_action_adapter_grad=action_meta_lora is not None,
                action_adapter_context=adapter_context,
            )
            patches.append(observed.patch_states)
            languages.append(observed.language_states)
            lattices.append(observed.owner_lattice)
        return torch.cat(patches), torch.cat(languages), torch.cat(lattices)

    @staticmethod
    @torch.no_grad()
    def embed_language_conditions(
        policy: torch.nn.Module, language_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Embed exact task language once for Stage 0 and Pass B alike."""

        return policy.model.paligemma_with_expert.embed_language_tokens(language_tokens)

    @torch.no_grad()
    def prepare_frame_prefix(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_embeddings: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the exact teacher-frame image/language prefix without state."""

        bridge = policy.model.paligemma_with_expert
        image_embeddings = bridge.embed_image(self._prepare_images(frames))
        selected_language = language_embeddings.index_select(0, frame_condition_ids)
        selected_mask = language_mask.index_select(0, frame_condition_ids)
        prefix = torch.cat((image_embeddings, selected_language), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    image_embeddings.shape[:2],
                    dtype=torch.bool,
                    device=frames.device,
                ),
                selected_mask,
            ),
            dim=1,
        )
        return prefix, prefix_padding

    def forward(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        video_offsets: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        suffix_noise: torch.Tensor | None = None,
        action_meta_lora: Any | None = None,
        install_action_meta_lora: bool = True,
    ) -> ECPVideoEncoderOutput:
        patch, language, lattice = self._native_frames(
            policy=policy,
            frames=frames,
            frame_condition_ids=frame_condition_ids,
            language_tokens=language_tokens,
            language_mask=language_mask,
            suffix_noise=suffix_noise,
            action_meta_lora=action_meta_lora,
            install_action_meta_lora=install_action_meta_lora,
        )
        patch, frame_mask = self._pad_videos(patch, video_offsets)
        language, _ = self._pad_videos(language, video_offsets)
        lattice, _ = self._pad_videos(lattice, video_offsets)
        counts = frame_mask.sum(1).to(language.dtype)
        language = language.sum(1) / counts[:, None, None]
        video_condition_ids = frame_condition_ids.index_select(
            0, video_offsets[:-1].to(frame_condition_ids.device)
        )
        video_language_mask = language_mask.index_select(0, video_condition_ids)
        language_weights = video_language_mask.to(language.dtype)
        language_summary = (
            language * language_weights[:, :, None]
        ).sum(1) / language_weights.sum(1, keepdim=True).clamp_min(1)
        patch_scores = torch.einsum(
            "vtpd,vd->vtp", patch, language_summary
        ) / patch.shape[-1] ** 0.5
        patch_summary = torch.einsum(
            "vtp,vtpd->vtd", patch_scores.softmax(-1), patch
        )
        final_indices = frame_mask.sum(1).clamp_min(1) - 1
        final_summary = patch_summary[
            torch.arange(patch_summary.shape[0], device=patch_summary.device),
            final_indices,
        ]
        scene_transition = torch.cat(
            (
                patch_summary[:, 0],
                final_summary,
                final_summary - patch_summary[:, 0],
            ),
            dim=-1,
        )
        candidates, confidence = self.matcher(
            patch, language, frame_mask, video_language_mask
        )
        bound = self.binding(candidates, confidence, lattice, frame_mask)
        candidate_weights = confidence.softmax(-1)
        frame_owner_evidence = torch.einsum(
            "vtm,vtmjd->vtjd", candidate_weights, bound
        ).masked_fill(~frame_mask[:, :, None, None], 0.0)
        program = self.segmenter(bound, confidence, frame_mask)
        summary_weights = program.presence / program.presence.sum(
            1, keepdim=True
        ).clamp_min(1e-6)
        summary = torch.einsum(
            "ve,ved->vd", summary_weights, program.process.mean(2)
        )
        return ECPVideoEncoderOutput(
            process=program.process,
            presence=program.presence,
            uncertainty=program.uncertainty,
            assignment=program.assignment,
            state_posterior=program.state_posterior,
            confidence=confidence,
            frame_mask=frame_mask,
            program_summary=summary,
            frame_owner_evidence=frame_owner_evidence,
            patch_states=patch,
            language_summary=language_summary,
            scene_transition=scene_transition,
        )


class ECPStage0Model(torch.nn.Module):
    """Native video encoder plus an auxiliary cross-episode action head."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        program_width: int = 128,
        event_slots: int = 8,
        action_phases: int = 10,
        presence_threshold_fraction: float = 0.08,
        max_frames_per_call: int = 8,
        fixed_probe_seed: int = 20260821,
    ) -> None:
        super().__init__()
        self.action_phases = action_phases
        self.encoder = ECPVideoEncoder(
            owners,
            prefix_width=prefix_width,
            expert_width=expert_width,
            program_width=program_width,
            event_slots=event_slots,
            presence_threshold_fraction=presence_threshold_fraction,
            max_frames_per_call=max_frames_per_call,
            fixed_probe_seed=fixed_probe_seed,
        )
        self.action_owner_score = torch.nn.Linear(program_width, 1, bias=False)
        self.action_head = torch.nn.Sequential(
            torch.nn.LayerNorm(program_width),
            torch.nn.Linear(program_width, program_width),
            torch.nn.GELU(),
            torch.nn.Linear(program_width, action_phases * 7),
        )

    def _predict_actions(self, owner_states: torch.Tensor) -> torch.Tensor:
        owner_weights = self.action_owner_score(
            torch.tanh(owner_states)
        ).squeeze(-1).softmax(-1)
        state = (owner_weights[..., None] * owner_states).sum(-2)
        return self.action_head(state).reshape(
            *owner_states.shape[:-2], self.action_phases, 7
        )

    def forward(self, **inputs: torch.Tensor | torch.nn.Module) -> ECPStage0Output:
        encoded = self.encoder(**inputs)
        return ECPStage0Output(
            process=encoded.process,
            presence=encoded.presence,
            uncertainty=encoded.uncertainty,
            assignment=encoded.assignment,
            state_posterior=encoded.state_posterior,
            confidence=encoded.confidence,
            frame_mask=encoded.frame_mask,
            program_summary=encoded.program_summary,
            frame_action_predictions=self._predict_actions(
                encoded.frame_owner_evidence
            ),
            event_action_predictions=self._predict_actions(encoded.process),
        )
