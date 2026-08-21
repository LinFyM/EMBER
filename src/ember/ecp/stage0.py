"""Stage 0 video orchestration and training-only action grounding."""

from __future__ import annotations

from dataclasses import dataclass

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
    action_phase_predictions: torch.Tensor


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
            width=program_width, event_slots=event_slots
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core = policy.model
        bridge = core.paligemma_with_expert
        with torch.no_grad():
            language_embeddings = bridge.embed_language_tokens(language_tokens)
        patches = []
        languages = []
        lattices = []
        probe = self.fixed_suffix_noise if suffix_noise is None else suffix_noise
        if probe.shape != self.fixed_suffix_noise.shape:
            raise ValueError("ECP observer probe must have shape [50, 32]")
        for start in range(0, frames.shape[0], self.max_frames_per_call):
            stop = min(start + self.max_frames_per_call, frames.shape[0])
            condition_ids = frame_condition_ids[start:stop]
            with torch.no_grad():
                image_embeddings = bridge.embed_image(
                    self._prepare_images(frames[start:stop])
                )
            selected_language = language_embeddings.index_select(0, condition_ids)
            selected_mask = language_mask.index_select(0, condition_ids)
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
            observed = self.observer(
                core,
                prefix,
                prefix_padding,
                probe[None].expand(stop - start, -1, -1),
                torch.ones(stop - start, device=frames.device),
            )
            patches.append(observed.patch_states)
            languages.append(observed.language_states)
            lattices.append(observed.owner_lattice)
        return torch.cat(patches), torch.cat(languages), torch.cat(lattices)

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        patch, language, lattice = self._native_frames(
            policy=policy,
            frames=frames,
            frame_condition_ids=frame_condition_ids,
            language_tokens=language_tokens,
            language_mask=language_mask,
            suffix_noise=suffix_noise,
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
        candidates, confidence = self.matcher(
            patch, language, frame_mask, video_language_mask
        )
        bound = self.binding(candidates, confidence, lattice, frame_mask)
        program = self.segmenter(bound, confidence, frame_mask)
        summary_weights = program.presence / program.presence.sum(
            1, keepdim=True
        ).clamp_min(1e-6)
        summary = torch.einsum(
            "ve,ved->vd", summary_weights, program.process.mean(2)
        )
        return (
            program.process,
            program.presence,
            program.uncertainty,
            program.assignment,
            program.state_posterior,
            confidence,
            frame_mask,
            summary,
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

    def forward(self, **inputs: torch.Tensor | torch.nn.Module) -> ECPStage0Output:
        (
            process,
            presence,
            uncertainty,
            assignment,
            posterior,
            confidence,
            frame_mask,
            summary,
        ) = self.encoder(**inputs)
        owner_weights = (
            self.action_owner_score(torch.tanh(process)).squeeze(-1).softmax(2)
        )
        event_state = torch.einsum("vej,vejd->ved", owner_weights, process)
        action = self.action_head(event_state).reshape(
            process.shape[0], process.shape[1], self.action_phases, 7
        )
        return ECPStage0Output(
            process=process,
            presence=presence,
            uncertainty=uncertainty,
            assignment=assignment,
            state_posterior=posterior,
            confidence=confidence,
            frame_mask=frame_mask,
            program_summary=summary,
            action_phase_predictions=action,
        )
