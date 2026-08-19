"""Language-prior and video-posterior inference in a fixed functional code space."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class FunctionalCodeInferenceError(RuntimeError):
    """Raised when the language/video inference boundary changes."""


@dataclass(frozen=True)
class FunctionalCodePosterior:
    """All baseline codes and process evidence from one amortized inference pass."""

    language_code: torch.Tensor
    video_code: torch.Tensor
    posterior_delta: torch.Tensor
    posterior_confidence: torch.Tensor
    combined_code: torch.Tensor
    per_video_program: torch.Tensor
    per_video_summary: torch.Tensor
    video_condition_ids: torch.Tensor
    action_phase_predictions: torch.Tensor


class _MLP(torch.nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def _sinusoidal_positions(
    positions: torch.Tensor, width: int, *, dtype: torch.dtype
) -> torch.Tensor:
    if width % 2 or positions.ndim != 1:
        raise FunctionalCodeInferenceError("sinusoidal positions require an even width")
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = positions.float()[:, None] * frequencies[None]
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype)


class LanguageVideoCodeInference(torch.nn.Module):
    """Infer one compact code without moving the complete-LoRA decoder.

    Each video is encoded independently as initial state, goal state, and ordered
    event tokens. Only those complete per-video programs are aggregated across K.
    """

    ACTION_TOKENS = 50

    def __init__(
        self,
        *,
        feature_width: int,
        hidden_width: int,
        code_width: int,
        attention_heads: int,
        temporal_layers: int,
        phase_queries: int,
        event_queries: int,
        dropout: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        sizes = (
            feature_width,
            hidden_width,
            code_width,
            attention_heads,
            temporal_layers,
            phase_queries,
            event_queries,
        )
        if (
            any(value <= 0 for value in sizes)
            or feature_width % attention_heads
            or not 0.0 <= dropout < 1.0
        ):
            raise FunctionalCodeInferenceError("invalid code-inference dimensions")
        self.feature_width = int(feature_width)
        self.code_width = int(code_width)
        self.phase_count = int(phase_queries)
        self.event_count = int(event_queries)

        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        self.language_query = torch.nn.Parameter(
            torch.randn(feature_width, generator=generator) / math.sqrt(feature_width)
        )
        self.video_query = torch.nn.Parameter(
            torch.randn(feature_width, generator=generator) / math.sqrt(feature_width)
        )
        self.phase_queries = torch.nn.Parameter(
            torch.randn(phase_queries, feature_width, generator=generator)
            / math.sqrt(feature_width)
        )
        self.event_queries = torch.nn.Parameter(
            torch.randn(event_queries, feature_width, generator=generator)
            / math.sqrt(feature_width)
        )
        self.frame_grounding = torch.nn.MultiheadAttention(
            feature_width, attention_heads, dropout=dropout, batch_first=True
        )
        self.action_phase_reader = torch.nn.MultiheadAttention(
            feature_width, attention_heads, dropout=dropout, batch_first=True
        )
        self.action_alignment_head = torch.nn.Linear(feature_width, 7)
        self.event_reader = torch.nn.MultiheadAttention(
            feature_width, attention_heads, dropout=dropout, batch_first=True
        )
        self.frame_projection = torch.nn.Linear(feature_width * 2, feature_width)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=feature_width,
            nhead=attention_heads,
            dim_feedforward=hidden_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = torch.nn.TransformerEncoder(
            layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        self.video_summary = _MLP(feature_width * 4, hidden_width, feature_width)
        self.video_key = torch.nn.Linear(feature_width, feature_width, bias=False)
        self.language_code = _MLP(feature_width, hidden_width, code_width)
        self.video_code = _MLP(feature_width, hidden_width, code_width)
        self.posterior_delta = _MLP(feature_width * 3, hidden_width, code_width)
        self.posterior_confidence = _MLP(feature_width * 3, hidden_width, 1)

    def _language_summary(
        self, language_tokens: torch.Tensor, valid_task_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum(
            "btw,w->bt", language_tokens.float(), self.language_query.float()
        ) / math.sqrt(self.feature_width)
        scores = scores.masked_fill(~valid_task_tokens, -torch.inf)
        weights = scores.softmax(dim=1).to(language_tokens.dtype)
        return torch.einsum("bt,btw->bw", weights, language_tokens)

    def _validate_feature_layout(
        self,
        *,
        language_tokens: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        frame_tokens: torch.Tensor,
        visual_patch_tokens: torch.Tensor,
        action_probe_tokens: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> tuple[int, int, int]:
        conditions, task_tokens, width = language_tokens.shape
        frames = frame_tokens.shape[0]
        videos = video_offsets.numel() - 1
        valid = (
            language_tokens.ndim == 3
            and width == self.feature_width
            and valid_task_tokens.shape == (conditions, task_tokens)
            and valid_task_tokens.dtype == torch.bool
            and valid_task_tokens.any(dim=1).all()
            and frame_tokens.shape == (frames, task_tokens, width)
            and visual_patch_tokens.ndim == 3
            and visual_patch_tokens.shape[0] == frames
            and visual_patch_tokens.shape[1] > 0
            and visual_patch_tokens.shape[2] == width
            and action_probe_tokens.shape
            == (frames, self.ACTION_TOKENS, width)
            and frame_condition_ids.shape == (frames,)
            and frame_condition_ids.dtype == torch.long
            and video_offsets.ndim == 1
            and condition_video_offsets.shape == (conditions + 1,)
            and videos > 0
        )
        if not valid:
            raise FunctionalCodeInferenceError("invalid language/video feature layout")
        return conditions, frames, videos

    @staticmethod
    def _validate_ownership(
        *,
        conditions: int,
        frames: int,
        videos: int,
        frame_condition_ids: torch.Tensor,
        frame_positions: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> None:
        if frame_positions.shape != (frames,):
            raise FunctionalCodeInferenceError("invalid language/video feature layout")
        offsets = video_offsets.detach().cpu().tolist()
        condition_offsets = condition_video_offsets.detach().cpu().tolist()
        if (
            offsets[0] != 0
            or offsets[-1] != frames
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
            or condition_offsets[0] != 0
            or condition_offsets[-1] != videos
            or any(
                right <= left
                for left, right in zip(condition_offsets, condition_offsets[1:])
            )
            or int(frame_condition_ids.min()) < 0
            or int(frame_condition_ids.max()) >= conditions
        ):
            raise FunctionalCodeInferenceError("invalid video or condition ownership")

    def _frame_features(
        self,
        *,
        language_summary: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        frame_tokens: torch.Tensor,
        visual_patch_tokens: torch.Tensor,
        action_probe_tokens: torch.Tensor,
        frame_condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = language_summary.index_select(0, frame_condition_ids)[:, None]
        frame_valid = valid_task_tokens.index_select(0, frame_condition_ids)
        grounded, _ = self.frame_grounding(
            query,
            frame_tokens,
            frame_tokens,
            key_padding_mask=~frame_valid,
            need_weights=False,
        )
        neutral_query = self.video_query.to(visual_patch_tokens)[None, None].expand(
            frame_tokens.shape[0], -1, -1
        )
        neutral_grounded, _ = self.frame_grounding(
            neutral_query,
            visual_patch_tokens,
            visual_patch_tokens,
            need_weights=False,
        )
        action_positions = _sinusoidal_positions(
            torch.arange(self.ACTION_TOKENS, device=frame_tokens.device),
            self.feature_width,
            dtype=action_probe_tokens.dtype,
        )
        action_content = action_probe_tokens + action_positions[None]
        phase_queries = self.phase_queries.to(action_content)[None].expand(
            frame_tokens.shape[0], -1, -1
        )
        phases, _ = self.action_phase_reader(
            phase_queries, action_content, action_content, need_weights=False
        )
        phase_summary = phases.mean(dim=1)
        conditioned_frames = self.frame_projection(
            torch.cat((grounded[:, 0], phase_summary), dim=-1)
        )
        video_only_frames = self.frame_projection(
            torch.cat(
                (neutral_grounded[:, 0], visual_patch_tokens.mean(dim=1)), dim=-1
            )
        )
        return (
            conditioned_frames,
            video_only_frames,
            self.action_alignment_head(phases),
        )

    def _video_programs(
        self,
        frame_features: torch.Tensor,
        frame_positions: torch.Tensor,
        video_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        programs = []
        summaries = []
        offsets = video_offsets.detach().cpu().tolist()
        for start, stop in zip(offsets, offsets[1:]):
            content = frame_features[start:stop]
            positions = frame_positions[start:stop].float()
            positions = (positions - positions[0]) / (positions[-1] - positions[0]).clamp_min(1)
            hidden = self.temporal(
                (content + _sinusoidal_positions(
                    positions, self.feature_width, dtype=content.dtype
                ))[None]
            )[0]
            event_query = self.event_queries.to(hidden)[None]
            events, _ = self.event_reader(
                event_query, hidden[None], hidden[None], need_weights=False
            )
            events = events[0]
            transitions = (
                hidden[1:] - hidden[:-1]
                if hidden.shape[0] > 1
                else torch.zeros_like(hidden[:1])
            )
            program = torch.cat((hidden[:1], hidden[-1:], events), dim=0)
            summary = self.video_summary(
                torch.cat(
                    (
                        hidden[0],
                        hidden[-1],
                        events.mean(dim=0),
                        transitions.mean(dim=0),
                    ),
                    dim=-1,
                )
            )
            programs.append(program)
            summaries.append(summary)
        return torch.stack(programs), torch.stack(summaries)

    def _aggregate_videos(
        self,
        language_summary: torch.Tensor,
        video_summary: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioned = []
        means = []
        variances = []
        condition_ids = []
        offsets = condition_video_offsets.detach().cpu().tolist()
        for condition, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            values = video_summary[start:stop]
            scores = torch.einsum(
                "vw,w->v",
                self.video_key(values).float(),
                language_summary[condition].float(),
            ) / math.sqrt(self.feature_width)
            weights = scores.softmax(dim=0).to(values.dtype)
            mean = values.mean(dim=0)
            conditioned.append(torch.einsum("v,vw->w", weights, values))
            means.append(mean)
            variances.append((values - mean).square().mean(dim=0))
            condition_ids.extend([condition] * (stop - start))
        return (
            torch.stack(conditioned),
            torch.stack(means),
            torch.stack(variances),
            torch.tensor(condition_ids, device=video_summary.device, dtype=torch.long),
        )

    def forward(
        self,
        *,
        language_tokens: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        frame_tokens: torch.Tensor,
        visual_patch_tokens: torch.Tensor,
        action_probe_tokens: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        frame_positions: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> FunctionalCodePosterior:
        conditions, frames, videos = self._validate_feature_layout(
            language_tokens=language_tokens,
            valid_task_tokens=valid_task_tokens,
            frame_tokens=frame_tokens,
            visual_patch_tokens=visual_patch_tokens,
            action_probe_tokens=action_probe_tokens,
            frame_condition_ids=frame_condition_ids,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
        )
        self._validate_ownership(
            conditions=conditions,
            frames=frames,
            videos=videos,
            frame_condition_ids=frame_condition_ids,
            frame_positions=frame_positions,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
        )
        language = self._language_summary(language_tokens, valid_task_tokens)
        frames, video_only_frames, action_phase_predictions = self._frame_features(
            language_summary=language,
            valid_task_tokens=valid_task_tokens,
            frame_tokens=frame_tokens,
            visual_patch_tokens=visual_patch_tokens,
            action_probe_tokens=action_probe_tokens,
            frame_condition_ids=frame_condition_ids,
        )
        programs, video_summaries = self._video_programs(
            frames, frame_positions, video_offsets
        )
        _, video_only_summaries = self._video_programs(
            video_only_frames, frame_positions, video_offsets
        )
        conditioned, _, video_variance, video_condition_ids = (
            self._aggregate_videos(
                language, video_summaries, condition_video_offsets
            )
        )
        posterior_features = torch.cat(
            (language, conditioned, video_variance), dim=-1
        )
        language_code = self.language_code(language)
        condition_offsets = condition_video_offsets.detach().cpu().tolist()
        video_only_mean = torch.stack(
            [
                video_only_summaries[left:right].mean(dim=0)
                for left, right in zip(
                    condition_offsets[:-1], condition_offsets[1:]
                )
            ]
        )
        video_code = self.video_code(video_only_mean)
        delta = self.posterior_delta(posterior_features)
        confidence = self.posterior_confidence(posterior_features).sigmoid()
        return FunctionalCodePosterior(
            language_code=language_code,
            video_code=video_code,
            posterior_delta=delta,
            posterior_confidence=confidence,
            combined_code=language_code + confidence * delta,
            per_video_program=programs,
            per_video_summary=video_summaries,
            video_condition_ids=video_condition_ids,
            action_phase_predictions=action_phase_predictions,
        )
