"""Full native evidence and repeated joint process-policy attention blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo

if TYPE_CHECKING:
    from ember.ecp.stage0 import ECPStage0Model


@dataclass(frozen=True)
class PolicyResponseEvidence:
    """Unpooled, source-separated, frame-aligned evidence for one video."""

    patches: torch.Tensor
    language: torch.Tensor
    language_valid: torch.Tensor
    response: torch.Tensor


class GatedMLP(torch.nn.Module):
    """A standard pre-norm gated residual MLP."""

    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = torch.nn.LayerNorm(width)
        self.input = torch.nn.Linear(width, 2 * hidden)
        self.output = torch.nn.Linear(hidden, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = self.input(self.norm(value)).chunk(2, dim=-1)
        return value + self.output(left * F.gelu(gate))


class PrefixTokenizer(torch.nn.Module):
    """Project frozen image and exact-language prefix tokens without pooling."""

    def __init__(self, prefix_width: int, width: int) -> None:
        super().__init__()
        self.patch_projection = torch.nn.Linear(prefix_width, width, bias=False)
        self.language_projection = torch.nn.Linear(prefix_width, width, bias=False)
        self.type_embedding = torch.nn.Embedding(2, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(
        self, video: FrozenPolicyResponseVideo
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.norm(
            self.patch_projection(video.patch_states) + self.type_embedding.weight[0]
        )
        language = self.norm(
            self.language_projection(video.language_states)
            + self.type_embedding.weight[1]
        )
        mask = video.language_mask
        if (
            patches.ndim != 3
            or language.ndim != 3
            or mask.shape != language.shape[:2]
            or patches.shape[0] != language.shape[0]
            or not patches.shape[1]
            or not torch.all(mask.any(1))
        ):
            raise ValueError("policy-response prefix topology changed")
        return patches, language, mask


class ResponseTokenizer(torch.nn.Module):
    """Project every layer/horizon once, before task-conditioned process reads."""

    def __init__(
        self,
        *,
        expert_width: int,
        width: int,
    ) -> None:
        super().__init__()
        self.state_projection = torch.nn.Linear(expert_width, width, bias=False)
        self.residual_projection = torch.nn.Linear(expert_width, width, bias=False)
        self.noise_projection = torch.nn.Linear(32, width, bias=False)
        self.velocity_projection = torch.nn.Linear(32, width, bias=False)
        self.layer_embedding = torch.nn.Embedding(19, width)
        self.horizon_embedding = torch.nn.Embedding(ACTION_HORIZON, width)
        self.channel_embedding = torch.nn.Embedding(8, width)
        self.norm = torch.nn.LayerNorm(width)

    @staticmethod
    def _even_odd(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.shape[1] != 2:
            raise ValueError("policy-response probe axis changed")
        return 0.5 * (value[:, 0] + value[:, 1]), 0.5 * (
            value[:, 0] - value[:, 1]
        )

    def forward(self, video: FrozenPolicyResponseVideo) -> torch.Tensor:
        states = video.layer_states
        if states.ndim != 5 or states.shape[1:4] != (2, 19, ACTION_HORIZON):
            raise ValueError("policy-response raw layer topology changed")
        frames = states.shape[0]
        fields = (
            self.state_projection(states),
            self.residual_projection(states[:, :, 1:] - states[:, :, :-1]),
            self.noise_projection(video.suffix_noise)[None].expand(frames, -1, -1, -1),
            self.velocity_projection(video.flow_velocity),
        )
        tokens = []
        for field, value in enumerate(fields):
            for probe, channel in enumerate(self._even_odd(value)):
                channel = channel + self.horizon_embedding.weight
                channel = channel + self.channel_embedding.weight[2 * field + probe]
                if channel.ndim == 4:
                    channel = channel + self.layer_embedding.weight[
                        :channel.shape[1]
                    ][None, :, None]
                    channel = channel.flatten(1, 2)
                tokens.append(channel)
        return self.norm(torch.cat(tokens, dim=1))


class JointProcessPolicyBlock(torch.nn.Module):
    """One joint block: policy feedback, grounded process read, time, whole policy."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.process_norm = torch.nn.LayerNorm(width)
        self.policy_norm = torch.nn.LayerNorm(width)
        self.evidence_norm = torch.nn.LayerNorm(width)
        self.policy_attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedback_attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.temporal_attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.process_read = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.policy_mixing = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.temporal_position = torch.nn.Linear(2, width, bias=False)
        self.process_mlp = GatedMLP(width)
        self.policy_mlp = GatedMLP(width)

    def _grounded_read(
        self, process: torch.Tensor, patches: torch.Tensor,
        language: torch.Tensor, language_valid: torch.Tensor, response: torch.Tensor,
    ) -> torch.Tensor:
        # Language conditions patch queries before the first full-response compression.
        for memory, valid in ((language, language_valid), (patches, None), (response, None)):
            memory = self.evidence_norm(memory)
            read, _ = self.policy_attention(
                self.process_norm(process), memory, memory,
                key_padding_mask=None if valid is None else ~valid, need_weights=False,
            )
            process = process + read
        return process

    def forward(
        self, process: torch.Tensor, policy: torch.Tensor, positions: torch.Tensor,
        patches: torch.Tensor, language: torch.Tensor,
        language_valid: torch.Tensor, response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames, work_tokens, width = process.shape
        if (
            positions.shape != (frames,) or policy.ndim != 2 or policy.shape[-1] != width
            or any(x.ndim != 3 or x.shape[0] != frames or x.shape[-1] != width
                   for x in (patches, language, response))
            or language_valid.shape != language.shape[:2]
        ):
            raise ValueError("joint process-policy evidence axes changed")
        policy_memory = self.policy_norm(policy)[None]
        feedback, _ = self.feedback_attention(
            self.process_norm(process).reshape(1, frames * work_tokens, width),
            policy_memory, policy_memory, need_weights=False,
        )
        process = self._grounded_read(
            process + feedback.reshape_as(process), patches, language, language_valid, response
        )
        temporal = self.process_norm(process).transpose(0, 1)
        position = self.temporal_position(torch.stack((positions, positions.square()), -1))
        read, _ = self.temporal_attention(
            temporal + position[None], temporal + position[None], temporal, need_weights=False
        )
        process = self.process_mlp(process + read.transpose(0, 1))
        memory = self.process_norm(process).reshape(1, frames * work_tokens, width)
        read, _ = self.process_read(self.policy_norm(policy)[None], memory, memory, need_weights=False)
        policy = policy + read[0]
        query = self.policy_norm(policy)[None]
        read, _ = self.policy_mixing(query, query, query, need_weights=False)
        return process, self.policy_mlp(policy + read[0])


class PolicyResponseEvidenceEncoder(torch.nn.Module):
    """Project exact prefix and full response while retaining every axis."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
    ) -> None:
        super().__init__()
        if not owners or width <= 0:
            raise ValueError("policy-response evidence topology changed")
        self.owners = tuple(owners)
        self.width = width
        self.prefix = PrefixTokenizer(prefix_width, width)
        self.response = ResponseTokenizer(
            expert_width=expert_width, width=width
        )

    def forward(
        self,
        video: FrozenPolicyResponseVideo,
        *,
        representation: str = "full",
    ) -> PolicyResponseEvidence:
        if representation != "full":
            raise ValueError("full policy-response is the only active representation")
        patches, language, language_valid = self.prefix(video)
        response = self.response(video)
        if (
            patches.shape[0] != video.frame_count
            or language.shape[0] != video.frame_count
            or language_valid.shape != language.shape[:2]
            or response.shape[:2] != (video.frame_count, 78 * ACTION_HORIZON)
        ):
            raise ValueError("policy-response evidence axes changed")
        return PolicyResponseEvidence(
            patches=patches,
            language=language,
            language_valid=language_valid,
            response=response,
        )

    @torch.no_grad()
    def initialize_from_stage0(self, stage0: "ECPStage0Model") -> dict[str, object]:
        """Reuse only G2-proven native projections and structural embeddings."""

        observer = stage0.encoder.observer
        source = observer.projector
        self.prefix.patch_projection.weight.copy_(observer.patch_projection.weight)
        self.prefix.language_projection.weight.copy_(observer.language_projection.weight)
        self.response.state_projection.weight.copy_(source.state_projection.weight)
        self.response.residual_projection.weight.copy_(source.delta_projection.weight)
        self.response.noise_projection.weight.copy_(source.noise_projection.weight)
        self.response.velocity_projection.weight.copy_(source.velocity_projection.weight)
        self.response.layer_embedding.weight[:18].copy_(source.layer_embedding.weight)
        binding = stage0.encoder.binding
        self.response.horizon_embedding.weight.copy_(binding.horizon_embedding)
        return {
            "reused": [
                "prefix_projections",
                "response_channel_projections",
                "layer_horizon_embeddings",
            ],
        }
