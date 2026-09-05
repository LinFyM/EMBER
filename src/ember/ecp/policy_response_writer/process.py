"""Tokenize frozen PI0.5 evidence without creating a learned video code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
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
    """Keep the full probe x horizon x response-channel field until attention."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        expert_width: int,
        width: int,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.state_projection = torch.nn.Linear(expert_width, width, bias=False)
        self.residual_projection = torch.nn.Linear(expert_width, width, bias=False)
        self.noise_projection = torch.nn.Linear(32, width, bias=False)
        self.velocity_projection = torch.nn.Linear(32, width, bias=False)
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(owners), width))
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        self.layer_embedding = torch.nn.Embedding(19, width)
        self.horizon_embedding = torch.nn.Embedding(ACTION_HORIZON, width)
        self.channel_embedding = torch.nn.Embedding(8, width)
        self.norm = torch.nn.LayerNorm(width)
        family_order = tuple(TargetFamily)
        self.register_buffer(
            "family_ids",
            torch.tensor([family_order.index(owner.family) for owner in owners]),
            persistent=False,
        )
        state_layers = []
        residual_layers = []
        for owner in owners:
            if owner.layer is not None:
                state_layers.append(owner.layer)
                residual_layers.append(owner.layer)
            elif owner.family is TargetFamily.ACTION_IN:
                state_layers.append(0)
                residual_layers.append(0)
            else:
                state_layers.append(18)
                residual_layers.append(17)
        self.register_buffer(
            "state_layers", torch.tensor(state_layers), persistent=False
        )
        self.register_buffer(
            "residual_layers", torch.tensor(residual_layers), persistent=False
        )
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)

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
        residuals = states[:, :, 1:] - states[:, :, :-1]
        state = self.state_projection(states.index_select(2, self.state_layers))
        residual = self.residual_projection(
            residuals.index_select(2, self.residual_layers)
        )
        frames = states.shape[0]
        noise = self.noise_projection(video.suffix_noise)[None, :, None].expand(
            frames, -1, len(self.owners), -1, -1
        )
        velocity = self.velocity_projection(video.flow_velocity)[:, :, None].expand(
            -1, -1, len(self.owners), -1, -1
        )
        channels = []
        for value in (state, residual, noise, velocity):
            channels.extend(self._even_odd(value))
        tokens = torch.stack(channels, dim=3)
        owner = (
            self.owner_embedding
            + self.family_embedding(self.family_ids)
            + self.layer_embedding(self.state_layers)
        )
        tokens = tokens + owner[None, :, None, None]
        tokens = tokens + self.horizon_embedding.weight[None, None, :, None]
        tokens = tokens + self.channel_embedding.weight[None, None, None]
        return self.norm(tokens.flatten(2, 3))


class UnifiedPolicyNativeFactorBlock(torch.nn.Module):
    """One copyable source-separated evidence, time, rank-side transformer block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.evidence_query_norm = torch.nn.LayerNorm(width)
        self.policy_memory_norm = torch.nn.LayerNorm(width)
        self.native_memory_norm = torch.nn.LayerNorm(width)
        self.policy_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.native_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.temporal_norm = torch.nn.LayerNorm(width)
        self.temporal_position = torch.nn.Linear(2, width, bias=False)
        self.temporal_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.factor_norm = torch.nn.LayerNorm(width)
        self.factor_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def _position(self, value: torch.Tensor) -> torch.Tensor:
        position = value.clamp(0.0, 1.0)
        return self.temporal_position(
            torch.stack((position, position.square()), dim=-1)
        )

    def _evidence_read(
        self,
        query: torch.Tensor,
        patches: torch.Tensor,
        language: torch.Tensor,
        language_valid: torch.Tensor,
        response: torch.Tensor,
        input_bank: torch.Tensor,
        output_bank: torch.Tensor,
    ) -> torch.Tensor:
        frames, ranks, sides, width = query.shape
        if (
            sides != 2
            or patches.ndim != 3
            or patches.shape[0] != frames
            or language.ndim != 3
            or language.shape[0] != frames
            or language_valid.shape != language.shape[:2]
            or not patches.shape[1]
            or response.ndim != 3
            or response.shape[0] != frames
            or input_bank.ndim != 3
            or output_bank.ndim != 3
            or input_bank.shape[0] != frames
            or output_bank.shape[0] != frames
            or any(
                value.shape[-1] != width
                for value in (
                    patches,
                    language,
                    response,
                    input_bank,
                    output_bank,
                )
            )
        ):
            raise ValueError("unified policy-native evidence axes changed")
        bank_tokens = max(input_bank.shape[1], output_bank.shape[1])
        input_memory = F.pad(
            input_bank, (0, 0, 0, bank_tokens - input_bank.shape[1])
        )
        output_memory = F.pad(
            output_bank, (0, 0, 0, bank_tokens - output_bank.shape[1])
        )
        bank = torch.stack((input_memory, output_memory), dim=1)
        bank_valid = torch.arange(bank_tokens, device=query.device)[
            None, None
        ] < torch.tensor(
            (input_bank.shape[1], output_bank.shape[1]), device=query.device
        )[None, :, None]
        bank_valid = bank_valid.expand(frames, -1, -1)

        native_memory = bank.reshape(frames * 2, bank_tokens, width)
        native_valid = bank_valid.reshape(frames * 2, bank_tokens)
        rows = query.permute(0, 2, 1, 3).reshape(frames * 2, ranks, width)
        normalized_query = self.evidence_query_norm(rows)
        native_memory = self.native_memory_norm(native_memory)

        def policy_read(
            memory: torch.Tensor, valid: torch.Tensor | None = None
        ) -> torch.Tensor:
            memory = memory[:, None].expand(-1, 2, -1, -1).reshape(
                frames * 2, -1, width
            )
            if valid is not None:
                valid = valid[:, None].expand(-1, 2, -1).reshape(frames * 2, -1)
            normalized_memory = self.policy_memory_norm(memory)
            attended, _ = self.policy_attention(
                normalized_query,
                normalized_memory,
                normalized_memory,
                key_padding_mask=None if valid is None else ~valid,
                need_weights=False,
            )
            return attended

        policy_readout = (
            policy_read(patches)
            + policy_read(language, language_valid)
            + policy_read(response)
        )
        native_read, _ = self.native_attention(
            normalized_query,
            native_memory,
            native_memory,
            key_padding_mask=~native_valid,
            need_weights=False,
        )
        return (policy_readout + native_read).reshape(
            frames, 2, ranks, width
        ).permute(0, 2, 1, 3)

    def _checkpointed_evidence_read(
        self,
        query: torch.Tensor,
        patches: torch.Tensor,
        language: torch.Tensor,
        language_valid: torch.Tensor,
        response: torch.Tensor,
        input_bank: torch.Tensor,
        output_bank: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._evidence_read(
                query,
                patches,
                language,
                language_valid,
                response,
                input_bank,
                output_bank,
            )
        return checkpoint(
            self._evidence_read,
            query,
            patches,
            language,
            language_valid,
            response,
            input_bank,
            output_bank,
            use_reentrant=False,
        )

    @staticmethod
    def _video_axes(
        frame: torch.Tensor,
        position: torch.Tensor,
        tokens: PolicyResponseEvidence,
        input_chunks: Sequence[torch.Tensor],
        output_chunks: Sequence[torch.Tensor],
    ) -> tuple[int, int, int]:
        """Validate one video and its complete, side-matched native bank."""
        if frame.ndim != 4 or frame.shape[2] != 2:
            raise ValueError("unified factor-side axes changed")
        frame_count, ranks, _, width = frame.shape
        if (
            position.shape != (frame_count,)
            or tokens.patches.shape[0] != frame_count
            or tokens.language.shape[0] != frame_count
            or tokens.language_valid.shape != tokens.language.shape[:2]
            or tokens.response.shape[0] != frame_count
            or not input_chunks
            or len(input_chunks) != len(output_chunks)
            or any(
                left.ndim != 3
                or right.ndim != 3
                or left.shape[0] != right.shape[0]
                or left.shape[-1] != width
                or right.shape[-1] != width
                or not left.shape[0]
                or not left.shape[1]
                or not right.shape[1]
                for left, right in zip(
                    input_chunks, output_chunks, strict=True
                )
            )
            or sum(chunk.shape[0] for chunk in input_chunks) != frame_count
        ):
            raise ValueError("unified evidence or bank axes changed")
        return frame_count, ranks, width

    def forward(
        self,
        frame_states: Sequence[torch.Tensor],
        frame_positions: Sequence[torch.Tensor],
        evidence: Sequence[PolicyResponseEvidence],
        input_bank_chunks: Sequence[Sequence[torch.Tensor]],
        output_bank_chunks: Sequence[Sequence[torch.Tensor]],
    ) -> tuple[torch.Tensor, ...]:
        frames = tuple(frame_states)
        positions = tuple(frame_positions)
        memories = tuple(evidence)
        input_banks = tuple(tuple(chunks) for chunks in input_bank_chunks)
        output_banks = tuple(tuple(chunks) for chunks in output_bank_chunks)
        if (
            not frames
            or len(frames) != len(positions)
            or len(frames) != len(memories)
            or len(frames) != len(input_banks)
            or len(frames) != len(output_banks)
        ):
            raise ValueError("unified factor video set changed")

        output = []
        for frame, position, tokens, input_chunks, output_chunks in zip(
            frames, positions, memories, input_banks, output_banks, strict=True
        ):
            frame_count, ranks, width = self._video_axes(
                frame, position, tokens, input_chunks, output_chunks
            )

            reads = []
            offset = 0
            for input_memory, output_memory in zip(
                input_chunks, output_chunks, strict=True
            ):
                count = input_memory.shape[0]
                local = frame[offset : offset + count]
                reads.append(
                    self._checkpointed_evidence_read(
                        local,
                        tokens.patches[offset : offset + count],
                        tokens.language[offset : offset + count],
                        tokens.language_valid[offset : offset + count],
                        tokens.response[offset : offset + count],
                        input_memory,
                        output_memory,
                    )
                )
                offset += count
            value = frame + torch.cat(tuple(reads), dim=0)

            temporal = value.permute(1, 2, 0, 3).reshape(
                ranks * 2, frame_count, width
            )
            normalized = self.temporal_norm(temporal)
            query_key = normalized + self._position(position.to(frame))[None]
            attended, _ = self.temporal_attention(
                query_key, query_key, normalized, need_weights=False
            )
            value = (temporal + attended).reshape(
                ranks, 2, frame_count, width
            ).permute(2, 0, 1, 3)

            factors = value.reshape(frame_count, ranks * 2, width)
            normalized = self.factor_norm(factors)
            attended, _ = self.factor_attention(
                normalized, normalized, normalized, need_weights=False
            )
            output.append(
                self.mlp(factors + attended).reshape(frame_count, ranks, 2, width)
            )
        return tuple(output)



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
            owners, expert_width=expert_width, width=width
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
            or response.shape[:2] != (video.frame_count, len(self.owners))
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
        self.response.family_embedding.weight.copy_(source.family_embedding.weight)
        self.response.layer_embedding.weight[:18].copy_(source.layer_embedding.weight)
        binding = stage0.encoder.binding
        self.response.owner_embedding.copy_(binding.owner_embedding)
        self.response.horizon_embedding.weight.copy_(binding.horizon_embedding)
        return {
            "reused": [
                "prefix_projections",
                "response_channel_projections",
                "owner_family_layer_horizon_embeddings",
            ],
        }
