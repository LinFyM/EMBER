"""Axial policy-response video encoder.

The encoder has one job: turn the frozen PI0.5 response field of each ordered
video into content-carrying event tokens. Capacity scales by repeating the same
frame, temporal, and event blocks. Frame position is used only in attention
queries and keys, so a repeated static frame cannot manufacture an event value
from position alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import G1_RESIDUAL_RANK
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo

if TYPE_CHECKING:
    from ember.ecp.stage0 import ECPStage0Model


@dataclass(frozen=True)
class PolicyResponseProcessOutput:
    """One independently encoded video with every semantic axis explicit."""

    events: torch.Tensor
    frame_tokens: torch.Tensor
    frame_innovations: torch.Tensor
    owner_language: torch.Tensor


class GatedMLP(torch.nn.Module):
    """A standard pre-norm gated residual MLP."""

    def __init__(self, width: int, expansion: int = 4, *, bias: bool = True) -> None:
        super().__init__()
        hidden = width * expansion
        # Do not normalize the centered dynamic path: LayerNorm would amplify
        # harmless floating-point residue from a repeated static sequence.
        self.norm = torch.nn.LayerNorm(width) if bias else torch.nn.Identity()
        self.input = torch.nn.Linear(width, 2 * hidden, bias=bias)
        self.output = torch.nn.Linear(hidden, width, bias=bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = self.input(self.norm(value)).chunk(2, dim=-1)
        return value + self.output(left * F.gelu(gate))


class OwnerLanguageReader(torch.nn.Module):
    """Let structural target queries read the exact contextualized language."""

    def __init__(self, owners: int, width: int) -> None:
        super().__init__()
        self.queries = torch.nn.Parameter(torch.empty(owners, width))
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(2 * width, width)
        self.norm = torch.nn.LayerNorm(width)
        torch.nn.init.normal_(self.queries, std=width**-0.5)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or mask.shape != tokens.shape[:1] or not torch.any(mask):
            raise ValueError("policy-response language token contract changed")
        logits = torch.einsum(
            "jd,ld->jl", self.queries, self.key(tokens)
        ) / math.sqrt(tokens.shape[-1])
        logits = logits.masked_fill(~mask[None], torch.finfo(logits.dtype).min)
        attended = torch.einsum("jl,ld->jd", logits.softmax(-1), self.value(tokens))
        return self.norm(self.output(torch.cat((self.queries, attended), dim=-1)))


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.patch_projection(video.patch_states)
        language = self.language_projection(video.language_states)
        mask = video.language_mask
        if (
            patches.ndim != 3
            or language.ndim != 3
            or mask.shape != language.shape[:2]
            or patches.shape[0] != language.shape[0]
        ):
            raise ValueError("policy-response prefix topology changed")
        patches = patches + self.type_embedding.weight[0]
        language = language + self.type_embedding.weight[1]
        memory = self.norm(torch.cat((patches, language), dim=1))
        valid = torch.cat(
            (
                torch.ones(
                    patches.shape[:2], dtype=torch.bool, device=patches.device
                ),
                mask,
            ),
            dim=1,
        )
        weights = mask.to(language.dtype)
        language_summary = (language * weights[:, :, None]).sum(0) / weights.sum(
            0
        ).clamp_min(1)[:, None]
        return memory, valid, language_summary, mask.any(0)


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


class FramePolicyResponseBlock(torch.nn.Module):
    """One copyable prefix read, full-response read, axial attention, and MLP."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.prefix_norm = torch.nn.LayerNorm(width)
        self.prefix_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.response_norm = torch.nn.LayerNorm(width)
        self.response_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.factor_norm = torch.nn.LayerNorm(width)
        self.factor_attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def forward(
        self,
        value: torch.Tensor,
        prefix: torch.Tensor,
        prefix_valid: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        frames, targets, ranks, width = value.shape
        if (
            prefix.shape[0] != frames
            or prefix_valid.shape != prefix.shape[:2]
            or response.shape[:2] != (frames, targets)
            or response.shape[-1] != width
        ):
            raise ValueError("frame policy-response block axes changed")
        rows = value.reshape(frames, targets * ranks, width)
        query = self.query_norm(rows)
        memory = self.prefix_norm(prefix)
        attended, _ = self.prefix_attention(
            query,
            memory,
            memory,
            key_padding_mask=~prefix_valid,
            need_weights=False,
        )
        rows = rows + attended

        query = self.query_norm(rows).reshape(frames * targets, ranks, width)
        memory = self.response_norm(response).reshape(
            frames * targets, response.shape[2], width
        )
        attended, _ = self.response_attention(
            query, memory, memory, need_weights=False
        )
        rows = rows + attended.reshape(frames, targets * ranks, width)

        normalized = self.factor_norm(rows)
        attended, _ = self.factor_attention(
            normalized, normalized, normalized, need_weights=False
        )
        return self.mlp(rows + attended).reshape(frames, targets, ranks, width)


class TemporalPolicyResponseBlock(torch.nn.Module):
    """Self-attend within each target-rank sequence; position enters Q/K only."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.position = torch.nn.Linear(2, width, bias=False)
        self.mlp = GatedMLP(width)

    def forward(self, value: torch.Tensor, frame_positions: torch.Tensor) -> torch.Tensor:
        frames, targets, ranks, width = value.shape
        if frame_positions.shape != (frames,):
            raise ValueError("temporal frame positions changed")
        rows = value.permute(1, 2, 0, 3).reshape(targets * ranks, frames, width)
        normalized = self.norm(rows)
        positions = frame_positions.to(value).clamp(0.0, 1.0)
        position = self.position(
            torch.stack((positions, positions.square()), dim=-1)
        )
        query_key = normalized + position[None]
        attended, _ = self.attention(
            query_key, query_key, normalized, need_weights=False
        )
        rows = self.mlp(rows + attended)
        return rows.reshape(targets, ranks, frames, width).permute(2, 0, 1, 3)


class OrderedEventBlock(torch.nn.Module):
    """One content-preserving event-axis attention and bias-free MLP."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm = torch.nn.Identity()
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.position = torch.nn.Linear(2, width, bias=False)
        self.mlp = GatedMLP(width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, events, _ = value.shape
        positions = torch.linspace(
            0.0, 1.0, events, device=value.device, dtype=value.dtype
        )
        position = self.position(
            torch.stack((positions, positions.square()), dim=-1)
        )
        normalized = self.norm(value)
        query_key = normalized + position[None]
        attended, _ = self.attention(
            query_key, query_key, normalized, need_weights=False
        )
        return self.mlp(value + attended)


class OrderedEventReadout(torch.nn.Module):
    """Compress centered frame content with simple first/last anchors."""

    def __init__(
        self, *, width: int, event_slots: int, heads: int, block_depth: int
    ) -> None:
        super().__init__()
        if event_slots < 2 or block_depth <= 0:
            raise ValueError("ordered event topology changed")
        self.event_slots = event_slots
        self.interior_queries = torch.nn.Parameter(
            torch.empty(max(event_slots - 2, 0), width)
        )
        self.query_context = torch.nn.Linear(width, width, bias=False)
        self.memory_norm = torch.nn.Identity()
        self.position = torch.nn.Linear(2, width, bias=False)
        self.read = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.blocks = torch.nn.ModuleList(
            OrderedEventBlock(width, heads) for _ in range(block_depth)
        )
        torch.nn.init.normal_(self.interior_queries, std=width**-0.5)

    def forward(
        self,
        innovations: torch.Tensor,
        frame_positions: torch.Tensor,
        query_context: torch.Tensor,
    ) -> torch.Tensor:
        frames, targets, ranks, width = innovations.shape
        if (
            frame_positions.shape != (frames,)
            or query_context.shape != (targets, ranks, width)
        ):
            raise ValueError("ordered event inputs changed")
        rows = innovations.permute(1, 2, 0, 3).reshape(
            targets * ranks, frames, width
        )
        normalized = self.memory_norm(rows)
        positions = frame_positions.to(innovations).clamp(0.0, 1.0)
        position = self.position(
            torch.stack((positions, positions.square()), dim=-1)
        )
        pieces = [rows[:, :1]]
        if self.interior_queries.shape[0]:
            context = self.query_context(query_context).reshape(
                targets * ranks, 1, width
            )
            query = self.interior_queries[None] + context
            interior, _ = self.read(
                query,
                normalized + position[None],
                normalized,
                need_weights=False,
            )
            pieces.append(interior)
        pieces.append(rows[:, -1:])
        events = torch.cat(tuple(pieces), dim=1)
        for block in self.blocks:
            events = block(events)
        return events.reshape(targets, ranks, self.event_slots, width).permute(
            2, 0, 1, 3
        )


class PolicyResponseProcessEncoder(torch.nn.Module):
    """Frozen response -> axial frame tokens -> ordered dynamic event tokens."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
        event_slots: int = 8,
        heads: int = 4,
        frame_blocks: int = 2,
        temporal_blocks: int = 2,
        event_blocks: int = 1,
    ) -> None:
        super().__init__()
        if (
            not owners
            or width % heads
            or min(frame_blocks, temporal_blocks, event_blocks) <= 0
        ):
            raise ValueError("axial policy-response topology changed")
        self.owners = tuple(owners)
        self.width = width
        self.event_slots = event_slots
        self.prefix = PrefixTokenizer(prefix_width, width)
        self.language_reader = OwnerLanguageReader(len(owners), width)
        self.response = ResponseTokenizer(
            owners, expert_width=expert_width, width=width
        )
        self.rank_embedding = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, width)
        )
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(owners), width))
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        self.seed = torch.nn.Sequential(
            torch.nn.Linear(4 * width, width),
            torch.nn.LayerNorm(width),
        )
        family_order = tuple(TargetFamily)
        self.register_buffer(
            "family_ids",
            torch.tensor([family_order.index(owner.family) for owner in owners]),
            persistent=False,
        )
        self.frame_blocks = torch.nn.ModuleList(
            FramePolicyResponseBlock(width, heads) for _ in range(frame_blocks)
        )
        self.temporal_blocks = torch.nn.ModuleList(
            TemporalPolicyResponseBlock(width, heads)
            for _ in range(temporal_blocks)
        )
        self.events = OrderedEventReadout(
            width=width,
            event_slots=event_slots,
            heads=heads,
            block_depth=event_blocks,
        )
        torch.nn.init.normal_(self.rank_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)

    def _query_seed(self, owner_language: torch.Tensor) -> torch.Tensor:
        targets = len(self.owners)
        ranks = G1_RESIDUAL_RANK
        if owner_language.shape != (targets, self.width):
            raise ValueError("policy-response owner language changed")
        rank = self.rank_embedding[None].expand(targets, -1, -1)
        owner = self.owner_embedding[:, None].expand(-1, ranks, -1)
        family = self.family_embedding(self.family_ids)[:, None].expand(
            -1, ranks, -1
        )
        language = owner_language[:, None].expand(-1, ranks, -1)
        return self.seed(torch.cat((rank, owner, family, language), dim=-1))

    def forward(
        self,
        video: FrozenPolicyResponseVideo,
        *,
        representation: str = "full",
    ) -> PolicyResponseProcessOutput:
        if representation != "full":
            raise ValueError("full policy-response is the only active representation")
        prefix, prefix_valid, language, language_mask = self.prefix(video)
        owner_language = self.language_reader(language, language_mask)
        seed = self._query_seed(owner_language)
        response = self.response(video)
        frame = seed[None].expand(video.frame_count, -1, -1, -1)
        for block in self.frame_blocks:
            frame = block(frame, prefix, prefix_valid, response)
        for block in self.temporal_blocks:
            frame = block(frame, video.frame_positions)
        innovations = frame - frame.mean(0, keepdim=True)
        events = self.events(innovations, video.frame_positions, seed)
        return PolicyResponseProcessOutput(
            events=events,
            frame_tokens=frame,
            frame_innovations=innovations,
            owner_language=owner_language,
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
        self.family_embedding.weight.copy_(source.family_embedding.weight)
        self.response.layer_embedding.weight[:18].copy_(source.layer_embedding.weight)
        binding = stage0.encoder.binding
        self.response.owner_embedding.copy_(binding.owner_embedding)
        self.response.horizon_embedding.weight.copy_(binding.horizon_embedding)
        self.owner_embedding.copy_(binding.owner_embedding)
        first = self.frame_blocks[0].response_attention
        width = self.width
        first.in_proj_weight[:width].copy_(binding.event_query.weight)
        first.in_proj_weight[width : 2 * width].copy_(binding.policy_key.weight)
        first.in_proj_weight[2 * width :].copy_(binding.policy_value.weight)
        first.in_proj_bias.zero_()
        first.out_proj.weight.copy_(torch.eye(width, device=first.out_proj.weight.device))
        first.out_proj.bias.zero_()
        return {
            "kind": "g2_native_projection_initialization",
            "reused": [
                "prefix_projections",
                "response_channel_projections",
                "owner_family_layer_horizon_embeddings",
                "first_full_horizon_response_attention",
            ],
            "fresh": [
                "frame_repeat_blocks",
                "temporal_repeat_blocks",
                "content_event_readout",
                "native_factor_composer",
            ],
        }
