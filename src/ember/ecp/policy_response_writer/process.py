"""Per-frame encoder for the native-temporal Policy-Response Writer.

The encoder preserves every frozen PI0.5 policy-response horizon and lets a
small set of target-rank states read it with repeatable transformer blocks.
Video-time modeling belongs to the downstream native-factor blocks, where it
can interact with the current video's real X/Y bank instead of passing through
an independent event bottleneck.
"""

from __future__ import annotations

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
class PolicyResponseFrameOutput:
    """One independently encoded video with real frames still explicit."""

    frame_tokens: torch.Tensor


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        memory = self.norm(
            torch.cat(
                (
                    patches + self.type_embedding.weight[0],
                    language + self.type_embedding.weight[1],
                ),
                dim=1,
            )
        )
        valid = torch.cat(
            (
                torch.ones(
                    patches.shape[:2], dtype=torch.bool, device=patches.device
                ),
                mask,
            ),
            dim=1,
        )
        if not torch.all(valid.any(1)):
            raise ValueError("policy-response prefix has an empty frame")
        return memory, valid


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
    """One copyable native-prefix, full-response, target-rank block."""

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
        memory = self.prefix_norm(prefix)
        attended, _ = self.prefix_attention(
            self.query_norm(rows),
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


class PolicyResponseFrameEncoder(torch.nn.Module):
    """Frozen PI0.5 response -> per-frame target-rank states."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
        heads: int = 4,
        frame_blocks: int = 2,
    ) -> None:
        super().__init__()
        if not owners or width % heads or frame_blocks <= 0:
            raise ValueError("frame policy-response topology changed")
        self.owners = tuple(owners)
        self.width = width
        self.prefix = PrefixTokenizer(prefix_width, width)
        self.response = ResponseTokenizer(
            owners, expert_width=expert_width, width=width
        )
        self.rank_embedding = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, width)
        )
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(owners), width))
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        family_order = tuple(TargetFamily)
        self.register_buffer(
            "family_ids",
            torch.tensor([family_order.index(owner.family) for owner in owners]),
            persistent=False,
        )
        self.frame_blocks = torch.nn.ModuleList(
            FramePolicyResponseBlock(width, heads) for _ in range(frame_blocks)
        )
        torch.nn.init.normal_(self.rank_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)

    def _seed(self) -> torch.Tensor:
        return (
            self.rank_embedding[None]
            + self.owner_embedding[:, None]
            + self.family_embedding(self.family_ids)[:, None]
        )

    def forward(
        self,
        video: FrozenPolicyResponseVideo,
        *,
        representation: str = "full",
    ) -> PolicyResponseFrameOutput:
        if representation != "full":
            raise ValueError("full policy-response is the only active representation")
        prefix, prefix_valid = self.prefix(video)
        response = self.response(video)
        frame = self._seed()[None].expand(video.frame_count, -1, -1, -1)
        for block in self.frame_blocks:
            frame = block(frame, prefix, prefix_valid, response)
        return PolicyResponseFrameOutput(frame_tokens=frame)

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
                "native_temporal_factor_blocks",
                "factor_side_signed_heads",
            ],
        }
