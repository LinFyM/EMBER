"""Repeatable policy-response frame and ordered-event blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo

if TYPE_CHECKING:
    from ember.ecp.stage0 import ECPStage0Model


@dataclass(frozen=True)
class PolicyResponseProcessOutput:
    """Per-video ordered events and the sole common/innovation decomposition."""

    events: torch.Tensor
    common: torch.Tensor
    innovations: torch.Tensor
    assignment: torch.Tensor
    occupancy: torch.Tensor
    presence: torch.Tensor
    frame_innovation: torch.Tensor
    owner_language: torch.Tensor


class GatedMLP(torch.nn.Module):
    def __init__(self, width: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = torch.nn.LayerNorm(width)
        self.input = torch.nn.Linear(width, 2 * hidden)
        self.output = torch.nn.Linear(hidden, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, gate = self.input(self.norm(value)).chunk(2, dim=-1)
        return value + self.output(left * F.gelu(gate))


class OwnerLanguageReader(torch.nn.Module):
    """Standard owner queries over exact contextualized language tokens."""

    def __init__(self, owners: int, width: int) -> None:
        super().__init__()
        self.queries = torch.nn.Parameter(torch.empty(owners, width))
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(2 * width, width),
            torch.nn.GELU(),
            torch.nn.LayerNorm(width),
        )
        torch.nn.init.normal_(self.queries, std=width**-0.5)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or mask.shape != tokens.shape[:1] or not torch.any(mask):
            raise ValueError("policy-response language token contract changed")
        key = self.key(tokens)
        value = self.value(tokens)
        logits = torch.einsum("jd,ld->jl", self.queries, key) / math.sqrt(key.shape[-1])
        logits = logits.masked_fill(~mask[None], torch.finfo(logits.dtype).min)
        attended = torch.einsum("jl,ld->jd", logits.softmax(-1), value)
        return self.output(torch.cat((self.queries, attended), dim=-1))


class TaskGroundedRelationReader(torch.nn.Module):
    """Language-ground patches into adjacent, local, initial, and goal relations."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.relation = torch.nn.Linear(3 * width, width, bias=False)
        self.relation_embedding = torch.nn.Parameter(torch.empty(4, width))
        self.token_score = torch.nn.Linear(width, 1, bias=False)
        self.confidence = torch.nn.Linear(width, 1)
        torch.nn.init.normal_(self.relation_embedding, std=width**-0.5)

    def forward(
        self,
        patches: torch.Tensor,
        language: torch.Tensor,
        language_mask: torch.Tensor,
        *,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames, _, width = patches.shape
        if (
            language.ndim != 2
            or language.shape[-1] != width
            or language_mask.shape != language.shape[:1]
        ):
            raise ValueError("policy-response grounded relation inputs changed")
        query = self.query(language)
        key = self.key(patches)
        value = self.value(patches)
        scores = torch.einsum("nd,tpd->tnp", query, key) / math.sqrt(width)
        grounded = torch.einsum("tnp,tpd->tnd", scores.softmax(-1), value)
        grounded = grounded.masked_fill(~language_mask[None, :, None], 0.0)
        time = torch.arange(frames, device=patches.device)
        previous = grounded.index_select(0, (time - 1).clamp_min(0))
        previous2 = grounded.index_select(0, (time - 2).clamp_min(0))
        initial = grounded[:1].expand(frames, -1, -1)
        relation_values = [
            grounded - previous,
            grounded - 0.5 * (previous + previous2),
            grounded - initial,
        ]
        if not causal:
            relation_values.append(grounded[-1:] - grounded)
        relations = torch.stack(relation_values, dim=1)
        count = int(relations.shape[1])
        current = grounded[:, None].expand(-1, count, -1, -1)
        lang = language[None, None].expand(frames, count, -1, -1)
        features = self.relation(
            torch.cat((relations, current, lang), dim=-1)
        ) + self.relation_embedding[None, :count, None]
        token_logits = self.token_score(torch.tanh(features)).squeeze(-1)
        token_logits = token_logits.masked_fill(
            ~language_mask[None, None], torch.finfo(token_logits.dtype).min
        )
        candidates = torch.einsum(
            "tmn,tmnd->tmd", token_logits.softmax(-1), features
        )
        return candidates, self.confidence(torch.tanh(candidates)).squeeze(-1)


class ResponseTokenizer(torch.nn.Module):
    """Keep state, residual, noise, velocity, even, and odd as distinct tokens."""

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
        self.coarse_embedding = torch.nn.Embedding(2, width)
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

    def _owner_bias(self) -> torch.Tensor:
        layers = self.state_layers
        return (
            self.owner_embedding
            + self.family_embedding(self.family_ids)
            + self.layer_embedding(layers)
        )

    def forward(
        self, video: FrozenPolicyResponseVideo, *, representation: str
    ) -> torch.Tensor:
        states = video.layer_states
        if states.ndim != 5 or states.shape[1:4] != (2, 19, ACTION_HORIZON):
            raise ValueError("policy-response raw layer topology changed")
        if representation == "coarse":
            final = self.state_projection(states[:, :, -1].mean(2))
            even, odd = self._even_odd(final)
            tokens = torch.stack((even, odd), dim=1)[:, None]
            tokens = tokens.expand(-1, len(self.owners), -1, -1)
            tokens = tokens + self._owner_bias()[None, :, None]
            tokens = tokens + self.coarse_embedding.weight[None, None]
            return self.norm(tokens)
        if representation != "full":
            raise ValueError("unknown policy-response representation arm")

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
        # Each entry is [frame, owner, horizon, width].
        tokens = torch.stack(channels, dim=3)
        tokens = tokens + self._owner_bias()[None, :, None, None]
        tokens = tokens + self.horizon_embedding.weight[None, None, :, None]
        tokens = tokens + self.channel_embedding.weight[None, None, None]
        return self.norm(tokens.flatten(2, 3))


class FramePolicyResponseBlock(torch.nn.Module):
    """One copyable response cross-attention, owner attention, and gated MLP."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.response_norm = torch.nn.LayerNorm(width)
        self.response_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.owner_norm = torch.nn.LayerNorm(width)
        self.owner_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def forward(
        self, relations: torch.Tensor, response_tokens: torch.Tensor
    ) -> torch.Tensor:
        frames, relation_count, owners, width = relations.shape
        if response_tokens.shape[:2] != (frames, owners):
            raise ValueError("frame-response owner axes changed")
        query = relations.permute(0, 2, 1, 3).reshape(
            frames * owners, relation_count, width
        )
        memory = response_tokens.reshape(frames * owners, -1, width)
        attended, _ = self.response_attention(
            self.query_norm(query),
            self.response_norm(memory),
            self.response_norm(memory),
            need_weights=False,
        )
        value = (query + attended).reshape(
            frames, owners, relation_count, width
        ).permute(0, 2, 1, 3)
        owner_rows = value.reshape(frames * relation_count, owners, width)
        normalized = self.owner_norm(owner_rows)
        attended, _ = self.owner_attention(
            normalized, normalized, normalized, need_weights=False
        )
        return self.mlp(
            (owner_rows + attended).reshape(frames, relation_count, owners, width)
        )


class EventBlock(torch.nn.Module):
    """One copyable ordered-event attention, owner attention, and gated MLP."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.event_norm = torch.nn.LayerNorm(width)
        self.event_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.owner_norm = torch.nn.LayerNorm(width)
        self.owner_attention = torch.nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.mlp = GatedMLP(width)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        event_count, owners, width = events.shape
        event_rows = events.permute(1, 0, 2)
        normalized = self.event_norm(event_rows)
        attended, _ = self.event_attention(
            normalized, normalized, normalized, need_weights=False
        )
        value = (event_rows + attended).permute(1, 0, 2)
        normalized = self.owner_norm(value)
        attended, _ = self.owner_attention(
            normalized, normalized, normalized, need_weights=False
        )
        return self.mlp(value + attended)


class BoundaryAnchoredEventEncoder(torch.nn.Module):
    """Monotone stay/advance segmentation with hard first and final anchors."""

    def __init__(
        self,
        *,
        width: int,
        event_slots: int,
        heads: int,
        block_depth: int,
        presence_threshold_fraction: float = 0.08,
    ) -> None:
        super().__init__()
        self.event_slots = event_slots
        self.presence_threshold_fraction = presence_threshold_fraction
        self.slot_queries = torch.nn.Parameter(torch.empty(event_slots, width))
        self.owner_pool = torch.nn.Linear(width, 1, bias=False)
        self.transition = torch.nn.Linear(width, event_slots)
        self.duration_bias = torch.nn.Parameter(torch.zeros(event_slots))
        self.event_positions = torch.nn.Parameter(torch.empty(event_slots, width))
        self.blocks = torch.nn.ModuleList(
            EventBlock(width, heads) for _ in range(block_depth)
        )
        torch.nn.init.normal_(self.slot_queries, std=width**-0.5)
        torch.nn.init.normal_(self.event_positions, std=width**-0.5)

    def _posterior(self, emission: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        frames, slots = emission.shape
        if slots != self.event_slots or frames < slots:
            raise ValueError("ordered events require at least one frame per slot")
        negative = torch.finfo(emission.dtype).min
        advance = F.logsigmoid(boundary + self.duration_bias)
        stay = F.logsigmoid(-(boundary + self.duration_bias))
        alpha = emission.new_full((frames, slots), negative)
        alpha[0, 0] = emission[0, 0]
        for time in range(1, frames):
            from_stay = alpha[time - 1] + stay[time - 1]
            from_advance = emission.new_full((slots,), negative)
            from_advance[1:] = alpha[time - 1, :-1] + advance[time - 1, :-1]
            alpha[time] = emission[time] + torch.logaddexp(
                from_stay, from_advance
            )
        beta = emission.new_full((frames, slots), negative)
        beta[-1, -1] = 0.0
        for time in range(frames - 2, -1, -1):
            remain = stay[time] + emission[time + 1] + beta[time + 1]
            move = emission.new_full((slots,), negative)
            move[:-1] = (
                advance[time, :-1]
                + emission[time + 1, 1:]
                + beta[time + 1, 1:]
            )
            beta[time] = torch.logaddexp(remain, move)
        return (alpha + beta).softmax(-1)

    def forward(
        self, relations: torch.Tensor, confidence: torch.Tensor
    ) -> PolicyResponseProcessOutput:
        frames, relation_count, owners, width = relations.shape
        owner_logits = self.owner_pool(torch.tanh(relations)).squeeze(-1)
        owner_weights = owner_logits.softmax(-1)
        tokens = torch.einsum("tmj,tmjd->tmd", owner_weights, relations)
        candidate_logits = torch.einsum(
            "tmd,ed->tem", tokens, self.slot_queries
        ) / math.sqrt(width)
        candidate_logits = candidate_logits + confidence[:, None]
        emission = torch.logsumexp(candidate_logits, dim=-1) - math.log(
            relation_count
        )
        frame_summary = torch.einsum(
            "tm,tmd->td", confidence.softmax(-1), tokens
        )
        posterior = self._posterior(emission, self.transition(frame_summary))
        relation_probability = candidate_logits.softmax(-1)
        assignment = posterior.transpose(0, 1)[:, :, None] * relation_probability.permute(
            1, 0, 2
        )
        occupancy = assignment.sum((1, 2))
        events = torch.einsum(
            "etm,tmjd->ejd", assignment, relations
        ) / occupancy.clamp_min(1e-6)[:, None, None]
        events = events + self.event_positions[:, None]
        for block in self.blocks:
            events = block(events)
        weights = occupancy / occupancy.sum().clamp_min(1e-6)
        common = torch.einsum("e,ejd->jd", weights, events)
        innovations = events - common[None]
        frame_innovation = torch.einsum(
            "etm,ejd->tjd", assignment, innovations
        )
        presence = -torch.expm1(
            -(occupancy / max(frames, 1)) / self.presence_threshold_fraction
        )
        return PolicyResponseProcessOutput(
            events=events,
            common=common,
            innovations=innovations,
            assignment=assignment,
            occupancy=occupancy,
            presence=presence,
            frame_innovation=frame_innovation,
            owner_language=events.new_empty(0),
        )


class PolicyResponseProcessEncoder(torch.nn.Module):
    """Frozen response -> grounded frames -> boundary-anchored ordered events."""

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
        event_blocks: int = 2,
        teacher_width: int = 32,
        teacher_seed: int = 20260902,
    ) -> None:
        super().__init__()
        if width % heads or min(frame_blocks, event_blocks) <= 0:
            raise ValueError("policy-response block topology changed")
        self.owners = tuple(owners)
        self.width = width
        self.event_slots = event_slots
        self.patch_projection = torch.nn.Linear(prefix_width, width, bias=False)
        self.language_projection = torch.nn.Linear(prefix_width, width, bias=False)
        self.language_reader = OwnerLanguageReader(len(owners), width)
        self.relations = TaskGroundedRelationReader(width)
        self.response = ResponseTokenizer(
            owners, expert_width=expert_width, width=width
        )
        self.frame_blocks = torch.nn.ModuleList(
            FramePolicyResponseBlock(width, heads) for _ in range(frame_blocks)
        )
        self.events = BoundaryAnchoredEventEncoder(
            width=width,
            event_slots=event_slots,
            heads=heads,
            block_depth=event_blocks,
        )
        self.prediction_probe = torch.nn.Embedding(2, width)
        self.prediction_horizon = torch.nn.Embedding(ACTION_HORIZON, width)
        self.prediction_head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, teacher_width),
        )
        generator = torch.Generator(device="cpu").manual_seed(teacher_seed)
        for name, input_width in (
            ("teacher_state", expert_width),
            ("teacher_residual", expert_width),
            ("teacher_velocity", 32),
            ("teacher_noise", 32),
        ):
            self.register_buffer(
                name,
                torch.randn(input_width, teacher_width, generator=generator)
                / math.sqrt(input_width),
                persistent=True,
            )

    def _compact_prefix(
        self, video: FrozenPolicyResponseVideo
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self.patch_projection(video.patch_states)
        language = self.language_projection(video.language_states)
        mask = video.language_mask
        if mask.ndim != 2 or mask.shape != language.shape[:2]:
            raise ValueError("policy-response language mask changed")
        weights = mask.to(language.dtype)
        language = (language * weights[:, :, None]).sum(0) / weights.sum(
            0
        ).clamp_min(1)[:, None]
        language_mask = mask.any(0)
        return patches, language, language_mask

    def forward(
        self,
        video: FrozenPolicyResponseVideo,
        *,
        representation: str = "full",
        causal: bool = False,
    ) -> PolicyResponseProcessOutput:
        patches, language, language_mask = self._compact_prefix(video)
        owner_language = self.language_reader(language, language_mask)
        relations, confidence = self.relations(
            patches, language, language_mask, causal=causal
        )
        relations = relations[:, :, None] + owner_language[None, None]
        response = self.response(video, representation=representation)
        for block in self.frame_blocks:
            relations = block(relations, response)
        result = self.events(relations, confidence)
        return PolicyResponseProcessOutput(
            events=result.events,
            common=result.common,
            innovations=result.innovations,
            assignment=result.assignment,
            occupancy=result.occupancy,
            presence=result.presence,
            frame_innovation=result.frame_innovation,
            owner_language=owner_language,
        )

    def fixed_teacher_response(
        self, video: FrozenPolicyResponseVideo
    ) -> torch.Tensor:
        """Fixed, non-trainable response target retaining owner/horizon/probe."""

        states = video.layer_states.float()
        residuals = states[:, :, 1:] - states[:, :, :-1]
        state = states.index_select(2, self.response.state_layers)
        residual = residuals.index_select(2, self.response.residual_layers)
        frames = states.shape[0]
        velocity = video.flow_velocity.float()[:, :, None].expand(
            -1, -1, len(self.owners), -1, -1
        )
        noise = video.suffix_noise.float()[None, :, None].expand(
            frames, -1, len(self.owners), -1, -1
        )
        return 0.5 * (
            state @ self.teacher_state
            + residual @ self.teacher_residual
            + velocity @ self.teacher_velocity
            + noise @ self.teacher_noise
        )

    def predict_future_delta(self, process: PolicyResponseProcessOutput) -> torch.Tensor:
        state = process.common + process.frame_innovation[-1]
        query = (
            state[:, None, None]
            + self.prediction_probe.weight[None, :, None]
            + self.prediction_horizon.weight[None, None]
        )
        return self.prediction_head(query).permute(1, 0, 2, 3)

    def causal_prediction_loss(
        self,
        video: FrozenPolicyResponseVideo,
        *,
        cutoffs: Sequence[int],
        future_offset: int = 1,
        representation: str = "full",
    ) -> torch.Tensor:
        """Predict frozen future-response changes from strictly sliced prefixes."""

        teacher = self.fixed_teacher_response(video).detach()
        losses = []
        for stop in map(int, cutoffs):
            current = stop - 1
            future = current + future_offset
            if stop < self.event_slots or future >= video.frame_count:
                raise ValueError("causal policy-response cutoff changed")
            prefix = video.frame_slice(stop)
            process = self(prefix, representation=representation, causal=True)
            prediction = self.predict_future_delta(process)
            target = teacher[future] - teacher[current]
            losses.append(
                F.smooth_l1_loss(
                    prediction.float(), target.float(), beta=1.0, reduction="mean"
                )
            )
        if not losses:
            raise ValueError("causal policy-response loss has no prefixes")
        return torch.stack(losses).mean()

    @torch.no_grad()
    def initialize_from_stage0(self, stage0: "ECPStage0Model") -> dict[str, object]:
        """Reuse the G2-proven projections and event initialization where shapes agree."""

        observer = stage0.encoder.observer
        source = observer.projector
        self.patch_projection.weight.copy_(observer.patch_projection.weight)
        self.language_projection.weight.copy_(observer.language_projection.weight)
        self.response.state_projection.weight.copy_(source.state_projection.weight)
        self.response.residual_projection.weight.copy_(source.delta_projection.weight)
        self.response.noise_projection.weight.copy_(source.noise_projection.weight)
        self.response.velocity_projection.weight.copy_(source.velocity_projection.weight)
        self.response.family_embedding.weight.copy_(source.family_embedding.weight)
        self.response.layer_embedding.weight[:18].copy_(source.layer_embedding.weight)
        self.relations.load_state_dict(stage0.encoder.matcher.state_dict())
        old_events = stage0.encoder.segmenter
        self.events.slot_queries.copy_(old_events.slot_queries)
        self.events.owner_pool.load_state_dict(old_events.owner_pool.state_dict())
        self.events.transition.load_state_dict(old_events.transition.state_dict())
        self.events.duration_bias.copy_(old_events.duration_bias)
        binding = stage0.encoder.binding
        self.response.owner_embedding.copy_(binding.owner_embedding)
        self.response.horizon_embedding.weight.copy_(binding.horizon_embedding)
        first = self.frame_blocks[0].response_attention
        width = self.width
        first.in_proj_weight[:width].copy_(binding.event_query.weight)
        first.in_proj_weight[width : 2 * width].copy_(binding.policy_key.weight)
        first.in_proj_weight[2 * width :].copy_(binding.policy_value.weight)
        first.in_proj_bias.zero_()
        first.out_proj.weight.copy_(torch.eye(width, device=first.out_proj.weight.device))
        first.out_proj.bias.zero_()
        return {
            "kind": "g2_stage0_component_initialization",
            "reused": [
                "prefix_projections",
                "separate_response_projections",
                "family_layer_horizon_embeddings",
                "task_grounded_relations",
                "event_emission_and_duration",
                "first_response_attention_qkv",
            ],
            "fresh": ["repeat_blocks", "causal_predictor", "composer"],
        }
