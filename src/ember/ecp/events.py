"""Task-grounded events and Event-Conditioned Horizon Binding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EventBindingOutput:
    candidates: torch.Tensor
    confidence: torch.Tensor
    bound_evidence: torch.Tensor


@dataclass(frozen=True)
class EventProgramOutput:
    process: torch.Tensor
    presence: torch.Tensor
    uncertainty: torch.Tensor
    assignment: torch.Tensor
    state_posterior: torch.Tensor


class TaskGroundedTransitionMatcher(torch.nn.Module):
    """Form four local transition hypotheses after language-grounding patches."""

    def __init__(self, *, width: int, candidates: int = 4) -> None:
        super().__init__()
        if candidates != 4:
            raise ValueError("ECP Stage 0 defines four complementary transition views")
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
        patch_states: torch.Tensor,
        language_queries: torch.Tensor,
        frame_mask: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, _, width = patch_states.shape
        query = self.query(language_queries)
        key = self.key(patch_states)
        value = self.value(patch_states)
        scores = torch.einsum("bnd,btpd->btnp", query, key) / math.sqrt(width)
        grounded = torch.einsum("btnp,btpd->btnd", scores.softmax(-1), value)
        grounded = grounded.masked_fill(~language_mask[:, None, :, None], 0.0)

        previous = torch.cat((grounded[:, :1], grounded[:, :-1]), dim=1)
        previous2 = torch.cat((grounded[:, :1], grounded[:, :1], grounded[:, :-2]), dim=1)
        local_context = 0.5 * (previous + previous2)
        initial = grounded[:, :1].expand(-1, frames, -1, -1)
        lengths = frame_mask.sum(1).clamp_min(1) - 1
        final = grounded[torch.arange(batch, device=grounded.device), lengths]
        final = final[:, None].expand(-1, frames, -1, -1)

        relations = torch.stack(
            (
                grounded - previous,
                grounded - local_context,
                grounded - initial,
                final - grounded,
            ),
            dim=2,
        )
        current = grounded[:, :, None].expand(-1, -1, 4, -1, -1)
        language = language_queries[:, None, None].expand(
            -1, frames, 4, -1, -1
        )
        token_features = self.relation(
            torch.cat((relations, current, language), dim=-1)
        ) + self.relation_embedding[None, None, :, None]
        token_logits = self.token_score(torch.tanh(token_features)).squeeze(-1)
        token_logits = token_logits.masked_fill(
            ~language_mask[:, None, None], torch.finfo(token_logits.dtype).min
        )
        candidates = torch.einsum(
            "btmn,btmnd->btmd", token_logits.softmax(-1), token_features
        )
        confidence = self.confidence(torch.tanh(candidates)).squeeze(-1)
        candidates = candidates.masked_fill(~frame_mask[:, :, None, None], 0.0)
        confidence = confidence.masked_fill(~frame_mask[:, :, None], -20.0)
        return candidates, confidence


class EventConditionedHorizonBinding(torch.nn.Module):
    """Bind each event candidate bidirectionally to every owner and horizon."""

    def __init__(
        self,
        *,
        width: int,
        owners: int = 38,
        horizons: int = 50,
    ) -> None:
        super().__init__()
        self.owners = owners
        self.horizons = horizons
        self.event_query = torch.nn.Linear(width, width, bias=False)
        self.event_value = torch.nn.Linear(width, width, bias=False)
        self.policy_key = torch.nn.Linear(width, width, bias=False)
        self.policy_value = torch.nn.Linear(width, width, bias=False)
        self.owner_embedding = torch.nn.Parameter(torch.empty(owners, width))
        self.horizon_embedding = torch.nn.Parameter(torch.empty(horizons, width))
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(3 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        torch.nn.init.normal_(self.owner_embedding, std=width**-0.5)
        torch.nn.init.normal_(self.horizon_embedding, std=width**-0.5)

    def forward(
        self,
        candidates: torch.Tensor,
        confidence: torch.Tensor,
        owner_lattice: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        policy = (
            owner_lattice
            + self.owner_embedding[None, None, :, None]
            + self.horizon_embedding[None, None, None]
        )
        event_query = self.event_query(candidates)
        event_value = self.event_value(candidates)
        policy_key = self.policy_key(policy)
        policy_value = self.policy_value(policy)
        scores = torch.einsum(
            "btmd,btjhd->btmjh", event_query, policy_key
        ) / math.sqrt(candidates.shape[-1])
        scores = scores + confidence[:, :, :, None, None]

        event_to_horizon = scores.softmax(-1)
        direct = torch.einsum(
            "btmjh,btjhd->btmjd", event_to_horizon, policy_value
        )
        horizon_to_event = scores.softmax(2)
        reverse_horizon = torch.einsum(
            "btmjh,btmd->btjhd", horizon_to_event, event_value
        )
        reverse = torch.einsum(
            "btmjh,btjhd->btmjd", event_to_horizon, reverse_horizon
        )
        event = event_value[:, :, :, None].expand(-1, -1, -1, self.owners, -1)
        bound = self.fusion(torch.cat((event, direct, reverse), dim=-1))
        return bound.masked_fill(~frame_mask[:, :, None, None, None], 0.0)


class OrderedEventSegmenter(torch.nn.Module):
    """Learn duration-aware monotone event slots with dynamic occupancy."""

    def __init__(
        self,
        *,
        width: int,
        event_slots: int = 8,
        candidates: int = 4,
    ) -> None:
        super().__init__()
        self.event_slots = event_slots
        self.candidates = candidates
        self.slot_queries = torch.nn.Parameter(torch.empty(event_slots, width))
        self.owner_pool = torch.nn.Linear(width, 1, bias=False)
        self.transition = torch.nn.Linear(width, event_slots)
        self.duration_bias = torch.nn.Parameter(torch.zeros(event_slots))
        self.start_logits = torch.nn.Parameter(torch.zeros(event_slots))
        self.forward_logits = torch.nn.Parameter(
            torch.zeros(event_slots, event_slots)
        )
        self.minimum_duration = torch.nn.Parameter(torch.zeros(event_slots))
        torch.nn.init.normal_(self.slot_queries, std=width**-0.5)

    def _posteriors(
        self,
        emission: torch.Tensor,
        boundary: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, slots = emission.shape
        log_advance = F.logsigmoid(boundary + self.duration_bias)
        log_stay = F.logsigmoid(-(boundary + self.duration_bias))
        negative = torch.finfo(emission.dtype).min
        source = torch.arange(slots, device=emission.device)[:, None]
        destination = torch.arange(slots, device=emission.device)[None]
        forward_mask = destination > source
        forward_choice = F.log_softmax(
            self.forward_logits[:-1].masked_fill(
                ~forward_mask[:-1], negative
            ),
            dim=-1,
        )
        forward_choice = torch.cat(
            (forward_choice, emission.new_full((1, slots), negative)), dim=0
        )
        transitions = (
            log_advance[..., :, None] + forward_choice[None, None]
        ).masked_fill(~forward_mask[None, None], negative)
        transitions = torch.where(
            torch.eye(slots, dtype=torch.bool, device=emission.device)[
                None, None
            ],
            log_stay[..., :, None],
            transitions,
        )

        alpha = emission.new_full((batch, frames, slots), negative)
        alpha[:, 0] = emission[:, 0] + self.start_logits.log_softmax(0)
        for time in range(1, frames):
            update = torch.logsumexp(
                alpha[:, time - 1, :, None] + transitions[:, time - 1],
                dim=1,
            ) + emission[:, time]
            alpha[:, time] = torch.where(
                frame_mask[:, time, None], update, alpha[:, time - 1]
            )

        beta = emission.new_zeros((batch, frames, slots))
        for time in range(frames - 2, -1, -1):
            update = torch.logsumexp(
                transitions[:, time]
                + emission[:, time + 1, None]
                + beta[:, time + 1, None],
                dim=-1,
            )
            beta[:, time] = torch.where(
                frame_mask[:, time + 1, None], update, beta[:, time + 1]
            )
        posterior = (alpha + beta).softmax(-1)
        return posterior.masked_fill(~frame_mask[:, :, None], 0.0)

    def forward(
        self,
        bound_evidence: torch.Tensor,
        confidence: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> EventProgramOutput:
        owner_logits = self.owner_pool(torch.tanh(bound_evidence)).squeeze(-1)
        owner_weights = owner_logits.softmax(-1)
        tokens = torch.einsum(
            "btmj,btmjd->btmd", owner_weights, bound_evidence
        )
        candidate_logits = torch.einsum(
            "btmd,ed->btem", tokens, self.slot_queries
        ) / math.sqrt(tokens.shape[-1])
        candidate_logits = candidate_logits + confidence[:, :, None]
        emission = torch.logsumexp(candidate_logits, dim=-1) - math.log(self.candidates)
        frame_summary = torch.einsum(
            "btm,btmd->btd", confidence.softmax(-1), tokens
        )
        boundary = self.transition(frame_summary)
        posterior = self._posteriors(emission, boundary, frame_mask)
        candidate_probability = candidate_logits.softmax(-1)
        assignment = torch.einsum(
            "bte,btem->betm", posterior, candidate_probability
        )
        occupancy = assignment.sum(dim=(2, 3))
        mass = occupancy.clamp_min(1e-6)
        process = torch.einsum(
            "betm,btmjd->bejd", assignment, bound_evidence
        ) / mass[:, :, None, None]
        second = torch.einsum(
            "betm,btmjd->bejd", assignment, bound_evidence.square()
        ) / mass[:, :, None, None]
        uncertainty = (second - process.square()).clamp_min(0.0).sqrt()
        duration_scale = F.softplus(self.minimum_duration)[None] + 1e-4
        presence = -torch.expm1(-occupancy / duration_scale)
        return EventProgramOutput(
            process=process,
            presence=presence,
            uncertainty=uncertainty,
            assignment=assignment,
            state_posterior=posterior,
        )
