"""Shared Program/candidate compatibility for bank-conditioned native anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional

from ember.ecp.contracts import ACTION_HORIZON, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.ecp.natural_program import NaturalProgram

from ember.ecp.bank_conditioning.operator import BankConditioningError


@dataclass(frozen=True)
class AnchorProgramState:
    """Task content in fixed owner/rank/event coordinates, never task IDs."""

    rank_event: torch.Tensor
    rank: torch.Tensor
    event_weights: torch.Tensor
    owner: torch.Tensor


class NativeCandidateEncoder(torch.nn.Module):
    """Nonlinear direction/log-magnitude encoder shared by native width."""

    def __init__(self, native_width: int, feature_width: int) -> None:
        super().__init__()
        hidden = 2 * feature_width
        self.native_width = int(native_width)
        self.direction = torch.nn.Sequential(
            torch.nn.Linear(native_width, hidden, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, feature_width, bias=False),
        )
        self.magnitude = torch.nn.Sequential(
            torch.nn.Linear(1, feature_width),
            torch.nn.SiLU(),
            torch.nn.Linear(feature_width, feature_width),
        )
        self.output = torch.nn.LayerNorm(feature_width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.native_width:
            raise BankConditioningError("native candidate width changed")
        value = value.float()
        magnitude = value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        direction = value / magnitude
        return self.output(
            self.direction(direction) + self.magnitude(magnitude.log())
        )


class ProgramNativeAnchorScorer(torch.nn.Module):
    """Produce bounded scalar anchor compatibilities from authorized content."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
        feature_width: int,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.feature_width = int(feature_width)
        if (
            not self.owners
            or self.program_width <= 0
            or self.event_slots <= 0
            or self.feature_width <= 0
        ):
            raise BankConditioningError("invalid native anchor topology")
        output_counts = tuple(native_output_group_count(owner) for owner in self.owners)
        self.maximum_groups = max(output_counts)
        width = self.program_width

        self.owner_embedding = torch.nn.Parameter(torch.empty(len(self.owners), width))
        self.rank_embedding = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, width)
        )
        self.event_embedding = torch.nn.Parameter(
            torch.empty(self.event_slots, width)
        )
        self.group_embedding = torch.nn.Parameter(
            torch.empty(self.maximum_groups, width)
        )
        self.program_context = torch.nn.Sequential(
            torch.nn.LayerNorm(4 * width),
            torch.nn.Linear(4 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.event_scalar = torch.nn.Linear(3, width, bias=False)
        self.rank_context = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.input_anchor_query = self._query_head(width)
        self.output_anchor_query = self._query_head(width)
        self.event_score = torch.nn.Linear(width, 1)
        self.group_gain = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, 1),
        )

        input_widths = sorted({owner.in_features for owner in self.owners})
        output_widths = sorted(
            {
                owner.out_features // native_output_group_count(owner)
                for owner in self.owners
            }
        )
        self.input_candidates = torch.nn.ModuleDict(
            {
                str(native_width): NativeCandidateEncoder(
                    native_width, self.feature_width
                )
                for native_width in input_widths
            }
        )
        self.output_candidates = torch.nn.ModuleDict(
            {
                str(native_width): NativeCandidateEncoder(
                    native_width, self.feature_width
                )
                for native_width in output_widths
            }
        )
        self.frame_event_metadata = torch.nn.Parameter(
            torch.empty(self.event_slots, self.feature_width)
        )
        self.probe_metadata = torch.nn.Parameter(
            torch.empty(G1_PROBE_COUNT, self.feature_width)
        )
        self.horizon_metadata = torch.nn.Parameter(
            torch.empty(ACTION_HORIZON, self.feature_width)
        )
        self.type_metadata = torch.nn.Parameter(
            torch.empty(len(OUTPUT_BANK_TYPES), self.feature_width)
        )
        self.time_metadata = torch.nn.Linear(2, self.feature_width, bias=False)

        for embedding in (
            self.owner_embedding,
            self.rank_embedding,
            self.event_embedding,
            self.group_embedding,
        ):
            torch.nn.init.normal_(embedding, std=width**-0.5)
        for embedding in (
            self.frame_event_metadata,
            self.probe_metadata,
            self.horizon_metadata,
            self.type_metadata,
        ):
            torch.nn.init.normal_(embedding, std=self.feature_width**-0.5)
        torch.nn.init.zeros_(self.event_score.weight)
        torch.nn.init.zeros_(self.event_score.bias)
        torch.nn.init.zeros_(self.group_gain[-1].weight)
        torch.nn.init.zeros_(self.group_gain[-1].bias)

    def _query_head(self, width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, 2 * self.feature_width),
        )

    def program_state(self, program: NaturalProgram) -> AnchorProgramState:
        targets = len(self.owners)
        expected = (targets, self.program_width)
        if (
            program.p_lang.shape != expected
            or program.p_scene.shape != expected
            or program.p_process.shape
            != (self.event_slots, targets, self.program_width)
            or program.rho.shape != (self.event_slots,)
            or program.tau.shape != (self.event_slots, 2)
            or program.sigma.shape
            != (self.event_slots, targets, self.program_width)
        ):
            raise BankConditioningError("anchor scorer Program schema changed")
        events = self.event_slots
        fields = torch.cat(
            (
                program.p_lang.float()[None].expand(events, -1, -1),
                program.p_scene.float()[None].expand(events, -1, -1),
                program.p_process.float(),
                program.sigma.float(),
            ),
            dim=-1,
        )
        rho = program.rho.float().clamp_min(1e-8)
        rho = rho / rho.sum()
        scalar = torch.cat((rho[:, None], program.tau.float()), dim=-1)
        event_owner = (
            self.program_context(fields)
            + self.event_scalar(scalar)[:, None]
            + self.event_embedding[:, None]
            + self.owner_embedding[None]
        )
        rank_event = self.rank_context(
            event_owner[:, :, None] + self.rank_embedding[None, None]
        ).permute(1, 2, 0, 3)
        event_logits = self.event_score(rank_event).squeeze(-1)
        event_weights = (event_logits + rho.log()[None, None]).softmax(-1)
        rank = torch.einsum("jre,jrew->jrw", event_weights, rank_event)
        owner = torch.einsum("e,ejw->jw", rho, event_owner)
        return AnchorProgramState(
            rank_event=rank_event,
            rank=rank,
            event_weights=event_weights,
            owner=owner,
        )

    def input_queries(self, state: AnchorProgramState) -> torch.Tensor:
        targets = len(self.owners)
        return self.input_anchor_query(state.rank_event).reshape(
            targets,
            G1_RESIDUAL_RANK,
            self.event_slots,
            2,
            self.feature_width,
        )

    def output_queries(
        self, state: AnchorProgramState, *, target: int, groups: int
    ) -> torch.Tensor:
        context = state.rank_event[target][None] + self.group_embedding[
            :groups, None, None
        ]
        return self.output_anchor_query(context).reshape(
            groups,
            G1_RESIDUAL_RANK,
            self.event_slots,
            2,
            self.feature_width,
        )

    def output_group_gains(
        self, state: AnchorProgramState, *, target: int, groups: int
    ) -> torch.Tensor:
        context = (
            state.rank[target][None] + self.group_embedding[:groups, None]
        ).detach()
        return torch.sigmoid(self.group_gain(context).squeeze(-1))

    def frame_metadata(
        self,
        assignment: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if (
            assignment.ndim != 2
            or assignment.shape[1] != self.event_slots
            or positions.shape != assignment.shape[:1]
        ):
            raise BankConditioningError("anchor frame metadata axes changed")
        frame_event = assignment.to(self.frame_event_metadata) @ self.frame_event_metadata
        points = positions.to(self.frame_event_metadata)
        time = self.time_metadata(torch.stack((points, 1.0 - points), dim=-1))
        return frame_event + time

    def candidate_metadata(
        self, frame: torch.Tensor, *, output: bool
    ) -> torch.Tensor:
        metadata = (
            frame[:, None, None]
            + self.probe_metadata[None, :, None]
            + self.horizon_metadata[None, None]
        )
        if output:
            metadata = metadata[:, :, :, None] + self.type_metadata[
                None, None, None
            ]
        return metadata

    def input_keys(self, value: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        encoded = self.input_candidates[str(value.shape[-1])](value)
        return functional.normalize(encoded + metadata, dim=-1)

    def output_keys(self, value: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        encoded = self.output_candidates[str(value.shape[-1])](value)
        return functional.normalize(encoded + metadata, dim=-1)

    def input_compatibility(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        if query.shape[-1] != self.feature_width or key.shape[-1] != self.feature_width:
            raise BankConditioningError("input anchor query/key width changed")
        score = torch.einsum("rebd,tphd->rebtph", query, key)
        return torch.tanh(score / math.sqrt(self.feature_width))

    def output_compatibility(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        if query.shape[-1] != self.feature_width or key.shape[-1] != self.feature_width:
            raise BankConditioningError("output anchor query/key width changed")
        score = torch.einsum("grebd,gtphud->grebtphu", query, key)
        return torch.tanh(score / math.sqrt(self.feature_width))
