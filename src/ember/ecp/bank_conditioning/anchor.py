"""Shared Program/candidate compatibility for bank-conditioned native anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.ecp.natural_program import NaturalProgram

from ember.ecp.bank_conditioning.compatibility import (
    NormalizedBilinearCompatibility,
)
from ember.ecp.bank_conditioning.operator import BankConditioningError


@dataclass(frozen=True)
class AnchorProgramState:
    """Task content in fixed owner/rank/event coordinates, never task IDs."""

    rank_event: torch.Tensor
    rank: torch.Tensor
    stable_rank_event: torch.Tensor
    stable_rank: torch.Tensor
    event_weights: torch.Tensor
    owner: torch.Tensor


class NativeCandidateEncoder(torch.nn.Module):
    """Fixed-owner native basis before the shared family feature trunk."""

    def __init__(self, native_width: int, feature_width: int) -> None:
        super().__init__()
        self.native_width = int(native_width)
        self.direction = torch.nn.Linear(native_width, feature_width, bias=False)
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
        direction = self.direction(direction)
        return self.output(
            direction + self.magnitude(magnitude.log())
        )


class CandidateFamilyTrunk(torch.nn.Module):
    """Share content statistics only after target-native information is kept."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class FixedOwnerQueryFiLM(torch.nn.Module):
    """Bounded query modulation with fixed LoRA-owner/group ownership."""

    def __init__(
        self,
        output_counts: Sequence[int],
        *,
        feature_width: int,
        event_slots: int,
    ) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.event_slots = int(event_slots)
        input_shape = (len(output_counts), 2, self.feature_width)
        self.input_scale = torch.nn.Parameter(torch.zeros(input_shape))
        self.input_shift = torch.nn.Parameter(torch.zeros(input_shape))
        self.output_scale = self._output_rows(output_counts)
        self.output_shift = self._output_rows(output_counts)

    def _output_rows(self, output_counts: Sequence[int]) -> torch.nn.ParameterList:
        return torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(groups, 2, self.feature_width))
                for groups in output_counts
            ]
        )

    def input(self, query: torch.Tensor, *, target: int) -> torch.Tensor:
        return self._modulate(
            query, scale=self.input_scale[target], shift=self.input_shift[target]
        )

    def output(self, query: torch.Tensor, *, target: int) -> torch.Tensor:
        return self._modulate(
            query, scale=self.output_scale[target], shift=self.output_shift[target]
        )

    def _modulate(
        self,
        query: torch.Tensor,
        *,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if scale.shape != shift.shape or query.shape[-4:] != (
            G1_RESIDUAL_RANK,
            self.event_slots,
            2,
            self.feature_width,
        ):
            raise BankConditioningError("native anchor query topology changed")
        leading = scale.shape[:-2]
        if query.shape[: len(leading)] != leading or scale.shape[-2:] != (
            2,
            self.feature_width,
        ):
            raise BankConditioningError("fixed-owner query modulation changed")
        broadcast = (*leading, 1, 1, 2, self.feature_width)
        return query * (1.0 + torch.tanh(scale).reshape(broadcast)) + torch.tanh(
            shift
        ).reshape(broadcast)


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
        families = tuple(TargetFamily)
        self.program_context = torch.nn.ModuleDict(
            {
                family.value: self._program_context(width)
                for family in families
            }
        )
        self.event_scalar = torch.nn.Linear(3, width, bias=False)
        self.rank_context = torch.nn.ModuleDict(
            {family.value: self._rank_context(width) for family in families}
        )
        self.input_anchor_query = torch.nn.ModuleDict(
            {family.value: self._query_head(width) for family in families}
        )
        self.output_anchor_query = torch.nn.ModuleDict(
            {family.value: self._query_head(width) for family in families}
        )
        # The family trunks learn a shared content coordinate system, while
        # these fixed-topology FiLM rows give each real LoRA owner an
        # independent gradient path.  They are not task or video tables: the
        # task dependence remains entirely in the Program-conditioned query.
        self.query_owner_film = FixedOwnerQueryFiLM(
            output_counts,
            feature_width=self.feature_width,
            event_slots=self.event_slots,
        )
        self.event_score = torch.nn.ModuleDict(
            {family.value: torch.nn.Linear(width, 1) for family in families}
        )
        self.group_gain = torch.nn.ModuleDict(
            {family.value: self._group_gain_head(width) for family in families}
        )
        self.input_candidates = self._candidate_encoders(output=False)
        self.output_candidates = self._candidate_encoders(output=True)
        self.input_candidate_trunks = self._candidate_trunks(families)
        self.output_candidate_trunks = self._candidate_trunks(families)
        self.input_compatibility_heads = self._compatibility_heads(families)
        self.output_compatibility_heads = self._compatibility_heads(families)
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
        self._reset_structured_parameters(families)

    def _compatibility_heads(
        self, families: Sequence[TargetFamily]
    ) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                family.value: NormalizedBilinearCompatibility(self.feature_width)
                for family in families
            }
        )

    def _candidate_encoders(self, *, output: bool) -> torch.nn.ModuleList:
        return torch.nn.ModuleList(
            NativeCandidateEncoder(
                (
                    owner.out_features // native_output_group_count(owner)
                    if output
                    else owner.in_features
                ),
                self.feature_width,
            )
            for owner in self.owners
        )

    def _candidate_trunks(
        self, families: Sequence[TargetFamily]
    ) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                family.value: CandidateFamilyTrunk(self.feature_width)
                for family in families
            }
        )

    def _reset_structured_parameters(
        self, families: Sequence[TargetFamily]
    ) -> None:
        for embedding in (
            self.owner_embedding,
            self.rank_embedding,
            self.event_embedding,
            self.group_embedding,
        ):
            torch.nn.init.normal_(embedding, std=self.program_width**-0.5)
        for embedding in (
            self.frame_event_metadata,
            self.probe_metadata,
            self.horizon_metadata,
            self.type_metadata,
        ):
            torch.nn.init.normal_(embedding, std=self.feature_width**-0.5)
        for family in families:
            torch.nn.init.zeros_(self.event_score[family.value].weight)
            torch.nn.init.zeros_(self.event_score[family.value].bias)
            torch.nn.init.zeros_(self.group_gain[family.value][-1].weight)
            torch.nn.init.zeros_(self.group_gain[family.value][-1].bias)

    @staticmethod
    def _program_context(width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(4 * width),
            torch.nn.Linear(4 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )

    @staticmethod
    def _rank_context(width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )

    @staticmethod
    def _group_gain_head(width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, 1),
        )

    def _query_head(self, width: int) -> torch.nn.Sequential:
        head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, 2 * self.feature_width),
        )
        final = head[-1]
        # Signed pooling begins with a genuine positive/negative pair instead
        # of two unrelated random branches that can nearly cancel after the
        # bank solve.  The rows remain independent parameters and may diverge
        # immediately under learned credit.
        with torch.no_grad():
            final.weight[self.feature_width :].copy_(
                -final.weight[: self.feature_width]
            )
            final.bias[self.feature_width :].copy_(
                -final.bias[: self.feature_width]
            )
        return head

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
        scalar_context = self.event_scalar(scalar)
        event_owner = torch.stack(
            tuple(
                self.program_context[owner.family.value](fields[:, target])
                + scalar_context
                + self.event_embedding
                + self.owner_embedding[target]
                for target, owner in enumerate(self.owners)
            ),
            dim=1,
        )
        rank_event = torch.stack(
            tuple(
                self.rank_context[owner.family.value](
                    event_owner[:, target, None] + self.rank_embedding[None]
                ).permute(1, 0, 2)
                for target, owner in enumerate(self.owners)
            )
        )
        event_logits = torch.stack(
            tuple(
                self.event_score[owner.family.value](rank_event[target]).squeeze(-1)
                for target, owner in enumerate(self.owners)
            )
        )
        event_weights = (event_logits + rho.log()[None, None]).softmax(-1)
        rank = torch.einsum("jre,jrew->jrw", event_weights, rank_event)
        # Scale/group-gain receive a deterministic P_lang-only view.  Reuse
        # the fixed owner/rank coordinates instead of retaining a second,
        # otherwise unsupervised neural branch beside the full-Program query.
        stable_rank = functional.layer_norm(
            program.p_lang.float()[:, None]
            + self.owner_embedding[:, None]
            + self.rank_embedding[None],
            (self.program_width,),
        )
        stable_rank_event = stable_rank[:, :, None].expand(
            -1, -1, self.event_slots, -1
        )
        owner = torch.einsum("e,ejw->jw", rho, event_owner)
        return AnchorProgramState(
            rank_event=rank_event,
            rank=rank,
            stable_rank_event=stable_rank_event,
            stable_rank=stable_rank,
            event_weights=event_weights,
            owner=owner,
        )

    def input_queries(self, state: AnchorProgramState) -> torch.Tensor:
        return torch.stack(
            tuple(
                self.query_owner_film.input(
                    self.input_anchor_query[owner.family.value](
                        state.rank_event[target]
                    ).reshape(
                        G1_RESIDUAL_RANK,
                        self.event_slots,
                        2,
                        self.feature_width,
                    ),
                    target=target,
                )
                for target, owner in enumerate(self.owners)
            )
        )

    def output_queries(
        self, state: AnchorProgramState, *, target: int, groups: int
    ) -> torch.Tensor:
        context = state.rank_event[target][None] + self.group_embedding[
            :groups, None, None
        ]
        family = self.owners[target].family.value
        return self.query_owner_film.output(
            self.output_anchor_query[family](context).reshape(
                groups,
                G1_RESIDUAL_RANK,
                self.event_slots,
                2,
                self.feature_width,
            ),
            target=target,
        )

    def output_group_gains(
        self, state: AnchorProgramState, *, target: int, groups: int
    ) -> torch.Tensor:
        context = (
            state.stable_rank[target][None] + self.group_embedding[:groups, None]
        ).detach()
        family = self.owners[target].family.value
        return torch.sigmoid(self.group_gain[family](context).squeeze(-1))

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

    def input_keys(
        self, value: torch.Tensor, metadata: torch.Tensor, *, target: int
    ) -> torch.Tensor:
        family = self.owners[target].family.value
        encoded = self.input_candidates[target](value)
        encoded = self.input_candidate_trunks[family](encoded)
        return functional.normalize(encoded + metadata, dim=-1)

    def output_keys(
        self, value: torch.Tensor, metadata: torch.Tensor, *, target: int
    ) -> torch.Tensor:
        family = self.owners[target].family.value
        encoded = self.output_candidates[target](value)
        encoded = self.output_candidate_trunks[family](encoded)
        return functional.normalize(encoded + metadata, dim=-1)

    def input_compatibility(
        self, query: torch.Tensor, key: torch.Tensor, *, target: int
    ) -> torch.Tensor:
        if (
            not 0 <= target < len(self.owners)
            or query.ndim != 4
            or query.shape[-1] != self.feature_width
            or key.shape[0] != self.event_slots
            or key.shape[-1] != self.feature_width
            or key.ndim != 5
        ):
            raise BankConditioningError("input anchor query/key width changed")
        family = self.owners[target].family.value
        return self.input_compatibility_heads[family](query, key)

    def output_compatibility(
        self, query: torch.Tensor, key: torch.Tensor, *, target: int
    ) -> torch.Tensor:
        if (
            not 0 <= target < len(self.owners)
            or query.ndim != 5
            or query.shape[-1] != self.feature_width
            or key.shape[0] != self.event_slots
            or key.shape[-1] != self.feature_width
            or key.ndim != 7
            or query.shape[0] != key.shape[1]
        ):
            raise BankConditioningError("output anchor query/key width changed")
        family = self.owners[target].family.value
        scorer = self.output_compatibility_heads[family]
        return torch.stack(
            tuple(
                scorer(query[group], key[:, group])
                for group in range(query.shape[0])
            )
        )
