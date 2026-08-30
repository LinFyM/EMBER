"""Program-conditioned primal directions in each real LoRA target space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as functional

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    native_output_group_count,
)
from ember.ecp.natural_program import NaturalProgram

from ember.ecp.bank_conditioning.operator import BankConditioningError


@dataclass(frozen=True)
class PrimalProgramState:
    """Full Program expressed in fixed owner/rank/event coordinates."""

    rank_event: torch.Tensor
    rank: torch.Tensor
    stable_rank: torch.Tensor
    event_weights: torch.Tensor
    owner: torch.Tensor


class ProgramNativePrimalScorer(torch.nn.Module):
    """Map a shared Program to target-native primal directions.

    The target rows are fixed LoRA ownership, not task lookup.  Task dependence
    enters only through the frozen G2 Program.  These primals are not emitted
    LoRA factors: the current video bank must first map them to legal dual
    queries and replay those queries over real native X/Y values.
    """

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        if not self.owners or self.program_width <= 0 or self.event_slots <= 0:
            raise BankConditioningError("invalid Program-primal topology")

        width = self.program_width
        output_counts = tuple(native_output_group_count(owner) for owner in self.owners)
        self.maximum_groups = max(output_counts)
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(self.owners), width))
        self.rank_embedding = torch.nn.Parameter(
            torch.empty(G1_RESIDUAL_RANK, width)
        )
        self.event_embedding = torch.nn.Parameter(torch.empty(self.event_slots, width))
        self.group_embedding = torch.nn.Parameter(
            torch.empty(self.maximum_groups, width)
        )
        families = tuple(TargetFamily)
        self.program_context = torch.nn.ModuleDict(
            {
                family.value: self._context(4 * width, width)
                for family in families
            }
        )
        self.event_scalar = torch.nn.Linear(3, width, bias=False)
        self.rank_context = torch.nn.ModuleDict(
            {family.value: self._context(width, width) for family in families}
        )
        self.input_trunk = torch.nn.ModuleDict(
            {family.value: self._context(width, width) for family in families}
        )
        self.output_trunk = torch.nn.ModuleDict(
            {family.value: self._context(width, width) for family in families}
        )
        self.event_score = torch.nn.ModuleDict(
            {family.value: torch.nn.Linear(width, 1) for family in families}
        )
        self.input_primal_heads = torch.nn.ModuleList(
            torch.nn.Linear(width, owner.in_features, bias=False)
            for owner in self.owners
        )
        self.output_primal_heads = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(
                    width,
                    owner.out_features // native_output_group_count(owner),
                    bias=False,
                )
                for _ in range(native_output_group_count(owner))
            )
            for owner in self.owners
        )
        self._reset(families)

    @staticmethod
    def _context(input_width: int, width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(input_width),
            torch.nn.Linear(input_width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )

    def _reset(self, families: Sequence[TargetFamily]) -> None:
        for embedding in (
            self.owner_embedding,
            self.rank_embedding,
            self.event_embedding,
            self.group_embedding,
        ):
            torch.nn.init.normal_(embedding, std=self.program_width**-0.5)
        for family in families:
            torch.nn.init.zeros_(self.event_score[family.value].weight)
            torch.nn.init.zeros_(self.event_score[family.value].bias)

    def _program_inputs(
        self, program: NaturalProgram
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            raise BankConditioningError("Program-primal schema changed")

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
        return fields, rho, self.event_scalar(scalar)

    def program_state(self, program: NaturalProgram) -> PrimalProgramState:
        fields, rho, scalar_context = self._program_inputs(program)
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
        stable_rank = functional.layer_norm(
            program.p_lang.float()[:, None]
            + self.owner_embedding[:, None]
            + self.rank_embedding[None],
            (self.program_width,),
        )
        owner_state = torch.einsum("e,ejw->jw", rho, event_owner)
        return PrimalProgramState(
            rank_event=rank_event,
            rank=rank,
            stable_rank=stable_rank,
            event_weights=event_weights,
            owner=owner_state,
        )

    def input_head_features(
        self, state: PrimalProgramState
    ) -> tuple[torch.Tensor, ...]:
        """Return the exact hidden rows consumed by each native input head."""

        return tuple(
            self.input_trunk[owner.family.value](state.rank[target])
            for target, owner in enumerate(self.owners)
        )

    def output_head_features(
        self, state: PrimalProgramState
    ) -> tuple[torch.Tensor, ...]:
        """Return ``[group, rank, width]`` rows consumed by output heads."""

        rows = []
        for target, owner in enumerate(self.owners):
            groups = native_output_group_count(owner)
            context = state.rank[target][None] + self.group_embedding[:groups, None]
            rows.append(self.output_trunk[owner.family.value](context))
        return tuple(rows)

    def input_primals(self, state: PrimalProgramState) -> tuple[torch.Tensor, ...]:
        features = self.input_head_features(state)
        return tuple(
            head(hidden)
            for head, hidden in zip(
                self.input_primal_heads, features, strict=True
            )
        )

    def input_event_queries(
        self, state: PrimalProgramState
    ) -> tuple[torch.Tensor, ...]:
        """Reuse each native head on unaggregated ``[rank,event]`` states."""

        return tuple(
            head(
                self.input_trunk[owner.family.value](
                    state.rank_event[target]
                )
            )
            for target, (owner, head) in enumerate(
                zip(self.owners, self.input_primal_heads, strict=True)
            )
        )

    def output_primals(
        self, state: PrimalProgramState
    ) -> tuple[torch.Tensor, ...]:
        features = self.output_head_features(state)
        return tuple(
            torch.stack(
                tuple(head(hidden[group]) for group, head in enumerate(heads))
            )
            for heads, hidden in zip(
                self.output_primal_heads, features, strict=True
            )
        )

    def output_event_queries(
        self, state: PrimalProgramState
    ) -> tuple[torch.Tensor, ...]:
        """Reuse owner/group heads on ``[group,rank,event]`` states."""

        rows = []
        for target, (owner, heads) in enumerate(
            zip(self.owners, self.output_primal_heads, strict=True)
        ):
            groups = native_output_group_count(owner)
            context = (
                state.rank_event[target][None]
                + self.group_embedding[:groups, None, None]
            )
            hidden = self.output_trunk[owner.family.value](context)
            rows.append(
                torch.stack(
                    tuple(head(hidden[group]) for group, head in enumerate(heads))
                )
            )
        return tuple(rows)
