"""Program/free query and real-native key parameterizations for PNBTT."""

from __future__ import annotations

from typing import Sequence

import torch

from ember.ecp.bank_conditioning.key_value_replay import safe_rms_normalize
from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.ecp.natural_program import NaturalProgram


PNBTT_SIDES = ("input", *OUTPUT_BANK_TYPES)
PNBTT_KEY_FAMILIES = (
    TargetFamily.Q,
    TargetFamily.V,
    TargetFamily.ACTION_IN,
    TargetFamily.ACTION_OUT,
)


def _native_key_value_width(owner: TargetOwner, side: int) -> int:
    if side == 0:
        return int(owner.in_features)
    return int(owner.out_features // native_output_group_count(owner))


class ProgramTangentQuery(torch.nn.Module):
    """The only Program path: complete Natural Program to low-dimensional query."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        program_width: int,
        event_slots: int,
        key_width: int,
        hidden_width: int,
        query_epsilon: float,
        residual_rank: int = G1_RESIDUAL_RANK,
    ) -> None:
        super().__init__()
        self.target_count = len(tuple(owners))
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.key_width = int(key_width)
        self.query_epsilon = float(query_epsilon)
        self.residual_rank = int(residual_rank)
        if self.residual_rank <= 0:
            raise ValueError("PNBTT residual rank must be positive")
        identity_width = 16
        self.target_identity = torch.nn.Embedding(self.target_count, identity_width)
        self.rank_identity = torch.nn.Embedding(self.residual_rank, identity_width)
        self.event_identity = torch.nn.Embedding(self.event_slots, identity_width)
        self.side_identity = torch.nn.Embedding(len(PNBTT_SIDES), identity_width)
        context_width = 4 * self.program_width + 3 + 4 * identity_width
        self.context_norm = torch.nn.LayerNorm(context_width)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(context_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.GELU(),
        )
        self.side_heads = torch.nn.ModuleList(
            torch.nn.Linear(hidden_width, self.key_width) for _ in PNBTT_SIDES
        )
        for head in self.side_heads:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        # Standard asymmetric LoRA zero: A is bank-realized at step zero while
        # every B side is exact zero, allowing first-step functional credit.
        generator = torch.Generator(device="cpu").manual_seed(20260902)
        self.side_heads[0].weight.data.normal_(
            mean=0.0, std=0.01, generator=generator
        )

    def forward(self, program: NaturalProgram) -> torch.Tensor:
        targets, events, width = (
            self.target_count,
            self.event_slots,
            self.program_width,
        )
        if (
            program.p_lang.shape != (targets, width)
            or program.p_scene.shape != (targets, width)
            or program.p_process.shape != (events, targets, width)
            or program.rho.shape != (events,)
            or program.tau.shape != (events, 2)
            or program.sigma.shape != (events, targets, width)
        ):
            raise BankConditioningError("PNBTT Natural Program schema changed")
        device = program.p_lang.device
        target_ids = torch.arange(targets, device=device)
        rank_ids = torch.arange(self.residual_rank, device=device)
        event_ids = torch.arange(events, device=device)
        side_ids = torch.arange(len(PNBTT_SIDES), device=device)
        base = torch.cat(
            (
                program.p_lang[:, None].expand(-1, events, -1),
                program.p_scene[:, None].expand(-1, events, -1),
                program.p_process.permute(1, 0, 2),
                program.sigma.permute(1, 0, 2),
                program.rho[None, :, None].expand(targets, -1, -1),
                program.tau[None].expand(targets, -1, -1),
            ),
            dim=-1,
        )
        shape = (targets, self.residual_rank, events, len(PNBTT_SIDES), -1)
        base = base[:, None, :, None].expand(*shape[:-1], base.shape[-1])
        target = self.target_identity(target_ids)[:, None, None, None].expand(
            *shape[:-1], -1
        )
        rank = self.rank_identity(rank_ids)[None, :, None, None].expand(
            *shape[:-1], -1
        )
        event = self.event_identity(event_ids)[None, None, :, None].expand(
            *shape[:-1], -1
        )
        side = self.side_identity(side_ids)[None, None, None].expand(
            *shape[:-1], -1
        )
        hidden = self.trunk(
            self.context_norm(torch.cat((base, target, rank, event, side), -1))
        )
        raw = torch.stack(
            tuple(
                self.side_heads[index](hidden[..., index, :])
                for index in range(len(PNBTT_SIDES))
            ),
            dim=3,
        )
        return safe_rms_normalize(raw, epsilon=self.query_epsilon)


class TaskLocalFreeTangentQuery(torch.nn.Module):
    """Training-only E1 query table; never part of the deployment compiler."""

    def __init__(
        self,
        task_ids: Sequence[int],
        owners: Sequence[TargetOwner],
        *,
        event_slots: int,
        key_width: int,
        query_epsilon: float,
        residual_rank: int = G1_RESIDUAL_RANK,
    ) -> None:
        super().__init__()
        self.task_ids = tuple(map(int, task_ids))
        self.query_epsilon = float(query_epsilon)
        self.residual_rank = int(residual_rank)
        if self.residual_rank <= 0:
            raise ValueError("PNBTT residual rank must be positive")
        self.raw_query = torch.nn.Parameter(
            torch.zeros(
                len(self.task_ids),
                len(tuple(owners)),
                self.residual_rank,
                int(event_slots),
                len(PNBTT_SIDES),
                int(key_width),
            )
        )
        generator = torch.Generator(device="cpu").manual_seed(20260902)
        self.raw_query.data[..., 0, :].normal_(
            mean=0.0, std=0.01, generator=generator
        )

    def _task_index(self, task_id: int) -> int:
        try:
            return self.task_ids.index(int(task_id))
        except ValueError as error:
            raise BankConditioningError("E1 task has no free tangent query") from error

    def forward(self, task_id: int) -> torch.Tensor:
        return safe_rms_normalize(
            self.raw_query[self._task_index(task_id)], epsilon=self.query_epsilon
        )

    def target(self, task_id: int, target: int) -> torch.Tensor:
        index = self._task_index(task_id)
        if not 0 <= int(target) < self.raw_query.shape[1]:
            raise BankConditioningError("E1 free-query target changed")
        return safe_rms_normalize(
            self.raw_query[index, int(target)], epsilon=self.query_epsilon
        )


class _LowRankTargetKeyResidual(torch.nn.Module):
    """Necessary target chart residual without a task/video lookup."""

    def __init__(self, input_width: int, key_width: int, rank: int) -> None:
        super().__init__()
        if min(input_width, key_width, rank) <= 0 or rank > min(
            input_width, key_width
        ):
            raise ValueError("invalid target key residual rank")
        self.down = torch.nn.Linear(input_width, rank, bias=False)
        self.up = torch.nn.Linear(rank, key_width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value))


class NativeTangentKey(torch.nn.Module):
    """Family-shared nonlinear chart plus target-specific low-rank residual."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        key_width: int,
        hidden_width: int,
        target_projection_rank: int,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.key_width = int(key_width)
        self.hidden_width = int(hidden_width)
        self.target_projection_rank = int(target_projection_rank)
        if min(self.key_width, self.hidden_width, self.target_projection_rank) <= 0:
            raise ValueError("invalid family key chart width")

        family_widths = []
        for family in PNBTT_KEY_FAMILIES:
            selected = tuple(owner for owner in self.owners if owner.family is family)
            if not selected:
                raise ValueError("PNBTT key family lost all target owners")
            input_widths = {owner.in_features for owner in selected}
            output_widths = {
                owner.out_features // native_output_group_count(owner)
                for owner in selected
            }
            if len(input_widths) != 1 or len(output_widths) != 1:
                raise ValueError("PNBTT family key chart dimensions changed")
            family_widths.append(
                (next(iter(input_widths)), next(iter(output_widths)))
            )
        self.family_widths = tuple(family_widths)

        # The first trunk is shared by all input targets in one family; the
        # second is shared by all four real Y bank types and all output groups.
        self.family_content_trunks = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Linear(width, self.hidden_width),
                    torch.nn.GELU(),
                )
                for width in widths
            )
            for widths in self.family_widths
        )
        self.family_side_heads = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(self.hidden_width, self.key_width, bias=False)
                for _ in PNBTT_SIDES
            )
            for _ in PNBTT_KEY_FAMILIES
        )
        self.family_metadata_projection = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(3, self.key_width, bias=False)
                for _ in PNBTT_SIDES
            )
            for _ in PNBTT_KEY_FAMILIES
        )
        self.family_side_norm = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.LayerNorm(self.key_width) for _ in PNBTT_SIDES
            )
            for _ in PNBTT_KEY_FAMILIES
        )
        self.target_side_residual = torch.nn.ModuleList(
            torch.nn.ModuleList(
                _LowRankTargetKeyResidual(
                    _native_key_value_width(owner, side),
                    self.key_width,
                    min(
                        self.target_projection_rank,
                        _native_key_value_width(owner, side),
                        self.key_width,
                    ),
                )
                for side in range(len(PNBTT_SIDES))
            )
            for owner in self.owners
        )

    def forward(
        self,
        *,
        target: int,
        side: int,
        normalized_values: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not 0 <= int(target) < len(self.owners)
            or not 0 <= int(side) < len(PNBTT_SIDES)
        ):
            raise BankConditioningError("PNBTT family key chart owner changed")
        owner = self.owners[int(target)]
        family = PNBTT_KEY_FAMILIES.index(owner.family)
        expected_width = self.family_widths[family][0 if side == 0 else 1]
        if normalized_values.shape[-1] != expected_width or metadata.shape != (
            *normalized_values.shape[:-1],
            3,
        ):
            raise BankConditioningError("PNBTT family key candidate shape changed")
        trunk = self.family_content_trunks[family][0 if side == 0 else 1]
        shared = self.family_side_heads[family][side](trunk(normalized_values))
        residual = self.target_side_residual[int(target)][int(side)](
            normalized_values
        )
        metadata_key = self.family_metadata_projection[family][side](
            metadata.to(shared)
        )
        return self.family_side_norm[family][side](
            shared + residual + metadata_key
        )
