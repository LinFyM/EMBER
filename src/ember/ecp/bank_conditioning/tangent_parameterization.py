"""Program/free query and real-native key parameterizations for PNBTT."""

from __future__ import annotations

from typing import Sequence

import torch

from ember.ecp.bank_conditioning.key_value_replay import safe_rms_normalize
from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.ecp.natural_program import NaturalProgram


PNBTT_SIDES = ("input", *OUTPUT_BANK_TYPES)


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
    ) -> None:
        super().__init__()
        self.target_count = len(tuple(owners))
        self.program_width = int(program_width)
        self.event_slots = int(event_slots)
        self.key_width = int(key_width)
        self.query_epsilon = float(query_epsilon)
        identity_width = 16
        self.target_identity = torch.nn.Embedding(self.target_count, identity_width)
        self.rank_identity = torch.nn.Embedding(G1_RESIDUAL_RANK, identity_width)
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
        rank_ids = torch.arange(G1_RESIDUAL_RANK, device=device)
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
        shape = (targets, G1_RESIDUAL_RANK, events, len(PNBTT_SIDES), -1)
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
    ) -> None:
        super().__init__()
        self.task_ids = tuple(map(int, task_ids))
        self.query_epsilon = float(query_epsilon)
        self.raw_query = torch.nn.Parameter(
            torch.zeros(
                len(self.task_ids),
                len(tuple(owners)),
                G1_RESIDUAL_RANK,
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


class NativeTangentKey(torch.nn.Module):
    """Map normalized real native candidates and legal metadata into key space."""

    def __init__(self, owners: Sequence[TargetOwner], *, key_width: int) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.key_width = int(key_width)
        self.input_projection = torch.nn.ModuleList(
            torch.nn.Linear(owner.in_features, self.key_width, bias=False)
            for owner in self.owners
        )
        self.output_projection = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(
                    owner.out_features // native_output_group_count(owner),
                    self.key_width,
                    bias=False,
                )
                for _ in OUTPUT_BANK_TYPES
            )
            for owner in self.owners
        )
        self.metadata_projection = torch.nn.ModuleList(
            torch.nn.Linear(3, self.key_width, bias=False) for _ in PNBTT_SIDES
        )
        self.side_norm = torch.nn.ModuleList(
            torch.nn.LayerNorm(self.key_width) for _ in PNBTT_SIDES
        )

    def forward(
        self,
        *,
        target: int,
        side: int,
        normalized_values: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if side == 0:
            content = self.input_projection[target](normalized_values)
        else:
            content = self.output_projection[target][side - 1](normalized_values)
        return self.side_norm[side](
            content + self.metadata_projection[side](metadata.to(content))
        )
