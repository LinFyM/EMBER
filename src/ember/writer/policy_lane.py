"""Conditioned policy-wide A/B generation with public-lane ownership."""

from __future__ import annotations

from typing import Sequence

import torch

from ember.writer.temporal import RMSNorm, VariableEpisodeInputError


class PolicyLaneComposer(torch.nn.Module):
    """Read Core and Procedure into one condition state per public LoRA lane."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        rank: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if min(width, heads, rank) <= 0 or width % heads:
            raise VariableEpisodeInputError("invalid policy-lane composer")
        self.width = int(width)
        self.rank = int(rank)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        queries = torch.empty(rank, width)
        queries.normal_(mean=0.0, std=0.02, generator=generator)
        self.lane_queries = torch.nn.Parameter(queries)
        self.core_norm = RMSNorm(width)
        self.procedure_norm = RMSNorm(width)
        self.core_reader = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.procedure_reader = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.fusion = torch.nn.Linear(2 * width, width, bias=False)
        self.output_norm = RMSNorm(width)

    @staticmethod
    def _validate(
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> None:
        if (
            core.ndim != 3
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or procedure.ndim != 3
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or core.shape[0] != procedure.shape[0]
            or core.shape[2] != procedure.shape[2]
            or not bool(valid_core.any(dim=1).all())
            or not bool(valid_procedure.any(dim=1).all())
        ):
            raise VariableEpisodeInputError("invalid policy-lane memories")

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(core, valid_core, procedure, valid_procedure)
        queries = self.lane_queries[None].expand(core.shape[0], -1, -1)
        core_read, _ = self.core_reader(
            queries,
            self.core_norm(core),
            core,
            key_padding_mask=~valid_core,
            need_weights=False,
        )
        procedure_read, _ = self.procedure_reader(
            queries + core_read,
            self.procedure_norm(procedure),
            procedure,
            key_padding_mask=~valid_procedure,
            need_weights=False,
        )
        lanes = self.output_norm(
            torch.nn.functional.gelu(
                self.fusion(torch.cat((core_read, procedure_read), dim=-1))
            )
        )
        if lanes.shape != (core.shape[0], self.rank, self.width) or not bool(
            torch.isfinite(lanes).all()
        ):
            raise VariableEpisodeInputError("invalid policy-lane condition states")
        return lanes


class PolicyLaneHyperdecoder(torch.nn.Module):
    """Generate one coupled policy-wide A/B direction per public LoRA lane."""

    def __init__(
        self,
        *,
        target_widths: Sequence[tuple[int, int]],
        rank: int,
        condition_width: int,
        hidden_width: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            not target_widths
            or min(rank, condition_width, hidden_width) <= 0
            or any(
                min(input_width, output_width) <= 0
                for input_width, output_width in target_widths
            )
        ):
            raise VariableEpisodeInputError("invalid policy-lane hyperdecoder")
        self.rank = int(rank)
        self.condition_width = int(condition_width)
        self.hidden_width = int(hidden_width)
        self.target_widths = tuple(
            (int(input_width), int(output_width))
            for input_width, output_width in target_widths
        )
        self.total_a_width = sum(value[0] for value in self.target_widths)
        self.total_b_width = sum(value[1] for value in self.target_widths)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        lane_input = torch.empty(rank, hidden_width, condition_width)
        lane_input.normal_(mean=0.0, std=0.02, generator=generator)
        self.lane_input = torch.nn.Parameter(lane_input)
        self.a_output = torch.nn.Parameter(
            torch.zeros(rank, self.total_a_width, hidden_width)
        )
        self.b_output = torch.nn.Parameter(
            torch.zeros(rank, self.total_b_width, hidden_width)
        )

    def hidden_states(self, lanes: torch.Tensor) -> torch.Tensor:
        if lanes.ndim != 3 or lanes.shape[1:] != (
            self.rank,
            self.condition_width,
        ):
            raise VariableEpisodeInputError("policy-lane condition shape changed")
        hidden = torch.nn.functional.gelu(
            torch.einsum("brd,rhd->brh", lanes, self.lane_input)
        )
        if not bool(torch.isfinite(hidden).all()):
            raise VariableEpisodeInputError("non-finite policy-lane hidden state")
        return hidden

    def forward(
        self,
        lanes: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        hidden = self.hidden_states(lanes)
        flat_a = torch.einsum("brh,roh->bro", hidden, self.a_output)
        flat_b = torch.einsum("brh,roh->bro", hidden, self.b_output)
        result = []
        a_start = b_start = 0
        for input_width, output_width in self.target_widths:
            a = flat_a[:, :, a_start : a_start + input_width]
            b = flat_b[:, :, b_start : b_start + output_width].transpose(1, 2)
            result.append((a, b))
            a_start += input_width
            b_start += output_width
        if a_start != self.total_a_width or b_start != self.total_b_width:
            raise VariableEpisodeInputError("policy-lane target slicing changed")
        return tuple(result)
