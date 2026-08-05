"""Conditioned mixing over policy-wide LoRA atom coordinates."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ember.writer.temporal import RMSNorm, VariableEpisodeInputError


class PolicyCoordinateComposer(torch.nn.Module):
    """Read Core and Procedure into rank-coordinate atom mixing matrices."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        rank: int,
        atom_count: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(width, heads, rank, atom_count) <= 0
            or width % heads
            or rank > atom_count
        ):
            raise VariableEpisodeInputError("invalid policy-coordinate composer")
        self.width = int(width)
        self.rank = int(rank)
        self.atom_count = int(atom_count)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)

        def parameter(rows: int) -> torch.nn.Parameter:
            value = torch.empty(rows, width)
            value.normal_(mean=0.0, std=0.02, generator=generator)
            return torch.nn.Parameter(value)

        self.coordinate_queries = parameter(rank)
        self.atom_a_keys = parameter(atom_count)
        self.atom_b_keys = parameter(atom_count)
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
        self.mixing_norm = RMSNorm(width)
        self.a_query = torch.nn.Linear(width, width, bias=False)
        self.b_query = torch.nn.Linear(width, width, bias=False)

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
            raise VariableEpisodeInputError("invalid policy-coordinate memories")

    def forward(
        self,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(core, valid_core, procedure, valid_procedure)
        queries = self.coordinate_queries[None].expand(core.shape[0], -1, -1)
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
        content = torch.nn.functional.gelu(
            self.fusion(torch.cat((core_read, procedure_read), dim=-1))
        )
        normalized = self.mixing_norm(content)
        scale = 1.0 / math.sqrt(self.width)
        mix_a = torch.einsum(
            "brd,kd->brk",
            self.a_query(normalized),
            self.atom_a_keys,
        ) * scale
        mix_b = torch.einsum(
            "brd,kd->brk",
            self.b_query(normalized),
            self.atom_b_keys,
        ) * scale
        if (
            mix_a.shape != (core.shape[0], self.rank, self.atom_count)
            or mix_b.shape != mix_a.shape
            or not bool(torch.isfinite(mix_a).all())
            or not bool(torch.isfinite(mix_b).all())
        ):
            raise VariableEpisodeInputError("invalid policy-atom mixing")
        return mix_a, mix_b


class PolicyWideAtomDictionary(torch.nn.Module):
    """Store shared-index rank-one atoms spanning every public policy target."""

    def __init__(
        self,
        *,
        target_widths: Sequence[tuple[int, int]],
        rank: int,
        atom_count: int,
    ) -> None:
        super().__init__()
        if (
            not target_widths
            or min(rank, atom_count) <= 0
            or rank > atom_count
            or any(
                min(input_width, output_width) <= 0
                for input_width, output_width in target_widths
            )
        ):
            raise VariableEpisodeInputError("invalid policy-wide atom dictionary")
        self.rank = int(rank)
        self.atom_count = int(atom_count)
        self.target_widths = tuple(
            (int(input_width), int(output_width))
            for input_width, output_width in target_widths
        )
        self.a_atoms = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(atom_count, input_width))
                for input_width, _ in self.target_widths
            ]
        )
        self.b_atoms = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(output_width, atom_count))
                for _, output_width in self.target_widths
            ]
        )

    def forward(
        self,
        mix_a: torch.Tensor,
        mix_b: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if (
            mix_a.ndim != 3
            or mix_a.shape[1:] != (self.rank, self.atom_count)
            or mix_b.shape != mix_a.shape
        ):
            raise VariableEpisodeInputError("policy dictionary mixing shape changed")
        result = []
        for a_atoms, b_atoms in zip(self.a_atoms, self.b_atoms, strict=True):
            a = torch.einsum("brk,ki->bri", mix_a, a_atoms)
            b = torch.einsum("ok,brk->bor", b_atoms, mix_b)
            result.append((a, b))
        return tuple(result)
