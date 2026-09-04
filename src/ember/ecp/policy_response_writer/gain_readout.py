"""Shared current-factor-conditioned relative group-gain readout."""

from __future__ import annotations

from typing import Sequence

import torch

from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import G1_RESIDUAL_RANK, native_output_group_count
from ember.ecp.policy_response_writer.process import GatedMLP


class FactorConditionedGroupGainReadout(torch.nn.Module):
    """Apply one copyable utility rule to every native output group.

    The readout has no target- or group-owned output rows.  Target, rank, and
    video identity arrive through the contextual query and the current signed
    native factors; group identity is a shared positional embedding.  Adding a
    target or another layer therefore adds tokens, not private readout weights.
    """

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        width: int,
        block_depth: int = 1,
    ) -> None:
        super().__init__()
        values = tuple(owners)
        if not values or width <= 0 or block_depth <= 0:
            raise ValueError("factor-conditioned group-gain topology changed")
        input_widths = sorted({owner.in_features for owner in values})
        output_widths = sorted(
            {
                owner.out_features // native_output_group_count(owner)
                for owner in values
            }
        )
        maximum_groups = max(native_output_group_count(owner) for owner in values)
        self.width = width
        self.input_encoder = torch.nn.ModuleDict(
            {
                str(value): torch.nn.Linear(value, width, bias=False)
                for value in input_widths
            }
        )
        self.output_encoder = torch.nn.ModuleDict(
            {
                str(value): torch.nn.Linear(value, width, bias=False)
                for value in output_widths
            }
        )
        self.group_embedding = torch.nn.Embedding(maximum_groups, width)
        self.fusion = torch.nn.Linear(4 * width, width)
        self.blocks = torch.nn.ModuleList(
            GatedMLP(width) for _ in range(block_depth)
        )
        self.output_norm = torch.nn.LayerNorm(width)
        self.output = torch.nn.Linear(width, 1)
        torch.nn.init.normal_(self.group_embedding.weight, std=width**-0.5)
        torch.nn.init.zeros_(self.output.weight)
        # Match the proved G1 opening.  The nonzero gain lets the first
        # functional backward train signed candidate directions, while the
        # conditional output weights receive credit immediately.
        torch.nn.init.constant_(self.output.bias, 0.1)

    def forward(
        self,
        query: torch.Tensor,
        input_factor: torch.Tensor,
        output_factor: torch.Tensor,
        *,
        groups: int,
    ) -> torch.Tensor:
        if (
            query.shape != (G1_RESIDUAL_RANK, self.width)
            or input_factor.ndim != 2
            or input_factor.shape[0] != G1_RESIDUAL_RANK
            or output_factor.ndim != 2
            or output_factor.shape[0] != G1_RESIDUAL_RANK
            or groups <= 0
            or output_factor.shape[1] % groups
            or groups > self.group_embedding.num_embeddings
        ):
            raise ValueError("factor-conditioned group-gain inputs changed")
        group_width = output_factor.shape[1] // groups
        input_key = str(input_factor.shape[1])
        output_key = str(group_width)
        if (
            input_key not in self.input_encoder
            or output_key not in self.output_encoder
        ):
            raise ValueError("factor-conditioned native width changed")

        input_context = self.input_encoder[input_key](input_factor)
        output_context = self.output_encoder[output_key](
            output_factor.reshape(G1_RESIDUAL_RANK, groups, group_width)
        )
        group_context = self.group_embedding.weight[:groups]
        token = self.fusion(
            torch.cat(
                (
                    query[:, None].expand(-1, groups, -1),
                    input_context[:, None].expand(-1, groups, -1),
                    output_context,
                    group_context[None].expand(G1_RESIDUAL_RANK, -1, -1),
                ),
                dim=-1,
            )
        )
        for block in self.blocks:
            token = block(token)
        logits = self.output(self.output_norm(token)).squeeze(-1)
        if logits.shape != (G1_RESIDUAL_RANK, groups):
            raise ValueError("factor-conditioned group-gain output changed")
        return logits
