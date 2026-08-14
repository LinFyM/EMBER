"""Gradient-open commitment of video memory into V6 factor families."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ember.writer.errors import WriterModelError
from ember.writer.temporal import RMSNorm


FACTOR_FAMILIES = (
    "q_a",
    "q_b",
    "v_a",
    "v_b",
    "action_in_a",
    "action_in_b",
    "action_out_a",
    "action_out_b",
)


class GradientOpenSemanticCommitment(torch.nn.Module):
    """Route video memory with exact-zero output and open first-step credit."""

    def __init__(
        self,
        *,
        width: int = 256,
        basis_count: int = 4,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if width <= 0 or basis_count <= 0:
            raise WriterModelError("invalid semantic factor-memory topology")
        self.width = int(width)
        self.basis_count = int(basis_count)
        self.language_norm = RMSNorm(width)
        self.basis_norm = RMSNorm(width)
        self.semantic_query = torch.nn.Linear(width, width, bias=False)
        self.basis_keys = torch.nn.Parameter(torch.empty(basis_count, width))
        self.family_maps = torch.nn.ParameterDict(
            {
                family: torch.nn.Parameter(
                    torch.zeros(basis_count, width, width)
                )
                for family in FACTOR_FAMILIES
            }
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(initialization_seed) + 0x53464D43
        )
        torch.nn.init.zeros_(self.semantic_query.weight)
        self.basis_keys.data.normal_(mean=0.0, std=0.02, generator=generator)
        self.register_buffer(
            "anchor_input_sign",
            torch.where(torch.arange(width) % 2 == 0, 1, -1).to(torch.int8),
            persistent=False,
        )

    def basis_weights(self, language_slots: torch.Tensor) -> torch.Tensor:
        """Return the language-only soft address for every policy slot."""

        if language_slots.ndim != 3 or language_slots.shape[-1] != self.width:
            raise WriterModelError("semantic address lost its policy-slot layout")
        query = self.semantic_query(self.language_norm(language_slots))
        keys = self.basis_norm(self.basis_keys)
        logits = torch.einsum("bsi,ki->bsk", query, keys) * self.width**-0.5
        return torch.softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)

    def hidden_residuals(
        self,
        factor_memory: torch.Tensor,
        basis_weights: torch.Tensor,
        *,
        anchor_input_weights: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        """Add learned deltas and V6-W1 anchored address tangents."""

        if (
            factor_memory.ndim != 3
            or factor_memory.shape[-1] != self.width
            or basis_weights.shape
            != (*factor_memory.shape[:2], self.basis_count)
        ):
            raise WriterModelError("invalid gradient-open semantic commitment")
        if set(anchor_input_weights) != set(FACTOR_FAMILIES):
            raise WriterModelError("factor anchor families changed")
        sign = self.anchor_input_sign
        result: dict[str, torch.Tensor] = {}
        for family, maps in self.family_maps.items():
            anchor_weight = anchor_input_weights[family]
            if anchor_weight.shape != (self.width, self.width):
                raise WriterModelError("factor anchor topology changed")
            learned = torch.einsum(
                "bski,bsk->bsi",
                torch.einsum("bsj,kij->bski", factor_memory, maps),
                basis_weights,
            )
            direct_anchor = torch.einsum(
                "bsj,ij->bsi", factor_memory, anchor_weight
            )
            signed_anchor = torch.einsum(
                "bsj,j,ij->bsi", factor_memory, sign, anchor_weight
            )
            anchored = (
                direct_anchor
                * (basis_weights[..., 0] - basis_weights[..., 1]).unsqueeze(-1)
                + signed_anchor
                * (basis_weights[..., 2] - basis_weights[..., 3]).unsqueeze(-1)
            )
            result[family] = learned + anchored
        return result

    def forward(
        self,
        factor_memory: torch.Tensor,
        language_slots: torch.Tensor,
        *,
        anchor_input_weights: Mapping[str, torch.Tensor],
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        weights = self.basis_weights(language_slots)
        return (
            self.hidden_residuals(
                factor_memory,
                weights,
                anchor_input_weights=anchor_input_weights,
            ),
            weights,
        )
