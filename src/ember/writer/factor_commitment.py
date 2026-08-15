"""Commit native Action-probe Value along V6 policy-aligned directions."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

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


class NativeProbeValueCommitment(torch.nn.Module):
    """Use native probe Value as coefficients over language-policy axes."""

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
        probe_value_memory: torch.Tensor,
        language_slots: torch.Tensor,
        basis_weights: torch.Tensor,
        *,
        anchor_input_weights: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        """Transport two video coefficients along task-shared hidden axes."""

        if (
            probe_value_memory.ndim != 3
            or probe_value_memory.shape[-1] != self.width
            or language_slots.shape != probe_value_memory.shape
            or basis_weights.shape
            != (*probe_value_memory.shape[:2], self.basis_count)
        ):
            raise WriterModelError("invalid native probe-Value commitment")
        if set(anchor_input_weights) != set(FACTOR_FAMILIES):
            raise WriterModelError("factor anchor families changed")
        language = self.language_norm(language_slots)
        signed_language = language * self.anchor_input_sign
        scale = self.width**-0.5
        direct_coefficient = (
            probe_value_memory * language
        ).sum(dim=-1, keepdim=True) * scale
        signed_coefficient = (
            probe_value_memory * signed_language
        ).sum(dim=-1, keepdim=True) * scale
        direct_route = (
            basis_weights[..., 0] - basis_weights[..., 1]
        ).unsqueeze(-1)
        signed_route = (
            basis_weights[..., 2] - basis_weights[..., 3]
        ).unsqueeze(-1)
        result: dict[str, torch.Tensor] = {}
        for family in FACTOR_FAMILIES:
            anchor_weight = anchor_input_weights[family]
            if anchor_weight.shape != (self.width, self.width):
                raise WriterModelError("factor anchor topology changed")
            direct_axis = F.gelu(
                torch.einsum("bsj,ij->bsi", language, anchor_weight)
            )
            signed_axis = F.gelu(
                torch.einsum("bsj,ij->bsi", signed_language, anchor_weight)
            )
            result[family] = (
                direct_route * direct_coefficient * direct_axis
                + signed_route * signed_coefficient * signed_axis
            )
        return result

    def forward(
        self,
        probe_value_memory: torch.Tensor,
        language_slots: torch.Tensor,
        *,
        anchor_input_weights: Mapping[str, torch.Tensor],
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        weights = self.basis_weights(language_slots)
        return (
            self.hidden_residuals(
                probe_value_memory,
                language_slots,
                weights,
                anchor_input_weights=anchor_input_weights,
            ),
            weights,
        )
