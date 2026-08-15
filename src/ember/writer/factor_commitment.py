"""Select native Action-probe Value through fixed language pre-addresses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from ember.writer.errors import WriterModelError


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


class PreAddressedFactorSelectiveNativeValue(torch.nn.Module):
    """Let each factor family select video Value under a fixed text address."""

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
        self.selectors = torch.nn.ParameterDict(
            {
                family: torch.nn.Parameter(
                    torch.zeros(basis_count, 2, width)
                )
                for family in FACTOR_FAMILIES
            }
        )
        basis_keys = torch.empty(basis_count, width)
        generator = torch.Generator(device="cpu").manual_seed(
            int(initialization_seed) + 0x53464D43
        )
        basis_keys.normal_(mean=0.0, std=0.02, generator=generator)
        self.register_buffer("basis_keys", basis_keys)
        self.register_buffer(
            "anchor_input_sign",
            torch.where(torch.arange(width) % 2 == 0, 1, -1).to(torch.int8),
            persistent=False,
        )

    @staticmethod
    def _rms_norm(value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(value.dtype)
        return value * scale

    def basis_weights(self, language_slots: torch.Tensor) -> torch.Tensor:
        """Return a frozen language-only pre-address for every policy slot."""

        if language_slots.ndim != 3 or language_slots.shape[-1] != self.width:
            raise WriterModelError("semantic address lost its policy-slot layout")
        query = self._rms_norm(language_slots)
        keys = self._rms_norm(self.basis_keys)
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
        """Select two video coefficients independently for every factor family."""

        if (
            probe_value_memory.ndim != 3
            or probe_value_memory.shape[-1] != self.width
            or language_slots.shape != probe_value_memory.shape
            or basis_weights.shape
            != (*probe_value_memory.shape[:2], self.basis_count)
        ):
            raise WriterModelError("invalid pre-addressed factor-selective Value")
        if set(anchor_input_weights) != set(FACTOR_FAMILIES):
            raise WriterModelError("factor anchor families changed")
        language = self._rms_norm(language_slots)
        signed_language = language * self.anchor_input_sign
        selector_input = probe_value_memory * language
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
            per_basis = torch.einsum(
                "bsj,koj->bsko", selector_input, self.selectors[family]
            )
            coefficients = (
                basis_weights[..., None] * per_basis
            ).sum(dim=-2)
            result[family] = (
                coefficients[..., :1] * direct_axis
                + coefficients[..., 1:] * signed_axis
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
