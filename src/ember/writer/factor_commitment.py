"""Gate native Action-probe Value with one shared language-video map."""

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


class SharedJointNativeValueGate(torch.nn.Module):
    """Use one joint language-video gate for every frozen factor family."""

    def __init__(self, *, width: int = 256) -> None:
        super().__init__()
        if width <= 0:
            raise WriterModelError("invalid shared native-Value gate topology")
        self.width = int(width)
        self.gate = torch.nn.Linear(width, 2, bias=False)
        torch.nn.init.zeros_(self.gate.weight)
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

    def gate_values(
        self,
        probe_value_memory: torch.Tensor,
        language_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Return two shared coefficients from video-required joint features."""

        if (
            probe_value_memory.ndim != 3
            or probe_value_memory.shape[-1] != self.width
            or language_slots.shape != probe_value_memory.shape
        ):
            raise WriterModelError("invalid shared joint native Value")
        language = self._rms_norm(language_slots)
        return self.gate(probe_value_memory * language)

    def hidden_residuals(
        self,
        probe_value_memory: torch.Tensor,
        language_slots: torch.Tensor,
        *,
        anchor_input_weights: Mapping[str, torch.Tensor],
        gate_values: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """Apply the same direct/signed coefficients to all factor families."""

        if set(anchor_input_weights) != set(FACTOR_FAMILIES):
            raise WriterModelError("factor anchor families changed")
        language = self._rms_norm(language_slots)
        coefficients = (
            self.gate_values(probe_value_memory, language_slots)
            if gate_values is None
            else gate_values
        )
        if coefficients.shape != (*probe_value_memory.shape[:2], 2):
            raise WriterModelError("shared native-Value gate changed shape")
        signed_language = language * self.anchor_input_sign
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
        coefficients = self.gate_values(probe_value_memory, language_slots)
        return (
            self.hidden_residuals(
                probe_value_memory,
                language_slots,
                anchor_input_weights=anchor_input_weights,
                gate_values=coefficients,
            ),
            coefficients,
        )
