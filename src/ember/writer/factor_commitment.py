"""Language-addressed commitment of video memory into V6 factor families."""

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


class SemanticFactorMemoryCommitment(torch.nn.Module):
    """Route one dynamic K-set memory through shared factor-family maps."""

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
        self.semantic_query.weight.data.normal_(
            mean=0.0, std=0.02, generator=generator
        )
        self.basis_keys.data.normal_(mean=0.0, std=0.02, generator=generator)

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
    ) -> Mapping[str, torch.Tensor]:
        """Map one video-derived Value into all eight frozen factor heads."""

        if (
            factor_memory.ndim != 3
            or factor_memory.shape[-1] != self.width
            or basis_weights.shape
            != (*factor_memory.shape[:2], self.basis_count)
        ):
            raise WriterModelError("invalid semantic factor-memory commitment")
        return {
            family: torch.einsum(
                "bski,bsk->bsi",
                torch.einsum("bsj,kij->bski", factor_memory, maps),
                basis_weights,
            )
            for family, maps in self.family_maps.items()
        }

    def forward(
        self,
        factor_memory: torch.Tensor,
        language_slots: torch.Tensor,
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        weights = self.basis_weights(language_slots)
        return self.hidden_residuals(factor_memory, weights), weights
