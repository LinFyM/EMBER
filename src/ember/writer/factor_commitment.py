"""Video-required fixed-A native-B residuals for the LPCP Writer."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from ember.writer.errors import WriterModelError


FACTOR_WIDTHS = {
    "q_a": 1024,
    "q_b": 2048,
    "v_a": 1024,
    "v_b": 256,
    "action_in_a": 32,
    "action_in_b": 1024,
    "action_out_a": 1024,
    "action_out_b": 32,
}
FACTOR_FAMILIES = tuple(FACTOR_WIDTHS)
NATIVE_B_WIDTHS = {
    family: width for family, width in FACTOR_WIDTHS.items() if family.endswith("_b")
}
NATIVE_B_FAMILIES = tuple(NATIVE_B_WIDTHS)


class AnchoredLinearNativeBResidual(torch.nn.Module):
    """Write joint video-language Value only into the LPCP-anchored B side."""

    EXPERT_SLOTS = 18 * 16
    ACTION_SLOTS = 16

    def __init__(self, *, width: int = 256) -> None:
        super().__init__()
        if width <= 0:
            raise WriterModelError("invalid anchored native-B topology")
        self.width = int(width)
        self.heads = torch.nn.ModuleDict(
            {
                family: torch.nn.Linear(width, output_width, bias=False)
                for family, output_width in NATIVE_B_WIDTHS.items()
            }
        )
        for head in self.heads.values():
            torch.nn.init.zeros_(head.weight)

    @staticmethod
    def _rms_norm(value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.to(torch.float32).square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(value.dtype)
        return value * scale

    def joint_value(
        self,
        probe_value_memory: torch.Tensor,
        language_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Return the video-required slot payload used by every direct head."""

        if (
            probe_value_memory.ndim != 3
            or probe_value_memory.shape[1:] != (320, self.width)
            or language_slots.shape != probe_value_memory.shape
        ):
            raise WriterModelError("invalid direct joint native Value")
        return (
            probe_value_memory * self._rms_norm(language_slots)
        ) / math.sqrt(self.width)

    def forward(
        self,
        probe_value_memory: torch.Tensor,
        language_slots: torch.Tensor,
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        """Emit layer/rank-owned B rows while the LPCP A side stays fixed."""

        joint = self.joint_value(probe_value_memory, language_slots)
        expert = joint[:, : self.EXPERT_SLOTS].reshape(
            joint.shape[0], 18, 16, self.width
        )
        action_in = joint[
            :, self.EXPERT_SLOTS : self.EXPERT_SLOTS + self.ACTION_SLOTS
        ]
        action_out = joint[:, -self.ACTION_SLOTS :]
        rows = {
            "q_b": self.heads["q_b"](expert),
            "v_b": self.heads["v_b"](expert),
            "action_in_b": self.heads["action_in_b"](action_in),
            "action_out_b": self.heads["action_out_b"](action_out),
        }
        return rows, joint
