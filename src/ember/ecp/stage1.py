"""Canonical Stage 1 graph: visible anchors, q_pi, and one complete compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.compiler import ECPCompilerOutput, LayerResolvedCompiler
from ember.ecp.contracts import TargetOwner
from ember.ecp.policy_teacher import (
    PolicyTeacherOutput,
    PrivilegedPolicyEvidence,
    PrivilegedPolicyTeacher,
)
from ember.ecp.program import ECPProgram, VisibleProgramProjector
from ember.ecp.stage0 import ECPVideoEncoderOutput
from ember.lora import LoRAContract


@dataclass(frozen=True)
class ECPStage1Output:
    anchors: ECPProgram
    teacher: PolicyTeacherOutput
    member_compilation: ECPCompilerOutput
    consensus_compilation: ECPCompilerOutput
    prior_compilation: ECPCompilerOutput


class ECPStage1Model(torch.nn.Module):
    """Train-only privileged teacher paired with the deployment compiler."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        contract: LoRAContract,
        template_state: Mapping[str, torch.Tensor],
        *,
        program_width: int = 128,
        compiler_width: int = 256,
        event_slots: int = 8,
        phase_width: int = 32,
        support_channels: int = 5,
        support_horizon_basis: int = 4,
        factor_head_init: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self.visible_program = VisibleProgramProjector(
            owners, width=program_width, event_slots=event_slots
        )
        self.policy_teacher = PrivilegedPolicyTeacher(
            owners,
            contract,
            width=program_width,
            phase_width=phase_width,
            event_slots=event_slots,
            support_channels=support_channels,
            support_horizon_basis=support_horizon_basis,
        )
        self.compiler = LayerResolvedCompiler(
            owners,
            contract,
            template_state,
            program_width=program_width,
            compiler_width=compiler_width,
            event_slots=event_slots,
            factor_head_init=factor_head_init,
        )

    def trainable_q_pi_compiler_parameters(
        self,
    ) -> tuple[torch.nn.Parameter, ...]:
        """Enable exactly the two shared Stage 1 mapping owners."""

        self.policy_teacher.requires_grad_(True)
        self.compiler.requires_grad_(True)
        parameters = tuple(
            (*self.policy_teacher.parameters(), *self.compiler.parameters())
        )
        ownership_valid = (
            bool(parameters)
            and all(parameter.requires_grad for parameter in parameters)
            and not any(
                parameter.requires_grad
                for parameter in self.visible_program.parameters()
            )
        )
        if not ownership_valid:
            raise ValueError("mapping-diverse q_pi/compiler ownership changed")
        return parameters

    def forward(
        self,
        encoded: ECPVideoEncoderOutput,
        evidence: PrivilegedPolicyEvidence,
        video_group_ids: torch.Tensor,
    ) -> ECPStage1Output:
        anchors = self.visible_program(encoded, video_group_ids, group_count=1)
        teacher = self.policy_teacher(anchors, evidence)
        return ECPStage1Output(
            anchors=anchors,
            teacher=teacher,
            member_compilation=self.compiler(teacher.member_programs),
            consensus_compilation=self.compiler(teacher.program),
            prior_compilation=self.compiler(anchors.prior_only()),
        )
