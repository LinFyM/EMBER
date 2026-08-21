"""Visible-event-anchored privileged policy teacher for ECP Stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.program import ECPProgram
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


@dataclass(frozen=True)
class PrivilegedPolicyEvidence:
    """Successful-policy evidence with no task identity or deployment route."""

    member_states: Mapping[str, torch.Tensor]
    phase_response: torch.Tensor
    reliability: torch.Tensor


@dataclass(frozen=True)
class PolicyTeacherOutput:
    program: ECPProgram
    member_programs: ECPProgram
    member_weights: torch.Tensor


def _family_name(family: TargetFamily) -> str:
    return str(family.value)


class PrivilegedPolicyTeacher(torch.nn.Module):
    """Attach successful adapters and trajectory response to visible event slots."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        contract: LoRAContract,
        *,
        width: int = 128,
        phase_width: int = 32,
        event_slots: int = 8,
    ) -> None:
        super().__init__()
        if width % 2:
            raise ValueError("privileged ECP width must be even")
        self.owners = owners
        self.contract = contract
        self.width = width
        self.event_slots = event_slots
        self.rank = int(contract.rank)
        half = width // 2
        family_widths: dict[str, tuple[int, int]] = {}
        for owner in owners:
            family_widths.setdefault(
                _family_name(owner.family),
                (owner.in_features, owner.out_features),
            )
        self.factor_a = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(in_features, half, bias=False)
                for name, (in_features, _) in family_widths.items()
            }
        )
        self.factor_b = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(out_features, half, bias=False)
                for name, (_, out_features) in family_widths.items()
            }
        )
        self.factor_norm = torch.nn.LayerNorm(width)
        self.phase_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(phase_width),
            torch.nn.Linear(phase_width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, width),
        )
        self.rank_embedding = torch.nn.Embedding(self.rank, width)
        self.owner_embedding = torch.nn.Embedding(len(owners), width)
        self.reliability_projection = torch.nn.Linear(1, width, bias=False)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(4 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.1))

    def _factor_tokens(
        self, states: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        rows = []
        member_count = -1
        rank_bias = self.rank_embedding.weight[None]
        for owner in self.owners:
            a = states[owner.target_name + LORA_A_SUFFIX]
            b = states[owner.target_name + LORA_B_SUFFIX]
            if (
                a.ndim != 3
                or b.ndim != 3
                or a.shape[1:] != (self.rank, owner.in_features)
                or b.shape[1:] != (owner.out_features, self.rank)
                or (member_count >= 0 and a.shape[0] != member_count)
            ):
                raise ValueError("privileged successful-adapter tensors changed shape")
            member_count = int(a.shape[0])
            family = _family_name(owner.family)
            token = torch.cat(
                (
                    self.factor_a[family](a.float()),
                    self.factor_b[family](b.transpose(1, 2).float()),
                ),
                dim=-1,
            )
            owner_bias = self.owner_embedding.weight[owner.index][None, None]
            rows.append(self.factor_norm(token + rank_bias + owner_bias))
        return torch.stack(rows, dim=1)

    def forward(
        self,
        anchors: ECPProgram,
        evidence: PrivilegedPolicyEvidence,
    ) -> PolicyTeacherOutput:
        if anchors.language.shape[0] != 1:
            raise ValueError("q_pi consumes one task-level visible Program at a time")
        factors = self._factor_tokens(evidence.member_states)
        members = factors.shape[0]
        if (
            evidence.phase_response.shape
            != (members, self.event_slots, self.phase_projection[0].normalized_shape[0])
            or evidence.reliability.shape != (members,)
            or members <= 0
        ):
            raise ValueError("privileged policy response evidence changed shape")
        phase = self.phase_projection(evidence.phase_response.float())
        scores = torch.einsum("med,mjrd->mejr", phase, factors) / math.sqrt(
            self.width
        )
        factor_event = torch.einsum(
            "mejr,mjrd->mejd", scores.softmax(-1), factors
        )
        reliability_feature = self.reliability_projection(
            evidence.reliability.float()[:, None]
        )[:, None, None]
        anchor = anchors.process[0][None].expand(members, -1, -1, -1)
        phase_owner = phase[:, :, None].expand(-1, -1, len(self.owners), -1)
        reliability_owner = reliability_feature.expand_as(phase_owner)
        correction = self.fusion(
            torch.cat(
                (anchor, factor_event, phase_owner, reliability_owner), dim=-1
            )
        )
        gate = anchors.presence[0][None, :, None, None]
        member_process = anchor + self.residual_scale.tanh() * gate * correction
        weights = evidence.reliability.float().clamp_min(1e-3)
        weights = weights / weights.sum()
        mean = torch.einsum("m,mejd->ejd", weights, member_process)
        disagreement = torch.einsum(
            "m,mejd->ejd", weights, (member_process.float() - mean).square()
        )
        uncertainty = (
            anchors.uncertainty[0].float().square() + disagreement
        ).clamp_min(1e-4).sqrt()
        consensus = ECPProgram(
            language=anchors.language,
            scene=anchors.scene,
            process=mean[None],
            presence=anchors.presence,
            uncertainty=uncertainty[None],
        )
        member_programs = ECPProgram(
            language=anchors.language.expand(members, -1, -1),
            scene=anchors.scene.expand(members, -1, -1),
            process=member_process,
            presence=anchors.presence.expand(members, -1),
            uncertainty=uncertainty[None].expand(members, -1, -1, -1),
        )
        return PolicyTeacherOutput(
            program=consensus,
            member_programs=member_programs,
            member_weights=weights,
        )
