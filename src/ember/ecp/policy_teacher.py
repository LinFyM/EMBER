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
    """Multi-policy evidence with no task identity or deployment route."""

    member_states: Mapping[str, torch.Tensor]
    phase_response: torch.Tensor
    reliability: torch.Tensor
    policy_response: torch.Tensor
    policy_response_weights: torch.Tensor


@dataclass(frozen=True)
class PolicyTeacherOutput:
    program: ECPProgram
    member_programs: ECPProgram
    member_weights: torch.Tensor
    evidence_gate: torch.Tensor
    evidence_gate_logits: torch.Tensor
    support_attention_entropy: torch.Tensor


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
        support_channels: int = 5,
        support_horizon_basis: int = 4,
    ) -> None:
        super().__init__()
        if width % 2:
            raise ValueError("privileged ECP width must be even")
        self.owners = owners
        self.contract = contract
        self.width = width
        self.event_slots = event_slots
        self.rank = int(contract.rank)
        self.support_channels = support_channels
        self.support_horizon_basis = support_horizon_basis
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
        self.reliability_projection = torch.nn.Linear(1, width, bias=False)
        self.support_key = torch.nn.Linear(width, width, bias=False)
        self.support_value = torch.nn.Linear(width, width, bias=False)
        self.support_query = torch.nn.Linear(2 * width, width, bias=False)
        self.support_channel_embedding = torch.nn.Embedding(
            support_channels, width
        )
        self.support_basis_embedding = torch.nn.Embedding(
            support_horizon_basis, width
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(5 * width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
            torch.nn.LayerNorm(width),
        )
        self.evidence_gate = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.evidence_gate.bias)

    def _support_tokens(
        self,
        *,
        anchor: torch.Tensor,
        phase_owner: torch.Tensor,
        response: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        members = anchor.shape[0]
        expected = (
            members,
            self.event_slots,
            len(self.owners),
            self.support_channels,
            self.support_horizon_basis,
            self.width,
        )
        if (
            response.shape != expected
            or weights.shape
            != (members, self.event_slots, self.support_channels)
            or bool((weights.sum(dim=-1) <= 0).any())
        ):
            raise ValueError("privileged policy-support response changed shape")
        content = response.float()
        values = self.support_value(content)
        channel = self.support_channel_embedding.weight[
            None, None, None, :, None
        ]
        basis = self.support_basis_embedding.weight[None, None, None, None]
        keys = self.support_key(content) + channel + basis
        query = self.support_query(torch.cat((anchor, phase_owner), dim=-1))
        scores = torch.einsum("meocpd,meod->meocp", keys, query)
        scores = scores / math.sqrt(self.width)
        expanded_weights = weights[:, :, None, :, None]
        scores = scores + expanded_weights.clamp_min(1e-8).log()
        scores = scores.masked_fill(expanded_weights <= 0, torch.finfo(scores.dtype).min)
        attention = scores.flatten(-2).softmax(-1).reshape_as(scores)
        support = torch.einsum("meocp,meocpd->meod", attention, values)
        entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=(-1, -2))
        return support, entropy

    def _factor_tokens(
        self, states: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        rows = []
        member_count = -1
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
            rows.append(self.factor_norm(token))
        return torch.stack(rows, dim=1)

    def forward(
        self,
        anchors: ECPProgram,
        evidence: PrivilegedPolicyEvidence,
        *,
        evidence_logit_offset: torch.Tensor | None = None,
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
        support, support_entropy = self._support_tokens(
            anchor=anchor,
            phase_owner=phase_owner,
            response=evidence.policy_response,
            weights=evidence.policy_response_weights,
        )
        correction = self.fusion(
            torch.cat(
                (
                    anchor,
                    factor_event,
                    phase_owner,
                    support,
                    reliability_owner,
                ),
                dim=-1,
            )
        )
        evidence_gate_logits = self.evidence_gate(correction)
        expected_offset = (1, self.event_slots, len(self.owners), 1)
        if (
            evidence_logit_offset is not None
            and evidence_logit_offset.shape != expected_offset
        ):
            raise ValueError("q_pi evidence-logit offset changed shape")
        if evidence_logit_offset is not None:
            evidence_gate_logits = (
                evidence_gate_logits + evidence_logit_offset.to(evidence_gate_logits)
            )
        evidence_gate = evidence_gate_logits.sigmoid()
        visible = anchors.presence[0][None, :, None, None]
        member_process = anchor + visible * evidence_gate * correction
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
            evidence_gate=evidence_gate,
            evidence_gate_logits=evidence_gate_logits,
            support_attention_entropy=support_entropy,
        )
