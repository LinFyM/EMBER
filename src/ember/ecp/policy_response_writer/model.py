"""Canonical complete LoRA Writer over frozen full policy-response evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.contracts import TargetOwner
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.composer import (
    CompleteLoRAFactors,
    CompletePolicyFactorGenerator,
)
from ember.ecp.policy_response_writer.process import PolicyResponseEvidenceEncoder
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract


@dataclass(frozen=True)
class PolicyResponseWriterOutput:
    factors: CompleteLoRAFactors


class CompletePolicyResponseWriter(torch.nn.Module):
    """Task-grounded video understanding and joint, complete policy generation."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        rank: int = 16,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
        heads: int = 4,
        blocks: int = 4,
        process_tokens: int = 8,
        identity_seed: int = 20260721,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.evidence = PolicyResponseEvidenceEncoder(
            owners, prefix_width=prefix_width, expert_width=expert_width, width=width
        )
        self.factor_writer = CompletePolicyFactorGenerator(
            owners, rank=rank, width=width, heads=heads, block_depth=blocks,
            process_tokens=process_tokens, identity_seed=identity_seed,
        )

    def forward(
        self, videos: Sequence[FrozenPolicyResponseVideo], *, representation: str = "full"
    ) -> PolicyResponseWriterOutput:
        values = tuple(videos)
        evidence = tuple(self.evidence(video, representation=representation) for video in values)
        return PolicyResponseWriterOutput(factors=self.factor_writer(values, evidence))

    @staticmethod
    def materialize(
        output: PolicyResponseWriterOutput, *, contract: LoRAContract
    ) -> dict[str, torch.Tensor]:
        factors = output.factors
        if len(factors.a) != len(contract.targets) or len(factors.b) != len(contract.targets):
            raise ValueError("complete Writer target count differs from the execution contract")
        state = {}
        for target, a, b in zip(contract.targets, factors.a, factors.b, strict=True):
            if a.shape != (contract.rank, target.in_features) or b.shape != (
                contract.rank, target.out_features
            ):
                raise ValueError(f"complete factor shape changed: {target.name}")
            state[target.name + LORA_A_SUFFIX] = a
            state[target.name + LORA_B_SUFFIX] = b.transpose(0, 1)
        return state

    @torch.no_grad()
    def initialize_from_stage0(self, stage0: torch.nn.Module) -> dict[str, object]:
        evidence = self.evidence.initialize_from_stage0(stage0)
        factor = self.factor_writer.initialize_from_stage0(stage0)
        return {
            "kind": "g2_evidence_projection_initialization",
            "reused": [*evidence["reused"], *factor["reused"]],
            "fresh": ["remaining_joint_process_policy_parameters", "complete_factor_heads"],
        }
