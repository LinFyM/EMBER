"""Canonical unified Policy-Native Factor Writer graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import NativeFactorResidual
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.composer import (
    UnifiedPolicyNativeFactorGenerator,
)
from ember.ecp.policy_response_writer.process import PolicyResponseEvidenceEncoder
from ember.lora import LoRAContract


@dataclass(frozen=True)
class PolicyResponseWriterOutput:
    residual: NativeFactorResidual


class UnifiedPolicyNativeFactorWriter(torch.nn.Module):
    """Input tokenizers plus one repeated factor-latent block family."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
        heads: int = 4,
        blocks: int = 4,
        pooling_frame_chunk: int = 4,
        task_local: bool = False,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.evidence = PolicyResponseEvidenceEncoder(
            owners,
            prefix_width=prefix_width,
            expert_width=expert_width,
            width=width,
        )
        self.factor_writer = UnifiedPolicyNativeFactorGenerator(
            owners,
            width=width,
            heads=heads,
            block_depth=blocks,
            pooling_frame_chunk=pooling_frame_chunk,
            task_local=task_local,
        )

    def forward(
        self,
        videos: Sequence[FrozenPolicyResponseVideo],
        *,
        s_ref: torch.Tensor,
        representation: str = "full",
    ) -> PolicyResponseWriterOutput:
        values = tuple(videos)
        evidence = tuple(
            self.evidence(video, representation=representation) for video in values
        )
        residual = self.factor_writer(values, evidence, s_ref=s_ref)
        return PolicyResponseWriterOutput(residual=residual)

    @staticmethod
    def materialize(
        output: PolicyResponseWriterOutput,
        *,
        carrier_state: Mapping[str, torch.Tensor],
        rank4_contract: LoRAContract,
        rank16_contract: LoRAContract,
        canonicalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        residual = residual_lora_state(
            output.residual, rank4_contract, canonicalize=canonicalize
        )
        return compose_rank12_plus_rank4(
            carrier_state=carrier_state,
            residual_state=residual,
            rank16_contract=rank16_contract,
        )

    @torch.no_grad()
    def initialize_from_stage0(self, stage0: torch.nn.Module) -> dict[str, object]:
        evidence = self.evidence.initialize_from_stage0(stage0)
        factor = self.factor_writer.initialize_from_stage0(stage0)
        return {
            "kind": "g2_native_projection_initialization",
            "reused": [*evidence["reused"], *factor["reused"]],
            "fresh": [
                "unified_policy_native_factor_blocks",
                "factor_side_signed_heads",
            ],
        }
