"""Canonical native-temporal Policy-Response Writer graph."""

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
from ember.ecp.policy_response_writer.composer import NativeTemporalFactorComposer
from ember.ecp.policy_response_writer.process import (
    PolicyResponseFrameEncoder,
    PolicyResponseFrameOutput,
)
from ember.lora import LoRAContract


@dataclass(frozen=True)
class PolicyResponseWriterOutput:
    residual: NativeFactorResidual
    frames: tuple[PolicyResponseFrameOutput, ...]


class PolicyResponseNativeTemporalWriter(torch.nn.Module):
    """One frame encoder and one native-temporal factor composer."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
        heads: int = 4,
        frame_blocks: int = 2,
        factor_blocks: int = 2,
        pooling_frame_chunk: int = 4,
        task_local: bool = False,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.process = PolicyResponseFrameEncoder(
            owners,
            prefix_width=prefix_width,
            expert_width=expert_width,
            width=width,
            heads=heads,
            frame_blocks=frame_blocks,
        )
        self.composer = NativeTemporalFactorComposer(
            owners,
            width=width,
            heads=heads,
            block_depth=factor_blocks,
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
        frames = tuple(
            self.process(video, representation=representation) for video in values
        )
        residual = self.composer(values, frames, s_ref=s_ref)
        return PolicyResponseWriterOutput(residual=residual, frames=frames)

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
        report = self.process.initialize_from_stage0(stage0)
        binding = stage0.encoder.binding
        projector = stage0.encoder.observer.projector
        self.composer.owner_embedding.copy_(binding.owner_embedding)
        self.composer.family_embedding.weight.copy_(projector.family_embedding.weight)
        self.composer.horizon_embedding.weight.copy_(binding.horizon_embedding)
        return {
            **report,
            "composer_reused": [
                "owner_embedding",
                "family_embedding",
                "horizon_embedding",
            ],
        }
