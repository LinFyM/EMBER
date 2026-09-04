"""Canonical Policy-Response Event-to-Factor Writer graph."""

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
from ember.ecp.policy_response_writer.composer import CurrentVideoNativeFactorComposer
from ember.ecp.policy_response_writer.process import (
    PolicyResponseProcessEncoder,
    PolicyResponseProcessOutput,
)
from ember.lora import LoRAContract


@dataclass(frozen=True)
class PolicyResponseWriterOutput:
    residual: NativeFactorResidual
    processes: tuple[PolicyResponseProcessOutput, ...]


class PolicyResponseEventToFactorWriter(torch.nn.Module):
    """One process encoder and one native composer, scaled by repeated blocks."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        prefix_width: int = 2048,
        expert_width: int = 1024,
        width: int = 128,
        event_slots: int = 8,
        heads: int = 4,
        frame_blocks: int = 2,
        event_blocks: int = 2,
        composer_blocks: int = 2,
        composer_gain_blocks: int = 1,
        pooling_frame_chunk: int = 4,
        task_local: bool = False,
    ) -> None:
        super().__init__()
        self.owners = tuple(owners)
        self.process = PolicyResponseProcessEncoder(
            owners,
            prefix_width=prefix_width,
            expert_width=expert_width,
            width=width,
            event_slots=event_slots,
            heads=heads,
            frame_blocks=frame_blocks,
            event_blocks=event_blocks,
        )
        self.composer = CurrentVideoNativeFactorComposer(
            owners,
            width=width,
            heads=heads,
            block_depth=composer_blocks,
            gain_block_depth=composer_gain_blocks,
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
        processes = tuple(
            self.process(video, representation=representation) for video in values
        )
        residual = self.composer(values, processes, s_ref=s_ref)
        return PolicyResponseWriterOutput(residual=residual, processes=processes)

    def causal_prediction_loss(
        self,
        videos: Sequence[FrozenPolicyResponseVideo],
        *,
        cutoffs: Sequence[Sequence[int]],
        future_offsets: Sequence[int],
        representation: str = "full",
    ) -> torch.Tensor:
        values = tuple(videos)
        selected = tuple(tuple(map(int, row)) for row in cutoffs)
        offsets = tuple(map(int, future_offsets))
        if len(values) != len(selected) or len(values) != len(offsets) or not values:
            raise ValueError("policy-response causal video set changed")
        return torch.stack(
            tuple(
                self.process.causal_prediction_loss(
                    video,
                    cutoffs=rows,
                    future_offset=offset,
                    representation=representation,
                )
                for video, rows, offset in zip(
                    values, selected, offsets, strict=True
                )
            )
        ).mean()

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
