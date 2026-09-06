"""Joint process-policy states directly generate the complete task LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.policy_response_writer.capture import FrozenPolicyResponseVideo
from ember.ecp.policy_response_writer.process import (
    JointProcessPolicyBlock,
    PolicyResponseEvidence,
)


@dataclass(frozen=True)
class CompleteLoRAFactors:
    """All target factors; B uses rank-first layout until materialization."""

    a: tuple[torch.Tensor, ...]
    b: tuple[torch.Tensor, ...]


class _FamilyFactorHead(torch.nn.Module):
    """Batch the same target family through shared learned A/B heads."""

    def __init__(
        self, *, width: int, output_width: int, a_template: torch.Tensor
    ) -> None:
        super().__init__()
        self.register_buffer("a_template", a_template)
        self.norm = torch.nn.LayerNorm(width)
        self.a_head = self._head(width, a_template.shape[-1])
        self.b_head = self._head(width, output_width)

    @staticmethod
    def _head(width: int, output_width: int) -> torch.nn.Sequential:
        head = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, output_width, bias=False),
        )
        torch.nn.init.zeros_(head[-1].weight)
        return head

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.norm(states)
        a = self.a_template + self.a_head(states[:, :, 0]).float()
        b = self.b_head(states[:, :, 1]).float()
        return a, b


class CompletePolicyFactorGenerator(torch.nn.Module):
    """One shared P/Q trunk, learned video-set read, and complete factor heads."""

    def __init__(
        self,
        owners: Sequence[TargetOwner],
        *,
        rank: int = 16,
        width: int = 128,
        heads: int = 4,
        block_depth: int = 4,
        process_tokens: int = 8,
        identity_seed: int = 20260721,
    ) -> None:
        super().__init__()
        if not owners or rank <= 0 or width % heads or min(block_depth, process_tokens) <= 0:
            raise ValueError("complete process-policy topology changed")
        self.owners = tuple(owners)
        self.rank = rank
        self.width = width
        self.process_seed = torch.nn.Parameter(torch.empty(process_tokens, width))
        self.owner_embedding = torch.nn.Parameter(torch.empty(len(owners), width))
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), width)
        families = tuple(TargetFamily)
        self.register_buffer("family_ids", torch.tensor([families.index(owner.family) for owner in owners]), persistent=False)
        self.rank_embedding = torch.nn.Parameter(torch.empty(rank, width))
        self.side_embedding = torch.nn.Parameter(torch.empty(2, width))
        for value in (self.process_seed, self.owner_embedding, self.rank_embedding, self.side_embedding):
            torch.nn.init.normal_(value, std=width**-0.5)
        self.blocks = torch.nn.ModuleList(
            JointProcessPolicyBlock(width, heads) for _ in range(block_depth)
        )
        self.set_query_norm = torch.nn.LayerNorm(width)
        self.set_memory_norm = torch.nn.LayerNorm(width)
        self.set_attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.factor_heads = torch.nn.ModuleDict()
        self.family_targets: dict[str, tuple[int, ...]] = {}
        generator = torch.Generator(device="cpu").manual_seed(identity_seed)
        templates = tuple(
            torch.empty(rank, owner.in_features).uniform_(
                -owner.in_features**-0.5, owner.in_features**-0.5, generator=generator
            )
            for owner in owners
        )
        families: dict[str, list[int]] = {}
        for target, owner in enumerate(owners):
            key = f"{owner.family.value}_{owner.in_features}_{owner.out_features}"
            families.setdefault(key, []).append(target)
        for key, targets in families.items():
            self.family_targets[key] = tuple(targets)
            self.register_buffer(f"indices_{key}", torch.tensor(targets), persistent=False)
            self.factor_heads[key] = _FamilyFactorHead(
                width=width,
                output_width=owners[targets[0]].out_features,
                a_template=torch.stack([templates[target] for target in targets]),
            )

    @torch.no_grad()
    def initialize_from_stage0(self, stage0: torch.nn.Module) -> dict[str, object]:
        """Retain the matching original P/Q owner and first attention initialization."""
        binding = stage0.encoder.binding
        self.owner_embedding.copy_(binding.owner_embedding)
        self.family_embedding.weight.copy_(stage0.encoder.observer.projector.family_embedding.weight)
        first = self.blocks[0].policy_attention
        first.in_proj_weight.copy_(torch.cat((
            binding.event_query.weight, binding.policy_key.weight, binding.policy_value.weight
        )))
        first.in_proj_bias.zero_()
        first.out_proj.weight.copy_(torch.eye(self.width, device=first.out_proj.weight.device))
        first.out_proj.bias.zero_()
        return {"reused": ["public_owner_family_embeddings", "first_grounded_full_response_attention"]}

    def _seed(self) -> torch.Tensor:
        return (
            (self.owner_embedding + self.family_embedding(self.family_ids))[:, None, None]
            + self.rank_embedding[None, :, None]
            + self.side_embedding[None, None]
        ).flatten(0, 2)

    def _video_policy(
        self, video: FrozenPolicyResponseVideo, evidence: PolicyResponseEvidence,
        seed: torch.Tensor,
    ) -> torch.Tensor:
        process = self.process_seed[None].expand(video.frame_count, -1, -1)
        policy = seed
        inputs = (
            video.frame_positions.to(process), evidence.patches, evidence.language,
            evidence.language_valid, evidence.response,
        )
        for block in self.blocks:
            if torch.is_grad_enabled():
                process, policy = checkpoint(block, process, policy, *inputs, use_reentrant=False)
            else:
                process, policy = block(process, policy, *inputs)
        return policy

    def forward(
        self, videos: Sequence[FrozenPolicyResponseVideo],
        evidence: Sequence[PolicyResponseEvidence],
    ) -> CompleteLoRAFactors:
        if not videos or len(videos) != len(evidence):
            raise ValueError("complete Writer needs aligned, nonempty video evidence")
        seed = self._seed()
        policies = tuple(
            self._video_policy(video, memory, seed)
            for video, memory in zip(videos, evidence, strict=True)
        )
        # Each independently encoded video contributes equally many policy states.
        # No video identity/position is added at the permutation-invariant set read.
        memory = self.set_memory_norm(torch.cat(policies, dim=0))[None]
        read, _ = self.set_attention(
            self.set_query_norm(seed)[None], memory, memory, need_weights=False
        )
        states = (seed + read[0]).reshape(len(self.owners), self.rank, 2, self.width)
        by_target: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for key, head in self.factor_heads.items():
            indices = getattr(self, f"indices_{key}")
            a, b = head(states.index_select(0, indices))
            for offset, target in enumerate(self.family_targets[key]):
                by_target[target] = (a[offset], b[offset])
        return CompleteLoRAFactors(
            a=tuple(by_target[target][0] for target in range(len(self.owners))),
            b=tuple(by_target[target][1] for target in range(len(self.owners))),
        )
