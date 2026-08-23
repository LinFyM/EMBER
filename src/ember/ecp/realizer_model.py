"""Small target-local decoder for frozen ECP effect codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.lora import LoRAContract


@dataclass(frozen=True)
class RealizerLoss:
    total: torch.Tensor
    factor: torch.Tensor
    effective: torch.Tensor
    null: torch.Tensor


class TargetFactorHead(nn.Module):
    def __init__(
        self,
        *,
        state_width: int,
        bottleneck: int,
        in_features: int,
        out_features: int,
        a_scale: float,
        b_scale: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.coefficients = nn.Sequential(
            nn.LayerNorm(state_width),
            nn.Linear(state_width, bottleneck),
            nn.GELU(),
        )
        self.a_basis = nn.Parameter(torch.empty(bottleneck, 4 * in_features))
        self.b_basis = nn.Parameter(torch.empty(bottleneck, out_features * 4))
        self.a_bias = nn.Parameter(torch.zeros(4 * in_features))
        self.b_bias = nn.Parameter(torch.zeros(out_features * 4))
        self.register_buffer("a_scale", torch.tensor(float(a_scale)))
        self.register_buffer("b_scale", torch.tensor(float(b_scale)))
        nn.init.normal_(self.a_basis, std=0.02 / bottleneck**0.5)
        nn.init.normal_(self.b_basis, std=0.02 / bottleneck**0.5)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.coefficients(state)
        a = (coefficients @ self.a_basis + self.a_bias) * self.a_scale
        b = (coefficients @ self.b_basis + self.b_bias) * self.b_scale
        return (
            a.reshape(state.shape[0], 4, self.in_features),
            b.reshape(state.shape[0], self.out_features, 4),
        )


class FixedEffectRealizer(nn.Module):
    """Preserve event/particle/owner structure before target-local decoding."""

    def __init__(
        self,
        *,
        contract: LoRAContract,
        owners: Sequence[TargetOwner],
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        token_width: int = 128,
        state_width: int = 256,
        bottleneck: int = 32,
    ) -> None:
        super().__init__()
        if (
            len(owners) != len(contract.targets)
            or len(owners) != 38
            or tuple(owner.index for owner in owners) != tuple(range(38))
            or a_scales.shape != (38,)
            or b_scales.shape != (38,)
            or token_width != 128
            or state_width != 256
        ):
            raise ValueError("fixed effect realizer topology changed")
        self.token_width = token_width
        self.event_embedding = nn.Parameter(torch.empty(8, token_width))
        self.owner_embedding = nn.Parameter(torch.empty(38, token_width))
        family_index = {family: index for index, family in enumerate(TargetFamily)}
        self.family_embedding = nn.Embedding(len(family_index), token_width)
        self.layer_embedding = nn.Embedding(19, token_width)
        self.register_buffer(
            "owner_family_ids",
            torch.tensor([family_index[owner.family] for owner in owners]),
        )
        self.register_buffer(
            "owner_layer_ids",
            torch.tensor([18 if owner.layer is None else owner.layer for owner in owners]),
        )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_width),
            nn.Linear(token_width, token_width),
            nn.GELU(),
            nn.Linear(token_width, token_width),
        )
        self.event_attention = nn.MultiheadAttention(
            token_width, num_heads=4, batch_first=True
        )
        self.event_score = nn.Linear(token_width, 1)
        self.particle_score = nn.Linear(token_width, 1)
        self.reliability = nn.Sequential(nn.Linear(1, state_width), nn.Tanh())
        self.owner_state = nn.Sequential(
            nn.LayerNorm(2 * state_width),
            nn.Linear(2 * state_width, state_width),
            nn.GELU(),
            nn.Linear(state_width, state_width),
        )
        self.heads = nn.ModuleList(
            TargetFactorHead(
                state_width=state_width,
                bottleneck=bottleneck,
                in_features=target.in_features,
                out_features=target.out_features,
                a_scale=float(a_scales[index]),
                b_scale=float(b_scales[index]),
            )
            for index, target in enumerate(contract.targets)
        )
        nn.init.normal_(self.event_embedding, std=0.02)
        nn.init.normal_(self.owner_embedding, std=0.02)
        nn.init.normal_(self.family_embedding.weight, std=0.02)
        nn.init.normal_(self.layer_embedding.weight, std=0.02)

    def encode(
        self,
        effect_code: torch.Tensor,
        particle_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if (
            effect_code.ndim != 5
            or effect_code.shape[2:] != (8, 38, 128)
            or particle_mask.shape != effect_code.shape[:2]
            or reliability.shape != (effect_code.shape[0],)
            or not particle_mask.any(1).all()
        ):
            raise ValueError("fixed effect code batch changed")
        batch, particles = effect_code.shape[:2]
        tokens = effect_code.float()
        tokens = tokens + self.event_embedding[None, None, :, None]
        tokens = tokens + self.owner_embedding[None, None, None]
        tokens = tokens + self.family_embedding(self.owner_family_ids)[None, None, None]
        tokens = tokens + self.layer_embedding(self.owner_layer_ids)[None, None, None]
        tokens = self.token_mlp(tokens)
        sequences = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch * particles * 38, 8, 128
        )
        attended, _ = self.event_attention(
            sequences, sequences, sequences, need_weights=False
        )
        event_weights = self.event_score(attended).softmax(dim=1)
        per_particle = (attended * event_weights).sum(1).reshape(
            batch, particles, 38, 128
        )
        scores = self.particle_score(per_particle).squeeze(-1)
        scores = scores.masked_fill(~particle_mask[:, :, None], -torch.inf)
        weights = scores.softmax(dim=1)
        mean = (per_particle * weights[..., None]).sum(1)
        variance = (
            (per_particle - mean[:, None]).square() * weights[..., None]
        ).sum(1)
        particle_state = torch.cat((mean, variance.clamp_min(1e-8).sqrt()), dim=-1)
        global_state = particle_state.mean(1, keepdim=True).expand(-1, 38, -1)
        state = self.owner_state(torch.cat((particle_state, global_state), dim=-1))
        return state + self.reliability(reliability[:, None])[:, None]

    def forward(
        self,
        effect_code: torch.Tensor,
        particle_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        state = self.encode(effect_code, particle_mask, reliability)
        return tuple(head(state[:, index]) for index, head in enumerate(self.heads))


def effective_inner_product(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    left_a, left_b = left
    right_a, right_b = right
    left_a = left_a.float()
    left_b = left_b.float()
    right_a = right_a.float()
    right_b = right_b.float()
    return torch.sum(
        (left_b.transpose(1, 2) @ right_b)
        * (left_a @ right_a.transpose(1, 2)),
        dim=(1, 2),
    )


def fixed_effect_realizer_loss(
    *,
    prediction: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target: Sequence[tuple[torch.Tensor, torch.Tensor]],
    null_prediction: Sequence[tuple[torch.Tensor, torch.Tensor]],
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    null_weight: float,
) -> RealizerLoss:
    if (
        not prediction
        or len(prediction) != len(target)
        or len(prediction) != len(null_prediction)
    ):
        raise ValueError("fixed effect realizer loss targets changed")
    factors, effects, nulls = [], [], []
    for index, (predicted, expected, zero) in enumerate(
        zip(prediction, target, null_prediction, strict=True)
    ):
        pa, pb = predicted
        ta, tb = expected
        za, zb = zero
        factors.append(
            (pa.float() - ta.float()).square().mean()
            / a_scales[index].square().clamp_min(1e-12)
            + (pb.float() - tb.float()).square().mean()
            / b_scales[index].square().clamp_min(1e-12)
        )
        target_energy = effective_inner_product(expected, expected).clamp_min(1e-8)
        error = (
            effective_inner_product(predicted, predicted)
            + target_energy
            - 2.0 * effective_inner_product(predicted, expected)
        ).clamp_min(0)
        effects.append((error / target_energy).mean())
        nulls.append(
            za.float().square().mean()
            / a_scales[index].square().clamp_min(1e-12)
            + zb.float().square().mean()
            / b_scales[index].square().clamp_min(1e-12)
        )
    factor = torch.stack(factors).mean()
    effective = torch.stack(effects).mean()
    null = torch.stack(nulls).mean()
    return RealizerLoss(
        total=factor + effective + float(null_weight) * null,
        factor=factor,
        effective=effective,
        null=null,
    )
