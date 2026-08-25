"""Verified set-valued policy-effect critic for G3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.contracts import TargetOwner
from ember.ecp.g1_objective import (
    sensitivity_normalized_update_losses,
)
from ember.ecp.policy_effects import ExecutionPolicyPrefix, PolicyEffectResponse
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_source_checkpoint import read_json


G3_EFFECT_BANK_SCHEMA = "ember_ecp_g3_verified_effect_bank_v1"


@dataclass(frozen=True)
class SharedCompilerEffectBank:
    prefix: ExecutionPolicyPrefix
    suffix_noise: torch.Tensor
    validity: torch.Tensor
    trajectory_ids: torch.Tensor
    carrier: PolicyEffectResponse
    members: PolicyEffectResponse
    reliability: torch.Tensor
    family_weights: torch.Tensor
    member_names: tuple[str, ...]
    projections: tuple[Mapping[str, torch.Tensor], ...]
    anchors: tuple[dict[str, Any], ...]

    @property
    def state_count(self) -> int:
        return int(self.suffix_noise.shape[0])

    @property
    def member_count(self) -> int:
        return int(self.reliability.numel())

    def validate(self, owners: Sequence[TargetOwner]) -> None:
        members = self.member_count
        states = self.state_count
        if (
            members not in (1, 2)
            or states != 4 * members
            or len(self.member_names) != members
            or len(self.projections) != members
            or len(self.anchors) != states
            or self.prefix.embeddings.shape[0] != states
            or self.prefix.padding.shape != self.prefix.embeddings.shape[:2]
            or self.suffix_noise.shape != (states, 50, 32)
            or self.validity.shape != (members, states)
            or self.validity.sum(1).tolist() != [4] * members
            or self.trajectory_ids.shape != (states,)
            or self.carrier.owner.shape != (states, 38, 4, 128)
            or self.members.owner.shape != (members, states, 38, 4, 128)
            or self.carrier.flow.shape != (states, 10, 50, 32)
            or self.members.flow.shape != (members, states, 10, 50, 32)
            or self.carrier.action.shape != (states, 10, 50, 7)
            or self.members.action.shape != (members, states, 10, 50, 7)
            or self.family_weights.shape != (members, len(owners))
            or not torch.allclose(
                self.family_weights.sum(1),
                torch.ones(members, device=self.family_weights.device),
            )
            or not torch.allclose(
                self.reliability.sum(),
                torch.ones((), device=self.reliability.device),
            )
        ):
            raise ValueError("G3 verified effect bank changed shape or ownership")


def _response(prefix: str, values: Mapping[str, torch.Tensor]) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=values[f"{prefix}_owner"],
        flow=values[f"{prefix}_flow"],
        action=values[f"{prefix}_action"],
    )


def load_shared_effect_bank(
    manifest_path: Path,
    *,
    contract: LoRAContract,
    owners: Sequence[TargetOwner],
    device: torch.device | str,
) -> SharedCompilerEffectBank:
    manifest = read_json(manifest_path.resolve())
    record = manifest.get("tensor_file", {})
    path = Path(str(record.get("path", ""))).resolve()
    metadata = manifest.get("metadata", {})
    members = tuple(metadata.get("members", ()))
    if (
        manifest.get("schema_version") != G3_EFFECT_BANK_SCHEMA
        or manifest.get("status") != "complete"
        or len(members) not in (1, 2)
        or not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or metadata.get("action_meta_installed") is not False
        or metadata.get("held_gradient_use") is not False
    ):
        raise ValueError("G3 verified effect-bank authority changed")
    values = load_file(str(path), device=str(device))
    projections = []
    for member in range(len(members)):
        state = {}
        for target in contract.targets:
            for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
                name = target.name + suffix
                state[name] = values[f"projection.{member}.{name}"]
        projections.append(state)
    bank = SharedCompilerEffectBank(
        prefix=ExecutionPolicyPrefix(
            embeddings=values["prefix_embeddings"],
            padding=values["prefix_padding"],
        ),
        suffix_noise=values["suffix_noise"],
        validity=values["validity"].bool(),
        trajectory_ids=values["trajectory_ids"].long(),
        carrier=_response("carrier", values),
        members=_response("members", values),
        reliability=values["reliability"].float(),
        family_weights=values["family_weights"].float(),
        member_names=tuple(str(row["name"]) for row in members),
        projections=tuple(projections),
        anchors=tuple(dict(row) for row in metadata.get("anchors", ())),
    )
    bank.validate(owners)
    return bank


def _response_values(response: PolicyEffectResponse) -> tuple[torch.Tensor, ...]:
    return response.owner, response.flow, response.action


def _member_scales(bank: SharedCompilerEffectBank) -> tuple[torch.Tensor, ...]:
    output = []
    for field, (members, carrier) in enumerate(zip(
        _response_values(bank.members), _response_values(bank.carrier), strict=True
    )):
        rows = []
        for member in range(bank.member_count):
            mask = bank.validity[member]
            squared = (
                members[member, mask].float() - carrier[mask].float()
            ).square()
            signal = (
                squared.mean((0, 2, 3)) if field == 0 else squared.mean()
            )
            rows.append(signal.clamp_min(1e-8))
        output.append(torch.stack(rows))
    return tuple(output)


def member_effect_losses(
    candidate: PolicyEffectResponse,
    bank: SharedCompilerEffectBank,
    *,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global single-member response loss plus an explicit family-balanced term."""

    if candidate.owner.shape != bank.carrier.owner.shape or temperature <= 0:
        raise ValueError("G3 candidate effect response changed")
    scales = _member_scales(bank)
    owner_losses = []
    flow_losses = []
    action_losses = []
    for member in range(bank.member_count):
        mask = bank.validity[member]
        per_target = (
            bank.members.owner[member, mask].float()
            - candidate.owner[mask].float()
        ).square().mean((0, 2, 3))
        owner_losses.append(
            (
                per_target
                / scales[0][member]
                * bank.family_weights[member]
            ).sum()
        )
        flow_losses.append(
            (
                bank.members.flow[member, mask].float()
                - candidate.flow[mask].float()
            ).square().mean()
            / scales[1][member]
        )
        action_losses.append(
            (
                bank.members.action[member, mask].float()
                - candidate.action[mask].float()
            ).square().mean()
            / scales[2][member]
        )
    owner = torch.stack(owner_losses)
    flow = torch.stack(flow_losses)
    action = torch.stack(action_losses)
    total = (owner + flow + action) / 3.0
    logits = bank.reliability.log() - total / temperature
    responsibilities = logits.softmax(0)
    global_loss = -temperature * torch.logsumexp(logits, 0)
    family_loss = (responsibilities.detach() * owner).sum()
    return global_loss, family_loss, responsibilities, total


def carrier_preservation_loss(
    candidate: PolicyEffectResponse, bank: SharedCompilerEffectBank
) -> torch.Tensor:
    scales = _member_scales(bank)
    owner_scale = torch.einsum("m,mj->j", bank.reliability, scales[0])
    owner_weight = torch.einsum(
        "m,mj->j", bank.reliability, bank.family_weights
    )
    owner = (
        (candidate.owner.float() - bank.carrier.owner.float())
        .square()
        .mean((0, 2, 3))
        / owner_scale.clamp_min(1e-8)
        * owner_weight
    ).sum()
    values = [owner]
    for candidate_value, carrier_value, scale in zip(
        (candidate.flow, candidate.action),
        (bank.carrier.flow, bank.carrier.action),
        scales[1:],
        strict=True,
    ):
        normalizer = (bank.reliability * scale).sum().clamp_min(1e-8)
        values.append(
            (candidate_value.float() - carrier_value.float()).square().mean()
            / normalizer
        )
    return torch.stack(values).mean()


def response_consistency_loss(
    first: PolicyEffectResponse,
    second: PolicyEffectResponse,
    bank: SharedCompilerEffectBank,
) -> torch.Tensor:
    if first.owner.shape != second.owner.shape:
        raise ValueError("G3 same-task response panels differ")
    scales = _member_scales(bank)
    owner_scale = torch.einsum("m,mj->j", bank.reliability, scales[0])
    owner_weight = torch.einsum(
        "m,mj->j", bank.reliability, bank.family_weights
    )
    values = [
        (
            (first.owner.float() - second.owner.float())
            .square()
            .mean((0, 2, 3))
            / owner_scale.clamp_min(1e-8)
            * owner_weight
        ).sum()
    ]
    for left, right, scale in zip(
        (first.flow, first.action),
        (second.flow, second.action),
        scales[1:],
        strict=True,
    ):
        normalizer = (bank.reliability * scale).sum().clamp_min(1e-8)
        values.append((left.float() - right.float()).square().mean() / normalizer)
    return torch.stack(values).mean()


def effective_update_loss(
    *,
    candidate_state: Mapping[str, torch.Tensor],
    bank: SharedCompilerEffectBank,
    contract: LoRAContract,
    s_ref: torch.Tensor,
    responsibilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    member_losses = sensitivity_normalized_update_losses(
        candidate_state=candidate_state,
        reference_states=bank.projections,
        contract=contract,
        s_ref=s_ref,
        sensitivity_weights=bank.family_weights,
    )
    return (responsibilities.detach() * member_losses).sum(), member_losses
