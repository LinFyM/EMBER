"""Set-valued policy-effect and update objectives for the G1 free-code oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.policy_effects import ExecutionPolicyPrefix, PolicyEffectResponse
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_source_checkpoint import read_json


EFFECT_BANK_SCHEMA = "ember_ecp_stage1_occupancy_effect_bank_v1"
CATEGORY_NAMES = ("initial", "successful", "candidate", "recovery")


@dataclass(frozen=True)
class G1EffectBank:
    prefix: ExecutionPolicyPrefix
    suffix_noise: torch.Tensor
    category_ids: torch.Tensor
    stage_ids: torch.Tensor
    progress: torch.Tensor
    source: PolicyEffectResponse
    carrier: PolicyEffectResponse
    members: PolicyEffectResponse
    member_reliability: torch.Tensor
    member_names: tuple[str, ...]
    anchors: tuple[dict[str, Any], ...]

    @property
    def state_count(self) -> int:
        return int(self.suffix_noise.shape[0])

    @property
    def member_count(self) -> int:
        return int(self.member_reliability.numel())

    def validate(self) -> None:
        states = self.state_count
        members = self.member_count
        if (
            states != 48
            or members != 3
            or len(self.member_names) != members
            or len(set(self.member_names)) != members
            or len(self.anchors) != states
            or self.prefix.embeddings.shape[0] != states
            or self.prefix.padding.shape != self.prefix.embeddings.shape[:2]
            or self.suffix_noise.shape != (states, 50, 32)
            or self.category_ids.shape != (states,)
            or self.stage_ids.shape != (states,)
            or self.progress.shape != (states,)
            or self.source.owner.shape != (states, 38, 4, 128)
            or self.carrier.owner.shape != self.source.owner.shape
            or self.members.owner.shape != (members, states, 38, 4, 128)
            or self.source.flow.shape != (states, 10, 50, 32)
            or self.carrier.flow.shape != self.source.flow.shape
            or self.members.flow.shape != (members, states, 10, 50, 32)
            or self.source.action.shape != (states, 10, 50, 7)
            or self.carrier.action.shape != self.source.action.shape
            or self.members.action.shape != (members, states, 10, 50, 7)
            or torch.bincount(self.category_ids.cpu(), minlength=4).tolist()
            != [8, 24, 8, 8]
        ):
            raise ValueError("G1 policy-effect bank changed shape or ownership")


@dataclass(frozen=True)
class VerifiedMemberObjective:
    bank: G1EffectBank
    validity: torch.Tensor
    scales: PolicyEffectResponse
    reliability: torch.Tensor
    temperature: float


@dataclass(frozen=True)
class G1SetLoss:
    global_effect: torch.Tensor
    member_effects: torch.Tensor
    responsibilities: torch.Tensor
    effective_update: torch.Tensor
    member_updates: torch.Tensor
    carrier_preservation: torch.Tensor


def _response(prefix: str, values: Mapping[str, torch.Tensor]) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=values[f"{prefix}_owner"],
        flow=values[f"{prefix}_flow"],
        action=values[f"{prefix}_action"],
    )


def load_g1_effect_bank(
    manifest_path: Path, *, device: torch.device | str
) -> G1EffectBank:
    manifest = read_json(manifest_path.resolve())
    record = manifest.get("tensor_file", {})
    tensor_path = Path(str(record.get("path", ""))).resolve()
    metadata = manifest.get("metadata", {})
    members = tuple(metadata.get("members", ()))
    if (
        manifest.get("schema_version") != EFFECT_BANK_SCHEMA
        or manifest.get("status") != "complete"
        or int(manifest.get("state_count", -1)) != 48
        or int(manifest.get("member_count", -1)) != 3
        or not tensor_path.is_file()
        or tensor_path.stat().st_size != int(record.get("bytes", -1))
        or metadata.get("action_meta_installed") is not False
    ):
        raise ValueError("G1 policy-effect authority changed")
    values = load_file(str(tensor_path), device=str(device))
    bank = G1EffectBank(
        prefix=ExecutionPolicyPrefix(
            embeddings=values["prefix_embeddings"],
            padding=values["prefix_padding"],
        ),
        suffix_noise=values["suffix_noise"],
        category_ids=values["category_ids"],
        stage_ids=values["stage_ids"],
        progress=values["progress"],
        source=_response("source", values),
        carrier=_response("carrier", values),
        members=_response("members", values),
        member_reliability=values["member_reliability"],
        member_names=tuple(str(row["name"]) for row in members),
        anchors=tuple(dict(row) for row in metadata.get("anchors", ())),
    )
    bank.validate()
    return bank


def initial_success_ids(results_path: Path, *, suite: str, task_id: int) -> set[int]:
    results = read_json(results_path.resolve())
    rows = [
        row
        for row in results.get("rows", ())
        if str(row.get("suite")) == suite and int(row.get("task_id", -1)) == task_id
    ]
    if len(rows) != 50 or {int(row["init_state_id"]) for row in rows} != set(range(50)):
        raise ValueError("G1 member fixed50 success authority changed")
    return {int(row["init_state_id"]) for row in rows if bool(row.get("success"))}


def verified_member_validity(
    bank: G1EffectBank,
    initial_success: Mapping[str, set[int]],
) -> torch.Tensor:
    if set(initial_success) != set(bank.member_names):
        raise ValueError("G1 member success authorities are incomplete")
    validity = torch.zeros(
        bank.member_count,
        bank.state_count,
        dtype=torch.bool,
        device=bank.suffix_noise.device,
    )
    member_index = {name: index for index, name in enumerate(bank.member_names)}
    for state, anchor in enumerate(bank.anchors):
        category = str(anchor.get("category"))
        if category == "initial":
            init_state = int(anchor["init_state_id"])
            for name, index in member_index.items():
                validity[index, state] = init_state in initial_success[name]
        elif category == "successful":
            validity[member_index[str(anchor["generator"])], state] = True
        elif category not in {"candidate", "recovery"}:
            raise ValueError("G1 effect anchor category changed")
    if any(int(validity[index, 8:32].sum()) != 8 for index in range(3)):
        raise ValueError("G1 successful-trajectory validity changed")
    return validity


def _response_fields(value: PolicyEffectResponse) -> tuple[torch.Tensor, ...]:
    return value.owner, value.flow, value.action


def member_response_scales(bank: G1EffectBank) -> PolicyEffectResponse:
    values = []
    for members, source in zip(
        _response_fields(bank.members), _response_fields(bank.source), strict=True
    ):
        reduction = tuple(range(2, members.ndim))
        signal = (
            (members.float() - source.float().unsqueeze(0)).square().mean(dim=reduction)
        )
        floor = 0.05 * signal.mean().clamp_min(1e-8)
        values.append(signal + floor)
    return PolicyEffectResponse(*values)


def build_verified_member_objective(
    bank: G1EffectBank,
    validity: torch.Tensor,
    *,
    temperature: float = 0.25,
) -> VerifiedMemberObjective:
    if validity.shape != (bank.member_count, bank.state_count) or temperature <= 0:
        raise ValueError("G1 verified-member objective changed")
    reliability = bank.member_reliability.float().clamp_min(1e-4)
    reliability = reliability / reliability.sum()
    return VerifiedMemberObjective(
        bank=bank,
        validity=validity,
        scales=member_response_scales(bank),
        reliability=reliability,
        temperature=float(temperature),
    )


def member_response_distances(
    candidate: PolicyEffectResponse,
    objective: VerifiedMemberObjective,
) -> torch.Tensor:
    result = None
    for candidate_value, member_value, scale in zip(
        _response_fields(candidate),
        _response_fields(objective.bank.members),
        _response_fields(objective.scales),
        strict=True,
    ):
        reduction = tuple(range(2, member_value.ndim))
        distance = (
            member_value.float() - candidate_value.float().unsqueeze(0)
        ).square().mean(dim=reduction) / scale
        result = distance if result is None else result + distance
    if result is None:
        raise RuntimeError("G1 policy response is empty")
    return result / 3.0


def verified_member_effects(
    candidate: PolicyEffectResponse,
    objective: VerifiedMemberObjective,
) -> torch.Tensor:
    per_state = member_response_distances(candidate, objective)
    losses = []
    for member in range(objective.bank.member_count):
        categories = []
        for category in (0, 1):
            mask = objective.validity[member] & (
                objective.bank.category_ids == category
            )
            if torch.any(mask):
                categories.append(per_state[member, mask].mean())
        if not categories:
            raise ValueError("G1 global member has no verified states")
        losses.append(torch.stack(categories).mean())
    return torch.stack(losses)


def global_member_effect_loss(
    member_losses: torch.Tensor,
    objective: VerifiedMemberObjective,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = objective.reliability.log() - member_losses / objective.temperature
    return (
        -objective.temperature * torch.logsumexp(logits, dim=0),
        torch.softmax(logits, dim=0),
    )


def carrier_preservation_loss(
    candidate: PolicyEffectResponse,
    objective: VerifiedMemberObjective,
) -> torch.Tensor:
    per_state = None
    for candidate_value, carrier_value, scale in zip(
        _response_fields(candidate),
        _response_fields(objective.bank.carrier),
        _response_fields(objective.scales),
        strict=True,
    ):
        reduction = tuple(range(1, candidate_value.ndim))
        distance = (candidate_value.float() - carrier_value.float()).square().mean(
            dim=reduction
        ) / scale.mean(0)
        per_state = distance if per_state is None else per_state + distance
    if per_state is None:
        raise RuntimeError("G1 carrier response is empty")
    category_means = [
        per_state[objective.bank.category_ids == category].mean()
        for category in (0, 2, 3)
    ]
    return torch.stack(category_means).mean() / 3.0


def low_rank_distance_squared(
    a: torch.Tensor,
    b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
) -> torch.Tensor:
    candidate_norm = ((b.transpose(0, 1) @ b) * (a @ a.transpose(0, 1))).sum()
    reference_norm = (
        (reference_b.transpose(0, 1) @ reference_b)
        * (reference_a @ reference_a.transpose(0, 1))
    ).sum()
    cross = (
        (b.transpose(0, 1) @ reference_b) * (a @ reference_a.transpose(0, 1))
    ).sum()
    return (candidate_norm + reference_norm - 2.0 * cross).clamp_min(0)


def family_balanced_sensitivity_weights(
    sensitivity: torch.Tensor,
    owners: Sequence[TargetOwner],
) -> torch.Tensor:
    if sensitivity.ndim != 2 or sensitivity.shape[1] != len(owners):
        raise ValueError("G1 policy sensitivity shape changed")
    result = torch.zeros_like(sensitivity.float())
    families = tuple(TargetFamily)
    for family in families:
        indices = [
            index for index, owner in enumerate(owners) if owner.family is family
        ]
        values = sensitivity[:, indices].float().abs()
        floor = 0.05 * values.mean(1, keepdim=True).clamp_min(1e-8)
        values = torch.maximum(values, floor)
        result[:, indices] = 0.25 * values / values.sum(1, keepdim=True)
    return result


def sensitivity_normalized_update_losses(
    *,
    candidate_state: Mapping[str, torch.Tensor],
    reference_states: Sequence[Mapping[str, torch.Tensor]],
    contract: LoRAContract,
    s_ref: torch.Tensor,
    sensitivity_weights: torch.Tensor,
) -> torch.Tensor:
    members = len(reference_states)
    if (
        members <= 0
        or s_ref.shape != (len(contract.targets),)
        or sensitivity_weights.shape != (members, len(contract.targets))
    ):
        raise ValueError("G1 effective-update authorities changed")
    losses = []
    for member, reference in enumerate(reference_states):
        per_target = []
        for target, scale in zip(contract.targets, s_ref, strict=True):
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            squared = low_rank_distance_squared(
                candidate_state[a_name].float(),
                candidate_state[b_name].float(),
                reference[a_name].float(),
                reference[b_name].float(),
            )
            matrix_mse = squared / float(target.in_features * target.out_features)
            per_target.append(matrix_mse / scale.float().square().clamp_min(1e-12))
        losses.append((torch.stack(per_target) * sensitivity_weights[member]).sum())
    return torch.stack(losses)


def g1_set_losses(
    *,
    candidate_response: PolicyEffectResponse,
    objective: VerifiedMemberObjective,
    candidate_state: Mapping[str, torch.Tensor],
    reference_states: Sequence[Mapping[str, torch.Tensor]],
    contract: LoRAContract,
    s_ref: torch.Tensor,
    sensitivity_weights: torch.Tensor,
) -> G1SetLoss:
    member_effects = verified_member_effects(candidate_response, objective)
    global_effect, responsibilities = global_member_effect_loss(
        member_effects, objective
    )
    member_updates = sensitivity_normalized_update_losses(
        candidate_state=candidate_state,
        reference_states=reference_states,
        contract=contract,
        s_ref=s_ref,
        sensitivity_weights=sensitivity_weights,
    )
    return G1SetLoss(
        global_effect=global_effect,
        member_effects=member_effects,
        responsibilities=responsibilities,
        effective_update=(responsibilities.detach() * member_updates).sum(),
        member_updates=member_updates,
        carrier_preservation=carrier_preservation_loss(candidate_response, objective),
    )
