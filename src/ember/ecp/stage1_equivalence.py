"""Occupancy-complete successful-policy effect evidence for ECP Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.policy_effects import ExecutionPolicyPrefix, PolicyEffectResponse
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.reward.rollout import policy_flow_noise_cpu


EFFECT_BANK_SCHEMA = "ember_ecp_stage1_occupancy_effect_bank_v1"
CATEGORY_NAMES = ("initial", "successful", "candidate", "recovery")


@dataclass(frozen=True)
class OccupancyAnchors:
    observations: tuple[Mapping[str, torch.Tensor], ...]
    suffix_noise: torch.Tensor
    selected_replans: tuple[int, ...]
    progress: torch.Tensor
    success: bool


@dataclass(frozen=True)
class InitialOccupancyAnchor:
    observation: Mapping[str, torch.Tensor]
    suffix_noise: torch.Tensor
    success: bool


@dataclass(frozen=True)
class Stage1EffectBank:
    prefix: ExecutionPolicyPrefix
    suffix_noise: torch.Tensor
    category_ids: torch.Tensor
    stage_ids: torch.Tensor
    progress: torch.Tensor
    source: PolicyEffectResponse
    carrier: PolicyEffectResponse
    members: PolicyEffectResponse
    member_reliability: torch.Tensor

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
            or self.prefix.embeddings.shape[0] != states
            or self.prefix.padding.shape != self.prefix.embeddings.shape[:2]
            or self.suffix_noise.shape != (states, 50, 32)
            or self.category_ids.shape != (states,)
            or self.stage_ids.shape != (states,)
            or self.progress.shape != (states,)
            or self.member_reliability.shape != (members,)
            or self.source.owner.shape != (states, 38, 4, 128)
            or self.carrier.owner.shape != self.source.owner.shape
            or self.members.owner.shape != (members, states, 38, 4, 128)
            or self.source.flow.shape != (states, 10, 50, 32)
            or self.carrier.flow.shape != self.source.flow.shape
            or self.members.flow.shape != (members, states, 10, 50, 32)
            or self.source.action.shape != (states, 10, 50, 7)
            or self.carrier.action.shape != self.source.action.shape
            or self.members.action.shape != (members, states, 10, 50, 7)
            or not torch.isfinite(self.progress).all()
            or not torch.isfinite(self.member_reliability).all()
        ):
            raise ValueError("ECP Stage 1 effect bank changed shape")
        counts = torch.bincount(self.category_ids.cpu(), minlength=4).tolist()
        if counts != [8, 24, 8, 8]:
            raise ValueError("ECP Stage 1 effect bank category balance changed")

    def to(self, device: torch.device | str) -> "Stage1EffectBank":
        return Stage1EffectBank(
            prefix=ExecutionPolicyPrefix(
                embeddings=self.prefix.embeddings.to(device),
                padding=self.prefix.padding.to(device),
            ),
            suffix_noise=self.suffix_noise.to(device),
            category_ids=self.category_ids.to(device),
            stage_ids=self.stage_ids.to(device),
            progress=self.progress.to(device),
            source=self.source.to(device),
            carrier=self.carrier.to(device),
            members=self.members.to(device),
            member_reliability=self.member_reliability.to(device),
        )


def stage_progress_at_replans(
    stage_predicates: Mapping[str, Any], *, replans: int, replan_steps: int = 5
) -> torch.Tensor:
    transitions = tuple(stage_predicates.get("transitions", ()))
    predicates = tuple(stage_predicates.get("predicates", ()))
    if not transitions or not predicates or replans <= 0:
        raise ValueError("ECP occupancy is missing BDDL progress")
    ever = [False] * len(predicates)
    values = []
    for replan in range(replans):
        current = transitions[0]["satisfied"]
        for transition in transitions[1:]:
            if int(transition["step"]) > replan * replan_steps:
                break
            current = transition["satisfied"]
        ever = [left or bool(right) for left, right in zip(ever, current, strict=True)]
        values.append(sum(ever) / len(ever))
    return torch.tensor(values, dtype=torch.float32)


def equal_time_progress_strata(
    progress: torch.Tensor, count: int = 8
) -> tuple[int, ...]:
    """Choose ordered unique anchors on a joint time/progress coordinate."""

    length = int(progress.numel())
    if count != 8 or length < count or progress.ndim != 1:
        raise ValueError("ECP occupancy trajectory cannot provide eight strata")
    time = torch.linspace(0.0, 1.0, length)
    coordinate = 0.5 * time + 0.5 * progress.float().cummax(0).values
    targets = torch.linspace(float(coordinate[0]), float(coordinate[-1]), count)
    selected: list[int] = []
    for slot, target in enumerate(targets):
        lower = selected[-1] + 1 if selected else 0
        upper = length - (count - slot)
        selected.append(
            min(
                range(lower, upper + 1),
                key=lambda index: (
                    abs(float(coordinate[index]) - float(target)),
                    index,
                ),
            )
        )
    return tuple(selected)


def load_occupancy_anchors(
    *,
    row: Mapping[str, Any],
    selected_replans: Sequence[int] | None = None,
    require_success: bool | None = None,
) -> OccupancyAnchors:
    record = row.get("occupancy_trajectory", {})
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError("ECP occupancy sidecar changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observations = tuple(payload.get("observations", ()))
    seeds = tuple(int(value) for value in payload.get("policy_noise_seeds", ()))
    success = bool(payload.get("success"))
    if (
        payload.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or len(observations) != len(seeds)
        or len(observations) != int(record.get("replans", -1))
        or success != bool(row.get("success"))
        or (require_success is not None and success is not require_success)
    ):
        raise ValueError("ECP occupancy trajectory contract changed")
    progress = stage_progress_at_replans(
        row["stage_predicates"], replans=len(observations)
    )
    indices = tuple(
        equal_time_progress_strata(progress)
        if selected_replans is None
        else (int(value) for value in selected_replans)
    )
    if (
        len(indices) != 8
        or tuple(sorted(set(indices))) != indices
        or indices[-1] >= len(observations)
    ):
        raise ValueError("ECP occupancy anchor selection changed")
    noise = torch.cat(
        [
            policy_flow_noise_cpu(seed=seeds[index], chunk_size=50, max_action_dim=32)
            for index in indices
        ]
    )
    return OccupancyAnchors(
        observations=tuple(observations[index] for index in indices),
        suffix_noise=noise,
        selected_replans=indices,
        progress=progress[list(indices)],
        success=success,
    )


def load_initial_occupancy_anchor(row: Mapping[str, Any]) -> InitialOccupancyAnchor:
    record = row.get("occupancy_trajectory", {})
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError("ECP initial occupancy sidecar changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observations = tuple(payload.get("observations", ()))
    seeds = tuple(int(value) for value in payload.get("policy_noise_seeds", ()))
    success = bool(payload.get("success"))
    if (
        payload.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or not observations
        or len(observations) != len(seeds)
        or len(observations) != int(record.get("replans", -1))
        or success != bool(row.get("success"))
    ):
        raise ValueError("ECP initial occupancy trajectory contract changed")
    return InitialOccupancyAnchor(
        observation=observations[0],
        suffix_noise=policy_flow_noise_cpu(
            seed=seeds[0], chunk_size=50, max_action_dim=32
        )[0],
        success=success,
    )


def stack_observations(
    observations: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("ECP Stage 1 has no policy observations")
    keys = set(observations[0])
    if any(set(row) != keys for row in observations):
        raise ValueError("ECP Stage 1 observation keys changed")
    return {
        name: torch.cat([row[name] for row in observations]) for name in sorted(keys)
    }


def _response_tensors(
    prefix: str, value: PolicyEffectResponse
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_owner": value.owner.detach().cpu().contiguous(),
        f"{prefix}_flow": value.flow.detach().cpu().contiguous(),
        f"{prefix}_action": value.action.detach().cpu().contiguous(),
    }


def save_effect_bank(
    root: Path, bank: Stage1EffectBank, metadata: Mapping[str, Any]
) -> Path:
    bank.validate()
    root.mkdir(parents=True, exist_ok=False)
    tensor_path = root / "effect_bank.safetensors"
    tensors = {
        "prefix_embeddings": bank.prefix.embeddings.detach().cpu().to(torch.bfloat16),
        "prefix_padding": bank.prefix.padding.detach().cpu(),
        "suffix_noise": bank.suffix_noise.detach().cpu(),
        "category_ids": bank.category_ids.detach().cpu(),
        "stage_ids": bank.stage_ids.detach().cpu(),
        "progress": bank.progress.detach().cpu(),
        "member_reliability": bank.member_reliability.detach().cpu(),
        **_response_tensors("source", bank.source),
        **_response_tensors("carrier", bank.carrier),
        **_response_tensors("members", bank.members),
    }
    save_file(tensors, str(tensor_path))
    manifest_path = root / "manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": EFFECT_BANK_SCHEMA,
            "status": "complete",
            "tensor_file": {
                "path": str(tensor_path.resolve()),
                "bytes": tensor_path.stat().st_size,
            },
            "state_count": bank.state_count,
            "member_count": bank.member_count,
            "metadata": dict(metadata),
        },
    )
    return manifest_path


def load_effect_bank(
    manifest_path: Path, device: torch.device | str
) -> Stage1EffectBank:
    manifest = read_json(manifest_path.resolve())
    record = manifest.get("tensor_file", {})
    path = Path(str(record.get("path", ""))).resolve()
    if (
        manifest.get("schema_version") != EFFECT_BANK_SCHEMA
        or manifest.get("status") != "complete"
        or int(manifest.get("state_count", -1)) != 48
        or int(manifest.get("member_count", -1)) != 3
        or not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
    ):
        raise ValueError("ECP Stage 1 effect-bank manifest changed")
    value = load_file(str(path), device=str(device))

    def response(prefix: str) -> PolicyEffectResponse:
        return PolicyEffectResponse(
            owner=value[f"{prefix}_owner"],
            flow=value[f"{prefix}_flow"],
            action=value[f"{prefix}_action"],
        )

    bank = Stage1EffectBank(
        prefix=ExecutionPolicyPrefix(
            value["prefix_embeddings"], value["prefix_padding"]
        ),
        suffix_noise=value["suffix_noise"],
        category_ids=value["category_ids"],
        stage_ids=value["stage_ids"],
        progress=value["progress"],
        source=response("source"),
        carrier=response("carrier"),
        members=response("members"),
        member_reliability=value["member_reliability"],
    )
    bank.validate()
    return bank
