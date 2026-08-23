"""Retained member-level evidence for the fixed ECP effect realizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.stage1_data import STAGE1_AUTHORITY_SCHEMA
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.reward.rollout import policy_flow_noise_cpu


EFFECT_PARTICLE_SHARD_SCHEMA = "ember_ecp_effect_particle_shard_v1"
EFFECT_PARTICLE_AUTHORITY_SCHEMA = "ember_ecp_effect_particle_authority_v1"
EFFECT_MEMBER_TENSOR_SCHEMA = "ember_ecp_effect_member_tensor_v1"


@dataclass(frozen=True)
class MemberAnchors:
    observations: tuple[Mapping[str, torch.Tensor], ...]
    suffix_noise: torch.Tensor
    trajectory_count: int
    trajectory_ids: torch.Tensor


def resolve_asset(asset_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def load_effect_member_rows(path: Path) -> tuple[dict[str, Any], ...]:
    manifest = read_json(path.resolve())
    rows = tuple(dict(row) for row in manifest.get("members", ()))
    if (
        manifest.get("schema_version") != STAGE1_AUTHORITY_SCHEMA
        or manifest.get("status") != "complete_mapping_diverse_authority"
        or len(rows) != 118
        or [int(row.get("index", -1)) for row in rows] != list(range(118))
    ):
        raise ValueError("ECP effect member authority changed")
    return rows


def balanced_member_shards(
    rows: Sequence[Mapping[str, Any]], shard_count: int
) -> tuple[tuple[int, ...], ...]:
    if shard_count <= 0:
        raise ValueError("effect evidence shard count must be positive")
    assignments: list[list[int]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    weighted = sorted(
        (
            (8 * len(row.get("trajectories", ())), int(row["index"]))
            for row in rows
        ),
        key=lambda value: (-value[0], value[1]),
    )
    for weight, index in weighted:
        shard = min(range(shard_count), key=lambda item: (loads[item], item))
        assignments[shard].append(index)
        loads[shard] += weight
    return tuple(tuple(sorted(values)) for values in assignments)


def load_member_anchors(
    row: Mapping[str, Any], *, asset_root: Path
) -> MemberAnchors:
    observations: list[Mapping[str, torch.Tensor]] = []
    noises: list[torch.Tensor] = []
    trajectory_ids: list[int] = []
    trajectories = tuple(row.get("trajectories", ()))
    if not 1 <= len(trajectories) <= 2:
        raise ValueError("effect member trajectory count changed")
    for trajectory_id, record in enumerate(trajectories):
        path = resolve_asset(asset_root, record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError("effect member trajectory asset changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        selected = tuple(int(value) for value in record["selected_replan_indices"])
        source_observations = tuple(payload.get("observations", ()))
        seeds = tuple(int(value) for value in payload.get("policy_noise_seeds", ()))
        if (
            payload.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
            or payload.get("success") is not True
            or str(payload.get("suite")) != str(row["suite"])
            or int(payload.get("task_id", -1)) != int(row["task_id"])
            or len(source_observations) != len(seeds)
            or len(selected) != 8
            or tuple(sorted(set(selected))) != selected
            or selected[-1] >= len(source_observations)
        ):
            raise ValueError("effect member successful anchors changed")
        for index in selected:
            observations.append(source_observations[index])
            noises.append(
                policy_flow_noise_cpu(
                    seed=seeds[index], chunk_size=50, max_action_dim=32
                )[0]
            )
            trajectory_ids.append(trajectory_id)
    return MemberAnchors(
        observations=tuple(observations),
        suffix_noise=torch.stack(noises),
        trajectory_count=len(trajectories),
        trajectory_ids=torch.tensor(trajectory_ids, dtype=torch.int64),
    )


def effect_member_tensors(
    *,
    owner_delta: torch.Tensor,
    residual: Mapping[str, torch.Tensor],
    trajectory_count: int,
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    particles = 2 * int(trajectory_count)
    if (
        owner_delta.shape != (particles, 8, 38, 4, 128)
        or not torch.isfinite(owner_delta).all()
    ):
        raise ValueError("effect owner particles changed shape")
    tensors: dict[str, torch.Tensor] = {
        "owner_delta": owner_delta.detach().cpu().to(torch.bfloat16).contiguous(),
        "particle_trajectory_ids": torch.arange(trajectory_count)
        .repeat_interleave(2)
        .to(torch.int64),
        "particle_probe_signs": torch.tensor([1, -1], dtype=torch.int64).repeat(
            trajectory_count
        ),
    }
    for index, target in enumerate(contract.targets):
        a = residual[target.name + LORA_A_SUFFIX]
        b = residual[target.name + LORA_B_SUFFIX]
        if (
            a.shape != (4, target.in_features)
            or b.shape != (target.out_features, 4)
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError("effect realizer target changed shape")
        tensors[f"target_{index:02d}_a"] = a.detach().cpu().contiguous()
        tensors[f"target_{index:02d}_b"] = b.detach().cpu().contiguous()
    return tensors


def save_effect_member(
    *,
    path: Path,
    owner_delta: torch.Tensor,
    residual: Mapping[str, torch.Tensor],
    trajectory_count: int,
    contract: LoRAContract,
) -> int:
    if path.exists():
        raise ValueError("effect member tensor already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        effect_member_tensors(
            owner_delta=owner_delta,
            residual=residual,
            trajectory_count=trajectory_count,
            contract=contract,
        ),
        str(path),
        metadata={"schema_version": EFFECT_MEMBER_TENSOR_SCHEMA},
    )
    return path.stat().st_size


def load_effect_member(
    path: Path, *, contract: LoRAContract, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    tensors = load_file(str(path.resolve()), device=str(device))
    owner_delta = tensors["owner_delta"]
    trajectory_ids = tensors["particle_trajectory_ids"]
    probe_signs = tensors["particle_probe_signs"]
    residual = {}
    for index, target in enumerate(contract.targets):
        residual[target.name + LORA_A_SUFFIX] = tensors[f"target_{index:02d}_a"]
        residual[target.name + LORA_B_SUFFIX] = tensors[f"target_{index:02d}_b"]
    if (
        owner_delta.ndim != 5
        or owner_delta.shape[1:] != (8, 38, 4, 128)
        or trajectory_ids.shape != (owner_delta.shape[0],)
        or probe_signs.shape != trajectory_ids.shape
    ):
        raise ValueError("effect member tensor authority changed")
    return owner_delta, residual, trajectory_ids, probe_signs


def aggregate_effect_shards(
    *, shard_manifests: Sequence[Path], output: Path
) -> Path:
    shards = [read_json(path.resolve()) for path in shard_manifests]
    if (
        not shards
        or any(row.get("schema_version") != EFFECT_PARTICLE_SHARD_SCHEMA for row in shards)
        or {int(row["shard_count"]) for row in shards} != {len(shards)}
        or {int(row["shard_index"]) for row in shards} != set(range(len(shards)))
        or any(row["config"] != shards[0]["config"] for row in shards)
        or any(row["source"] != shards[0]["source"] for row in shards)
    ):
        raise ValueError("effect particle shards changed")
    rows = sorted(
        (dict(value) for shard in shards for value in shard.get("members", ())),
        key=lambda value: int(value["index"]),
    )
    if [int(row["index"]) for row in rows] != list(range(118)):
        raise ValueError("effect particle member capture is incomplete")
    result = {
        "schema_version": EFFECT_PARTICLE_AUTHORITY_SCHEMA,
        "status": "complete_effect_particle_authority",
        "repository": shards[0]["repository"],
        "config": shards[0]["config"],
        "source": shards[0]["source"],
        "shards": [str(path.resolve()) for path in shard_manifests],
        "member_count": len(rows),
        "task_count": len({str(row["asset_key"]) for row in rows}),
        "particles": {
            "probe_axis_retained": True,
            "trajectory_axis_retained": True,
            "on_policy_successful_states_only": True,
            "shape_per_particle": [8, 38, 4, 128],
        },
        "members": rows,
        "information_wall": {
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "task_id_model_input": False,
            "held_optimizer_steps": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output.resolve(), result)
    return output.resolve()
