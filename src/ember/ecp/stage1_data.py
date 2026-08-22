"""Namespaced videos and successful-policy evidence for ECP Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from ember.ecp.low_rank import canonicalize_low_rank_factors
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import read_json
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority


@dataclass(frozen=True)
class ECPStage1Task:
    ordinal: int
    global_task_id: int
    suite: str
    task_id: int
    language: str
    path: Path
    expected_bytes: int
    episode_lengths: tuple[int, ...]
    fold_role: str
    asset_key: str = ""
    domain: str = "target40"

    def video_authority(self) -> WriterTaskAuthority:
        return WriterTaskAuthority(
            task_id=self.ordinal,
            language=self.language,
            path=self.path,
            expected_bytes=self.expected_bytes,
        )


@dataclass(frozen=True)
class ECPStage1Member:
    index: int
    ordinal: int
    global_task_id: int
    suite: str
    task_id: int
    member: str
    expert_step: int
    init_state_id: int
    fold_role: str
    reliability: float
    checkpoint: Path
    trajectories: tuple["ECPStage1Trajectory", ...]
    asset_key: str


@dataclass(frozen=True)
class ECPStage1Trajectory:
    path: Path
    expected_bytes: int
    selected_replan_indices: tuple[int, ...]


@dataclass(frozen=True)
class PackedStage1Videos:
    frames: torch.Tensor
    video_offsets: torch.Tensor
    frame_condition_ids: torch.Tensor
    video_group_ids: torch.Tensor
    demo_indices: tuple[int, ...]


@dataclass(frozen=True)
class ECPStage1EvidenceBank:
    members: tuple[ECPStage1Member, ...]
    member_states: Mapping[str, torch.Tensor]
    phase_response: torch.Tensor
    reliability: torch.Tensor

    def member_indices(self, ordinal: int) -> tuple[int, ...]:
        indices = tuple(
            member.index for member in self.members if member.ordinal == int(ordinal)
        )
        if not indices:
            raise ValueError("ECP Stage 1 task has no successful policy evidence")
        return indices

def gauge_canonicalize_factors(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact-SVD LoRA factors with a deterministic rank sign gauge."""

    return canonicalize_low_rank_factors(a, b)


def gauge_canonicalize_lora_state(
    state: Mapping[str, torch.Tensor], contract: LoRAContract
) -> dict[str, torch.Tensor]:
    """Place every LoRA target in the same compact rank coordinate system."""

    result: dict[str, torch.Tensor] = {}
    for owner in contract.targets:
        name_a = owner.name + LORA_A_SUFFIX
        name_b = owner.name + LORA_B_SUFFIX
        result[name_a], result[name_b] = gauge_canonicalize_factors(
            state[name_a], state[name_b]
        )
    return result


STAGE1_AUTHORITY_SCHEMA = "ember_ecp_stage1_mapping_diverse_authority_v1"


def _asset_path(asset_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def load_stage1_tasks(
    *, authority_manifest: Path, data_root: Path
) -> tuple[ECPStage1Task, ...]:
    manifest = read_json(authority_manifest)
    rows = tuple(dict(row) for row in manifest.get("tasks", ()))
    if (
        manifest.get("schema_version") != STAGE1_AUTHORITY_SCHEMA
        or manifest.get("status") != "complete_mapping_diverse_authority"
        or len(rows) != 95
    ):
        raise ValueError("ECP Stage 1 mapping-diverse task authority changed")
    tasks = []
    for ordinal, row in enumerate(rows):
        path = data_root / str(row["hdf5_relative_path"])
        task = ECPStage1Task(
            ordinal=int(row["ordinal"]),
            global_task_id=int(row["global_task_id"]),
            suite=str(row["suite"]),
            task_id=int(row["task_id"]),
            language=str(row["language"]),
            path=path,
            expected_bytes=int(row["hdf5_bytes"]),
            episode_lengths=tuple(int(value) for value in row["episode_lengths"]),
            fold_role=str(row["fold_role"]),
            asset_key=str(row["asset_key"]),
            domain=str(row["domain"]),
        )
        if (
            task.ordinal != ordinal
            or task.asset_key
            != (
                f"{'source90' if task.domain == 'libero90_nonheld' else 'target40'}:"
                f"{task.global_task_id}"
            )
            or task.domain not in {"libero90_nonheld", "target_train"}
            or not path.is_file()
            or path.stat().st_size != task.expected_bytes
            or len(task.episode_lengths) != 50
        ):
            raise ValueError(f"ECP Stage 1 task authority changed: {ordinal}")
        tasks.append(task)
    if (
        sum(task.fold_role == "fit" for task in tasks) != 90
        or sum(task.fold_role == "held_transform_only" for task in tasks) != 5
        or sum(task.domain == "libero90_nonheld" for task in tasks) != 71
        or len({task.asset_key for task in tasks}) != 95
    ):
        raise ValueError("ECP Stage 1 split differs from fixed fit90/held5")
    return tuple(tasks)


def load_stage1_evidence_bank(
    *,
    authority_manifest: Path,
    asset_root: Path,
    contract: LoRAContract,
    device: torch.device | str,
) -> ECPStage1EvidenceBank:
    manifest = read_json(authority_manifest)
    selected = tuple(dict(row) for row in manifest.get("members", ()))
    phase_cell = manifest.get("phase_response", {})
    if (
        manifest.get("schema_version") != STAGE1_AUTHORITY_SCHEMA
        or manifest.get("status") != "complete_mapping_diverse_authority"
        or len(selected) != 118
        or phase_cell.get("tensor") != "member_phase_response"
        or tuple(int(value) for value in phase_cell.get("shape", ())) != (118, 8, 32)
    ):
        raise ValueError("ECP Stage 1 successful-policy authority changed")
    phase_path = _asset_path(asset_root, phase_cell["path"])
    tensors = load_file(str(phase_path), device=str(device))
    fingerprints = tensors["member_phase_response"]
    if fingerprints.shape != (118, 8, 32):
        raise ValueError("ECP Stage 1 phase response must be 118x8x32")
    loaded = [
        _load_stage1_member(
            row=row,
            index=index,
            asset_root=asset_root,
            contract=contract,
            device=device,
        )
        for index, row in enumerate(selected)
    ]
    members = [row[0] for row in loaded]
    states = [row[1] for row in loaded]
    reliability = [row[0].reliability for row in loaded]
    if (
        len({member.asset_key for member in members}) != 95
        or {member.ordinal for member in members} != set(range(95))
        or any(
            member.asset_key
            != manifest["tasks"][member.ordinal]["asset_key"]
            for member in members
        )
    ):
        raise ValueError("ECP Stage 1 member-to-task ownership changed")
    stacked = {
        name: torch.stack([state[name] for state in states]) for name in states[0]
    }
    return ECPStage1EvidenceBank(
        members=tuple(members),
        member_states=stacked,
        phase_response=fingerprints.float(),
        reliability=torch.tensor(reliability, device=device),
    )


def _load_stage1_member(
    *,
    row: Mapping[str, Any],
    index: int,
    asset_root: Path,
    contract: LoRAContract,
    device: torch.device | str,
) -> tuple[ECPStage1Member, dict[str, torch.Tensor]]:
    checkpoint = _asset_path(asset_root, row["checkpoint"])
    trajectories = tuple(
        ECPStage1Trajectory(
            path=_asset_path(asset_root, value["path"]),
            expected_bytes=int(value["bytes"]),
            selected_replan_indices=tuple(
                int(item) for item in value["selected_replan_indices"]
            ),
        )
        for value in row.get("trajectories", ())
    )
    reliability = float(row["reliability"])
    valid_trajectories = trajectories and all(
        len(value.selected_replan_indices) == 8
        and value.path.is_file()
        and value.path.stat().st_size == value.expected_bytes
        for value in trajectories
    )
    if (
        int(row["index"]) != index
        or not (checkpoint / "adapter.safetensors").is_file()
        or not 0.0 < reliability <= 1.0
        or not valid_trajectories
    ):
        raise ValueError("ECP Stage 1 member asset changed")
    member = ECPStage1Member(
        index=index,
        ordinal=int(row["ordinal"]),
        global_task_id=int(row["global_task_id"]),
        suite=str(row["suite"]),
        task_id=int(row["task_id"]),
        member=str(row["member"]),
        expert_step=int(row["expert_step"]),
        init_state_id=int(row["init_state_id"]),
        fold_role=str(row["fold_role"]),
        reliability=reliability,
        checkpoint=checkpoint,
        trajectories=trajectories,
        asset_key=str(row["asset_key"]),
    )
    state = load_file(str(checkpoint / "adapter.safetensors"), device=str(device))
    validate_lora_state(state, contract)
    return member, gauge_canonicalize_lora_state(state, contract)


def build_stage1_video_store(
    tasks: Sequence[ECPStage1Task], *, frame_stride: int
) -> RawTeacherVideoStore:
    return RawTeacherVideoStore(
        tuple(task.video_authority() for task in tasks), frame_stride=frame_stride
    )


def stage1_demo_indices(
    *, ordinal: int, visit: int, seed: int, k: int
) -> tuple[int, ...]:
    if not 1 <= k <= 4:
        raise ValueError("ECP Stage 1 supports one to four visible videos")
    order = np.random.default_rng(
        np.random.SeedSequence([seed, ordinal, visit])
    ).permutation(50)
    return tuple(int(value) for value in order[:k])


def tokenize_stage1_languages(
    tasks: tuple[ECPStage1Task, ...],
    *,
    tokenizer_path: Path,
    max_length: int,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    tokenizer = Pi05TeacherPrefixTokenizer(tokenizer_path, max_length, str(device))
    tokens, masks, _ = tokenizer([task.language for task in tasks])
    return {
        task.ordinal: (tokens[index : index + 1], masks[index : index + 1])
        for index, task in enumerate(tasks)
    }


def pack_stage1_videos(
    *,
    store: RawTeacherVideoStore,
    ordinal: int,
    visit: int,
    seed: int,
    k: int,
    device: torch.device,
) -> PackedStage1Videos:
    demos = stage1_demo_indices(ordinal=ordinal, visit=visit, seed=seed, k=k)
    videos = [store.load(ordinal, demo) for demo in demos]
    counts = [int(video.frames.shape[0]) for video in videos]
    frames = torch.from_numpy(
        np.concatenate([video.frames for video in videos])
    ).to(device=device, non_blocking=True)
    offsets = torch.tensor(
        [0, *np.cumsum(counts).tolist()], dtype=torch.long, device=device
    )
    return PackedStage1Videos(
        frames=frames,
        video_offsets=offsets,
        frame_condition_ids=torch.zeros(
            frames.shape[0], dtype=torch.long, device=device
        ),
        video_group_ids=torch.zeros(k, dtype=torch.long, device=device),
        demo_indices=demos,
    )
