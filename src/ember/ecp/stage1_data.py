"""Train24 videos and successful-policy evidence for ECP Stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from ember.ecp.policy_teacher import PrivilegedPolicyEvidence
from ember.ecp.low_rank import canonicalize_low_rank_factors
from ember.ecp.stage1_support import PolicySupportTask
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
    trajectory_path: Path
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

    def evidence(
        self, ordinal: int, support: PolicySupportTask
    ) -> PrivilegedPolicyEvidence:
        index = torch.tensor(
            self.member_indices(ordinal),
            dtype=torch.long,
            device=self.phase_response.device,
        )
        if support.member_indices != self.member_indices(ordinal):
            raise ValueError("policy-support member ordering changed")
        return PrivilegedPolicyEvidence(
            member_states={
                name: value.index_select(0, index)
                for name, value in self.member_states.items()
            },
            phase_response=self.phase_response.index_select(0, index),
            reliability=self.reliability.index_select(0, index),
            policy_response=support.policy_response,
            policy_response_weights=support.policy_response_weights,
        )


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


def load_stage1_tasks(
    *,
    target_manifest: Path,
    selection_path: Path,
    data_root: Path,
) -> tuple[ECPStage1Task, ...]:
    manifest = read_json(target_manifest)
    selection = read_json(selection_path)
    selected: dict[int, dict[str, Any]] = {}
    for row in selection.get("rows", ()):
        ordinal = int(row["ordinal"])
        previous = selected.setdefault(ordinal, dict(row))
        if (
            int(previous["global_task_id"]) != int(row["global_task_id"])
            or previous["fold_role"] != row["fold_role"]
        ):
            raise ValueError("successful-member task ownership changed")
    train_rows = {
        int(row["global_task_id"]): row
        for row in manifest.get("tasks", ())
        if row.get("split_role") == "train"
    }
    tasks = []
    for ordinal in range(24):
        selected_row = selected[ordinal]
        global_id = int(selected_row["global_task_id"])
        row = train_rows[global_id]
        path = data_root / str(row["hdf5"]["relative_path"])
        task = ECPStage1Task(
            ordinal=ordinal,
            global_task_id=global_id,
            suite=str(row["suite"]),
            task_id=int(row["task_id"]),
            language=str(row["language"]),
            path=path,
            expected_bytes=int(row["hdf5"]["bytes"]),
            episode_lengths=tuple(
                int(value) for value in row["demonstrations"]["episode_lengths"]
            ),
            fold_role=str(selected_row["fold_role"]),
        )
        if (
            not path.is_file()
            or path.stat().st_size != task.expected_bytes
            or task.suite != str(selected_row["suite"])
            or task.task_id != int(selected_row["task_id"])
            or len(task.episode_lengths) != 50
        ):
            raise ValueError(f"ECP Stage 1 task authority changed: {ordinal}")
        tasks.append(task)
    if (
        sum(task.fold_role == "fit" for task in tasks) != 19
        or sum(task.fold_role == "held_transform_only" for task in tasks) != 5
    ):
        raise ValueError("ECP Stage 1 split differs from fixed fit19/held5")
    return tuple(tasks)


def _member_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(row["ordinal"]), int(row["expert_step"]), int(row["init_state_id"])


def _expert_assets(
    analysis: Mapping[str, Any], asset_root: Path
) -> tuple[
    dict[tuple[int, str, int], Path],
    dict[tuple[int, str, int, int], Path],
]:
    checkpoints: dict[tuple[int, str, int], Path] = {}
    trajectories: dict[tuple[int, str, int, int], Path] = {}
    for step_text, relative in analysis.get("panels", {}).items():
        step = int(step_text)
        root = (asset_root / str(relative)).resolve()
        contract = read_json(root / "run_contract.json")
        for row in contract.get("adapter", {}).get("tasks", ()):
            key = (step, str(row["suite"]), int(row["task_id"]))
            checkpoint = Path(str(row["checkpoint"])).resolve()
            if not (checkpoint / "adapter.safetensors").is_file():
                raise ValueError("successful expert checkpoint is missing")
            checkpoints[key] = checkpoint
        results = read_json(root / "results.json")
        for row in results.get("rows", ()):
            capture = row.get("occupancy_trajectory", {})
            path = Path(str(capture.get("path", ""))).resolve()
            if row.get("success") is True and path.is_file():
                trajectories[
                    (
                        step,
                        str(row["suite"]),
                        int(row["task_id"]),
                        int(row["init_state_id"]),
                    )
                ] = path
    if {key[0] for key in checkpoints} != {250, 500, 1000, 2000}:
        raise ValueError("successful expert panel family changed")
    return checkpoints, trajectories


def load_stage1_evidence_bank(
    *,
    selection_path: Path,
    phase_analysis_path: Path,
    phase_code_root: Path,
    asset_root: Path,
    contract: LoRAContract,
    device: torch.device | str,
) -> ECPStage1EvidenceBank:
    selection = read_json(selection_path)
    analysis = read_json(phase_analysis_path)
    phase_result = read_json(phase_code_root / "result.json")
    if (
        selection.get("schema_version")
        != "ember_successful_expert_equivalence_selection_v1"
        or analysis.get("schema_version")
        != "ember_successful_expert_equivalence_phase_analysis_v1"
        or analysis.get("decision") != "advance_to_phase_aligned_fixed_decoder"
        or phase_result.get("schema_version")
        != "ember_successful_expert_equivalence_phase_codes_v1"
    ):
        raise ValueError("ECP Stage 1 successful-policy authority changed")
    selected = sorted(
        (dict(row) for row in selection.get("rows", ())), key=_member_key
    )
    phase_members = sorted(
        (dict(row) for row in phase_result.get("members", ())), key=_member_key
    )
    if (
        len(selected) != 47
        or [_member_key(row) for row in selected]
        != [_member_key(row) for row in phase_members]
    ):
        raise ValueError("ECP Stage 1 member ordering changed")
    tensors = load_file(
        str(phase_code_root / "phase_codes.safetensors"), device=str(device)
    )
    fingerprints = tensors["member_phase_fingerprints"]
    if fingerprints.shape != (47, 256):
        raise ValueError("ECP Stage 1 phase response must be 47x8x32")
    checkpoints, trajectories = _expert_assets(analysis, asset_root)
    members = []
    states = []
    reliability = []
    for index, row in enumerate(selected):
        step = int(row["expert_step"])
        checkpoint = checkpoints[(step, str(row["suite"]), int(row["task_id"]))]
        trajectory = trajectories[
            (
                step,
                str(row["suite"]),
                int(row["task_id"]),
                int(row["init_state_id"]),
            )
        ]
        successes = int(row["checkpoint_successes"][str(step)])
        selected_indices = tuple(
            int(value) for value in phase_members[index]["selected_replan_indices"]
        )
        if len(selected_indices) != 8:
            raise ValueError("successful member lost its eight phase states")
        members.append(
            ECPStage1Member(
                index=index,
                ordinal=int(row["ordinal"]),
                global_task_id=int(row["global_task_id"]),
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                member=str(row["member"]),
                expert_step=step,
                init_state_id=int(row["init_state_id"]),
                fold_role=str(row["fold_role"]),
                reliability=successes / 50.0,
                checkpoint=checkpoint,
                trajectory_path=trajectory,
                selected_replan_indices=selected_indices,
            )
        )
        state = load_file(str(checkpoint / "adapter.safetensors"), device=str(device))
        validate_lora_state(state, contract)
        states.append(gauge_canonicalize_lora_state(state, contract))
        reliability.append(successes / 50.0)
    stacked = {
        name: torch.stack([state[name] for state in states]) for name in states[0]
    }
    return ECPStage1EvidenceBank(
        members=tuple(members),
        member_states=stacked,
        phase_response=fingerprints.reshape(47, 8, 32).float(),
        reliability=torch.tensor(reliability, device=device),
    )


def build_stage1_video_store(
    tasks: Sequence[ECPStage1Task], *, frame_stride: int
) -> RawTeacherVideoStore:
    return RawTeacherVideoStore(
        tuple(task.video_authority() for task in tasks), frame_stride=frame_stride
    )


def stage1_demo_indices(*, ordinal: int, visit: int, seed: int, k: int) -> tuple[int, ...]:
    if not 1 <= k <= 4:
        raise ValueError("ECP Stage 1 supports one to four visible videos")
    order = np.random.default_rng(
        np.random.SeedSequence([seed, ordinal, visit])
    ).permutation(50)
    return tuple(int(value) for value in order[:k])


def _frame_count(raw_frames: int, stride: int) -> int:
    count = (raw_frames - 1) // stride + 1
    return count + int((raw_frames - 1) % stride != 0)


def build_stage1_schedule(
    *,
    config: Mapping[str, Any],
    tasks: tuple[ECPStage1Task, ...],
    world_size: int,
    total_task_visits: int,
    mode: str,
) -> tuple[tuple[int, int], ...]:
    fit = tuple(int(value) for value in config["roles"]["fit_task_ordinals"])
    if mode == "profile":
        return tuple((fit[0], visit) for visit in range(total_task_visits))
    visits = int(config["optimization"]["visits_per_fit_task"])
    balance_rounds = int(
        config["optimization"]["task_balance_block_rounds"]
    )
    expected_balance_rounds = world_size // math.gcd(len(fit), world_size)
    if balance_rounds != expected_balance_rounds or visits % balance_rounds:
        raise ValueError("ECP Stage 1 task-balance block changed")
    balance_block_visits = len(fit) * balance_rounds
    if any(
        int(value) % balance_block_visits
        for value in config["optimization"]["stage_stop_task_visits"]
    ):
        raise ValueError("ECP Stage 1 decision prefix is not task-equal")
    stride = int(config["data"]["frame_stride"])
    k = int(config["data"]["visible_videos_per_visit"])
    seed = int(config["data"]["pair_seed"])
    by_ordinal = {task.ordinal: task for task in tasks}
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["optimization"]["seed"])
    )
    schedule = []
    for visit_start in range(0, visits, balance_rounds):
        rows = []
        for visit in range(visit_start, visit_start + balance_rounds):
            for ordinal in fit:
                demos = stage1_demo_indices(
                    ordinal=ordinal, visit=visit, seed=seed, k=k
                )
                cost = sum(
                    _frame_count(
                        by_ordinal[ordinal].episode_lengths[demo], stride
                    )
                    for demo in demos
                )
                rows.append((cost, ordinal, visit))
        rows.sort(reverse=True)
        groups = [
            rows[index : index + world_size]
            for index in range(0, len(rows), world_size)
        ]
        for group_index in torch.randperm(
            len(groups), generator=generator
        ).tolist():
            group = groups[group_index]
            rank_order = torch.randperm(
                len(group), generator=generator
            ).tolist()
            schedule.extend(
                (group[index][1], group[index][2]) for index in rank_order
            )
    if len(schedule) != total_task_visits:
        raise ValueError("ECP Stage 1 task-equal schedule changed")
    return tuple(schedule)


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
