"""Successful-trajectory flow panels used by the phase-aligned decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from lerobot.utils.constants import ACTION
from safetensors.torch import load_file

from ember.functional_adaptation.functional_response import (
    FunctionalResponseTarget,
    build_functional_response_target,
)
from ember.functional_adaptation.phase_alignment import arc_length_phase_indices
from ember.functional_adaptation.phase_decoder_codes import (
    PhaseDecoderCodeAuthority,
    PhaseDecoderMember,
)
from ember.lora import LoRAContract
from ember.pi05_source_checkpoint import read_json


@dataclass(frozen=True)
class CachedPhasePanel:
    batch: Mapping[str, torch.Tensor]
    target: FunctionalResponseTarget
    policy_seed: int


@dataclass(frozen=True)
class PhaseMemberSource:
    member: PhaseDecoderMember
    trajectory_path: Path
    trajectory_bytes: int
    expert_checkpoint: Path


@dataclass(frozen=True)
class ProjectedOccupancySource:
    trajectory_path: Path
    trajectory_bytes: int
    success: bool


def _capture_panels(
    analysis_path: Path, repo_root: Path
) -> dict[int, tuple[dict[Any, Any], dict[Any, Any]]]:
    analysis = read_json(analysis_path.resolve())
    roots = {
        int(step): (repo_root / Path(str(path))).resolve()
        for step, path in analysis.get("panels", {}).items()
    }
    if set(roots) != {250, 500, 1000, 2000}:
        raise ValueError("phase decoder capture panel family changed")
    panels = {}
    for step, root in roots.items():
        contract = read_json(root / "run_contract.json")
        results = read_json(root / "results.json")
        rows = {
            (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): row
            for row in results.get("rows", ())
        }
        adapters = {
            (str(row["suite"]), int(row["task_id"])): row
            for row in contract.get("adapter", {}).get("tasks", ())
        }
        capture = contract.get("diagnostic_occupancy_capture", {})
        if (
            results.get("mode") != "formal"
            or results.get("role") != "development_train"
            or contract.get("git", {}).get("dirty_paths") != []
            or int(capture.get("selected_step", -1)) != step
        ):
            raise ValueError("phase decoder capture authority changed")
        panels[step] = (rows, adapters)
    return panels


def load_phase_member_sources(
    *,
    analysis_path: Path,
    codes: PhaseDecoderCodeAuthority,
    repo_root: Path,
) -> tuple[PhaseMemberSource, ...]:
    """Resolve each fixed member to its successful replay and expert adapter."""

    panels = _capture_panels(analysis_path, repo_root)
    sources = []
    for member in codes.members:
        rows, adapters = panels[member.expert_step]
        key = (member.suite, member.task_id, member.init_state_id)
        row = rows.get(key)
        adapter = adapters.get(key[:2])
        capture = {} if row is None else row.get("occupancy_trajectory", {})
        trajectory = Path(str(capture.get("path", ""))).resolve()
        checkpoint = (
            None
            if adapter is None
            else Path(str(adapter.get("checkpoint", ""))).resolve()
        )
        if (
            row is None
            or adapter is None
            or row.get("success") is not True
            or int(row.get("task_expert", {}).get("step", -1))
            != member.expert_step
            or not trajectory.is_file()
            or trajectory.stat().st_size != int(capture.get("bytes", -1))
            or checkpoint is None
            or not (checkpoint / "adapter.safetensors").is_file()
        ):
            raise ValueError("phase decoder member capture changed")
        sources.append(
            PhaseMemberSource(
                member=member,
                trajectory_path=trajectory,
                trajectory_bytes=trajectory.stat().st_size,
                expert_checkpoint=checkpoint,
            )
        )
    return tuple(sources)


def _trajectory_batches(
    source: PhaseMemberSource, *, device: torch.device
) -> tuple[dict[str, torch.Tensor], ...]:
    value = torch.load(
        source.trajectory_path, map_location="cpu", weights_only=False
    )
    observations = tuple(value.get("observations", ()))
    actions = tuple(value.get("action_chunks", ()))
    indices = source.member.selected_replan_indices
    if (
        value.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or value.get("success") is not True
        or len(observations) != len(actions)
        or indices[-1] >= len(observations)
    ):
        raise ValueError("phase decoder successful trajectory changed")
    result = []
    for start in range(0, len(indices), 2):
        selected = indices[start : start + 2]
        keys = set(observations[selected[0]])
        if any(set(observations[index]) != keys for index in selected):
            raise ValueError("phase decoder observation keys changed")
        batch = {
            name: torch.cat([observations[index][name] for index in selected]).to(
                device, non_blocking=True
            )
            for name in sorted(keys)
        }
        batch[ACTION] = torch.cat([actions[index] for index in selected]).to(
            device, non_blocking=True
        )
        result.append(batch)
    if len(result) != 4 or any(int(batch[ACTION].shape[0]) != 2 for batch in result):
        raise ValueError("phase decoder member must expose four paired phase panels")
    return tuple(result)


def cache_phase_member_panels(
    *,
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    sources: tuple[PhaseMemberSource, ...],
    member_index: int,
    device: torch.device,
    policy_seed: int,
) -> tuple[CachedPhasePanel, ...]:
    """Cache four paired batch-two flow targets for one successful member."""

    source = sources[member_index]
    expert = load_file(
        str(source.expert_checkpoint / "adapter.safetensors"), device=str(device)
    )
    panels = []
    for panel_index, batch in enumerate(_trajectory_batches(source, device=device)):
        seed = int(policy_seed) + member_index * 101 + panel_index
        target = build_functional_response_target(
            policy,
            identity_state,
            expert,
            contract,
            batch,
            policy_seed=seed,
        )
        panels.append(CachedPhasePanel(batch=batch, target=target, policy_seed=seed))
    return tuple(panels)


def load_projected_occupancy_sources(
    *,
    root: Path,
    codes: PhaseDecoderCodeAuthority,
) -> dict[int, ProjectedOccupancySource]:
    """Bind every fit member to the decoded policy's matching trajectory."""

    resolved = root.resolve()
    contract = read_json(resolved / "run_contract.json")
    results = read_json(resolved / "results.json")
    capture = contract.get("diagnostic_occupancy_capture", {})
    adapter = contract.get("adapter", {})
    result_rows = tuple(results.get("rows", ()))
    rows = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): row
        for row in result_rows
    }
    fit_indices = [
        index
        for index, member in enumerate(codes.members)
        if member.fold_role == "fit"
    ]
    expected_keys = {
        (
            codes.members[index].suite,
            codes.members[index].task_id,
            codes.members[index].init_state_id,
        )
        for index in fit_indices
    }
    if (
        results.get("mode") != "formal"
        or results.get("role") != "development_train"
        or len(result_rows) != 30
        or len(rows) != 30
        or set(rows) != expected_keys
        or contract.get("git", {}).get("dirty_paths") != []
        or capture.get("schema_version")
        != "ember_phase_decoder_fit_projected_occupancy_capture_v1"
        or int(capture.get("selected_rows", -1)) != 30
        or int(capture.get("member_count", -1)) != 37
        or Path(str(capture.get("selection_path", ""))).resolve()
        != codes.root / "result.json"
        or adapter.get("schema_version")
        != "ember_pi05_functional_decoder_projected_task_expert_eval_adapter_v1"
        or adapter.get("projection", {}).get("schema")
        != "ember_phase_aligned_functional_decoder_train24_projection_v1"
    ):
        raise ValueError("phase decoder projected occupancy authority changed")
    sources = {}
    for index in fit_indices:
        member = codes.members[index]
        row = rows[(member.suite, member.task_id, member.init_state_id)]
        occupancy = row.get("occupancy_trajectory", {})
        path = Path(str(occupancy.get("path", ""))).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(occupancy.get("bytes", -1))
            or int(occupancy.get("replans", -1)) < 8
        ):
            raise ValueError("phase decoder projected trajectory changed")
        sources[index] = ProjectedOccupancySource(
            trajectory_path=path,
            trajectory_bytes=path.stat().st_size,
            success=bool(row["success"]),
        )
    return sources


def _projected_trajectory_batches(
    source: ProjectedOccupancySource, *, device: torch.device
) -> tuple[dict[str, torch.Tensor], ...]:
    value = torch.load(
        source.trajectory_path, map_location="cpu", weights_only=False
    )
    observations = tuple(value.get("observations", ()))
    actions = tuple(value.get("action_chunks", ()))
    if (
        value.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or bool(value.get("success")) != source.success
        or len(observations) != len(actions)
        or len(observations) < 8
    ):
        raise ValueError("phase decoder projected trajectory payload changed")
    action_sequence = torch.cat(actions).float().flatten(1)
    indices = tuple(
        int(value)
        for value in arc_length_phase_indices(action_sequence, count=8).tolist()
    )
    result = []
    for start in range(0, len(indices), 2):
        selected = indices[start : start + 2]
        keys = set(observations[selected[0]])
        if any(set(observations[index]) != keys for index in selected):
            raise ValueError("phase decoder projected observation keys changed")
        batch = {
            name: torch.cat([observations[index][name] for index in selected]).to(
                device, non_blocking=True
            )
            for name in sorted(keys)
        }
        batch[ACTION] = torch.cat([actions[index] for index in selected]).to(
            device, non_blocking=True
        )
        result.append(batch)
    return tuple(result)


def cache_projected_occupancy_panels(
    *,
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    member_sources: tuple[PhaseMemberSource, ...],
    occupancy_sources: Mapping[int, ProjectedOccupancySource],
    member_index: int,
    device: torch.device,
    policy_seed: int,
) -> tuple[CachedPhasePanel, ...]:
    """Query one privileged expert on four decoded-policy occupancy panels."""

    member_source = member_sources[member_index]
    occupancy = occupancy_sources[member_index]
    expert = load_file(
        str(member_source.expert_checkpoint / "adapter.safetensors"),
        device=str(device),
    )
    panels = []
    for panel_index, batch in enumerate(
        _projected_trajectory_batches(occupancy, device=device)
    ):
        seed = int(policy_seed) + member_index * 101 + panel_index
        target = build_functional_response_target(
            policy,
            identity_state,
            expert,
            contract,
            batch,
            policy_seed=seed,
        )
        panels.append(CachedPhasePanel(batch=batch, target=target, policy_seed=seed))
    if len(panels) != 4:
        raise ValueError("phase decoder projected occupancy must expose four panels")
    return tuple(panels)
