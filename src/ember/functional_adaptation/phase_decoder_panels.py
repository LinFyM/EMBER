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
