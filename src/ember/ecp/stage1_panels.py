"""Successful multi-phase functional panels for ECP Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from lerobot.utils.constants import ACTION

from ember.ecp.stage1_data import ECPStage1EvidenceBank, ECPStage1Member
from ember.functional_adaptation.functional_response import (
    FunctionalResponseTarget,
    build_functional_response_target,
)
from ember.lora import LoRAContract


@dataclass(frozen=True)
class ECPStage1FunctionalPanel:
    batch: Mapping[str, torch.Tensor]
    target: FunctionalResponseTarget
    policy_seed: int


def _trajectory_batches(
    member: ECPStage1Member, *, device: torch.device
) -> tuple[dict[str, torch.Tensor], ...]:
    value = torch.load(
        member.trajectory_path, map_location="cpu", weights_only=False
    )
    observations = tuple(value.get("observations", ()))
    actions = tuple(value.get("action_chunks", ()))
    indices = member.selected_replan_indices
    if (
        value.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or value.get("success") is not True
        or len(observations) != len(actions)
        or indices[-1] >= len(observations)
    ):
        raise ValueError("ECP Stage 1 successful occupancy changed")
    panels = []
    for start in range(0, len(indices), 2):
        selected = indices[start : start + 2]
        keys = set(observations[selected[0]])
        if any(set(observations[index]) != keys for index in selected):
            raise ValueError("ECP Stage 1 observation panel keys changed")
        batch = {
            name: torch.cat([observations[index][name] for index in selected]).to(
                device, non_blocking=True
            )
            for name in sorted(keys)
        }
        batch[ACTION] = torch.cat([actions[index] for index in selected]).to(
            device, non_blocking=True
        )
        panels.append(batch)
    if len(panels) != 4:
        raise ValueError("ECP Stage 1 member must expose four phase panels")
    return tuple(panels)


def cache_stage1_functional_panels(
    *,
    policy: torch.nn.Module,
    identity_state: Mapping[str, torch.Tensor],
    evidence_bank: ECPStage1EvidenceBank,
    contract: LoRAContract,
    device: torch.device,
    policy_seed: int,
    fit_only: bool,
    member_indices: set[int] | None = None,
) -> dict[int, tuple[ECPStage1FunctionalPanel, ...]]:
    result = {}
    for member in evidence_bank.members:
        if fit_only and member.fold_role != "fit":
            continue
        if member_indices is not None and member.index not in member_indices:
            continue
        expert = {
            name: value[member.index]
            for name, value in evidence_bank.member_states.items()
        }
        panels = []
        for panel_index, batch in enumerate(
            _trajectory_batches(member, device=device)
        ):
            seed = int(policy_seed) + member.index * 101 + panel_index
            panels.append(
                ECPStage1FunctionalPanel(
                    batch=batch,
                    target=build_functional_response_target(
                        policy,
                        identity_state,
                        expert,
                        contract,
                        batch,
                        policy_seed=seed,
                    ),
                    policy_seed=seed,
                )
            )
        result[member.index] = tuple(panels)
    return result
