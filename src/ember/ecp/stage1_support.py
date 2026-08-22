"""Privileged multi-policy support evidence for ECP Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from lerobot.utils.constants import ACTION

from ember.ecp.policy_response import (
    FrozenTargetActivationEffects,
    TargetActivationEffectLoss,
    target_activation_effect_distillation_loss,
    target_activation_effects_from_payload,
)
from ember.functional_adaptation.functional_response import pi05_flow_response
from ember.lora import LoRAContract
from ember.pi05_source_checkpoint import read_json


SUPPORT_BANK_SCHEMA = "ember_ecp_stage1_policy_support_bank_v4"
SUPPORT_TASK_SCHEMA = "ember_ecp_stage1_policy_support_task_v4"
SUPPORT_CHANNELS = (
    "successful_expert_minus_source",
    "successful_shared_minus_source",
    "learner_expert_minus_source",
    "learner_policy_minus_source",
    "learner_shared_minus_source",
)
SUPPORT_PRESERVATION_RESPONSE_PROXIMITY = "response_proximity"
SUPPORT_PRESERVATION_BASELINE_BARRIER = "baseline_relative_response_barrier"


@dataclass(frozen=True)
class LearnerOccupancySource:
    ordinal: int
    suite: str
    task_id: int
    init_state_id: int
    success: bool
    trajectory_path: Path
    trajectory_bytes: int
    projected_adapter: Path
    projected_adapter_bytes: int


@dataclass(frozen=True)
class PolicySupportPanel:
    panel_id: int
    kind: str
    trajectory_path: Path
    trajectory_bytes: int
    selected_indices: tuple[int, int]
    policy_seed: int
    source_response: torch.Tensor
    shared_response: torch.Tensor
    expert_responses: torch.Tensor
    expert_weights: torch.Tensor
    outcome_weight: float
    source_support_weight: float
    shared_support_weight: float
    learner_success: bool | None
    activation_effects: FrozenTargetActivationEffects | None = None


@dataclass(frozen=True)
class CachedPolicySupportPanel:
    panel: PolicySupportPanel
    batch: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class PolicySupportTask:
    ordinal: int
    member_indices: tuple[int, ...]
    policy_response: torch.Tensor
    policy_response_weights: torch.Tensor
    panels: tuple[PolicySupportPanel, ...]

    def panel_for_visit(self, visit: int) -> PolicySupportPanel:
        successful = tuple(panel for panel in self.panels if panel.kind == "successful")
        learner = tuple(panel for panel in self.panels if panel.kind == "learner")
        if not successful:
            raise ValueError("policy-support task has no successful panel")
        family = learner if visit % 2 and learner else successful
        index = (visit // 2) % len(family)
        return family[index]


@dataclass(frozen=True)
class PolicySupportBank:
    root: Path
    tasks: Mapping[int, PolicySupportTask]

    def task(self, ordinal: int) -> PolicySupportTask:
        try:
            return self.tasks[int(ordinal)]
        except KeyError as error:
            raise ValueError(f"policy-support task {ordinal} was not loaded") from error


@dataclass(frozen=True)
class PolicySupportLoss:
    successful_response: torch.Tensor
    learner_response: torch.Tensor
    source_support: torch.Tensor
    shared_support: torch.Tensor
    expert_set_disagreement: torch.Tensor

    @property
    def response(self) -> torch.Tensor:
        return self.successful_response + self.learner_response


def load_learner_occupancy_sources(
    *, root: Path, tasks: Sequence[Any]
) -> dict[int, tuple[LearnerOccupancySource, ...]]:
    """Load the target-fit learner trajectories without old decoder objects."""

    resolved = root.resolve()
    contract = read_json(resolved / "run_contract.json")
    results = read_json(resolved / "results.json")
    capture = contract.get("diagnostic_occupancy_capture", {})
    adapter = contract.get("adapter", {})
    rows = tuple(results.get("rows", ()))
    task_by_key = {(str(task.suite), int(task.task_id)): task for task in tasks}
    if (
        results.get("mode") != "formal"
        or results.get("role") != "development_train"
        or len(rows) != 30
        or len(
            {
                (row["suite"], int(row["task_id"]), int(row["init_state_id"]))
                for row in rows
            }
        )
        != 30
        or contract.get("git", {}).get("dirty_paths") != []
        or capture.get("schema_version")
        != "ember_phase_decoder_fit_projected_occupancy_capture_v1"
        or int(capture.get("selected_rows", -1)) != 30
        or int(capture.get("selected_tasks", -1)) != 19
        or capture.get("held_data_use") is not False
        or adapter.get("schema_version")
        != "ember_pi05_functional_decoder_projected_task_expert_eval_adapter_v1"
    ):
        raise ValueError("ECP learner-occupancy authority changed")
    result: dict[int, list[LearnerOccupancySource]] = {}
    for row in rows:
        ordinal, source = _learner_occupancy_source(row, task_by_key)
        result.setdefault(ordinal, []).append(source)
    expected = {
        int(task.ordinal)
        for task in tasks
        if task.fold_role == "fit" and getattr(task, "domain", None) == "target_train"
    }
    if len(expected) != 19 or set(result) != expected:
        raise ValueError("learner occupancy no longer covers target fit19")
    return {
        ordinal: tuple(sorted(values, key=lambda row: row.init_state_id))
        for ordinal, values in result.items()
    }


def _learner_occupancy_source(
    row: Mapping[str, Any], task_by_key: Mapping[tuple[str, int], Any]
) -> tuple[int, LearnerOccupancySource]:
    key = (str(row["suite"]), int(row["task_id"]))
    task = task_by_key.get(key)
    occupancy = row.get("occupancy_trajectory", {})
    expert = row.get("task_expert", {})
    trajectory = Path(str(occupancy.get("path", ""))).resolve()
    projected = Path(str(expert.get("projected_adapter", ""))).resolve()
    if (
        task is None
        or task.fold_role != "fit"
        or getattr(task, "domain", None) != "target_train"
        or not trajectory.is_file()
        or trajectory.stat().st_size != int(occupancy.get("bytes", -1))
        or int(occupancy.get("replans", -1)) < 8
        or not projected.is_file()
        or projected.stat().st_size
        != int(expert.get("projected_adapter_bytes", -1))
        or int(expert.get("global_task_id", -1)) != int(task.global_task_id)
    ):
        raise ValueError("ECP learner-occupancy row changed")
    source = LearnerOccupancySource(
        ordinal=int(task.ordinal),
        suite=key[0],
        task_id=key[1],
        init_state_id=int(row["init_state_id"]),
        success=bool(row["success"]),
        trajectory_path=trajectory,
        trajectory_bytes=trajectory.stat().st_size,
        projected_adapter=projected,
        projected_adapter_bytes=projected.stat().st_size,
    )
    return int(task.ordinal), source


def _panel_from_payload(
    value: Mapping[str, Any], *, contract: LoRAContract
) -> PolicySupportPanel:
    selected = tuple(int(index) for index in value["selected_indices"])
    expert_responses = value["expert_responses"].float()
    panel = PolicySupportPanel(
        panel_id=int(value["panel_id"]),
        kind=str(value["kind"]),
        trajectory_path=Path(str(value["trajectory_path"])).resolve(),
        trajectory_bytes=int(value["trajectory_bytes"]),
        selected_indices=(selected[0], selected[1]),
        policy_seed=int(value["policy_seed"]),
        source_response=value["source_response"].float(),
        shared_response=value["shared_response"].float(),
        expert_responses=expert_responses,
        expert_weights=value["expert_weights"].float(),
        outcome_weight=float(value["outcome_weight"]),
        source_support_weight=float(value["source_support_weight"]),
        shared_support_weight=float(value["shared_support_weight"]),
        learner_success=(
            None
            if value.get("learner_success") is None
            else bool(value["learner_success"])
        ),
        activation_effects=target_activation_effects_from_payload(
            value,
            contract=contract,
            expert_count=int(expert_responses.shape[0]),
        ),
    )
    if (
        panel.kind not in {"successful", "learner"}
        or len(set(panel.selected_indices)) != 2
        or panel.source_response.ndim != 3
        or panel.shared_response.shape != panel.source_response.shape
        or panel.expert_responses.ndim != 4
        or panel.expert_responses.shape[1:] != panel.source_response.shape
        or panel.expert_weights.shape != (panel.expert_responses.shape[0],)
        or not panel.trajectory_path.is_file()
        or panel.trajectory_path.stat().st_size != panel.trajectory_bytes
        or not 0.0 < panel.outcome_weight <= 1.0
        or not 0.0 <= panel.source_support_weight <= 1.0
        or not 0.0 <= panel.shared_support_weight <= 1.0
    ):
        raise ValueError("ECP policy-support panel changed")
    return panel


def load_policy_support_bank(
    *,
    manifest_path: Path,
    evidence_bank: Any,
    contract: LoRAContract,
    task_ordinals: set[int],
    device: torch.device,
) -> PolicySupportBank:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    rows = {int(row["ordinal"]): row for row in manifest.get("tasks", ())}
    learner_required = {
        int(value)
        for value in manifest.get("learner_panels_required_task_ordinals", ())
    }
    if (
        manifest.get("schema_version") != SUPPORT_BANK_SCHEMA
        or tuple(manifest.get("support_channels", ())) != SUPPORT_CHANNELS
        or int(manifest.get("event_slots", -1)) != 8
        or int(manifest.get("owners", -1)) != 38
        or int(manifest.get("horizon_basis", -1)) != 4
        or int(manifest.get("program_width", -1)) != 128
        or manifest.get("target_local_activation_effect_panels") is not True
        or set(rows) != set(range(95))
        or len(learner_required) != 19
        or not learner_required <= set(rows)
        or not task_ordinals <= set(rows)
    ):
        raise ValueError("ECP policy-support bank manifest changed")
    tasks = {}
    for ordinal in sorted(task_ordinals):
        row = rows[ordinal]
        path = (manifest_path.parent / str(row["file"])).resolve()
        tasks[ordinal] = _load_policy_support_task(
            path=path,
            expected_bytes=int(row["bytes"]),
            ordinal=ordinal,
            evidence_bank=evidence_bank,
            contract=contract,
            device=device,
            learner_required=ordinal in learner_required,
        )
    return PolicySupportBank(root=manifest_path.parent, tasks=tasks)


def _load_policy_support_task(
    *,
    path: Path,
    expected_bytes: int,
    ordinal: int,
    evidence_bank: Any,
    contract: LoRAContract,
    device: torch.device,
    learner_required: bool,
) -> PolicySupportTask:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError("ECP policy-support task asset changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    member_indices = tuple(int(index) for index in value["member_indices"])
    response = value["policy_response"]
    weights = value["policy_response_weights"]
    panels = tuple(
        _panel_from_payload(panel, contract=contract) for panel in value["panels"]
    )
    expected_members = evidence_bank.member_indices(ordinal)
    invalid = (
        value.get("schema_version") != SUPPORT_TASK_SCHEMA
        or int(value.get("ordinal", -1)) != ordinal
        or member_indices != expected_members
        or response.shape
        != (len(member_indices), 8, 38, len(SUPPORT_CHANNELS), 4, 128)
        or weights.shape != (len(member_indices), 8, len(SUPPORT_CHANNELS))
        or not torch.isfinite(response).all()
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or not panels
        or (learner_required and not any(panel.kind == "learner" for panel in panels))
        or any(panel.activation_effects is None for panel in panels)
    )
    if invalid:
        raise ValueError("ECP policy-support task payload changed")
    return PolicySupportTask(
        ordinal=ordinal,
        member_indices=member_indices,
        policy_response=response.to(device=device, dtype=torch.float32),
        policy_response_weights=weights.to(device=device, dtype=torch.float32),
        panels=panels,
    )


def _panel_batch_from_trajectory(
    panel: PolicySupportPanel,
    value: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observations = tuple(value.get("observations", ()))
    actions = tuple(value.get("action_chunks", ()))
    if (
        value.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or len(observations) != len(actions)
        or max(panel.selected_indices) >= len(observations)
        or (panel.kind == "successful" and value.get("success") is not True)
        or (
            panel.kind == "learner"
            and bool(value.get("success")) != panel.learner_success
        )
    ):
        raise ValueError("ECP policy-support trajectory payload changed")
    keys = set(observations[panel.selected_indices[0]])
    if any(set(observations[index]) != keys for index in panel.selected_indices):
        raise ValueError("ECP policy-support observation keys changed")
    batch = {
        name: torch.cat(
            [observations[index][name] for index in panel.selected_indices]
        ).to(device, non_blocking=True)
        for name in sorted(keys)
    }
    batch[ACTION] = torch.cat(
        [actions[index] for index in panel.selected_indices]
    ).to(device, non_blocking=True)
    return batch


def cache_policy_support_panels(
    *,
    bank: PolicySupportBank,
    requests: set[tuple[int, int]],
    device: torch.device,
) -> dict[tuple[int, int], CachedPolicySupportPanel]:
    """Load only panels that this rank will actually query."""

    selected: dict[Path, list[tuple[int, PolicySupportPanel]]] = {}
    for ordinal, panel_id in sorted(requests):
        task = bank.task(ordinal)
        matches = tuple(panel for panel in task.panels if panel.panel_id == panel_id)
        if len(matches) != 1:
            raise ValueError("ECP policy-support panel identity changed")
        panel = matches[0]
        selected.setdefault(panel.trajectory_path, []).append((ordinal, panel))
    result = {}
    for trajectory_path, rows in selected.items():
        value = torch.load(
            trajectory_path, map_location="cpu", weights_only=False
        )
        for ordinal, panel in rows:
            panel_id = panel.panel_id
            target = PolicySupportPanel(
                **{
                    **panel.__dict__,
                    "source_response": panel.source_response.to(
                        device, non_blocking=True
                    ),
                    "shared_response": panel.shared_response.to(
                        device, non_blocking=True
                    ),
                    "expert_responses": panel.expert_responses.to(
                        device, non_blocking=True
                    ),
                    "expert_weights": panel.expert_weights.to(
                        device, non_blocking=True
                    ),
                    "activation_effects": (
                        None
                        if panel.activation_effects is None
                        else panel.activation_effects.to(device)
                    ),
                }
            )
            result[(ordinal, panel_id)] = CachedPolicySupportPanel(
                panel=target,
                batch=_panel_batch_from_trajectory(
                    panel, value, device=device
                ),
            )
    return result


def policy_support_distillation_loss(
    *,
    policy: torch.nn.Module,
    candidate_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    cached: CachedPolicySupportPanel,
    preservation: str = SUPPORT_PRESERVATION_RESPONSE_PROXIMITY,
) -> PolicySupportLoss:
    panel = cached.panel
    candidate = pi05_flow_response(
        policy,
        candidate_state,
        contract,
        cached.batch,
        policy_seed=panel.policy_seed,
    ).float()
    return policy_support_loss_from_response(
        candidate=candidate,
        panel=panel,
        preservation=preservation,
    )


def policy_support_activation_distillation_loss(
    *,
    policy: torch.nn.Module,
    candidate_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    cached: CachedPolicySupportPanel,
    preservation: str,
) -> tuple[PolicySupportLoss, TargetActivationEffectLoss]:
    panel = cached.panel
    if panel.activation_effects is None:
        raise ValueError("target-local activation panel is unavailable")
    candidate = pi05_flow_response(
        policy,
        candidate_state,
        contract,
        cached.batch,
        policy_seed=panel.policy_seed,
    ).float()
    support = policy_support_loss_from_response(
        candidate=candidate,
        panel=panel,
        preservation=preservation,
    )
    activation = target_activation_effect_distillation_loss(
        candidate_state=candidate_state,
        targets=panel.activation_effects,
        contract=contract,
        expert_weights=panel.expert_weights,
        outcome_weight=panel.outcome_weight,
    )
    return support, activation


def shared_prior_response_distillation_loss(
    *,
    policy: torch.nn.Module,
    candidate_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    cached: CachedPolicySupportPanel,
) -> torch.Tensor:
    """Match a learned prior-only compiler output to the stable shared policy."""

    panel = cached.panel
    candidate = pi05_flow_response(
        policy,
        candidate_state,
        contract,
        cached.batch,
        policy_seed=panel.policy_seed,
    ).float()
    source = panel.source_response.to(candidate).float()
    shared = panel.shared_response.to(candidate).float()
    experts = panel.expert_responses.to(candidate).float()
    weights = panel.expert_weights.to(candidate).clamp_min(1e-4)
    weights = weights / weights.sum()
    consensus = torch.einsum("m,mbhd->bhd", weights, experts)
    normalization = (consensus - source).square().mean().clamp_min(1e-8)
    return (candidate - shared).square().mean() / normalization


def policy_support_loss_from_response(
    *,
    candidate: torch.Tensor,
    panel: PolicySupportPanel,
    preservation: str = SUPPORT_PRESERVATION_RESPONSE_PROXIMITY,
) -> PolicySupportLoss:
    """Score a frozen response against one cached multi-policy support panel."""

    source = panel.source_response.to(candidate).float()
    shared = panel.shared_response.to(candidate).float()
    experts = panel.expert_responses.to(candidate).float()
    member_weights = panel.expert_weights.to(candidate).clamp_min(1e-4)
    member_weights = member_weights / member_weights.sum()
    expert_energy = (experts - source[None]).square().mean(dim=(1, 2, 3))

    def expert_response_error(value: torch.Tensor) -> torch.Tensor:
        error = (experts - value[None]).square().mean(dim=(1, 2, 3))
        normalized = member_weights * error / expert_energy.clamp_min(1e-8)
        return normalized.sum() * float(panel.outcome_weight)

    response = expert_response_error(candidate)
    consensus = torch.einsum("m,mbhd->bhd", member_weights, experts)
    normalization = (consensus - source).square().mean().clamp_min(1e-8)
    if preservation == SUPPORT_PRESERVATION_RESPONSE_PROXIMITY:
        source_support = (
            (candidate - source).square().mean()
            / normalization
            * float(panel.source_support_weight)
        )
        shared_support = (
            (candidate - shared).square().mean()
            / normalization
            * float(panel.shared_support_weight)
        )
    elif preservation == SUPPORT_PRESERVATION_BASELINE_BARRIER:
        source_support = torch.relu(response - expert_response_error(source))
        shared_support = torch.relu(response - expert_response_error(shared))
        source_support = source_support * float(panel.source_support_weight)
        shared_support = shared_support * float(panel.shared_support_weight)
    else:
        raise ValueError(f"unsupported policy-support preservation: {preservation}")
    disagreement = torch.einsum(
        "m,mbhd->bhd", member_weights, (experts - consensus[None]).square()
    ).mean() / normalization
    zero = candidate.new_zeros(())
    return PolicySupportLoss(
        successful_response=response if panel.kind == "successful" else zero,
        learner_response=response if panel.kind == "learner" else zero,
        source_support=source_support,
        shared_support=shared_support,
        expert_set_disagreement=disagreement,
    )
