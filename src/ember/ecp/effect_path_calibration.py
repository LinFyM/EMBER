"""Objectives and adjudication for known-success ECP rank4 paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.ecp.policy_effects import PolicyEffectResponse
from ember.ecp.stage1_equivalence import CATEGORY_NAMES, Stage1EffectBank
from ember.ecp.stage1_objective import (
    RealizationConfig,
    member_distances,
    member_response_scales,
    reference_distances,
)
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic


TASK_RESULT_SCHEMA = "ember_ecp_effect_path_calibration_task_v1"
AGGREGATE_SCHEMA = "ember_ecp_effect_path_calibration_gate_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VerifiedMemberObjective:
    scales: PolicyEffectResponse
    validity: torch.Tensor
    category_ids: torch.Tensor
    reliability: torch.Tensor
    temperature: float


def build_verified_member_validity(
    *,
    anchors: Sequence[Mapping[str, Any]],
    member_names: Sequence[str],
    initial_success: Mapping[str, Mapping[int, bool]],
) -> torch.Tensor:
    """Build the conservative member-state mask fixed by the Phase 2A card."""

    if len(anchors) != 48 or len(member_names) != 3:
        raise ValueError("effect-path member-state panel changed")
    validity = torch.zeros(len(member_names), len(anchors), dtype=torch.bool)
    member_index = {name: index for index, name in enumerate(member_names)}
    if len(member_index) != len(member_names):
        raise ValueError("effect-path member names overlap")
    for state, anchor in enumerate(anchors):
        category = str(anchor["category"])
        if category == "initial":
            init_state_id = int(anchor["init_state_id"])
            for member, index in member_index.items():
                validity[index, state] = bool(
                    initial_success[member].get(init_state_id, False)
                )
        elif category == "successful":
            generator = str(anchor["generator"])
            if generator not in member_index:
                raise ValueError("successful anchor lost its generating member")
            validity[member_index[generator], state] = True
        elif category not in {"candidate", "recovery"}:
            raise ValueError("effect-path anchor category changed")
    if any(int(validity[index, 8:32].sum()) != 8 for index in range(3)):
        raise ValueError("on-policy successful validity changed")
    return validity


def build_verified_member_objective(
    bank: Stage1EffectBank,
    validity: torch.Tensor,
    config: RealizationConfig,
) -> VerifiedMemberObjective:
    bank.validate()
    if validity.shape != (bank.member_count, bank.state_count):
        raise ValueError("verified member-state validity changed shape")
    reliability = bank.member_reliability.float().clamp_min(1e-4)
    reliability = reliability / reliability.sum()
    return VerifiedMemberObjective(
        scales=member_response_scales(bank, config),
        validity=validity.bool(),
        category_ids=bank.category_ids,
        reliability=reliability,
        temperature=float(config.temperature),
    )


def verified_member_losses(
    candidate: PolicyEffectResponse,
    bank: Stage1EffectBank,
    objective: VerifiedMemberObjective,
    config: RealizationConfig,
) -> torch.Tensor:
    """Score each global member only on its verified trajectory support."""

    per_state = member_distances(candidate, bank.members, objective.scales, config)
    losses = []
    for member in range(bank.member_count):
        category_losses = []
        for category in (0, 1):
            mask = objective.validity[member] & (objective.category_ids == category)
            if torch.any(mask):
                category_losses.append(per_state[member, mask].mean())
        if not category_losses:
            raise ValueError("global member has no verified states")
        losses.append(torch.stack(category_losses).mean())
    return torch.stack(losses)


def global_particle_loss(
    member_losses: torch.Tensor,
    objective: VerifiedMemberObjective,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = objective.reliability.log() - member_losses / objective.temperature
    return (
        -objective.temperature * torch.logsumexp(logits, dim=0),
        torch.softmax(logits, dim=0),
    )


def carrier_drift_by_category(
    candidate: PolicyEffectResponse,
    bank: Stage1EffectBank,
    objective: VerifiedMemberObjective,
    config: RealizationConfig,
) -> dict[str, float]:
    per_state = reference_distances(
        candidate, bank.carrier, objective.scales, config
    )
    return {
        name: float(per_state[bank.category_ids == category].mean())
        for category, name in enumerate(CATEGORY_NAMES)
    }


def summarize_task_paths(
    rows: Sequence[Mapping[str, Any]], members: Sequence[str]
) -> dict[str, Any]:
    endpoint_improved = 0
    first_improved = 0
    late_minimum = 0
    per_member = {}
    for member in members:
        path = sorted(
            (row for row in rows if row["member"] == member),
            key=lambda row: float(row["alpha"]),
        )
        carrier = float(path[0]["matching_verified_loss"])
        endpoint = float(path[-1]["matching_verified_loss"])
        first = float(path[1]["matching_verified_loss"])
        minimum = min(float(row["matching_verified_loss"]) for row in path)
        late = min(
            float(row["matching_verified_loss"])
            for row in path
            if float(row["alpha"]) >= 0.75
        )
        flags = {
            "endpoint_improved": endpoint < carrier,
            "first_nonzero_improved": first < carrier,
            "minimum_is_late": late <= minimum + 1e-9,
        }
        endpoint_improved += int(flags["endpoint_improved"])
        first_improved += int(flags["first_nonzero_improved"])
        late_minimum += int(flags["minimum_is_late"])
        per_member[member] = flags
    carrier_global = float(
        next(row for row in rows if float(row["alpha"]) == 0.0)[
            "global_particle_loss"
        ]
    )
    best_endpoint_global = min(
        float(row["global_particle_loss"])
        for row in rows
        if float(row["alpha"]) == 1.0
    )
    return {
        "endpoint_improved_members": endpoint_improved,
        "first_nonzero_improved_members": first_improved,
        "late_minimum_members": late_minimum,
        "any_first_nonzero_improved": first_improved > 0,
        "carrier_global_particle_loss": carrier_global,
        "best_endpoint_global_particle_loss": best_endpoint_global,
        "global_particle_improved": best_endpoint_global < carrier_global,
        "per_member": per_member,
    }


def aggregate_results(
    *, config: Mapping[str, Any], root: Path, output: Path
) -> Path:
    results = [read_json(path) for path in sorted(root.glob("task_*.json"))]
    expected = tuple(int(value) for value in config["tasks"]["ordinals"])
    by_ordinal = {int(value["task"]["ordinal"]): value for value in results}
    if set(by_ordinal) != set(expected) or any(
        value.get("schema_version") != TASK_RESULT_SCHEMA for value in results
    ):
        raise ValueError("effect-path task results are incomplete")
    summaries = [by_ordinal[ordinal]["summary"] for ordinal in expected]
    endpoint = sum(int(value["endpoint_improved_members"]) for value in summaries)
    first = sum(int(value["first_nonzero_improved_members"]) for value in summaries)
    late = sum(int(value["late_minimum_members"]) for value in summaries)
    first_task_coverage = sum(
        bool(value["any_first_nonzero_improved"]) for value in summaries
    )
    global_tasks = sum(bool(value["global_particle_improved"]) for value in summaries)
    thresholds = config["gate"]
    gate = {
        "matching_endpoint_improvement": endpoint,
        "matching_endpoint_improvement_pass": endpoint
        >= int(thresholds["matching_endpoint_improvement_required"]),
        "first_nonzero_improvement": first,
        "first_nonzero_improvement_pass": first
        >= int(thresholds["first_nonzero_improvement_minimum"]),
        "first_nonzero_task_coverage": first_task_coverage,
        "first_nonzero_task_coverage_pass": first_task_coverage
        >= int(thresholds["task_coverage_required"]),
        "late_path_minimum": late,
        "late_path_minimum_pass": late
        >= int(thresholds["late_path_minimum_minimum"]),
        "global_objective_improved_tasks": global_tasks,
        "global_objective_improved_tasks_pass": global_tasks
        >= int(thresholds["global_objective_improved_tasks_required"]),
    }
    single_pass = bool(
        gate["matching_endpoint_improvement_pass"]
        and gate["first_nonzero_improvement_pass"]
        and gate["first_nonzero_task_coverage_pass"]
        and gate["late_path_minimum_pass"]
    )
    global_pass = bool(gate["global_objective_improved_tasks_pass"])
    gate["single_member_pass"] = single_pass
    gate["global_particle_pass"] = global_pass
    gate["pass"] = single_pass and global_pass
    decision = (
        "pass_proceed_balanced_svd_coordinate"
        if gate["pass"]
        else (
            "single_member_supported_mixture_non_pass"
            if single_pass
            else "effect_coordinate_non_pass"
        )
    )
    payload = {
        "schema_version": AGGREGATE_SCHEMA,
        "date": "2026-08-24",
        "decision": decision,
        "repository": git_state(REPO_ROOT),
        "task_results": [
            {
                "ordinal": ordinal,
                "global_task_id": int(by_ordinal[ordinal]["task"]["global_task_id"]),
                "path": str((root / f"task_{ordinal}.json").resolve()),
                "summary": by_ordinal[ordinal]["summary"],
            }
            for ordinal in expected
        ],
        "gate": gate,
        "information_wall": dict(config["information_wall"]),
    }
    if output.exists():
        raise ValueError("effect-path aggregate output already exists")
    write_json_atomic(output, payload)
    return output
