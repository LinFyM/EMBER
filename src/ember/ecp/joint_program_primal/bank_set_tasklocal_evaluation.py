"""Final diagnostics and formal aggregation for EBSRI S0/S1."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.ecp.contracts import TargetFamily
from ember.ecp.joint_program_primal.bank_set_tasklocal_contract import (
    BANK_CONDITIONED_PRIMAL_STAGE,
    BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA,
    BANK_SET_TASKLOCAL_SCHEMA,
)
from ember.ecp.joint_program_primal.routing_control import (
    BANK_SET_S0_STAGE,
    BANK_SET_S1_STAGE,
    load_routing_control_config,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


TASKS = (1, 93)
FAMILIES = tuple(TargetFamily)
RESULT_SCHEMA = "ember_ecp_program_through_bank_tasklocal_result_v1"
AGGREGATE_SCHEMA = BANK_SET_TASKLOCAL_AGGREGATE_SCHEMA


def _free_parameter_roots(names: Sequence[str]) -> set[str]:
    return {name.split(".", 1)[0] for name in names if name.startswith("free_")}


@dataclass(frozen=True)
class EffectiveTarget:
    a: tuple[torch.Tensor, ...]
    b: tuple[torch.Tensor, ...]


class CorrectionCollector:
    """Accumulate bounded correction moments without retaining candidate tensors."""

    _HISTOGRAM_BINS = 128

    def __init__(self, bound: float) -> None:
        self.bound = float(bound)
        self._rows: dict[
            tuple[str, str],
            list[
                tuple[
                    int,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]
            ],
        ] = defaultdict(list)

    def observe(self, side: str, owner: Any, correction: torch.Tensor) -> None:
        if side not in {"input", "output"} or correction.numel() <= 0:
            raise RuntimeError("bank-set correction diagnostic axes changed")
        value = correction.detach().float().reshape(-1)
        absolute = value.abs()
        self._rows[(side, owner.family.value)].append(
            (
                int(value.numel()),
                value.sum(),
                value.square().sum(),
                absolute.max(),
                (absolute >= 0.95 * self.bound).sum(),
                torch.histc(
                    absolute.clamp_max(self.bound),
                    bins=self._HISTOGRAM_BINS,
                    min=0.0,
                    max=self.bound,
                ),
            )
        )

    def _statistics(self, keys: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        rows = tuple(row for key in keys for row in self._rows[key])
        count = sum(row[0] for row in rows)
        signed_sum = torch.stack(tuple(row[1] for row in rows)).sum()
        square_sum = torch.stack(tuple(row[2] for row in rows)).sum()
        maximum = torch.stack(tuple(row[3] for row in rows)).max()
        near_bound = torch.stack(tuple(row[4] for row in rows)).sum()
        histogram = torch.stack(tuple(row[5] for row in rows)).sum(0)
        percentile_rank = max(1, math.ceil(0.95 * count))
        percentile_bin = int(
            torch.searchsorted(
                histogram.cumsum(0),
                histogram.new_tensor(float(percentile_rank)),
            )
        )
        return {
            "count": count,
            "rms": float((square_sum / count).sqrt()),
            "p95_absolute": min(
                self.bound,
                (percentile_bin + 1) * self.bound / self._HISTOGRAM_BINS,
            ),
            "maximum_absolute": float(maximum),
            "signed_mean": float(signed_sum / count),
            "near_bound_fraction": float(near_bound / count),
        }

    def finalize(self) -> dict[str, Any]:
        expected = {
            (side, family.value)
            for side in ("input", "output")
            for family in FAMILIES
        }
        if set(self._rows) != expected:
            raise RuntimeError("bank-set correction diagnostic family coverage changed")
        return {
            "bound": self.bound,
            "near_bound_definition": "absolute_correction_at_least_0.95_times_bound",
            "p95_method": (
                f"upper_edge_of_{self._HISTOGRAM_BINS}_bin_online_histogram"
            ),
            "all": self._statistics(tuple(sorted(self._rows))),
            "by_side_family": {
                f"{side}:{family}": self._statistics(((side, family),))
                for side, family in sorted(self._rows)
            },
        }


def effective_rank4_diagnostics(
    runtime: Any,
    output: Any,
    target: EffectiveTarget,
    denominators: Mapping[TargetFamily, torch.Tensor],
) -> dict[str, Any]:
    family_rows: dict[TargetFamily, dict[str, torch.Tensor]] = {
        family: {
            "candidate_norm": output.residual.scales.new_zeros(()),
            "target_norm": output.residual.scales.new_zeros(()),
            "cross": output.residual.scales.new_zeros(()),
            "squared_error": output.residual.scales.new_zeros(()),
        }
        for family in FAMILIES
    }
    targets = []
    for owner, actual_a, actual_b, target_a, target_b in zip(
        runtime.owners,
        output.residual.a,
        output.residual.b,
        target.a,
        target.b,
        strict=True,
    ):
        actual_b_matrix = actual_b.float().transpose(0, 1)
        target_b_matrix = target_b.float().transpose(0, 1)
        candidate_norm = (
            (actual_b_matrix.transpose(0, 1) @ actual_b_matrix)
            * (actual_a.float() @ actual_a.float().transpose(0, 1))
        ).sum()
        target_norm = (
            (target_b_matrix.transpose(0, 1) @ target_b_matrix)
            * (target_a.float() @ target_a.float().transpose(0, 1))
        ).sum()
        cross = (
            (actual_b_matrix.transpose(0, 1) @ target_b_matrix)
            * (actual_a.float() @ target_a.float().transpose(0, 1))
        ).sum()
        squared_error = (candidate_norm + target_norm - 2.0 * cross).clamp_min(0)
        cosine = cross / (candidate_norm * target_norm).clamp_min(1e-24).sqrt()
        row = family_rows[owner.family]
        row["candidate_norm"] = row["candidate_norm"] + candidate_norm
        row["target_norm"] = row["target_norm"] + target_norm
        row["cross"] = row["cross"] + cross
        row["squared_error"] = row["squared_error"] + squared_error
        targets.append(
            {
                "target": owner.target_name,
                "family": owner.family.value,
                "squared_error": float(squared_error),
                "cosine": float(cosine.clamp(-1.0, 1.0)),
            }
        )
    families = {}
    for family, row in family_rows.items():
        cosine = row["cross"] / (
            row["candidate_norm"] * row["target_norm"]
        ).clamp_min(1e-24).sqrt()
        families[family.value] = {
            "squared_error": float(row["squared_error"]),
            "normalized_squared_error": float(
                row["squared_error"] / denominators[family]
            ),
            "cosine": float(cosine.clamp(-1.0, 1.0)),
        }
    return {
        "families": families,
        "targets": targets,
        "negative_family_cosines": sorted(
            family for family, row in families.items() if row["cosine"] < 0.0
        ),
    }


def _formal_checkpoint_valid(root: Path, *, macro: int, stage: str) -> bool:
    checkpoint = root / "checkpoints" / f"macro_{macro:08d}"
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    return all(
        (
            manifest.get("stage") == stage,
            int(manifest.get("next_macro", -1)) == macro,
            int(manifest.get("world_size", -1)) == 1,
            set(files)
            == {"ecp.safetensors", "trainer_state.pt", "rank_00_state.pt"},
            all(
                (checkpoint / name).is_file()
                and (checkpoint / name).stat().st_size == int(record.get("bytes", -1))
                for name, record in files.items()
            ),
        )
    )


def _metric_rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_formal_task(
    root: Path, *, stage: str, config_path: Path, config: Mapping[str, Any]
) -> tuple[int, str, dict[str, Any]]:
    result = read_json(root / "evaluation.json")
    completion = read_json(root / "completion.json")
    contract = read_json(root / "run_contract.json")
    metrics = _metric_rows(root)
    task = int(result.get("task", -1))
    qualification = contract.get("task_local_qualification", {})
    inventory = contract.get("inventory", {})
    arm_names = [row.get("name") for row in qualification.get("arms", ())]
    expected_model = dict(config["model"])
    expected_model["replay_frame_chunk_size"] = int(
        expected_model["replay_frame_chunk_size_by_task"][str(task)]
    )
    if stage != BANK_CONDITIONED_PRIMAL_STAGE:
        expected_model["interaction_group_batch_size"] = int(
            expected_model["interaction_group_batch_size_by_task"][str(task)]
        )
    expected_optimization = dict(config["optimization"])
    expected_optimization["functional_policy_microbatch_size"] = int(
        expected_optimization["functional_policy_microbatch_size_by_task"][str(task)]
    )
    trainable_names = inventory.get("writer_trainable_parameter_names", ())
    expected_free = (
        {"free_correct", "free_wrong"} if stage == BANK_SET_S0_STAGE else set()
    )
    expected_set_encoder_trainable = stage in {
        BANK_SET_S1_STAGE,
        BANK_CONDITIONED_PRIMAL_STAGE,
    }
    teacher = result.get("wrong_teacher", {})
    summary_mass = result.get("summary_event_mass")
    valid = all(
        (
            task in TASKS,
            result.get("schema_version") == RESULT_SCHEMA,
            result.get("status") == "complete",
            result.get("stage") == stage,
            int(result.get("completed_optimizer_steps", -1)) == 110,
            completion.get("stage") == stage,
            int(completion.get("task", -1)) == task,
            int(completion.get("completed_optimizer_steps", -1)) == 110,
            completion.get("gate") == result.get("evaluation", {}).get("gate"),
            contract.get("stage") == stage,
            contract.get("mode") == "formal",
            contract.get("config", {}).get("path") == str(config_path),
            int(contract.get("config", {}).get("bytes", -1))
            == config_path.stat().st_size,
            contract.get("model") == expected_model,
            contract.get("optimization") == expected_optimization,
            int(contract.get("diagnostic", {}).get("task", -1)) == task,
            qualification.get("program_language_authority") == task,
            qualification.get("wrong_bank_task")
            == int(config["task_local"]["wrong_task_by_task"][str(task)]),
            arm_names
            == [
                "correct_fit0",
                "correct_fit1",
                "correct_held",
                "wrong_fit0",
                "wrong_fit1",
            ],
            [row.get("receives_gradient") for row in qualification.get("arms", ())]
            == [True, True, False, True, False],
            qualification.get("panel_b_receives_gradient") is False,
            contract.get("information_wall", {}).get("action_meta_installed") is False,
            int(inventory.get("action_meta_module_count", -1)) == 0,
            int(inventory.get("action_meta_parameter_count", -1)) == 0,
            _free_parameter_roots(trainable_names) == expected_free,
            any("set_encoder" in name for name in trainable_names)
            is expected_set_encoder_trainable,
            int(inventory.get("writer_trainable_parameter_count", 0)) > 0,
            (
                summary_mass is None
                if stage == BANK_SET_S1_STAGE
                else float(summary_mass.get("minimum", 0.0)) > 0.0
            ),
            float(teacher.get("panel_a_recovery_after_update", math.inf)) <= 0.25,
            float(teacher.get("panel_a_recovery_after_update", math.inf))
            < float(teacher.get("panel_a_recovery_before_update", -math.inf)),
            result.get("information_wall", {}).get("action_meta_installed") is False,
            result.get("information_wall", {}).get("single_complete_rank16") is True,
            result.get("information_wall", {}).get("shuffled_or_reversed_use") is False,
            result.get("information_wall", {}).get("panel_b_backward_calls") == 0,
            len(metrics) == 110,
            [int(row.get("optimizer_step", -1)) for row in metrics]
            == list(range(1, 111)),
            _formal_checkpoint_valid(root, macro=70, stage=stage),
            _formal_checkpoint_valid(root, macro=110, stage=stage),
        )
    )
    if not valid:
        raise ValueError(f"invalid task-local formal evidence: {root}")
    commit = str(contract.get("git", {}).get("authority_commit", ""))
    if len(commit) != 40:
        raise ValueError("task-local formal commit authority changed")
    evaluation = result["evaluation"]
    arms = evaluation["arms"]
    row = {
        "output_root": str(root),
        "role": result["role"],
        "gate": evaluation["gate"],
        "checks": evaluation["checks"],
        "functional_recovery": {
            name: float(value["functional_recovery"]) for name, value in arms.items()
        },
        "negative_family_cosines": {
            name: value["effective_rank4"]["negative_family_cosines"]
            for name, value in arms.items()
        },
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "step_seconds": {
            "median": statistics.median(float(value["step_seconds"]) for value in metrics),
            "maximum": max(float(value["step_seconds"]) for value in metrics),
        },
        "peak_cuda_allocated_bytes": max(
            int(value["peak_cuda_allocated_bytes"]) for value in metrics
        ),
    }
    if stage != BANK_CONDITIONED_PRIMAL_STAGE:
        row["maximum_near_bound_fraction"] = max(
            float(value["correction"]["all"]["near_bound_fraction"])
            for value in arms.values()
        )
    else:
        row["primal_response"] = {
            name: value["primal_response"] for name, value in arms.items()
        }
    return task, commit, row


def aggregate_tasklocal(
    *,
    config_path: Path,
    task_output_dirs: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, Any]:
    """Seal the two independent S0/S1 task-local functional conclusions."""

    config_path = config_path.resolve()
    config = load_routing_control_config(config_path)
    stage = str(config.get("stage", ""))
    if (
        config.get("schema_version") != BANK_SET_TASKLOCAL_SCHEMA
        or stage
        not in {
            BANK_SET_S0_STAGE,
            BANK_SET_S1_STAGE,
            BANK_CONDITIONED_PRIMAL_STAGE,
        }
        or len(task_output_dirs) != len(TASKS)
    ):
        raise ValueError("bank-set task-local aggregate authority changed")
    task_rows: dict[int, Any] = {}
    commits = set()
    for root in map(Path.resolve, task_output_dirs):
        task, commit, row = _load_formal_task(
            root, stage=stage, config_path=config_path, config=config
        )
        if task in task_rows:
            raise ValueError("duplicate task-local formal task")
        task_rows[task] = row
        commits.add(commit)
    if set(task_rows) != set(TASKS) or len(commits) != 1:
        raise ValueError("task-local formal tasks do not share one authority")
    science_pass = all(
        row["gate"] == "pass" and all(row["checks"].values())
        for row in task_rows.values()
    )
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "complete",
        "stage": stage,
        "authority_commit": next(iter(commits)),
        "gate": "pass" if science_pass else "non_pass",
        "tasks": {str(task): task_rows[task] for task in TASKS},
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("task-local aggregate output root is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "aggregate.json", aggregate)
    return aggregate
