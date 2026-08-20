#!/usr/bin/env python3
"""Diagnose a role-disjoint, set-valued adapter manifold in effective-BA space."""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.functional_adaptation.decoder_training import (
    authority_path,
    decoder_task_split,
    expert_records,
    inspect_train24_expert_bank,
    load_functional_adapter_config,
)
from ember.functional_adaptation.fingerprint_codes import (
    load_functional_fingerprint_code_targets,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ember_pi05_role_disjoint_equivalence_diagnostic_v1"


def _step_path(value: str) -> tuple[int, Path]:
    try:
        step_text, path_text = value.split("=", 1)
        step = int(step_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("evaluation must be STEP=PATH") from error
    path = Path(path_text).resolve()
    if step <= 0 or not (path / "results.json").is_file():
        raise argparse.ArgumentTypeError("evaluation result is missing")
    return step, path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--fingerprint-root", type=Path, required=True)
    parser.add_argument(
        "--evaluation",
        type=_step_path,
        action="append",
        required=True,
        metavar="STEP=PATH",
    )
    parser.add_argument("--success-threshold", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> None:
    for name in (
        "config",
        "source_run",
        "checkpoint",
        "expert_bank_root",
        "fingerprint_root",
    ):
        value = getattr(args, name).resolve()
        if not value.exists():
            raise ValueError(f"missing role-disjoint diagnostic input: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise ValueError("role-disjoint diagnostic output already exists")
    if args.success_threshold <= 0:
        raise ValueError("success threshold must be positive")


def _task_metadata(config: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = read_json(authority_path(config, "target_data_manifest", REPO_ROOT))
    return {
        (str(row["suite"]), int(row["task_id"])): row
        for row in manifest["tasks"]
        if row["split_role"] == "train"
    }


def _successes(
    evaluations: Mapping[int, Path],
    metadata: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for step, root in evaluations.items():
        payload = read_json(root / "results.json")
        if int(payload["overall"]["episodes"]) != 1200:
            raise ValueError("task-expert evaluation is not the fixed 24x50 panel")
        by_task: dict[int, int] = {}
        for row in payload["per_task"]:
            task = metadata[(str(row["suite"]), int(row["task_id"]))]
            by_task[int(task["global_task_id"])] = int(row["successes"])
        if len(by_task) != 24:
            raise ValueError("task-expert evaluation does not cover train24")
        result[step] = by_task
    return result


def _selected_steps(
    successes: Mapping[int, Mapping[int, int]],
    task_ids: Sequence[int],
    *,
    threshold: int,
) -> dict[int, tuple[int, ...]]:
    steps = tuple(sorted(successes))
    selected: dict[int, tuple[int, ...]] = {}
    for task_id in task_ids:
        passing = tuple(
            step for step in steps if int(successes[step][task_id]) >= threshold
        )
        if not passing:
            best = max(steps, key=lambda step: (successes[step][task_id], step))
            passing = (best,)
        selected[task_id] = passing
    return selected


def _pair_names(contract: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            target.name + LORA_A_SUFFIX,
            target.name + LORA_B_SUFFIX,
        )
        for target in contract.targets
    )


def _adapter_gram(
    states: Sequence[Mapping[str, torch.Tensor]],
    pairs: Sequence[tuple[str, str]],
    *,
    device: torch.device,
) -> torch.Tensor:
    size = len(states)
    gram = torch.zeros(size, size, dtype=torch.float64, device=device)
    for name_a, name_b in pairs:
        a = torch.stack(
            [state[name_a].to(device=device, dtype=torch.float32) for state in states]
        )
        b = torch.stack(
            [state[name_b].to(device=device, dtype=torch.float32) for state in states]
        )
        left = torch.einsum("nor,mos->nmrs", b, b)
        right = torch.einsum("nri,msi->nmrs", a, a)
        gram += (left * right).sum(dim=(-1, -2)).double()
    return gram.cpu()


def _weights(
    task_ids: Sequence[int],
    chosen: Mapping[int, Sequence[int]],
    adapter_index: Mapping[tuple[int, int], int],
    adapter_count: int,
) -> torch.Tensor:
    result = torch.zeros(len(task_ids), adapter_count, dtype=torch.float64)
    for row, task_id in enumerate(task_ids):
        steps = tuple(chosen[task_id])
        for step in steps:
            result[row, adapter_index[(task_id, step)]] = 1.0 / len(steps)
    return result


def _geometry(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gram: torch.Tensor,
) -> dict[str, float]:
    pred_energy = float(prediction @ gram @ prediction)
    target_energy = float(target @ gram @ target)
    dot = float(prediction @ gram @ target)
    error = max(pred_energy + target_energy - 2.0 * dot, 0.0)
    return {
        "relative_l2": math.sqrt(error / max(target_energy, 1e-24)),
        "cosine": dot / max(math.sqrt(pred_energy * target_energy), 1e-24),
        "norm_ratio": math.sqrt(pred_energy / max(target_energy, 1e-24)),
    }


def _summarize(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        name: float(sum(float(row[name]) for row in rows) / len(rows))
        for name in ("relative_l2", "cosine", "norm_ratio")
    }


def _affine_panel(
    *,
    fit_codes: torch.Tensor,
    held_codes: torch.Tensor,
    fit_targets: torch.Tensor,
    held_targets: torch.Tensor,
    held_members: Sequence[Sequence[torch.Tensor]],
    gram: torch.Tensor,
    fit_task_ids: Sequence[int],
    held_task_ids: Sequence[int],
) -> dict[str, Any]:
    fit_design = torch.cat(
        (fit_codes.double(), torch.ones(len(fit_codes), 1, dtype=torch.float64)),
        dim=1,
    )
    held_design = torch.cat(
        (held_codes.double(), torch.ones(len(held_codes), 1, dtype=torch.float64)),
        dim=1,
    )
    inverse = torch.linalg.pinv(fit_design)
    fit_coefficients = fit_design @ inverse
    held_coefficients = held_design @ inverse
    fit_predictions = fit_coefficients @ fit_targets
    held_predictions = held_coefficients @ fit_targets
    fit_rows = [
        _geometry(fit_predictions[index], fit_targets[index], gram)
        for index in range(len(fit_task_ids))
    ]
    held_rows = []
    for index, task_id in enumerate(held_task_ids):
        prototype = _geometry(held_predictions[index], held_targets[index], gram)
        members = [
            _geometry(held_predictions[index], member, gram)
            for member in held_members[index]
        ]
        held_rows.append(
            {
                "global_task_id": int(task_id),
                "prototype": prototype,
                "best_member_relative_l2": min(row["relative_l2"] for row in members),
                "best_member_cosine": max(row["cosine"] for row in members),
                "member_count": len(members),
                "affine_coefficient_l1": float(held_coefficients[index].abs().sum()),
                "affine_coefficient_l2": float(held_coefficients[index].norm()),
            }
        )
    held_prototypes = [row["prototype"] for row in held_rows]
    singular = torch.linalg.svdvals(fit_design)
    return {
        "design_rank": int(torch.linalg.matrix_rank(fit_design)),
        "design_condition_number": float(singular.max() / singular.min()),
        "fit_prototype": _summarize(fit_rows),
        "held_prototype": _summarize(held_prototypes),
        "held_best_member": {
            "relative_l2": float(
                sum(row["best_member_relative_l2"] for row in held_rows)
                / len(held_rows)
            ),
            "cosine": float(
                sum(row["best_member_cosine"] for row in held_rows) / len(held_rows)
            ),
        },
        "held_tasks": held_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _resolve_inputs(args)
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal role-disjoint diagnostic requires clean pushed code")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("requested role-disjoint diagnostic CUDA device is unavailable")
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    mechanism = config["train24_mechanism"]
    evaluations = dict(args.evaluation)
    if len(evaluations) != len(args.evaluation):
        raise ValueError("role-disjoint diagnostic evaluation steps are duplicated")
    required_step = int(mechanism["expert_step"])
    if required_step not in evaluations:
        raise ValueError("role-disjoint diagnostic requires the canonical expert step")
    banks = {
        step: inspect_train24_expert_bank(
            config,
            REPO_ROOT,
            source_run=args.source_run,
            checkpoint=args.checkpoint,
            bank_root=args.expert_bank_root,
            expert_step=step,
        )
        for step in sorted(evaluations)
    }
    record_maps = {
        step: {row.global_task_id: row for row in expert_records(bank)}
        for step, bank in banks.items()
    }
    canonical_records = expert_records(banks[required_step])
    split = decoder_task_split(
        canonical_records,
        fold_count=int(mechanism["fold_count"]),
        held_out_fold=int(mechanism["held_out_fold"]),
    )
    fit_ids = tuple(row.global_task_id for row in split.fit)
    held_ids = tuple(row.global_task_id for row in split.held)
    codes = load_functional_fingerprint_code_targets(
        args.fingerprint_root,
        expected_train_task_ids=fit_ids,
        expected_held_task_ids=held_ids,
        code_width=int(config["decoder"]["train24_smoke_code_width"]),
        device="cpu",
        expected_fit_surface="train24_fit_only_pca_whitening",
    )
    metadata = _task_metadata(config)
    success = _successes(evaluations, metadata)
    all_ids = (*fit_ids, *held_ids)
    equivalence_steps = _selected_steps(
        success, all_ids, threshold=args.success_threshold
    )
    selected_keys = {
        (task_id, step)
        for task_id in all_ids
        for step in (*equivalence_steps[task_id], required_step)
    }
    ordered_keys = tuple(sorted(selected_keys))
    adapter_index = {key: index for index, key in enumerate(ordered_keys)}
    states = tuple(
        load_file(
            str(record_maps[step][task_id].checkpoint / "adapter.safetensors"),
            device="cpu",
        )
        for task_id, step in ordered_keys
    )
    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", REPO_ROOT)
    )
    gram = _adapter_gram(states, _pair_names(contract), device=device)
    equivalence_fit = _weights(
        fit_ids, equivalence_steps, adapter_index, len(ordered_keys)
    )
    equivalence_held = _weights(
        held_ids, equivalence_steps, adapter_index, len(ordered_keys)
    )
    step_only = {task_id: (required_step,) for task_id in all_ids}
    step_fit = _weights(fit_ids, step_only, adapter_index, len(ordered_keys))
    step_held = _weights(held_ids, step_only, adapter_index, len(ordered_keys))
    equivalence_members = [
        [
            _weights(
                (task_id,),
                {task_id: (step,)},
                adapter_index,
                len(ordered_keys),
            )[0]
            for step in equivalence_steps[task_id]
        ]
        for task_id in held_ids
    ]
    step_members = [[step_held[index]] for index in range(len(held_ids))]
    result = {
        "schema_version": SCHEMA,
        "formal_authority": True,
        "repository": repository,
        "host": socket.gethostname(),
        "device": str(device),
        "surface": "target_train24_source_meta_role_disjoint",
        "inputs": {
            "config": str(args.config),
            "source_run": str(args.source_run),
            "checkpoint": str(args.checkpoint),
            "expert_bank_root": str(args.expert_bank_root),
            "fingerprint_root": str(args.fingerprint_root),
            "evaluations": {
                str(step): str(root) for step, root in sorted(evaluations.items())
            },
        },
        "fit_global_task_ids": list(fit_ids),
        "held_global_task_ids": list(held_ids),
        "expert_steps": sorted(evaluations),
        "success_threshold": args.success_threshold,
        "equivalence_selection": {
            str(task_id): {
                "steps": list(equivalence_steps[task_id]),
                "successes": {
                    str(step): int(success[step][task_id])
                    for step in equivalence_steps[task_id]
                },
                "fallback_to_best_when_none_pass": all(
                    success[step][task_id] < args.success_threshold
                    for step in success
                ),
            }
            for task_id in all_ids
        },
        "selected_adapter_count": len(ordered_keys),
        "step2000_single_label": _affine_panel(
            fit_codes=codes.train_codes,
            held_codes=codes.held_codes,
            fit_targets=step_fit,
            held_targets=step_held,
            held_members=step_members,
            gram=gram,
            fit_task_ids=fit_ids,
            held_task_ids=held_ids,
        ),
        "thresholded_equivalence_or_best_ceiling_prototype": _affine_panel(
            fit_codes=codes.train_codes,
            held_codes=codes.held_codes,
            fit_targets=equivalence_fit,
            held_targets=equivalence_held,
            held_members=equivalence_members,
            gram=gram,
            fit_task_ids=fit_ids,
            held_task_ids=held_ids,
        ),
        "interpretation_boundary": {
            "diagnostic_only": True,
            "materialized_rank16_adapter": False,
            "closed_loop_claim": False,
            "checkpoint_family_independent_seeds": False,
            "checkpoint_family_note": "same task-local optimization trajectory at five retained steps",
            "held_outcomes_used_only_to_define_diagnostic_equivalence_or_best_ceiling_set": True,
            "held_fingerprint_requires_privileged_step2000_expert": True,
            "fingerprint_code_representative_step": required_step,
            "prototype_average_is_not_assumed_rank16_or_closed_loop_successful": True,
            "validation_or_test_tasks_read": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "content_hash_policy": "disabled_by_owner",
    }
    args.output_dir.mkdir(parents=True)
    write_json_atomic(args.output_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    run(_args())
