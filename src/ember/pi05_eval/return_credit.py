"""Registered Gaussian decisions retained from canonical PI05 rollouts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.exploration import exploration_covariance
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = Path("configs/return_credit_direction_v1/experiment_spec.json")


def authority() -> dict[str, Any]:
    spec = read_json(REPO_ROOT / SPEC_PATH)
    if (spec.get("schema_version") != "ember_return_credit_direction_v1"
            or spec.get("study_id") != "return_credit_direction_causality_20260925"
            or spec["parent"]["macro"] != 1155
            or len(spec["collection"]["conditions"]) != 8
            or spec["collection"]["episodes"] != 128
            or spec["evaluation"]["total_episodes"] != 256):
        raise Pi05EvaluationError("return-credit scientific registration changed")
    return spec


def _seed(parts: Sequence[int]) -> int:
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)


def exploration_seed(task: int, state: int, replica: int, replan: int) -> int:
    return _seed((20260925, task, state, replica, replan, 0x524C))


def reservoir_seed(task: int, state: int, replica: int) -> int:
    return _seed((20260925, task, state, replica, 0xDEC1))


def validate_collection(contract: Mapping[str, Any], task: Mapping[str, Any]) -> Mapping[str, Any]:
    scope = contract.get("return_credit_collection")
    if scope is None:
        raise Pi05EvaluationError("return-credit collection registration missing")
    spec = authority()
    task_id, state, replica = (int(scope[key]) for key in ("global_task", "init_state_id", "replica"))
    registered = next((row for row in spec["collection"]["conditions"] if row["task"] == task_id), None)
    study = Path(spec["resources"]["study_root"]).resolve()
    output = Path(scope.get("output", "")).resolve()
    engineering = scope.get("engineering_smoke") is True
    expected = (study / "development_smoke" / "task_002_state_00" if engineering else
                study / "collection" / "groups" / f"task_{task_id:03d}_state_{state:02d}")
    if (registered is None or state not in registered["init_state_ids"]
            or replica not in registered["replicas"]
            or (engineering and (task_id, state, replica) != (2, 0, 0))
            or scope.get("teacher_demo") != registered["teacher_demo"]
            or (task["suite"], int(task["task_id"])) !=
               (("libero_spatial", task_id) if task_id < 10 else
                ("libero_object", task_id - 10) if task_id < 20 else
                ("libero_goal", task_id - 20) if task_id < 30 else
                ("libero_10", task_id - 30))
            or output != expected or contract["rng"]["inference_seed"] != 7
            or contract["policy"]["num_inference_steps"] != 10
            or contract["policy"]["replan_steps"] != 5):
        raise Pi05EvaluationError("return-credit task, teacher, state, replica or policy scope changed")
    return scope


def explore_and_retain(chunks: torch.Tensor, slots: Sequence[dict[str, Any]], *,
                       raw_inputs: Sequence[Mapping[str, Any]],
                       processed: Sequence[Mapping[str, Any]], noise: torch.Tensor,
                       task: Mapping[str, Any], contract: Mapping[str, Any]) -> torch.Tensor:
    """Save original inputs and a reward-independent four-decision reservoir."""
    scope = validate_collection(contract, task)
    if (chunks.shape != (len(slots), 50, 7) or noise.shape != (len(slots), 50, 32)
            or len(raw_inputs) != len(slots) or len(processed) != len(slots)):
        raise Pi05EvaluationError("return-credit native flow or query shape changed")
    result = chunks.clone()
    lower = torch.linalg.cholesky(exploration_covariance(device="cpu", dtype=torch.float32))
    for index, slot in enumerate(slots):
        state, replica, task_id = (int(scope[key]) for key in ("init_state_id", "replica", "global_task"))
        if int(slot["init_state_id"]) != state:
            raise Pi05EvaluationError("return-credit replay state differs from registration")
        replan = int(slot["replan_index"])
        seed = exploration_seed(task_id, state, replica, replan)
        eta = torch.randn(35, generator=torch.Generator(device="cpu").manual_seed(seed),
                          dtype=torch.float32) @ lower.T
        mu = chunks[index].detach().float().cpu()
        latent = mu[:5, :7].reshape(35) + eta
        result[index, :5, :7] = latent.reshape(5, 7).to(result)
        record = {
            "replan_index": replan, "control_step": int(slot["steps"]),
            "policy_noise_seed": policy_noise_seed(
                7, str(task["suite"]), int(task["task_id"]), state, replan),
            "exploration_seed": seed,
            "raw_input": {key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor)
                          else value for key, value in raw_inputs[index].items()},
            "processed": {key: value.detach().cpu().contiguous() for key, value in processed[index].items()
                          if isinstance(value, torch.Tensor)},
            "flow_noise": noise[index].detach().float().cpu().contiguous(),
            "mu_old_full": mu.contiguous(), "latent_u": latent.contiguous(),
            "applied_normalized": result[index, :5, :7].detach().float().cpu().contiguous(),
        }
        bank = slot.get("return_credit_reservoir")
        if bank is None:
            bank = {"rng": np.random.default_rng(reservoir_seed(task_id, state, replica)),
                    "seen": 0, "items": []}
            slot["return_credit_reservoir"] = bank
        bank["seen"] += 1
        if len(bank["items"]) < 4:
            bank["items"].append(record)
        else:
            choice = int(bank["rng"].integers(0, bank["seen"]))
            if choice < 4:
                bank["items"][choice] = record
    return result


def save_decisions(contract: Mapping[str, Any], task: Mapping[str, Any],
                   slot: Mapping[str, Any]) -> dict[str, Any]:
    scope = validate_collection(contract, task)
    bank = slot.get("return_credit_reservoir", {"seen": 0, "items": []})
    count = int(bank["seen"])
    items = sorted(bank["items"], key=lambda row: row["replan_index"])
    if len(items) != min(4, count) or count != len(slot["policy_noise_seeds"]):
        raise Pi05EvaluationError("return-credit Q/M reservoir or replan count changed")
    for item in items:
        start = int(item["control_step"])
        executed = max(0, min(5, int(slot["steps"]) - start))
        item["executed_mask"] = [index < executed for index in range(5)]
    output = Path(scope["output"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"replica_{scope['replica']}_decisions.pt"
    partial = path.with_suffix(".partial")
    if path.exists() or partial.exists():
        raise Pi05EvaluationError("return-credit decision evidence already exists")
    payload = {"schema_version": "ember_return_credit_decisions_v1",
               "global_task": scope["global_task"], "teacher_demo": scope["teacher_demo"],
               "init_state_id": scope["init_state_id"], "replica": scope["replica"],
               "Q": count, "M": len(items), "decisions": items}
    torch.save(payload, partial)
    os.replace(partial, path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "Q": count, "M": len(items), "replica": scope["replica"],
            "teacher_demo": scope["teacher_demo"]}
