#!/usr/bin/env python3
"""Registered frozen first-slot actions and non-held 76/77 FM reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ember.lora import validate_lora_state
from ember.pi05_eval.crossed_video_field import read_json
from ember.pi05_eval.crossed_video_predict import (
    CrossedVideoPredictor, _query_input, scaled_osc_actions,
)
from ember.pi05_evaluation import make_policy_noise
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.writer.evaluation import inspect_horizon_writer_bank
from ember.writer.function_credit import paired_functional_credit
from ember.writer.learning_data import WriterTrainingData
from ember.writer.materialization import source_matches
from ember.writer.runtime import autocast, build_runtime
from ember.writer.support_slot_credit import ARMS, authority, root


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO.parent / "EMBER"
QUERY_ROOT = Path("/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924")
MODELS = ("P1155", *ARMS)


def _commit() -> str:
    state = git_state(REPO)
    if (state["branch"] or state["dirty_paths"]
            or not git_state_is_clean_pushed_or_frozen_authority(state)):
        raise ValueError("formal support-slot prediction needs its clean detached implementation")
    return state["commit"]


def _bank(model: str, phase: str, commit: str) -> dict:
    path = root() / "materialization" / phase / model / "manifest.json"
    manifest = read_json(path)
    tasks = tuple((row["suite"], int(row["task_id"])) for row in manifest["tasks"])
    checked = inspect_horizon_writer_bank(
        manifest_path=path, source=manifest["source"], task_keys=tasks,
        evaluation_role=manifest["evaluation_role"], require_formal=True,
        support_slot_model=model, support_slot_phase=phase,
    )
    if checked["support_slot_credit"]["bank_generation_commit"] != commit:
        raise ValueError("support-slot probe bank and prediction implementation differ")
    return checked


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def first_slot() -> None:
    commit = _commit()
    banks = {model: _bank(model, "first_slot", commit) for model in ARMS}
    original = read_json(QUERY_ROOT / "manifest.json")
    if (original["policy"]["num_inference_steps"] != 10
            or original["policy"]["chunk_size"] != 50):
        raise ValueError("sealed original query policy semantics changed")
    if any(not source_matches(bank["source"], original["model"]) for bank in banks.values()):
        raise ValueError("first-slot bank and original query source models differ")
    queries = [json.loads(line) for line in (QUERY_ROOT / "query_index.jsonl").read_text().splitlines()]
    selected = [row for row in queries if row["global_task_id"] in (14, 21)
                and row["control_step"] == 0 and row["init_state_id"] in (0, 10, 20, 30, 40)]
    if len(queries) != 300 or len(selected) != 10 or len({row["query_id"] for row in selected}) != 10:
        raise ValueError("registered original t0 query subset changed")
    plans = {}
    for model, bank in banks.items():
        conditions = {row["condition_id"]: row for row in bank["conditions"]}
        episodes = {(int(task["global_task_id"]), int(row["init_state_id"])): row
                    for task in bank["tasks"] for row in task["episodes"]}
        if len(episodes) != 10:
            raise ValueError("first-slot bank lost the ten registered states")
        plans[model] = (conditions, episodes)
    predictor = CrossedVideoPredictor(original, batch_size=3)
    rows = []
    output = root() / "prediction" / "first_slot"
    try:
        with torch.inference_mode():
            for query in selected:
                task, state = int(query["global_task_id"]), int(query["init_state_id"])
                if int(query["global_replan_index"]) != 0:
                    raise ValueError("first-slot action probe must use original t0 query")
                group = []
                for model in ARMS:
                    conditions, episodes = plans[model]
                    episode = episodes[(task, state)]
                    if episode["condition_id"] != query["C_diagonal_condition_id"]:
                        raise ValueError("first-slot teacher differs from original correct-video schedule")
                    group.append({"kind": model, **conditions[episode["condition_id"]]})
                processed = predictor.processor(_query_input(query))
                batch = {key: value for key, value in processed.items() if isinstance(value, torch.Tensor)}
                predictor.policy.reset()
                noise, seeds = make_policy_noise(
                    [{"init_state_id": state, "replan_index": 0}], root_seed=7,
                    suite=query["suite"], task_id=int(query["task_id"]),
                    chunk_size=50, max_action_dim=32, device=predictor.device)
                if seeds != (int(query["policy_noise_seed"]),):
                    raise ValueError("first-slot policy noise differs from original query")
                normalized, environment, _ = predictor._forward_group(group, batch, noise, pilot=False)
                osc = scaled_osc_actions(environment, query["osc_channel_audit"])
                if (normalized.shape != (3, 50, 7) or environment.shape != (3, 50, 7)
                        or osc.shape != (3, 5, 6) or not np.isfinite(osc).all()):
                    raise ValueError("first-slot action output shape or finite check failed")
                path = output / f'{query["query_id"].replace(":", "_")}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    np.savez_compressed(handle, normalized=normalized, environment=environment,
                                        osc_first5=osc, noise=noise[0].float().cpu().numpy(),
                                        models=np.asarray(ARMS))
                for index, model in enumerate(ARMS):
                    rows.append({"query_id": query["query_id"], "global_task_id": task,
                                 "init_state_id": state, "model": model,
                                 "condition_id": group[index]["condition_id"],
                                 "teacher_demo_indices": group[index]["teacher_demo_indices"],
                                 "policy_noise_seed": seeds[0], "prediction": str(path),
                                 "prediction_index": index, "implementation_commit": commit,
                                 "gradients": False, "environment_steps": 0})
    finally:
        predictor.close()
    if len(rows) != 30:
        raise ValueError("first-slot prediction matrix incomplete")
    _write_rows(output / "rows.jsonl", rows)


def donor_fm() -> None:
    import json as _json

    commit = _commit()
    config = _json.loads((REPO / "configs/relational_support_causality_v1/train_C_S00.json").read_text())
    donor = _json.loads((REPO / "configs/relational_support_causality_v1/train_C_S10.json").read_text())
    banks = {model: _bank(model, "donor_fm", commit) for model in MODELS}
    device = torch.device("cuda:0")
    runtime = build_runtime(ASSETS, config, device)
    runtime.policy.requires_grad_(False).eval()
    s00 = WriterTrainingData(ASSETS, config["data"], camera_view=config["observer"]["camera_view"])
    s10 = WriterTrainingData(ASSETS, donor["data"], camera_view=donor["observer"]["camera_view"])
    rows = []
    try:
        with torch.inference_mode(), autocast(device):
            for task, data in ((76, s10), (77, s00)):
                raw, trace = data.diagnostic_batch(task, seed=20260925, count=16, teacher_demo=46)
                if (len(trace["action_demos"]) != 16
                        or not set(trace["action_demos"]) <= {47, 48, 49}):
                    raise ValueError("donor FM query episodes differ from 47..49")
                batch = runtime.processor.training_batch(raw)
                for model in MODELS:
                    bank = banks[model]
                    condition = next(row for row in bank["conditions"] if row["global_task_id"] == task)
                    state = load_file(condition["adapter"]["path"], device=str(device))
                    validate_lora_state(state, runtime.lora)
                    for index in range(16):
                        sliced = {name: value[index:index + 1] if isinstance(value, torch.Tensor)
                                  and value.ndim and len(value) == 16 else value
                                  for name, value in batch.items()}
                        credit = paired_functional_credit(runtime.policy, state, runtime.lora, sliced,
                            seed=trace["policy_rng_seed"], device=device, random_batch=16,
                            offset=index, microbatch=1, condition_weight=1.0, backward=False)
                        if (credit["lora_cotangent"] or credit["compiled_forward_calls"] != 1
                                or not np.isfinite(credit["flow_loss"])):
                            raise ValueError("donor FM inference or loss invalid")
                        rows.append({"task": task, "model": model, "query_index": index,
                            "query_demo": int(trace["action_demos"][index]),
                            "query_frame": int(trace["action_frames"][index]),
                            "teacher_demo": 46, "policy_rng_seed": trace["policy_rng_seed"],
                            "loss": credit["flow_loss"], "condition_id": condition["condition_id"],
                            "implementation_commit": commit, "gradients": False,
                            "flow_mode": "random_tau_full_H", "ten_flow_predictions": 0})
    finally:
        s00.close()
        s10.close()
    if len(rows) != 128:
        raise ValueError("donor FM matrix incomplete")
    _write_rows(root() / "prediction" / "donor_fm" / "rows.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("first_slot", "donor_fm"))
    args = parser.parse_args()
    if authority()["resources"]["study_root"] != str(root()):
        raise ValueError("support-slot study root changed")
    _commit()
    (first_slot if args.phase == "first_slot" else donor_fm)()


if __name__ == "__main__":
    main()
