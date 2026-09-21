#!/usr/bin/env python3
"""Run the fixed Writer stability diagnostics without changing formal models."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, sha256_file, write_json_atomic
from ember.writer.runtime import VideoConditionCache
from ember.writer.stability_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    HIGH_LR,
    LOW_LR,
    PROBE_NOISE_SEEDS,
    PROBE_TEACHER_QUERY,
    current_training_data,
    event_window,
    load_writer,
    optimizer_step,
    probe_dataset,
    probe_items,
    probe_one,
    save_diagnostic_checkpoint,
    source_policy,
    validate_optimizer_state,
    writer_condition_gradient,
    writer_frozen_policy,
    writer_probe_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/user/ymdai/projects/EMBER")
STUDY_ROOT = Path("/data0/user/ymdai/ember_runs/writer_stability_diagnostics_20260921")
OLD_ROOT = Path("/data0/user/ymdai/ember_runs/video_teaching_20260919/training")
NEW_ROOT = Path("/data0/user/ymdai/ember_runs/coverage_retraining_20260920")
SOURCE_CHECKPOINT = Path(
    "/data1/user/ymdai/projects/EMBER/runs/outputs/"
    "pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000"
)
WRITER_ASSETS = {
    "O1200": OLD_ROOT / "checkpoints/macro_00001200",
    "O1500": OLD_ROOT / "checkpoints/macro_00001500",
    "N1000": NEW_ROOT / "training/writer/checkpoints/macro_00001000",
    "N1800": NEW_ROOT / "training/writer/checkpoints/macro_00001800",
}
MTBC_CHECKPOINT = NEW_ROOT / "training/mtbc/checkpoints/step_00000300"
MTBC_EVAL_CONTRACT = NEW_ROOT / "evaluation/mtbc_00000300/run_contract.json"


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type != "cuda":
        raise ValueError("diagnostics require CUDA")
    torch.cuda.set_device(result)
    from ember.writer.topology import bind_current_process_to_cuda_numa

    if not bind_current_process_to_cuda_numa(torch.cuda.current_device()):
        raise ValueError("diagnostics require GPU-local NUMA placement")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    return result


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _json_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=isinstance(value, dict))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refuse empty diagnostic table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"diagnostic table has inconsistent fields: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _json_cell(value) for key, value in row.items()} for row in rows)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _current_run() -> dict[str, Any]:
    return read_json(NEW_ROOT / "training/writer/run_contract.json")


def register(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    for name, checkpoint in WRITER_ASSETS.items():
        assets[name] = {
            "kind": "writer",
            "checkpoint": str(checkpoint.resolve()),
            "weights": _record(checkpoint / "ecp.safetensors"),
            "trainer_state": _record(checkpoint / "trainer_state.pt"),
            "checkpoint_manifest": _record(checkpoint / "checkpoint_manifest.json"),
            "run_contract": _record(checkpoint.parent.parent / "run_contract.json"),
        }
    mtbc_eval = read_json(MTBC_EVAL_CONTRACT)
    assets["M300"] = {
        "kind": "mtbc",
        "checkpoint": str(MTBC_CHECKPOINT.resolve()),
        "lora": _record(MTBC_CHECKPOINT / "lora.safetensors"),
        "manifest": _record(MTBC_CHECKPOINT / "checkpoint_manifest.json"),
        "config": mtbc_eval["adapter"]["config"],
    }
    assets["S1000"] = {
        "kind": "source",
        "checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "manifest": _record(SOURCE_CHECKPOINT / "checkpoint_manifest.json"),
    }
    write_json_atomic(output / "registration.json", {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "registered",
        "design": "docs/writer_stability_diagnostics_design.md",
        "fresh_training": False,
        "test_use": False,
        "checkpoint_selection_use": False,
        "asset_names": list(assets),
        "stage_order": ["E0", "E1", "E3-corrected"],
        "output_root": str(output.resolve()),
    })
    write_json_atomic(output / "assets.json", {"schema_version": DIAGNOSTIC_SCHEMA, "assets": assets})


def e0(output: Path, device: torch.device) -> None:
    current = _current_run()
    rows = []
    for name in ("O1200", "N1800"):
        loaded = load_writer(name=name, checkpoint=WRITER_ASSETS[name], asset_root=ASSET_ROOT, device=device)
        data = current_training_data(ASSET_ROOT, current, planned_updates=1801)
        draw = event_window(data, first_step=1801, last_step=1801)[0][0]
        cache = VideoConditionCache(loaded.runtime, data, 2**30)
        optimizer = validate_optimizer_state(loaded.runtime.state, loaded.optimizer, loaded.parameter_names)
        for component in ("main", "teaching"):
            gradients, metrics = writer_condition_gradient(loaded, data, cache, draw, component=component)
            rows.append({
                "asset": name,
                "component": component,
                "task": draw["task"],
                "teacher_demo": draw["video_demos"][0],
                "frame_chunk": loaded.run["config"]["observer"]["frame_chunk"],
                "checkpoint_world_size": loaded.run["topology"]["world_size"],
                **optimizer,
                **metrics,
                "all_parameter_gradients_finite": all(torch.isfinite(value).all() for value in gradients),
            })
        data.close()
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
    if {row["asset"] for row in rows} != {"O1200", "N1800"} or len(rows) != 4:
        raise ValueError("E0 did not complete both parent/interface paths")
    write_json_atomic(output / "e0_integrity.json", {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "rows": rows,
        "stop_reason": None,
    })


def _load_frozen(asset: str, device: torch.device):
    current = _current_run()
    if asset in WRITER_ASSETS:
        loaded = load_writer(name=asset, checkpoint=WRITER_ASSETS[asset], asset_root=ASSET_ROOT, device=device)
        loaded.runtime.state.eval()
        return loaded, None
    if asset == "S1000":
        return None, source_policy(name=asset, asset_root=ASSET_ROOT, current_run=current, device=device)
    if asset == "M300":
        adapter = read_json(MTBC_EVAL_CONTRACT)["adapter"]
        config_path = ASSET_ROOT / "configs/libero_24_8_8_coverage_v1/mtbc.json"
        if sha256_file(config_path) != adapter["config"]["sha256"]:
            raise ValueError("canonical MT-BC config differs from the frozen evaluation authority")
        return None, source_policy(
            name=asset, asset_root=ASSET_ROOT, current_run=current, device=device,
            mtbc_config=config_path, mtbc_checkpoint=MTBC_CHECKPOINT,
        )
    raise ValueError(f"unknown diagnostic asset: {asset}")


def _probe_rows(asset: str, device: torch.device, *, compact: bool = False) -> list[dict[str, Any]]:
    current = _current_run()
    data = current_training_data(ASSET_ROOT, current, planned_updates=1)
    queries = probe_dataset(ASSET_ROOT, data)
    loaded, frozen = _load_frozen(asset, device)
    rows = []
    try:
        if loaded is not None:
            return _probe_loaded_rows(loaded, data, compact=compact)
        for task in data.task_ids:
            for pair_index, fraction_index, position, raw in probe_items(queries, task):
                if compact and (pair_index, fraction_index) != (0, 0):
                    continue
                teacher_demo, query_demo = PROBE_TEACHER_QUERY[pair_index]
                model = frozen
                for noise_index, seed in enumerate(PROBE_NOISE_SEEDS[:1] if compact else PROBE_NOISE_SEEDS):
                    metric = probe_one(model, raw, seed=seed)
                    rows.append({
                        "asset": asset,
                        "global_task_id": task,
                        "suite": data.tasks[task].suite,
                        "suite_task_id": data.tasks[task].suite_task_id,
                        "teacher_demo": teacher_demo,
                        "query_demo": query_demo,
                        "query_fraction": (0.25, 0.75)[fraction_index],
                        "query_position": position,
                        "noise_seed": seed,
                        **metric,
                    })
    finally:
        queries.close()
        data.close()
    return rows


def _probe_loaded_rows(loaded, data, *, compact: bool = False) -> list[dict[str, Any]]:
    queries = probe_dataset(ASSET_ROOT, data)
    cache = VideoConditionCache(loaded.runtime, data, 2**30)
    rows = []
    try:
        for task in data.task_ids:
            for pair_index, fraction_index, position, raw in probe_items(queries, task):
                if compact and (pair_index, fraction_index) != (0, 0):
                    continue
                teacher_demo, query_demo = PROBE_TEACHER_QUERY[pair_index]
                generated = writer_probe_state(loaded, cache, task=task, teacher_demo=teacher_demo)
                model = writer_frozen_policy(loaded, generated)
                seeds = PROBE_NOISE_SEEDS[:1] if compact else PROBE_NOISE_SEEDS
                for seed in seeds:
                    rows.append({
                        "asset": loaded.name,
                        "global_task_id": task,
                        "suite": data.tasks[task].suite,
                        "suite_task_id": data.tasks[task].suite_task_id,
                        "teacher_demo": teacher_demo,
                        "query_demo": query_demo,
                        "query_fraction": (0.25, 0.75)[fraction_index],
                        "query_position": position,
                        "noise_seed": seed,
                        **probe_one(model, raw, seed=seed),
                    })
    finally:
        queries.close()
    return rows


def e1_probe(output: Path, device: torch.device, asset: str) -> None:
    rows = _probe_rows(asset, device)
    expected = 36 * 8
    if len(rows) != expected:
        raise ValueError(f"E1 frozen probe row count differs: {len(rows)} != {expected}")
    _write_csv(output / "parts" / f"frozen_probe_rows_{asset}.csv", rows)
    for row in rows:
        _append_jsonl(output / "parts" / f"probe_manifest_{asset}.jsonl", {
            key: row[key] for key in (
                "asset", "global_task_id", "teacher_demo", "query_demo", "query_fraction", "query_position", "noise_seed"
            )
        })


def e1_rollout(output: Path, device: torch.device, asset: str, physical_gpu_id: int) -> None:
    from ember.writer.stability_rollouts import run_asset_rollouts

    loaded, frozen = _load_frozen(asset, device)
    if loaded is not None:
        loaded.runtime.state.eval()
    rows = run_asset_rollouts(
        asset=asset,
        asset_root=ASSET_ROOT,
        output=output / "rollouts" / "e1" / asset,
        physical_gpu_id=physical_gpu_id,
        current_evaluation_contract=NEW_ROOT / "evaluation/source_validation/run_contract.json",
        loaded_writer=loaded,
        frozen_policy=frozen,
    )
    _write_csv(output / "parts" / f"frozen_rollout_rows_{asset}.csv", rows)


def e3_rollout(output: Path, device: torch.device, branch: str, local_step: int,
               physical_gpu_id: int) -> None:
    from ember.writer.stability_rollouts import (
        E3_FULL_CAPTURE_CONDITIONS,
        E3_HELD_TASKS,
        run_asset_rollouts,
    )

    parent_asset = "O1200" if branch.startswith("O-") else "N1800"
    loaded = load_writer(
        name=f"{branch}@{local_step}", checkpoint=WRITER_ASSETS[parent_asset],
        asset_root=ASSET_ROOT, device=device,
    )
    checkpoint = output / "shadow" / branch / f"local_{local_step:03d}" / "ecp.safetensors"
    loaded.runtime.state.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    loaded.runtime.state.eval()
    rows = run_asset_rollouts(
        asset=loaded.name,
        asset_root=ASSET_ROOT,
        output=output / "rollouts" / "e3" / branch / f"local_{local_step:03d}",
        physical_gpu_id=physical_gpu_id,
        current_evaluation_contract=NEW_ROOT / "evaluation/source_validation/run_contract.json",
        loaded_writer=loaded,
        frozen_policy=None,
        panel_ids=E3_HELD_TASKS,
        compact_capture=True,
        full_capture_conditions=E3_FULL_CAPTURE_CONDITIONS,
    )
    normalized = [{**row, "branch": branch, "local_step": local_step, "parent_asset": parent_asset}
                  for row in rows]
    _write_csv(output / "parts" / f"shadow_rollout_rows_{branch}_{local_step:03d}.csv", normalized)


def _action_delta_mse(row: Mapping[str, Any], parent: Mapping[str, Any]) -> float:
    def tensor(value: Any) -> torch.Tensor:
        return torch.tensor(json.loads(value) if isinstance(value, str) else value, dtype=torch.float32)

    left = tensor(row["inference_first5_actions"])
    right = tensor(parent["inference_first5_actions"])
    if left.shape != right.shape:
        raise ValueError("virtual probe and parent action shapes differ")
    return float((left - right).square().mean())


def _probe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["global_task_id"]), int(row["teacher_demo"]), int(row["query_demo"]),
        float(row["query_fraction"]), int(row["query_position"]), int(row["noise_seed"]),
    )


def _parent_compact_probes(output: Path, asset: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = output / "frozen_probe_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["asset"] == asset
            and int(row["teacher_demo"]) == 46
            and int(row["query_demo"]) == 47
            and float(row["query_fraction"]) == 0.25
            and int(row["noise_seed"]) == PROBE_NOISE_SEEDS[0]
        ]
    parents = {_probe_key(row): row for row in rows}
    if len(rows) != 36 or len(parents) != 36:
        raise ValueError(f"E3 parent compact probe is incomplete for {asset}: {len(rows)}")
    return parents


def e3(output: Path, device: torch.device, branch: str) -> None:
    definitions = {
        "O-H": ("O1200", HIGH_LR), "O-L": ("O1200", LOW_LR),
        "N-H": ("N1800", HIGH_LR), "N-L": ("N1800", LOW_LR),
    }
    asset, lr = definitions[branch]
    current = _current_run()
    parent_step = int(WRITER_ASSETS[asset].name.rsplit("_", 1)[-1])
    loaded = load_writer(name=asset, checkpoint=WRITER_ASSETS[asset], asset_root=ASSET_ROOT,
                         device=device, fixed_lr=lr)
    data = current_training_data(ASSET_ROOT, current, planned_updates=1836)
    batches = event_window(data, first_step=1801, last_step=1836)
    cache = VideoConditionCache(loaded.runtime, data, 2**30)
    branch_root = output / "shadow" / branch
    branch_root.mkdir(parents=True, exist_ok=False)
    steps, exposures = [], []
    parent_by_key = _parent_compact_probes(output, asset)
    for local_step, batch in enumerate(batches, start=1):
        started = time.perf_counter()
        gradients, losses = [], []
        for draw in batch:
            gradient, metrics = writer_condition_gradient(
                loaded, data, cache, draw, component="joint", task_weight=0.25
            )
            gradients.append(gradient)
            losses.append(metrics)
            exposures.append({
                "branch": branch, "local_step": local_step, "parent_step": parent_step,
                "global_task_id": draw["task"], "occurrence": draw["occurrence"],
                "teacher_demo": draw["video_demos"][0], "query_seed": draw["query_seed"],
            })
        result = optimizer_step(loaded, gradients=gradients, weights=[1.0] * 4, lr=lr)
        steps.append({
            "branch": branch, "parent_asset": asset, "local_step": local_step,
            "parent_step": parent_step, "optimizer_step": parent_step + local_step,
            "lr": lr, "seconds": time.perf_counter() - started,
            "mean_main_loss": sum(row["main_loss"] for row in losses) / 4,
            "mean_teaching_loss": sum(row["teaching_loss"] for row in losses) / 4,
            **result,
        })
    if len(steps) != 36 or len(exposures) != 144:
        raise ValueError("E3 corrected window did not produce 36 updates and 144 task events")
    save_diagnostic_checkpoint(
        loaded, branch_root / "local_036", local_step=36, parent_step=parent_step
    )
    probe_rows = []
    for row in _probe_loaded_rows(loaded, data, compact=True):
        parent = parent_by_key[_probe_key(row)]
        probe_rows.append({
            "branch": branch, "parent_asset": asset, "local_step": 36,
            **{key: value for key, value in row.items() if key != "asset"},
            "parent_full_mse": float(parent["full_mse"]),
            "parent_tau1_first5_mse": float(parent["tau1_first5_mse"]),
            "parent_inference_first5_mse": float(parent["inference_first5_mse"]),
            "action_delta_mse_vs_parent": _action_delta_mse(row, parent),
        })
    if len(probe_rows) != 36:
        raise ValueError(f"E3 compact terminal probe row count differs: {len(probe_rows)} != 36")
    _write_csv(output / "parts" / f"shadow_training_steps_{branch}.csv", steps)
    _write_csv(output / "parts" / f"shadow_task_exposures_{branch}.csv", exposures)
    _write_csv(output / "parts" / f"shadow_probe_rows_{branch}.csv", probe_rows)
    write_json_atomic(branch_root / "completion.json", {
        "schema_version": DIAGNOSTIC_SCHEMA, "status": "complete", "branch": branch,
        "parent_asset": asset, "parent_step": parent_step, "local_updates": 36,
        "fixed_lr": lr, "scheduler_steps": 0, "test_use": False,
    })
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    for name in ("e0", "e1-probe", "e1-rollout", "e3", "e3-rollout"):
        command = sub.add_parser(name)
        command.add_argument("--device", default="cuda:0")
        if name in {"e1-probe", "e1-rollout"}:
            command.add_argument("--asset", required=True)
        if name == "e1-rollout":
            command.add_argument("--physical-gpu-id", type=int, required=True)
        if name == "e3-rollout":
            command.add_argument("--branch", choices=("O-H", "O-L", "N-H", "N-L"), required=True)
            command.add_argument("--local-step", type=int, choices=(36,), required=True)
            command.add_argument("--physical-gpu-id", type=int, required=True)
        if name == "e3":
            command.add_argument("--branch", choices=("O-H", "O-L", "N-H", "N-L"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        register(output)
        return
    device = _device(args.device)
    if args.command == "e0":
        e0(output, device)
    elif args.command == "e1-probe":
        e1_probe(output, device, args.asset)
    elif args.command == "e1-rollout":
        e1_rollout(output, device, args.asset, args.physical_gpu_id)
    elif args.command == "e3":
        e3(output, device, args.branch)
    else:
        e3_rollout(output, device, args.branch, args.local_step, args.physical_gpu_id)


if __name__ == "__main__":
    main()
