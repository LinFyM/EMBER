#!/usr/bin/env python3
"""Run the fixed Writer stability diagnostics without changing formal models."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import VideoConditionCache
from ember.writer.stability_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    HIGH_LR,
    LOW_LR,
    PROBE_NOISE_SEEDS,
    PROBE_TEACHER_QUERY,
    TEACHING_WEIGHT,
    combine_gradients,
    current_training_data,
    displacement_from_parent,
    event_window,
    gradient_dot,
    grouped_gradient_metrics,
    load_writer,
    optimizer_step,
    probe_dataset,
    probe_items,
    probe_one,
    restore_parent,
    save_diagnostic_checkpoint,
    snapshot_parent,
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
        "stage_order": ["E0", "E1", "E2", "E3"],
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
        return None, source_policy(
            name=asset, asset_root=ASSET_ROOT, current_run=current, device=device,
            mtbc_config=Path(adapter["config"]["path"]), mtbc_checkpoint=MTBC_CHECKPOINT,
        )
    raise ValueError(f"unknown diagnostic asset: {asset}")


def _probe_rows(asset: str, device: torch.device, *, compact: bool = False) -> list[dict[str, Any]]:
    current = _current_run()
    data = current_training_data(ASSET_ROOT, current, planned_updates=1)
    queries = probe_dataset(ASSET_ROOT, data)
    loaded, frozen = _load_frozen(asset, device)
    cache = VideoConditionCache(loaded.runtime, data, 2**30) if loaded is not None else None
    rows = []
    try:
        for task in data.task_ids:
            for pair_index, fraction_index, position, raw in probe_items(queries, task):
                if compact and (pair_index, fraction_index) != (0, 0):
                    continue
                teacher_demo, query_demo = PROBE_TEACHER_QUERY[pair_index]
                model = frozen
                if loaded is not None:
                    generated = writer_probe_state(loaded, cache, task=task, teacher_demo=teacher_demo)
                    model = writer_frozen_policy(loaded, generated)
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


def _sum_gradient(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor], scale: float) -> tuple[torch.Tensor, ...]:
    return tuple(a + b.mul(scale) for a, b in zip(left, right, strict=True))


def _gradient_gram(asset: str, names: Sequence[str], gradients: Mapping[str, Sequence[Sequence[torch.Tensor]]], tasks: Sequence[int]):
    rows = []
    for component, vectors in gradients.items():
        for i, left_task in enumerate(tasks):
            for j, right_task in enumerate(tasks):
                rows.append({
                    "asset": asset,
                    "component": component,
                    "left_task": left_task,
                    "right_task": right_task,
                    "dot": gradient_dot(vectors[i], vectors[j]),
                })
    return rows


def _norm_matched_step(loaded, parent_model, gradient, displacement: float) -> None:
    loaded.runtime.state.load_state_dict(parent_model, strict=True)
    norm = math.sqrt(sum(float(value.square().sum()) for value in gradient))
    if not math.isfinite(norm) or norm <= 0 or displacement < 0:
        raise FloatingPointError("invalid norm-matched displacement")
    scale = displacement / norm
    with torch.no_grad():
        for parameter, value in zip(loaded.runtime.state.parameters(), gradient, strict=True):
            parameter.add_(value.to(parameter), alpha=-scale)


def e2(output: Path, device: torch.device, asset: str) -> None:
    if asset not in {"O1200", "N1800"}:
        raise ValueError("E2 only admits O1200 or N1800")
    current = _current_run()
    loaded = load_writer(name=asset, checkpoint=WRITER_ASSETS[asset], asset_root=ASSET_ROOT,
                         device=device, fixed_lr=HIGH_LR)
    data = current_training_data(ASSET_ROOT, current, planned_updates=1809)
    batches = event_window(data, first_step=1801, last_step=1809)
    cache = VideoConditionCache(loaded.runtime, data, 2**30)
    tasks = [int(draw["task"]) for batch in batches for draw in batch]
    if len(tasks) != len(set(tasks)) != 0 or set(tasks) != set(data.task_ids):
        raise ValueError("E2 events1801..1809 do not cover each current task exactly once")
    q_vectors, s_vectors, j_vectors, metric_rows = [], [], [], []
    draws = {int(draw["task"]): draw for batch in batches for draw in batch}
    for task in tasks:
        q, qm = writer_condition_gradient(loaded, data, cache, draws[task], component="main")
        s, sm = writer_condition_gradient(loaded, data, cache, draws[task], component="teaching")
        j = _sum_gradient(q, s, TEACHING_WEIGHT)
        q_vectors.append(q)
        s_vectors.append(s)
        j_vectors.append(j)
        metric_rows.append({
            "asset": asset, "global_task_id": task, "suite": data.tasks[task].suite,
            "suite_task_id": data.tasks[task].suite_task_id,
            "q_s_inner_product": gradient_dot(q, s),
            **{("q_" + key): value for key, value in grouped_gradient_metrics(loaded.parameter_names, q).items()},
            **{("s_" + key): value for key, value in grouped_gradient_metrics(loaded.parameter_names, s).items()},
            **{("j_" + key): value for key, value in grouped_gradient_metrics(loaded.parameter_names, j).items()},
            **qm, **{("teaching_" + key): value for key, value in sm.items()},
        })
    _write_csv(output / "parts" / f"task_gradient_metrics_{asset}.csv", metric_rows)
    gram = _gradient_gram(asset, loaded.parameter_names,
                          {"gQ": q_vectors, "gS": s_vectors, "gJ": j_vectors}, tasks)
    _write_csv(output / "parts" / f"task_gradient_gram_{asset}.csv", gram)

    parent = snapshot_parent(loaded)
    parent_model = parent[0]
    update_specs = []
    for index, batch in enumerate(batches, start=1):
        batch_tasks = [int(draw["task"]) for draw in batch]
        positions = [tasks.index(task) for task in batch_tasks]
        update_specs.extend((
            (f"batch_{index:02d}_joint", [j_vectors[p] for p in positions], [0.25] * 4),
            (f"batch_{index:02d}_main", [q_vectors[p] for p in positions], [0.25] * 4),
        ))
    update_specs.extend((
        ("all36_joint", j_vectors, [1 / 36] * 36),
        ("all36_main", q_vectors, [1 / 36] * 36),
        ("zero_current_gradient", [tuple(torch.zeros_like(v) for v in q_vectors[0])], [1.0]),
    ))
    virtual_rows = []
    for update_name, vectors, weights in update_specs:
        restore_parent(loaded, parent)
        combined = tuple(sum(vector[i].mul(weight) for vector, weight in zip(vectors, weights, strict=True))
                         for i in range(len(vectors[0])))
        step = optimizer_step(loaded, gradients=[combined], weights=[1.0], lr=HIGH_LR)
        virtual_rows.append({"asset": asset, "update": update_name, "measurement": "adam", **step})
        if update_name != "zero_current_gradient":
            _norm_matched_step(loaded, parent_model, combined, step["parameter_displacement_norm"])
            virtual_rows.append({
                "asset": asset, "update": update_name, "measurement": "norm_matched_raw_gradient",
                "preclip_grad_norm": math.sqrt(sum(float(v.square().sum()) for v in combined)),
                "parameter_displacement_norm": displacement_from_parent(loaded, parent_model), "lr": HIGH_LR,
            })
    restore_parent(loaded, parent)
    _write_csv(output / "parts" / f"virtual_update_rows_{asset}.csv", virtual_rows)
    data.close()


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
    data = current_training_data(ASSET_ROOT, current, planned_updates=1872)
    batches = event_window(data, first_step=1801, last_step=1872)
    cache = VideoConditionCache(loaded.runtime, data, 2**30)
    branch_root = output / "shadow" / branch
    branch_root.mkdir(parents=True, exist_ok=False)
    steps, exposures = [], []
    save_diagnostic_checkpoint(loaded, branch_root / "local_000", local_step=0, parent_step=parent_step)
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
        if local_step in {18, 36, 72}:
            save_diagnostic_checkpoint(loaded, branch_root / f"local_{local_step:03d}",
                                       local_step=local_step, parent_step=parent_step)
    _write_csv(output / "parts" / f"shadow_training_steps_{branch}.csv", steps)
    _write_csv(output / "parts" / f"shadow_task_exposures_{branch}.csv", exposures)
    write_json_atomic(branch_root / "completion.json", {
        "schema_version": DIAGNOSTIC_SCHEMA, "status": "complete", "branch": branch,
        "parent_asset": asset, "parent_step": parent_step, "local_updates": 72,
        "fixed_lr": lr, "scheduler_steps": 0, "test_use": False,
    })
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    for name in ("e0", "e1-probe", "e2", "e3"):
        command = sub.add_parser(name)
        command.add_argument("--device", default="cuda:0")
        if name in {"e1-probe", "e2"}:
            command.add_argument("--asset", required=True)
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
    elif args.command == "e2":
        e2(output, device, args.asset)
    else:
        e3(output, device, args.branch)


if __name__ == "__main__":
    main()
