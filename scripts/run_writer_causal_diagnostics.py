#!/usr/bin/env python3
"""Run the registered Writer causal diagnostics without changing a formal model."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import torch

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.causal_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    PATH_ARMS,
    PATH_TASKS,
    PATH_TRAJECTORY_CONDITIONS,
    CausalRawInputCache,
    apply_virtual_candidate,
    compile_path_panel,
    gradient_linearity_residual,
    measure_batch_gradients,
    measure_gradient_windows,
    probe_path,
    registered_path_probes,
    rollout_path_states,
    validate_causal_parameter_groups,
)
from ember.writer.runtime import VideoConditionCache
from ember.writer.stability_diagnostics import (
    current_training_data,
    event_window,
    load_writer,
    probe_one,
    save_diagnostic_checkpoint,
    snapshot_parent,
    restore_parent,
    source_policy,
)


# The detached runtime supplies code only; task authorities and frozen assets remain canonical and read-only.
ASSET_ROOT = Path("/data1/user/ymdai/projects/EMBER")
STUDY_ROOT = Path("/data0/user/ymdai/ember_runs/writer_causal_diagnostics_20260922")
OLD_CHECKPOINT = Path("/data0/user/ymdai/ember_runs/video_teaching_20260919/training/checkpoints/macro_00001200")
CURRENT_ROOT = Path("/data0/user/ymdai/ember_runs/coverage_retraining_cross_episode_aux_20260922")
CURRENT_TRAINING = CURRENT_ROOT / "training/writer"
MTBC_ROOT = Path("/data0/user/ymdai/ember_runs/coverage_retraining_20260920")
MTBC_CHECKPOINT = MTBC_ROOT / "training/mtbc/checkpoints/step_00000300"
EVALUATION_CONTRACT = MTBC_ROOT / "evaluation/source_validation/run_contract.json"
FIXED_LR = 0.0002992056748283996
WRITER_ASSETS = {
    "O1200": OLD_CHECKPOINT,
    "C600": CURRENT_TRAINING / "checkpoints/macro_00000600",
    "C1200": CURRENT_TRAINING / "checkpoints/macro_00001200",
}


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("causal diagnostics require CUDA")
    torch.cuda.set_device(device)
    from ember.writer.topology import bind_current_process_to_cuda_numa

    if not bind_current_process_to_cuda_numa(torch.cuda.current_device()):
        raise ValueError("causal diagnostics require GPU-local NUMA placement")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    return device


def _json_cell(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":"), sort_keys=True) if isinstance(value, (list, tuple, dict)) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty causal table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key)) for key in fields})


def _write_empty_csv(path: Path, fields: Sequence[str]) -> None:
    """Preserve a registered output schema when the time-gated D3 is not started."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=sorted(fields)).writeheader()


def _read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _current_run() -> dict[str, Any]:
    return read_json(CURRENT_TRAINING / "run_contract.json")


def _cost(output: Path, key: str, started: float, **extra: Any) -> None:
    write_json_atomic(output / "parts" / f"cost_{key}.json", {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "key": key,
        "seconds": time.perf_counter() - started,
        **extra,
    })


def _release(*objects: Any) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def register(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    assets = {
        name: {
            "kind": "writer", "checkpoint": str(checkpoint.resolve()),
            "weights": _record(checkpoint / "ecp.safetensors"),
            "trainer_state": _record(checkpoint / "trainer_state.pt"),
            "manifest": _record(checkpoint / "checkpoint_manifest.json"),
        }
        for name, checkpoint in WRITER_ASSETS.items()
    }
    assets["M300"] = {"kind": "mtbc", "checkpoint": str(MTBC_CHECKPOINT.resolve()),
                       "weights": _record(MTBC_CHECKPOINT / "lora.safetensors")}
    source = _current_run()["source"]
    assets["Source"] = {"kind": "source", "checkpoint": source["checkpoint"], "model_path": source["model_path"]}
    write_json_atomic(output / "registration.json", {
        "schema_version": DIAGNOSTIC_SCHEMA, "status": "registered",
        "design": "docs/writer_causal_diagnostics_20260922.md", "assets": assets,
        "fresh_training": False, "validation_use": False, "test_use": False,
        "checkpoint_selection_use": False, "gpu_wallclock_limit_seconds": 7200,
        "d1_probe_count": 544, "d1_rollout_count": 112, "d2_unique_conditions": 72,
        "d2_virtual_candidates": 16, "d3_actual_updates": 54, "d3_rollout_count": 48,
    })


def _load_writer(asset: str, device: torch.device, *, fixed_lr: float | None = None):
    if asset not in WRITER_ASSETS:
        raise ValueError(f"unknown Writer asset: {asset}")
    return load_writer(name=asset, checkpoint=WRITER_ASSETS[asset], asset_root=ASSET_ROOT, device=device, fixed_lr=fixed_lr)


def _baseline_policy(asset: str, device: torch.device):
    current = _current_run()
    if asset == "Source":
        return source_policy(name=asset, asset_root=ASSET_ROOT, current_run=current, device=device)
    if asset == "M300":
        return source_policy(
            name=asset, asset_root=ASSET_ROOT, current_run=current, device=device,
            mtbc_config=ASSET_ROOT / "configs/libero_24_8_8_coverage_v1/mtbc.json",
            mtbc_checkpoint=MTBC_CHECKPOINT,
        )
    raise ValueError(f"unknown frozen baseline: {asset}")


def _optimizer_lr(loaded) -> float:
    values = {float(group["lr"]) for group in loaded.optimizer.param_groups}
    if len(values) != 1:
        raise ValueError("causal Writer optimizer has inconsistent parameter-group learning rates")
    return values.pop()


def d1_path(output: Path, device: torch.device, asset: str) -> None:
    started, current = time.perf_counter(), _current_run()
    data = current_training_data(ASSET_ROOT, current, planned_updates=1)
    loaded = _load_writer(asset, device)
    try:
        loaded.runtime.state.eval()
        requests = registered_path_probes(data)
        panel = compile_path_panel(loaded, CausalRawInputCache(loaded, data), check_cc=True)
        rows, reconstruction = [], []
        for request in requests:
            cc = panel[(request.task, request.cohort.label, "CC")]
            cc_row = probe_path(loaded, cc, request, reference=cc)
            rows.append(cc_row)
            reconstruction.append(cc.cc_reconstruction_max_abs)
            for arm in PATH_ARMS[1:]:
                rows.append(probe_path(loaded, panel[(request.task, request.cohort.label, arm)], request,
                                       reference=cc, reference_metric=cc_row))
        maximum = max(value for value in reconstruction if value is not None)
        if len(rows) != 160 or maximum > 1e-5:
            raise ValueError("causal CC reconstruction or path probe coverage failed")
        _write_csv(output / "parts" / f"path_probe_rows_{asset}.csv", rows)
        write_json_atomic(output / "parts" / f"path_cc_reconstruction_{asset}.json", {
            "asset": asset, "max_abs": maximum, "conditions": len(reconstruction),
        })
        _cost(output, f"d1_path_{asset}", started, rows=len(rows))
    finally:
        data.close()
        _release(loaded)


def d1_baseline(output: Path, device: torch.device, asset: str) -> None:
    started, current = time.perf_counter(), _current_run()
    data, policy = current_training_data(ASSET_ROOT, current, planned_updates=1), _baseline_policy(asset, device)
    try:
        rows = []
        for request in registered_path_probes(data):
            rows.append({
                "asset": asset, "arm": "CC", "global_task_id": request.task,
                "suite": data.tasks[request.task].suite, "suite_task_id": data.tasks[request.task].suite_task_id,
                "cohort": request.cohort.label, "teacher_demo": request.cohort.teacher_demo,
                "query_demo": request.cohort.query_demo, "other_demo": request.cohort.other_demo,
                "donor_global_task_id": None, "query_fraction": request.fraction,
                "query_position": request.query_position, "noise_seed": request.noise_seed,
                "cc_reconstruction_max_abs": None, "action_delta_mse_vs_cc": 0.0,
                **probe_one(policy, request.raw, seed=request.noise_seed),
            })
        if len(rows) != 32:
            raise ValueError("causal baseline probe coverage changed")
        _write_csv(output / "parts" / f"path_probe_rows_{asset}.csv", rows)
        _cost(output, f"d1_baseline_{asset}", started, rows=len(rows))
    finally:
        data.close()
        _release(policy)


def d1_rollout(output: Path, device: torch.device, asset: str, arm: str, physical_gpu_id: int) -> None:
    from ember.writer.stability_rollouts import run_asset_rollouts

    if asset == "O1200" and arm not in {"CC", "WW"}:
        raise ValueError("O1200 D1 rollout only registers CC and WW")
    started, current = time.perf_counter(), _current_run()
    data, loaded = current_training_data(ASSET_ROOT, current, planned_updates=1), _load_writer(asset, device)
    try:
        loaded.runtime.state.eval()
        panel = compile_path_panel(loaded, CausalRawInputCache(loaded, data), arms=(arm,), check_cc=arm == "CC")
        states, evidence = rollout_path_states(panel, arm=arm, asset=asset)
        rows = run_asset_rollouts(
            asset=asset, asset_root=ASSET_ROOT, output=output / "rollouts" / "d1" / asset / arm,
            physical_gpu_id=physical_gpu_id, current_evaluation_contract=EVALUATION_CONTRACT,
            loaded_writer=loaded, frozen_policy=None, panel_ids=PATH_TASKS, state_ids=(0, 1),
            compact_capture=True, full_capture_conditions=PATH_TRAJECTORY_CONDITIONS,
            precompiled_states=states, precompiled_evidence=evidence,
        )
        rows = [{**row, "path_arm": arm, "d1_asset": asset} for row in rows]
        if len(rows) != 16:
            raise ValueError("causal D1 rollout row count changed")
        _write_csv(output / "parts" / f"path_rollout_rows_{asset}_{arm}.csv", rows)
        _cost(output, f"d1_rollout_{asset}_{arm}", started, rows=len(rows))
    finally:
        data.close()
        _release(loaded)


def d2(output: Path, device: torch.device, asset: str) -> None:
    started, current = time.perf_counter(), _current_run()
    data = current_training_data(ASSET_ROOT, current, planned_updates=612)
    loaded = _load_writer(asset, device)
    try:
        groups = validate_causal_parameter_groups(loaded)
        batches = event_window(data, first_step=604, last_step=612)
        cache = VideoConditionCache(loaded.runtime, data, 1 << 30)
        windows, gradient_rows = measure_gradient_windows(loaded, data, cache, batches)
        linearity = max(gradient_linearity_residual(values["q"], values["a"], values["joint"])
                        for values in windows.values())
        if linearity > 1e-6:
            raise ValueError(f"causal native joint gradient decomposition residual is too large: {linearity}")
        loaded.runtime.state.eval()
        lr = _optimizer_lr(loaded)
        gradient_rows = [{"asset": asset, "linearity_max_abs": linearity,
                          "auxiliary_gradient_definition": "native_joint_vjp_minus_main_vjp",
                          "parameter_groups": groups, **row}
                         for row in gradient_rows]
        requests = registered_path_probes(data)
        outer_parent = snapshot_parent(loaded)
        virtual_rows, probe_rows = [], []
        for window, values in windows.items():
            restore_parent(loaded, outer_parent)
            parent_panel = compile_path_panel(loaded, CausalRawInputCache(loaded, data), arms=("CC",))
            parents = {(
                request.task, request.cohort.label, request.query_position
            ): probe_path(loaded, parent_panel[(request.task, request.cohort.label, "CC")], request,
                          reference=parent_panel[(request.task, request.cohort.label, "CC")])
                       for request in requests}
            for candidate in ("J", "Q", "M", "Z"):
                restore_parent(loaded, outer_parent)
                update = apply_virtual_candidate(loaded, q=values["q"], a=values["a"], joint=values["joint"],
                                                 candidate=candidate, lr=lr)
                panel = compile_path_panel(loaded, CausalRawInputCache(loaded, data), arms=("CC",))
                for request in requests:
                    compiled = panel[(request.task, request.cohort.label, "CC")]
                    row = probe_path(loaded, compiled, request, reference=compiled)
                    parent = parents[(request.task, request.cohort.label, request.query_position)]
                    action = torch.tensor(row["inference_first5_actions"], dtype=torch.float32)
                    prior = torch.tensor(parent["inference_first5_actions"], dtype=torch.float32)
                    probe_rows.append({"asset": asset, "window": window, "candidate": candidate,
                                       "action_delta_mse_vs_parent": float((action - prior).square().mean()), **row})
                virtual_rows.append({"asset": asset, "window": window, "draw_count": values["draw_count"],
                                     "main_loss": values["losses"]["main_loss"],
                                     "auxiliary_loss": values["losses"]["auxiliary_loss"], **update})
        restore_parent(loaded, outer_parent)
        if len(virtual_rows) != 8 or len(probe_rows) != 256:
            raise ValueError("causal D2 virtual candidate coverage changed")
        _write_csv(output / "parts" / f"gradient_rows_{asset}.csv", gradient_rows)
        _write_csv(output / "parts" / f"virtual_update_rows_{asset}.csv", virtual_rows)
        _write_csv(output / "parts" / f"virtual_probe_rows_{asset}.csv", probe_rows)
        _cost(output, f"d2_{asset}", started, virtual_candidates=len(virtual_rows), probes=len(probe_rows))
    finally:
        data.close()
        _release(loaded)


def d3(output: Path, device: torch.device, branch: str, physical_gpu_id: int) -> None:
    from ember.writer.stability_rollouts import run_asset_rollouts

    if branch not in {"J", "Q", "M"}:
        raise ValueError("causal D3 branch must be J, Q, or M")
    started, current = time.perf_counter(), _current_run()
    data = current_training_data(ASSET_ROOT, current, planned_updates=618)
    loaded = _load_writer("C600", device, fixed_lr=FIXED_LR)
    branch_root = output / "microtrain" / branch
    try:
        batches, cache = event_window(data, first_step=601, last_step=618), VideoConditionCache(loaded.runtime, data, 1 << 30)
        steps = []
        for local_step, draws in enumerate(batches, start=1):
            step_started = time.perf_counter()
            q, a, joint, metrics = measure_batch_gradients(loaded, data, cache, draws)
            update = apply_virtual_candidate(loaded, q=q, a=a, joint=joint, candidate=branch, lr=FIXED_LR)
            steps.append({"branch": branch, "local_step": local_step, "parent_step": 600,
                          "global_event_step": 600 + local_step, "seconds": time.perf_counter() - step_started,
                          **metrics, **update})
        if len(steps) != 18:
            raise ValueError("causal D3 update count changed")
        loaded.runtime.state.eval()
        save_diagnostic_checkpoint(loaded, branch_root / "final", local_step=18, parent_step=600)
        panel = compile_path_panel(loaded, CausalRawInputCache(loaded, data), arms=("CC", "WW"))
        probes = []
        for request in registered_path_probes(data):
            cc = panel[(request.task, request.cohort.label, "CC")]
            cc_row = probe_path(loaded, cc, request, reference=cc)
            probes.append({"branch": branch, **cc_row})
            probes.append({"branch": branch, **probe_path(loaded, panel[(request.task, request.cohort.label, "WW")], request,
                                                             reference=cc, reference_metric=cc_row)})
        states, evidence = rollout_path_states(panel, arm="CC", asset=f"C600-{branch}")
        rollout_rows = run_asset_rollouts(
            asset=f"C600-{branch}", asset_root=ASSET_ROOT, output=branch_root / "rollouts",
            physical_gpu_id=physical_gpu_id, current_evaluation_contract=EVALUATION_CONTRACT,
            loaded_writer=loaded, frozen_policy=None, panel_ids=PATH_TASKS, state_ids=(0, 1),
            compact_capture=True, full_capture_conditions=PATH_TRAJECTORY_CONDITIONS,
            precompiled_states=states, precompiled_evidence=evidence,
        )
        if len(probes) != 64 or len(rollout_rows) != 16:
            raise ValueError("causal D3 endpoint coverage changed")
        _write_csv(output / "parts" / f"microtrain_steps_{branch}.csv", steps)
        _write_csv(output / "parts" / f"microtrain_probe_rows_{branch}.csv", probes)
        _write_csv(output / "parts" / f"microtrain_rollout_rows_{branch}.csv", [{"branch": branch, **row} for row in rollout_rows])
        write_json_atomic(branch_root / "completion.json", {
            "schema_version": DIAGNOSTIC_SCHEMA, "status": "complete", "branch": branch,
            "updates": 18, "fixed_lr": FIXED_LR, "intermediate_probe_use": False,
        })
        _cost(output, f"d3_{branch}", started, updates=len(steps), probes=len(probes), rollouts=len(rollout_rows))
    finally:
        data.close()
        _release(loaded)


def _number(value: str | None) -> float:
    return float(value) if value not in (None, "", "None") else float("nan")


def _probe_effects(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Mapping[str, str]]] = {}
    for row in rows:
        if row.get("asset") not in {"O1200", "C600", "C1200"}:
            continue
        key = tuple(row.get(name, "") for name in ("asset", "global_task_id", "cohort", "query_position", "noise_seed", "query_fraction"))
        grouped.setdefault(key, {})[row["arm"]] = row
    effects = []
    for key, arms in grouped.items():
        if not set(PATH_ARMS) <= set(arms):
            continue
        for metric in ("full_mse", "first5_mse", "tau1_first5_mse", "inference_first5_mse"):
            cc, cw, wc, ww = (_number(arms[label].get(metric)) for label in ("CC", "CW", "WC", "WW"))
            for label, value in (("cc_minus_cw", cc - cw), ("wc_minus_ww", wc - ww),
                                 ("interaction", cc - cw - wc + ww)):
                effects.append({"kind": "probe", "metric": metric, "effect": label, "value": value,
                                "asset": key[0], "global_task_id": key[1], "cohort": key[2],
                                "query_position": key[3], "noise_seed": key[4], "query_fraction": key[5]})
    return effects


def _rollout_effects(rollouts: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    effects = []
    for asset in {row.get("d1_asset", "") for row in rollouts}:
        by_arm = {arm: [row for row in rollouts if row.get("d1_asset") == asset and row.get("path_arm") == arm]
                  for arm in PATH_ARMS}
        for arm, values in by_arm.items():
            if values:
                effects.append({"kind": "rollout", "metric": "success", "effect": "count", "value": sum(row.get("success") == "True" for row in values),
                                "asset": asset, "arm": arm, "total": len(values)})
    return effects


def _effects(rows: Sequence[Mapping[str, str]], rollouts: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [*_probe_effects(rows), *_rollout_effects(rollouts)]


def finalize(output: Path, *, d3_status: str) -> None:
    if d3_status not in {"complete", "not_started_insufficient_preregistered_time"}:
        raise ValueError(f"unsupported D3 finalization status: {d3_status}")
    started = time.perf_counter()
    parts = output / "parts"
    path_rows = _read_rows(sorted(parts.glob("path_probe_rows_*.csv")))
    rollout_rows = _read_rows(sorted(parts.glob("path_rollout_rows_*.csv")))
    gradients = _read_rows(sorted(parts.glob("gradient_rows_*.csv")))
    updates = _read_rows(sorted(parts.glob("virtual_update_rows_*.csv")))
    micro_steps = _read_rows(sorted(parts.glob("microtrain_steps_*.csv")))
    micro_probes = _read_rows(sorted(parts.glob("microtrain_probe_rows_*.csv")))
    micro_rollouts = _read_rows(sorted(parts.glob("microtrain_rollout_rows_*.csv")))
    expected_micro = (54, 192, 48) if d3_status == "complete" else (0, 0, 0)
    if (len(path_rows), len(rollout_rows), len(updates), len(micro_steps), len(micro_probes), len(micro_rollouts)) != (544, 112, 16, *expected_micro):
        raise ValueError("causal diagnostic final row counts are incomplete")
    _write_csv(output / "path_probe_rows.csv", path_rows)
    _write_csv(output / "path_rollout_rows.csv", rollout_rows)
    _write_csv(output / "gradient_rows.csv", gradients)
    _write_csv(output / "virtual_update_rows.csv", updates)
    if d3_status == "complete":
        _write_csv(output / "microtrain_steps.csv", micro_steps)
        _write_csv(output / "microtrain_probe_rows.csv", micro_probes)
        _write_csv(output / "microtrain_rollout_rows.csv", micro_rollouts)
    else:
        _write_empty_csv(output / "microtrain_steps.csv", ("branch", "local_step", "global_event_step"))
        _write_empty_csv(output / "microtrain_probe_rows.csv", ("branch", "global_task_id", "arm"))
        _write_empty_csv(output / "microtrain_rollout_rows.csv", ("branch", "global_task_id", "success"))
    _write_csv(output / "path_effects.csv", _effects(path_rows, rollout_rows))
    trajectory = [{key: row.get(key) for key in ("d1_asset", "path_arm", "branch", "global_task_id", "suite", "task_id", "init_state_id", "success", "steps")}
                  for row in [*rollout_rows, *micro_rollouts]]
    _write_csv(output / "trajectory_manifest.csv", trajectory)
    _cost(output, "finalize", started)
    costs = [read_json(path) for path in sorted(parts.glob("cost_*.json"))]
    write_json_atomic(output / "costs.json", {"schema_version": DIAGNOSTIC_SCHEMA, "stages": costs,
                                                 "total_recorded_seconds": sum(float(row["seconds"]) for row in costs)})
    write_json_atomic(output / "completion.json", {
        "schema_version": DIAGNOSTIC_SCHEMA, "status": "complete", "fresh_training": False,
        "validation_use": False, "test_use": False, "checkpoint_selection_use": False,
        "d3": {"status": d3_status},
        "rows": {"path_probes": 544, "path_rollouts": 112, "virtual_updates": 16,
                 "micro_updates": expected_micro[0], "micro_probes": expected_micro[1], "micro_rollouts": expected_micro[2]},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    for name in ("d1-path", "d1-baseline", "d2"):
        item = sub.add_parser(name)
        item.add_argument("--asset", required=True)
        item.add_argument("--device", default="cuda:0")
    item = sub.add_parser("d1-rollout")
    item.add_argument("--asset", choices=("O1200", "C600"), required=True)
    item.add_argument("--arm", choices=PATH_ARMS, required=True)
    item.add_argument("--device", default="cuda:0")
    item.add_argument("--physical-gpu-id", type=int, required=True)
    item = sub.add_parser("d3")
    item.add_argument("--branch", choices=("J", "Q", "M"), required=True)
    item.add_argument("--device", default="cuda:0")
    item.add_argument("--physical-gpu-id", type=int, required=True)
    item = sub.add_parser("finalize")
    item.add_argument("--d3-status", choices=("complete", "not_started_insufficient_preregistered_time"), default="complete")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        register(output)
        return
    if args.command == "finalize":
        finalize(output, d3_status=args.d3_status)
        return
    device = _device(args.device)
    if args.command == "d1-path":
        d1_path(output, device, args.asset)
    elif args.command == "d1-baseline":
        d1_baseline(output, device, args.asset)
    elif args.command == "d1-rollout":
        d1_rollout(output, device, args.asset, args.arm, args.physical_gpu_id)
    elif args.command == "d2":
        d2(output, device, args.asset)
    else:
        d3(output, device, args.branch, args.physical_gpu_id)


if __name__ == "__main__":
    main()
