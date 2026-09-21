#!/usr/bin/env python3
"""Run the frozen Writer output-space and code diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import RawTeacherVideoStore
from ember.writer.output_space_diagnostics import (
    FAMILIES,
    checkpoint_b_templates,
    checkpoint_head_matrix,
    column_space,
    compile_with_codes,
    condition_geometry,
    cross_video_geometry,
    space_overlap,
    space_summary,
    transform_state,
)
from ember.writer.stability_diagnostics import load_writer
from ember.writer.stability_rollouts import (
    CURRENT_PROTOCOL,
    DiagnosticWriterAdapter,
    E3_HELD_TASKS,
    compile_panel,
    run_asset_rollouts,
)


ASSET_ROOT = Path("/data1/user/ymdai/projects/EMBER")
STUDY_ROOT = Path("/data0/user/ymdai/ember_runs/writer_output_space_diagnostics_20260921")
STABILITY_ROOT = Path("/data0/user/ymdai/ember_runs/writer_stability_diagnostics_20260921")
OLD_ROOT = Path("/data0/user/ymdai/ember_runs/video_teaching_20260919/training")
NEW_ROOT = Path("/data0/user/ymdai/ember_runs/coverage_retraining_20260920")
CHECKPOINTS = {
    "O1200": OLD_ROOT / "checkpoints/macro_00001200",
    "N1000": NEW_ROOT / "training/writer/checkpoints/macro_00001000",
    "N1800": NEW_ROOT / "training/writer/checkpoints/macro_00001800",
}
A1_TASKS = (5, 7, 12, 37)
A1_DEMOS = (46, 48)
SCHEMA = "ember_writer_output_space_diagnostics_v1"


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type != "cuda":
        raise ValueError("video compilation and rollout require CUDA")
    torch.cuda.set_device(result)
    from ember.writer.topology import bind_current_process_to_cuda_numa

    if not bind_current_process_to_cuda_numa(torch.cuda.current_device()):
        raise ValueError("diagnostics require GPU-local NUMA placement")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    return result


def _json_cell(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":"), sort_keys=isinstance(value, dict)) \
        if isinstance(value, (dict, list, tuple)) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty output table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"inconsistent output fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _json_cell(value) for key, value in row.items()} for row in rows)


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _baseline_rows() -> list[dict[str, str]]:
    path = STABILITY_ROOT / "parts/frozen_rollout_rows_O1200.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["global_task_id"]) in E3_HELD_TASKS]
    keys = {(int(row["global_task_id"]), int(row["init_state_id"])) for row in rows}
    expected = {(task, state) for task in E3_HELD_TASKS for state in range(4)}
    if len(rows) != 16 or keys != expected:
        raise ValueError("existing O1200 held reference is not the registered 16-row panel")
    return rows


def register(output: Path, runtime_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None:
        raise ValueError("registration requires the clean pushed runtime commit")
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    for name, checkpoint in CHECKPOINTS.items():
        files = (checkpoint / "ecp.safetensors", checkpoint / "checkpoint_manifest.json")
        if not all(path.is_file() for path in files):
            raise ValueError(f"missing registered Writer checkpoint: {name}")
        assets[name] = {"checkpoint": str(checkpoint.resolve()),
                        "weights": _record(files[0]), "manifest": _record(files[1])}
    baseline = _baseline_rows()
    trajectories = STABILITY_ROOT / "rollouts/e1/O1200/trajectories"
    for row in baseline:
        path = trajectories / (
            f"{row['suite']}_task_{int(row['task_id']):02d}_state_{int(row['init_state_id']):03d}.pt"
        )
        if not path.is_file():
            raise ValueError(f"missing O1200 replay trajectory: {path.name}")
    _write_csv(output / "baseline_o1200_rows.csv", baseline)
    write_json_atomic(output / "registration.json", {
        "schema_version": SCHEMA,
        "status": "registered",
        "design": "docs/writer_output_space_projection_design.md",
        "runtime_commit": runtime_commit,
        "assets": assets,
        "a1_tasks": list(A1_TASKS), "a1_demos": list(A1_DEMOS),
        "a2_tasks": list(E3_HELD_TASKS), "a2_states": list(range(4)),
        "arms": ["SELF", "NEWSPACE", "SHRINK"],
        "fresh_training": False, "backward": False, "test_use": False,
        "checkpoint_selection_use": False,
    })


def a0(output: Path) -> None:
    summaries, singular_rows, overlap_rows, bases = [], [], [], {}
    template_rows = []
    for asset, checkpoint in CHECKPOINTS.items():
        template_rows.append({"asset": asset, **checkpoint_b_templates(checkpoint)})
        for family in FAMILIES:
            summary, basis = space_summary(asset, checkpoint, family)
            bases[(asset, family)] = basis
            singular = summary.pop("singular_values")
            summaries.append(summary)
            singular_rows.extend({"asset": asset, "family": family, "value_kind": "head_singular",
                                  "index": index, "value": value}
                                 for index, value in enumerate(singular))
    for family in FAMILIES:
        for left_index, left in enumerate(CHECKPOINTS):
            for right in tuple(CHECKPOINTS)[left_index + 1:]:
                row = space_overlap(left, bases[(left, family)], right, bases[(right, family)], family)
                cosines = row.pop("principal_cosines")
                overlap_rows.append(row)
                singular_rows.extend({"asset": f"{left}__{right}", "family": family,
                                      "value_kind": "principal_cosine", "index": index, "value": value}
                                     for index, value in enumerate(cosines))
    if any(row["b_nonzero_values"] != 0 for row in template_rows):
        raise ValueError("a registered Writer checkpoint has nonzero B templates")
    _write_csv(output / "a0_space_summary.csv", summaries)
    _write_csv(output / "a0_singular_and_principal_values.csv", singular_rows)
    _write_csv(output / "a0_space_overlap.csv", overlap_rows)
    write_json_atomic(output / "a0_completion.json", {
        "schema_version": SCHEMA, "status": "complete", "assets": list(CHECKPOINTS),
        "families": list(FAMILIES), "template_checks": template_rows,
    })


def _a1_tasks(asset_root: Path):
    tasks = load_learning_tasks(asset_root, A1_TASKS, role="train", protocol_path=CURRENT_PROTOCOL)
    if set(tasks) != set(A1_TASKS):
        raise ValueError("A1 train task authorities changed")
    return tasks


def a1(output: Path, asset_root: Path, asset: str, device: torch.device) -> None:
    if asset not in {"O1200", "N1000"}:
        raise ValueError("A1 only reads O1200 and N1000")
    loaded = load_writer(name=asset, checkpoint=CHECKPOINTS[asset], asset_root=asset_root, device=device)
    loaded.runtime.state.eval()
    tasks = _a1_tasks(asset_root)
    store = RawTeacherVideoStore(tuple(task.authority for task in tasks.values()), frame_stride=5,
                                 camera_view=loaded.run["config"]["observer"]["camera_view"])
    heads = {family: checkpoint_head_matrix(CHECKPOINTS[asset], family) for family in FAMILIES}
    scale = float(loaded.runtime.lora.alpha) / float(loaded.runtime.lora.rank)
    tensors, code_rows, factor_rows, cross_rows = {}, [], [], []
    try:
        per_video = {}
        for task_id in A1_TASKS:
            task = tasks[task_id]
            for demo in A1_DEMOS:
                video = store.load(task_id, demo)
                condition = loaded.runtime.prepare(
                    (torch.from_numpy(video.frames),), (torch.from_numpy(video.frame_indices),),
                    task.authority.language,
                )
                state, codes = compile_with_codes(loaded, condition)
                for family in FAMILIES:
                    tensors[f"task{task_id}.demo{demo}.{family}"] = codes[family]
                per_video[(task_id, demo)] = codes
                rows, factors = condition_geometry(
                    asset=asset, task=task_id, demo=demo, state=state, codes=codes,
                    heads=heads, scale=scale,
                )
                code_rows.extend(rows)
                factor_rows.extend(factors)
        for task_id in A1_TASKS:
            for family in FAMILIES:
                cross_rows.append({
                    "asset": asset, "global_task_id": task_id, "family": family,
                    "left_demo": A1_DEMOS[0], "right_demo": A1_DEMOS[1],
                    **cross_video_geometry(
                        per_video[(task_id, A1_DEMOS[0])][family],
                        per_video[(task_id, A1_DEMOS[1])][family],
                    ),
                })
    finally:
        store.close()
    save_file(tensors, str(output / f"a1_codes_{asset}.safetensors"))
    _write_csv(output / f"a1_code_geometry_{asset}.csv", code_rows)
    _write_csv(output / f"a1_factor_geometry_{asset}.csv", factor_rows)
    _write_csv(output / f"a1_cross_video_geometry_{asset}.csv", cross_rows)
    write_json_atomic(output / f"a1_completion_{asset}.json", {
        "schema_version": SCHEMA, "status": "complete", "asset": asset,
        "compilations": len(A1_TASKS) * len(A1_DEMOS), "backward": False,
        "code_tensors": len(tensors), "factor_rows": len(factor_rows),
    })


def _head_bases() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    old, new = {}, {}
    for family in FAMILIES:
        old[family] = column_space(checkpoint_head_matrix(CHECKPOINTS["O1200"], family))[0]
        new[family] = column_space(checkpoint_head_matrix(CHECKPOINTS["N1000"], family))[0]
    return old, new


def _first_replan_inputs(evidence: Mapping[str, Mapping[str, Any]], device: torch.device):
    root = STABILITY_ROOT / "rollouts/e1/O1200/trajectories"
    groups = []
    for task_id in E3_HELD_TASKS:
        selected = sorted((key, row) for key, row in evidence.items()
                          if int(row["global_task_id"]) == task_id)
        if len(selected) != 4:
            raise ValueError("SELF replay evidence is incomplete")
        trajectories = []
        for key, row in selected:
            path = root / (
                f"{row['suite']}_task_{int(row['task_id']):02d}_state_{int(row['init_state_id']):03d}.pt"
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != "ember_pi05_occupancy_trajectory_v1":
                raise ValueError("existing O1200 trajectory schema changed")
            trajectories.append((
                key, row, payload["observations"][0], int(payload["policy_noise_seeds"][0]),
                payload["action_chunks"][0],
            ))
        processed = [item[2] for item in trajectories]
        batch = {key: torch.cat([row[key] for row in processed], dim=0).to(device)
                 for key in processed[0] if isinstance(processed[0][key], torch.Tensor)}
        noises = []
        for _key, _row, _observation, seed, _chunk in trajectories:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            noises.append(torch.randn((50, 32), generator=generator, dtype=torch.float32))
        reference = torch.cat([chunk for _key, _row, _observation, _seed, chunk in trajectories], dim=0)
        groups.append((task_id, trajectories, batch, torch.stack(noises).to(device), reference))
    return groups


@torch.no_grad()
def _predict_replans(loaded, states, evidence, groups) -> dict[tuple[int, int], torch.Tensor]:
    adapter = DiagnosticWriterAdapter(loaded, states, evidence)
    result = {}
    try:
        for task_id, trajectories, batch, noise, _reference in groups:
            prepared = [adapter.prepare_episode(suite=row["suite"], task_id=int(row["task_id"]),
                                                init_state_id=int(row["init_state_id"]))
                        for _key, row, _observation, _seed, _chunk in trajectories]
            chunks = adapter.predict_action_chunk(prepared, batch, noise=noise, num_steps=10).detach().cpu()
            for index, (_key, row, _observation, _seed, _chunk) in enumerate(trajectories):
                result[(task_id, int(row["init_state_id"]))] = chunks[index:index + 1]
    finally:
        adapter.close()
    return result


def _error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    difference = left.float() - right.float()
    return float(difference.abs().max()), float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(right.float()).clamp_min(1e-30)
    )


def _self_replan_check(loaded, original, transformed, evidence, device) -> list[dict[str, Any]]:
    groups = _first_replan_inputs(evidence, device)
    original_chunks = _predict_replans(loaded, original, evidence, groups)
    self_chunks = _predict_replans(loaded, transformed, evidence, groups)
    rows = []
    for task_id, trajectories, _batch, _noise, reference in groups:
        for index, (_key, row, _observation, _seed, _chunk) in enumerate(trajectories):
            key = (task_id, int(row["init_state_id"]))
            stored = reference[index:index + 1]
            original_max, original_relative = _error(original_chunks[key], stored)
            self_max, self_relative = _error(self_chunks[key], stored)
            pair_max, pair_relative = _error(self_chunks[key], original_chunks[key])
            rows.append({
                "global_task_id": task_id, "init_state_id": key[1],
                "original_vs_stored_max_abs": original_max,
                "original_vs_stored_relative_l2": original_relative,
                "self_vs_stored_max_abs": self_max,
                "self_vs_stored_relative_l2": self_relative,
                "self_vs_original_max_abs": pair_max,
                "self_vs_original_relative_l2": pair_relative,
                "finite": all(torch.isfinite(value).all() for value in (stored, original_chunks[key], self_chunks[key])),
            })
    return rows


def a2(output: Path, asset_root: Path, arm: str, device: torch.device, physical_gpu_id: int) -> None:
    loaded = load_writer(name="O1200", checkpoint=CHECKPOINTS["O1200"], asset_root=asset_root, device=device)
    loaded.runtime.state.eval()
    original, evidence = compile_panel(loaded, asset_root=asset_root, task_ids=E3_HELD_TASKS)
    old_bases, new_bases = _head_bases()
    scale = float(loaded.runtime.lora.alpha) / float(loaded.runtime.lora.rank)
    transformed, projection_rows = {}, []
    for key in sorted(original):
        state, rows = transform_state(original[key], arm=arm, old_bases=old_bases,
                                      new_bases=new_bases, scale=scale)
        transformed[key] = state
        condition = evidence[key]
        projection_rows.extend({
            "global_task_id": int(condition["global_task_id"]),
            "init_state_id": int(condition["init_state_id"]),
            "teacher_demo": int(condition["teacher_demo"]), **row,
        } for row in rows)
        evidence[key] = {**condition, "asset": arm, "intervention": arm}
    if arm == "SELF":
        _write_csv(output / "a2_self_first_replan.csv",
                   _self_replan_check(loaded, original, transformed, evidence, device))
    rows = run_asset_rollouts(
        asset=arm, asset_root=asset_root, output=output / "rollouts" / arm,
        physical_gpu_id=physical_gpu_id,
        current_evaluation_contract=NEW_ROOT / "evaluation/source_validation/run_contract.json",
        loaded_writer=loaded, frozen_policy=None, panel_ids=E3_HELD_TASKS,
        compact_capture=True, precompiled_states=transformed, precompiled_evidence=evidence,
    )
    _write_csv(output / f"a2_projection_rows_{arm}.csv", projection_rows)
    _write_csv(output / f"a2_rollout_rows_{arm}.csv", rows)
    write_json_atomic(output / f"a2_completion_{arm}.json", {
        "schema_version": SCHEMA, "status": "complete", "arm": arm,
        "rollouts": len(rows), "projection_rows": len(projection_rows),
        "backward": False, "optimizer_updates": 0, "test_use": False,
    })
    del loaded
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY_ROOT)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--runtime-commit")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    sub.add_parser("a0")
    a1_parser = sub.add_parser("a1")
    a1_parser.add_argument("--asset", choices=("O1200", "N1000"), required=True)
    a1_parser.add_argument("--device", default="cuda:0")
    a2_parser = sub.add_parser("a2")
    a2_parser.add_argument("--arm", choices=("SELF", "NEWSPACE", "SHRINK"), required=True)
    a2_parser.add_argument("--device", default="cuda:0")
    a2_parser.add_argument("--physical-gpu-id", type=int, required=True)
    args = parser.parse_args()
    output, asset_root = args.output.resolve(), args.asset_root.resolve()
    if args.command == "register":
        register(output, args.runtime_commit or "")
    elif args.command == "a0":
        a0(output)
    elif args.command == "a1":
        a1(output, asset_root, args.asset, _device(args.device))
    else:
        a2(output, asset_root, args.arm, _device(args.device), args.physical_gpu_id)


if __name__ == "__main__":
    main()
