"""Finite frozen Gamma Value materialization; existing native caches only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open

from ember.operator_writer.control_calibration import ControlCalibration
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import source_reference_matches

STUDY = "calibrated_action_memory_control_20261007"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
PRIOR = ROOT.parent / "privileged_action_memory_control_20261007"
CALIBRATION = ROOT.parent / "control_calibrated_read_write_20261003"
CHECKPOINT = CALIBRATION / "control_calibrated_read_write/train/attempts/fresh/checkpoints/macro_00000450/ecp.safetensors"
ARMS = ("bare_endpoint", "calibrated_value")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def load_bare(row, source):
    path = CALIBRATION / "bare_native" / f"task{row['global_task_id']:03d}_demo{row['teacher_demo']:02d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    bare = {key: saved[key] for key in ("schema", "H0", "mu0", "frame_indices", "source")}
    if (bare["schema"] != "ember_bare_native_control_features_v1"
            or not source_reference_matches(bare["source"], source)
            or bare["H0"].shape != (len(bare["frame_indices"]), 50, 1024)
            or bare["mu0"].shape != (len(bare["frame_indices"]), 5, 7)
            or not torch.isfinite(bare["H0"]).all() or not torch.isfinite(bare["mu0"]).all()
            or bare["H0"].requires_grad or bare["mu0"].requires_grad):
        raise ValueError(f"Existing bare native schema/source/full50 contract changed: {path}")
    indices = bare["frame_indices"].tolist()
    if len(indices) != len(set(indices)):
        raise ValueError("Existing bare cache repeats a frame")
    mapping = {frame: position for position, frame in enumerate(indices)}
    if any(frame not in mapping or frame + 5 not in mapping for frame in row["frames"]):
        raise ValueError(f"Legal departure/arrival missing: {path}")
    dep = torch.tensor([mapping[frame] for frame in row["frames"]])
    arr = torch.tensor([mapping[frame + 5] for frame in row["frames"]])
    return path, bare, dep, arr


def metadata(registration, arm, conditions, git):
    return {**registration, "schema_version": "ember_calibrated_action_memory_manifest_v1",
            "study_id": STUDY, "root": str(ROOT), "arm": arm, "conditions": conditions,
            "kind": "privileged_action_memory_controller", "source_inference_steps": 0,
            "source_value_mode": "existing_statefree_tau1_endpoint",
            "source_state_used": False, "source_flow_tau": 1,
            "source_noise_seed": 1729, "source_noise_shape": [50, 32],
            "source_native_forwards": 0, "rollout_flow_steps": 0,
            "value_shape": [5, 7], "valid_length": 5,
            "value_inverse_quantile_count": 1, "new_geometry_restores": 0,
            "geometry_registration": str(PRIOR / "memory/registration.json"),
            "gamma": {"checkpoint": str(CHECKPOINT), "training_git": "2c630fb3",
                      "input": "existing_H0_mu0_only", "reader_git": git,
                      "parameter_dtype": "float32", "autocast_dtype": "bfloat16",
                      "native_positions": 50, "heads": 4, "rms_eps": 1e-6},
            "reading_git": git}


def save_arm(registration, arm, values, condition_rows, git):
    stats = json.loads(Path(registration["normalization"]).read_text())["stats"]["action"]
    low, high = torch.tensor(stats["q01"], dtype=torch.float32), torch.tensor(stats["q99"], dtype=torch.float32)
    conditions, cursor = [], 0
    for row in condition_rows:
        count = len(row["frames"])
        normalized = values[cursor:cursor + count].float().cpu()
        raw = Pi05LiberoProcessor._quantile_transform(normalized, low, high, inverse=True)
        if normalized.shape != (count, 5, 7) or not torch.isfinite(raw).all():
            raise ValueError("Actual Value shape/finite contract changed")
        path = ROOT / "memory" / arm / f"task{row['global_task_id']:03d}_demo{row['teacher_demo']:02d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, frames=np.asarray(row["frames"], dtype=np.int64),
                                normalized_actions5=normalized.numpy(), raw_actions5=raw.numpy())
        conditions.append({**row, "value_path": str(path)})
        cursor += count
    if cursor != len(values):
        raise ValueError("Materialization changed the logical transition stream")
    write_json(ROOT / "memory" / f"{arm}_manifest.json", metadata(registration, arm, conditions, git))
    write_json(ROOT / "launch" / f"{arm}_ready.json", {"arm": arm, "conditions": len(conditions), "transitions": cursor})


def run(git):
    started = time.time()
    torch.set_grad_enabled(False)
    registration = json.loads((PRIOR / "memory/registration.json").read_text())
    condition_rows, departures, arrivals, endpoints = [], [], [], []
    for row in registration["conditions"]:
        path, bare, dep, arr = load_bare(row, registration["source"])
        condition_rows.append({**row, "bare_path": str(path),
                               "bare_source": bare["source"], "bare_schema": bare["schema"],
                               "departure_cache_positions": dep.tolist(), "arrival_cache_positions": arr.tolist()})
        departures.append(bare["H0"][dep])
        arrivals.append(bare["H0"][arr])
        endpoints.append(bare["mu0"][dep])
    h0, h1, mu = map(torch.cat, (departures, arrivals, endpoints))
    if len(condition_rows) != 144 or len(mu) != 4556:
        raise ValueError("Fixed144/4556 existing memory coverage changed")
    save_arm(registration, "bare_endpoint", mu, condition_rows, git)
    model = ControlCalibration()
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as saved:
        weights = {key.removeprefix("gamma."): saved.get_tensor(key)
                   for key in saved.keys() if key.startswith("gamma.")}
    if len(weights) != 8 or any(weight.dtype != torch.float32 for weight in weights.values()):
        raise ValueError("Frozen Gamma tensor count/dtype changed")
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False).to("cuda")
    output = torch.empty_like(mu)
    profiles = []
    first_count = min(2048, len(mu))

    def calculate(begin, count, batch, preserve):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        clock = time.perf_counter()
        for offset in range(begin, begin + count, batch):
            stop = min(offset + batch, begin + count)
            dep = h0[offset:stop].to("cuda")
            arr = h1[offset:stop].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual = model(dep, arr)[:, :5].float()
            calibrated = mu[offset:stop].to("cuda") + residual
            if not torch.isfinite(calibrated).all():
                raise ValueError("Frozen Gamma produced nonfinite Values")
            if preserve:
                output[offset:stop] = calibrated.cpu()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - clock
        return {"batch": batch, "transitions": count, "seconds": elapsed,
                "transitions_per_second": count / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    # Both profiles use the same first fixed block; keep its first valid result.
    profiles.append(calculate(0, first_count, 1024, True))
    profiles.append(calculate(0, first_count, 2048, False))
    selected = max(profiles, key=lambda row: row["transitions_per_second"])["batch"]
    remainder = calculate(first_count, len(mu) - first_count, selected, True)
    save_arm(registration, "calibrated_value", output, condition_rows, git)
    write_json(ROOT / "memory/materialization.json", {
        "elapsed_seconds": time.time() - started, "conditions": 144, "transitions": len(mu),
        "profiles": profiles, "selected_transition_batch": selected, "remainder": remainder,
        "stop_amplifying": "Two profiles are the authorized maximum; fixed4556 transitions only",
        "first_profile_values_preserved": True, "source_native_forwards": 0,
        "geometry_restores": 0, "gradient_updates": 0, "teacher_HDF_reads": 0,
        "gamma_tensor_count": len(weights), "gamma_training_git": "2c630fb3", "reader_git": git,
        "information_wall": "Gamma received existing full50 H0 pairs only; mu0 added afterwards"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader-git", required=True)
    args = parser.parse_args()
    run(args.reader_git)


if __name__ == "__main__":
    main()
