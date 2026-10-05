"""Bounded, retired-after-study hand-axis heads and array-only readback.

Labels and task/video metadata never enter Head. This module opens no dataset,
native runtime, policy, HDF file or physical environment.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDY = "native_hand_axis_readout_20261006"
HELDOUT_TASKS = (29, 34, 38, 73)
GROUPS = ("all", "tilt_gt30", "pre_positive")
MEANS = ("angle_mean_deg", "tilt_ae_mean_deg", "vector_mse", "adjacent_change_mse",
         "true_change_mean", "pred_change_mean", "true_change_rms", "pred_change_rms",
         "true_angular_change_mean_deg", "pred_angular_change_mean_deg", "centered_mse")


class Head(nn.Module):
    """One frame's entire token collection -> world hand +Z unit vector."""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        if self.input_dim not in (1024, 512):
            raise ValueError("this study has only hbar1024 and visualKV512")
        # Construct shared shapes first. The different projection widths must
        # not consume a different number of RNG draws before their initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7)
            self.input_norm = nn.LayerNorm(128)
            self.query = nn.Parameter(torch.randn(1, 1, 128))
            self.attention = nn.MultiheadAttention(128, 4, dropout=0., batch_first=True)
            self.read_norm = nn.LayerNorm(128)
            self.ffn_norm = nn.LayerNorm(128)
            self.ffn = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
            self.output = nn.Linear(128, 3)
            nn.init.zeros_(self.output.weight)
            with torch.no_grad():
                self.output.bias.copy_(torch.tensor([0., 0., -1.]))
            self.input_proj = nn.Linear(self.input_dim, 128)

    def forward(self, values):
        if values.ndim != 3 or values.shape[-1] != self.input_dim or values.shape[1] < 1:
            raise ValueError("Head requires a nonempty B,N,input_dim token collection")
        tokens = self.input_norm(self.input_proj(values.detach()))
        query = self.query.expand(len(tokens), -1, -1)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        read = self.read_norm(query + attended).squeeze(1)
        read = read + self.ffn(self.ffn_norm(read))
        return F.normalize(self.output(read).float(), dim=-1, eps=1e-6)


def make_heads():
    return {"H": Head(1024), "KV": Head(512)}


def axis_labels(ee_ori):
    """Exp([r]_x) e_z; sinc forms give the continuous zero-angle limit."""
    r = np.asarray(ee_ori, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3 or not np.isfinite(r).all():
        raise ValueError("ee_ori must be finite N,3 axis-angle arrays")
    theta = np.linalg.norm(r, axis=1)
    first = np.stack((r[:, 1], -r[:, 0], np.zeros(len(r))), axis=1)
    second = np.stack((r[:, 0] * r[:, 2], r[:, 1] * r[:, 2],
                       -r[:, 0] ** 2 - r[:, 1] ** 2), axis=1)
    z = np.array([0., 0., 1.]) + np.sinc(theta / np.pi)[:, None] * first
    z += (.5 * np.sinc(theta / (2 * np.pi)) ** 2)[:, None] * second
    return (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)


def _unit_array(value, n, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite {n},3")
    if not np.allclose(np.linalg.norm(value, axis=1), 1., atol=1e-4, rtol=0.):
        raise ValueError(f"{name}: directions must be unit vectors")
    return value


def _unit_mean(value):
    norm = float(np.linalg.norm(value))
    return (value / norm).tolist() if norm >= 1e-6 else None


def _angle(a, b):
    dot = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.rad2deg(np.arccos(np.clip(dot, -1., 1.)))


def _tilt(z):
    return np.rad2deg(np.arccos(np.clip(-z[:, 2] / np.linalg.norm(z, axis=1), -1., 1.)))


def constant_baselines(manifest, labels):
    """All constant estimates use only fit clips, equal video then task weight."""
    means = defaultdict(list)
    for clip in manifest["clips"]:
        if clip["role"] == "fit":
            means[int(clip["task"])].append(np.asarray(labels[clip["key"]]["z"]).mean(0))
    task_means = {task: np.mean(values, axis=0) for task, values in means.items()}
    global_axis = _unit_mean(np.mean(list(task_means.values()), axis=0))
    task_axes = {task: _unit_mean(value) for task, value in task_means.items()}
    if set(task_axes) != set(map(int, manifest["fit_tasks"])) or set(task_axes) & set(HELDOUT_TASKS):
        raise ValueError("task constants must come from all and only fit32")
    predictions = {"vertical": {}, "fit_global": {}, "fit_task": {}}
    for clip in manifest["clips"]:
        key, task = clip["key"], int(clip["task"])
        n = len(clip["raw_indices"])
        predictions["vertical"][key] = np.tile([0., 0., -1.], (n, 1))
        if global_axis is not None:
            predictions["fit_global"][key] = np.tile(global_axis, (n, 1))
        if clip["role"] == "eval" and task in task_axes and task_axes[task] is not None:
            predictions["fit_task"][key] = np.tile(task_axes[task], (n, 1))
    return predictions, {"vertical": [0., 0., -1.], "fit_global": global_axis,
        "fit_task_axes": task_axes, "fit_clip_keys": [c["key"] for c in manifest["clips"] if c["role"] == "fit"],
        "undefined_mean_axis": "norm<1e-6 has no defined unit direction; unavailable rows retain null metrics"}


def _clip_stats(z, p, mask):
    n = int(mask.sum())
    edges = mask[:-1] & mask[1:]
    row = {"nframes": n, "nadjacent": int(edges.sum()), "nclips_total": 1,
           "nclips_nonempty": int(n > 0 and p is not None), **{name: None for name in MEANS},
           "angle_median_deg": None, "tilt_ae_median_deg": None, "_angle": [], "_tilt": []}
    if not n or p is None:
        return row
    angle, tilt_ae = _angle(p[mask], z[mask]), np.abs(_tilt(p[mask]) - _tilt(z[mask]))
    row.update(angle_mean_deg=float(angle.mean()), tilt_ae_mean_deg=float(tilt_ae.mean()),
        vector_mse=float(np.square(p[mask] - z[mask]).sum(-1).mean()),
        centered_mse=float(np.square((p[mask] - p[mask].mean(0)) - (z[mask] - z[mask].mean(0))).sum(-1).mean()),
        angle_median_deg=float(np.median(angle)), tilt_ae_median_deg=float(np.median(tilt_ae)),
        _angle=[(float(x), 1 / n) for x in angle], _tilt=[(float(x), 1 / n) for x in tilt_ae])
    if edges.any():
        dz, dp = np.diff(z, axis=0)[edges], np.diff(p, axis=0)[edges]
        row.update(adjacent_change_mse=float(np.square(dp - dz).sum(-1).mean()),
            true_change_mean=float(np.linalg.norm(dz, axis=1).mean()), pred_change_mean=float(np.linalg.norm(dp, axis=1).mean()),
            true_change_rms=float(np.sqrt(np.square(dz).sum(-1).mean())),
            pred_change_rms=float(np.sqrt(np.square(dp).sum(-1).mean())),
            true_angular_change_mean_deg=float(_angle(z[:-1][edges], z[1:][edges]).mean()),
            pred_angular_change_mean_deg=float(_angle(p[:-1][edges], p[1:][edges]).mean()))
    return row


def _distribution(rows, field):
    present = [row[field] for row in rows if row[field]]
    return [(value, weight / len(present)) for values in present for value, weight in values]


def _median(distribution):
    if not distribution:
        return None
    ordered = sorted(distribution)
    cumulative = np.cumsum([weight for _, weight in ordered])
    index = int(np.searchsorted(cumulative, .5 - 1e-12))
    if index + 1 < len(ordered) and abs(cumulative[index] - .5) < 1e-12:
        return (ordered[index][0] + ordered[index + 1][0]) / 2
    return ordered[index][0]


def _reduce(rows, unit):
    result = {name: sum(row[name] for row in rows) for name in ("nframes", "nadjacent", "nclips_total", "nclips_nonempty")}
    result["weight_unit"] = unit
    result["metric_units"] = {}
    for name in MEANS:
        values = [row[name] for row in rows if row[name] is not None]
        result[name] = float(np.mean(values)) if values else None
        result["metric_units"][name] = len(values)
    for field, name in (("_angle", "angle_median_deg"), ("_tilt", "tilt_ae_median_deg")):
        result[field] = _distribution(rows, field)
        result[name] = _median(result[field])
    return result


def _public(row):
    return {name: value for name, value in row.items() if not name.startswith("_")}


def _validate(manifest, labels, predictions):
    fit, held = set(map(int, manifest["fit_tasks"])), set(map(int, manifest["heldout_tasks"]))
    if len(fit) != 32 or held != set(HELDOUT_TASKS) or fit & held:
        raise ValueError("fixed split is fit32 and head-heldout29/34/38/73")
    clips = manifest["clips"]
    keys = {clip["key"] for clip in clips}
    if len(clips) != 136 or len(keys) != 136 or set(labels) != keys or set(predictions) != {"H", "KV"}:
        raise ValueError("all136 fixed clips and both H/KV predictions are required")
    actual = {(int(c["task"]), int(c["demo"]), c["role"]) for c in clips}
    wanted = {(task, demo, "fit") for task in fit for demo in (16, 17)}
    wanted |= {(task, demo, "eval") for task in fit | held for demo in (42, 43)}
    if actual != wanted:
        raise ValueError("clip task/demo/role differs from the fixed contract")
    for arm in predictions:
        if set(predictions[arm]) != keys:
            raise ValueError(f"{arm} predictions do not cover the entire endpoint")
    for clip in clips:
        key, indices = clip["key"], clip["raw_indices"]
        n = len(indices)
        if (key != f"t{int(clip['task']):03d}_d{int(clip['demo']):02d}" or not n
                or indices[0] != 0 or any(b <= a for a, b in zip(indices, indices[1:]))
                or any(b - a != 5 for a, b in zip(indices[:-2], indices[1:-1]))
                or (n > 1 and not 1 <= indices[-1] - indices[-2] <= 5)
                or clip.get("nframes", n) != n):
            raise ValueError(f"{key}: invalid registered frame grid/key")
        label = labels[key]
        _unit_array(label["z"], n, key + "/z")
        pre = np.asarray(label["pre_positive"])
        if pre.shape != (n,) or pre.dtype.kind != "b":
            raise ValueError(f"{key}: pre_positive must be boolean N")
        for arm in predictions:
            _unit_array(predictions[arm][key], n, key + "/" + arm)


def metrics(manifest, labels, predictions):
    """Equal clip -> equal task metrics; no feature or dataset access."""
    _validate(manifest, labels, predictions)
    baselines, baseline_info = constant_baselines(manifest, labels)
    all_predictions = {**predictions, **baselines}
    clip_rows, task_buckets = [], defaultdict(list)
    frame_records, key_task_records = [], []
    for clip in manifest["clips"]:
        key, task = clip["key"], int(clip["task"])
        partition = "fit" if clip["role"] == "fit" else ("eval_holdout" if task in HELDOUT_TASKS else "eval_seen")
        z = np.asarray(labels[key]["z"], dtype=np.float64)
        masks = {"all": np.ones(len(z), dtype=bool), "tilt_gt30": _tilt(z) > 30.,
                 "pre_positive": np.asarray(labels[key]["pre_positive"])}
        frames = {**clip, "source": labels[key].get("source"), "z": z.tolist(),
                  "pre_positive": masks["pre_positive"].tolist(), "predictions": {}}
        for arm, arm_predictions in all_predictions.items():
            if arm == "fit_task" and partition != "eval_seen":
                continue  # Never construct or score a heldout-task label baseline.
            p = np.asarray(arm_predictions[key], dtype=np.float64) if key in arm_predictions else None
            frames["predictions"][arm] = p.tolist() if p is not None else None
            for group, mask in masks.items():
                row = {"arm": arm, "partition": partition, "group": group, "key": key,
                    "task": task, "demo": int(clip["demo"]), "source": labels[key].get("source"),
                    "raw_indices": clip["raw_indices"], **_clip_stats(z, p, mask)}
                clip_rows.append(row)
                task_buckets[(arm, partition, group, task)].append(row)
        frame_records.append(frames)
        if task in HELDOUT_TASKS:
            key_task_records.append(frames)
    task_rows, aggregate_buckets = [], defaultdict(list)
    for (arm, partition, group, task), rows in sorted(task_buckets.items()):
        row = {"arm": arm, "partition": partition, "group": group, "task": task, **_reduce(rows, "equal_clips")}
        task_rows.append(row)
        aggregate_buckets[(arm, partition, group)].append(row)
    aggregate = [{"arm": arm, "partition": partition, "group": group, "ntasks_total": len(rows),
        "ntasks_nonempty": sum(row["nclips_nonempty"] > 0 for row in rows), **_reduce(rows, "equal_tasks_after_equal_clips")}
        for (arm, partition, group), rows in sorted(aggregate_buckets.items())]
    scopes = {name: {"tasks": list(HELDOUT_TASKS) if name == "eval_holdout" else sorted(map(int, manifest["fit_tasks"])),
        "parent_policy_has_seen_tasks": True, "arms": {}} for name in ("fit", "eval_seen", "eval_holdout")}
    for row in aggregate:
        scopes[row["partition"]]["arms"].setdefault(row["arm"], {})[row["group"]] = _public(row)
    paired = defaultdict(dict)
    for row in clip_rows:
        paired[(row["key"], row["group"])][row["arm"]] = row
    comparisons = []
    for (key, group), rows in sorted(paired.items()):
        difference = None
        if rows["H"]["angle_mean_deg"] is not None:
            difference = rows["H"]["angle_mean_deg"] - rows["KV"]["angle_mean_deg"]
        comparisons.append({"key": key, "task": rows["H"]["task"], "group": group,
            "H_minus_KV_angle_mean_deg": difference, "all_arms": {arm: _public(row) for arm, row in rows.items()}})
    return {"schema_version": "ember_native_hand_axis_readback_v1", "study": STUDY,
        "scope": "136 original train36 clips; heldout4 is held out only from head fitting; the frozen parent policy has seen all36 tasks",
        "definitions": {"weights": "nonempty clip distributions equally weighted within task, then nonempty tasks equally weighted; empty units retain null/count0 and coverage counts",
            "median": "weighted median of frame error mixture using the same clip/task weights, not median of clip medians",
            "vector_mse": "mean(sum_3((prediction-label)**2)); matches mean(||zhat-z||²)",
            "adjacent": "original neighboring sampled frames including terminal frame; both endpoints must belong to the group; no gaps bridged",
            "change": "mean and RMS of adjacent vector difference norms; angular changes also reported in degrees",
            "centered_mse": "each clip/group prediction and label are separately centered before mean squared vector norm",
            "pre_positive": "caller-supplied obs index before first positive gripper command; not a grasp/contact label",
            "meaningful_reference": "5 degrees absolute / 20% relative are magnitude references, not significance gates",
            "boundary": "finite readout-class diagnosis only; no claim of causal information loss, root cause, repaired control, policy-heldout generalization or new EMBER score"},
        "fit_tasks": sorted(map(int, manifest["fit_tasks"])), "heldout_tasks": list(HELDOUT_TASKS),
        "scopes": scopes, "arms": {"H": "learned hbar head", "KV": "learned final visualKV head",
            "vertical": "constant world downward axis", "fit_global": "fit-only equal-task/equal-video global unit axis",
            "fit_task": "fit-only task unit axis, scoring eval_seen only"},
        "baselines": baseline_info, "per_clip": list(map(_public, clip_rows)), "per_task": list(map(_public, task_rows)),
        "aggregate": list(map(_public, aggregate)), "frame_records": frame_records,
        "adverse_examples": {"all_clip_group_comparisons": comparisons, "all_key_tasks_full_frames": key_task_records,
            "selection": "all clips/groups, both signs and empty groups retained; all29/34/38/73 videos retained without selecting favorable frames"}}


def _write_tsv(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def analyze(manifest, labels, predictions, output_path):
    """Write complete arrays/provenance, tables and bounded descriptive report."""
    report = metrics(manifest, labels, predictions)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("per_clip", "per_task", "aggregate"):
        _write_tsv(output / (name + ".tsv"), report[name])
    for name, value in (("metrics.json", {key: value for key, value in report.items() if key != "frame_records"}),
                        ("adverse_examples.json", report["adverse_examples"])):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = ["# Frozen native hand-axis readout", "", report["scope"], "", report["definitions"]["boundary"], "",
        "Metrics use equal clip then equal task weights. Medians use the weighted frame mixture; MSE sums three vector coordinates.",
        "Change metrics compare original adjacent sampled observations; groups never bridge missing frames. Empty groups are retained.", "",
        "|Partition|Arm|Tasks with frames|Angle mean/median °|Tilt AE °|Vector MSE|Adjacent-change MSE|Centered MSE|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "null" if value is None else f"{value:.5g}"
    for row in report["aggregate"]:
        if row["group"] == "all":
            lines.append(f"|{row['partition']}|{row['arm']}|{row['ntasks_nonempty']}/{row['ntasks_total']}|"
                f"{fmt(row['angle_mean_deg'])}/{fmt(row['angle_median_deg'])}|{fmt(row['tilt_ae_mean_deg'])}|"
                f"{fmt(row['vector_mse'])}|{fmt(row['adjacent_change_mse'])}|{fmt(row['centered_mse'])}|")
    empty = [row for row in report["per_clip"] if not row["nframes"]]
    lines += ["", f"Empty clip/arm/group rows: {len(empty)}. Complete subgroup metrics and every adverse comparison are in TSV/JSON.",
        "All29/34/38/73 frame labels/predictions and all clips with either sign of H−KV are retained.",
        "Vertical and global-axis baselines use only fit data; per-task constants are scored only on eval32, never heldout4.",
        "A zero fit mean has no unit direction and remains unavailable; no heldout label supplies a fallback.", "",
        "The 5°/20% references describe effect magnitude only. Decoding does not establish repaired control or a root cause."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
