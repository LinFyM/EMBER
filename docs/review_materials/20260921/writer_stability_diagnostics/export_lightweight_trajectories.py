#!/usr/bin/env python3
"""Export the registered 16 existing E1 trajectories without model execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch


ASSETS = ("O1200", "N1800")
TASKS = (("libero_spatial", 3), ("libero_10", 1))
REPLAN_STEPS = 5


def _camera(observation: dict, key: str) -> np.ndarray:
    value = observation[key][0].permute(1, 2, 0).numpy()
    value = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(value).resize((128, 128), Image.Resampling.BILINEAR))


def _video(path: Path, observations: tuple[dict, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=6, codec="libx264", quality=6, macro_block_size=None) as writer:
        for observation in observations:
            base = _camera(observation, "observation.images.base_0_rgb")
            wrist = _camera(observation, "observation.images.left_wrist_0_rgb")
            writer.append_data(np.concatenate((base, wrist), axis=1))


def export(source: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    action_rows, manifest = [], []
    for asset in ASSETS:
        for suite, task_id in TASKS:
            for state_id in range(4):
                stem = f"{asset}_{suite}_task_{task_id:02d}_state_{state_id:03d}"
                source_path = (
                    source / asset / "trajectories"
                    / f"{suite}_task_{task_id:02d}_state_{state_id:03d}.pt"
                )
                payload = torch.load(source_path, map_location="cpu", weights_only=False)
                observations = tuple(payload["observations"])
                chunks = tuple(payload["action_chunks"])
                if len(observations) != len(chunks):
                    raise ValueError(f"trajectory replan rows differ: {source_path}")
                video_path = output / "videos" / f"{stem}.mp4"
                _video(video_path, observations)
                emitted = 0
                for replan_index, chunk in enumerate(chunks):
                    remaining = int(payload["steps"]) - replan_index * REPLAN_STEPS
                    executed = max(0, min(REPLAN_STEPS, remaining))
                    actions = chunk[0, :executed].to(torch.float32).numpy()
                    for offset, action in enumerate(actions):
                        action_rows.append({
                            "asset": asset,
                            "suite": suite,
                            "task_id": task_id,
                            "init_state_id": state_id,
                            "control_step": replan_index * REPLAN_STEPS + offset,
                            **{f"normalized_action_{index}": float(value) for index, value in enumerate(action)},
                        })
                        emitted += 1
                if emitted != int(payload["steps"]):
                    raise ValueError(f"executed action count differs: {source_path}")
                manifest.append({
                    "asset": asset,
                    "suite": suite,
                    "task_id": task_id,
                    "init_state_id": state_id,
                    "success": bool(payload["success"]),
                    "control_steps": int(payload["steps"]),
                    "frame_control_step_stride": REPLAN_STEPS,
                    "video": str(video_path.relative_to(output)),
                    "video_bytes": video_path.stat().st_size,
                    "video_frames": len(observations),
                })

    for name, rows in (("manifest.csv", manifest), ("executed_normalized_actions.csv", action_rows)):
        path = output / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "contract.json").write_text(json.dumps({
        "schema_version": "ember_writer_e1_lightweight_trajectories_v1",
        "source_execution": "existing_E1_only",
        "model_forward_or_rollout": False,
        "assets": list(ASSETS),
        "tasks": [{"suite": suite, "task_id": task} for suite, task in TASKS],
        "init_state_ids": list(range(4)),
        "conditions": len(manifest),
        "frame_control_step_stride": REPLAN_STEPS,
        "video_layout": "base_camera_left_wrist_camera_side_by_side_128px_each",
        "action_scope": "actually_executed_prefix_from_normalized_50x7_policy_chunks",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
