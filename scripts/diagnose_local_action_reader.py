"""Bounded frozen local-reader audit; historical runtime owns all model/data code.

This entrypoint is retired after the registered audit, with its frozen commit
and outputs retained. It is not a second Writer or a deployment action path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(root, arm, step):
    values = [json.loads(line) for line in (root / arm / "local_diagnostics.jsonl").read_text().splitlines()]
    return {(r["task"], r["clip"]): r for r in values if r["step"] == step}


def task_moments(handle, normalize):
    means, seconds, count = [], [], 0
    for demo in range(16, 42):
        actions = normalize(torch.from_numpy(handle[f"data/demo_{demo}/actions"][:])).numpy()
        # p is uniform in [0, N-16]; post-action RGB[p:p+16:5] maps to a[p+1:p+16].
        windows = np.stack([actions[p + 1:p + 16].reshape(-1) for p in range(len(actions) - 15)]).astype(np.float64)
        means.append(windows.mean(0))
        seconds.append(windows.T @ windows / len(windows))
        count += len(windows)
    mean = np.mean(means, axis=0)
    covariance = np.mean(seconds, axis=0) - np.outer(mean, mean)
    eigenvalues, basis = np.linalg.eigh(covariance)
    return mean, eigenvalues.clip(min=0), basis, count


def moments(args, config, output):
    from ember.pi05_processing import Pi05LiberoProcessor
    from ember.writer.function_credit import local_flow_sample
    from ember.writer.learning_data import load_learning_tasks

    root = args.asset_root
    stats = read(root / "configs/pi05_source_corpus_v1/source_normalization.json")["stats"]
    processor = Pi05LiberoProcessor(stats, root / "models/tokenizers/openpi/paligemma_tokenizer.model", 200, "cpu")
    tasks = load_learning_tasks(root, config["data"]["task_ids"])
    original = rows(args.runs, "ordered", 200)
    initial = rows(args.runs, "ordered", 0)
    reference = rows(args.runs, "frame_set", 200)
    assert set(original) == set(initial) == set(reference) == {(t, c) for t in tasks for c in range(16)}
    arrays, result = {}, []
    for task, authority in tasks.items():
        with h5py.File(authority.authority.path, "r") as handle:
            mean, eigenvalues, basis, count = task_moments(handle, processor.normalize_action)
            arrays[str(task)] = mean.reshape(15, 7)
            for clip in range(16):
                row = original[task, clip]
                for field in ("local_action_demo", "local_start_frame", "local_flow_seed", "local_action_start", "local_action_stop"):
                    assert row[field] == initial[task, clip][field] == reference[task, clip][field]
                assert row["local_action_demo"] == 42 + clip // 4 and row["gradients"] is False
                raw = handle[f'data/demo_{row["local_action_demo"]}/actions'][row["local_action_start"]:row["local_action_stop"]]
                action = processor.normalize_action(torch.from_numpy(raw))
                noisy, times, target = local_flow_sample(action, seed=row["local_flow_seed"], draws=8)
                zero_loss = float(target.square().mean())
                assert abs(zero_loss - initial[task, clip]["local_flow_loss"]) < 1e-5
                t = times.double().numpy()[:, None]
                centered = noisy.reshape(8, 105).double().numpy() - (1 - t) * mean
                coefficient = (t - (1 - t) * eigenvalues) / (t * t + (1 - t) ** 2 * eigenvalues)
                velocity = -mean + ((centered @ basis) * coefficient) @ basis.T
                result.append({"task": task, "suite": authority.suite, "clip": clip,
                    "training_windows": count, "gaussian_fm": float(np.mean((velocity - target.reshape(8, 105).numpy()) ** 2)),
                    "ordered_fm": row["local_flow_loss"], "frame_set_fm": reference[task, clip]["local_flow_loss"],
                    "task_mean_action_mse": float(np.mean((mean - action.reshape(-1).numpy()) ** 2)),
                    "initial_zero_fm": zero_loss})
    np.savez(output / "inverse_action_task_means.npz", **arrays)
    write(output / "inverse_action_marginal.json", {"scope": "train24 only; moments from action16:42; scores42:46",
        "deployment_baseline": False, "gradient_updates": 0, "rows": result})
    print(json.dumps({field: float(np.mean([r[field] for r in result])) for field in
        ("gaussian_fm", "ordered_fm", "frame_set_fm", "task_mean_action_mse", "initial_zero_fm")}), flush=True)


@torch.no_grad()
def decode_clip(runtime, frames, indices, language, seed):
    from ember.writer.attention import position_encoding
    from ember.writer.native import autocast

    condition = runtime.observer.prepare(frames, indices, language)
    responses, inputs = runtime.observer.read(condition)
    device = runtime.observer.device
    with autocast(device):
        video = runtime.state.writer.encode(responses, *inputs)[0]
        memory = video.flatten(0, 1)
        routing = torch.zeros_like(memory)
        if runtime.state.writer.config.process_mode == "ordered":
            positions = torch.arange(4, device=device)
            routing = position_encoding(positions, video.shape[-1], video.dtype)
            routing = routing[:, None].expand_as(video).flatten(0, 1)
        prior = memory.new_full((1, len(memory)), -math.log(len(memory)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        samples = torch.randn(8, 15, 7, generator=generator).to(device)
        for step in range(10):
            flow_time = torch.full((8,), 1 - step / 10, device=device)
            velocity = runtime.state.reader(samples, flow_time, memory, routing, prior)
            samples = samples - .1 * velocity.float()
        if not torch.isfinite(samples).all():
            raise ValueError("frozen local decoder returned nonfinite actions")
    return samples.cpu().numpy()


def decode(args, config, output):
    from safetensors.torch import load_file
    from ember.writer.learning_data import WriterTrainingData
    from ember.writer.runtime import build_runtime

    target = output / f"inverse_action_{args.arm}.json"
    if target.exists():
        raise ValueError("audit output exists; do not overwrite or blindly rerun")
    checkpoint = args.runs / args.arm / "checkpoints/macro_00000200"
    assert read(checkpoint / "checkpoint_manifest.json")["next_macro"] == 200
    runtime = build_runtime(args.asset_root, config, torch.device("cuda:0"))
    runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors")), strict=True)
    runtime.state.requires_grad_(False).eval()
    runtime.policy.requires_grad_(False).eval()
    data = WriterTrainingData(args.asset_root, config["data"], camera_view=config["observer"].get("camera_view", "agentview"))
    original = rows(args.runs, args.arm, 200)
    result, predictions, truths = [], [], []
    started = time.monotonic()
    with np.load(output / "inverse_action_task_means.npz") as means:
        for task, authority in data.tasks.items():
            for clip in range(16):
                frames, indices, raw, trace = data.local_action_clip(task, clip, diagnostic=True)
                assert all(value == original[task, clip][field] for field, value in trace.items())
                # Targets are normalized/scored only after sampling, never passed to decode_clip.
                samples = decode_clip(runtime, frames, indices, authority.authority.language, trace["local_flow_seed"])
                truth = runtime.processor.normalize_action(raw).cpu().numpy()
                estimate = samples.mean(0)
                result.append({"task": task, "suite": authority.suite, "clip": clip, **trace,
                    "mean_action_mse": float(np.mean((estimate - truth) ** 2)),
                    "sample_action_mse": float(np.mean((samples - truth[None]) ** 2)),
                    "task_mean_action_mse": float(np.mean((means[str(task)] - truth) ** 2)),
                    "xyz_mean_action_mse": float(np.mean((estimate[:, :3] - truth[:, :3]) ** 2)),
                    "gripper_mean_action_mse": float(np.mean((estimate[:, 6] - truth[:, 6]) ** 2))})
                predictions.append(samples)
                truths.append(truth)
            print(json.dumps({"arm": args.arm, "task": task, "completed_clips": len(result),
                "elapsed_seconds": time.monotonic() - started}), flush=True)
    np.savez(output / f"inverse_action_{args.arm}_samples.npz", samples=np.stack(predictions), truth=np.stack(truths))
    write(target, {"status": "complete", "checkpoint": str(checkpoint), "runtime_commit": args.runtime_commit,
        "gradient_updates": 0, "flow_steps": 10, "noise_draws": 8, "rows": result,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("moments", "decode"), required=True)
    parser.add_argument("--arm", choices=("ordered", "frame_set"), default="ordered")
    args = parser.parse_args()
    args.runtime_commit = subprocess.check_output(["git", "-C", str(args.runtime_root), "rev-parse", "HEAD"], text=True).strip()
    if args.runtime_commit != "5f4f440c992e470aaffcc873c847373ce6df43e5":
        raise ValueError("local-action frozen runtime identity changed")
    if subprocess.check_output(["git", "-C", str(args.runtime_root), "status", "--porcelain"], text=True).strip():
        raise ValueError("historical model runtime is dirty")
    sys.path.insert(0, str(args.runtime_root / "src"))
    torch.set_num_threads(4)
    args.runs = args.asset_root / "runs/outputs/local_action_grounded_20260912"
    output = args.asset_root / "runs/analysis/local_action_grounded_20260912"
    config = read(args.runs / args.arm / "run_contract.json")["config"]
    (moments if args.phase == "moments" else decode)(args, config, output)


if __name__ == "__main__":
    main()
