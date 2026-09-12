"""One bounded train-side source-input audit; no Writer or parameter updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(root):
    from ember.writer.learning_data import load_learning_tasks

    original = root / "runs/outputs/local_action_grounded_20260912/ordered/local_diagnostics.jsonl"
    rows = [json.loads(line) for line in original.read_text().splitlines()]
    rows = sorted((r for r in rows if r["step"] == 200), key=lambda r: (r["task"], r["clip"]))
    tasks = load_learning_tasks(root, sorted({r["task"] for r in rows}))
    assert len(tasks) == 24 and len(rows) == 384
    assert {(r["task"], r["clip"]) for r in rows} == {(t, c) for t in tasks for c in range(16)}
    for row in rows:
        assert row["local_action_demo"] == 42 + row["clip"] // 4
        assert row["local_action_start"] == row["local_start_frame"] + 1
        assert row["local_action_stop"] == row["local_start_frame"] + 16
        assert row["gradients"] is False
    return tasks, rows


def source_model(root, device):
    from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
    from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
    from ember.pi05_source_setup import load_policy

    reuse = read(root / "configs/pi05_writer_data_v1.json")["authorities"]
    authorities = load_evaluation_authorities(root / "configs/pi05_target_evaluation_v1.json", root)
    checkpoint = root / reuse["source_checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
    policy = load_policy(Path(source["model_path"]), read(root / reuse["source_base_config"]), device)
    policy.eval().requires_grad_(False)
    policy.model.gradient_checkpointing_disable()
    assert policy.config.chunk_size == 50 and policy.config.max_action_dim == 32
    tokenizer = root / reuse["tokenizer"]
    stats = read(root / reuse["source_normalization"])["stats"]
    return (policy, Pi05LiberoProcessor(stats, tokenizer, 200, str(device)),
            Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)), source)


def mean_states(tasks):
    means = {}
    for task, entry in tasks.items():
        with h5py.File(entry.authority.path, "r") as handle:
            episodes = []
            for demo in range(16, 42):
                obs = handle[f"data/demo_{demo}/obs"]
                state = np.concatenate((obs["ee_states"][:], obs["gripper_states"][:]), -1)
                assert state.ndim == 2 and state.shape[1] == 8 and np.isfinite(state).all()
                episodes.append(state.mean(0, dtype=np.float64))
            means[task] = torch.from_numpy(np.mean(episodes, axis=0).astype(np.float32))
    return means


@torch.no_grad()
def predict(policy, processor, tokenizer, query, arm, mean_state, seed, device):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    batch = {}
    for source, target in (("camera1", "base_0_rgb"), ("camera2", "left_wrist_0_rgb")):
        pixels = torch.as_tensor(query[f"observation.images.{source}"], device=device)
        if pixels.dtype == torch.uint8:
            pixels = pixels.float().div(255)
        batch[f"observation.images.{target}"] = pixels.unsqueeze(0).expand(8, -1, -1, -1)
    if arm == "state_free":
        tokens, mask, _ = tokenizer([query["task"]])
    else:
        state = mean_state if arm == "state_mean" else torch.as_tensor(query["observation.state"])
        tokens, mask = processor._tokenize_prompts(state.to(device).reshape(1, 8), [query["task"]])
    batch[OBS_LANGUAGE_TOKENS] = tokens.expand(8, -1)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = mask.expand(8, -1)
    noise = torch.randn((8, 50, 32), generator=torch.Generator().manual_seed(seed)).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
    assert generated.shape[:2] == (8, 50) and generated.shape[2] >= 7
    assert torch.isfinite(generated).all()
    return generated[:, :15, :7].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("state_free", "state_mean", "state_true"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    tasks, rows = manifest(root)
    if args.validate_only:
        print(json.dumps({"tasks": len(tasks), "positions": len(rows), "episodes": [42, 43, 44, 45]}))
        return
    assert args.arm is not None
    output = root / "runs/analysis/source_state_input_20260913"
    assert output.is_dir() and not (output / f"{args.arm}.json").exists()
    from ember.writer.data import FunctionalQueryDataset

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    policy, processor, tokenizer, source = source_model(root, device)
    dataset = FunctionalQueryDataset(tuple(t.authority for t in tasks.values()), demo_indices=(42, 43, 44, 45),
                                     action_chunk_size=15, action_start_offset=1)
    index = dataset.task_episode_rows
    means = mean_states(tasks)
    action_means = np.load(root / "runs/analysis/local_action_grounded_20260912/inverse_action_task_means.npz")
    results, samples, labels = [], [], []
    started = time.monotonic()
    for row in rows:
        task, clip = row["task"], row["clip"]
        query = dataset[index[task][row["local_action_demo"]][row["local_start_frame"]]]
        assert not query["action_is_pad"].any()
        seed = 20260914 + 100 * task + clip
        # Only the listed observation fields enter the inference function.
        observation = {k: query[k] for k in ("observation.images.camera1", "observation.images.camera2",
                                            "observation.state", "task")}
        generated = predict(policy, processor, tokenizer, observation, args.arm, means[task], seed, device)
        truth = processor.normalize_action(torch.from_numpy(query["action"])).cpu().numpy()
        mean_mse = float(np.mean((action_means[str(task)] - truth) ** 2))
        results.append({"task": task, "suite": tasks[task].suite, "clip": clip,
                        "demo": row["local_action_demo"], "frame": row["local_start_frame"],
                        "action_start": row["local_action_start"], "action_stop": row["local_action_stop"],
                        "noise_seed": seed, "mean_action_mse": float(np.mean((generated.mean(0) - truth) ** 2)),
                        "single_action_mse": float(np.mean((generated - truth) ** 2)),
                        "task_action_mean_mse": mean_mse})
        samples.append(generated)
        labels.append(truth)
        if len(results) % 16 == 0:
            print(json.dumps({"arm": args.arm, "complete": len(results), "seconds": time.monotonic() - started}), flush=True)
    torch.cuda.synchronize()
    result = {"arm": args.arm, "source": source, "gradient_updates": 0, "lora_generations": 0,
              "rollouts": 0, "flow_steps": 10, "noise_samples": 8, "rows": results,
              "wall_seconds": time.monotonic() - started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    np.savez(output / f"{args.arm}_samples.npz", samples=np.stack(samples), labels=np.stack(labels))
    write(output / f"{args.arm}.json", result)
    dataset.close()
    print(json.dumps({"arm": args.arm, "complete": len(results), "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
