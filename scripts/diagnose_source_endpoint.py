"""Bounded camera x native endpoint audit; retired after the registered readout."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np
import torch


@torch.no_grad()
def predict(policy, tokenizer, observation, view, steps, seed, probe_seed, device):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    count = 9 if steps == 1 else 8
    cameras = (("camera1", "base_0_rgb"), ("camera2", "left_wrist_0_rgb"))
    batch = {}
    for source, target in cameras[:1 if view == "agentview" else 2]:
        pixels = torch.as_tensor(observation[f"observation.images.{source}"], device=device)
        pixels = pixels.float().div(255) if pixels.dtype == torch.uint8 else pixels.float()
        batch[f"observation.images.{target}"] = pixels.unsqueeze(0).expand(count, -1, -1, -1)
    tokens, mask, _ = tokenizer([observation["task"]])
    batch[OBS_LANGUAGE_TOKENS] = tokens.expand(count, -1)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = mask.expand(count, -1)
    noise = torch.randn((8, 50, 32), generator=torch.Generator().manual_seed(seed))
    if steps == 1:
        probe = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(probe_seed))
        noise = torch.cat((noise, probe))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = policy.predict_action_chunk(batch, noise=noise.to(device), num_steps=steps)
    assert generated.shape[:2] == (count, 50) and generated.shape[2] >= 7
    assert torch.isfinite(generated).all()
    return generated[:, :15, :7].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--cell", choices=("agentview_full10", "agentview_t1", "dual_t1"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.asset_root.resolve()
    loader = root / ".codex/worktrees/source-state-audit/scripts/diagnose_source_state_input.py"
    spec = importlib.util.spec_from_file_location("source_state_frozen", loader)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    tasks, rows = helper.manifest(root)
    observer = helper.read(root / "runs/outputs/execution_aligned_video_20260912/ordered/run_contract.json")["config"]["observer"]
    assert observer["camera_view"] == "agentview" and observer["flow_time"] == 1
    probe_seed = int(observer["probe_seed"])
    if args.validate_only:
        print({"tasks": len(tasks), "positions": len(rows), "probe_seed": probe_seed})
        return
    assert args.cell is not None
    view, stage = args.cell.split("_")
    steps = 1 if stage == "t1" else 10
    output = root / "runs/analysis/source_state_input_20260913"
    destination = output / f"endpoint_{args.cell}.json"
    assert output.is_dir() and not destination.exists()
    from ember.writer.data import FunctionalQueryDataset

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    policy, processor, tokenizer, source = helper.source_model(root, device)
    dataset = FunctionalQueryDataset(tuple(t.authority for t in tasks.values()), demo_indices=(42, 43, 44, 45),
                                     action_chunk_size=15, action_start_offset=1)
    action_means = np.load(root / "runs/analysis/local_action_grounded_20260912/inverse_action_task_means.npz")
    results, samples, labels = [], [], []
    started = time.monotonic()
    for row in rows:
        task, clip = row["task"], row["clip"]
        query = dataset[dataset.task_episode_rows[task][row["local_action_demo"]][row["local_start_frame"]]]
        assert not query["action_is_pad"].any()
        observation = {k: query[k] for k in ("observation.images.camera1", "observation.images.camera2", "task")}
        seed = 20260914 + 100 * task + clip
        generated = predict(policy, tokenizer, observation, view, steps, seed, probe_seed, device)
        truth = processor.normalize_action(torch.from_numpy(query["action"])).cpu().numpy()
        result = {"task": task, "suite": tasks[task].suite, "clip": clip,
                  "demo": row["local_action_demo"], "frame": row["local_start_frame"],
                  "action_start": row["local_action_start"], "action_stop": row["local_action_stop"],
                  "noise_seed": seed, "mean_action_mse": float(np.mean((generated[:8].mean(0) - truth) ** 2)),
                  "task_action_mean_mse": float(np.mean((action_means[str(task)] - truth) ** 2))}
        if steps == 1:
            result["public_probe_mse"] = float(np.mean((generated[8] - truth) ** 2))
        results.append(result)
        samples.append(generated)
        labels.append(truth)
        if len(results) % 16 == 0:
            print({"cell": args.cell, "complete": len(results), "seconds": time.monotonic() - started}, flush=True)
    torch.cuda.synchronize()
    np.savez(output / f"endpoint_{args.cell}_samples.npz", samples=np.stack(samples), labels=np.stack(labels))
    helper.write(destination, {"cell": args.cell, "source": source, "loader_commit": "6fa6177f",
                 "gradient_updates": 0, "lora_generations": 0, "rollouts": 0, "flow_steps": steps,
                 "paired_noise_samples": 8, "public_probe_seed": probe_seed if steps == 1 else None,
                 "rows": results, "wall_seconds": time.monotonic() - started,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
    dataset.close()


if __name__ == "__main__":
    main()
