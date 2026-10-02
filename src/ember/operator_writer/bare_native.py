"""Action-free F0 feature cache; separate from train-only control labels.

Public beta-native remains in Runtime.compile and is never served from here.
Cache construction uses the same canonical state-free native call on bare F0.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from ember.pi05_eval_contract import git_state, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks
from ember.writer.runtime import autocast
from .native import _NativeFrameCall

SCHEMA = "ember_bare_native_control_features_v1"
FIELDS = {"schema", "H0", "mu0", "frame_indices", "source", "reading_git"}


def feature_path(spec, task, demo):
    return Path(spec["run_root"]) / "bare_native" / f"task{task:03d}_demo{demo:02d}.pt"


def validate(value, indices):
    if (set(value) != FIELDS or value["schema"] != SCHEMA
            or value["H0"].shape != (len(indices), 50, 1024)
            or value["mu0"].shape != (len(indices), 5, 7)
            or not torch.equal(value["frame_indices"].cpu(), torch.as_tensor(indices).cpu())
            or not torch.isfinite(value["H0"]).all() or not torch.isfinite(value["mu0"]).all()):
        raise ValueError("bare cache fields/indices must be pure aligned F0 features")


def load_features(spec, task, demo, indices, device, source):
    value = torch.load(feature_path(spec, task, demo), map_location="cpu", weights_only=True, mmap=True)
    validate(value, indices)
    if value["source"] != source:
        raise ValueError("bare-source identity differs from the actual consumer")
    return {key: value[key].to(device, non_blocking=True)
            for key in ("H0", "mu0", "frame_indices")}


def write_features(spec, task, demo, h, mu, indices, source, reading_git):
    path = feature_path(spec, task, demo)
    value = {"schema": SCHEMA, "H0": h.float().cpu().contiguous(),
             "mu0": mu.float().cpu().contiguous(), "frame_indices": torch.as_tensor(indices).cpu(),
             "source": source, "reading_git": reading_git}
    validate(value, indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_bare_frames(wrapper, pixels, tokens, mask, probe, chunk, profile=None):
    """H and mu use the actual velocity from the same native call, no second forward."""
    hs, mus = [], []
    start = 0
    while start < len(pixels):
        trial = (min(128, chunk), min(128, chunk), chunk)
        requested = trial[len(profile)] if profile is not None and len(profile) < 3 else chunk
        physical = min(requested, len(pixels) - start)
        began = time.monotonic()
        captured = []
        hook = wrapper.policy.model.action_out_proj.register_forward_hook(
            lambda _module, _args, output: captured.append(output))
        try:
            with torch.no_grad(), autocast(probe.device):
                h, = wrapper(pixels[start:start + physical], tokens, mask)
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].shape != (len(h), 50, 32):
            raise ValueError("bare native lost the same-call full50 velocity")
        hs.append(h.detach().float().cpu())
        mus.append((probe[None] - captured[0].float())[:, :5, :7].detach().cpu())
        elapsed = time.monotonic() - began
        if profile is not None and len(profile) < 3 and physical == requested:
            profile.append({"frame_start": start, "frames": physical, "seconds": elapsed,
                            "frames_per_second": physical / elapsed,
                            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30})
            if len(profile) == 3:
                chunk = profile[2]["frames"] if profile[2]["frames_per_second"] > profile[1]["frames_per_second"] else profile[1]["frames"]
        start += physical
    return torch.cat(hs), torch.cat(mus)


def cache(spec, args):
    """One source load per independent frame-balanced cache worker, no label owner."""
    import ember.pi05_evaluation  # Initialize the existing facade before its internal loader.
    from ember.pi05_eval.worker_setup import load_policy
    from ember.writer.topology import bind_current_process_to_cuda_numa
    from .run import frozen_git
    from .data import TASKS

    reading_git = frozen_git()
    bind_current_process_to_cuda_numa(0)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    asset, cfg = args.asset_root, spec["source"]
    auth = load_evaluation_authorities(asset / cfg["evaluation_config"], asset)
    cp = asset / cfg["checkpoint"]
    source = inspect_source_checkpoint(auth, cp.parent.parent, cp, evaluation_mode="formal")
    tokenpath = asset / cfg["tokenizer"]
    stats = read_json(asset / cfg["normalization"])["stats"]
    policy, _processor, _ = load_policy(Path(source["model_path"]), stats, tokenpath,
                                       read_json(asset / "configs/pi05_target_evaluation_v1.json")["policy"])
    policy.requires_grad_(False).eval()
    if any("lora_" in name for name, _ in policy.named_parameters()):
        raise ValueError("F0 cache must never load public or generated LoRA")
    tokenizer = Pi05TeacherPrefixTokenizer(tokenpath, 200, "cuda:0")
    probe = torch.randn(50, 32, generator=torch.Generator().manual_seed(1729)).cuda()
    wrapper = _NativeFrameCall(policy, (), probe)
    rows = []
    for role, ids in (("train", TASKS), ("validation", spec["evaluation"]["task_ids"])):
        tasks = load_learning_tasks(asset, ids, role=role, protocol_path=cfg["data_protocol"])
        for task, row in tasks.items():
            rows.extend((task, demo, length, row) for demo, length in enumerate(row.episode_lengths[:50]))
    # Greedy frame load, independent of task labels and GPU identities.
    bins, loads = [[] for _ in range(args.cache_shards)], [0] * args.cache_shards
    for row in sorted(rows, key=lambda x: (-x[2], x[0], x[1])):
        shard = min(range(args.cache_shards), key=lambda i: loads[i])
        bins[shard].append(row)
        loads[shard] += (row[2] - 1) // 5 + 2
    todo = bins[args.cache_shard]
    store = RawTeacherVideoStore(tuple({r[0]: r[3].authority for r in todo}.values()),
                                frame_stride=5, camera_view="dual")
    started, frames, reused = time.monotonic(), 0, 0
    chunks, records, profile, packing_failures = args.frame_chunk, [], [], []
    try:
        for task in sorted({r[0] for r in todo}):
            selected = [r for r in todo if r[0] == task]
            missing = []
            for t, demo, length, row in selected:
                path = feature_path(spec, t, demo)
                if path.exists():
                    reused += 1
                else:
                    missing.append((t, demo, row, store.load(t, demo)))
            if not missing:
                continue
            pixels = torch.from_numpy(np.concatenate([r[3].frames for r in missing])).cuda()
            tokens, mask, _ = tokenizer([missing[0][2].authority.language])
            beg = time.monotonic()
            while True:
                try:
                    h, mu = read_bare_frames(wrapper, pixels, tokens, mask, probe, chunks, profile)
                    break
                except torch.cuda.OutOfMemoryError:
                    packing_failures.append({"task": task, "frames": len(pixels), "frame_chunk": chunks,
                                             "reason": "CUDA OOM; same authorized frames retried"})
                    if chunks <= 64:
                        raise
                    chunks //= 2
                    profile.clear()
                    torch.cuda.empty_cache()
            offset = 0
            for t, demo, _row, video in missing:
                stop = offset + len(video.frames)
                write_features(spec, t, demo, h[offset:stop].clone(), mu[offset:stop].clone(),
                               video.frame_indices, source, reading_git)
                offset = stop
            if len(profile) == 3:
                chunks = profile[2]["frames"] if profile[2]["frames_per_second"] > profile[1]["frames_per_second"] else profile[1]["frames"]
            frames += len(h)
            records.append({"task": task, "frames": len(h), "videos": len(missing),
                            "frame_chunk": chunks, "seconds": time.monotonic() - beg})
            del pixels, h, mu, missing
    finally:
        store.close()
    record = {"schema": SCHEMA, "complete": True, "source": source, "reading_git": reading_git,
              "shard": args.cache_shard, "shards": args.cache_shards, "videos": len(todo),
              "frames_new": frames, "reused": reused, "records": records, "packing_profile": profile,
              "packing_failures": packing_failures,
              "seconds": time.monotonic() - started,
              "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
              "labels_read": 0, "state_reward_read": 0, "Source_loads": 1}
    write_json_atomic(Path(spec["run_root"]) / "bare_native" / f"completion_shard{args.cache_shard}.json", record)
