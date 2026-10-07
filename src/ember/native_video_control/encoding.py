"""Encode each fixed readout condition once under the terminal frozen model."""
from __future__ import annotations

import time
import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import write_json_atomic
from .checkpoint import inspect_checkpoint
from .credit import native_memory
from .data import NativeData
from .evaluation import evaluation_conditions, publish_manifest, save_memory
from .runtime import build_runtime, frozen_git
from .specification import specification


def encode(args):
    git = frozen_git()
    spec = specification()
    inspect_checkpoint(args.checkpoint, arm=args.arm, terminal=True)
    panel = "correct" if args.condition == "correct" else "same_task_other"
    rows = evaluation_conditions(panel)
    if args.arm == "M":
        if panel != "correct":
            raise ValueError("M has no additional other-video environment panel")
        return publish_manifest(args.checkpoint, arm="M", condition=panel,
                                asset_root=args.asset_root, encoding_git=git, memories={})
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    runtime = build_runtime(args.asset_root, spec, device, "V")
    state = load_file(str(args.checkpoint / "ecp.safetensors"), device="cpu")
    runtime.controller.load_state_dict(state, strict=True)
    runtime.controller.eval()
    data = NativeData(args.asset_root, spec, query_labels=False)
    memories = {}
    started = time.monotonic()
    try:
        for row in rows:
            condition, raw_count, sampled = data.condition(runtime, row["global_task_id"], row["teacher_demo"])
            with torch.no_grad():
                c = native_memory(runtime, condition, frame_chunk=args.frame_chunk, backward=False)
            record = save_memory(args.checkpoint, row, c, condition[1], panel=panel)
            record.update(raw_frames=raw_count, sampled_frames=sampled)
            memories[row["condition_id"]] = record
        path = publish_manifest(args.checkpoint, arm="V", condition=panel, asset_root=args.asset_root,
                                encoding_git=git, memories=memories)
        write_json_atomic(path.parent / "encoding_completion.json", {"status": "complete", "conditions": len(rows),
            "seconds": time.monotonic() - started, "checkpoint": str(args.checkpoint), "git": git,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30})
        return path
    finally:
        data.close()
