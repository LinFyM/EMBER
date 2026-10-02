"""At most three discarded real macro updates through the canonical trainer."""
from __future__ import annotations
import time

import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import write_json_atomic
from . import control_calibration as calibration


def validate(spec, args, output):
    if (spec["task"] != calibration.TASK or args.resume is not None
            or args.mode != calibration.MODE or output.exists() and (output / "run_contract.json").exists()
            or len(list((calibration.ROOT / "profile").glob("*/run_contract.json"))) >= 3
            or args.microbatch not in (7, 14, 28) or not 4 <= args.frame_chunk <= 128):
        raise ValueError("profile requires at most three registered real updates and fresh state")


def profile(spec, args):
    from .run import prepare_train, update
    session = prepare_train(spec, args)
    started = time.monotonic()
    try:
        step = next(k for k in range(450) if 38 in session.data.tasks_for_step(k)
                    and session.data.event(k, 38)["teacher_demo"] == 36)
        events = [session.data.event(step, t) for t in session.data.tasks_for_step(step)]
        raw, sampled = session.data.videos.frame_counts(38, 36)
        if sampled != 105 or any(len(e["queries"]) != 28 for e in events):
            raise ValueError("profile changed the sealed longest105-frame/28-query event")
        update(session, step, 0)
        torch.cuda.synchronize()
        if session.context.is_main:
            write_json_atomic(session.output / "completion.json", {
                "profile_only": True, "discarded_complete_updates": 1,
                "formal_initialization_restored_by_fresh_process": True,
                "event_macro": step + 1, "events": events, "longest_sampled_frames": sampled,
                "seconds_including_update": time.monotonic() - started,
                "world_size": session.context.world_size, "microbatch": args.microbatch,
                "frame_chunk": args.frame_chunk, "git": session.contract["git"],
                "actual_auxiliary_and_full_FM_replay": True, "optimizer_updates": 1})
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()
