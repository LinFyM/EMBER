"""At most three physical layouts on one longest registered training condition."""
from __future__ import annotations

import time
import traceback
import torch

from ember.pi05_source_checkpoint import capture_rng, restore_rng, write_json_atomic
from .credit import condition_credit
from .training import prepare


def run_profile(args):
    session = prepare(args)
    if session.context.world_size != 1:
        raise ValueError("this finite one-condition profile needs one physical rank")
    candidates = [tuple(map(int, value.split("x"))) for value in args.profile_configs.split(",")]
    if not 1 <= len(candidates) <= 3 or any(not 0 < micro <= 28 or chunk < 1 for micro, chunk in candidates):
        raise ValueError("only three positive physical layouts may be profiled")
    longest = None
    count = 0
    for step in range(128):
        for task in session.data.tasks_for_step(step):
            event = session.data.event(step, task)
            raw, frames = session.data.videos.frame_counts(task, event["teacher_demo"])
            candidate = (frames, raw, -step, -task, event)
            if longest is None or candidate[:4] > longest[:4]:
                longest = candidate
            count += 1
    event = longest[-1]
    rng = capture_rng(session.context)
    records = []
    try:
        for attempt, (micro, chunk) in enumerate(candidates, 1):
            session.optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(session.context.device)
            started = time.monotonic()
            row = {"attempt": attempt, "event": event, "microbatch": micro, "frame_chunk": chunk,
                   "queries_registered": 28, "updates": 0}
            try:
                credit = condition_credit(session.runtime, session.data, event,
                                          microbatch=micro, frame_chunk=chunk, check_initial=True)
                torch.cuda.synchronize(session.context.device)
                row.update(status="complete", credit=credit)
            except torch.cuda.OutOfMemoryError:
                row.update(status="OOM", error=traceback.format_exc())
            row.update(seconds=time.monotonic() - started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(session.context.device) / 2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(session.context.device) / 2**30,
                       free_bytes=torch.cuda.mem_get_info(session.context.device)[0])
            if row["status"] == "complete":
                row["queries_per_second"] = 28 / row["seconds"]
            records.append(row)
            write_json_atomic(session.output / f"profile_{attempt:02d}.json", row)
            session.optimizer.zero_grad(set_to_none=True)
            restore_rng(rng, session.context)
            torch.cuda.empty_cache()
        valid = [row for row in records if row["status"] == "complete"]
        if not valid:
            raise ValueError("all registered physical layouts failed on the longest actual condition")
        chosen = min(valid, key=lambda row: row["seconds"])
        write_json_atomic(session.output / "completion.json", {"status": "complete", "kind": "profile",
            "arm": "V", "metadata_conditions_considered": count, "queries_registered": len(records) * 28,
            "updates": 0, "chosen": {key: chosen[key] for key in ("microbatch", "frame_chunk", "seconds",
            "queries_per_second", "peak_allocated_gib", "peak_reserved_gib")}, "records": records,
            "stopped_growth_reason": "three registered physical layouts exhausted; choose measured best"})
    finally:
        session.data.close()
