"""Frozen, bounded CPU task23 replay dispatch; no automatic successor work."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback


def worker(root: str, attempt: str, row: dict, arms: tuple[str, ...]) -> list[dict]:
    started, cpu_started = time.monotonic(), time.process_time()
    path = Path(root) / "attempts" / attempt / "rows" / f"{row['model']}_state{row['state']:03d}"
    from ember.pi05_eval.drawer_state_replay import replay
    results = []
    try:
        for arm in arms:
            result = replay(Path(root), row, path / arm)
            results.append(result)
            if arm == "recorded_parent" and not result["deviation"]["interpretable_parent"]:
                break  # no counterfactual for a non-admitted command consumer
    except Exception:
        path.mkdir(parents=True, exist_ok=True)
        failure = dict(model=row["model"], state=row["state"], arms=arms,
                       traceback=traceback.format_exc(), completed_arms=[r["arm"] for r in results],
                       wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
                       peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (path / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--phase", choices=("admission", "panel", "remaining"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only maximum eight workers, no policy or rendering")
    root, run = args.root, args.root / "attempts" / args.attempt
    run.mkdir(parents=True, exist_ok=True)
    rows = json.loads((root / "inputs.json").read_text())["rows"]
    contract = json.loads((root / "run_contract.json").read_text())
    deadline = datetime.datetime.fromisoformat(contract["hard_deadline_utc"].replace("Z", "+00:00")).timestamp()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("frozen run source must be clean")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 0:
        raise ValueError("frozen run source must be detached")
    started, cpu_started = time.monotonic(), time.process_time()
    if (root / "launch/retired.json").exists():
        raise ValueError("Closed task23 batch cannot run new physics")
    jobs = []
    if args.phase == "admission":
        jobs = [(row, ("recorded_parent",)) for row in rows if row["state"] == 0]
    else:
        if args.phase == "remaining":
            previous = json.loads((run / "panel_completion.json").read_text())
            if previous["exceptions"] or len(previous["invalid"]) != 1:
                raise ValueError("remaining-only continuation requires the recorded single-row validity boundary")
        for row in rows:
            parent = run / "rows" / f"{row['model']}_state{row['state']:03d}" / "recorded_parent.json"
            if row["state"] == 0:
                if not parent.is_file() or not json.loads(parent.read_text())["deviation"]["interpretable_parent"]:
                    raise ValueError("both init0 parent rows must pass admission before panel")
                continue  # init0 admission rows are already part of the fixed 100
            else:
                if args.phase == "remaining" and parent.is_file():
                    saved = json.loads(parent.read_text())
                    if saved["source"] != row:
                        raise ValueError("existing source identity changed")
                    continue  # retain admitted and non-admitted rows without repeating either
                jobs.append((row, ("recorded_parent",)))
    start = dict(commit=commit, cwd=os.getcwd(), phase=args.phase, workers=args.workers,
                 command=list(__import__("sys").argv), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], renderer="none",
                 job_count=len(jobs), max_seconds_remaining=deadline-time.time())
    (run / f"{args.phase}_start.json").write_text(json.dumps(start, indent=2) + "\n")
    outcomes, exceptions, invalid = [], [], []
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    context = multiprocessing.get_context("spawn")
    # Submit at most one job per worker; stop new jobs on any parent mismatch.
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
    pending, cursor, stopped = {}, 0, False
    try:
        while cursor < len(jobs) or pending:
            while cursor < len(jobs) and len(pending) < args.workers and not stopped:
                if time.time() >= deadline:
                    stopped = True
                    exceptions.append({"boundary": "wall-clock budget before next job"})
                    break
                row, arms = jobs[cursor]
                pending[pool.submit(worker, str(root), args.attempt, row, arms)] = (row["model"], row["state"])
                cursor += 1
            if not pending:
                break
            ready, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                identity = pending.pop(future)
                try:
                    current = future.result()
                    outcomes.extend(current)
                    bad = [r for r in current if r["arm"] == "recorded_parent"
                           and not r["deviation"]["interpretable_parent"]]
                    if bad:
                        invalid.extend(dict(model=r["model"], state=r["state"], deviation=r["deviation"]) for r in bad)
                        stopped = True
                except Exception:
                    exceptions.append(dict(identity=identity, traceback=traceback.format_exc()))
                    stopped = True
            if stopped and not pending:
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    completion = dict(**start, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      wall_seconds=time.monotonic()-started, parent_cpu_seconds=time.process_time()-cpu_started,
                      child_cpu_seconds=(children_after.ru_utime+children_after.ru_stime-
                                         children_before.ru_utime-children_before.ru_stime),
                      completed_arms=len(outcomes), invalid=invalid, exceptions=exceptions,
                      submitted_jobs=cursor, unscheduled_jobs=len(jobs)-cursor,
                      exit_code=0 if not invalid and not exceptions else 2)
    (run / f"{args.phase}_completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion), flush=True)
    return completion["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
