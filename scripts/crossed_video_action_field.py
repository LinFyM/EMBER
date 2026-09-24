"""Bounded execution of the registered frozen crossed-video diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.crossed_video_field import acquire_queries, build_manifest, read_json
from ember.pi05_eval.crossed_video_predict import CrossedVideoPredictor


def _git_commit(repo: Path, *, formal: bool) -> str:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    commit = output("rev-parse", "HEAD")
    if formal and output("status", "--porcelain=v1"):
        raise Pi05EvaluationError("formal crossed-video runtime must be clean")
    if formal:
        head = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo,
                              capture_output=True, text=True)
        pushed = subprocess.run(["git", "merge-base", "--is-ancestor", commit,
                                 "origin/main"], cwd=repo, capture_output=True)
        if head.returncode == 0 or pushed.returncode != 0:
            raise Pi05EvaluationError("formal crossed-video commit is not detached and pushed")
    return commit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "queries", "predict"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--panel", choices=("pilot", "remaining"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    formal = args.mode == "formal"
    commit = _git_commit(repo, formal=formal)
    if args.phase == "prepare":
        manifest = build_manifest(repo, root, formal=formal)
        manifest["implementation_commit"] = commit
        manifest["mode"] = args.mode
        _write_json(root / "manifest.json", manifest)
        print(json.dumps({"phase": "prepare", "commit": commit,
                          "queries": len(manifest["query_plan"])}))
        return
    manifest = read_json(root / "manifest.json")
    if manifest["mode"] != args.mode or manifest["implementation_commit"] != commit:
        raise Pi05EvaluationError("crossed-video manifest implementation identity changed")
    if args.phase == "queries":
        acquired = acquire_queries(manifest, root, smoke=not formal)
        path = root / "query_index.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for query in acquired:
                handle.write(json.dumps(query, sort_keys=True) + "\n")
        print(json.dumps({"phase": "queries", "count": len(acquired), "index": str(path)}))
        return
    if args.panel is None or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise Pi05EvaluationError("crossed-video prediction shard or panel is invalid")
    query_index = _read_index(root / "query_index.jsonl")
    expected = 300 if formal else 3
    if len(query_index) != expected or len({row["query_id"] for row in query_index}) != expected:
        raise Pi05EvaluationError("crossed-video query index incomplete")
    pilot_id = "libero_goal:1:0:0"
    selected = [query for query in query_index
                if (query["query_id"] == pilot_id) == (args.panel == "pilot")]
    if len(selected) != (1 if args.panel == "pilot" else expected - 1):
        raise Pi05EvaluationError("crossed-video registered panel incomplete")
    selected = [row for index, row in enumerate(selected)
                if index % args.shard_count == args.shard_index]
    predictor = CrossedVideoPredictor(manifest, batch_size=args.batch_size)
    started = time.monotonic()
    rows = []
    try:
        for query in selected:
            stem = query["query_id"].replace(":", "_")
            output = root / "predictions" / f"{stem}.npz"
            rows.append(predictor.predict_query(
                manifest, query, output, pilot=args.panel == "pilot"))
    finally:
        predictor.close()
    worker = {
        "schema_version": "ember_crossed_video_field_worker_v1",
        "phase": args.panel, "mode": args.mode,
        "implementation_commit": commit,
        "batch_size": args.batch_size, "shard_index": args.shard_index,
        "shard_count": args.shard_count, "query_count": len(rows),
        "prediction_count": sum(row["predictions"] for row in rows),
        "seconds": time.monotonic() - started, "rows": rows,
        "exit": "normal completion",
    }
    path = root / "workers" / f"{args.panel}_{args.shard_index}_of_{args.shard_count}.json"
    _write_json(path, worker)
    print(json.dumps({"phase": args.panel, "queries": len(rows),
                      "predictions": worker["prediction_count"],
                      "seconds": worker["seconds"], "worker": str(path)}))


if __name__ == "__main__":
    main()
