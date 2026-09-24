"""Bounded original-query prediction for the frozen four-cell readout intervention."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.crossed_video_field import read_json
from ember.pi05_eval.readout_predict import ReadoutPredictor
from ember.pi05_eval.readout_state import GROUPS, derive_states
from ember.pi05_lora import load_pi05_lora_contract


PILOT_ID = "libero_goal:1:0:0"
QUERY_ROOT = Path("/data0/user/ymdai/ember_runs/crossed_video_action_field_20260924")


def _commit(repo: Path, formal: bool) -> str:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    commit = output("rev-parse", "HEAD")
    if formal:
        if output("status", "--porcelain=v1") or subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo, capture_output=True,
        ).returncode == 0 or subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
            cwd=repo, capture_output=True,
        ).returncode != 0:
            raise Pi05EvaluationError("formal readout runtime is not clean detached pushed commit")
    return commit


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _queries() -> list[dict]:
    path = QUERY_ROOT / "query_index.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != 300 or len({row["query_id"] for row in rows}) != 300
            or {(row["global_task_id"], row["control_step"])
                for row in rows} != {(task, step) for task in (14, 21) for step in (0, 10, 20)}):
        raise Pi05EvaluationError("sealed original 300 query panel changed")
    return rows


def _prepare(root: Path, repo: Path, commit: str, mode: str,
             spec: dict, bank: dict, original_bank_path: Path) -> None:
    if (root / "manifest.json").exists() or (root / "derived_states").exists():
        raise Pi05EvaluationError("readout study preparation already exists")
    conditions = {row["condition_id"]: row
                  for row in read_json(original_bank_path)["conditions"]
                  if int(row["global_task_id"]) in (14, 21)}
    if len(conditions) != 100:
        raise Pi05EvaluationError("readout original correct condition count changed")
    lora = load_pi05_lora_contract(Path(bank["lora_contract"]["path"]))
    derived = derive_states(conditions=conditions, lora=lora,
                            root=root / "derived_states",
                            original_manifest=original_bank_path)
    manifest = {
        "schema_version": "ember_readout_realization_execution_v1",
        "implementation_commit": commit, "mode": mode,
        "spec_path": str(repo / "configs/readout_realization_causality_v1/experiment_spec.json"),
        "query_root": str(QUERY_ROOT),
        "crossed_manifest": {"path": str(QUERY_ROOT / "manifest.json"),
                             "bytes": (QUERY_ROOT / "manifest.json").stat().st_size},
        "derived_manifest": {"path": str(root / "derived_states" / "manifest.json"),
                             "bytes": (root / "derived_states" / "manifest.json").stat().st_size},
        "condition_count": len(derived["conditions"]),
        "predictions": 1200, "rollouts": 400,
    }
    _write(root / "manifest.json", manifest)
    for group in GROUPS:
        for phase, states in (("pilot", [0]), ("remaining", list(range(1, 50)))):
            _write(root / "panels" / f"{group}_{phase}.json", {
                "schema_version": "ember_readout_realization_panel_v1",
                "study_id": spec["study_id"], "group": group, "phase": phase,
                "state_ids": states, "task_ids": [14, 21],
                "training_gradient_use": False, "checkpoint_selection_use": False,
                "validation_use": False, "test_use": False,
            })
    print(json.dumps({"phase": "prepare", "conditions": len(conditions),
                      "commit": commit}))


def _predict(root: Path, commit: str, args: argparse.Namespace, crossed: dict) -> None:
    manifest = read_json(root / "manifest.json")
    if (manifest["implementation_commit"] != commit or manifest["mode"] != args.mode
            or manifest["derived_manifest"]["bytes"] !=
            Path(manifest["derived_manifest"]["path"]).stat().st_size
            or not 0 <= args.shard_index < args.shard_count
            or args.panel not in ("pilot", "remaining")):
        raise Pi05EvaluationError("readout prediction implementation or panel changed")
    derived = read_json(Path(manifest["derived_manifest"]["path"]))
    selected = [row for row in _queries()
                if (row["query_id"] == PILOT_ID) == (args.panel == "pilot")]
    if len(selected) != (1 if args.panel == "pilot" else 299):
        raise Pi05EvaluationError("readout selected query count changed")
    selected = [row for index, row in enumerate(selected)
                if index % args.shard_count == args.shard_index]
    predictor = ReadoutPredictor(crossed, derived)
    started = time.monotonic()
    rows = []
    try:
        for query in selected:
            stem = query["query_id"].replace(":", "_")
            rows.append(predictor.predict_query(
                crossed, query, root / "predictions" / f"{stem}.npz",
                pilot=args.panel == "pilot"))
    finally:
        predictor.close()
    worker = {
        "schema_version": "ember_readout_realization_worker_v1",
        "phase": args.panel, "mode": args.mode,
        "implementation_commit": commit,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "query_count": len(rows), "prediction_count": sum(row["predictions"] for row in rows),
        "seconds": time.monotonic() - started, "rows": rows,
        "exit": "normal completion",
    }
    path = root / "workers" / f"{args.panel}_{args.shard_index}_of_{args.shard_count}.json"
    _write(path, worker)
    print(json.dumps({"phase": args.panel, "queries": len(rows),
                      "predictions": worker["prediction_count"], "worker": str(path)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "predict"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--panel", choices=("pilot", "remaining"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    commit = _commit(repo, args.mode == "formal")
    spec = read_json(repo / "configs/readout_realization_causality_v1/experiment_spec.json")
    if (spec["schema_version"] != "ember_readout_realization_causality_spec_v1"
            or args.mode == "formal" and root != Path(spec["run_root"]).resolve()):
        raise Pi05EvaluationError("readout study root or specification changed")
    crossed = read_json(QUERY_ROOT / "manifest.json")
    bank = crossed["banks"]["C_correct"]
    original_bank_path = Path(bank["manifest"]["path"])
    if (not original_bank_path.is_file()
            or original_bank_path.stat().st_size != bank["manifest"]["bytes"]
            or original_bank_path.parent != Path(spec["bank_root"]).resolve()):
        raise Pi05EvaluationError("readout original C bank identity changed")
    if args.phase == "prepare":
        _prepare(root, repo, commit, args.mode, spec, bank, original_bank_path)
    else:
        _predict(root, commit, args, crossed)


if __name__ == "__main__":
    main()
