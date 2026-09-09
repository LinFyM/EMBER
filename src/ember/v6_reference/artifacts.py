"""Exploratory v6 checkpoint provenance and frozen train24 action diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.v6_reference import contract


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def inspect_checkpoint(checkpoint: Path, run: Mapping[str, Any] | None = None):
    checkpoint = checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = dict(run) if run is not None else read_json(run_path)
    config = contract.validate_config(run["config"])
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    world = int(manifest.get("world_size", 0))
    files = {"ecp.safetensors", "trainer_state.pt", *(f"rank_{rank:02d}_state.pt" for rank in range(world))}
    expected_run = {"schema_version": contract.RUN_SCHEMA, "stage": contract.STAGE,
                    "mode": "exploratory", "model_config": config["model"]}
    expected_manifest = {"schema_version": ECP_CHECKPOINT_SCHEMA, "stage": contract.STAGE,
                         "run_contract_schema": contract.RUN_SCHEMA, "next_macro": macro}
    if ({key: run.get(key) for key in expected_run} != expected_run
            or {key: manifest.get(key) for key in expected_manifest} != expected_manifest
            or not contract.execution_authority(run.get("git", {}))
            or macro not in config["evidence"]["checkpoint_updates"]
            or not 1 <= world <= 4 or run.get("topology", {}).get("world_size") != world
            or set(manifest.get("files", {})) != files):
        raise ValueError("v6 materialization requires a complete registered exploratory checkpoint")
    for name, item in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"exploratory v6 checkpoint file changed: {name}")
    _inspect_training_state(checkpoint, macro, config)
    with safe_open(str(checkpoint / "ecp.safetensors"), framework="pt", device="cpu") as handle:
        probe = "writer.semantic_encoder.fixed_suffix_noise"
        if probe not in handle.keys() or tuple(handle.get_slice(probe).get_shape()) != (50, 32):
            raise ValueError("v6 checkpoint lost its original public probe")
    return run, {"path": str(checkpoint), "macro": macro, "kind": contract.BANK_KIND,
                 "weights": _record(checkpoint / "ecp.safetensors"),
                 "manifest": _record(checkpoint / "checkpoint_manifest.json"),
                 "run_contract": _record(run_path), "training_commit": run["git"]["commit"]}


def _inspect_training_state(checkpoint, macro, config):
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True, weights_only=True)
    expected_training = {"schema_version": contract.TRAINING_SCHEMA, "updates": macro,
                         "update_version": contract.UPDATE_VERSION, "data_version": config["data"]["version"]}
    expected = {"schema_version": ECP_CHECKPOINT_SCHEMA, "stage": contract.STAGE, "next_macro": macro,
                "training_state": expected_training, "metrics_rows": macro * 4, "scaler": None}
    if ({key: trainer.get(key) for key in expected} != expected
            or trainer.get("sampler_state", {}).get("next_step") != macro
            or not trainer.get("optimizer") or not trainer.get("scheduler")):
        raise ValueError("v6 complete learning state or sampling cursor changed")


def diagnose(args) -> dict[str, Any]:
    """No optimizer, sampler draw, validation-task action, or gradient is created."""
    from ember.pi05_eval_contract import git_state
    from ember.pi05_source_checkpoint import DistributedContext
    from ember.v6_reference.runtime import build_runtime, V6SupervisedEngine
    from ember.writer.learning_data import WriterTrainingData
    from ember.writer.materialization import source_matches
    from ember.writer.topology import bind_current_process_to_cuda_numa

    repository = Path(__file__).resolve().parents[3]
    if not contract.execution_authority(git_state(repository)):
        raise ValueError("v6 diagnostic requires the clean pushed detached reference checkout")
    run, checkpoint = inspect_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("full frozen action diagnostics require an explicitly scheduled CUDA device")
    if args.output.exists():
        raise ValueError("diagnostic output already exists")
    torch.cuda.set_device(device)
    if not bind_current_process_to_cuda_numa(torch.cuda.current_device()):
        raise ValueError("v6 action diagnostic requires GPU-local NUMA placement")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = run["config"]
    execution = {**config, "runtime": {"policy_microbatch": args.policy_microbatch}}
    runtime = build_runtime(args.asset_root.resolve(), config, device)
    if not source_matches(runtime.source, run["source"]):
        raise ValueError("frozen v6 diagnostic changed its source policy")
    runtime.state.load_state_dict(load_file(checkpoint["weights"]["path"], device=str(device)), strict=True)
    runtime.state.requires_grad_(False).eval()
    data = WriterTrainingData(args.asset_root.resolve(), config["data"])
    before = data.sampler_state()
    context = DistributedContext(0, device.index or 0, 1, device)
    engine = V6SupervisedEngine(runtime, data, context, execution)
    spec = config["evidence"]["supervised_validation"]
    try:
        rows = [engine.validate(task, spec["teacher_video_pool"][task % len(spec["teacher_video_pool"])],
                                seed=spec["seed"] + task, queries=spec["queries_per_task"])
                for task in spec["task_ids"]]
        if data.sampler_state() != before:
            raise RuntimeError("frozen diagnostic advanced the training sampler")
    finally:
        data.close()
    result = {"schema_version": "ember_exploratory_v6_frozen_actions_v1", "checkpoint": checkpoint,
              "rows": rows, "queries": sum(row["queries"] for row in rows),
              "mean_flow_loss": sum(row["flow_loss"] for row in rows) / len(rows),
              "gradients": False, "sampler_unchanged": True, "scientific_qualification": False}
    write_json_atomic(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    inspect = commands.add_parser("inspect", help="CPU metadata-only complete checkpoint inspection")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    diagnostic = commands.add_parser("diagnose", help="frozen train24 held-episode FM; requires a scheduled GPU")
    diagnostic.add_argument("--checkpoint", type=Path, required=True)
    diagnostic.add_argument("--asset-root", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--device", default="cuda:0")
    diagnostic.add_argument("--policy-microbatch", type=int, default=8)
    diagnostic.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.operation == "inspect":
        run, record = inspect_checkpoint(args.checkpoint)
        print(json.dumps({"checkpoint": record, "method": contract.method_metadata(run)}, indent=2))
    else:
        result = diagnose(args)
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
