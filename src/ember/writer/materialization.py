"""Compile sealed joint Writer checkpoints into per-episode complete LoRAs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.lora import validate_lora_state
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.data import RawTeacherVideoStore


RUN_SCHEMA = "ember_layered_relation_writer_joint_run_v1"
STAGE = "layered_relation_writer_fresh_joint"
BANK_SCHEMA = "ember_layered_writer_lora_bank_v1"
BANK_KIND = "layered_writer_lora_bank"
ADAPTER_SCHEMA = "ember_layered_writer_materialized_adapter_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def frozen_authority(state: Mapping[str, Any]) -> bool:
    return state.get("branch") == "" and git_state_is_clean_pushed_or_frozen_authority(state)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def source_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("source_run", "checkpoint", "model_path")
    return all(left.get(key) and right.get(key) and
               Path(left[key]).resolve() == Path(right[key]).resolve() for key in keys)


def inspect_joint_checkpoint(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check retained checkpoint authority without loading optimizer/RNG or hashes."""
    checkpoint = checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    run_path = checkpoint.parent.parent / "run_contract.json"
    run, manifest = read_json(run_path), read_json(checkpoint / "checkpoint_manifest.json")
    world_size = int(manifest.get("world_size", 0))
    expected = {"ecp.safetensors", "trainer_state.pt", *(f"rank_{rank:02d}_state.pt" for rank in range(world_size))}
    if (macro <= 0 or not 1 <= world_size <= 6 or run.get("schema_version") != RUN_SCHEMA
            or run.get("stage") != STAGE or run.get("mode") != "formal"
            or not frozen_authority(run.get("git", {}))
            or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
            or manifest.get("stage") != STAGE or manifest.get("run_contract_schema") != RUN_SCHEMA
            or manifest.get("next_macro") != macro or set(manifest.get("files", {})) != expected):
        raise ValueError("materialization requires a complete formal fresh-joint Writer checkpoint")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"joint Writer checkpoint file changed: {name}")
    return run, {"path": str(checkpoint), "macro": macro,
                 "weights": file_record(checkpoint / "ecp.safetensors"),
                 "manifest": file_record(checkpoint / "checkpoint_manifest.json"),
                 "run_contract": file_record(run_path), "training_commit": run["git"]["commit"]}


def _fixed_video_selection(fixed_videos, *, mode, tasks, cardinality, pool):
    fixed = {str(key): sorted(map(int, value)) for key, value in (fixed_videos or {}).items()}
    if fixed and (mode != "fixed_per_task" or set(fixed) != set(map(str, tasks)) or
                  any(len(value) != cardinality or len(set(value)) != cardinality or not set(value) <= set(pool)
                      for value in fixed.values())):
        raise ValueError("fixed diagnostic videos must provide one distinct K-set for every task")
    return fixed


def selection_contract(
    *, role: str, task_ids: Sequence[int], cardinality: int, arm: str, mode: str,
    seed: int, init_state_ids: Sequence[int], video_pool: Sequence[int],
    fixed_videos: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    tasks, states, pool = tuple(task_ids), tuple(init_state_ids), tuple(video_pool)
    if (role not in {"development_train", "validation"} or cardinality not in (1, 2, 4)
            or arm not in {"correct", "same_task_other"} or mode not in {"fixed_per_task", "per_init_ordinal"}
            or not tasks or len(set(tasks)) != len(tasks) or seed < 0
            or not states or tuple(sorted(set(states))) != states or not set(states) <= set(range(50))
            or len(set(pool)) != len(pool) or not set(pool) <= set(range(50)) or len(pool) < cardinality):
        raise ValueError("invalid explicit task/condition selection; Test and final controls are excluded")
    fixed = _fixed_video_selection(fixed_videos, mode=mode, tasks=tasks, cardinality=cardinality, pool=pool)
    if role == "validation" and mode != "per_init_ordinal":
        raise ValueError("validation banks use per-init-state random video sets; fixed sets are train diagnostics")
    return {"evaluation_role": role, "task_ids": list(tasks), "K": cardinality, "arm": arm,
            "mode": mode, "seed": seed, "init_state_ids": list(states), "video_pool": sorted(pool),
            "fixed_videos": fixed, "outcome_dependence": False, "gradient_use": False,
            "without_replacement": True, "video_ordinal_rule": "init_state_id" if mode == "per_init_ordinal" else "fixed_zero"}


def paired_video_sets(selection: Mapping[str, Any], task: int, ordinal: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Same seed/ordinal yields a correct set and a disjoint alternative set."""
    k = int(selection["K"])
    rng = random.Random(int(selection["seed"]) + 1_000_003 * task + 7_919 * ordinal)
    fixed = selection["fixed_videos"].get(str(task))
    correct = tuple(sorted(fixed if fixed is not None else rng.sample(selection["video_pool"], k)))
    remaining = [demo for demo in selection["video_pool"] if demo not in correct]
    other = tuple(sorted(rng.sample(remaining, k))) if len(remaining) >= k else ()
    if selection["arm"] == "same_task_other" and len(other) != k:
        raise ValueError("same-task-other requires at least K additional disjoint videos")
    return correct, other


def condition_id(task: int, demos: Sequence[int]) -> str:
    return f"task_{task:02d}_demos_" + "_".join(f"{demo:02d}" for demo in sorted(demos))


def planned_episodes(selection: Mapping[str, Any], task: int) -> list[dict[str, Any]]:
    rows = []
    for state in selection["init_state_ids"]:
        ordinal = state if selection["mode"] == "per_init_ordinal" else 0
        correct, other = paired_video_sets(selection, task, ordinal)
        demos = correct if selection["arm"] == "correct" else other
        rows.append({"init_state_id": state, "video_ordinal": ordinal,
                     "condition_id": condition_id(task, demos), "teacher_demo_indices": list(demos),
                     "paired_correct_demos": list(correct), "paired_other_demos": list(other)})
    return rows


def method_metadata(run: Mapping[str, Any]) -> dict[str, Any]:
    return {"model_config": run["model_config"], "observer": run["config"]["observer"],
            "checkpoint_state": "strict entire Writer+Meta+public probe", "frame_stride": 5,
            "include_last_frame": True, "camera": "agentview_rotated_180", "execution_rank": 16,
            "native_response_shape": [18, 50, 1024], "generated_tensor_count": 76}


def adapter_metadata(condition: str, checkpoint: Mapping[str, Any]) -> dict[str, str]:
    return {"schema_version": ADAPTER_SCHEMA, "condition_id": condition,
            "writer_checkpoint": str(checkpoint["path"]), "macro": str(checkpoint["macro"])}


def _compile_condition(runtime, store, task, demos, output, checkpoint):
    from ember.writer.native import autocast

    videos = tuple(store.load(task.authority.task_id, demo) for demo in demos)
    if any(video.raw_frame_count != task.episode_lengths[demo] for demo, video in zip(demos, videos, strict=True)):
        raise ValueError("actual teacher frame count differs from its data authority")
    condition = runtime.observer.prepare(
        tuple(torch.from_numpy(video.frames) for video in videos),
        tuple(torch.from_numpy(video.frame_indices) for video in videos), task.authority.language,
    )
    with torch.no_grad(), autocast(runtime.observer.device):
        generated = runtime.state.writer(runtime.observer.responses(condition), condition.frame_indices,
                                         condition.language_embeddings, condition.language_mask)
    state = {name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
             for name, value in generated.items()}
    validate_lora_state(state, runtime.lora)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Writer generated nonfinite LoRA parameters")
    identifier = condition_id(task.authority.task_id, demos)
    path = output / f"{identifier}.safetensors"
    save_file(state, str(path), metadata=adapter_metadata(identifier, checkpoint))
    return {"condition_id": identifier, "global_task_id": task.authority.task_id,
            "suite": task.suite, "task_id": task.suite_task_id, "language": task.authority.language,
            "teacher_demo_indices": list(demos), "teacher_videos": [
                {"demo_index": demo, "raw_frame_count": video.raw_frame_count,
                 "sampled_frame_count": len(video.frame_indices), "frame_indices": video.frame_indices.tolist()}
                for demo, video in zip(demos, videos, strict=True)],
            "adapter": file_record(path), "writer_invocations": 1, "single_complete_rank16": True}


def materialize(
    *, asset_root: Path, checkpoint: Path, output: Path,
    selection: Mapping[str, Any], device: torch.device,
) -> Path:
    from ember.writer.learning_data import load_learning_tasks
    from ember.writer.runtime import build_joint_runtime
    from ember.writer.evaluation import validate_task_scope

    repository = git_state(REPO_ROOT)
    if not frozen_authority(repository):
        raise ValueError("materialization requires a clean pushed detached checkout")
    run, checkpoint_record = inspect_joint_checkpoint(checkpoint)
    role = "train" if selection["evaluation_role"] == "development_train" else "validation"
    tasks = load_learning_tasks(asset_root, selection["task_ids"], role=role)
    rows = [{"global_task_id": task, "suite": value.suite, "task_id": value.suite_task_id,
             "language": value.authority.language, "split_role": role,
             "teacher_source": file_record(value.authority.path), "episodes": planned_episodes(selection, task)}
            for task, value in tasks.items()]
    validate_task_scope(rows, selection["evaluation_role"], asset_root)
    # Preserve the checkpoint's explicit defaults even if later defaults evolve.
    runtime_config = {**run["config"], "model": run["model_config"]}
    runtime = build_joint_runtime(asset_root, runtime_config, device)
    if not source_matches(runtime.source, run["source"]):
        raise ValueError("joint runtime uses a different frozen source checkpoint")
    runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.state.requires_grad_(False).eval()
    runtime.policy.eval()
    expected_probe = torch.randn(50, 32, generator=torch.Generator().manual_seed(int(run["config"]["observer"]["probe_seed"])))
    if not torch.equal(runtime.state.probe.cpu(), expected_probe):
        raise ValueError("checkpoint public probe differs from its declared seed")
    if runtime.lora.rank != 16 or len(runtime.lora.targets) != 38:
        raise ValueError("materialization must produce one complete 38-target rank16 LoRA")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    store = RawTeacherVideoStore(tuple(value.authority for value in tasks.values()), frame_stride=5)
    conditions = {}
    try:
        for row in rows:
            task = tasks[row["global_task_id"]]
            for episode in row["episodes"]:
                key = episode["condition_id"]
                if key not in conditions:
                    conditions[key] = _compile_condition(runtime, store, task, episode["teacher_demo_indices"], output, checkpoint_record)
                    print(json.dumps({"condition": key, "compiled": len(conditions),
                                      "frames": sum(video["sampled_frame_count"] for video in conditions[key]["teacher_videos"])}), flush=True)
    finally:
        store.close()
    lora_path = asset_root / read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]["lora_contract"]
    manifest = {"schema_version": BANK_SCHEMA, "kind": BANK_KIND, "status": "sealed",
                "arm": selection["arm"], "evaluation_role": selection["evaluation_role"], "selection": dict(selection),
                "asset_root": str(asset_root.resolve()), "source": runtime.source,
                "writer_checkpoint": checkpoint_record, "materialization_git": repository,
                "lora_contract": file_record(lora_path), "method": method_metadata(run),
                "tasks": rows, "conditions": list(conditions.values()), "single_complete_rank16": True,
                "information_wall": {"deployment_inputs": ["exact language", "ordered RGB videos", "original frame indices"],
                    "teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
                    "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
                    "writer_invocations_per_unique_condition": 1, "total_writer_invocations": len(conditions),
                    "outcome_dependent_video_selection": False, "shuffled_reversed_wrong_no_video": False}}
    path = output / "manifest.json"
    write_json_atomic(path, manifest)
    return path


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("development_train", "validation"), required=True)
    parser.add_argument("--task-ids", type=_integers, required=True)
    parser.add_argument("--k", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--arm", choices=("correct", "same_task_other"), default="correct")
    parser.add_argument("--selection-mode", choices=("fixed_per_task", "per_init_ordinal"), default="per_init_ordinal")
    parser.add_argument("--video-pool", type=_integers, default=tuple(range(50)))
    parser.add_argument("--fixed-videos-json", type=Path)
    parser.add_argument("--state-count", type=int, choices=(10, 50), default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if torch.device(args.device).type == "cuda":
        from ember.writer.topology import bind_current_process_to_cuda_numa

        torch.cuda.set_device(torch.device(args.device))
        if not bind_current_process_to_cuda_numa(torch.cuda.current_device()):
            raise ValueError("materialization requires GPU-local NUMA placement")
        torch.backends.cuda.matmul.allow_tf32 = True
    selection = selection_contract(role=args.role, task_ids=args.task_ids, cardinality=args.k,
        arm=args.arm, mode=args.selection_mode, seed=args.seed, init_state_ids=tuple(range(args.state_count)),
        video_pool=args.video_pool, fixed_videos=read_json(args.fixed_videos_json) if args.fixed_videos_json else None)
    print(materialize(asset_root=args.asset_root.resolve(), checkpoint=args.checkpoint.resolve(),
                      output=args.output, selection=selection, device=torch.device(args.device)), flush=True)
