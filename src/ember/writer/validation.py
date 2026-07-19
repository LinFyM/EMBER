"""Eight-rank cross-category validation for the direct complete-LoRA Writer."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy
from ember.gate_zero_oracle_report_runtime import _closed_loop_metrics
from ember.gate_zero_runtime import build_lora_config, set_global_seed
from ember.gate_zero_task_local_rl.runtime import scoped_policy_execution_horizon
from ember.writer.core import (
    _validate_writer_checkpoint,
    load_writer_contract,
    sha256_file,
)
from ember.writer.data import WriterSpecAuthority
from ember.writer.direct_fit import direct_final_path, fit_direct_lora
from ember.writer.train import (
    _encode_spec_feature,
    _lora_targets,
    _paths,
    _writer_model,
    repository_root,
)
from ember.writer.validation_contract import (
    WriterValidationError,
    aggregate_validation_rows,
    load_validation_contract,
    require,
    validation_work_for_rank,
)


@dataclass(frozen=True)
class ParallelContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def primary(self) -> bool:
        return self.rank == 0

def _parallel(spec: Mapping[str, Any]) -> ParallelContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != spec["parallel"]["world_size"]:
        raise WriterValidationError("Writer validation requires eight ranks")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        "gloo",
        init_method="env://",
        timeout=timedelta(hours=3),
    )
    return ParallelContext(rank, local_rank, world_size)

def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WriterValidationError(f"invalid JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise WriterValidationError(f"invalid JSON authority: {path}")
    return value

def _authorities(
    spec: Mapping[str, Any], *, output_root: Path, data_root: Path
) -> dict[int, WriterSpecAuthority]:
    manifest = output_root / spec["authority"]["canonical_manifest_relative_path"]
    require(
        sha256_file(manifest),
        spec["authority"]["canonical_manifest_sha256"],
        "manifest SHA256",
    )
    records = _json(manifest).get("tasks", [])
    by_id = {row.get("task_index"): row for row in records if isinstance(row, dict)}
    dataset_root = data_root / spec["authority"]["dataset_relative_path"]
    result = {}
    for task_id in spec["evaluation"]["task_ids"]:
        row = by_id.get(task_id)
        if not isinstance(row, dict) or row.get("split") != "validation":
            raise WriterValidationError(f"task {task_id} left validation split")
        hdf5 = row["hdf5"]
        result[task_id] = WriterSpecAuthority(
            task_id,
            row["language"],
            dataset_root / hdf5["filename"],
            hdf5["bytes"],
            hdf5["sha256"],
        )
    return result

def _source_checkpoint(spec: Mapping[str, Any], output_root: Path) -> Path:
    path = (
        output_root
        / spec["authority"]["source_base_output_relative_path"]
        / "checkpoints"
        / f"{spec['authority']['source_base_checkpoint_step']:06d}"
    )
    require(
        sha256_file(path / "ember_checkpoint_manifest.json"),
        spec["authority"]["source_base_checkpoint_manifest_sha256"],
        "source-base checkpoint",
    )
    return path

def _wrap_lora(
    policy: Any,
    *,
    writer_spec: Mapping[str, Any],
    targets: Sequence[str],
) -> tuple[Any, dict[str, torch.Tensor]]:
    lora = writer_spec["lora"]
    policy = policy.wrap_with_peft(
        peft_config=build_lora_config(
            targets=targets,
            rank=lora["rank"],
            alpha=lora["alpha"],
            dropout=lora["dropout"],
            init_lora_weights="gaussian",
            base_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        )
    )
    state = {name: value for name, value in policy.named_parameters() if ".lora_" in name}
    require(
        sum(value.numel() for value in state.values()),
        lora["expected_parameter_count"],
        "evaluation LoRA capacity",
    )
    return policy, state

def _restore_lora(
    target: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]
) -> None:
    if set(target) != set(state):
        raise WriterValidationError("evaluation LoRA parameter identity changed")
    with torch.no_grad():
        for name, value in target.items():
            source = state[name]
            if value.shape != source.shape or value.dtype != source.dtype:
                raise WriterValidationError(f"evaluation LoRA metadata changed: {name}")
            value.copy_(source.to(value.device))

def _writer_lora(
    policy: Any,
    *,
    lora_state: Mapping[str, torch.Tensor],
    authority: WriterSpecAuthority,
    writer_spec: Mapping[str, Any],
    writer_checkpoint: Path,
) -> None:
    template = {
        name: value.detach().cpu().to(torch.float32) for name, value in lora_state.items()
    }
    writer = _writer_model(writer_spec, template, next(policy.parameters()).device)
    incompatible = writer.load_state_dict(
        load_file(writer_checkpoint / "writer.safetensors"), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise WriterValidationError("Writer checkpoint model is incomplete")
    writer.eval()
    bounds = writer_spec["data"]["writer_spec_episode_bounds"]
    feature = _encode_spec_feature(
        policy,
        authority,
        demo_indices=list(range(bounds[0], bounds[1] + 1)),
        positions=writer_spec["data"]["teaching_video_frame_positions"],
    )
    with torch.inference_mode():
        generated = {
            key: value[0].detach()
            for key, value in writer(feature.to(next(writer.parameters()).device)[None]).items()
        }
    _restore_lora(lora_state, generated)

def _open_runtime(
    *,
    task_id: int,
    arm: str,
    authority: WriterSpecAuthority,
    writer_spec: Mapping[str, Any],
    source_checkpoint: Path,
    mature_path: Path,
    writer_checkpoint: Path,
    output_dir: Path,
) -> tuple[Any, ...]:
    set_global_seed(writer_spec["train"]["seed"])
    runtime = list(
        _load_policy(
            source_checkpoint / "pretrained_model",
            {"task_suite": "libero_90", "task_id": task_id},
        )
    )
    if arm == "frozen_base":
        return tuple(runtime)
    policy, lora_state = _wrap_lora(
        runtime[0], writer_spec=writer_spec, targets=_lora_targets(mature_path)
    )
    if arm == "matched_direct_task_local_lora":
        final = direct_final_path(output_dir / "direct_lora" / f"task_{task_id:03d}")
        manifest = _json(final / "manifest.json")
        state_path = final / "trainable_state.safetensors"
        require(sha256_file(state_path), manifest["state_sha256"], "direct final")
        _restore_lora(lora_state, load_file(state_path))
    elif arm == "writer_cold_start":
        _writer_lora(
            policy,
            lora_state=lora_state,
            authority=authority,
            writer_spec=writer_spec,
            writer_checkpoint=writer_checkpoint,
        )
    else:
        raise WriterValidationError(f"unknown validation arm: {arm}")
    policy.eval()
    runtime[0] = policy
    return tuple(runtime)

def _episode_rows(
    result: Mapping[str, Any],
    *,
    task_id: int,
    task_category: str,
    arm: str,
    horizon: int,
    policy_rng_seed: int,
) -> list[dict[str, Any]]:
    values = zip(
        result["seeds"],
        result["official_rollout_init_state_indices"],
        result["successes"],
        result["sum_rewards"],
        result["max_rewards"],
        result["episode_steps"],
        result["time_to_success"],
        strict=True,
    )
    return [
        {
            "surface": "validation",
            "task_id": task_id,
            "task_category": task_category,
            "arm": arm,
            "execution_horizon": horizon,
            "policy_rng_seed": policy_rng_seed,
            "evaluator_seed": seed,
            "physical_init_state_index": init_state,
            "success": bool(success),
            "sum_reward": float(reward),
            "max_reward": float(maximum),
            "episode_steps": int(steps),
            "time_to_success": time_to_success,
        }
        for seed, init_state, success, reward, maximum, steps, time_to_success in values
    ]

def _rollout_spec(evaluation: Mapping[str, Any], policy_seed: int, retain: bool) -> dict[str, Any]:
    return {
        "report": {
            "rollout_batch_size": evaluation["rollouts_per_policy_seed"],
            "official_rollout_init_state_indices": evaluation["physical_init_state_indices"],
            "seed_start": evaluation["evaluator_seed_start"],
            "warmup_seed_start": evaluation["warmup_seed_start"],
            "policy_rng_seed": policy_seed,
        },
        "resources": {
            "retain_one_video_per_report_arm": retain,
            "return_episode_data": True,
        },
    }


def _evaluate_arm(
    *,
    task_id: int,
    arm: str,
    authority: WriterSpecAuthority,
    spec: Mapping[str, Any],
    writer_spec: Mapping[str, Any],
    source_checkpoint: Path,
    mature_path: Path,
    writer_checkpoint: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    runtime = _open_runtime(
        task_id=task_id,
        arm=arm,
        authority=authority,
        writer_spec=writer_spec,
        source_checkpoint=source_checkpoint,
        mature_path=mature_path,
        writer_checkpoint=writer_checkpoint,
        output_dir=output_dir,
    )
    evaluation = spec["evaluation"]
    rows: list[dict[str, Any]] = []
    videos: list[str] = []
    try:
        for horizon in evaluation["execution_horizons"]:
            for policy_seed in evaluation["policy_rng_seeds"]:
                retain = (
                    policy_seed == evaluation["retain_video_policy_seed"]
                    and horizon == evaluation["retain_video_horizon"]
                )
                condition = f"{arm}_h{horizon}_p{policy_seed}"
                with scoped_policy_execution_horizon(
                    runtime[0], execution_horizon=horizon, expected_model_chunk_size=50
                ):
                    result = _closed_loop_metrics(
                        runtime=runtime,
                        task_id=task_id,
                        condition=condition,
                        language=authority.language,
                        spec=_rollout_spec(evaluation, policy_seed, retain),
                        output_dir=output_dir,
                    )
                if result["mechanics_valid"] is not True:
                    raise WriterValidationError("validation rollout mechanics failed")
                rows.extend(
                    _episode_rows(
                        result,
                        task_id=task_id,
                        task_category=evaluation["task_categories"][str(task_id)],
                        arm=arm,
                        horizon=horizon,
                        policy_rng_seed=policy_seed,
                    )
                )
                videos.extend(result["video_paths"])
    finally:
        del runtime
        torch.cuda.empty_cache()
    return rows, videos


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_shard(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = _json(path)
    return list(payload["rows"]), list(payload["videos"])


def _checksums(output_dir: Path) -> None:
    files = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.name != "checksums.sha256"
        and not path.name.startswith("gpu_telemetry_")
        and "recovery" not in path.parts
    ]
    (output_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def _prepare_output(output_dir: Path, *, resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "writer_cold_start_validation_result.json").exists():
        raise WriterValidationError("validation output is already complete")
    if not resume:
        unexpected = [
            path.name
            for path in output_dir.iterdir()
            if not (path.is_file() and path.name.startswith("gpu_telemetry_"))
        ]
        if unexpected:
            raise WriterValidationError(f"refusing non-fresh validation output: {unexpected}")
    (output_dir / "shards").mkdir(exist_ok=resume)


def _eval_info(
    spec: Mapping[str, Any], aggregate: Mapping[str, Any], videos: Sequence[str]
) -> dict[str, Any]:
    evaluation = spec["evaluation"]
    video_by_arm: dict[tuple[int, str], list[str]] = {}
    for relative in videos:
        parts = Path(relative).parts
        task_id = int(parts[1].split("_")[-1])
        condition = parts[2]
        arm = next(value for value in evaluation["arms"] if condition.startswith(value))
        video_by_arm.setdefault((task_id, arm), []).append(relative)
    per_task = []
    primary_horizon = str(evaluation["primary_execution_horizon"])
    for task_id in evaluation["task_ids"]:
        for arm in evaluation["arms"]:
            cell = aggregate["per_task"][str(task_id)][arm][primary_horizon]
            per_task.append(
                {
                    "task_group": f"validation:{arm}",
                    "task_id": task_id,
                    "metrics": {
                        "successes": [True] * cell["successes"]
                        + [False] * (cell["episodes"] - cell["successes"]),
                        "sum_rewards": [],
                        "video_paths": video_by_arm.get((task_id, arm), []),
                    },
                }
            )
    return {
        "overall": {
            "status": "writer_cold_start_validation_completed",
            "surface": "validation_only",
            "episodes": aggregate["raw_episode_rows"],
            "primary_horizon": evaluation["primary_execution_horizon"],
        },
        "per_task": per_task,
    }


def _publish_result(
    *,
    spec: Mapping[str, Any],
    rows: list[dict[str, Any]],
    videos: list[str],
    output_dir: Path,
    writer_checkpoint: Path,
    contract_path: Path,
    wall_seconds: float,
) -> dict[str, Any]:
    evaluation = spec["evaluation"]
    aggregate = aggregate_validation_rows(
        rows,
        task_ids=evaluation["task_ids"],
        arms=evaluation["arms"],
        horizons=evaluation["execution_horizons"],
        expected_rollouts=evaluation["rollouts_per_task_arm"],
        bootstrap_seed=evaluation["paired_bootstrap_seed"],
        bootstrap_replicates=evaluation["paired_bootstrap_replicates"],
    )
    writer_manifest = _json(writer_checkpoint / "writer_checkpoint_manifest.json")
    writer_stage = _json(
        writer_checkpoint.parents[1] / "writer_cold_start_stage_result.json"
    )
    direct_manifests = {
        str(task_id): _json(
            direct_final_path(output_dir / "direct_lora" / f"task_{task_id:03d}")
            / "manifest.json"
        )
        for task_id in evaluation["task_ids"]
    }
    result = {
        "schema_version": 1,
        "status": "writer_cold_start_validation_completed",
        "surface": "validation_only",
        "validation_contract_sha256": sha256_file(contract_path),
        "writer_checkpoint": {
            "step": spec["authority"]["writer_checkpoint_step"],
            "writer_state_sha256": sha256_file(writer_checkpoint / "writer.safetensors"),
        },
        "training": {
            "writer": {
                "source_tasks": 60,
                "functional_episode_bounds": [8, 39],
                "completed_step": writer_manifest["step"],
                "consumed_query_frames": writer_manifest["sampler"]["consumed_query_frames"],
                "wall_seconds": writer_stage["wall_seconds"],
                "environment_interactions": 0,
            },
            "matched_direct_task_local_lora": {
                "validation_tasks": evaluation["task_ids"],
                "support_episode_bounds": spec["direct_baseline"]["support_episode_bounds"],
                "per_task": direct_manifests,
                "environment_interactions": 0,
                "information_role": spec["direct_baseline"]["role"],
            },
        },
        "evaluation": {
            "task_ids": evaluation["task_ids"],
            "task_categories": evaluation["task_categories"],
            "policy_rng_seeds": evaluation["policy_rng_seeds"],
            "physical_init_state_indices": evaluation["physical_init_state_indices"],
            "rollouts_per_task_arm": evaluation["rollouts_per_task_arm"],
            "execution_horizons": evaluation["execution_horizons"],
        },
        "aggregate": aggregate,
        "task_categories": evaluation["task_categories"],
        "resources": {
            "cuda_visible_physical_gpus": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"
            ),
            "world_size": spec["parallel"]["world_size"],
            "wall_seconds": wall_seconds,
        },
        "test_held_accessed": False,
    }
    _atomic_json(output_dir / "writer_cold_start_validation_result.json", result)
    with (output_dir / "episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(output_dir / "eval_info.json", _eval_info(spec, aggregate, videos))
    build_eval_gallery(output_dir)
    _checksums(output_dir)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    spec = load_validation_contract(args.config, repo_root=root)
    paths = _paths(root)
    writer_spec = load_writer_contract(
        root / spec["authority"]["writer_contract_relative_path"],
        phase0_path=paths["phase0"],
        split_path=paths["split"],
        gate_zero_path=paths["gate_zero"],
        mature_lora_path=paths["mature"],
    )
    context = _parallel(spec)
    started = time.perf_counter()
    try:
        if context.primary:
            _prepare_output(args.output_dir, resume=args.resume)
        torch.distributed.barrier()
        authorities = _authorities(spec, output_root=args.output_root, data_root=args.data_root)
        source_checkpoint = _source_checkpoint(spec, args.output_root)
        require(
            int(args.writer_checkpoint.name),
            spec["authority"]["writer_checkpoint_step"],
            "Writer checkpoint step",
        )
        _validate_writer_checkpoint(args.writer_checkpoint, world_size=8)
        work = validation_work_for_rank(spec, rank=context.rank, world_size=context.world_size)
        task_id = work["direct_fit_task"]
        if task_id is not None:
            fit_direct_lora(
                spec=spec,
                writer_spec=writer_spec,
                authority=authorities[task_id],
                source_checkpoint=source_checkpoint,
                mature_path=paths["mature"],
                output_dir=args.output_dir,
                validation_contract_sha256=sha256_file(args.config),
            )
        local_rows, local_videos = [], []
        for task_id, arm in work["evaluation_arms"]:
            shard_path = args.output_dir / "shards" / f"task_{task_id:03d}_{arm}.json"
            if shard_path.exists():
                if not args.resume:
                    raise WriterValidationError("validation arm shard already exists")
                rows, videos = _read_shard(shard_path)
            else:
                rows, videos = _evaluate_arm(
                    task_id=task_id,
                    arm=arm,
                    authority=authorities[task_id],
                    spec=spec,
                    writer_spec=writer_spec,
                    source_checkpoint=source_checkpoint,
                    mature_path=paths["mature"],
                    writer_checkpoint=args.writer_checkpoint,
                    output_dir=args.output_dir,
                )
                _atomic_json(shard_path, {"rows": rows, "videos": videos})
                print(json.dumps({
                    "event": "writer_validation_arm_complete",
                    "rank": context.rank,
                    "task_id": task_id,
                    "arm": arm,
                    "episodes": len(rows),
                    "performance_withheld_until_complete": True,
                }, sort_keys=True), flush=True)
            local_rows.extend(rows)
            local_videos.extend(videos)
        gathered: list[Any] | None = [None] * context.world_size if context.primary else None
        torch.distributed.gather_object(
            {"rows": local_rows, "videos": local_videos}, gathered, dst=0
        )
        if not context.primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        assert gathered is not None
        rows = [row for shard in gathered for row in shard["rows"]]
        videos = [video for shard in gathered for video in shard["videos"]]
        result = _publish_result(
            spec=spec,
            rows=rows,
            videos=videos,
            output_dir=args.output_dir,
            writer_checkpoint=args.writer_checkpoint,
            contract_path=args.config,
            wall_seconds=time.perf_counter() - started,
        )
        update_latest_link(
            args.output_dir, args.output_root / "writer_cold_start" / "validation_latest"
        )
        return result
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--writer-checkpoint", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    for name in ("config", "output_root", "data_root", "output_dir", "writer_checkpoint"):
        if not getattr(args, name).is_absolute():
            raise WriterValidationError(f"--{name.replace('_', '-')} must be absolute")
    result = run(args)
    if result.get("status") != "non_primary_rank_complete":
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
