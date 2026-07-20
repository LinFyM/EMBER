#!/usr/bin/env python3
"""Eight-rank official LIBERO evaluation for a frozen source embodiment base."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ember.libero_evaluation import (
    EvaluationContractError,
    aggregate_rows,
    batched_with_padding,
    canonical_sha256,
    environment_seed,
    load_evaluation_config,
    partition_fixed_state_ids,
    policy_seed,
    resolve_role,
    sha256_file,
    validate_complete_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/source_base_eval_v1.json"
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument(
        "--role", choices=("source_development", "validation"), required=True
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {"commit": commit, "branch": branch, "dirty_paths": dirty}


def _prepare_libero_config(output_dir: Path) -> tuple[Path, dict[str, str]]:
    """Create a run-local non-interactive LIBERO path authority."""

    package = importlib.util.find_spec("libero")
    if package is None or package.origin is None:
        raise EvaluationContractError("installed LIBERO package cannot be located")
    benchmark_root = Path(package.origin).resolve().parent / "libero"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    for name in ("benchmark_root", "bddl_files", "init_states", "assets"):
        if not Path(paths[name]).exists():
            raise EvaluationContractError(f"LIBERO {name} path is missing: {paths[name]}")
    config_dir = output_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(config_dir / "config.yaml", paths)
    return config_dir, paths


def _set_fixed_init_states(env: Any, state_ids: tuple[int, ...]) -> None:
    if hasattr(env, "set_attr"):
        env.set_attr("init_state_id", list(state_ids))
        return
    if env.__class__.__name__ == "_LazyAsyncVectorEnv" and hasattr(env, "_ensure"):
        env._ensure()  # The pinned wrapper does not forward VectorEnv.set_attr.
        env._env.set_attr("init_state_id", list(state_ids))
        return
    raise EvaluationContractError(
        f"vector environment {type(env).__name__} cannot set exact fixed-state IDs"
    )


def _seed_policy(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _episode_rows(
    rollout_data: dict[str, torch.Tensor],
    *,
    valid_count: int,
    task_id: int,
    language: str,
    state_ids: tuple[int, ...],
    env_seeds: tuple[int, ...],
    policy_rng_seed: int,
    rank: int,
    batch_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(valid_count):
        done = rollout_data["done"][index].to(torch.int64)
        done_index = int(torch.argmax(done).item())
        success = bool(rollout_data["success"][index, : done_index + 1].any().item())
        reward = rollout_data["reward"][index, : done_index + 1]
        rows.append(
            {
                "task_id": task_id,
                "language": language,
                "init_state_id": state_ids[index],
                "env_seed": env_seeds[index],
                "policy_seed": policy_rng_seed,
                "rank": rank,
                "batch_index": batch_index,
                "success": success,
                "steps": done_index + 1,
                "sum_reward": float(reward.sum().item()),
                "max_reward": float(reward.max().item()),
            }
        )
    return rows


def main() -> int:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or not 0 <= local_rank < torch.cuda.device_count():
        raise EvaluationContractError("evaluation rank has no eligible CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    dist.init_process_group("gloo")

    config = load_evaluation_config(args.config.resolve(), REPO_ROOT)
    if world_size != int(config["parallel"]["world_size"]):
        raise EvaluationContractError("launch world size differs from sealed evaluation contract")
    manifest_path = REPO_ROOT / config["protocol"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role = resolve_role(config, manifest, args.role)
    task_ids = role.task_ids
    state_count = int(config["environment"]["fixed_init_state_count"])
    if args.mode == "smoke":
        task_ids = task_ids[: int(config["smoke"]["task_count"])]
        state_count = int(config["smoke"]["fixed_init_state_count"])
    assigned_states = partition_fixed_state_ids(state_count, world_size, rank)
    batches = batched_with_padding(
        assigned_states, int(config["parallel"]["envs_per_rank"])
    )
    language_by_task = {
        int(record["task_index"]): str(record["language"])
        for record in manifest["tasks"]
    }

    if rank == 0:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise EvaluationContractError(f"output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        required_policy_files = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        missing = [
            name
            for name in required_policy_files
            if not (args.policy_path / name).is_file()
        ]
        if missing:
            raise EvaluationContractError(f"source policy is incomplete: {missing}")
        git = _git_state()
        if args.mode == "formal" and git["dirty_paths"]:
            raise EvaluationContractError("formal evaluation requires a clean committed worktree")
        libero_config_dir, libero_paths = _prepare_libero_config(args.output_dir)
        contract = {
            "schema_version": "ember_source_base_eval_launch_v1",
            "mode": args.mode,
            "role": role.name,
            "required_split": role.required_split,
            "task_ids": list(task_ids),
            "fixed_init_state_count": state_count,
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config.resolve()),
            "policy_path": str(args.policy_path.resolve()),
            "policy_files": {
                name: sha256_file(args.policy_path / name) for name in required_policy_files
            },
            "git": git,
            "runtime": {
                "world_size": world_size,
                "envs_per_rank": config["parallel"]["envs_per_rank"],
                "one_policy_process_per_gpu": True,
                "task_synchronous": True,
            },
            "environment": config["environment"],
            "libero_paths": libero_paths,
            "policy": config["policy"],
            "rng": config["rng"],
        }
        contract["contract_sha256"] = canonical_sha256(contract)
        _write_json_atomic(args.output_dir / "run_contract.json", contract)
    dist.barrier()

    libero_config_dir = args.output_dir / "libero_config"
    if not (libero_config_dir / "config.yaml").is_file():
        raise EvaluationContractError("run-local LIBERO config was not created")
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_dir.resolve())

    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.scripts.lerobot_eval import rollout

    env_common = config["environment"]
    rename_map = dict(config["policy"]["rename_map"])
    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    if not isinstance(policy_config, SmolVLAConfig):
        raise EvaluationContractError("source checkpoint is not a SmolVLA policy")
    policy_config.device = str(device)
    policy_config.pretrained_path = args.policy_path
    policy_config.use_amp = config["policy"]["precision"] == "bfloat16"
    if policy_config.n_action_steps != int(config["policy"]["action_execution_horizon"]):
        raise EvaluationContractError("policy action horizon differs from sealed h50 evaluation")
    feature_env = LiberoEnv(
        task=env_common["suite"],
        task_ids=[task_ids[0]],
        episode_length=env_common["max_horizon"],
        obs_type="pixels_agent_pos",
        camera_name=env_common["camera_name"],
        init_states=True,
        observation_height=env_common["observation_height"],
        observation_width=env_common["observation_width"],
        control_mode=env_common["control_mode"],
    )
    policy = make_policy(policy_config, env_cfg=feature_env, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=feature_env, policy_cfg=policy_config
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    rows: list[dict[str, Any]] = []
    task_timings: list[dict[str, Any]] = []
    for task_id in task_ids:
        started = time.monotonic()
        env_config = LiberoEnv(
            task=env_common["suite"],
            task_ids=[task_id],
            episode_length=env_common["max_horizon"],
            obs_type="pixels_agent_pos",
            camera_name=env_common["camera_name"],
            init_states=True,
            observation_height=env_common["observation_height"],
            observation_width=env_common["observation_width"],
            control_mode=env_common["control_mode"],
        )
        envs = make_env(
            env_config,
            n_envs=int(config["parallel"]["envs_per_rank"]),
            use_async_envs=bool(config["parallel"]["use_async_envs"]),
        )
        env = envs[env_common["suite"]][task_id]
        try:
            for batch_index, (state_ids, valid_count) in enumerate(batches):
                _set_fixed_init_states(env, state_ids)
                env_seeds = tuple(
                    environment_seed(
                        int(config["rng"]["environment_seed_base"]), task_id, state_id
                    )
                    for state_id in state_ids
                )
                policy_rng_seed = policy_seed(
                    int(config["rng"]["policy_seed_base"]),
                    task_id,
                    batch_index,
                    rank,
                )
                _seed_policy(policy_rng_seed)
                precision_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if config["policy"]["precision"] == "bfloat16"
                    else nullcontext()
                )
                with torch.inference_mode(), precision_context:
                    rollout_data = rollout(
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        seeds=list(env_seeds),
                    )
                rows.extend(
                    _episode_rows(
                        rollout_data,
                        valid_count=valid_count,
                        task_id=task_id,
                        language=language_by_task[task_id],
                        state_ids=state_ids,
                        env_seeds=env_seeds,
                        policy_rng_seed=policy_rng_seed,
                        rank=rank,
                        batch_index=batch_index,
                    )
                )
        finally:
            close_envs(envs)
        task_timings.append(
            {
                "task_id": task_id,
                "rank": rank,
                "episodes": len(assigned_states),
                "wall_seconds": time.monotonic() - started,
            }
        )
        dist.barrier()

    rank_payload = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device": str(device),
        "egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
        "assigned_state_ids": list(assigned_states),
        "rows": rows,
        "task_timings": task_timings,
    }
    _write_json_atomic(args.output_dir / f"rank_{rank:02d}.json", rank_payload)
    dist.barrier()

    if rank == 0:
        all_rows: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        for source_rank in range(world_size):
            payload = json.loads(
                (args.output_dir / f"rank_{source_rank:02d}.json").read_text(encoding="utf-8")
            )
            all_rows.extend(payload["rows"])
            timings.extend(payload["task_timings"])
        all_rows = validate_complete_rows(all_rows, task_ids, state_count)
        result = {
            "schema_version": "ember_fresh_eval_results_v1",
            "mode": args.mode,
            "role": role.name,
            "task_ids": list(task_ids),
            "fixed_init_state_count": state_count,
            "rows": all_rows,
            "metrics": aggregate_rows(all_rows),
            "task_rank_timings": timings,
        }
        _write_json_atomic(args.output_dir / "results.json", result)
        print(json.dumps({"event": "complete", **result["metrics"]["overall"]}, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
