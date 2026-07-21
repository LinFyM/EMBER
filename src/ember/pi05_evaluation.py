"""Generic pi0.5 feasibility evaluation on the sealed LIBERO 24/8/8 protocol."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ember.libero_evaluation import sha256_file
from ember.pi05_assets import (
    Pi05EvaluationError,
    load_protocol,
    prepare_libero_config,
    write_json_atomic,
)
from ember.pi05_processing import Pi05LiberoProcessor


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, text=True, capture_output=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty_paths": run("status", "--porcelain").splitlines(),
    }


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return quat[:3] * (2.0 * np.arccos(quat[3]) / den)


def _policy_input(obs: dict[str, Any], language: str) -> dict[str, Any]:
    import torch

    def image(value: np.ndarray) -> torch.Tensor:
        value = np.ascontiguousarray(value[::-1, ::-1])
        return torch.from_numpy(value).permute(2, 0, 1).float().div_(255.0)

    base = image(obs["agentview_image"])
    wrist = image(obs["robot0_eye_in_hand_image"])
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return {
        "observation.images.base_0_rgb": base,
        "observation.images.left_wrist_0_rgb": wrist,
        "observation.images.right_wrist_0_rgb": torch.zeros_like(base),
        "observation.state": torch.from_numpy(state),
        "task": language,
    }


def _load_policy(
    model_path: Path, stats: dict[str, Any], tokenizer_path: Path, device: str
) -> tuple[Any, Any, Any]:
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION

    config = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(config, PI05Config):
        raise Pi05EvaluationError("generic checkpoint did not resolve to PI05Config")
    config.device = device
    config.n_action_steps = 10
    config.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    policy = PI05Policy.from_pretrained(
        model_path, config=config, local_files_only=True, strict=True
    ).to(device).eval()
    processor = Pi05LiberoProcessor(
        stats, tokenizer_path, config.tokenizer_max_length, device
    )
    torch.set_grad_enabled(False)
    return policy, processor, processor.unnormalize_action


def _select_test_task(protocol: dict[str, Any], suite: str, task_id: int) -> dict[str, Any]:
    matches = [
        row for row in protocol["test_tasks"] if row["suite"] == suite and row["task_id"] == task_id
    ]
    if len(matches) != 1:
        raise Pi05EvaluationError(f"{suite} task {task_id} is not a sealed test task")
    return matches[0]


def _validate_held_action_isolation(
    protocol: dict[str, Any], normalization: dict[str, Any]
) -> None:
    mapping = normalization.get("local_to_global_task_ids", {})
    for role, count in (("validation", 8), ("test", 8)):
        expected = sorted(
            int(mapping[suite][str(task_id)])
            for suite, roles in protocol["split"]["suites"].items()
            for task_id in roles[role]
        )
        observed = sorted(normalization.get(f"{role}_global_task_ids_not_read", []))
        if len(expected) != count or observed != expected:
            raise Pi05EvaluationError(f"normalization does not prove {role} action isolation")


def evaluate_test_task(
    *,
    repo_root: Path,
    protocol_path: Path,
    normalization_path: Path,
    model_path: Path,
    model_manifest_path: Path,
    tokenizer_path: Path,
    tokenizer_manifest_path: Path,
    suite_name: str,
    task_id: int,
    output_dir: Path,
    episode_limit: int | None = None,
    env_count: int | None = None,
) -> None:
    protocol = load_protocol(protocol_path)
    task_contract = _select_test_task(protocol, suite_name, task_id)
    feasibility = protocol["pi05_feasibility"]
    expected_episodes = int(feasibility["num_trials_per_task"])
    episode_count = expected_episodes if episode_limit is None else int(episode_limit)
    formal_env_count = int(feasibility["envs_per_policy_process"])
    active_env_count = formal_env_count if env_count is None else int(env_count)
    if not 1 <= episode_count <= expected_episodes:
        raise Pi05EvaluationError("episode limit is outside 1..50")
    if not 1 <= active_env_count <= episode_count:
        raise Pi05EvaluationError("environment count is outside 1..episode count")
    if episode_limit is None and active_env_count != formal_env_count:
        raise Pi05EvaluationError("formal evaluation requires the sealed environment count")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise Pi05EvaluationError(f"output directory is not empty: {output_dir}")
    git = _git_state(repo_root)
    if episode_limit is None and git["dirty_paths"]:
        raise Pi05EvaluationError("formal evaluation requires a clean Git worktree")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    weights = model_path / model_manifest["weights_filename"]
    if weights.stat().st_size != int(model_manifest["weights_bytes"]):
        raise Pi05EvaluationError("model weight size differs from the sealed manifest")
    tokenizer_manifest = json.loads(tokenizer_manifest_path.read_text(encoding="utf-8"))
    if tokenizer_path.stat().st_size != int(tokenizer_manifest["bytes"]):
        raise Pi05EvaluationError("tokenizer size differs from the sealed manifest")
    if sha256_file(tokenizer_path) != tokenizer_manifest["sha256"]:
        raise Pi05EvaluationError("tokenizer hash differs from the sealed manifest")
    paths = prepare_libero_config(output_dir / "libero_config")
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if len(visible_devices) != 1 or not visible_devices[0].strip().isdigit():
        raise Pi05EvaluationError(
            "each evaluator requires one numeric physical CUDA_VISIBLE_DEVICES entry"
        )
    os.environ["MUJOCO_EGL_DEVICE_ID"] = visible_devices[0].strip()
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Pi05EvaluationError("each evaluator must see exactly one CUDA device")
    torch.cuda.set_device(0)
    seed = int(feasibility["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if normalization.get("protocol_sha256") != sha256_file(protocol_path):
        raise Pi05EvaluationError("normalization does not belong to the sealed protocol")
    _validate_held_action_isolation(protocol, normalization)
    policy, preprocess, postprocess = _load_policy(
        model_path, normalization["stats"], tokenizer_path, "cuda:0"
    )
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    if task.language != task_contract["language"]:
        raise Pi05EvaluationError("installed LIBERO task language differs from sealed contract")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_path = (
        Path(get_libero_path("init_states"))
        / suite_name
        / task_contract["init_states_file"]
    )
    init_states = suite.get_task_init_states(task_id)
    if sha256_file(bddl_path) != task_contract["bddl_sha256"]:
        raise Pi05EvaluationError("installed BDDL differs from sealed contract")
    if sha256_file(init_path) != task_contract["init_states_sha256"]:
        raise Pi05EvaluationError("installed fixed init states differ from sealed contract")
    if len(init_states) < expected_episodes:
        raise Pi05EvaluationError("installed task has fewer than 50 fixed init states")
    envs = [
        OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=int(feasibility["render_resolution"]),
            camera_widths=int(feasibility["render_resolution"]),
        )
        for _ in range(active_env_count)
    ]
    try:
        for env in envs:
            env.seed(seed)
        rows = _rollout_rows(
            envs,
            init_states,
            task_contract,
            feasibility,
            policy,
            preprocess,
            postprocess,
            episode_count,
        )
    finally:
        for env in envs:
            env.close()
    result = {
        "schema_version": 1,
        "arm": "generic_pi05_base_zero_shot",
        "suite": suite_name,
        "task_id": task_id,
        "language": task.language,
        "successes": sum(bool(row["success"]) for row in rows),
        "episodes": len(rows),
        "rows": rows,
        "protocol_sha256": sha256_file(protocol_path),
        "normalization_sha256": sha256_file(normalization_path),
        "model_path": str(model_path.resolve()),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "model_sha256": model_manifest["weights_sha256"],
        "model_revision": model_manifest["model_revision"],
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest_path),
        "tokenizer_sha256": tokenizer_manifest["sha256"],
        "git": git,
        "libero_paths": paths,
        "runtime_seconds": sum(float(row["wall_seconds"]) for row in rows),
        "wall_clock_seconds": max(float(row["finished_at"]) for row in rows),
        "envs_per_policy_process": active_env_count,
    }
    write_json_atomic(output_dir / "results.json", result)


def _rollout_rows(
    envs: list[Any],
    init_states: Any,
    task: dict[str, Any],
    config: dict[str, Any],
    policy: Any,
    preprocess: Any,
    postprocess: Any,
    episode_count: int,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    dummy = np.asarray(config["dummy_action"], dtype=np.float32)
    max_steps = int(config["max_steps"][task["suite"]])
    replan_steps = int(config["replan_steps"])
    evaluation_start = time.monotonic()

    def start_episode(env: Any, episode_index: int) -> dict[str, Any]:
        env.reset()
        obs = env.set_init_state(init_states[episode_index])
        for _ in range(int(config["num_steps_wait"])):
            obs, _, _, _ = env.step(dummy)
        return {
            "episode_index": episode_index,
            "obs": obs,
            "steps": 0,
            "action_plan": deque(),
            "start": time.monotonic(),
        }

    policy.reset()
    next_episode = min(len(envs), episode_count)
    slots: list[dict[str, Any] | None] = [
        start_episode(env, episode_index)
        for env, episode_index in zip(envs, range(next_episode), strict=False)
    ]
    while any(slot is not None for slot in slots):
        planning_slots = [slot for slot in slots if slot is not None and not slot["action_plan"]]
        if planning_slots:
            processed = [
                preprocess(_policy_input(slot["obs"], task["language"]))
                for slot in planning_slots
            ]
            batch = {
                key: torch.cat([item[key] for item in processed], dim=0)
                for key in processed[0]
                if isinstance(processed[0][key], torch.Tensor)
            }
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)
                actions = postprocess(chunk).detach().cpu().numpy()
            for slot, plan in zip(planning_slots, actions, strict=True):
                slot["action_plan"].extend(plan[:replan_steps])

        for slot_index, (env, slot) in enumerate(zip(envs, slots, strict=True)):
            if slot is None:
                continue
            obs, _, done, _ = env.step(slot["action_plan"].popleft())
            slot["obs"] = obs
            slot["steps"] += 1
            if not bool(done) and slot["steps"] < max_steps:
                continue
            finished = time.monotonic()
            rows.append(
                {
                    "suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "language": task["language"],
                    "init_state_id": int(slot["episode_index"]),
                    "env_seed": int(config["seed"]),
                    "policy_seed": int(config["seed"]),
                    "success": bool(done),
                    "steps": int(slot["steps"]),
                    "wall_seconds": finished - float(slot["start"]),
                    "finished_at": finished - evaluation_start,
                }
            )
            if next_episode < episode_count:
                slots[slot_index] = start_episode(env, next_episode)
                next_episode += 1
            else:
                slots[slot_index] = None
    return sorted(rows, key=lambda row: row["init_state_id"])


def aggregate_results(protocol_path: Path, input_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    expected = {(row["suite"], int(row["task_id"])) for row in protocol["test_tasks"]}
    result_paths = sorted(input_root.rglob("results.json"))
    results = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in result_paths]
    actual = {(row["suite"], int(row["task_id"])) for _, row in results}
    if actual != expected or len(results) != len(expected):
        raise Pi05EvaluationError(f"incomplete task results: expected={expected} actual={actual}")
    all_rows = [episode for _, result in results for episode in result["rows"]]
    keys = {(row["suite"], int(row["task_id"]), int(row["init_state_id"])) for row in all_rows}
    if len(all_rows) != 400 or len(keys) != 400:
        raise Pi05EvaluationError("formal aggregate requires 400 unique episode rows")
    per_task = sorted(
        (
            {
                "suite": result["suite"],
                "task_id": result["task_id"],
                "language": result["language"],
                "successes": result["successes"],
                "episodes": result["episodes"],
                "success_rate": result["successes"] / result["episodes"],
                "results_sha256": sha256_file(path),
            }
            for path, result in results
        ),
        key=lambda row: (row["suite"], row["task_id"]),
    )
    successes = sum(int(row["success"]) for row in all_rows)
    write_json_atomic(
        output_path,
        {
            "schema_version": 1,
            "arm": "generic_pi05_base_zero_shot",
            "protocol_sha256": sha256_file(protocol_path),
            "per_task": per_task,
            "overall": {"successes": successes, "episodes": 400, "success_rate": successes / 400},
            "rows": sorted(all_rows, key=lambda row: (row["suite"], row["task_id"], row["init_state_id"])),
        },
    )
