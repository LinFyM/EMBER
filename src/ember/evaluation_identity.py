"""Gate -1 probe for LIBERO seed, init-state, observation, and policy identity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ember.identity_evidence import (
    IdentityProbeError,
    as_numpy,
    canonical_tree_summary,
    compare_trees,
    load_probe_spec,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _slice_batch(tree: Any, index: int, batch_size: int) -> Any:
    if isinstance(tree, Mapping):
        return {key: _slice_batch(value, index, batch_size) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_slice_batch(value, index, batch_size) for value in tree)
    if isinstance(tree, list):
        return [_slice_batch(value, index, batch_size) for value in tree]
    array = as_numpy(tree)
    if array is not None and array.ndim and array.shape[0] == batch_size:
        return np.array(array[index], copy=True)
    return copy.deepcopy(tree)


def _repeat_batch(tree: Any, batch_size: int) -> Any:
    if isinstance(tree, Mapping):
        return {key: _repeat_batch(value, batch_size) for key, value in tree.items()}
    array = as_numpy(tree)
    if array is not None:
        return np.repeat(np.expand_dims(array, axis=0), batch_size, axis=0)
    return copy.deepcopy(tree)


def _environment_config(spec: Mapping[str, Any]) -> Any:
    from lerobot.envs.configs import LiberoEnv

    return LiberoEnv(
        task=spec["task_suite"],
        task_ids=[spec["task_id"]],
        camera_name_mapping={
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
        control_mode="relative",
    )


def _make_condition_env(spec: Mapping[str, Any], condition: Mapping[str, Any]) -> Any:
    from lerobot.envs.factory import make_env

    all_envs = make_env(
        _environment_config(spec),
        n_envs=condition["batch_size"],
        use_async_envs=condition["mode"] == "async",
    )
    return all_envs[spec["task_suite"]][spec["task_id"]]


def _capture_mechanics_condition(
    spec: Mapping[str, Any], condition: Mapping[str, Any], identity_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    from lerobot.envs.libero import get_libero_dummy_action

    batch_size = condition["batch_size"]
    seeds = list(range(spec["seed_start"], spec["seed_start"] + batch_size))
    env = _make_condition_env(spec, condition)
    runtime: dict[int, dict[str, Any]] = {}
    try:
        before = list(env.call("init_state_id"))
        if before != list(range(batch_size)):
            raise IdentityProbeError(f"Unexpected initial init-state IDs for {condition['name']}: {before}")
        reset_observation, _ = env.reset(seed=seeds)
        after = list(env.call("init_state_id"))
        expected_after = [index + batch_size for index in range(batch_size)]
        if after != expected_after:
            raise IdentityProbeError(f"Unexpected reset stride for {condition['name']}: {after}")
        fixed_steps: list[Any] = []
        dummy = np.asarray(get_libero_dummy_action(), dtype=np.float32)
        for _ in range(spec["fixed_steps"]):
            observation, reward, terminated, truncated, _ = env.step(
                np.repeat(dummy[None, :], batch_size, axis=0)
            )
            fixed_steps.append(
                {
                    "observation": copy.deepcopy(observation),
                    "reward": np.array(reward, copy=True),
                    "terminated": np.array(terminated, copy=True),
                    "truncated": np.array(truncated, copy=True),
                }
            )
        episodes = []
        for index in range(batch_size):
            reset = _slice_batch(reset_observation, index, batch_size)
            fixed = [_slice_batch(step, index, batch_size) for step in fixed_steps]
            runtime[index] = {"reset": reset, "fixed_trajectory": fixed}
            episodes.append(
                {
                    **identity_rows[index],
                    "reset_observation": canonical_tree_summary(reset),
                    "fixed_trajectory": canonical_tree_summary(fixed),
                }
            )
        return {
            "name": condition["name"],
            "mode": condition["mode"],
            "batch_size": batch_size,
            "seeds": seeds,
            "init_state_ids_before_reset": before,
            "init_state_ids_after_reset": after,
            "episodes": episodes,
        }, runtime
    finally:
        env.close()


def _mechanics_comparisons(
    spec: Mapping[str, Any], runtime: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    comparisons = []
    for pair in spec["comparison_pairs"]:
        for index in pair["shared_indices"]:
            left = runtime[pair["left"]][index]
            right = runtime[pair["right"]][index]
            comparisons.append(
                {
                    "left": pair["left"],
                    "right": pair["right"],
                    "logical_index": index,
                    "reset_observation": compare_trees(
                        left["reset"],
                        right["reset"],
                        atol=spec["observation_atol"],
                        rtol=spec["observation_rtol"],
                    ),
                    "fixed_trajectory": compare_trees(
                        left["fixed_trajectory"],
                        right["fixed_trajectory"],
                        atol=spec["observation_atol"],
                        rtol=spec["observation_rtol"],
                    ),
                }
            )
    return comparisons


def _prepare_authority(spec: Mapping[str, Any], maximum_batch: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite, get_task_init_states

    suite = _get_suite(spec["task_suite"])
    task = suite.get_task(spec["task_id"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = get_task_init_states(suite, spec["task_id"])
    rows = []
    for index in range(maximum_batch):
        rows.append(
            {
                "logical_index": index,
                "seed": spec["seed_start"] + index,
                "init_state_index": index,
                "init_state_sha256": canonical_tree_summary(init_states[index])["sha256"],
            }
        )
    authority = {
        "task_name": task.name,
        "task_language": task.language,
        "bddl_path": str(bddl),
        "bddl_sha256": _sha256_file(bddl),
        "init_state_path": str(init_file),
        "init_state_file_sha256": _sha256_file(init_file),
        "init_state_count": len(init_states),
        "camera_name_mapping": {
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
        "control_mode": "relative",
    }
    return authority, rows


def _load_policy(policy_path: Path, spec: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    env_config = _environment_config(spec)
    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    policy_config.pretrained_path = policy_path
    policy_config.device = "cuda"
    policy_config.empty_cameras = 1
    policy = make_policy(cfg=policy_config, env_cfg=env_config, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=policy_config
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _select_policy_action(runtime: tuple[Any, Any, Any, Any, Any], observation: Any, tasks: list[str]) -> np.ndarray:
    import torch
    from lerobot.scripts.lerobot_eval import preprocess_observation
    from lerobot.utils.constants import ACTION

    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = runtime
    batch = preprocess_observation(copy.deepcopy(observation))
    batch["task"] = tasks
    batch = env_preprocessor(batch)
    batch = preprocessor(batch)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocessor(action)
    action = env_postprocessor({ACTION: action})[ACTION]
    return action.detach().cpu().numpy()


def _policy_batch_probe(
    spec: Mapping[str, Any], runtime: tuple[Any, Any, Any, Any, Any], canonical_observation: Any, language: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from lerobot.utils.random_utils import set_seed

    policy = runtime[0]
    records = []
    reference: np.ndarray | None = None
    first_nonexact = None
    first_outside = None
    for batch_size in spec["policy_batch_sizes"]:
        set_seed(spec["policy_rng_seed"])
        policy.reset()
        observation = _repeat_batch(canonical_observation, batch_size)
        action = _select_policy_action(runtime, observation, [language] * batch_size)
        first = np.array(action[0], copy=True)
        comparison = None
        if reference is None:
            reference = first
        else:
            comparison = compare_trees(
                reference, first, atol=spec["action_atol"], rtol=spec["action_rtol"]
            )
            if not comparison["exact"] and first_nonexact is None:
                first_nonexact = batch_size
            if not comparison["within_tolerance"] and first_outside is None:
                first_outside = batch_size
        records.append(
            {
                "batch_size": batch_size,
                "action": canonical_tree_summary(action),
                "first_action": canonical_tree_summary(first),
                "first_action_values": first.tolist(),
                "versus_batch_1": comparison,
            }
        )
        if first_outside is not None and spec["stop_policy_batch_after_first_tolerance_mismatch"]:
            break
    return records, {
        "first_nonexact_batch": first_nonexact,
        "first_outside_tolerance_batch": first_outside,
        "tested_batch_sizes": [record["batch_size"] for record in records],
    }


def _capture_policy_condition(
    spec: Mapping[str, Any], condition: Mapping[str, Any], runtime: tuple[Any, Any, Any, Any, Any], language: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    from lerobot.utils.random_utils import set_seed

    batch_size = condition["batch_size"]
    seeds = list(range(spec["seed_start"], spec["seed_start"] + batch_size))
    env = _make_condition_env(spec, condition)
    policy = runtime[0]
    try:
        observation, _ = env.reset(seed=seeds)
        set_seed(spec["policy_rng_seed"])
        policy.reset()
        actions = []
        observations = []
        outcomes = []
        for _ in range(spec["policy_steps"]):
            action = _select_policy_action(runtime, observation, [language] * batch_size)
            observation, reward, terminated, truncated, _ = env.step(action)
            actions.append(np.array(action, copy=True))
            observations.append(copy.deepcopy(observation))
            outcomes.append(
                {
                    "reward": np.array(reward, copy=True),
                    "terminated": np.array(terminated, copy=True),
                    "truncated": np.array(truncated, copy=True),
                }
            )
        per_index = {}
        episodes = []
        for index in range(batch_size):
            episode = {
                "actions": [_slice_batch(value, index, batch_size) for value in actions],
                "observations": [_slice_batch(value, index, batch_size) for value in observations],
                "outcomes": [_slice_batch(value, index, batch_size) for value in outcomes],
            }
            per_index[index] = episode
            episodes.append(
                {
                    "logical_index": index,
                    "seed": seeds[index],
                    "actions": canonical_tree_summary(episode["actions"]),
                    "observations": canonical_tree_summary(episode["observations"]),
                    "outcomes": canonical_tree_summary(episode["outcomes"]),
                }
            )
        return {"name": condition["name"], "episodes": episodes}, per_index
    finally:
        env.close()


def _policy_trajectory_comparisons(
    spec: Mapping[str, Any], records: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    comparisons = []
    for pair in spec["comparison_pairs"]:
        for index in pair["shared_indices"]:
            left = records[pair["left"]][index]
            right = records[pair["right"]][index]
            comparisons.append(
                {
                    "left": pair["left"],
                    "right": pair["right"],
                    "logical_index": index,
                    "actions": compare_trees(
                        left["actions"], right["actions"],
                        atol=spec["action_atol"], rtol=spec["action_rtol"],
                    ),
                    "observations": compare_trees(
                        left["observations"], right["observations"],
                        atol=spec["observation_atol"], rtol=spec["observation_rtol"],
                    ),
                    "outcomes": compare_trees(
                        left["outcomes"], right["outcomes"], atol=0.0, rtol=0.0
                    ),
                }
            )
    return comparisons


def _mechanics_stop_reason(spec: Mapping[str, Any], comparisons: list[dict[str, Any]]) -> str | None:
    if spec["stop_on_reset_mismatch"] and any(
        not item["reset_observation"]["within_tolerance"] for item in comparisons
    ):
        return "reset_observation_mismatch"
    if spec["stop_on_fixed_trajectory_mismatch"] and any(
        not item["fixed_trajectory"]["within_tolerance"] for item in comparisons
    ):
        return "fixed_trajectory_mismatch"
    return None


def _run_mechanics_layer(
    spec: Mapping[str, Any], identity_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    runtime = {}
    reports = []
    for condition in spec["env_conditions"]:
        report, condition_runtime = _capture_mechanics_condition(spec, condition, identity_rows)
        reports.append(report)
        runtime[condition["name"]] = condition_runtime
    comparisons = _mechanics_comparisons(spec, runtime)
    layer = {"conditions": reports, "comparisons": comparisons}
    return layer, runtime, _mechanics_stop_reason(spec, comparisons)


def _run_policy_layer(
    spec: Mapping[str, Any], policy_path: Path, mechanics_runtime: Mapping[str, Any], language: str
) -> dict[str, Any]:
    policy_runtime = _load_policy(policy_path, spec)
    canonical = mechanics_runtime["sync_b1"][0]["reset"]
    batch_report, batch_decision = _policy_batch_probe(
        spec, policy_runtime, canonical, language
    )
    trajectory_reports = []
    trajectory_runtime = {}
    for condition in spec["env_conditions"]:
        report, runtime = _capture_policy_condition(spec, condition, policy_runtime, language)
        trajectory_reports.append(report)
        trajectory_runtime[condition["name"]] = runtime
    return {
        "batch_probe": batch_report,
        "batch_decision": batch_decision,
        "trajectory_conditions": trajectory_reports,
        "trajectory_comparisons": _policy_trajectory_comparisons(spec, trajectory_runtime),
    }


def run_probe(
    spec_path: Path,
    policy_path: Path,
    output_dir: Path,
    physical_gpu: int,
    *,
    mechanics_only: bool = False,
) -> dict[str, Any]:
    """Run the predeclared mechanics and policy identity layers."""

    spec = load_probe_spec(spec_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "probe_result.json"
    if result_path.exists():
        raise IdentityProbeError(f"Refusing to overwrite completed probe: {result_path}")
    started = time.time()
    maximum_env_batch = max(item["batch_size"] for item in spec["env_conditions"])
    authority, identity_rows = _prepare_authority(spec, maximum_env_batch)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "surface": spec["surface"],
        "config_path": str(spec_path.resolve()),
        "config_sha256": _sha256_file(spec_path),
        "policy_path": str(policy_path.resolve()),
        "physical_gpu": physical_gpu,
        "spec": spec,
        "authority": authority,
        "mechanics": {},
        "policy": {},
        "started_unix": started,
    }
    mechanics, mechanics_runtime, stop_reason = _run_mechanics_layer(spec, identity_rows)
    result["mechanics"] = mechanics
    result["status"] = "stopped" if stop_reason else "mechanics_passed"
    result["stop_reason"] = stop_reason
    _atomic_json(result_path, result)
    if stop_reason:
        result["finished_unix"] = time.time()
        result["wall_seconds"] = result["finished_unix"] - started
        _atomic_json(result_path, result)
        return result
    if mechanics_only:
        result["status"] = "mechanics_completed"
        result["finished_unix"] = time.time()
        result["wall_seconds"] = result["finished_unix"] - started
        _atomic_json(result_path, result)
        return result

    result["policy"] = _run_policy_layer(
        spec, policy_path, mechanics_runtime, authority["task_language"]
    )
    result["status"] = "completed"
    result["stop_reason"] = None
    result["finished_unix"] = time.time()
    result["wall_seconds"] = result["finished_unix"] - started
    _atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Gate -1 evaluation-identity probe")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--mechanics-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run_probe(
            args.config,
            args.policy_path,
            args.output_dir,
            args.physical_gpu,
            mechanics_only=args.mechanics_only,
        )
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(args.output_dir / "failure_packet.json", failure)
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "stop_reason": result.get("stop_reason"),
                "result": str((args.output_dir / "probe_result.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {"completed", "mechanics_completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
