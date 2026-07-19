"""Gate -1 prompt/specification diagnostic on a pinned official-overlap surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ember.contracts import load_contract, validate_contract
from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy, _make_condition_env
from ember.libero_data import sha256_file


class SpecificationProbeError(RuntimeError):
    """Raised when the probe contract or matched evaluation mechanics are invalid."""


EXPECTED_CONDITIONS = ["correct", "no_spec", "scene_only", "swapped"]


class ResetAuditEnv:
    """Transparent vector-env proxy that records the reset performed upstream."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.reset_events: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        before = list(self._env.call("init_state_id"))
        result = self._env.reset(*args, **kwargs)
        after = list(self._env.call("init_state_id"))
        seeds = kwargs.get("seed")
        self.reset_events.append(
            {"before": before, "after": after, "seeds": None if seeds is None else list(seeds)}
        )
        return result


def _pair_map(spec: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for pair in spec.get("hard_negative_pairs", []):
        left, right = pair.get("left"), pair.get("right")
        if not isinstance(left, int) or not isinstance(right, int) or left == right:
            raise SpecificationProbeError("each hard-negative pair must contain two task IDs")
        if left in mapping or right in mapping:
            raise SpecificationProbeError("hard-negative pair members must be unique")
        mapping[left], mapping[right] = right, left
    return mapping


def _validate_surface(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise SpecificationProbeError("unsupported specification probe schema")
    if spec.get("surface") != "official_overlap_mechanics_only":
        raise SpecificationProbeError("pilot must remain on the official-overlap surface")
    task_ids = spec.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
        raise SpecificationProbeError("task IDs must be a non-empty unique list")
    if spec.get("conditions") != EXPECTED_CONDITIONS:
        raise SpecificationProbeError("the four prompt conditions are fixed")
    if spec.get("episodes_per_task") != spec.get("batch_size"):
        raise SpecificationProbeError("each arm must use one fixed batch")
    if not spec.get("use_async_envs"):
        raise SpecificationProbeError("pilot mode is predeclared as async")
    if not isinstance(spec.get("scene_only_prompt"), str) or not spec["scene_only_prompt"]:
        raise SpecificationProbeError("scene-only prompt must be non-empty")
    mapping = _pair_map(spec)
    if set(mapping) != set(task_ids) or any(mapping.get(value) != key for key, value in mapping.items()):
        raise SpecificationProbeError("hard-negative pair map must be involutive and cover pilot tasks")
    if spec.get("pilot_advancement", {}).get("gate_decision_authorized") is not False:
        raise SpecificationProbeError("the overlap pilot cannot authorize a Gate decision")


def _validate_evaluation_contract(spec: dict[str, Any]) -> None:
    evaluation = spec.get("evaluation_contract", {})
    required_true = {
        "same_batch_across_arms",
        "same_mode_across_arms",
        "same_seed_init_mapping_across_arms",
        "same_policy_rng_seed_across_arms",
        "upstream_rollout_unchanged",
    }
    missing = sorted(key for key in required_true if evaluation.get(key) is not True)
    if missing:
        raise SpecificationProbeError(f"matched evaluation contract is incomplete: {missing}")
    if evaluation.get("cross_batch_pooling") is not False:
        raise SpecificationProbeError("cross-batch pooling must remain disabled")


def validate_specification_spec(spec: dict[str, Any]) -> None:
    _validate_surface(spec)
    _validate_evaluation_contract(spec)


def load_specification_spec(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    validate_specification_spec(spec)
    return spec


def resolve_prompt(
    spec: dict[str, Any], task_id: int, condition: str, languages: dict[int, str]
) -> str:
    if task_id not in languages:
        raise SpecificationProbeError(f"missing task language: {task_id}")
    if condition == "correct":
        return languages[task_id]
    if condition == "no_spec":
        return ""
    if condition == "scene_only":
        return spec["scene_only_prompt"]
    if condition == "swapped":
        return languages[_pair_map(spec)[task_id]]
    raise SpecificationProbeError(f"unknown prompt condition: {condition}")


def _set_vector_attr(env: Any, name: str, values: list[Any]) -> None:
    target = env._env if isinstance(env, ResetAuditEnv) else env
    setter = getattr(target, "set_attr", None)
    if callable(setter):
        setter(name, values)
        return
    ensure = getattr(target, "_ensure", None)
    if callable(ensure):
        ensure()
    underlying = getattr(target, "_env", None)
    setter = getattr(underlying, "set_attr", None)
    if not callable(setter):
        raise SpecificationProbeError("vector environment cannot override task_description")
    setter(name, values)


def apply_prompt_override(env: Any, prompt: str, *, batch_size: int) -> dict[str, Any]:
    """Change only the policy-visible description and prove task identity stayed fixed."""

    task_before = list(env.call("task"))
    prompt_before = list(env.call("task_description"))
    _set_vector_attr(env, "task_description", [prompt] * batch_size)
    task_after = list(env.call("task"))
    prompt_after = list(env.call("task_description"))
    valid = (
        task_before == task_after
        and len(task_after) == batch_size
        and prompt_after == [prompt] * batch_size
    )
    return {
        "task_before": task_before,
        "task_after": task_after,
        "prompt_before": prompt_before,
        "prompt_after": prompt_after,
        "mechanically_valid": valid,
    }


def _condition_rows(
    arms: list[dict[str, Any]], left: str, right: str
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    by_key = {
        (int(arm["task_id"]), str(arm["condition"])): np.asarray(arm["successes"], dtype=float)
        for arm in arms
    }
    task_ids = sorted(task_id for task_id, condition in by_key if condition == left)
    rows = []
    for task_id in task_ids:
        try:
            left_values = by_key[(task_id, left)]
            right_values = by_key[(task_id, right)]
        except KeyError as error:
            raise SpecificationProbeError("paired comparison is missing an arm") from error
        if left_values.shape != right_values.shape or left_values.ndim != 1:
            raise SpecificationProbeError("paired arms must have equal one-dimensional episodes")
        rows.append((task_id, left_values, right_values))
    if not rows:
        raise SpecificationProbeError("paired comparison has no tasks")
    return rows


def paired_gap_summary(
    arms: list[dict[str, Any]], *, left: str, right: str, seed: int, replicates: int
) -> dict[str, Any]:
    """Hierarchical paired bootstrap: tasks first, then paired episodes within task."""

    rows = _condition_rows(arms, left, right)
    task_gaps = [(left_values - right_values).mean() for _, left_values, right_values in rows]
    observed = float(np.mean(task_gaps) * 100.0)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        task_draw = rng.integers(0, len(rows), size=len(rows))
        sampled_gaps = []
        for task_position in task_draw:
            _, left_values, right_values = rows[int(task_position)]
            episode_draw = rng.integers(0, len(left_values), size=len(left_values))
            sampled_gaps.append((left_values[episode_draw] - right_values[episode_draw]).mean())
        draws[replicate] = np.mean(sampled_gaps) * 100.0
    return {
        "left": left,
        "right": right,
        "task_count": len(rows),
        "episodes_per_task": int(len(rows[0][1])),
        "gap_pp": observed,
        "ci95_pp": np.quantile(draws, [0.025, 0.975]).tolist(),
        "task_gap_pp": {str(task_id): float(gap * 100.0) for (task_id, _, _), gap in zip(rows, task_gaps)},
    }


def decide_pilot(
    spec: dict[str, Any], arms: list[dict[str, Any]], *, mechanics_valid: bool
) -> dict[str, Any]:
    decision = {"gate_decision_authorized": False}
    if not mechanics_valid:
        return {**decision, "status": "stopped", "reason": "mechanics_identity_failure"}
    minimum = spec["pilot_advancement"]["minimum_correct_successes_per_task"]
    correct = {arm["task_id"]: sum(arm["successes"]) for arm in arms if arm["condition"] == "correct"}
    if any(value < minimum for value in correct.values()):
        return {
            **decision,
            "status": "stopped",
            "reason": "correct_arm_zero_success",
            "correct_successes": correct,
        }
    if set(correct) != set(spec["task_ids"]):
        return {**decision, "status": "stopped", "reason": "missing_correct_arm"}
    return {
        **decision,
        "status": "pilot_completed_scale_candidate",
        "reason": "mechanics_valid_and_correct_arm_competent",
        "correct_successes": correct,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _task_authorities(spec: dict[str, Any]) -> tuple[Any, dict[int, str], list[dict[str, Any]]]:
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite, get_task_init_states

    suite = _get_suite(spec["task_suite"])
    languages: dict[int, str] = {}
    authorities = []
    for task_id in spec["task_ids"]:
        task = suite.get_task(task_id)
        languages[task_id] = task.language
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = np.asarray(get_task_init_states(suite, task_id))
        row_hashes = [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in init_states[: spec["batch_size"]]
        ]
        authorities.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
                "bddl_filename": task.bddl_file,
                "bddl_sha256": sha256_file(bddl_path),
                "init_state_filename": task.init_states_file,
                "init_state_file_sha256": sha256_file(init_path),
                "init_state_indices": list(range(spec["batch_size"])),
                "init_state_sha256": row_hashes,
            }
        )
    return suite, languages, authorities


def _relative_videos(output_dir: Path, paths: list[str]) -> list[str]:
    relative = []
    for raw_path in paths:
        resolved = Path(raw_path).resolve()
        try:
            relative.append(resolved.relative_to(output_dir.resolve()).as_posix())
        except ValueError as error:
            raise SpecificationProbeError("evaluation video escaped the output directory") from error
    return relative


def _run_upstream_eval(
    *,
    spec: dict[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any],
    env: Any,
    videos_dir: Path,
) -> tuple[dict[str, Any], float]:
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    set_seed(spec["policy_rng_seed"])
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = runtime
    started = time.time()
    metrics = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=spec["episodes_per_task"],
        max_episodes_rendered=spec["max_videos_per_arm"],
        videos_dir=videos_dir,
        start_seed=spec["seed_start"],
    )
    return metrics, time.time() - started


def _evaluate_arm(
    *,
    spec: dict[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any],
    task_id: int,
    condition: str,
    prompt: str,
    output_dir: Path,
) -> dict[str, Any]:
    batch_size = spec["batch_size"]
    env_spec = {**spec, "task_id": task_id}
    base_env = _make_condition_env(
        env_spec,
        {"name": f"{task_id}_{condition}", "batch_size": batch_size, "mode": "async"},
    )
    env = ResetAuditEnv(base_env)
    try:
        override = apply_prompt_override(env, prompt, batch_size=batch_size)
        videos_dir = output_dir / "videos" / f"task_{task_id}" / condition
        metrics, elapsed = _run_upstream_eval(
            spec=spec, runtime=runtime, env=env, videos_dir=videos_dir
        )
        final_init_ids = list(env.call("init_state_id"))
    finally:
        env.close()
    if len(env.reset_events) != 1:
        raise SpecificationProbeError("upstream rollout must perform exactly one explicit batch reset")
    reset_event = env.reset_events[0]
    expected_before = list(range(batch_size))
    expected_after = list(range(batch_size, 2 * batch_size))
    mechanics_valid = (
        override["mechanically_valid"]
        and reset_event["before"] == expected_before
        and reset_event["after"] == expected_after
        and reset_event["seeds"]
        == list(range(spec["seed_start"], spec["seed_start"] + batch_size))
    )
    episodes = metrics["per_episode"]
    return {
        "task_id": task_id,
        "condition": condition,
        "prompt": prompt,
        "seeds": list(range(spec["seed_start"], spec["seed_start"] + batch_size)),
        "init_state_indices": expected_before,
        "init_state_ids_before_reset": reset_event["before"],
        "init_state_ids_after_reset": reset_event["after"],
        "init_state_ids_after_rollout": final_init_ids,
        "prompt_override": override,
        "mechanics_valid": mechanics_valid,
        "successes": [bool(episode["success"]) for episode in episodes],
        "sum_rewards": [float(episode["sum_reward"]) for episode in episodes],
        "max_rewards": [float(episode["max_reward"]) for episode in episodes],
        "video_paths": _relative_videos(output_dir, metrics.get("video_paths", [])),
        "eval_seconds": elapsed,
    }


def _comparison_report(spec: dict[str, Any], arms: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for offset, right in enumerate(("no_spec", "scene_only", "swapped")):
        report = paired_gap_summary(
            arms,
            left="correct",
            right=right,
            seed=spec["bootstrap_seed"] + offset,
            replicates=spec["bootstrap_replicates"],
        )
        threshold_key = (
            "correct_vs_swapped_success_pp"
            if right == "swapped"
            else "full_vs_no_spec_success_pp"
        )
        report["descriptive_threshold_pp"] = spec["gate_thresholds"][threshold_key]
        report["descriptive_threshold_met"] = report["gap_pp"] >= report["descriptive_threshold_pp"]
        comparisons[f"correct_vs_{right}"] = report
    comparisons["same_init_counterfactual_goal_switch"] = {
        "status": "not_measured",
        "reason": "prompt override leaves the environment goal unchanged",
    }
    return comparisons


def _eval_info(spec: dict[str, Any], arms: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": {
            "surface": spec["surface"],
            "episodes": sum(len(arm["successes"]) for arm in arms),
            "successes": sum(sum(arm["successes"]) for arm in arms),
            "gate_decision_authorized": False,
        },
        "per_task": [
            {
                "task_group": f"{spec['task_suite']}:{arm['condition']}",
                "task_id": arm["task_id"],
                "metrics": {
                    "successes": arm["successes"],
                    "sum_rewards": arm["sum_rewards"],
                    "max_rewards": arm["max_rewards"],
                    "video_paths": arm["video_paths"],
                },
            }
            for arm in arms
        ],
    }


def _write_checksums(output_dir: Path) -> None:
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in files]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_contract_alignment(spec: dict[str, Any], contract: dict[str, Any]) -> None:
    if spec["gate_thresholds"] != contract["gate_minus_one"]["thresholds"]:
        raise SpecificationProbeError("Gate thresholds differ from the Phase 0 contract")
    expected_role = contract["models"]["smolvla_libero_smoke"]["role"]
    if spec["policy"]["role"] != expected_role:
        raise SpecificationProbeError("official overlap checkpoint role changed")


def _run_arms(
    *,
    spec: dict[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any],
    languages: dict[int, str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    arms = []
    for task_id in spec["task_ids"]:
        for condition in spec["conditions"]:
            arm = _evaluate_arm(
                spec=spec,
                runtime=runtime,
                task_id=task_id,
                condition=condition,
                prompt=resolve_prompt(spec, task_id, condition, languages),
                output_dir=output_dir,
            )
            arms.append(arm)
            print(
                f"task={task_id} condition={condition} successes={sum(arm['successes'])}/{len(arm['successes'])}",
                flush=True,
            )
            if condition == "correct" and sum(arm["successes"]) == 0:
                break
        if arms[-1]["condition"] == "correct" and sum(arms[-1]["successes"]) == 0:
            break
    return arms


def _result_document(
    *,
    spec_path: Path,
    contract_path: Path,
    spec: dict[str, Any],
    contract: dict[str, Any],
    authorities: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    decision: dict[str, Any],
    comparisons: dict[str, Any],
    physical_gpu: int,
) -> dict[str, Any]:
    model = contract["models"]["smolvla_libero_smoke"]
    return {
        "schema_version": 1,
        "status": decision["status"],
        "surface": spec["surface"],
        "config_filename": spec_path.name,
        "config_sha256": sha256_file(spec_path),
        "phase0_contract_sha256": sha256_file(contract_path),
        "policy": {
            "role": spec["policy"]["role"],
            "revision": model["revision"],
            "weight_sha256": model["weight_sha256"],
        },
        "spec": spec,
        "task_authorities": authorities,
        "arms": arms,
        "comparisons": comparisons,
        "decision": decision,
        "interpretation": spec["interpretation"],
        "resources": {
            "cuda_visible_devices": str(physical_gpu),
            **spec["resource_contract"],
        },
    }


def run_probe(
    *,
    spec_path: Path,
    contract_path: Path,
    policy_path: Path,
    output_dir: Path,
    latest_link: Path | None,
    physical_gpu: int,
) -> dict[str, Any]:
    spec = load_specification_spec(spec_path)
    contract = load_contract(contract_path)
    validate_contract(contract)
    _validate_contract_alignment(spec, contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SpecificationProbeError(f"refusing non-empty output directory: {output_dir}")
    _, languages, authorities = _task_authorities(spec)
    runtime = _load_policy(policy_path, {**spec, "task_id": spec["task_ids"][0]})
    arms = _run_arms(spec=spec, runtime=runtime, languages=languages, output_dir=output_dir)
    mechanics_valid = all(arm["mechanics_valid"] for arm in arms)
    decision = decide_pilot(spec, arms, mechanics_valid=mechanics_valid)
    comparisons = _comparison_report(spec, arms) if decision["status"] != "stopped" else {}
    result = _result_document(
        spec_path=spec_path,
        contract_path=contract_path,
        spec=spec,
        contract=contract,
        authorities=authorities,
        arms=arms,
        decision=decision,
        comparisons=comparisons,
        physical_gpu=physical_gpu,
    )
    _atomic_json(output_dir / "probe_result.json", result)
    _atomic_json(output_dir / "eval_info.json", _eval_info(spec, arms))
    build_eval_gallery(output_dir)
    _write_checksums(output_dir)
    if latest_link is not None:
        update_latest_link(output_dir, latest_link)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.physical_gpu < 0 or args.physical_gpu > 7:
        parser.error("--physical-gpu must be between 0 and 7")
    return args


def main() -> int:
    args = _parse_args()
    try:
        result = run_probe(
            spec_path=args.config.resolve(),
            contract_path=args.contract.resolve(),
            policy_path=args.policy_path.resolve(),
            output_dir=args.output_dir.resolve(),
            latest_link=args.latest_link,
            physical_gpu=args.physical_gpu,
        )
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            args.output_dir / "failure_packet.json",
            {
                "schema_version": 1,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps({"status": result["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
