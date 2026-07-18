"""Probe language-to-action causality from one cached LIBERO reset observation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import traceback
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from ember.contracts import load_contract, validate_contract
from ember.eval_artifacts import update_latest_link
from ember.identity_evidence import canonical_tree_summary
from ember.specification_probe import (
    load_specification_spec,
    resolve_prompt,
    sha256_file,
)


class LanguageActionProbeError(RuntimeError):
    """Raised when the same-observation language-action contract is violated."""


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate_frozen_pilot(pilot: dict[str, Any]) -> None:
    if pilot.get("surface") != "official_overlap_mechanics_only":
        raise LanguageActionProbeError("action probe must remain on the overlap surface")
    if pilot.get("task_suite") != "libero_spatial" or pilot.get("task_ids") != [0, 1]:
        raise LanguageActionProbeError("pilot task pair changed")
    if pilot.get("batch_size") != 8 or pilot.get("episodes_per_task") != 8:
        raise LanguageActionProbeError("fixed batch-8 pilot contract changed")
    if pilot.get("use_async_envs") is not True:
        raise LanguageActionProbeError("fixed async pilot contract changed")
    if pilot.get("conditions") != ["correct", "no_spec", "scene_only", "swapped"]:
        raise LanguageActionProbeError("pilot prompt conditions changed")
    if pilot.get("evaluation_contract", {}).get("cross_batch_pooling") is not False:
        raise LanguageActionProbeError("cross-batch pooling must remain disabled")


def _validate_action_thresholds(spec: dict[str, Any]) -> None:
    if spec.get("substantive_plan_max_abs_delta") != 0.01:
        raise LanguageActionProbeError("substantive action threshold changed")
    if spec.get("known_cross_batch_max_abs_delta") != 0.002254:
        raise LanguageActionProbeError("recorded cross-batch diagnostic changed")
    if spec["substantive_plan_max_abs_delta"] <= 4 * spec["known_cross_batch_max_abs_delta"]:
        raise LanguageActionProbeError("substantive threshold no longer clears batch-shape scale")
    if spec.get("minimum_substantive_fraction") != 0.8:
        raise LanguageActionProbeError("minimum action-path fraction changed")


def _validate_mechanics_contract(spec: dict[str, Any]) -> None:
    mechanics = spec.get("mechanics_contract", {})
    required_true = {
        "same_cached_reset_observation_across_conditions",
        "same_fixed_batch_shape_across_conditions",
        "same_policy_rng_seed_across_conditions",
        "policy_reset_before_each_condition",
        "correct_repeat_required",
        "upstream_policy_forward_unchanged",
    }
    if any(mechanics.get(field) is not True for field in required_true):
        raise LanguageActionProbeError("same-observation mechanics contract is incomplete")
    if mechanics.get("cross_batch_pooling") is not False:
        raise LanguageActionProbeError("cross-batch pooling must remain disabled")


def validate_action_spec(spec: dict[str, Any], pilot: dict[str, Any]) -> None:
    _validate_frozen_pilot(pilot)
    expected = {
        "schema_version": 1,
        "surface": pilot["surface"],
        "conditions": pilot["conditions"],
        "repeat_condition": "correct",
        "primary_comparison": "correct_vs_swapped",
        "planned_action_steps": 10,
        "environment_resets_per_task": 1,
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            label = "conditions" if field == "conditions" else field
            raise LanguageActionProbeError(f"action probe {label} changed")
    _validate_action_thresholds(spec)
    _validate_mechanics_contract(spec)
    if spec.get("interpretation", {}).get("gate_decision_authorized") is not False:
        raise LanguageActionProbeError("action diagnostic cannot authorize a Gate decision")


def load_action_spec(path: Path, pilot: dict[str, Any]) -> dict[str, Any]:
    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    validate_action_spec(spec, pilot)
    return spec


def compare_action_plans(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
    substantive_delta: float,
) -> dict[str, Any]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 3:
        raise LanguageActionProbeError("action plans must share [batch, step, action] shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise LanguageActionProbeError("action plan contains a non-finite value")
    delta = np.abs(left - right)
    episodes = []
    for index in range(left.shape[0]):
        episode_delta = delta[index]
        differs = not np.allclose(left[index], right[index], atol=atol, rtol=rtol)
        maximum = float(episode_delta.max())
        episodes.append(
            {
                "episode_index": index,
                "differs_beyond_tolerance": differs,
                "substantive": maximum >= substantive_delta,
                "mean_absolute_delta": float(episode_delta.mean()),
                "maximum_absolute_delta": maximum,
                "rms_delta": float(np.sqrt(np.mean(np.square(episode_delta)))),
                "first_action_maximum_absolute_delta": float(episode_delta[0].max()),
            }
        )
    differing = sum(row["differs_beyond_tolerance"] for row in episodes)
    substantive = sum(row["substantive"] for row in episodes)
    return {
        "shape": list(left.shape),
        "atol": atol,
        "rtol": rtol,
        "substantive_plan_max_abs_delta": substantive_delta,
        "differing_episodes": differing,
        "differing_fraction": differing / len(episodes),
        "substantive_episodes": substantive,
        "substantive_fraction": substantive / len(episodes),
        "overall_mean_absolute_delta": float(delta.mean()),
        "overall_maximum_absolute_delta": float(delta.max()),
        "episodes": episodes,
    }


def decide_action_probe(
    spec: dict[str, Any], *, repeat_stable: bool, primary: dict[str, Any]
) -> dict[str, Any]:
    decision = {"gate_decision_authorized": False}
    if not repeat_stable:
        return {**decision, "status": "stopped", "reason": "correct_repeat_instability"}
    if primary["substantive_fraction"] < spec["minimum_substantive_fraction"]:
        return {
            **decision,
            "status": "language_action_path_ambiguous",
            "reason": "swapped_plan_fraction_below_predeclared_diagnostic_threshold",
        }
    return {
        **decision,
        "status": "language_action_path_present",
        "reason": "stable_repeat_and_substantive_same_observation_prompt_effect",
    }


def _validate_prior_result(
    spec: dict[str, Any], pilot: dict[str, Any], prior_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    if sha256_file(prior_path) != spec["prior_pilot_result_sha256"]:
        raise LanguageActionProbeError("prior pilot result hash changed")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("config_sha256") != spec["pilot_config_sha256"]:
        raise LanguageActionProbeError("prior pilot config authority changed")
    expected_model = contract["models"]["smolvla_libero_smoke"]
    if prior.get("policy", {}).get("weight_sha256") != expected_model["weight_sha256"]:
        raise LanguageActionProbeError("prior pilot policy authority changed")
    aligned_fields = ("task_ids", "batch_size", "seed_start", "conditions", "policy_rng_seed")
    if any(prior.get("spec", {}).get(field) != pilot.get(field) for field in aligned_fields):
        raise LanguageActionProbeError("prior pilot evaluation identity changed")
    correct = {
        int(arm["task_id"]): sum(bool(value) for value in arm["successes"])
        for arm in prior.get("arms", [])
        if arm.get("condition") == "correct"
    }
    minimum = pilot["pilot_advancement"]["minimum_correct_successes_per_task"]
    if set(correct) != set(pilot["task_ids"]) or any(value < minimum for value in correct.values()):
        raise LanguageActionProbeError("prior pilot lacks correct-arm competence")
    return {
        "result_sha256": spec["prior_pilot_result_sha256"],
        "status": prior["status"],
        "correct_successes": {str(task_id): value for task_id, value in correct.items()},
        "gate_decision_authorized": False,
    }


def _task_authority(pilot: dict[str, Any]) -> tuple[dict[int, str], list[dict[str, Any]]]:
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite, get_task_init_states

    suite = _get_suite(pilot["task_suite"])
    languages: dict[int, str] = {}
    authorities = []
    for task_id in pilot["task_ids"]:
        task = suite.get_task(task_id)
        init_states = np.asarray(get_task_init_states(suite, task_id))
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        languages[task_id] = task.language
        authorities.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "language": task.language,
                "bddl_filename": task.bddl_file,
                "bddl_sha256": sha256_file(bddl),
                "init_state_filename": task.init_states_file,
                "init_state_file_sha256": sha256_file(init_file),
                "init_state_indices": list(range(pilot["batch_size"])),
                "init_state_sha256": [
                    hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                    for row in init_states[: pilot["batch_size"]]
                ],
            }
        )
    return languages, authorities


def _capture_reset_observation(pilot: dict[str, Any], task_id: int) -> tuple[Any, dict[str, Any]]:
    from ember.evaluation_identity import _make_condition_env

    batch_size = pilot["batch_size"]
    seeds = list(range(pilot["seed_start"], pilot["seed_start"] + batch_size))
    condition = {"name": "cached_reset", "mode": "async", "batch_size": batch_size}
    env = _make_condition_env({**pilot, "task_id": task_id}, condition)
    try:
        before = list(env.call("init_state_id"))
        observation, _ = env.reset(seed=seeds)
        observation = copy.deepcopy(observation)
        after = list(env.call("init_state_id"))
    finally:
        env.close()
    if before != list(range(batch_size)) or after != list(range(batch_size, 2 * batch_size)):
        raise LanguageActionProbeError("reset/init-state identity changed")
    return observation, {
        "task_id": task_id,
        "seeds": seeds,
        "init_state_ids_before_reset": before,
        "init_state_ids_after_reset": after,
        "reset_observation": canonical_tree_summary(observation),
    }


def _action_plan(
    runtime: tuple[Any, Any, Any, Any, Any],
    observation: Any,
    prompt: str,
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    from ember.evaluation_identity import _select_policy_action
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    runtime[0].reset()
    actions = []
    for _ in range(steps):
        action = _select_policy_action(
            runtime, copy.deepcopy(observation), [prompt] * batch_size
        )
        if action.shape != (batch_size, 7) or not np.isfinite(action).all():
            raise LanguageActionProbeError("policy action shape or finiteness changed")
        actions.append(np.asarray(action).copy())
    return np.stack(actions, axis=1)


def _task_probe(
    action_spec: dict[str, Any],
    pilot: dict[str, Any],
    runtime: tuple[Any, Any, Any, Any, Any],
    languages: dict[int, str],
    task_id: int,
) -> dict[str, Any]:
    observation, reset = _capture_reset_observation(pilot, task_id)
    plans: dict[str, np.ndarray] = {}
    prompts: dict[str, str] = {}
    for condition in action_spec["conditions"]:
        prompt = resolve_prompt(pilot, task_id, condition, languages)
        prompts[condition] = prompt
        plans[condition] = _action_plan(
            runtime,
            observation,
            prompt,
            batch_size=pilot["batch_size"],
            steps=action_spec["planned_action_steps"],
            seed=pilot["policy_rng_seed"],
        )
    repeat = _action_plan(
        runtime,
        observation,
        prompts["correct"],
        batch_size=pilot["batch_size"],
        steps=action_spec["planned_action_steps"],
        seed=pilot["policy_rng_seed"],
    )
    repeat_report = compare_action_plans(
        plans["correct"],
        repeat,
        atol=action_spec["repeat_atol"],
        rtol=action_spec["repeat_rtol"],
        substantive_delta=action_spec["substantive_plan_max_abs_delta"],
    )
    comparisons = {}
    for condition in action_spec["conditions"]:
        if condition == "correct":
            continue
        comparisons[f"correct_vs_{condition}"] = compare_action_plans(
            plans["correct"],
            plans[condition],
            atol=action_spec["action_atol"],
            rtol=action_spec["action_rtol"],
            substantive_delta=action_spec["substantive_plan_max_abs_delta"],
        )
    return {
        "task_id": task_id,
        "reset": reset,
        "prompts": prompts,
        "action_plans": {condition: plan.tolist() for condition, plan in plans.items()},
        "correct_repeat_action_plan": repeat.tolist(),
        "correct_repeat": repeat_report,
        "comparisons": comparisons,
    }


def _aggregate_comparison(tasks: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows = [row for task in tasks for row in task["comparisons"][key]["episodes"]]
    substantive = sum(row["substantive"] for row in rows)
    differing = sum(row["differs_beyond_tolerance"] for row in rows)
    return {
        "task_count": len(tasks),
        "episode_count": len(rows),
        "differing_episodes": differing,
        "differing_fraction": differing / len(rows),
        "substantive_episodes": substantive,
        "substantive_fraction": substantive / len(rows),
        "per_task_substantive_fraction": {
            str(task["task_id"]): task["comparisons"][key]["substantive_fraction"]
            for task in tasks
        },
        "overall_maximum_absolute_delta": max(
            task["comparisons"][key]["overall_maximum_absolute_delta"] for task in tasks
        ),
    }


def _render_report(result: dict[str, Any]) -> str:
    rows = []
    for key, summary in result["aggregate_comparisons"].items():
        width = min(100.0, 100.0 * summary["substantive_fraction"])
        rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{summary['substantive_episodes']}/{summary['episode_count']}</td>"
            f"<td><div class='bar'><i style='width:{width:.1f}%'></i></div>{summary['substantive_fraction']:.3f}</td>"
            f"<td>{summary['overall_maximum_absolute_delta']:.5f}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMBER same-observation language-action probe</title><style>
body{{font:16px system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #555;padding:.55rem;text-align:left}}
.bar{{display:inline-block;width:180px;height:.75rem;background:#333;margin-right:.5rem}}.bar i{{display:block;height:100%;background:#58a6ff}}
code{{white-space:pre-wrap}}
</style></head><body><h1>EMBER · same-observation language→action</h1>
<p><strong>{html.escape(result['status'])}</strong></p>
<p>Official-overlap diagnostic only. The cached reset observation and batch shape are identical across prompts; this is not correct paired-goal switching and cannot pass Gate -1.</p>
<table><thead><tr><th>comparison</th><th>substantive episodes</th><th>fraction</th><th>max |Δaction|</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Correct-repeat stable: <strong>{result['repeat_stable']}</strong>.</p>
<p><a href="probe_result.json">probe_result.json</a> · <a href="checksums.sha256">checksums.sha256</a></p>
</body></html>"""


def _write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    _atomic_text(
        output_dir / "checksums.sha256",
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
    )


def _run_policy_tasks(
    action_spec: dict[str, Any], pilot: dict[str, Any], policy_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    from ember.evaluation_identity import _load_policy

    languages, authorities = _task_authority(pilot)
    runtime = _load_policy(policy_path, {**pilot, "task_id": pilot["task_ids"][0]})
    task_results = [
        _task_probe(action_spec, pilot, runtime, languages, task_id)
        for task_id in pilot["task_ids"]
    ]
    repeat_stable = all(
        task["correct_repeat"]["differing_episodes"] == 0 for task in task_results
    )
    aggregate = {
        key: _aggregate_comparison(task_results, key)
        for key in ("correct_vs_no_spec", "correct_vs_scene_only", "correct_vs_swapped")
    }
    return authorities, task_results, repeat_stable, aggregate


def _result_document(
    *,
    action_spec_path: Path,
    contract_path: Path,
    action_spec: dict[str, Any],
    pilot: dict[str, Any],
    contract: dict[str, Any],
    prior: dict[str, Any],
    authorities: list[dict[str, Any]],
    task_results: list[dict[str, Any]],
    repeat_stable: bool,
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    physical_gpu: int,
) -> dict[str, Any]:
    import torch

    model = contract["models"]["smolvla_libero_smoke"]
    return {
        "schema_version": 1,
        "status": decision["status"],
        "surface": action_spec["surface"],
        "action_config_filename": action_spec_path.name,
        "action_config_sha256": sha256_file(action_spec_path),
        "pilot_config_sha256": action_spec["pilot_config_sha256"],
        "phase0_contract_sha256": sha256_file(contract_path),
        "prior_pilot": prior,
        "policy": {
            "role": model["role"],
            "revision": model["revision"],
            "weight_sha256": model["weight_sha256"],
        },
        "spec": action_spec,
        "pilot_identity": {
            field: pilot[field]
            for field in ("task_suite", "task_ids", "batch_size", "seed_start", "policy_rng_seed")
        },
        "task_authorities": authorities,
        "task_results": task_results,
        "repeat_stable": repeat_stable,
        "aggregate_comparisons": aggregate,
        "decision": decision,
        "interpretation": action_spec["interpretation"],
        "resources": {
            "physical_gpu": physical_gpu,
            **action_spec["resource_contract"],
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        },
    }


def run_probe(
    *,
    action_spec_path: Path,
    pilot_spec_path: Path,
    prior_result_path: Path,
    contract_path: Path,
    policy_path: Path,
    output_dir: Path,
    latest_link: Path | None,
    physical_gpu: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise LanguageActionProbeError(f"refusing existing output directory: {output_dir}")
    pilot = load_specification_spec(pilot_spec_path)
    action_spec = load_action_spec(action_spec_path, pilot)
    if sha256_file(pilot_spec_path) != action_spec["pilot_config_sha256"]:
        raise LanguageActionProbeError("pilot config hash changed")
    contract = load_contract(contract_path)
    validate_contract(contract)
    prior = _validate_prior_result(action_spec, pilot, prior_result_path, contract)
    output_dir.mkdir(parents=True)
    authorities, task_results, repeat_stable, aggregate = _run_policy_tasks(
        action_spec, pilot, policy_path
    )
    decision = decide_action_probe(
        action_spec,
        repeat_stable=repeat_stable,
        primary=aggregate[action_spec["primary_comparison"]],
    )
    result = _result_document(
        action_spec_path=action_spec_path,
        contract_path=contract_path,
        action_spec=action_spec,
        pilot=pilot,
        contract=contract,
        prior=prior,
        authorities=authorities,
        task_results=task_results,
        repeat_stable=repeat_stable,
        aggregate=aggregate,
        decision=decision,
        physical_gpu=physical_gpu,
    )
    _atomic_json(output_dir / "probe_result.json", result)
    _atomic_text(output_dir / "index.html", _render_report(result))
    _write_checksums(output_dir)
    if latest_link is not None:
        update_latest_link(output_dir, latest_link)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-config", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
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
    output_preexisted = args.output_dir.exists()
    try:
        result = run_probe(
            action_spec_path=args.action_config.resolve(),
            pilot_spec_path=args.pilot_config.resolve(),
            prior_result_path=args.prior_result.resolve(),
            contract_path=args.contract.resolve(),
            policy_path=args.policy_path.resolve(),
            output_dir=args.output_dir.resolve(),
            latest_link=args.latest_link,
            physical_gpu=args.physical_gpu,
        )
    except Exception as error:
        if not output_preexisted:
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
