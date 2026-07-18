"""Validate a same-state, native-BDDL paired-goal surface on LIBERO-90 source tasks."""

from __future__ import annotations

import argparse
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


class CounterfactualGoalProbeError(RuntimeError):
    """Raised when the source-only paired-goal mechanics contract is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    prefix = f"{array.dtype.str}:{array.shape}:".encode("ascii")
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


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


def _validated_task_ids(spec: dict[str, Any]) -> list[int]:
    task_ids = spec.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 2:
        raise CounterfactualGoalProbeError("probe requires exactly two distinct task IDs")
    if len(set(task_ids)) != 2 or not all(isinstance(value, int) for value in task_ids):
        raise CounterfactualGoalProbeError("probe requires exactly two distinct task IDs")
    return task_ids


def _validate_task_pair(spec: dict[str, Any]) -> None:
    task_ids = _validated_task_ids(spec)
    if spec.get("source_task_ids") != task_ids:
        raise CounterfactualGoalProbeError("both paired tasks must be declared source tasks")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or [task.get("task_id") for task in tasks] != task_ids:
        raise CounterfactualGoalProbeError("task authorities must follow the paired task IDs")
    if any(task.get("split") != "source" for task in tasks):
        raise CounterfactualGoalProbeError("task authorities must remain source-only")
    if any(task.get("scene_id") != spec.get("scene_id") for task in tasks):
        raise CounterfactualGoalProbeError("paired tasks must share the declared scene")
    if len({task.get("state_dimension") for task in tasks}) != 1:
        raise CounterfactualGoalProbeError("paired tasks must declare one state dimension")
    if spec.get("initial_state_source_task_id") not in task_ids:
        raise CounterfactualGoalProbeError("initial-state authority must be one paired task")


def _validate_frozen_protocol(spec: dict[str, Any]) -> None:
    for field in ("initial_state_indices", "demonstration_indices"):
        if spec.get(field) != list(range(8)):
            raise CounterfactualGoalProbeError(f"{field} must remain the frozen first eight rows")
    expected_strings = {
        "terminal_state_selector": "last_recorded_state",
        "state_identity": "exact_flattened_mujoco_state",
        "success_evaluator": "pinned_libero_native_bddl_check_success",
    }
    for field, expected in expected_strings.items():
        if spec.get(field) != expected:
            label = "native BDDL evaluator" if field == "success_evaluator" else field
            raise CounterfactualGoalProbeError(f"frozen {label} changed")
    if spec.get("counterfactual_switch_threshold") != 0.8:
        raise CounterfactualGoalProbeError("counterfactual switch threshold must remain 0.8")
    if spec.get("claim_boundary", {}).get("gate_decision_authorized") is not False:
        raise CounterfactualGoalProbeError("mechanics probe cannot authorize a Gate decision")


def validate_probe_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise CounterfactualGoalProbeError("unsupported same-init probe schema")
    if spec.get("surface") != "libero90_source_executable_goal_mechanics":
        raise CounterfactualGoalProbeError("probe must remain on the LIBERO-90 source surface")
    if spec.get("task_suite") != "libero_90":
        raise CounterfactualGoalProbeError("probe must use the pinned libero_90 suite")
    _validate_task_pair(spec)
    _validate_frozen_protocol(spec)


def load_probe_spec(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    validate_probe_spec(spec)
    return spec


def _validate_evidence_shape(
    spec: dict[str, Any], initial_rows: list[dict[str, Any]], terminal_rows: list[dict[str, Any]]
) -> None:
    task_ids = spec["task_ids"]
    keys = {str(task_id) for task_id in task_ids}
    if len(initial_rows) != len(spec["initial_state_indices"]):
        raise CounterfactualGoalProbeError("initial-state evidence count changed")
    expected_terminal = len(task_ids) * len(spec["demonstration_indices"])
    if len(terminal_rows) != expected_terminal:
        raise CounterfactualGoalProbeError("terminal-state evidence count changed")

    if not all(set(row.get("success_by_task", {})) == keys for row in initial_rows + terminal_rows):
        raise CounterfactualGoalProbeError("goal-evaluator matrix is incomplete")


def _origin_switch_summary(
    spec: dict[str, Any], terminal_rows: list[dict[str, Any]], origin: int
) -> dict[str, Any]:
    rows = [row for row in terminal_rows if row.get("origin_task_id") == origin]
    if len(rows) != len(spec["demonstration_indices"]):
        raise CounterfactualGoalProbeError("terminal evidence is not balanced by origin")
    paired = next(task_id for task_id in spec["task_ids"] if task_id != origin)
    correct = sum(
        row["state_identity_exact"]
        and row["success_by_task"][str(origin)]
        and not row["success_by_task"][str(paired)]
        for row in rows
    )
    return {
        "paired_task_id": paired,
        "correct_switches": correct,
        "rows": len(rows),
        "switch_fraction": correct / len(rows),
    }


def assess_goal_switch(
    spec: dict[str, Any],
    initial_rows: list[dict[str, Any]],
    terminal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize exact-state identity and bidirectional native-goal specificity."""

    _validate_evidence_shape(spec, initial_rows, terminal_rows)
    initial_passed = all(
        row["state_identity_exact"] and not any(row["success_by_task"].values())
        for row in initial_rows
    )
    per_origin = {
        str(origin): _origin_switch_summary(spec, terminal_rows, origin)
        for origin in spec["task_ids"]
    }
    minimum = min(value["switch_fraction"] for value in per_origin.values())
    passed = initial_passed and minimum >= spec["counterfactual_switch_threshold"]
    return {
        "status": "mechanics_pass" if passed else "mechanics_failed",
        "initial_state_contract_passed": initial_passed,
        "per_origin": per_origin,
        "minimum_switch_fraction": minimum,
        "threshold": spec["counterfactual_switch_threshold"],
        "threshold_met": minimum >= spec["counterfactual_switch_threshold"],
        "gate_decision_authorized": False,
    }


def validate_paired_source_authority(
    spec: dict[str, Any], manifest_path: Path, seal_path: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate a frozen two-task source authority against the resealed manifest."""
    seal_sha = _sha256_file(seal_path)
    if seal_sha != spec["split_seal_sha256"]:
        raise CounterfactualGoalProbeError("split seal hash differs from the frozen probe")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not set(spec["task_ids"]).issubset(set(seal["active_split"]["source"])):
        raise CounterfactualGoalProbeError("paired task is absent from the sealed source split")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["split_reseal"]["record_sha256"] != seal_sha:
        raise CounterfactualGoalProbeError("manifest was not built from the frozen split seal")
    dataset = contract["datasets"]["libero_90"]
    if manifest["dataset"]["revision"] != dataset["revision"]:
        raise CounterfactualGoalProbeError("manifest dataset revision differs from the contract")
    by_id = {int(task["task_index"]): task for task in manifest["tasks"]}
    declared = {int(task["task_id"]): task for task in spec["tasks"]}
    for task_id in spec["task_ids"]:
        task = by_id.get(task_id)
        expected = declared[task_id]
        if task is None or task["split"] != "source":
            raise CounterfactualGoalProbeError("manifest does not authorize the source task")
        if task["scene"] != spec["scene_id"] or task["language"] != expected["language"]:
            raise CounterfactualGoalProbeError("manifest task specification changed")
        for field, manifest_value in (
            ("bddl_sha256", task["bddl"]["sha256"]),
            ("hdf5_sha256", task["hdf5"]["sha256"]),
            ("hdf5_filename", task["hdf5"]["filename"]),
            ("hdf5_bytes", task["hdf5"]["bytes"]),
            ("state_dimension", task["init_states"]["state_dimension"]),
        ):
            if expected[field] != manifest_value:
                raise CounterfactualGoalProbeError(f"manifest authority changed: {field}")
    authority = {
        "manifest_sha256": _sha256_file(manifest_path),
        "split_seal_sha256": seal_sha,
        "dataset_revision": manifest["dataset"]["revision"],
        "libero_runtime_commit": manifest["upstream_authority"]["libero_runtime_commit"],
    }
    return authority, {task_id: by_id[task_id] for task_id in spec["task_ids"]}


def _model_layout(environment: Any) -> dict[str, Any]:
    model = environment.sim.model

    def names(kind: str, count: int) -> list[str | None]:
        getter = getattr(model, f"{kind}_id2name")
        return [getter(index) for index in range(count)]

    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(model.na),
        "nu": int(model.nu),
        "joints": names("joint", int(model.njnt)),
        "bodies": names("body", int(model.nbody)),
        "geoms": names("geom", int(model.ngeom)),
        "sites": names("site", int(model.nsite)),
        "actuators": names("actuator", int(model.nu)),
        "joint_qpos_addresses": np.asarray(model.jnt_qposadr).tolist(),
        "joint_dof_addresses": np.asarray(model.jnt_dofadr).tolist(),
        "joint_types": np.asarray(model.jnt_type).tolist(),
    }


def _evaluate_state(environments: dict[int, Any], state: np.ndarray) -> dict[str, Any]:
    post_states: dict[int, np.ndarray] = {}
    success: dict[str, bool] = {}
    for task_id, environment in environments.items():
        environment.set_init_state(state)
        post_states[task_id] = np.asarray(environment.get_sim_state()).copy()
        success[str(task_id)] = bool(environment.check_success())
    reference = post_states[next(iter(environments))]
    exact = all(np.array_equal(reference, value) for value in post_states.values())
    maximum_delta = max(float(np.max(np.abs(reference - value))) for value in post_states.values())
    return {
        "state_identity_exact": exact,
        "maximum_absolute_state_delta": maximum_delta,
        "post_state_sha256_by_task": {
            str(task_id): _array_digest(value) for task_id, value in post_states.items()
        },
        "success_by_task": success,
    }


def _make_source_environments(
    spec: dict[str, Any], suite: Any, get_libero_path: Any, parse_problem: Any, env_type: Any
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    declared = {int(task["task_id"]): task for task in spec["tasks"]}
    environments: dict[int, Any] = {}
    authorities = []
    for task_id in spec["task_ids"]:
        task = suite.get_task(task_id)
        expected = declared[task_id]
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        problem = parse_problem(str(bddl))
        init_states = np.asarray(suite.get_task_init_states(task_id))
        if task.language != expected["language"] or _sha256_file(bddl) != expected["bddl_sha256"]:
            raise CounterfactualGoalProbeError("pinned BDDL task authority changed")
        if _json_digest(problem["goal_state"]) != expected["goal_sha256"]:
            raise CounterfactualGoalProbeError("native BDDL goal predicate changed")
        if init_states.shape[1] != expected["state_dimension"]:
            raise CounterfactualGoalProbeError("native init-state dimension changed")
        environments[task_id] = env_type(
            bddl_file_name=str(bddl),
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
        )
        environments[task_id].reset()
        authorities.append(
            {
                "task_id": task_id,
                "language": task.language,
                "bddl_filename": task.bddl_file,
                "bddl_sha256": expected["bddl_sha256"],
                "goal_state": problem["goal_state"],
                "goal_sha256": expected["goal_sha256"],
                "state_dimension": int(init_states.shape[1]),
            }
        )
    return environments, authorities


def _layout_hashes(environments: dict[int, Any]) -> dict[str, str]:
    hashes = {
        str(task_id): _json_digest(_model_layout(environment))
        for task_id, environment in environments.items()
    }
    if len(set(hashes.values())) != 1:
        raise CounterfactualGoalProbeError("paired MuJoCo state layouts are not identical")
    return hashes


def _probe_initial_states(
    spec: dict[str, Any], suite: Any, environments: dict[int, Any]
) -> list[dict[str, Any]]:
    source_id = spec["initial_state_source_task_id"]
    states = np.asarray(suite.get_task_init_states(source_id))
    return [
        {
            "source_task_id": source_id,
            "init_state_index": index,
            "input_state_sha256": _array_digest(states[index]),
            **_evaluate_state(environments, states[index]),
        }
        for index in spec["initial_state_indices"]
    ]


def _probe_terminal_states(
    spec: dict[str, Any], dataset_root: Path, environments: dict[int, Any]
) -> list[dict[str, Any]]:
    import h5py

    declared = {int(task["task_id"]): task for task in spec["tasks"]}
    rows = []
    for task_id in spec["task_ids"]:
        expected = declared[task_id]
        hdf5_path = dataset_root / expected["hdf5_filename"]
        if not hdf5_path.is_file() or hdf5_path.stat().st_size != expected["hdf5_bytes"]:
            raise CounterfactualGoalProbeError("source HDF5 file authority changed")
        with h5py.File(hdf5_path, "r") as handle:
            for index in spec["demonstration_indices"]:
                states = np.asarray(handle["data"][f"demo_{index}"]["states"])
                if states.ndim != 2 or states.shape[1] != expected["state_dimension"]:
                    raise CounterfactualGoalProbeError("source demonstration state shape changed")
                rows.append(
                    {
                        "origin_task_id": task_id,
                        "demonstration_index": index,
                        "recorded_state_count": int(len(states)),
                        "input_state_sha256": _array_digest(states[-1]),
                        **_evaluate_state(environments, states[-1]),
                    }
                )
    return rows


def _runtime_probe(
    spec: dict[str, Any], dataset_root: Path, libero_config_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_root.resolve())
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark_dict
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = get_benchmark_dict()[spec["task_suite"]]()
    environments: dict[int, Any] = {}
    try:
        environments, authorities = _make_source_environments(
            spec, suite, get_libero_path, robosuite_parse_problem, ControlEnv
        )
        layout_hashes = _layout_hashes(environments)
        initial_rows = _probe_initial_states(spec, suite, environments)
        terminal_rows = _probe_terminal_states(spec, dataset_root, environments)
        runtime = {
            "task_authorities": authorities,
            "model_layout_sha256_by_task": layout_hashes,
            "model_layout_identical": True,
        }
        return initial_rows, terminal_rows, runtime
    finally:
        for environment in environments.values():
            environment.close()


def _render_report(result: dict[str, Any]) -> str:
    assessment = result["assessment"]
    rows = []
    for task_id, summary in assessment["per_origin"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(task_id)}</td>"
            f"<td>{summary['paired_task_id']}</td>"
            f"<td>{summary['correct_switches']}/{summary['rows']}</td>"
            f"<td>{summary['switch_fraction']:.3f}</td>"
            "</tr>"
        )
    goals = "".join(
        f"<li><strong>task {item['task_id']}</strong>: "
        f"{html.escape(item['language'])}<br><code>{html.escape(json.dumps(item['goal_state']))}</code></li>"
        for item in result["runtime"]["task_authorities"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMBER same-state goal mechanics</title><style>
body{{font:16px system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}}
code{{white-space:pre-wrap}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #555;padding:.55rem;text-align:left}}
.ok{{color:#7ee787}} .warn{{color:#f2cc60}}
</style></head><body><h1>EMBER · same-state executable-goal mechanics</h1>
<p class="{'ok' if assessment['status'] == 'mechanics_pass' else 'warn'}"><strong>{assessment['status']}</strong></p>
<p>Source-only native-BDDL mechanics evidence. This report does not pass Gate -1 and does not authorize Writer training.</p>
<h2>Paired source goals</h2><ul>{goals}</ul>
<h2>Exact-state goal matrix</h2><table><thead><tr><th>origin task</th><th>paired evaluator</th><th>specific switches</th><th>fraction</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Initial-state neutral contract: <strong>{assessment['initial_state_contract_passed']}</strong>; threshold: {assessment['threshold']:.2f}.</p>
<p><a href="probe_result.json">probe_result.json</a> · <a href="checksums.sha256">checksums.sha256</a></p>
</body></html>"""


def _write_checksums(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    _atomic_text(
        output_dir / "checksums.sha256",
        "".join(f"{_sha256_file(path)}  {path.name}\n" for path in files),
    )


def run_probe(
    *,
    spec_path: Path,
    contract_path: Path,
    seal_path: Path,
    manifest_path: Path,
    dataset_root: Path,
    libero_config_root: Path,
    output_dir: Path,
    latest_link: Path | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise CounterfactualGoalProbeError(f"refusing existing output directory: {output_dir}")
    spec = load_probe_spec(spec_path)
    contract = load_contract(contract_path)
    validate_contract(contract)
    threshold = contract["gate_minus_one"]["thresholds"]["counterfactual_correct_switch_fraction"]
    if spec["counterfactual_switch_threshold"] != threshold:
        raise CounterfactualGoalProbeError("probe threshold differs from the Phase 0 contract")
    authority, _ = validate_paired_source_authority(
        spec, manifest_path, seal_path, contract
    )
    output_dir.mkdir(parents=True)
    initial_rows, terminal_rows, runtime = _runtime_probe(
        spec, dataset_root, libero_config_root
    )
    assessment = assess_goal_switch(spec, initial_rows, terminal_rows)
    result = {
        "schema_version": 1,
        "status": assessment["status"],
        "surface": spec["surface"],
        "config_filename": spec_path.name,
        "config_sha256": _sha256_file(spec_path),
        "phase0_contract_sha256": _sha256_file(contract_path),
        "authority": authority,
        "spec": spec,
        "runtime": runtime,
        "initial_state_rows": initial_rows,
        "terminal_state_rows": terminal_rows,
        "assessment": assessment,
        "interpretation": {
            "authorized": spec["claim_boundary"]["authorized_claim"],
            "not_authorized": spec["claim_boundary"]["not_authorized"],
        },
        "resources": spec["resources"],
    }
    _atomic_json(output_dir / "probe_result.json", result)
    _atomic_text(output_dir / "index.html", _render_report(result))
    _write_checksums(output_dir)
    if latest_link is not None:
        update_latest_link(output_dir, latest_link)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--libero-config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_preexisted = args.output_dir.exists()
    try:
        result = run_probe(
            spec_path=args.config.resolve(),
            contract_path=args.contract.resolve(),
            seal_path=args.seal.resolve(),
            manifest_path=args.manifest.resolve(),
            dataset_root=args.dataset_root.resolve(),
            libero_config_root=args.libero_config_root.resolve(),
            output_dir=args.output_dir.resolve(),
            latest_link=args.latest_link,
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
