"""Build the paired four-arm G1 Native-Factor capacity Gate report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.g1_assets import authority_path, load_g1_config
from ember.ecp.g1_runtime import REPO_ROOT
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.static_task_lora import STATIC_TASK_LORA_KIND


G1_GATE_SCHEMA = "ember_ecp_native_factor_g1_gate_v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _rows(
    result: Mapping[str, Any], held_keys: set[tuple[str, int]]
) -> dict[tuple[str, int, int], dict[str, Any]]:
    selected = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): dict(row)
        for row in result.get("rows", ())
        if (str(row["suite"]), int(row["task_id"])) in held_keys
    }
    expected = {
        (suite, task_id, state) for suite, task_id in held_keys for state in range(50)
    }
    if len(selected) != 250 or set(selected) != expected:
        raise ValueError("G1 arm is not exact held5 strict250")
    return selected


def _normalized_tokenizer(result: Mapping[str, Any]) -> dict[str, Any]:
    cell = result.get("tokenizer", {})
    return {
        "path": str(Path(str(cell.get("path", ""))).resolve()),
        "bytes": int(cell.get("bytes", -1)),
        "manifest_name": Path(str(cell.get("manifest_path", ""))).name,
    }


def _normalized_normalization(result: Mapping[str, Any]) -> dict[str, Any]:
    cell = result.get("normalization", {})
    return {
        "name": Path(str(cell.get("path", ""))).name,
        "bytes": int(cell.get("bytes", -1)),
        "source_only_numeric_reads": cell.get("source_only_numeric_reads"),
        "validation_or_test_numeric_reads": cell.get(
            "validation_or_test_numeric_reads"
        ),
    }


def _task_authority(
    contract: Mapping[str, Any], held: set[tuple[str, int]]
) -> dict[Any, Any]:
    fields = (
        "language",
        "bddl_file",
        "bddl_bytes",
        "init_states_file",
        "init_states_bytes",
        "horizon",
        "init_state_ids",
    )
    return {
        (str(row["suite"]), int(row["task_id"])): {
            field: row.get(field) for field in fields
        }
        for row in contract.get("tasks", ())
        if (str(row["suite"]), int(row["task_id"])) in held
    }


def _arm(*, name: str, path: Path, held_keys: set[tuple[str, int]]) -> dict[str, Any]:
    path = path.resolve()
    result = read_json(path)
    contract = read_json(path.parent / "run_contract.json")
    rows = _rows(result, held_keys)
    successes = sum(bool(row["success"]) for row in rows.values())
    if (
        result.get("schema_version") != "ember_pi05_target_eval_results_v2"
        or result.get("mode") != "formal"
        or result.get("role") != "development_train"
        or contract.get("mode") != "formal"
        or contract.get("role") != "development_train"
        or result.get("contract_reference") != contract.get("contract_reference")
    ):
        raise ValueError(f"G1 {name} arm is not a formal development-train result")
    per_task = {
        key: sum(
            bool(row["success"]) for row_key, row in rows.items() if row_key[:2] == key
        )
        for key in held_keys
    }
    return {
        "name": name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "contract_reference": str(result["contract_reference"]),
        "arm": str(result["arm"]),
        "successes": successes,
        "breadth": sum(value > 0 for value in per_task.values()),
        "per_task": per_task,
        "rows": rows,
        "model": result.get("model"),
        "tokenizer": _normalized_tokenizer(result),
        "normalization": _normalized_normalization(result),
        "rng": contract.get("rng"),
        "task_authority": _task_authority(contract, held_keys),
        "adapter": result.get("adapter"),
    }


def _paired_authority(arms: Sequence[Mapping[str, Any]]) -> None:
    reference = arms[0]
    fields = ("model", "tokenizer", "normalization", "rng", "task_authority")
    for arm in arms[1:]:
        if any(arm[field] != reference[field] for field in fields):
            raise ValueError(
                "G1 four-arm source, task, normalization, tokenizer, or RNG differs"
            )
    keys = set(reference["rows"])
    for key in keys:
        base = reference["rows"][key]
        identity = (
            base["language"],
            int(base["env_seed"]),
            int(base["policy_seed_root"]),
        )
        for arm in arms[1:]:
            row = arm["rows"][key]
            if (
                row["language"],
                int(row["env_seed"]),
                int(row["policy_seed_root"]),
            ) != identity:
                raise ValueError("G1 four-arm paired row identity differs")
            common = min(
                len(base["policy_noise_seeds"]), len(row["policy_noise_seeds"])
            )
            if (
                base["policy_noise_seeds"][:common]
                != row["policy_noise_seeds"][:common]
            ):
                raise ValueError("G1 four-arm paired policy-noise schedule differs")


def build_g1_gate_report(
    *, config_path: Path, asset_root: Path, free_results: Path, output_path: Path
) -> dict[str, Any]:
    config = load_g1_config(config_path.resolve())
    asset_root = asset_root.resolve()
    held_globals = tuple(int(value) for value in config["tasks"]["global_task_ids"])
    manifest = read_json(
        authority_path(config, "target_manifest", asset_root=asset_root)
    )
    held_rows = [
        row
        for row in manifest.get("tasks", ())
        if int(row["global_task_id"]) in held_globals
    ]
    held_keys = {(str(row["suite"]), int(row["task_id"])) for row in held_rows}
    if len(held_rows) != 5 or len(held_keys) != 5:
        raise ValueError("G1 Gate held5 authority changed")
    paths = {
        "carrier": authority_path(config, "carrier_strict250", asset_root=asset_root),
        "direct": authority_path(config, "direct_latest_strict", asset_root=asset_root),
        "mobile_rank4": authority_path(
            config, "mobile_latest_strict250", asset_root=asset_root
        ),
        "free_code": free_results.resolve(),
    }
    arms = {
        name: _arm(name=name, path=path, held_keys=held_keys)
        for name, path in paths.items()
    }
    _paired_authority(tuple(arms.values()))
    free_adapter = arms["free_code"]["adapter"] or {}
    free_tasks = tuple(free_adapter.get("tasks", ()))
    unique_complete = (
        free_adapter.get("kind") == STATIC_TASK_LORA_KIND
        and isinstance(free_adapter.get("training_commit"), str)
        and len(free_adapter["training_commit"]) == 40
        and isinstance(free_adapter.get("shared_run_contract"), Mapping)
        and free_adapter.get("single_complete_rank16") is True
        and free_adapter.get("rank_partition") == {"carrier": [0, 12], "task": [12, 16]}
        and len(free_tasks) == 5
        and len({str(row.get("adapter_path")) for row in free_tasks}) == 5
        and all(row.get("single_complete_rank16") is True for row in free_tasks)
    )
    carrier = arms["carrier"]
    mobile = arms["mobile_rank4"]
    free = arms["free_code"]
    denominator = int(mobile["successes"]) - int(carrier["successes"])
    if denominator <= 0:
        raise ValueError("G1 recovery denominator is non-positive")
    recovery = (int(free["successes"]) - int(carrier["successes"])) / denominator
    retained = sum(
        bool(carrier["rows"][key]["success"]) and bool(free["rows"][key]["success"])
        for key in carrier["rows"]
    )
    gained = sum(
        not bool(carrier["rows"][key]["success"]) and bool(free["rows"][key]["success"])
        for key in carrier["rows"]
    )
    lost = sum(
        bool(carrier["rows"][key]["success"]) and not bool(free["rows"][key]["success"])
        for key in carrier["rows"]
    )
    retained_failures = 250 - retained - gained - lost
    breadth = sum(value > 0 for value in free["per_task"].values())
    tasks_above = sum(
        free["per_task"][key] > carrier["per_task"][key] for key in held_keys
    )
    suite_successes = {
        suite: sum(
            value
            for (task_suite, _task_id), value in free["per_task"].items()
            if task_suite == suite
        )
        for suite in SUITES
    }
    gate = config["gate"]
    checks = {
        "relative_recovery": recovery >= float(gate["relative_recovery_minimum"]),
        "breadth": breadth >= int(gate["breadth_minimum"]),
        "tasks_above_carrier": tasks_above >= int(gate["tasks_above_carrier_minimum"]),
        "carrier_successes_retained": retained
        >= int(gate["carrier_successes_retained_minimum"]),
        "goal_nonzero": suite_successes["libero_goal"] > 0,
        "long_nonzero": suite_successes["libero_10"] > 0,
        "single_complete_rank16": unique_complete,
    }
    payload = {
        "schema_version": G1_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "non_pass",
        "question": (
            "whether real native X/Y banks plus signed pooling contain a strong "
            "closed-loop rank4 residual for every held task"
        ),
        "claim_boundary": "task-local free-code capacity only; shared Program attention is unproven",
        "strict250": {
            name: {
                "path": arm["path"],
                "bytes": arm["bytes"],
                "contract_reference": arm["contract_reference"],
                "arm": arm["arm"],
                "successes": arm["successes"],
                "breadth": arm["breadth"],
            }
            for name, arm in arms.items()
        },
        "per_task": [
            {
                "suite": key[0],
                "task_id": key[1],
                **{name: arm["per_task"][key] for name, arm in arms.items()},
            }
            for key in sorted(held_keys)
        ],
        "per_suite": [
            {
                "suite": suite,
                **{
                    name: sum(
                        value
                        for (task_suite, _task_id), value in arm["per_task"].items()
                        if task_suite == suite
                    )
                    for name, arm in arms.items()
                },
            }
            for suite in SUITES
        ],
        "metrics": {
            "relative_recovery": recovery,
            "recovery_numerator": int(free["successes"]) - int(carrier["successes"]),
            "recovery_denominator": denominator,
            "breadth": breadth,
            "tasks_above_carrier": tasks_above,
            "suite_successes": suite_successes,
            "carrier_successes_retained": retained,
            "gained": gained,
            "lost": lost,
            "retained_failures": retained_failures,
            "churn": gained + lost,
        },
        "checks": checks,
        "paired_authority": {
            "task_state_keys": 250,
            "source_model": True,
            "tokenizer": True,
            "source_normalization": True,
            "environment_and_policy_rng": True,
        },
        "single_complete_rank16": unique_complete,
        "shuffled_or_reversed_use": False,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        if read_json(output_path) != payload:
            raise ValueError("existing G1 Gate report differs")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_native_factor_g1_v1.json",
    )
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--free-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_g1_gate_report(
        config_path=args.config,
        asset_root=args.asset_root,
        free_results=args.free_results,
        output_path=args.output,
    )
    print(f"G1 Gate: {report['status']} {report['metrics']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
