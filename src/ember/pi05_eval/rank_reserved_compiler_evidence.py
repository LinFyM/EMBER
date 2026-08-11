"""Decision evidence for the one-time old134 compiler-only diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_FAMILY,
    load_rank_reserved_config,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.analysis import _assert_row_pairing
from ember.pi05_eval.paired_metrics import paired_transition_summary
from ember.pi05_eval.rank_reserved_compiler_diagnostic import (
    COMPILER_DIAGNOSTIC_ONLINE_COMMIT,
    COMPILER_DIAGNOSTIC_ONLINE_ROOT,
    COMPILER_DIAGNOSTIC_SOURCE_COMMIT,
    COMPILER_DIAGNOSTIC_SOURCE_ROOT,
    compiler_diagnostic_output_path,
    validate_compiler_diagnostic_contract,
)
from ember.pi05_eval.rank_reserved_gate import _panel, rank_reserved_macro0_evidence
from ember.pi05_eval_contract import load_run_contract
from ember.pi05_source_checkpoint import write_json_atomic


COMPILER_DIAGNOSTIC_EVIDENCE_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_compiler_only_decision_evidence_v1"
)
COMPILER_DIAGNOSTIC_EVIDENCE = "compiler_only_diagnostic_evidence.json"


def _success_set_record(keys: set[tuple[str, int, int]]) -> dict[str, Any]:
    ordered = sorted(keys)
    task_counts: dict[tuple[str, int], int] = {}
    for suite, task_id, _ in ordered:
        task_counts[(suite, task_id)] = task_counts.get((suite, task_id), 0) + 1
    return {
        "count": len(ordered),
        "per_task": [
            {"suite": suite, "task_id": task_id, "successes": count}
            for (suite, task_id), count in sorted(task_counts.items())
        ],
        "episode_keys": [list(key) for key in ordered],
    }


def _success_set_evidence(
    old_success: set[tuple[str, int, int]],
    compiler_success: set[tuple[str, int, int]],
    online_success: set[tuple[str, int, int]],
) -> dict[str, Any]:
    arms = {
        "old134": old_success,
        "compiler_only": compiler_success,
        "online128": online_success,
    }
    return {
        "arms": {name: _success_set_record(values) for name, values in arms.items()},
        "three_way_intersection": _success_set_record(
            old_success & compiler_success & online_success
        ),
        "three_way_union": _success_set_record(
            old_success | compiler_success | online_success
        ),
        "compiler_only_retained_from_old134": _success_set_record(
            compiler_success & old_success
        ),
        "compiler_only_shared_with_online128": _success_set_record(
            compiler_success & online_success
        ),
        "compiler_only_only": _success_set_record(
            compiler_success - old_success - online_success
        ),
        "old134_only_relative_to_compiler": _success_set_record(
            old_success - compiler_success
        ),
        "online128_only_relative_to_compiler": _success_set_record(
            online_success - compiler_success
        ),
    }


def _validated_panels(output_dir: Path) -> dict[str, Any]:
    contract = load_run_contract(output_dir / "run_contract.json")
    authority = validate_compiler_diagnostic_contract(
        output_dir, contract, require_cache_ready=True
    )
    config = load_rank_reserved_config(RANK_RESERVED_CANONICAL_CONFIG)
    original = rank_reserved_macro0_evidence(config)
    expected = authority["original_gate_b_nonpass"]
    if (
        original.get("passed") is not False
        or original.get("immutable_reference_valid") is not True
        or original.get("old_full_rank_macro0", {}).get("correct")
        != int(authority["source_old134"]["correct"])
        or original.get("old_full_rank_macro0", {}).get("breadth")
        != int(authority["source_old134"]["breadth"])
        or original.get("new_rank14_macro0", {}).get("correct")
        != int(expected["correct"])
        or original.get("new_rank14_macro0", {}).get("breadth")
        != int(expected["breadth"])
        or original.get("paired_transition", {}).get("overall", {}).get("gained")
        != int(expected["gained_from_old134"])
        or original.get("paired_transition", {}).get("overall", {}).get("lost")
        != int(expected["lost_from_old134"])
    ):
        raise Pi05EvaluationError("original Gate B no longer recomputes as 128/nonpass")
    old_root = compiler_diagnostic_output_path(
        COMPILER_DIAGNOSTIC_SOURCE_ROOT, label="compiler-only old134 evidence"
    )
    online_root = compiler_diagnostic_output_path(
        COMPILER_DIAGNOSTIC_ONLINE_ROOT, label="compiler-only online128 evidence"
    )
    _, old_rows, old_summary = _panel(
        old_root,
        family="v6_condition_residual_v2",
        macro=0,
        kind="historical_v6_macro400_load_only",
        expected_commit=COMPILER_DIAGNOSTIC_SOURCE_COMMIT,
    )
    _, compiler_rows, compiler_summary = _panel(
        output_dir,
        family=RANK_RESERVED_FAMILY,
        macro=0,
        kind="v6_qv_rank14_zero_program_load_only",
        expected_commit=str(contract["git"]["commit"]),
        config_path=RANK_RESERVED_CANONICAL_CONFIG,
    )
    _, online_rows, online_summary = _panel(
        online_root,
        family=RANK_RESERVED_FAMILY,
        macro=0,
        kind="v6_qv_rank14_zero_program_load_only",
        expected_commit=COMPILER_DIAGNOSTIC_ONLINE_COMMIT,
        config_path=RANK_RESERVED_CANONICAL_CONFIG,
    )
    _assert_row_pairing(old_rows, compiler_rows, require_same_actual_video=True)
    _assert_row_pairing(compiler_rows, online_rows, require_same_actual_video=True)
    _assert_row_pairing(old_rows, online_rows, require_same_actual_video=True)
    if (
        int(old_summary["overall"]["successes"])
        != int(authority["source_old134"]["correct"])
        or int(old_summary["nonzero_task_breadth"])
        != int(authority["source_old134"]["breadth"])
    ):
        raise Pi05EvaluationError("old134 paired reference changed")
    return {
        "contract": contract,
        "authority": authority,
        "config": config,
        "original": original,
        "old_rows": old_rows,
        "compiler_rows": compiler_rows,
        "online_rows": online_rows,
        "old_summary": old_summary,
        "compiler_summary": compiler_summary,
        "online_summary": online_summary,
    }


def compiler_diagnostic_evidence(output_dir: Path) -> dict[str, Any]:
    """Build the old134/compiler-only/online128 paired evidence triangle."""

    output_dir = output_dir.resolve()
    panels = _validated_panels(output_dir)
    old_rows = panels["old_rows"]
    compiler_rows = panels["compiler_rows"]
    online_rows = panels["online_rows"]
    transitions = {
        "old134_to_compiler_only": paired_transition_summary(
            list(old_rows.values()), list(compiler_rows.values())
        ),
        "compiler_only_to_online128": paired_transition_summary(
            list(compiler_rows.values()), list(online_rows.values())
        ),
        "old134_to_online128": paired_transition_summary(
            list(old_rows.values()), list(online_rows.values())
        ),
    }
    successes = {
        "old": {key for key, row in old_rows.items() if row["success"]},
        "compiler": {
            key for key, row in compiler_rows.items() if row["success"]
        },
        "online": {key for key, row in online_rows.items() if row["success"]},
    }
    compiler_summary = panels["compiler_summary"]
    correct = int(compiler_summary["overall"]["successes"])
    breadth = int(compiler_summary["nonzero_task_breadth"])
    gates = panels["config"]["evaluation"]["gates"]
    counterfactual_passed = (
        correct >= int(gates["macro0_correct_min"])
        and breadth >= int(gates["macro0_breadth_min"])
        and int(transitions["old134_to_compiler_only"]["overall"]["lost"])
        <= int(gates["macro0_lost_to_paired_old134_max"])
    )
    evidence = {
        "schema_version": COMPILER_DIAGNOSTIC_EVIDENCE_SCHEMA,
        "scientific_role": panels["authority"]["decision"]["scientific_role"],
        "original_gate_b": panels["original"],
        "original_gate_b_passed": False,
        "old134": {
            "correct": int(panels["old_summary"]["overall"]["successes"]),
            "breadth": int(panels["old_summary"]["nonzero_task_breadth"]),
        },
        "compiler_only": {"correct": correct, "breadth": breadth},
        "online_regenerated": {
            "correct": int(panels["online_summary"]["overall"]["successes"]),
            "breadth": int(panels["online_summary"]["nonzero_task_breadth"]),
        },
        "triangle": transitions,
        "success_sets": _success_set_evidence(
            successes["old"], successes["compiler"], successes["online"]
        ),
        "counterfactual_gate_passed": counterfactual_passed,
        "counterfactual_gate": {
            "correct_min": int(gates["macro0_correct_min"]),
            "breadth_min": int(gates["macro0_breadth_min"]),
            "lost_max": int(gates["macro0_lost_to_paired_old134_max"]),
        },
        "retroactively_changes_original_gate_b": False,
        "authorizes_cycle1": False,
    }
    write_json_atomic(output_dir / COMPILER_DIAGNOSTIC_EVIDENCE, evidence)
    return evidence


def compiler_evidence_run(args: Any) -> dict[str, Any]:
    result = compiler_diagnostic_evidence(args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return result
