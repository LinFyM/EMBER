"""Registered load-only behavior gates for the active rank-reserved Writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_CONFIG_SCHEMA,
    RANK_RESERVED_FAMILY,
    RANK_RESERVED_PROGRAM_REFERENCE,
    RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA,
    load_rank_reserved_config,
    rank_reserved_asset,
    rank_reserved_output_path,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.analysis import (
    SIX_ARM_CONDITIONS,
    _assert_row_pairing,
    _formal_panel_index,
    _method_macro,
    _validated_roots,
    _writer_family,
)
from ember.pi05_eval.paired_metrics import (
    EpisodeKey,
    paired_transition_summary,
    summarize_panel,
)
from ember.pi05_eval_contract import (
    git_state_is_clean_pushed_or_frozen_authority,
)


RANK_RESERVED_DECISION_EVIDENCE_SCHEMA = (
    "ember_pi05_v6_qv_rank_reserved_load_only_decision_evidence_v1"
)


def _fail(message: str) -> None:
    raise Pi05EvaluationError(message)


def _root(relative: str) -> Path:
    try:
        return rank_reserved_output_path(
            relative,
            label="rank-reserved evidence root",
        )
    except Exception as error:
        raise Pi05EvaluationError(
            "rank-reserved evidence root escaped canonical outputs"
        ) from error


def _asset_checkpoint(config: Mapping[str, Any], name: str) -> Path:
    try:
        selected = config["assets"][name]
        checkpoint = (
            RANK_RESERVED_CANONICAL_CONFIG.parents[1] / str(selected["checkpoint"])
        ).resolve()
        rank_reserved_asset(config, checkpoint)
    except Exception as error:
        raise Pi05EvaluationError(
            "rank-reserved asset checkpoint escaped its canonical authority"
        ) from error
    return checkpoint


def _config_record_matches(record: object, expected: Path) -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        observed = Path(str(record.get("path", "")))
        return (
            record.get("schema") == RANK_RESERVED_CONFIG_SCHEMA
            and int(record.get("bytes", -1)) == expected.stat().st_size
            and observed.name == expected.name
            and observed.parent.name == "configs"
        )
    except (OSError, TypeError, ValueError):
        return False


def _checkpoint_record_matches(asset: Mapping[str, Any], expected: Path) -> bool:
    try:
        observed = Path(str(asset.get("checkpoint", "")))
        if expected.resolve() != RANK_RESERVED_PROGRAM_REFERENCE.resolve():
            return observed.resolve() == expected.resolve()
        manifest = asset.get("manifest", {})
        manifest_path = Path(str(manifest.get("path", "")))
        return (
            observed.name == expected.name
            and observed.parent.name == "configs"
            and manifest_path.name == expected.name
            and manifest_path.parent.name == "configs"
            and int(manifest.get("bytes", -1)) == expected.stat().st_size
            and manifest.get("schema") == RANK_RESERVED_PROGRAM_REFERENCE_SCHEMA
        )
    except (OSError, TypeError, ValueError):
        return False


def _panel(
    root: Path,
    *,
    family: str,
    macro: int,
    kind: str,
    checkpoint: Path | None = None,
    expected_commit: str | None = None,
    config_path: Path | None = None,
) -> tuple[
    Mapping[str, Any],
    dict[EpisodeKey, Mapping[str, Any]],
    dict[str, Any],
]:
    normalized = root.resolve()
    results = _validated_roots((normalized,))
    result = results[str(normalized)]
    rows = _formal_panel_index(result)
    asset = result["adapter"]["writer_asset"]
    try:
        checkpoint_matches = checkpoint is None or _checkpoint_record_matches(
            asset, checkpoint
        )
        config_matches = config_path is None or _config_record_matches(
            result["adapter"].get("config"), config_path
        )
    except (OSError, RuntimeError):
        checkpoint_matches = False
        config_matches = False
    if not (
        _writer_family(result["adapter"])[0] == family
        and result["adapter"].get("video_condition") == "correct"
        and _method_macro(
            result,
            allowed_macros=(macro,),
            context="rank-reserved load-only gate",
        )
        == macro
        and asset.get("kind") == kind
        and checkpoint_matches
        and config_matches
        and (
            expected_commit is None
            or result.get("paired_control", {}).get("git", {}).get("commit")
            == expected_commit
        )
    ):
        _fail("rank-reserved load-only evidence identity changed")
    return result, rows, summarize_panel(list(rows.values()))


def _record(
    root: Path,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    asset = result["adapter"]["writer_asset"]
    return {
        "root": str(root.resolve()),
        "family": _writer_family(result["adapter"])[0],
        "method_macro": int(asset["method_macro"]),
        "checkpoint_kind": asset["kind"],
        "commit": result.get("paired_control", {}).get("git", {}).get("commit"),
        "correct": int(summary["overall"]["successes"]),
        "breadth": int(summary["nonzero_task_breadth"]),
    }


def rank_reserved_macro0_evidence(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute Gate B from the immutable old macro0 and new rank14 macro0."""

    evaluation = config["evaluation"]
    old_reference = evaluation["immutable_references"]["old_full_rank_macro0_correct"]
    old_root = _root(str(old_reference["root"]))
    new_root = _root(str(evaluation["registered_roots"]["macro0_correct"]))
    old_result, old_rows, old_summary = _panel(
        old_root,
        family="v6_condition_residual_v2",
        macro=0,
        kind="historical_v6_macro400_load_only",
        expected_commit=str(old_reference["commit"]),
    )
    macro0_checkpoint = _asset_checkpoint(config, "macro0")
    new_result, new_rows, new_summary = _panel(
        new_root,
        family=RANK_RESERVED_FAMILY,
        macro=0,
        kind=str(config["assets"]["macro0"]["kind"]),
        checkpoint=macro0_checkpoint,
        config_path=RANK_RESERVED_CANONICAL_CONFIG,
    )
    _assert_row_pairing(old_rows, new_rows, require_same_actual_video=True)
    transition = paired_transition_summary(
        list(old_rows.values()),
        list(new_rows.values()),
    )
    old_correct = int(old_summary["overall"]["successes"])
    old_breadth = int(old_summary["nonzero_task_breadth"])
    new_correct = int(new_summary["overall"]["successes"])
    new_breadth = int(new_summary["nonzero_task_breadth"])
    gates = evaluation["gates"]
    reference_valid = old_correct == int(
        old_reference["correct"]
    ) and old_breadth == int(old_reference["breadth"])
    passed = (
        reference_valid
        and new_correct >= int(gates["macro0_correct_min"])
        and new_breadth >= int(gates["macro0_breadth_min"])
        and int(transition["overall"]["lost"])
        <= int(gates["macro0_lost_to_paired_old134_max"])
    )
    return {
        "schema_version": RANK_RESERVED_DECISION_EVIDENCE_SCHEMA,
        "stage": "macro0_base_gate",
        "passed": passed,
        "immutable_reference_valid": reference_valid,
        "old_full_rank_macro0": _record(old_root, old_result, old_summary),
        "new_rank14_macro0": _record(new_root, new_result, new_summary),
        "paired_transition": transition,
        "gate": {
            "correct_min": int(gates["macro0_correct_min"]),
            "breadth_min": int(gates["macro0_breadth_min"]),
            "lost_max": int(gates["macro0_lost_to_paired_old134_max"]),
        },
        "per_suite_is_diagnostic_not_a_hard_gate": True,
    }


def rank_reserved_cycle1_evidence(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute Gate C and the four-root old/new compiler comparison."""

    macro0 = rank_reserved_macro0_evidence(config)
    if not macro0["passed"]:
        _fail("rank-reserved cycle1 is blocked by the macro0 base gate")
    evaluation = config["evaluation"]
    old_reference = evaluation["immutable_references"][
        "old_native_reward_cycle1_correct"
    ]
    old_reward_root = _root(str(old_reference["root"]))
    new_macro0_root = _root(str(evaluation["registered_roots"]["macro0_correct"]))
    new_cycle1_root = _root(str(evaluation["registered_roots"]["cycle1_correct"]))
    old_reward_result, old_reward_rows, old_reward_summary = _panel(
        old_reward_root,
        family="v6_reward_credit_program_v1",
        macro=1,
        kind="v6_condition_program_residual_checkpoint",
        expected_commit=str(old_reference["commit"]),
    )
    new_macro0_result, new_macro0_rows, new_macro0_summary = _panel(
        new_macro0_root,
        family=RANK_RESERVED_FAMILY,
        macro=0,
        kind=str(config["assets"]["macro0"]["kind"]),
        checkpoint=_asset_checkpoint(config, "macro0"),
        config_path=RANK_RESERVED_CANONICAL_CONFIG,
    )
    new_commit = str(
        new_macro0_result.get("paired_control", {}).get("git", {}).get("commit", "")
    )
    new_cycle1_result, new_cycle1_rows, new_cycle1_summary = _panel(
        new_cycle1_root,
        family=RANK_RESERVED_FAMILY,
        macro=1,
        kind=str(config["assets"]["cycle1"]["kind"]),
        checkpoint=_asset_checkpoint(config, "cycle1"),
        expected_commit=new_commit,
        config_path=RANK_RESERVED_CANONICAL_CONFIG,
    )
    _assert_row_pairing(
        new_macro0_rows,
        new_cycle1_rows,
        require_same_actual_video=True,
    )
    _assert_row_pairing(
        old_reward_rows,
        new_cycle1_rows,
        require_same_actual_video=True,
    )
    program_transition = paired_transition_summary(
        list(new_macro0_rows.values()),
        list(new_cycle1_rows.values()),
    )
    compiler_transition = paired_transition_summary(
        list(old_reward_rows.values()),
        list(new_cycle1_rows.values()),
    )
    old_reference_valid = int(old_reward_summary["overall"]["successes"]) == int(
        old_reference["correct"]
    ) and int(old_reward_summary["nonzero_task_breadth"]) == int(
        old_reference["breadth"]
    )
    gates = evaluation["gates"]
    overall = program_transition["overall"]
    new_correct = int(new_cycle1_summary["overall"]["successes"])
    new_breadth = int(new_cycle1_summary["nonzero_task_breadth"])
    passed = (
        old_reference_valid
        and new_correct >= int(gates["cycle1_correct_min"])
        and new_breadth >= int(gates["cycle1_breadth_min"])
        and int(overall["lost"]) <= int(gates["cycle1_lost_to_macro0_max"])
        and int(overall["gained"]) > int(overall["lost"])
    )
    return {
        "schema_version": RANK_RESERVED_DECISION_EVIDENCE_SCHEMA,
        "stage": "cycle1_program_gate",
        "passed": passed,
        "macro0_gate": macro0,
        "immutable_reward_reference_valid": old_reference_valid,
        "old_native_reward_cycle1": _record(
            old_reward_root,
            old_reward_result,
            old_reward_summary,
        ),
        "new_rank14_macro0": _record(
            new_macro0_root,
            new_macro0_result,
            new_macro0_summary,
        ),
        "new_rank14_plus2_cycle1": _record(
            new_cycle1_root,
            new_cycle1_result,
            new_cycle1_summary,
        ),
        "paired_program_transition": program_transition,
        "paired_compiler_transition": compiler_transition,
        "gate": {
            "correct_min": int(gates["cycle1_correct_min"]),
            "breadth_min": int(gates["cycle1_breadth_min"]),
            "lost_max": int(gates["cycle1_lost_to_macro0_max"]),
            "gained_must_exceed_lost": True,
        },
    }


def _implementation_lineage_matches(
    config: Mapping[str, Any], evaluation_commit: str
) -> bool:
    evidence = config["evaluation"].get("online_smoke_evidence")
    if not isinstance(evidence, Mapping):
        return False
    implementation_commit = str(evidence.get("run_commit", ""))
    if len(implementation_commit) != 40 or len(evaluation_commit) != 40:
        return False
    if implementation_commit == evaluation_commit:
        return True
    allowed = {
        "AGENTS.md",
        "README.md",
        "configs/pi05_v6_qv_rank_reserved_native_reward_v1.json",
        "docs/active_session_handoff.md",
        "docs/execution_brief.md",
        "docs/action_forecast_writer_qv_rank_reserved_native_reward_design.md",
        "task_plan.md",
        "findings.md",
        "progress.md",
    }
    try:
        ancestry = subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                evaluation_commit,
            ),
            cwd=RANK_RESERVED_CANONICAL_CONFIG.parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        changed = subprocess.run(
            (
                "git",
                "diff",
                "--name-only",
                implementation_commit,
                evaluation_commit,
            ),
            cwd=RANK_RESERVED_CANONICAL_CONFIG.parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    paths = {line for line in changed.stdout.splitlines() if line}
    return ancestry.returncode == 0 and bool(paths) and paths <= allowed


def _registered_gate_root(
    config: Mapping[str, Any],
    *,
    macro: int,
    condition: str,
) -> Path:
    roots = config["evaluation"]["registered_roots"]
    if macro == 0:
        if condition != "correct":
            _fail("rank14 macro0 authorizes only the correct-video base gate")
        return _root(str(roots["macro0_correct"]))
    if macro == 1:
        return _root(
            str(
                roots["cycle1_correct"]
                if condition == "correct"
                else roots["cycle1_controls"][condition]
            )
        )
    _fail("rank-reserved evaluation macro is not registered")


def _registered_contract_matches(
    evaluation_contract: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    family: str,
    asset: Mapping[str, Any],
    selected: Mapping[str, Any],
    adapter_checkpoint: Path,
    selected_checkpoint: Path,
    condition: str,
    output_dir: Path,
    registered: Path,
) -> bool:
    adapter = evaluation_contract.get("adapter", {})
    tasks = evaluation_contract.get("tasks", ())
    commit = str(evaluation_contract.get("git", {}).get("commit", ""))
    return all(
        (
            family == RANK_RESERVED_FAMILY,
            asset.get("kind") == selected["kind"],
            adapter_checkpoint == selected_checkpoint,
            bool(asset.get("enable_program_residual"))
            == bool(selected["enable_program_residual"]),
            adapter.get("video_condition") == condition,
            adapter.get("video_schedule", {}).get("sampling_mode")
            == "without_replacement",
            evaluation_contract.get("mode") == "formal",
            evaluation_contract.get("role") == "validation",
            evaluation_contract.get("output_dir") == str(output_dir.resolve()),
            output_dir.resolve() == registered,
            Path(
                str(evaluation_contract["writer_lora_cache"]["root"])
            ).resolve()
            == output_dir.resolve() / "writer_lora_cache",
            evaluation_contract["writer_lora_cache"]["identity"].get(
                "implementation_commit"
            )
            == commit,
            _implementation_lineage_matches(config, commit),
            len(tasks) == 8,
            all(
                task.get("split_role") == "validation"
                and tuple(task.get("init_state_ids", ())) == tuple(range(50))
                for task in tasks
            ),
            git_state_is_clean_pushed_or_frozen_authority(
                evaluation_contract.get("git", {})
            ),
        )
    )


def _enforce_gate_order(
    config: Mapping[str, Any],
    *,
    macro: int,
    condition: str,
    evaluation_commit: str,
) -> None:
    if macro != 1:
        return
    macro0_evidence = rank_reserved_macro0_evidence(config)
    if not macro0_evidence["passed"]:
        _fail("rank-reserved cycle1 is blocked by the macro0 base gate")
    if condition == "correct":
        return
    cycle1_evidence = rank_reserved_cycle1_evidence(config)
    if not cycle1_evidence["passed"]:
        _fail("rank-reserved controls are blocked by the cycle1 gate")
    if cycle1_evidence["new_rank14_plus2_cycle1"]["commit"] != evaluation_commit:
        _fail("rank-reserved controls must use the cycle1 evaluation commit")


def validate_registered_rank_reserved_output(
    args: Any,
    output_dir: Path,
    evaluation_contract: Mapping[str, Any],
) -> None:
    """Permit only the registered Gate-B/Gate-C formal roots in order."""

    if (
        getattr(args, "mode", None) != "formal"
        or getattr(args, "expert_manifold_config", None) is None
        or getattr(args, "expert_manifold_checkpoint", None) is None
    ):
        return
    config_path = args.expert_manifold_config.resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError(
            "rank-reserved evaluation config is unreadable"
        ) from error
    if raw.get("schema_version") != RANK_RESERVED_CONFIG_SCHEMA:
        return
    config = load_rank_reserved_config(config_path)
    if (
        config["evaluation"]["formal_status"]
        != "sealed_from_live_a40_rank_reserved_deployment_profile"
    ):
        _fail("formal rank-reserved evaluation is blocked by the live profile")
    condition = getattr(args, "expert_manifold_video_condition", None)
    if condition not in SIX_ARM_CONDITIONS:
        _fail(
            "formal rank-reserved evaluation requires one registered six-arm condition"
        )
    adapter = evaluation_contract.get("adapter", {})
    asset = adapter.get("writer_asset", {})
    try:
        family = _writer_family(adapter)[0]
        macro = int(asset.get("method_macro", -1))
        selected = config["assets"]["macro0" if macro == 0 else "cycle1"]
        selected_checkpoint = _asset_checkpoint(
            config, "macro0" if macro == 0 else "cycle1"
        )
        adapter_checkpoint = Path(str(asset.get("checkpoint", ""))).resolve()
    except (KeyError, OSError, RuntimeError, ValueError, Pi05EvaluationError):
        _fail("rank-reserved evaluation contract is invalid")
    registered = _registered_gate_root(
        config,
        macro=macro,
        condition=condition,
    )
    if not _registered_contract_matches(
        evaluation_contract,
        config=config,
        family=family,
        asset=asset,
        selected=selected,
        adapter_checkpoint=adapter_checkpoint,
        selected_checkpoint=selected_checkpoint,
        condition=condition,
        output_dir=output_dir,
        registered=registered,
    ):
        _fail("rank-reserved evaluation output is not its pre-registered root")
    _enforce_gate_order(
        config,
        macro=macro,
        condition=condition,
        evaluation_commit=str(
            evaluation_contract.get("git", {}).get("commit", "")
        ),
    )


def validate_prepared_rank_reserved_contract(
    output_dir: Path,
    evaluation_contract: Mapping[str, Any],
) -> None:
    """Re-run ordered v9 registration gates on launcher start and resume."""

    adapter = evaluation_contract.get("adapter", {})
    if adapter.get("schema_version") != (
        "ember_pi05_v6_qv_rank_reserved_native_reward_eval_adapter_v9"
    ):
        return
    validate_registered_rank_reserved_output(
        SimpleNamespace(
            mode=evaluation_contract.get("mode"),
            expert_manifold_config=Path(adapter["config"]["path"]),
            expert_manifold_checkpoint=Path(adapter["writer_asset"]["checkpoint"]),
            expert_manifold_video_condition=adapter["video_condition"],
        ),
        output_dir,
        evaluation_contract,
    )
