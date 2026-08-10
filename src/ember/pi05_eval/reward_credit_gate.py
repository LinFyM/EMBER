"""Strict paired support gate for Reward-Credit cycle1 to cycle2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_eval.analysis import (
    SIX_ARM_CONDITIONS,
    _assert_row_pairing,
    _formal_panel_index,
    _method_macro,
    _validated_roots,
    _writer_family,
    six_arm_paired_analysis,
)
from ember.pi05_eval.paired_metrics import (
    EpisodeKey,
    paired_transition_summary,
    summarize_panel,
)


DECISION_EVIDENCE_SCHEMA = "ember_pi05_v6_reward_credit_decision_evidence_v1"
SIX_ARM_EVIDENCE_SCHEMA = "ember_pi05_v6_reward_credit_six_arm_evidence_v1"
CONTROL_TRIGGER_EVIDENCE_SCHEMA = (
    "ember_pi05_v6_reward_credit_control_trigger_evidence_v1"
)


def _fail(message: str) -> None:
    raise Pi05EvaluationError(message)


def _registered_roots(
    run_decision: Mapping[str, Any],
    config_decision: Mapping[str, Any],
    *,
    macro: int,
    condition: str,
) -> tuple[Any, Any]:
    if condition == "correct":
        key = f"macro{macro}_registered_root"
        return run_decision[key], config_decision[key]
    key = f"macro{macro}_control_registered_roots"
    return run_decision[key][condition], config_decision[key][condition]


def _evaluation_contract_matches(
    contract: Mapping[str, Any],
    *,
    checkpoint: Path,
    output_dir: Path,
    run_commit: str,
    requested_condition: str,
) -> bool:
    adapter = contract.get("adapter", {})
    tasks = contract.get("tasks", ())
    try:
        adapter_checkpoint: Path | None = Path(
            str(adapter.get("writer_asset", {}).get("checkpoint", ""))
        ).resolve()
        family = _writer_family(adapter)[0]
    except (OSError, RuntimeError, Pi05EvaluationError):
        adapter_checkpoint = None
        family = ""
    observed = {
        "mode": contract.get("mode"),
        "role": contract.get("role"),
        "output_dir": contract.get("output_dir"),
        "family": family,
        "condition": adapter.get("video_condition"),
        "sampling": adapter.get("video_schedule", {}).get("sampling_mode"),
        "checkpoint": adapter_checkpoint,
        "commit": contract.get("git", {}).get("commit"),
    }
    expected = {
        "mode": "formal",
        "role": "validation",
        "output_dir": str(output_dir),
        "family": "v6_reward_credit_program_v1",
        "condition": requested_condition,
        "sampling": "without_replacement",
        "checkpoint": checkpoint,
        "commit": run_commit,
    }
    return (
        observed == expected
        and len(tasks) == 8
        and all(
            task.get("split_role") == "validation"
            and tuple(task.get("init_state_ids", ())) == tuple(range(50))
            for task in tasks
        )
        and git_state_is_clean_pushed_or_frozen_authority(contract.get("git", {}))
    )


def _registration_matches(
    *,
    checkpoint: Path,
    training_root: Path,
    configured_training_root: Path,
    configured_output: Path,
    output_dir: Path,
    macro: int,
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint_contract: Any,
    registered: Any,
    run_commit: str,
    requested_condition: str,
    evaluation_contract: Mapping[str, Any],
) -> bool:
    return (
        checkpoint.parent.name == "checkpoints"
        and checkpoint
        == configured_training_root / "checkpoints" / f"macro_{macro:08d}"
        and training_root.resolve() == configured_training_root
        and run.get("schema_version") == V6_PRIOR_RUN_SCHEMA
        and run.get("mode") == "formal"
        and run.get("config", {}).get("schema") == V6_PRIOR_CONFIG_SCHEMA
        and manifest.get("schema_version") == V6_PRIOR_CHECKPOINT_SCHEMA
        and manifest.get("next_macro") == macro
        and manifest.get("metrics_rows") == macro
        and isinstance(checkpoint_contract, Mapping)
        and checkpoint_contract.get("run_schema") == V6_PRIOR_RUN_SCHEMA
        and checkpoint_contract.get("mode") == "formal"
        and checkpoint_contract.get("config", {}).get("schema")
        == V6_PRIOR_CONFIG_SCHEMA
        and checkpoint_contract.get("git_commit") == run_commit
        and isinstance(registered, str)
        and bool(registered)
        and Path(registered).resolve() == configured_output == output_dir
        and _evaluation_contract_matches(
            evaluation_contract,
            checkpoint=checkpoint,
            output_dir=output_dir,
            run_commit=run_commit,
            requested_condition=requested_condition,
        )
    )


def validate_registered_reward_credit_output(
    args: Any,
    output_dir: Path,
    evaluation_contract: Mapping[str, Any],
) -> None:
    """Reject Reward-Credit strict400 outside its pre-registered root."""

    if (
        getattr(args, "mode", None) != "formal"
        or getattr(args, "expert_manifold_config", None) is None
        or getattr(args, "expert_manifold_checkpoint", None) is None
    ):
        return
    config_path = args.expert_manifold_config.resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError(
            "Reward-Credit evaluation config is unreadable"
        ) from error
    if config.get("schema_version") != V6_PRIOR_CONFIG_SCHEMA:
        return
    condition = getattr(args, "expert_manifold_video_condition", None)
    if condition not in SIX_ARM_CONDITIONS:
        raise Pi05EvaluationError(
            "formal Reward-Credit evaluation requires one registered six-arm condition"
        )

    checkpoint = args.expert_manifold_checkpoint.resolve()
    training_root = checkpoint.parent.parent
    try:
        run = json.loads(
            (training_root / "run_contract.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (checkpoint / "manifest.json").read_text(encoding="utf-8")
        )
        macro = manifest["next_macro"]
        if type(macro) is not int or macro not in (1, 2):
            raise KeyError("next_macro")
        checkpoint_contract = manifest["checkpoint_contract"]
        run_decision = run["decision_evaluation"]
        config_formal = config["formal_run"]
        config_decision = config_formal["decision_evaluation"]
        registered, configured = _registered_roots(
            run_decision,
            config_decision,
            macro=macro,
            condition=condition,
        )
        run_commit = run["git"]["commit"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise Pi05EvaluationError(
            "Reward-Credit evaluation registration is incomplete"
        ) from error
    config_repo_root = config_path.parents[1]
    configured_training_root = (
        config_repo_root / config_formal["registered_output_root"]
    ).resolve()
    configured_output = (config_repo_root / str(configured)).resolve()
    valid = _registration_matches(
        checkpoint=checkpoint,
        training_root=training_root,
        configured_training_root=configured_training_root,
        configured_output=configured_output,
        output_dir=output_dir,
        macro=macro,
        run=run,
        manifest=manifest,
        checkpoint_contract=checkpoint_contract,
        registered=registered,
        run_commit=run_commit,
        requested_condition=condition,
        evaluation_contract=evaluation_contract,
    )
    if not valid:
        raise Pi05EvaluationError(
            "Reward-Credit evaluation output is not its pre-registered root"
        )
    if condition != "correct":
        resolved_decision = {
            **config_decision,
            **{
                f"macro{candidate_macro}_registered_root": str(
                    (
                        config_repo_root
                        / config_decision[f"macro{candidate_macro}_registered_root"]
                    ).resolve()
                )
                for candidate_macro in (1, 2)
            },
        }
        load_reward_credit_control_trigger_evidence(
            training_root=training_root,
            current_checkpoint=checkpoint,
            macro=macro,
            expected_commit=run_commit,
            decision_evaluation=resolved_decision,
            decision_gates=config_formal["decision_gates"],
        )


def _baseline_record(
    result: Mapping[str, Any],
    *,
    root: Path,
    expected_commit: str,
    expected_correct: int,
    expected_breadth: int,
) -> tuple[dict[EpisodeKey, Mapping[str, Any]], dict[str, Any]]:
    rows = _formal_panel_index(result)
    summary = summarize_panel(list(rows.values()))
    correct = int(summary["overall"]["successes"])
    breadth = int(summary["nonzero_task_breadth"])
    if not (
        _writer_family(result["adapter"])[0] == "v6_condition_residual_v2"
        and result["adapter"].get("video_condition") == "correct"
        and _method_macro(
            result,
            allowed_macros=(0,),
            context="Reward-Credit macro0 reference",
        )
        == 0
        and result.get("paired_control", {}).get("git", {}).get("commit")
        == expected_commit
        and correct == expected_correct
        and breadth == expected_breadth
    ):
        _fail("Reward-Credit macro0 reference changed")
    return rows, {
        "root": str(root.resolve()),
        "git_commit": expected_commit,
        "correct": correct,
        "breadth": breadth,
    }


def _candidate_rows(
    result: Mapping[str, Any],
    *,
    resume_checkpoint: Path,
    expected_commit: str,
    expected_macro: int = 1,
) -> dict[EpisodeKey, Mapping[str, Any]]:
    rows = _formal_panel_index(result)
    asset = result["adapter"].get("writer_asset", {})
    manifest = asset.get("manifest", {})
    try:
        checkpoint_matches = Path(str(asset.get("checkpoint", ""))).resolve() == (
            resume_checkpoint.resolve()
        )
    except (OSError, RuntimeError):
        checkpoint_matches = False
    if not (
        _writer_family(result["adapter"])[0] == "v6_reward_credit_program_v1"
        and result["adapter"].get("video_condition") == "correct"
        and _method_macro(
            result,
            allowed_macros=(expected_macro,),
            context=f"Reward-Credit macro{expected_macro} gate",
        )
        == expected_macro
        and result.get("paired_control", {}).get("git", {}).get("commit")
        == expected_commit
        and checkpoint_matches
        and asset.get("training_mode") == "formal"
        and isinstance(manifest, Mapping)
        and manifest.get("schema") == V6_PRIOR_CHECKPOINT_SCHEMA
    ):
        _fail(f"Reward-Credit macro{expected_macro} checkpoint identity changed")
    return rows


def _registered_candidate_correct(
    *,
    root: Path,
    checkpoint: Path,
    macro: int,
    expected_commit: str,
) -> int:
    normalized = root.resolve()
    results = _validated_roots((normalized,))
    rows = _candidate_rows(
        results[str(normalized)],
        resume_checkpoint=checkpoint,
        expected_commit=expected_commit,
        expected_macro=macro,
    )
    return int(summarize_panel(list(rows.values()))["overall"]["successes"])


def load_reward_credit_control_trigger_evidence(
    *,
    training_root: Path,
    current_checkpoint: Path,
    macro: int,
    expected_commit: str,
    decision_evaluation: Mapping[str, Any],
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize only informative same-checkpoint controls after correct400."""

    if macro not in (1, 2):
        _fail("Reward-Credit control trigger macro changed")
    correct_root = Path(
        str(decision_evaluation[f"macro{macro}_registered_root"])
    ).resolve()
    correct = _registered_candidate_correct(
        root=correct_root,
        checkpoint=current_checkpoint,
        macro=macro,
        expected_commit=expected_commit,
    )
    first_threshold = int(decision_gates["first_full_six_arm_correct_min"])
    goal_threshold = int(decision_gates["goal_full_six_arm_correct_min"])
    previous_correct: int | None = None
    required = correct >= first_threshold
    reason = "first_checkpoint_at_or_above_control_threshold"
    if macro == 2:
        previous_correct = _registered_candidate_correct(
            root=Path(str(decision_evaluation["macro1_registered_root"])),
            checkpoint=training_root / "checkpoints/macro_00000001",
            macro=1,
            expected_commit=expected_commit,
        )
        first_new_trigger = previous_correct < first_threshold <= correct
        goal_checkpoint_trigger = correct >= goal_threshold
        required = first_new_trigger or goal_checkpoint_trigger
        reason = (
            "first_checkpoint_at_or_above_control_threshold"
            if first_new_trigger
            else "goal_candidate_requires_same_checkpoint_controls"
        )
    if not required:
        _fail("Reward-Credit control condition is not authorized by correct400")
    return {
        "schema_version": CONTROL_TRIGGER_EVIDENCE_SCHEMA,
        "macro": macro,
        "correct_root": str(correct_root),
        "correct": correct,
        "previous_correct": previous_correct,
        "reason": reason,
        "support_gate_independent": True,
    }


def reward_credit_decision_evidence(
    results_by_root: Mapping[str, Mapping[str, Any]],
    *,
    macro0_root: Path,
    macro1_root: Path,
    resume_checkpoint: Path,
    expected_macro0_commit: str,
    expected_macro0_correct: int,
    expected_macro0_breadth: int,
    expected_current_commit: str,
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {str(macro0_root.resolve()), str(macro1_root.resolve())}
    if set(results_by_root) != expected:
        _fail("Reward-Credit decision roots changed")
    baseline_rows, baseline = _baseline_record(
        results_by_root[str(macro0_root.resolve())],
        root=macro0_root,
        expected_commit=expected_macro0_commit,
        expected_correct=expected_macro0_correct,
        expected_breadth=expected_macro0_breadth,
    )
    candidate_rows = _candidate_rows(
        results_by_root[str(macro1_root.resolve())],
        resume_checkpoint=resume_checkpoint,
        expected_commit=expected_current_commit,
    )
    _assert_row_pairing(baseline_rows, candidate_rows, require_same_actual_video=True)
    candidate_values = list(candidate_rows.values())
    summary = summarize_panel(candidate_values)
    transition = paired_transition_summary(
        list(baseline_rows.values()), candidate_values
    )["overall"]
    correct = int(summary["overall"]["successes"])
    breadth = int(summary["nonzero_task_breadth"])
    checks = {
        "correct": correct >= int(decision_gates["macro1_support_correct_min"]),
        "lost_to_macro0": int(transition["lost"])
        <= int(decision_gates["macro1_support_lost_to_macro0_max"]),
        "breadth": breadth >= int(decision_gates["macro1_support_breadth_min"]),
        "gained_exceeds_lost": (
            not bool(decision_gates["macro1_support_gained_must_exceed_lost"])
            or int(transition["gained"]) > int(transition["lost"])
        ),
    }
    return {
        "schema_version": DECISION_EVIDENCE_SCHEMA,
        "macro0": baseline,
        "macro1": {
            "root": str(macro1_root.resolve()),
            "git_commit": expected_current_commit,
            "checkpoint": str(resume_checkpoint.resolve()),
            "correct": correct,
            "breadth": breadth,
        },
        "transition": {
            name: transition[name]
            for name in (
                "retained_success",
                "gained",
                "lost",
                "retained_failure",
                "net",
                "churn",
            )
        },
        "checks": checks,
        "passed": all(checks.values()),
        "six_arm_required": correct
        >= int(decision_gates["first_full_six_arm_correct_min"]),
    }


def reward_credit_six_arm_evidence(
    analysis: Mapping[str, Any],
    *,
    macro: int,
    decision_evaluation: Mapping[str, Any],
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate registered same-checkpoint controls and report the Goal causality gate."""

    if macro not in (1, 2):
        _fail("Reward-Credit six-arm macro changed")
    expected = {
        "correct": str(
            Path(decision_evaluation[f"macro{macro}_registered_root"]).resolve()
        ),
        **{
            condition: str(Path(path).resolve())
            for condition, path in decision_evaluation[
                f"macro{macro}_control_registered_roots"
            ].items()
        },
    }
    observed = {
        str(row["condition"]): str(Path(row["root"]).resolve())
        for row in analysis.get("roots", ())
    }
    if (
        analysis.get("method_family") != "v6_reward_credit_program_v1"
        or int(analysis.get("winner", {}).get("method_macro", -1)) != macro
        or observed != expected
    ):
        _fail("Reward-Credit six-arm roots or checkpoint changed")
    arm_successes = {
        condition: int(analysis["arms"][condition]["overall"]["successes"])
        for condition in SIX_ARM_CONDITIONS
    }
    correct = arm_successes["correct"]
    same_ratio = arm_successes["same_task_other"] / max(correct, 1)
    negative_checks = {
        condition: correct > arm_successes[condition]
        for condition in (
            "cross_suite_wrong",
            "shuffled",
            "reversed",
            "no_video",
        )
    }
    same_robust = same_ratio >= float(
        decision_gates["goal_same_task_other_correct_ratio_min"]
    )
    threshold_reached = correct >= int(decision_gates["goal_full_six_arm_correct_min"])
    goal_checks = {
        "correct_threshold": threshold_reached,
        "correct_exceeds_all_negative_controls": (
            not bool(decision_gates["goal_correct_strictly_exceeds_negative_controls"])
            or all(negative_checks.values())
        ),
        "same_task_other_robust": same_robust,
    }
    return {
        "schema_version": SIX_ARM_EVIDENCE_SCHEMA,
        "macro": macro,
        "arm_successes": arm_successes,
        "same_task_other_to_correct_ratio": same_ratio,
        "negative_control_checks": negative_checks,
        "goal_checks": goal_checks,
        "goal_passed": all(goal_checks.values()),
    }


def load_reward_credit_six_arm_evidence(
    *,
    correct_root: Path,
    control_roots: Mapping[str, Path],
    macro: int,
    decision_evaluation: Mapping[str, Any],
    decision_gates: Mapping[str, Any],
) -> dict[str, Any]:
    roots = (correct_root, *control_roots.values())
    analysis = six_arm_paired_analysis(_validated_roots(roots))
    return reward_credit_six_arm_evidence(
        analysis,
        macro=macro,
        decision_evaluation=decision_evaluation,
        decision_gates=decision_gates,
    )


def reward_credit_six_arm_evidence_from_config(
    analysis: Mapping[str, Any],
    *,
    config_path: Path,
    expected_bytes: int,
) -> dict[str, Any]:
    """Bind a retained six-arm audit to the exact Reward-Credit config roots."""

    config_path = config_path.resolve()
    try:
        if config_path.stat().st_size != expected_bytes:
            _fail("Reward-Credit six-arm config bytes changed")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != V6_PRIOR_CONFIG_SCHEMA:
            _fail("Reward-Credit six-arm config schema changed")
        macro = int(analysis["winner"]["method_macro"])
        raw = config["formal_run"]["decision_evaluation"]
        repo_root = config_path.parents[1]
        decision = {
            **raw,
            f"macro{macro}_registered_root": str(
                (repo_root / raw[f"macro{macro}_registered_root"]).resolve()
            ),
            f"macro{macro}_control_registered_roots": {
                condition: str((repo_root / path).resolve())
                for condition, path in raw[
                    f"macro{macro}_control_registered_roots"
                ].items()
            },
        }
        gates = config["formal_run"]["decision_gates"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise Pi05EvaluationError("Reward-Credit six-arm config is invalid") from error
    return reward_credit_six_arm_evidence(
        analysis,
        macro=macro,
        decision_evaluation=decision,
        decision_gates=gates,
    )


def load_reward_credit_decision_evidence(**kwargs: Any) -> dict[str, Any]:
    roots = (kwargs["macro0_root"], kwargs["macro1_root"])
    return reward_credit_decision_evidence(_validated_roots(roots), **kwargs)
