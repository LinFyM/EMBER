"""Paired functional Gate for candidate-level Program--bank interaction."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.joint_program_primal.evaluation import (
    FAMILY_NAMES,
    _family_record,
    _mean_recovery,
    _normalized,
    _panel_value,
    _positive_control_losses,
    _task_conditions,
)
from ember.ecp.joint_program_primal.evaluation_gate import (
    _distribution,
)
from ember.ecp.joint_program_primal.program_bank_interaction_training import (
    PROGRAM_BANK_INTERACTION_COMPLETION_SCHEMA,
    generated_interaction_rank16,
)
from ember.ecp.joint_program_primal.routing_control import (
    PROGRAM_BANK_INTERACTION_RUN_SCHEMA,
    PROGRAM_BANK_INTERACTION_STAGE,
    ROUTING_TASK_IDS,
    load_routing_control_config,
    prepare_routing_control_runtime,
)
from ember.ecp.joint_program_primal.routing_initialization import (
    R5_SHARED_FUNCTIONAL_CHART,
)
from ember.ecp.joint_program_primal.train_step import (
    prepare_program_bank_condition,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


PROGRAM_BANK_INTERACTION_GATE_SCHEMA = (
    "ember_ecp_program_bank_candidate_interaction_gate_v4"
)
PROGRAM_BANK_INTERACTION_EVALUATION_SCHEMA = (
    "ember_ecp_program_bank_candidate_interaction_evaluation_task_v4"
)
PROGRAM_BANK_INTERACTION_GATE_REPORT_SCHEMA = (
    "ember_ecp_program_bank_candidate_interaction_gate_report_v4"
)


def _git_commit_is_ancestor(
    repo_root: Path, ancestor: str, descendant: str
) -> bool:
    if len(ancestor) != 40 or len(descendant) != 40:
        return False
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _tracked_json_config_authority(
    recorded: Mapping[str, Any],
    runtime_path: Path,
    runtime_config: Mapping[str, Any],
    *,
    repo_root: Path,
    training_commit: str,
    relative_path: str,
) -> bool:
    recorded_path = Path(str(recorded.get("path", "")))
    tracked_path = Path(relative_path)
    if (
        tracked_path.is_absolute()
        or ".." in tracked_path.parts
        or len(recorded_path.parts) < len(tracked_path.parts)
        or recorded_path.parts[-len(tracked_path.parts) :]
        != tracked_path.parts
        or not runtime_path.is_file()
        or runtime_path.name != tracked_path.name
        or len(training_commit) != 40
    ):
        return False
    blob = subprocess.run(
        ["git", "show", f"{training_commit}:{tracked_path.as_posix()}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if blob.returncode != 0:
        return False
    try:
        tracked_config = json.loads(blob.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        int(recorded.get("bytes", -1))
        == len(blob.stdout)
        == runtime_path.stat().st_size
        and tracked_config == runtime_config
        and read_json(runtime_path) == runtime_config
    )


def _worker_commit_authority_matches(
    workers: Sequence[Mapping[str, Any]],
) -> bool:
    training = {
        str(row.get("checkpoint", {}).get("training_commit", ""))
        for row in workers
    }
    evaluators = {
        str(row.get("authority", {}).get("evaluator_commit", ""))
        for row in workers
    }
    worker_commits = {
        str(row.get("git", {}).get("commit", "")) for row in workers
    }
    return (
        len(training) == len(evaluators) == len(worker_commits) == 1
        and worker_commits == evaluators
        and all(len(value) == 40 for value in (*training, *evaluators))
    )


def load_program_bank_interaction_gate(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    evaluation = config.get("evaluation", {})
    gate = config.get("gate", {})
    efficiency = config.get("efficiency_diagnostics", {})
    wall = config.get("information_wall", {})
    if (
        config.get("schema_version") != PROGRAM_BANK_INTERACTION_GATE_SCHEMA
        or config.get("status")
        != "active_base_score_conditioned_bank_interaction_qualification"
        or config.get("training_config")
        != "configs/pi05_ecp_program_bank_candidate_interaction_v4.json"
        or config.get("authorities", {}).get("positive_control_root")
        != (
            "runs/outputs/"
            "pi05_ecp_j2_functional_positive_control_10task_"
            "c4704cb_gpu01p012345_20260829"
        )
        or config.get("checkpoint_optimizer_steps") != [70, 110]
        or evaluation.get("evaluated_task_ids") != list(ROUTING_TASK_IDS)
        or evaluation.get("functional_panel") != "panel_b"
        or evaluation.get("panel_visits") != 16
        or evaluation.get("correct_views") != "two_fit_plus_same_task_held"
        or evaluation.get("unseen_wrong_meta_bank_task") != 2
        or evaluation.get("unseen_wrong_target_bank_task") != 74
        or evaluation.get("unseen_wrong_video")
        != "lowest_sorted_task_holdout_video"
        or evaluation.get("conditioning_language") != "correct_task_exact_language"
        or evaluation.get("wrong_recovery_denominator")
        != "mean_two_correct_fit_free_primal_panel_b_losses"
        or evaluation.get("control_arms")
        != [
            "correct_interaction_on",
            "unseen_wrong_interaction_on",
            "same_unseen_wrong_interaction_off",
        ]
        or evaluation.get("selected_family_report_targets")
        != [0, 16, 34, 1, 17, 35, 36, 37]
        or gate.get("correct_fit_recovery_median_minimum") != 0.85
        or gate.get("same_task_held_recovery_median_minimum") != 0.80
        or gate.get("held_to_fit_minimum") != 0.85
        or gate.get("unseen_wrong_recovery_median_maximum") != 0.25
        or gate.get("correct_minus_wrong_median_minimum") != 0.50
        or gate.get("correct_better_than_wrong_required_tasks") != 10
        or gate.get("wrong_off_minus_on_median_minimum") != 0.40
        or gate.get("family_median_minimum") != 0.0
        or gate.get("maximum_adjacent_correct_fit_median_drop") != 0.05
        or efficiency.get("qualification_gate") is not False
        or efficiency.get("reported_metrics")
        != [
            "training_global_step_seconds_maximum",
            "training_global_step_seconds_median",
            "evaluation_to_training_wall",
        ]
        or wall.get("panel_b_backward_calls") != 0
        or wall.get("same_task_held_backward_calls") != 0
        or wall.get("unseen_wrong_backward_calls") != 0
        or wall.get("validation_or_test_reads") != 0
        or wall.get("fixed_routing_token_training_only") is not True
        or wall.get("wrong_bank_exact_language_fixed") is not True
        or wall.get("action_meta_installed") is not False
        or wall.get("single_complete_rank16") is not True
        or wall.get("shuffled_or_reversed_use") is not False
        or wall.get("deployment_candidate") is not False
    ):
        raise ValueError("unsupported Program-bank interaction Gate config")
    return config


def _training_world_size(
    runtime: Any,
    run_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> int:
    world_size = int(manifest.get("world_size", -1))
    topology = run_contract.get("world_topology", ())
    allowed = set(map(int, runtime.config["profile"]["allowed_world_sizes"]))
    if (
        world_size not in allowed
        or world_size > 6
        or not isinstance(topology, list)
        or len(topology) != world_size
        or sorted(int(row.get("rank", -1)) for row in topology)
        != list(range(world_size))
        or sorted(int(row.get("local_rank", -1)) for row in topology)
        != list(range(world_size))
        or len({str(row.get("device", "")) for row in topology}) != world_size
        or len({str(row.get("hostname", "")) for row in topology}) != 1
    ):
        raise ValueError("interaction training world authority changed")
    return world_size


def _checkpoint_authority(
    runtime: Any,
    *,
    compiler_run: Path,
    compiler_checkpoint: Path,
    gate: Mapping[str, Any],
    evaluator_commit: str,
) -> dict[str, Any]:
    compiler_run = compiler_run.resolve()
    compiler_checkpoint = compiler_checkpoint.resolve()
    step = checkpoint_macro(compiler_checkpoint)
    run_contract = read_json(compiler_run / "run_contract.json")
    manifest = read_json(compiler_checkpoint / "checkpoint_manifest.json")
    training_world_size = _training_world_size(runtime, run_contract, manifest)
    run_git = run_contract.get("git", {})
    training_commit = str(run_git.get("commit", ""))
    run_config = run_contract.get("config", {})
    run_base = run_contract.get("base_g3_config", {})
    initialization = run_contract.get("primal_scorer_initialization", {})
    inventory = run_contract.get("inventory", {})
    trainable_names = tuple(inventory.get("trainable_parameter_names", ()))
    expected_r5_checkpoint = str(
        (
            runtime.args.asset_root
            / runtime.config["authorities"]["r5_primal_scorer_checkpoint"]
        ).resolve()
    )
    expected_r5_gate = str(
        (
            runtime.args.asset_root
            / runtime.config["authorities"]["r5_gate_aggregate"]
        ).resolve()
    )
    files = manifest.get("files", {})
    expected = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(
            f"rank_{rank:02d}_state.pt"
            for rank in range(training_world_size)
        ),
    }
    repo_root = Path(__file__).resolve().parents[4]
    config_authority_matches = _tracked_json_config_authority(
        run_config,
        runtime.args.config,
        runtime.config,
        repo_root=repo_root,
        training_commit=training_commit,
        relative_path=str(gate["training_config"]),
    )
    base_config_authority_matches = _tracked_json_config_authority(
        run_base,
        runtime.args.base_config,
        read_json(runtime.args.base_config),
        repo_root=repo_root,
        training_commit=training_commit,
        relative_path=str(runtime.config["authorities"]["base_g3_config"]),
    )
    if (
        compiler_checkpoint.parent.parent != compiler_run
        or step not in set(map(int, gate["checkpoint_optimizer_steps"]))
        or run_contract.get("schema_version")
        != PROGRAM_BANK_INTERACTION_RUN_SCHEMA
        or run_contract.get("stage") != PROGRAM_BANK_INTERACTION_STAGE
        or run_contract.get("phase") != "joint"
        or run_contract.get("mode") != "formal"
        or run_git.get("branch") != ""
        or training_commit != run_git.get("authority_commit")
        or not _git_commit_is_ancestor(
            repo_root, training_commit, evaluator_commit
        )
        or not config_authority_matches
        or not base_config_authority_matches
        or run_contract.get("model") != runtime.config["model"]
        or run_contract.get("optimization") != runtime.config["optimization"]
        or run_contract.get("task_split") != runtime.config["task_split"]
        or run_contract.get("throughput_gate")
        != runtime.config["throughput_gate"]
        or run_contract.get("information_wall")
        != runtime.config["information_wall"]
        or initialization.get("kind") != R5_SHARED_FUNCTIONAL_CHART
        or initialization.get("checkpoint") != expected_r5_checkpoint
        or initialization.get("gate_aggregate") != expected_r5_gate
        or int(initialization.get("optimizer_step", -1)) != 110
        or initialization.get("fixed_routing_token_loaded") is not False
        or initialization.get("task_lookup_parameters_loaded") is not False
        or int(inventory.get("action_meta_module_count", -1)) != 0
        or int(inventory.get("action_meta_parameter_count", -1)) != 0
        or int(inventory.get("source_policy_trainable_parameter_count", -1)) != 0
        or int(inventory.get("native_stage0_trainable_parameter_count", -1)) != 0
        or int(inventory.get("scale_trainable_parameter_count", -1)) != 0
        or int(inventory.get("task_video_frame_free_parameter_count", -1)) != 0
        or not trainable_names
        or any(
            not name.startswith("compiler.interaction_scorer.")
            for name in trainable_names
        )
        or run_contract.get("diagnostic", {}).get(
            "candidate_interaction_qualification"
        )
        is not True
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != PROGRAM_BANK_INTERACTION_STAGE
        or int(manifest.get("next_macro", -1)) != step
        or manifest.get("run_contract_schema")
        != PROGRAM_BANK_INTERACTION_RUN_SCHEMA
        or set(files) != expected
    ):
        raise ValueError("interaction checkpoint authority changed")
    for name, record in files.items():
        candidate = compiler_checkpoint / name
        if not candidate.is_file() or candidate.stat().st_size != int(record["bytes"]):
            raise ValueError(f"interaction checkpoint file changed: {name}")
    runtime.writer_state.load_state_dict(
        load_file(
            str(compiler_checkpoint / "ecp.safetensors"),
            device=str(runtime.context.device),
        ),
        strict=True,
    )
    runtime.writer_state.requires_grad_(False).eval()
    runtime.program.eval()
    runtime.compiler.eval()
    if any(parameter.requires_grad for parameter in runtime.policy.parameters()):
        raise RuntimeError("interaction evaluator unfroze source policy")
    return {
        "optimizer_step": step,
        "path": str(compiler_checkpoint),
        "compiler_run": str(compiler_run),
        "training_commit": str(run_contract["git"]["commit"]),
        "world_size": training_world_size,
        "tensor_bytes": int(files["ecp.safetensors"]["bytes"]),
        "run_contract_bytes": (compiler_run / "run_contract.json").stat().st_size,
        "training_config_bytes": runtime.args.config.stat().st_size,
    }


def _unseen_wrong_condition(runtime: Any, task_id: int) -> tuple[int, Any]:
    meta = set(map(int, runtime.config["task_split"]["gradient_meta"]))
    wrong_task = 2 if task_id in meta else 74
    return wrong_task, _task_conditions(runtime, wrong_task)[0]


def interaction_task_assignments(
    runtime: Any, worker_count: int
) -> tuple[tuple[int, ...], ...]:
    if not 1 <= worker_count <= 6:
        raise ValueError("interaction evaluator worker count changed")
    costs = {}
    for task in ROUTING_TASK_IDS:
        first, second, held = _task_conditions(runtime, task)
        _, wrong = _unseen_wrong_condition(runtime, task)
        costs[task] = (
            first.sampled_frames
            + second.sampled_frames
            + held.sampled_frames
            + 2 * wrong.sampled_frames
        )
    rows: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    maximum_tasks = (len(ROUTING_TASK_IDS) + worker_count - 1) // worker_count
    for task in sorted(ROUTING_TASK_IDS, key=lambda value: (-costs[value], value)):
        eligible = [
            worker
            for worker in range(worker_count)
            if len(rows[worker]) < maximum_tasks
        ]
        worker = min(eligible, key=lambda value: (loads[value], value))
        rows[worker].append(task)
        loads[worker] += costs[task]
    return tuple(tuple(sorted(row)) for row in rows)


def _functional_recovery(value: Mapping[str, Any]) -> float:
    result = value.get("functional_recovery")
    if result is None:
        raise RuntimeError("interaction functional denominator is non-positive")
    return float(result)


def _positive_control_panel_b(
    root: Path, task_id: int
) -> tuple[dict[int, float], dict[str, Any], float]:
    losses, authority = _positive_control_losses(root, task_id)
    result = read_json(Path(authority["path"]))
    evaluation = result["evaluation"]
    rows = (*evaluation["fit_videos"], evaluation["held_video"])
    carriers = {float(row["panel_b"]["carrier_loss"]) for row in rows}
    if len(carriers) != 1:
        raise ValueError("interaction positive-control carrier changed")
    return losses, authority, carriers.pop()


def _evaluate_task(
    runtime: Any,
    *,
    task_id: int,
    gate: Mapping[str, Any],
    positive_control_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    teacher_reads = runtime.native_teachers.tensor_reads
    first, second, held = _task_conditions(runtime, task_id)
    correct_conditions = (first, second, held)
    free_reference, free_authority, positive_carrier = _positive_control_panel_b(
        positive_control_root, task_id
    )
    correct_rows: dict[int, dict[str, Any]] = {}
    outputs = {}
    for condition in correct_conditions:
        with torch.inference_mode():
            state, output, _ = generated_interaction_rank16(
                runtime,
                program_task_id=task_id,
                bank_condition=condition,
            )
        correct_rows[condition.video_demo] = _normalized(
            _panel_value(runtime, task_id=task_id, state=state),
            free_reference[condition.video_demo],
        )
        outputs[condition.video_demo] = output
        del state
    observed_carriers = {
        float(row["carrier_loss"]) for row in correct_rows.values()
    }
    if (
        len(observed_carriers) != 1
        or abs(observed_carriers.pop() - positive_carrier) > 1e-6
    ):
        raise RuntimeError("interaction Panel-B carrier authority changed")

    wrong_task, wrong_condition = _unseen_wrong_condition(runtime, task_id)
    with torch.inference_mode():
        wrong_on_state, _wrong_on_output, wrong_on_metrics = (
            generated_interaction_rank16(
                runtime,
                program_task_id=task_id,
                bank_condition=wrong_condition,
            )
        )
        wrong_off_state, _wrong_off_output, wrong_off_metrics = (
            generated_interaction_rank16(
                runtime,
                program_task_id=task_id,
                bank_condition=wrong_condition,
                interaction_off=True,
            )
        )
    reference_loss = statistics.fmean(
        free_reference[condition.video_demo] for condition in (first, second)
    )
    wrong_on = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_on_state),
        reference_loss,
    )
    wrong_off = _normalized(
        _panel_value(runtime, task_id=task_id, state=wrong_off_state),
        reference_loss,
    )
    del wrong_on_state, wrong_off_state

    for metrics in (wrong_on_metrics, wrong_off_metrics):
        if (
            int(metrics.get("conditioning_language_authority_id", -1))
            != task_id
            or int(metrics.get("video_bank_authority_id", -1)) != wrong_task
        ):
            raise RuntimeError("interaction wrong-bank exact-language pairing changed")

    if runtime.native_teachers.tensor_reads != teacher_reads:
        raise RuntimeError("interaction deployment evaluation read factor teachers")
    indices = tuple(map(int, gate["evaluation"]["selected_family_report_targets"]))
    family = {
        str(condition.video_demo): _family_record(
            runtime,
            task_id=task_id,
            condition=condition,
            output=outputs[condition.video_demo],
            indices=indices,
        )
        for condition in correct_conditions
    }
    diagnostic_teacher_reads = runtime.native_teachers.tensor_reads - teacher_reads
    inventory = runtime.run_contract["inventory"]
    if (
        int(inventory.get("action_meta_module_count", -1)) != 0
        or int(inventory.get("action_meta_parameter_count", -1)) != 0
    ):
        raise RuntimeError("interaction evaluator loaded Action Meta")
    fit_recovery = _mean_recovery(
        [correct_rows[row.video_demo] for row in (first, second)]
    )
    held_recovery = _functional_recovery(correct_rows[held.video_demo])
    wrong_on_recovery = _functional_recovery(wrong_on)
    wrong_off_recovery = _functional_recovery(wrong_off)
    return {
        "schema_version": PROGRAM_BANK_INTERACTION_EVALUATION_SCHEMA,
        "task": task_id,
        "role": runtime.panels[task_id].role,
        "fit_videos": [first.video_demo, second.video_demo],
        "held_video": held.video_demo,
        "correct": {str(key): value for key, value in correct_rows.items()},
        "functional_summary": {
            "fit_recovery": fit_recovery,
            "held_video_recovery": held_recovery,
            "held_to_fit": held_recovery / max(float(fit_recovery), 1e-12),
            "unseen_wrong_on_recovery": wrong_on_recovery,
            "unseen_wrong_off_recovery": wrong_off_recovery,
            "correct_minus_wrong": float(fit_recovery) - wrong_on_recovery,
            "wrong_off_minus_on": wrong_off_recovery - wrong_on_recovery,
            "correct_better_than_wrong": float(fit_recovery) > wrong_on_recovery,
        },
        "controls": {
            "unseen_wrong_task": wrong_task,
            "unseen_wrong_video": wrong_condition.video_demo,
            "conditioning_language_task": task_id,
            "free_primal_reference": {
                "method": "mean_two_correct_fit_free_primal_panel_b_losses",
                "loss": reference_loss,
            },
            "wrong_interaction_on": wrong_on,
            "wrong_interaction_off": wrong_off,
            "wrong_on_condition_metrics": wrong_on_metrics,
            "wrong_off_condition_metrics": wrong_off_metrics,
        },
        "family_diagnostic": family,
        "free_primal_authority": free_authority,
        "information_wall": {
            "deployment_native_teacher_tensor_reads": 0,
            "diagnostic_native_teacher_tensor_reads": diagnostic_teacher_reads,
            "panel_b_backward_calls": 0,
            "same_task_held_backward_calls": 0,
            "unseen_wrong_backward_calls": 0,
            "validation_or_test_reads": 0,
            "action_meta_installed": False,
            "action_meta_module_count": 0,
            "action_meta_parameter_count": 0,
            "single_complete_rank16": True,
            "K1_identity": True,
            "shuffled_or_reversed_use": False,
            "fixed_routing_token_training_only": True,
            "wrong_bank_exact_language_fixed": True,
            "deployment_candidate": False,
        },
        "task_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
    }


def evaluate_program_bank_interaction_worker(args: argparse.Namespace) -> None:
    state = git_state(Path(__file__).resolve().parents[4])
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal interaction evaluation requires detached authority")
    gate = load_program_bank_interaction_gate(args.gate_config)
    if args.config != (args.asset_root / gate["training_config"]).resolve():
        raise ValueError("interaction evaluator training authority changed")
    training_config = load_routing_control_config(args.config)
    positive_root = (
        args.asset_root / gate["authorities"]["positive_control_root"]
    ).resolve()
    if positive_root != (
        args.asset_root / training_config["authorities"]["positive_control_root"]
    ).resolve():
        raise ValueError("interaction positive-control authority changed")
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("interaction evaluator worker index changed")
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1:
        raise ValueError("interaction workers must be independent single GPUs")
    if (
        len(args.compiler_checkpoints) != 2
        or len(args.output_dirs) != 2
        or [checkpoint_macro(path) for path in args.compiler_checkpoints]
        != list(map(int, gate["checkpoint_optimizer_steps"]))
    ):
        raise ValueError("interaction paired checkpoint panel changed")
    runtime_args = argparse.Namespace(
        config=args.config,
        base_config=args.base_config,
        mode="profile",
        phase="joint",
        task=None,
        asset_root=args.asset_root,
        source_run=args.source_run,
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        data_root=args.data_root,
        output_dir=args.output_dirs[0]
        / f"worker_{args.worker_index:02d}_runtime",
        condition_cache_root=args.condition_cache_root,
        program_bank_condition_cache_root=(
            args.program_bank_condition_cache_root
        ),
        resume=None,
        stop_after_step=1,
        log_every=1,
        skip_routing_initialization=True,
    )
    runtime = None
    started = time.monotonic()
    try:
        runtime = prepare_routing_control_runtime(runtime_args, context)
        assignments = interaction_task_assignments(runtime, args.worker_count)
        teacher_reads = runtime.native_teachers.tensor_reads
        for task_id in assignments[args.worker_index]:
            first, second, held = _task_conditions(runtime, task_id)
            _, wrong = _unseen_wrong_condition(runtime, task_id)
            for condition in (first, second, held, wrong):
                prepared, _ = prepare_program_bank_condition(
                    runtime,
                    language_authority_id=task_id,
                    bank_condition=condition,
                )
                del prepared
        if runtime.native_teachers.tensor_reads != teacher_reads:
            raise RuntimeError("interaction cache prewarm read factor teachers")
        torch.cuda.synchronize(context.device)
        setup_seconds = time.monotonic() - started
        authority = {
            "compiler_run": str(args.compiler_run),
            "training_config": dict(
                read_json(args.compiler_run / "run_contract.json")["config"]
            ),
            "evaluator_config": {
                "path": str(args.config),
                "bytes": args.config.stat().st_size,
            },
            "gate_config": {
                "path": str(args.gate_config),
                "bytes": args.gate_config.stat().st_size,
            },
            "positive_control_root": str(positive_root),
            "evaluator_commit": str(state["commit"]),
        }
        for compiler_checkpoint, output_dir in zip(
            args.compiler_checkpoints, args.output_dirs, strict=True
        ):
            checkpoint_started = time.monotonic()
            checkpoint = _checkpoint_authority(
                runtime,
                compiler_run=args.compiler_run,
                compiler_checkpoint=compiler_checkpoint,
                gate=gate,
                evaluator_commit=str(state["commit"]),
            )
            torch.cuda.reset_peak_memory_stats(context.device)
            rows = []
            for task_id in assignments[args.worker_index]:
                rows.append(
                    _evaluate_task(
                        runtime,
                        task_id=task_id,
                        gate=gate,
                        positive_control_root=positive_root,
                    )
                )
                runtime.panel_batch_cache.clear()
            torch.cuda.synchronize(context.device)
            evaluation_seconds = time.monotonic() - checkpoint_started
            worker_dir = output_dir / f"worker_{args.worker_index:02d}"
            if worker_dir.exists():
                raise ValueError("interaction worker output already exists")
            worker_dir.mkdir(parents=True)
            payload = {
                "schema_version": PROGRAM_BANK_INTERACTION_EVALUATION_SCHEMA,
                "status": "complete",
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "assignments": [list(row) for row in assignments],
                "authority": authority,
                "checkpoint": checkpoint,
                "tasks": rows,
                "shared_runtime_setup_seconds": setup_seconds,
                "checkpoint_evaluation_seconds": evaluation_seconds,
                "elapsed_seconds": setup_seconds / 2.0 + evaluation_seconds,
                "physical_visible_device": __import__("os").environ.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "git": {"commit": state["commit"], "branch": state["branch"]},
            }
            write_json_atomic(worker_dir / "result.json", payload)
            write_json_atomic(
                worker_dir / "completion.json",
                {
                    "schema_version": PROGRAM_BANK_INTERACTION_EVALUATION_SCHEMA,
                    "worker_index": args.worker_index,
                    "task_count": len(rows),
                },
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _load_workers(
    output_dir: Path, worker_count: int, compiler_run: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks, workers = [], []
    for worker in range(worker_count):
        root = output_dir / f"worker_{worker:02d}"
        payload = read_json(root / "result.json")
        completion = read_json(root / "completion.json")
        rows = payload.get("tasks", [])
        if (
            payload.get("schema_version")
            != PROGRAM_BANK_INTERACTION_EVALUATION_SCHEMA
            or payload.get("status") != "complete"
            or int(payload.get("worker_index", -1)) != worker
            or int(payload.get("worker_count", -1)) != worker_count
            or completion.get("schema_version")
            != PROGRAM_BANK_INTERACTION_EVALUATION_SCHEMA
            or int(completion.get("worker_index", -1)) != worker
            or int(completion.get("task_count", -1)) != len(rows)
        ):
            raise ValueError("interaction worker evidence changed")
        assignments = payload.get("assignments")
        if (
            not isinstance(assignments, list)
            or len(assignments) != worker_count
            or [int(row["task"]) for row in rows]
            != list(map(int, assignments[worker]))
        ):
            raise ValueError("interaction worker assignment changed")
        tasks.extend(rows)
        workers.append(payload)
    assignments = [row.get("assignments") for row in workers]
    checkpoints = [row.get("checkpoint") for row in workers]
    authorities = [row.get("authority") for row in workers]
    if (
        len(tasks) != 10
        or sorted(int(row["task"]) for row in tasks) != list(ROUTING_TASK_IDS)
        or any(value != assignments[0] for value in assignments[1:])
        or sorted(int(task) for row in assignments[0] for task in row)
        != list(ROUTING_TASK_IDS)
        or any(value != checkpoints[0] for value in checkpoints[1:])
        or any(value != authorities[0] for value in authorities[1:])
        or Path(str(checkpoints[0].get("compiler_run", ""))).resolve()
        != compiler_run.resolve()
        or Path(str(authorities[0].get("compiler_run", ""))).resolve()
        != compiler_run.resolve()
        or not _worker_commit_authority_matches(workers)
    ):
        raise ValueError("interaction task coverage changed")
    return sorted(tasks, key=lambda row: int(row["task"])), workers


def _training_metrics_authority(compiler_run: Path) -> dict[str, float]:
    rows = []
    with (compiler_run / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    completion = read_json(compiler_run / "completion.json")
    if (
        len(rows) != 110
        or [int(row.get("optimizer_step", -1)) for row in rows]
        != list(range(1, 111))
        or completion.get("schema_version")
        != PROGRAM_BANK_INTERACTION_COMPLETION_SCHEMA
        or completion.get("status") != "complete"
        or completion.get("stage") != PROGRAM_BANK_INTERACTION_STAGE
        or completion.get("run_contract_schema")
        != PROGRAM_BANK_INTERACTION_RUN_SCHEMA
        or int(completion.get("completed_optimizer_steps", -1)) != 110
        or int(completion.get("completed_effective_steps", -1)) != 100
        or int(completion.get("metrics_rows", -1)) != 110
        or completion.get("checkpoint_optimizer_steps") != [70, 110]
        or completion.get("deployment_candidate") is not False
    ):
        raise ValueError("interaction training completion authority changed")
    elapsed = [float(row.get("elapsed_seconds", math.nan)) for row in rows]
    updates = [float(row.get("global_step_seconds", math.nan)) for row in rows]
    if (
        not all(math.isfinite(value) and value > 0.0 for value in (*elapsed, *updates))
        or any(right < left for left, right in zip(elapsed, elapsed[1:]))
        or abs(float(completion.get("elapsed_seconds", math.nan)) - elapsed[-1])
        > 1e-6
    ):
        raise ValueError("interaction training timing evidence changed")
    return {
        "elapsed_seconds": elapsed[-1],
        "global_step_seconds_maximum": max(updates),
        "global_step_seconds_median": statistics.median(updates),
    }


def _family_value(row: Mapping[str, Any], family: str) -> float:
    return float(
        row["family_diagnostic"][str(row["held_video"])]["family_recovery"][
            family
        ]
    )


def _information_wall_pass(tasks: Sequence[Mapping[str, Any]]) -> bool:
    expected_wall = {
        "deployment_native_teacher_tensor_reads": 0,
        "panel_b_backward_calls": 0,
        "same_task_held_backward_calls": 0,
        "unseen_wrong_backward_calls": 0,
        "validation_or_test_reads": 0,
        "action_meta_installed": False,
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "single_complete_rank16": True,
        "K1_identity": True,
        "shuffled_or_reversed_use": False,
        "fixed_routing_token_training_only": True,
        "wrong_bank_exact_language_fixed": True,
        "deployment_candidate": False,
    }
    return all(
        {
            key: value
            for key, value in row["information_wall"].items()
            if key != "diagnostic_native_teacher_tensor_reads"
        }
        == expected_wall
        and int(row["information_wall"]["diagnostic_native_teacher_tensor_reads"])
        >= 0
        for row in tasks
    )


def _evaluation_summary(
    *,
    tasks: Sequence[Mapping[str, Any]],
    workers: Sequence[Mapping[str, Any]],
    compiler_run: Path,
) -> dict[str, Any]:
    field = lambda name: _distribution(
        [float(row["functional_summary"][name]) for row in tasks]
    )
    fit = field("fit_recovery")
    held = field("held_video_recovery")
    wall_pass = _information_wall_pass(tasks)
    training_timing = _training_metrics_authority(compiler_run)
    evaluation_wall = max(float(row["elapsed_seconds"]) for row in workers)
    training_wall = float(training_timing["elapsed_seconds"])
    return {
        "checkpoint_optimizer_step": int(workers[0]["checkpoint"]["optimizer_step"]),
        "correct_fit": fit,
        "same_task_held": held,
        "held_to_fit": (
            held["median"] / fit["median"]
            if held is not None and fit is not None and fit["median"] > 0
            else None
        ),
        "unseen_wrong_on": field("unseen_wrong_on_recovery"),
        "unseen_wrong_off": field("unseen_wrong_off_recovery"),
        "correct_minus_wrong": field("correct_minus_wrong"),
        "wrong_off_minus_on": field("wrong_off_minus_on"),
        "correct_better_than_wrong_tasks": sum(
            bool(row["functional_summary"]["correct_better_than_wrong"])
            for row in tasks
        ),
        "family": {
            name: _distribution([_family_value(row, name) for row in tasks])
            for name in FAMILY_NAMES
        },
        "information_wall_pass": wall_pass,
        "evaluation_wall_seconds": evaluation_wall,
        "training_wall_seconds": training_wall,
        "training_global_step_seconds_maximum": training_timing[
            "global_step_seconds_maximum"
        ],
        "training_global_step_seconds_median": training_timing[
            "global_step_seconds_median"
        ],
        "evaluation_to_training_wall": evaluation_wall / max(training_wall, 1e-12),
        "tasks": tasks,
    }


def _checks(
    gate: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, bool]:
    thresholds = gate["gate"]
    fit = summary["correct_fit"]
    held = summary["same_task_held"]
    wrong = summary["unseen_wrong_on"]
    margin = summary["correct_minus_wrong"]
    off_on = summary["wrong_off_minus_on"]
    return {
        "correct_fit": fit is not None
        and fit["count"] == 10
        and fit["median"]
        >= float(thresholds["correct_fit_recovery_median_minimum"]),
        "same_task_held": held is not None
        and held["count"] == 10
        and held["median"]
        >= float(thresholds["same_task_held_recovery_median_minimum"]),
        "held_to_fit": summary["held_to_fit"] is not None
        and summary["held_to_fit"] >= float(thresholds["held_to_fit_minimum"]),
        "unseen_wrong": wrong is not None
        and wrong["count"] == 10
        and wrong["median"]
        <= float(thresholds["unseen_wrong_recovery_median_maximum"]),
        "correct_minus_wrong": margin is not None
        and margin["count"] == 10
        and margin["median"]
        >= float(thresholds["correct_minus_wrong_median_minimum"]),
        "correct_better_than_wrong": summary["correct_better_than_wrong_tasks"]
        == int(thresholds["correct_better_than_wrong_required_tasks"]),
        "wrong_off_minus_on": off_on is not None
        and off_on["count"] == 10
        and off_on["median"]
        >= float(thresholds["wrong_off_minus_on_median_minimum"]),
        **{
            f"family_{name}_not_systematically_reversed": value is not None
            and value["count"] == 10
            and value["median"] >= float(thresholds["family_median_minimum"])
            for name, value in summary["family"].items()
        },
        "information_wall": bool(summary["information_wall_pass"]),
    }


def aggregate_program_bank_interaction_evaluation(
    *,
    output_dir: Path,
    gate_config: Path,
    compiler_run: Path,
    worker_count: int,
    previous_report: Path | None = None,
) -> dict[str, Any]:
    state = git_state(Path(__file__).resolve().parents[4])
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal interaction aggregation requires detached authority")
    gate = load_program_bank_interaction_gate(gate_config)
    tasks, workers = _load_workers(output_dir, worker_count, compiler_run)
    worker_authority = workers[0]["authority"]
    expected_gate = {
        "path": str(gate_config),
        "bytes": gate_config.stat().st_size,
    }
    run_contract = read_json(compiler_run / "run_contract.json")
    expected_training = run_contract.get("config")
    if not isinstance(expected_training, Mapping):
        raise ValueError("interaction training config authority changed")
    evaluator_training_path = (
        gate_config.parent / Path(str(gate["training_config"])).name
    ).resolve()
    expected_evaluator_training = {
        "path": str(evaluator_training_path),
        "bytes": evaluator_training_path.stat().st_size,
    }
    if (
        worker_authority.get("gate_config") != expected_gate
        or worker_authority.get("training_config") != expected_training
        or worker_authority.get("evaluator_config")
        != expected_evaluator_training
        or not _tracked_json_config_authority(
            expected_training,
            evaluator_training_path,
            read_json(evaluator_training_path),
            repo_root=Path(__file__).resolve().parents[4],
            training_commit=str(run_contract.get("git", {}).get("commit", "")),
            relative_path=str(gate["training_config"]),
        )
        or Path(str(worker_authority.get("compiler_run", ""))).resolve()
        != compiler_run.resolve()
        or worker_authority.get("evaluator_commit") != state.get("commit")
        or workers[0]["checkpoint"].get("training_commit")
        != run_contract.get("git", {}).get("commit")
        or not _git_commit_is_ancestor(
            Path(__file__).resolve().parents[4],
            str(run_contract.get("git", {}).get("commit", "")),
            str(state.get("commit", "")),
        )
    ):
        raise ValueError("interaction aggregation authority changed")
    summary = _evaluation_summary(
        tasks=tasks, workers=workers, compiler_run=compiler_run
    )
    checkpoint_step = int(summary["checkpoint_optimizer_step"])
    checks = _checks(gate, summary)
    primary_pass = all(checks.values())
    stability: dict[str, Any] = {
        "status": "pending_adjacent_checkpoint",
        "pass": False,
    }
    if previous_report is not None:
        previous = read_json(previous_report)
        drop = float(previous["summary"]["correct_fit"]["median"]) - float(
            summary["correct_fit"]["median"]
        )
        stable = (
            previous.get("schema_version")
            == PROGRAM_BANK_INTERACTION_GATE_REPORT_SCHEMA
            and previous.get("status") == "complete"
            and int(previous["checkpoint"]["optimizer_step"]) == 70
            and checkpoint_step == 110
            and Path(str(previous["checkpoint"]["path"])).resolve()
            == (compiler_run / "checkpoints/macro_00000070").resolve()
            and Path(str(workers[0]["checkpoint"]["path"])).resolve()
            == (compiler_run / "checkpoints/macro_00000110").resolve()
            and previous.get("compiler_run") == str(compiler_run.resolve())
            and previous.get("gate_config")
            == {
                "path": str(gate_config),
                "bytes": gate_config.stat().st_size,
            }
            and previous.get("training_commit")
            == workers[0]["checkpoint"]["training_commit"]
            and previous.get("evaluator_commit")
            == workers[0]["authority"]["evaluator_commit"]
            and previous.get("aggregation_commit") == state["commit"]
            and int(previous.get("worker_count", -1)) == worker_count
            and bool(previous.get("primary_pass"))
            and all(bool(value) for value in previous.get("checks", {}).values())
            and drop
            <= float(
                gate["gate"]["maximum_adjacent_correct_fit_median_drop"]
            )
        )
        stability = {
            "status": "evaluated",
            "previous_optimizer_step": 70,
            "current_optimizer_step": checkpoint_step,
            "correct_fit_median_drop": drop,
            "previous_primary_pass": bool(previous.get("primary_pass")),
            "pass": stable,
        }
    report = {
        "schema_version": PROGRAM_BANK_INTERACTION_GATE_REPORT_SCHEMA,
        "status": "complete",
        "checkpoint": dict(workers[0]["checkpoint"]),
        "summary": summary,
        "checks": checks,
        "primary_pass": primary_pass,
        "adjacent_checkpoint": stability,
        "gate_pass": primary_pass and bool(stability["pass"]),
        "scientific_scope": {
            "deployment_candidate": False,
            "can_pass_g3": False,
            "question": (
                "can one continuous candidate interaction preserve R5 capacity "
                "and neutralize unseen wrong banks"
            ),
        },
        "worker_count": worker_count,
        "compiler_run": str(compiler_run.resolve()),
        "training_commit": workers[0]["checkpoint"]["training_commit"],
        "evaluator_commit": workers[0]["authority"]["evaluator_commit"],
        "aggregation_commit": state["commit"],
        "worker_commits": sorted({row["git"]["commit"] for row in workers}),
        "gate_config": expected_gate,
    }
    write_json_atomic(output_dir / "aggregate.json", report)
    write_json_atomic(
        output_dir / "completion.json",
        {
            "schema_version": PROGRAM_BANK_INTERACTION_GATE_REPORT_SCHEMA,
            "checkpoint_optimizer_step": checkpoint_step,
            "primary_pass": primary_pass,
            "gate_pass": report["gate_pass"],
        },
    )
    return report
