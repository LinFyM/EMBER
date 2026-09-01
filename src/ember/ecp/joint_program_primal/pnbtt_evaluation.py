"""Zero-gradient Panel-B evaluation of one PNBTT E1 checkpoint."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.joint_program_primal.pnbtt_runtime import (
    PNBTT_E1_STAGE,
    PNBTT_TASKLOCAL_RUN_SCHEMA,
    PNBTTTaskLocalRuntime,
    prepare_pnbtt_tasklocal_runtime,
)
from ember.ecp.joint_program_primal.pnbtt_tasklocal import (
    generated_rank16,
    local_tasks,
    prepare_e1_arms,
)
from ember.ecp.joint_program_primal.train_step import functional_panel_batch
from ember.ecp.stage0_train_step import _gather_records
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_value,
)


PNBTT_E1_EVALUATION_SCHEMA = "ember_ecp_pnbtt_e1_evaluation_v1"
PNBTT_E1_QUALIFICATION_SCHEMA = "ember_ecp_pnbtt_e1_qualification_v1"
_TRAINING_IDENTITY_KEYS = (
    "schema_version",
    "stage",
    "mode",
    "git",
    "config",
    "base_config",
    "source_checkpoint",
    "tokenizer",
    "data_root",
    "condition_cache",
    "program_bank_condition_cache_root",
    "task_local",
    "model",
    "optimization",
    "information_wall",
    "inventory",
)


def _training_authority(
    runtime: PNBTTTaskLocalRuntime, checkpoint: Path
) -> tuple[Path, dict[str, Any]]:
    training_root = checkpoint.parent.parent.resolve()
    if checkpoint.parent != training_root / "checkpoints":
        raise ValueError("PNBTT E1 checkpoint escaped its training root")
    contract = read_json(training_root / "run_contract.json")
    if any(
        contract.get(key) != runtime.run_contract.get(key)
        for key in _TRAINING_IDENTITY_KEYS
    ):
        raise ValueError("PNBTT E1 evaluation/training authority changed")
    return training_root, contract


def _load_writer_checkpoint(
    runtime: PNBTTTaskLocalRuntime, checkpoint: Path
) -> dict[str, Any]:
    macro = checkpoint_macro(checkpoint)
    if macro not in runtime.checkpoint_steps:
        raise ValueError("PNBTT E1 evaluation macro was not preregistered")
    training_root, training_contract = _training_authority(runtime, checkpoint)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    tensor_path = checkpoint / "ecp.safetensors"
    expected_files = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(
            f"rank_{rank:02d}_state.pt"
            for rank in range(runtime.context.world_size)
        ),
    }
    if (
        manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != PNBTT_E1_STAGE
        or manifest.get("run_contract_schema") != PNBTT_TASKLOCAL_RUN_SCHEMA
        or int(manifest.get("next_macro", -1)) != macro
        or int(manifest.get("world_size", -1)) != runtime.context.world_size
        or set(manifest.get("files", {})) != expected_files
    ):
        raise ValueError("PNBTT E1 evaluation checkpoint authority changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"PNBTT E1 checkpoint file changed: {name}")
    runtime.writer_state.load_state_dict(
        load_file(str(tensor_path), device=str(runtime.context.device)), strict=True
    )
    runtime.writer_state.requires_grad_(False).eval()
    return {
        "path": str(checkpoint),
        "macro": macro,
        "tensor_bytes": tensor_path.stat().st_size,
        "training_world_size": int(manifest["world_size"]),
        "training_root": str(training_root),
        "training_git_commit": training_contract["git"]["commit"],
        "training_config_bytes": int(training_contract["config"]["bytes"]),
    }


def _write_adjacent_qualification(
    runtime: PNBTTTaskLocalRuntime, *, evaluation_root: Path
) -> None:
    expected_macros = tuple(runtime.checkpoint_steps)
    if len(expected_macros) != 2:
        raise RuntimeError("PNBTT E1 adjacent checkpoint contract changed")
    evaluations = []
    for macro in expected_macros:
        path = evaluation_root / f"macro_{macro:08d}" / "evaluation.json"
        if not path.is_file():
            return
        row = read_json(path)
        if (
            row.get("schema_version") != PNBTT_E1_EVALUATION_SCHEMA
            or row.get("status") != "complete"
            or int(row.get("checkpoint", {}).get("macro", -1)) != macro
        ):
            raise ValueError("PNBTT E1 adjacent evaluation authority changed")
        evaluations.append((path, row))

    training_roots = {
        row["checkpoint"]["training_root"] for _, row in evaluations
    }
    if len(training_roots) != 1:
        raise ValueError("PNBTT E1 adjacent evaluations mixed training runs")
    training_root = Path(training_roots.pop())
    if evaluation_root != training_root / "evaluations":
        raise ValueError("PNBTT E1 evaluation root changed")
    completion = read_json(training_root / "completion.json")
    training_complete = (
        completion.get("stage") == PNBTT_E1_STAGE
        and int(completion.get("completed_optimizer_steps", -1))
        == max(expected_macros)
    )
    overall_gates = tuple(row["gate"] for _, row in evaluations)
    task_gates = {
        str(task): tuple(
            next(
                task_row["gate"]
                for task_row in row["tasks"]
                if int(task_row["authority_id"]) == task
            )
            for _, row in evaluations
        )
        for task in map(int, runtime.config["task_local"]["task_ids"])
    }
    checks = {
        "both_checkpoint_gates_pass": all(gate == "pass" for gate in overall_gates),
        "overall_conclusion_consistent": len(set(overall_gates)) == 1,
        "per_task_conclusion_consistent": all(
            len(set(gates)) == 1 for gates in task_gates.values()
        ),
        "training_complete": training_complete,
    }
    qualification = {
        "schema_version": PNBTT_E1_QUALIFICATION_SCHEMA,
        "status": "complete",
        "stage": PNBTT_E1_STAGE,
        "training_root": str(training_root),
        "checkpoint_macros": list(expected_macros),
        "evaluations": [
            {"path": str(path), "gate": row["gate"]}
            for path, row in evaluations
        ],
        "task_gates": task_gates,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "non_pass",
    }
    write_json_atomic(evaluation_root / "qualification.json", qualification)


def _functional_value(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    seed: int,
) -> float:
    by_task = runtime.config["optimization"].get(
        "functional_policy_microbatch_size_by_task", {}
    )
    microbatch = int(
        by_task.get(
            str(task),
            runtime.config["optimization"]["functional_policy_microbatch_size"],
        )
    )
    value, details = functional_lora_loss_value(
        runtime.policy,
        state,
        runtime.ranks.contract,
        batch=batch,
        policy_rng_seed=seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=microbatch,
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(value)):
        raise RuntimeError("PNBTT E1 Panel-B functional value changed")
    return float(value)


def _free_losses(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arms: Mapping[str, Any],
) -> dict[str, float]:
    root = (
        runtime.args.asset_root
        / runtime.config["authorities"]["positive_control_root"]
    ).resolve()
    source = read_json(root / f"task_{task:03d}" / "result.json")
    rows = (*source["evaluation"]["fit_videos"], source["evaluation"]["held_video"])
    by_video = {
        int(row["video_demo"]): float(row["panel_b"]["free_primal_loss"])
        for row in rows
    }
    fit_mean = statistics.fmean(
        by_video[arms[name].video_demo]
        for name in ("correct_fit0", "correct_fit1")
    )
    return {
        "correct_fit0": by_video[arms["correct_fit0"].video_demo],
        "correct_fit1": by_video[arms["correct_fit1"].video_demo],
        "correct_held": by_video[arms["correct_held"].video_demo],
        "wrong_fit0": fit_mean,
        "wrong_fit1": fit_mean,
    }


def _evaluate_arm(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arm: Any,
    free_loss: float,
    visits: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        state, output = generated_rank16(
            runtime, task=task, arm=arm, canonicalize=True
        )
        solve = output.solve_metrics.detach().float()
        concentration = torch.maximum(solve[:, 4], solve[:, 5])
        near_bound = float(
            (
                concentration
                >= float(runtime.config["gate"]["near_bound_weight_threshold"])
            )
            .float()
            .mean()
        )
        conditioning = output.conditioning_metrics.detach().float().cpu().tolist()
    rows = []
    for visit in range(visits):
        batch, panel = functional_panel_batch(
            runtime,
            task_id=task,
            panel_name="b",
            visit_index=visit,
        )
        loss = _functional_value(
            runtime,
            task=task,
            state=state,
            batch=batch,
            seed=panel.policy_rng_seed,
        )
        rows.append(
            {
                "visit": visit,
                "carrier_loss": float(panel.flow_loss),
                "generated_loss": loss,
                "benefit_over_carrier": float(panel.flow_loss) - loss,
            }
        )
    carrier = statistics.fmean(row["carrier_loss"] for row in rows)
    generated = statistics.fmean(row["generated_loss"] for row in rows)
    denominator = carrier - float(free_loss)
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("PNBTT E1 positive-control denominator changed")
    return {
        "video_demo": arm.video_demo,
        "bank_task": arm.bank_task,
        "visits": rows,
        "carrier_loss": carrier,
        "generated_loss": generated,
        "free_primal_loss": float(free_loss),
        "free_primal_benefit": denominator,
        "functional_recovery": (carrier - generated) / denominator,
        "near_bound_fraction": near_bound,
        "conditioning": conditioning,
        "condition_metrics": arm.condition_metrics,
    }


def _evaluate_task(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arms: Mapping[str, Any],
) -> dict[str, Any]:
    visits = 16 if runtime.args.mode == "formal" else 1
    free = _free_losses(runtime, task=task, arms=arms)
    evaluated = {
        name: _evaluate_arm(
            runtime,
            task=task,
            arm=arm,
            free_loss=free[name],
            visits=visits,
        )
        for name, arm in arms.items()
    }
    correct = [
        evaluated[name]["functional_recovery"]
        for name in ("correct_fit0", "correct_fit1")
    ]
    held = float(evaluated["correct_held"]["functional_recovery"])
    wrong = [
        evaluated[name]["functional_recovery"]
        for name in ("wrong_fit0", "wrong_fit1")
    ]
    gate = runtime.config["gate"]
    checks = {
        "correct_fit_each": min(correct) >= float(gate["correct_fit_each_minimum"]),
        "correct_held": held >= float(gate["correct_held_minimum"]),
        "wrong_each": max(wrong) <= float(gate["wrong_each_maximum"]),
        "margin": min(correct) - max(wrong)
        >= float(gate["minimum_correct_minus_maximum_wrong"]),
        "all_pairs": min((*correct, held)) > max(wrong),
        "near_bound": max(
            float(row["near_bound_fraction"]) for row in evaluated.values()
        )
        <= float(gate["maximum_near_bound_fraction"]),
    }
    return {
        "authority_id": task,
        "role": runtime.panels[task].role,
        "arms": evaluated,
        "checks": checks,
        "gate": "pass" if all(checks.values()) else "non_pass",
    }


def evaluate_pnbtt_tasklocal(args: Any) -> None:
    checkpoint = args.evaluate_checkpoint
    if checkpoint is None or args.resume is not None or args.phase != "joint":
        raise ValueError("PNBTT E1 evaluation arguments changed")
    expected_output = checkpoint.parent.parent / "evaluations" / checkpoint.name
    if args.output_dir != expected_output:
        raise ValueError("PNBTT E1 formal evaluation output layout changed")
    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime: PNBTTTaskLocalRuntime | None = None
    try:
        runtime = prepare_pnbtt_tasklocal_runtime(args, context)
        checkpoint_record = _load_writer_checkpoint(runtime, checkpoint)
        arms = prepare_e1_arms(runtime)
        started = time.monotonic()
        local = [
            _evaluate_task(runtime, task=task, arms=arms[task])
            for task in local_tasks(runtime)
        ]
        tasks = sorted(
            _gather_records(local, runtime.context.world_size),
            key=lambda row: int(row["authority_id"]),
        )
        if tuple(int(row["authority_id"]) for row in tasks) != tuple(
            map(int, runtime.config["task_local"]["task_ids"])
        ):
            raise RuntimeError("PNBTT E1 evaluation lost a task")
        result = {
            "schema_version": PNBTT_E1_EVALUATION_SCHEMA,
            "status": "complete",
            "stage": PNBTT_E1_STAGE,
            "checkpoint": checkpoint_record,
            "tasks": tasks,
            "gate": "pass" if all(row["gate"] == "pass" for row in tasks) else "non_pass",
            "elapsed_seconds": time.monotonic() - started,
            "information_wall": {
                "correct_held_backward_calls": 0,
                "wrong_fit1_backward_calls": 0,
                "panel_b_backward_calls": 0,
                "validation_or_test_reads": 0,
                "action_meta_installed": False,
                "single_complete_rank16": True,
                "shuffled_or_reversed_use": False,
            },
        }
        if context.is_main:
            write_json_atomic(args.output_dir / "evaluation.json", result)
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "stage": PNBTT_E1_STAGE,
                    "checkpoint_macro": checkpoint_record["macro"],
                    "gate": result["gate"],
                },
            )
            _write_adjacent_qualification(
                runtime, evaluation_root=args.output_dir.parent
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
