"""Registered fresh full/public learning studies on the canonical operator trainer."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json

TASK = "operator_joint_public_fresh_20260930"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
MODE = "joint"
LOSS = "full_plus_public_beta"
SPEC_NAME = "joint_public_fresh_spec.json"
CHECKPOINTS = (90, 180, 270, 360, 450)
JOINT = {"loss_variant": LOSS, "full_loss_coefficient": 1.0,
         "public_loss_coefficient": 1.0, "public_credit": "direct_A_B0_only",
         "query_reuse": "same_112_query_action_tau_noise", "internal_mode": "T",
         "initialization": "fresh_original_identity_and_module_seeds"}

CONTEXT_TASK = "operator_context_value_fresh_20261001"
CONTEXT_ROOT = Path("/data1/user/ymdai/ember_runs") / CONTEXT_TASK
CONTEXT_MODE = "context"
CONTEXT_SPEC_NAME = "context_value_fresh_spec.json"
MODES = (MODE, CONTEXT_MODE)
CONTEXT = {
    "normalization": "parameterless_RMS1024_eps1e-6",
    "shared_projection": [1024, 256], "qkv_width": 256, "heads": 4, "head_width": 64,
    "rope": {"base": 10000, "pairing": "adjacent_even_odd", "position": "real_frame_index/5",
             "consumers": ["Q", "K"]},
    "attention": "full_real_frames_bidirectional_per_horizon_softmax",
    "output_projection": [256, 256], "target_U": [256, 256],
    "gate": "GELU(PK+C_hbar+U_context)*D_delta_hbar",
    "last_frame_context": True, "write_count": "N-1", "bias": False,
    "dropout": 0.0, "ffn": False, "extra_block": False, "cross_horizon_average": False,
    "initialization": "old_modules_first_then_independent_CPU_seed7_Linear_defaults_U_zero",
    "added_trainable_parameters": 3014656, "native_credit_detached": False,
}


def registered(spec: dict) -> bool:
    return spec.get("task") in (TASK, CONTEXT_TASK)


def settings(spec: dict) -> tuple[Path, str, dict]:
    if spec.get("task") == TASK:
        return ROOT, MODE, JOINT
    if spec.get("task") == CONTEXT_TASK:
        return CONTEXT_ROOT, CONTEXT_MODE, {**JOINT, "internal_mode": CONTEXT_MODE}
    raise ValueError("unregistered fresh full/public study")


def expected_context_spec(base: dict, events: dict) -> dict:
    spec = expected_spec(base, events)
    return {**spec, "task": CONTEXT_TASK,
            "design": "docs/designs/operator_read_write_learning_design.md#36",
            "run_root": str(CONTEXT_ROOT),
            "execution": {**spec["execution"], "modes": [CONTEXT_MODE]},
            "joint": {**JOINT, "internal_mode": CONTEXT_MODE},
            "operator": {**spec["operator"], "context_value": CONTEXT}}


def expected_spec(base: dict, events: dict) -> dict:
    execution = {**base["execution"], "modes": [MODE], "world_sizes": [1, 2, 3, 4],
                 "updates_per_mode": 450, "queries_per_mode": 50400,
                 "checkpoints": list(CHECKPOINTS), "only_selected_checkpoint": 450}
    execution.pop("world_size")
    return {**base, "task": TASK,
            "design": "docs/designs/operator_read_write_learning_design.md#35",
            "run_root": str(ROOT), "events": events, "execution": execution,
            "joint": JOINT, "evaluation": {**base["evaluation"], "bank_macro": 450},
            "budget": {"new_gpu_hours_hard": 16, "peak_new_gib": 32,
                       "expected_wall_hours": [3, 5], "report_gpu_hours": 12}}


def validate_request(spec: dict, args) -> None:
    _, mode, joint = settings(spec)
    if (args.mode != mode or spec.get("joint") != joint
            or not args.attempt or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.attempt) is None
            or getattr(args, "pilot_arm", None) is not None
            or args.microbatch not in (28, 14, 7) or args.frame_chunk not in (8, 4)
            or args.stop_after_macro not in (None, *CHECKPOINTS[:-1])
            or (args.resume is None) != (args.attempt == "fresh")):
        raise ValueError("joint450 requires fresh identity or its own complete ECP resume")


def validate_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    from .run import complete_checkpoint, resume_contract_compatible

    root, mode, joint = settings(spec)
    attempts = root / mode / "train/attempts"
    if (output.parent.resolve() != attempts.resolve()
            or contract.get("loss_variant") != LOSS or contract.get("joint") != joint
            or (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists()):
        raise ValueError("joint450 attempt output or loss identity changed")
    if args.resume is None:
        if any(attempts.glob("*/run_contract.json")):
            raise ValueError("joint450 fresh initialization already has an owned attempt")
        return
    checkpoint = args.resume.resolve()
    parent = checkpoint.parent.parent
    complete = [path for path in attempts.glob("*/checkpoints/macro_*")
                if complete_checkpoint(path)]
    if (parent.parent.resolve() != attempts.resolve() or parent == output.resolve()
            or checkpoint.name not in {f"macro_{step:08d}" for step in CHECKPOINTS[:-1]}
            or checkpoint not in complete
            or checkpoint.name != max(path.name for path in complete)
            or not resume_contract_compatible(read_json(parent / "run_contract.json"), contract,
                                               allow_topology_change=True)):
        raise ValueError("joint450 resume requires latest complete owned same-loss ECP")


def inspect_source(spec: dict, checkpoint: Path) -> dict:
    """Keep actual training Git/spec identity separate from the current reader."""
    from .run import SCHEMA, STAGE, complete_checkpoint, frozen_git

    frozen_git()
    root, mode, joint = settings(spec)
    checkpoint = checkpoint.resolve()
    output = checkpoint.parent.parent
    if (output.parent.resolve() != (root / mode / "train/attempts").resolve()
            or checkpoint.name != "macro_00000450" or not complete_checkpoint(checkpoint)):
        raise ValueError("joint readout requires its complete owned 450 ECP")
    run = read_json(output / "run_contract.json")
    training_spec = Path(run["spec"])
    training_repo = training_spec.parents[2]
    code = git_state(training_repo)
    if (read_json(training_spec) != spec or code["branch"] or code["dirty_paths"]
            or code["commit"] != run["git"]["commit"]):
        raise ValueError("joint training frozen code/spec identity changed")
    import subprocess

    refs = subprocess.run(["git", "branch", "-r", "--contains", code["commit"]],
                          cwd=training_repo, check=True, capture_output=True, text=True).stdout
    if run["git"].get("pushed_ref") not in {row.strip() for row in refs.splitlines()}:
        raise ValueError("joint training source was not pushed")
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True,
                         weights_only=True)
    expected_sampler = {"schema_version": spec["events"]["schema_version"],
                        "seed": 20260928, "tasks": spec["events"]["task_ids"],
                        "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28,
                        "teacher_rounds": [[20260928, 1, "task"], [20260928, 1, "task", 1]],
                        "teacher_visits_per_round": 50, "teacher_demo_pool": list(range(50))}
    facts = ((run.get("schema_version"), SCHEMA), (run.get("stage"), STAGE),
             (run.get("mode"), mode), (run.get("joint"), joint),
             (run.get("loss_variant"), LOSS), (run.get("operator"), spec["operator"]),
             (run.get("optimizer"), spec["optimization"]), (run.get("events"), spec["events"]),
             (run.get("source_trainable"), 0), (run.get("sampler"), expected_sampler),
             (run.get("continuation"), None), (run.get("pilot_arm"), None),
             (trainer.get("stage"), STAGE), (trainer.get("next_macro"), 450),
             (trainer.get("metrics_rows"), 450), (trainer.get("scheduler", {}).get("last_epoch"), 450),
             (bool(trainer.get("optimizer", {}).get("param_groups")), True),
             (trainer.get("scaler"), None),
             (trainer.get("training_state"), {"updates": 450, "mode": mode, "loss_variant": LOSS}),
             (trainer.get("sampler_state"), {**expected_sampler, "next_step": 450}))
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    completion = read_json(output / "completion.json")
    if (any(actual != wanted for actual, wanted in facts) or not valid_metrics(metrics, mode=mode)
            or completion.get("updates") != 450 or completion.get("checkpoint") != str(checkpoint)):
        raise ValueError("joint source loss/ECP/optimizer/sampler/completion changed")
    return run


def valid_metrics(metrics: list, *, mode: str = MODE) -> bool:
    if len(metrics) != 450 or [row["update"] for row in metrics] != list(range(1, 451)):
        return False
    for row in metrics:
        if (row.get("mode") != mode or row.get("loss_variant") != LOSS
                or row.get("queries") != 112 or len(row.get("jobs", ())) != 4):
            return False
        if any(job.get("loss_variant") != LOSS or not job.get("public_query_reuse")
               for job in row["jobs"]):
            return False
    return True
