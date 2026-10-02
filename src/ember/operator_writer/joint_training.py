"""Registered full/public studies and their bounded continuation contracts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import Pi05SourceTrainingError, read_json
from . import support_diversity, prefix_change

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
CONTEXT_CONTINUATION_TASK = "operator_context_value_continuation900_20261001"
CONTEXT_CONTINUATION_ROOT = ROOT.parent / CONTEXT_CONTINUATION_TASK
CONTEXT_CONTINUATION_SPEC_NAME = "context_value_continuation900_spec.json"
CONTEXT_CONTINUATION_CHECKPOINTS = (540, 630, 720, 810, 900)
CONTEXT_PARENT_GIT = "5f4f76e7179948f7945a7a98814f55c8a7f78610"
CONTEXT_PARENT_SPEC_PATH = Path("/data1/user/ymdai/projects/EMBER-context-value-formal"
                                "/configs/operator_read_write_v1/context_value_fresh_spec.json")
CONTEXT_PARENT_CHECKPOINT = (CONTEXT_ROOT / "context/train/attempts/fresh/checkpoints"
                             / "macro_00000450")
CONTEXT_SPEC_NAME = "context_value_fresh_spec.json"
SELF_READ_TASK = "operator_self_conditioned_native_fresh_20261001"
SELF_READ_ROOT = ROOT.parent / SELF_READ_TASK
SELF_READ_MODE = "self_read"
SELF_READ_SPEC_NAME = "self_conditioned_native_fresh_spec.json"
CONDITIONAL_TASK = "conditional_read_write_fresh_20261001"
CONDITIONAL_ROOT = ROOT.parent / CONDITIONAL_TASK
CONDITIONAL_MODE = "conditional_read_write"
CONDITIONAL_SPEC_NAME = "conditional_read_write_fresh_spec.json"
CONDITIONAL_CONTINUATION_TASK = "conditional_read_write_continuation900_20261002"
CONDITIONAL_CONTINUATION_ROOT = ROOT.parent / CONDITIONAL_CONTINUATION_TASK
CONDITIONAL_CONTINUATION_SPEC_NAME = "conditional_read_write_continuation900_spec.json"
CONDITIONAL_PARENT_GIT = "a0e0248d96568e42b24a3d1c4102e2ca6e35a40c"
CONDITIONAL_PARENT_SPEC_PATH = Path("/data1/user/ymdai/projects/EMBER-conditional-read-write-mlp-formal"
                                   "/configs/operator_read_write_v1/conditional_read_write_fresh_spec.json")
CONDITIONAL_PARENT_CHECKPOINT = (CONDITIONAL_ROOT / CONDITIONAL_MODE
    / "train/attempts/resume360_native_packing/checkpoints/macro_00000450")
CONDITIONAL_ORIGIN_GIT = "797ae01f3d35d4740a636b15f020c4ef55477845"
CONDITIONAL_ORIGIN_SPEC = Path("/data1/user/ymdai/projects/EMBER-conditional-read-write-r2-formal"
                               "/configs/operator_read_write_v1/conditional_read_write_fresh_spec.json")
CONDITIONAL = {"loss_variant": "full", "full_loss_coefficient": 1.0,
               "public_loss_coefficient": 0.0, "public_credit": "none_no_public_objective",
               "query_reuse": "112_cross_episode_query_action_tau_noise",
               "internal_mode": CONDITIONAL_MODE,
               "initialization": "fresh_legal_identity_and_seed7_modules"}
CONDITIONAL_OPERATOR = {
    "rank": 128, "alpha": 128, "targets": 38, "identity_seed": 20260721,
    "module_seed": 7, "probe_seed": 1729, "probe_shape": [50, 32],
    "teaching_camera": "dual", "frame_stride": 5, "frame_chunk": 8,
    "source_frozen": True, "value_width": 256, "memory_dtype": "float32",
    "memory_step": 1.0, "additional_loss": False,
    "conditional_read_write": {
        "native_reads": 1, "teacher_state": "State_prompt_segment_omitted",
        "native_base": "public_A0_B0", "hidden_shape": ["N", 50, 1024],
        "hidden_normalization": "parameterless_token_RMS_eps1e-6",
        "layers": 4, "layer_parameter_sharing": False, "width": 1024,
        "heads": 16, "head_width": 64, "ffn_width": 4096, "dropout": 0.0,
        "position": "fixed_sinusoidal_real_frame_index_and_horizon_0_49_disjoint512_halves",
        "context_normalization": "pre_LayerNorm",
        "attention": "same_frame_all_visible_prior_frames_visible_future_frames_masked",
        "dynamic_initialization": "zero_first_frame_then_backward_hidden_difference",
        "dynamic_stream": "zero_preserving_biasfree_attention_and_gated_FFN",
        "head_context": "concat_final_c_and_real_H", "head_dynamic": "final_d",
        "transition": "X_t_minus_1_with_c_t_d_t",
        "S": "zero_initial_true_X_association_with_A0_X",
        "M": "zero_initial_compiled_after_final_A0_plus_S_using_K_and_S_X",
        "output": "unique_A0_plus_S_B0_plus_M", "rank_scale": 1.0,
        "credit": "complete_full_FM_same_version_cotangent_replay_no_detach",
    },
}
MODES = (MODE, CONTEXT_MODE, SELF_READ_MODE, CONDITIONAL_MODE, prefix_change.MODE)
SELF_READ = {"native_reads": 2, "writer_parameter_sharing": "same_Context_module",
             "memory_initialization": "zero_each_read", "native_installation": "beta_then_beta_plus_M0",
             "writer_public_base": "original_beta_both_reads", "output": "beta_plus_M1_only",
             "credit": "complete_composite_no_detach", "intermediate_loss": False,
             "added_trainable_parameters": 0, "action_out_feedback": False}
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
    return spec.get("task") in (TASK, CONTEXT_TASK, CONTEXT_CONTINUATION_TASK, SELF_READ_TASK, CONDITIONAL_TASK, CONDITIONAL_CONTINUATION_TASK, support_diversity.TASK, prefix_change.TASK)


def settings(spec: dict) -> tuple[Path, str, dict]:
    if spec.get("task") == prefix_change.TASK:
        return prefix_change.ROOT, prefix_change.MODE, prefix_change.JOINT
    if spec.get("task") == support_diversity.TASK:
        return support_diversity.ROOT, CONDITIONAL_MODE, CONDITIONAL
    if spec.get("task") == TASK:
        return ROOT, MODE, JOINT
    if spec.get("task") in (CONTEXT_TASK, CONTEXT_CONTINUATION_TASK):
        root = CONTEXT_ROOT if spec["task"] == CONTEXT_TASK else CONTEXT_CONTINUATION_ROOT
        return root, CONTEXT_MODE, {**JOINT, "internal_mode": CONTEXT_MODE}
    if spec.get("task") == SELF_READ_TASK:
        return SELF_READ_ROOT, SELF_READ_MODE, {**JOINT, "internal_mode": SELF_READ_MODE}
    if spec.get("task") in (CONDITIONAL_TASK, CONDITIONAL_CONTINUATION_TASK):
        root = CONDITIONAL_ROOT if spec["task"] == CONDITIONAL_TASK else CONDITIONAL_CONTINUATION_ROOT
        return root, CONDITIONAL_MODE, CONDITIONAL
    raise ValueError("unregistered fresh full/public study")


def expected_context_spec(base: dict, events: dict) -> dict:
    spec = expected_spec(base, events)
    return {**spec, "task": CONTEXT_TASK,
            "design": "docs/designs/operator_read_write_learning_design.md#36",
            "run_root": str(CONTEXT_ROOT),
            "execution": {**spec["execution"], "modes": [CONTEXT_MODE]},
            "joint": {**JOINT, "internal_mode": CONTEXT_MODE},
            "operator": {**spec["operator"], "context_value": CONTEXT}}


def expected_self_read_spec(context: dict) -> dict:
    return {**context, "task": SELF_READ_TASK,
            "design": "docs/designs/operator_read_write_learning_design.md#40",
            "run_root": str(SELF_READ_ROOT),
            "execution": {**context["execution"], "modes": [SELF_READ_MODE],
                          "world_sizes": list(range(1, 7))},
            "joint": {**JOINT, "internal_mode": SELF_READ_MODE},
            "operator": {**context["operator"], "self_conditioned_native": SELF_READ},
            "budget": {"new_gpu_hours_hard": 24, "peak_new_gib": 36,
                       "expected_wall_hours": [4, 6], "report_gpu_hours": 18}}


def expected_conditional_spec(events_source: dict) -> dict:
    """Reuse only sealed data/events and optimization; replace the old model/objective."""
    return {**events_source, "task": CONDITIONAL_TASK,
            "design": "docs/designs/conditional_read_write_architecture.md#13",
            "run_root": str(CONDITIONAL_ROOT), "operator": CONDITIONAL_OPERATOR,
            "execution": {**events_source["execution"], "modes": [CONDITIONAL_MODE],
                          "world_sizes": list(range(1, 7))}, "joint": CONDITIONAL,
            "budget": {"new_gpu_hours_hard": 40, "peak_new_gib": 80,
                       "expected_wall_hours": [6, 12], "report_gpu_hours": 30}}


def expected_conditional_continuation_spec(parent: dict) -> dict:
    """Extend the fixed graph and absolute v3 stream to its second teacher round."""
    return {**parent, "task": CONDITIONAL_CONTINUATION_TASK,
            "design": "docs/designs/conditional_read_write_architecture.md#14",
            "run_root": str(CONDITIONAL_CONTINUATION_ROOT),
            "execution": {**parent["execution"], "updates_per_mode": 900,
                          "queries_per_mode": 100800,
                          "checkpoints": list(CONTEXT_CONTINUATION_CHECKPOINTS),
                          "only_selected_checkpoint": 900},
            "evaluation": {**parent["evaluation"], "bank_macro": 900,
                           "conditional_adjacent_macro": 810,
                           "adjacent_trigger": "complete_valid_900_correct400_successes_strictly_gt153"},
            "continuation": {"parent_run_root": str(CONDITIONAL_ROOT), "parent_macro": 450,
                             "parent_training_git": CONDITIONAL_PARENT_GIT,
                             "parent_event_schema": parent["events"]["schema_version"],
                             "sampler_migration": "none_keep_v3_cursor_and_second_teacher_round"},
            "budget": {"new_gpu_hours_hard": 20, "peak_new_gib": 64,
                       "expected_wall_hours": [3, 5], "report_gpu_hours": 16}}

def expected_context_continuation_spec(parent: dict) -> dict:
    return {**parent, "task": CONTEXT_CONTINUATION_TASK,
            "design": "docs/designs/operator_read_write_learning_design.md#37",
            "run_root": str(CONTEXT_CONTINUATION_ROOT),
            "execution": {**parent["execution"], "updates_per_mode": 900,
                          "queries_per_mode": 100800,
                          "checkpoints": list(CONTEXT_CONTINUATION_CHECKPOINTS),
                          "only_selected_checkpoint": 900},
            "evaluation": {**parent["evaluation"], "bank_macro": 900,
                           "conditional_adjacent_macro": 810,
                           "adjacent_trigger": "complete_valid_900_correct400_successes_strictly_gt153"},
            "continuation": {"parent_run_root": str(CONTEXT_ROOT), "parent_macro": 450,
                             "parent_training_git": CONTEXT_PARENT_GIT,
                             "parent_event_schema": parent["events"]["schema_version"],
                             "sampler_migration": "none_keep_v3_cursor_and_second_teacher_round"},
            "budget": {"new_gpu_hours_hard": 16, "peak_new_gib": 32,
                       "expected_wall_hours": [3, 4], "report_gpu_hours": 12}}


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
    continuation = spec["task"] in (CONTEXT_CONTINUATION_TASK, CONDITIONAL_CONTINUATION_TASK, support_diversity.TASK)
    frames = (4, 8, 16, 32) if mode in (CONDITIONAL_MODE, prefix_change.MODE) else (8, 4)
    checkpoints = (support_diversity.CHECKPOINTS if spec["task"] == support_diversity.TASK else
                   CONTEXT_CONTINUATION_CHECKPOINTS if continuation else CHECKPOINTS)
    bad_resume = (args.resume is None or args.attempt == "fresh") if continuation else (
        (args.resume is None) != (args.attempt == "fresh"))
    if (args.mode != mode or spec.get("joint") != joint
            or not args.attempt or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.attempt) is None
            or getattr(args, "pilot_arm", None) is not None
            or args.microbatch not in (28, 14, 7) or args.frame_chunk not in frames
            or args.stop_after_macro not in (None, *checkpoints[:-1]) or bad_resume):
        raise ValueError("joint/context requires registered fresh identity or complete same-loss ECP continuation")


def conditional_contract(contract: dict) -> bool:
    return (contract.get("mode") == CONDITIONAL_MODE
            and contract.get("operator") == CONDITIONAL_OPERATOR
            and contract.get("joint") == CONDITIONAL
            and contract.get("loss_variant") == "full")


def _inspect_frozen_source(run: dict, spec: dict) -> None:
    import subprocess

    training_spec = Path(run["spec"])
    training_repo = training_spec.parents[2]
    code = git_state(training_repo)
    if (read_json(training_spec) != spec or code["branch"] or code["dirty_paths"]
            or code["commit"] != run["git"]["commit"]):
        raise ValueError("joint training frozen code/spec identity changed")
    refs = subprocess.run(["git", "branch", "-r", "--contains", code["commit"]],
                          cwd=training_repo, check=True, capture_output=True, text=True).stdout
    if run["git"].get("pushed_ref") not in {row.strip() for row in refs.splitlines()}:
        raise ValueError("joint training source was not pushed")


def _conditional_resume_record(parent: dict, current: dict) -> dict:
    if not conditional_contract(parent) or not conditional_contract(current):
        raise ValueError("conditional source migration cannot change the scientific contract")
    old = parent.get("source_resume")
    if old is None:
        if (parent["git"]["commit"] != CONDITIONAL_ORIGIN_GIT
                or parent["spec"] != str(CONDITIONAL_ORIGIN_SPEC)):
            raise ValueError("conditional source migration requires its actual797 origin")
    elif (old.get("origin_training_git") != CONDITIONAL_ORIGIN_GIT
          or old.get("origin_training_spec") != str(CONDITIONAL_ORIGIN_SPEC)
          or old.get("current_training_git") != parent["git"]
          or old.get("current_training_spec") != parent["spec"]):
        raise ValueError("conditional source migration lost its actual origin lineage")
    origin = read_json(CONDITIONAL_ORIGIN_SPEC)
    registered_specs = {CONDITIONAL_TASK: origin,
                        CONDITIONAL_CONTINUATION_TASK: expected_conditional_continuation_spec(origin),
                        support_diversity.TASK: support_diversity.expected_spec(origin)}
    parent_spec, current_spec = (read_json(Path(run["spec"])) for run in (parent, current))
    for run, spec in ((parent, parent_spec), (current, current_spec)):
        if (spec != registered_specs.get(spec.get("task"))
                or run.get("continuation") != spec.get("continuation")
                or run.get("events") != spec["events"]):
            raise ValueError("conditional resumed training spec changed scientific content")
    if current_spec["task"] == support_diversity.TASK:
        parent_sampler = parent["sampler"].get("parent_sampler", {**parent["sampler"], "next_step": 450})
        wanted = support_diversity.sampler_state(parent_sampler, 450)
        if current["sampler"] != {key: value for key, value in wanted.items() if key != "next_step"}:
            raise ValueError("support fork lost its parent or fixed event/weight sampler")
    _inspect_frozen_source(parent, parent_spec)
    if parent_spec["task"] != current_spec["task"]:
        if (parent_spec["task"] != CONDITIONAL_TASK
                or current_spec["task"] not in (CONDITIONAL_CONTINUATION_TASK, support_diversity.TASK)
                or parent["git"]["commit"] != CONDITIONAL_PARENT_GIT
                or parent["spec"] != str(CONDITIONAL_PARENT_SPEC_PATH)
                or Path(current["parent_checkpoint"]).resolve() != CONDITIONAL_PARENT_CHECKPOINT.resolve()):
            raise ValueError("conditional second-round continuation requires its actual450 parent")
    return {"origin_training_git": CONDITIONAL_ORIGIN_GIT,
            "origin_training_spec": str(CONDITIONAL_ORIGIN_SPEC),
            "parent_checkpoint": current["parent_checkpoint"],
            "parent_training_git": parent["git"], "parent_training_spec": parent["spec"],
            "current_training_git": current["git"], "current_training_spec": current["spec"],
            "migration": ("registered_support_distribution_fork" if current_spec["task"] == support_diversity.TASK
                          and parent_spec["task"] != current_spec["task"] else "engineering_same_science_complete_ECP"),
            "execution": "automatic_target_execution"}


def register_conditional_resume(spec: dict, args, contract: dict) -> None:
    if spec.get("task") not in (CONDITIONAL_TASK, CONDITIONAL_CONTINUATION_TASK, support_diversity.TASK) or args.resume is None:
        return
    contract["parent_checkpoint"] = str(args.resume.resolve())
    parent = read_json(args.resume.resolve().parent.parent / "run_contract.json")
    contract["source_resume"] = _conditional_resume_record(parent, contract)


def conditional_resume_compatible(parent: dict, current: dict) -> bool:
    try:
        return current.get("source_resume") == _conditional_resume_record(parent, current)
    except (KeyError, OSError, ValueError, Pi05SourceTrainingError):
        return False


def validate_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    from .run import complete_checkpoint, resume_contract_compatible

    if spec["task"] in (CONTEXT_CONTINUATION_TASK, CONDITIONAL_CONTINUATION_TASK, support_diversity.TASK):
        _validate_joint_continuation(spec, args, contract, output)
        return
    root, mode, joint = settings(spec)
    attempts = root / mode / "train/attempts"
    if (output.parent.resolve() != attempts.resolve()
            or contract.get("loss_variant") != joint["loss_variant"] or contract.get("joint") != joint
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


def _validate_joint_continuation(spec: dict, args, contract: dict, output: Path) -> None:
    from .run import complete_checkpoint, packing_compatible

    root, mode, joint = settings(spec)
    attempts = root / mode / "train/attempts"
    checkpoint = args.resume.resolve()
    macro = int(checkpoint.name.removeprefix("macro_"))
    checkpoints = tuple(spec["execution"]["checkpoints"])
    if (output.parent.resolve() != attempts.resolve() or output.resolve() == checkpoint.parent.parent
            or (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists()
            or contract.get("loss_variant") != joint["loss_variant"] or contract.get("joint") != joint
            or macro not in (450, *checkpoints[:-1])
            or not complete_checkpoint(checkpoint)):
        raise ValueError("context continuation requires a new attempt and complete registered ECP")
    latest = max((int(path.name.removeprefix("macro_")) for path in attempts.glob(
        "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
    if (macro == 450 and latest != -1) or (macro != 450 and macro != latest):
        raise ValueError("context continuation must resume latest owned complete ECP")
    if macro == 450:
        conditional = mode == CONDITIONAL_MODE
        parent_checkpoint = CONDITIONAL_PARENT_CHECKPOINT if conditional else CONTEXT_PARENT_CHECKPOINT
        parent_spec = CONDITIONAL_PARENT_SPEC_PATH if conditional else CONTEXT_PARENT_SPEC_PATH
        parent_git = CONDITIONAL_PARENT_GIT if conditional else CONTEXT_PARENT_GIT
        if checkpoint != parent_checkpoint.resolve():
            raise ValueError("joint continuation parent must be the actual sealed fresh450 ECP")
        old = inspect_source(read_json(parent_spec), checkpoint)
        fixed = ("schema_version", "stage", "mode", "source", "lora", "operator", "optimizer",
                 "events", "sampler", "trainable_names", "source_trainable", "information_wall",
                 "loss_variant", "joint")
        if spec["task"] == support_diversity.TASK:
            fixed = tuple(key for key in fixed if key not in ("events", "sampler"))
            if spec != support_diversity.expected_spec(read_json(CONDITIONAL_PARENT_SPEC_PATH)):
                raise ValueError("support-distribution fork differs from registered §15")
        if (old["git"]["commit"] != parent_git
                or any(old.get(key) != contract.get(key) for key in fixed)):
            raise ValueError("context parent source, model, labels, loss or optimizer changed")
        if conditional and not conditional_resume_compatible(old, contract):
            raise ValueError("conditional continuation lost the actual450 source lineage")
    else:
        if checkpoint.parent.parent.parent.resolve() != attempts.resolve():
            raise ValueError("context continuation checkpoint is outside owned attempts")
        old = read_json(checkpoint.parent.parent / "run_contract.json")
        if mode == CONDITIONAL_MODE:
            from .run import resume_contract_compatible
            if not resume_contract_compatible(old, contract, allow_topology_change=True):
                raise ValueError("conditional continuation source or scientific contract changed")
            return
        mutable = {"topology", "microbatch", "frame_chunk", "parent_checkpoint"}
        if (not packing_compatible(old, contract)
                or {key: value for key, value in old.items() if key not in mutable}
                != {key: value for key, value in contract.items() if key not in mutable}):
            raise ValueError("context same-window resume scientific/source contract changed")


def _completed_metrics_source(root: Path, mode: str, checkpoint: Path, target: int) -> Path:
    """Read an earlier boundary only from the actual completed resume lineage."""
    attempts = root / mode / "train/attempts"
    completed = [path.parent for path in attempts.glob("*/completion.json")
                 if read_json(path).get("updates") == target]
    if len(completed) != 1:
        raise ValueError("readout requires one actually completed terminal training attempt")
    output, cursor = completed[0], completed[0]
    seen = set()
    while cursor != checkpoint.parent.parent:
        if cursor in seen or cursor.parent.resolve() != attempts.resolve():
            raise ValueError("readout checkpoint is outside completed continuation lineage")
        seen.add(cursor)
        parent = Path(read_json(cursor / "run_contract.json")["parent_checkpoint"])
        if parent.parent.parent == checkpoint.parent.parent and int(parent.name.removeprefix("macro_")) < int(checkpoint.name.removeprefix("macro_")):
            raise ValueError("readout checkpoint was not retained by the actual resume boundary")
        cursor = parent.parent.parent
    return output


def inspect_source(spec: dict, checkpoint: Path, *, _fixed_c12_630: bool = False) -> dict:
    """Keep actual training Git/spec identity separate from the current reader."""
    from .run import SCHEMA, STAGE, complete_checkpoint, frozen_git

    frozen_git()
    if spec["task"] == support_diversity.TASK and checkpoint.resolve() == support_diversity.C12_CHECKPOINT.resolve():
        run = inspect_source(read_json(support_diversity.C12_SPEC), checkpoint, _fixed_c12_630=True)
        if run["git"]["commit"] != support_diversity.C12_GIT:
            raise ValueError("fixed C12_630 actual training source changed")
        return run
    root, mode, joint = settings(spec)
    checkpoint = checkpoint.resolve()
    output = checkpoint.parent.parent
    diversity = spec["task"] == support_diversity.TASK
    continuation = spec["task"] in (CONTEXT_CONTINUATION_TASK, CONDITIONAL_CONTINUATION_TASK, support_diversity.TASK)
    macro = int(checkpoint.name.removeprefix("macro_"))
    target = 630 if diversity else 900 if continuation else 450
    allowed = (630,) if diversity else (630, 810, 900) if _fixed_c12_630 else (810, 900) if continuation else (450,)
    if mode == prefix_change.MODE:
        target, allowed = macro, (270, 450)
    if _fixed_c12_630 and checkpoint != support_diversity.C12_CHECKPOINT.resolve():
        raise ValueError("C12 diagnostic source must be its pre-fixed630")
    if (output.parent.resolve() != (root / mode / "train/attempts").resolve()
            or macro not in allowed or not complete_checkpoint(checkpoint)):
        raise ValueError("joint/context readout requires its complete owned registered endpoint")
    run = read_json(output / "run_contract.json")
    _inspect_frozen_source(run, spec)
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True,
                         weights_only=True)
    expected_sampler = {"schema_version": spec["events"]["schema_version"],
                        "seed": 20260928, "tasks": spec["events"]["task_ids"],
                        "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28,
                        "teacher_rounds": [[20260928, 1, "task"], [20260928, 1, "task", 1]],
                        "teacher_visits_per_round": 50, "teacher_demo_pool": list(range(50))}
    if diversity:
        from .data import TASKS
        parent_sampler = {**expected_sampler, "schema_version": "ember_operator_read_write_events_v3", "tasks": list(TASKS)}
        expected_sampler = {key: value for key, value in support_diversity.sampler_state(parent_sampler, macro).items() if key != "next_step"}
    facts = ((run.get("schema_version"), SCHEMA), (run.get("stage"), STAGE),
             (run.get("mode"), mode), (run.get("joint"), joint),
             (run.get("loss_variant"), joint["loss_variant"]), (run.get("operator"), spec["operator"]),
             (run.get("optimizer"), spec["optimization"]), (run.get("events"), spec["events"]),
             (run.get("source_trainable"), 0), (run.get("sampler"), expected_sampler),
             (run.get("continuation"), spec.get("continuation")), (run.get("pilot_arm"), None),
             (trainer.get("stage"), STAGE), (trainer.get("next_macro"), macro),
             (trainer.get("metrics_rows"), macro), (trainer.get("scheduler", {}).get("last_epoch"), macro),
             (bool(trainer.get("optimizer", {}).get("param_groups")), True),
             (trainer.get("scaler"), None),
             (trainer.get("training_state"), {"updates": macro, "mode": mode, "loss_variant": joint["loss_variant"]}),
             (trainer.get("sampler_state"), {**expected_sampler, "next_step": macro}))
    terminal = _completed_metrics_source(root, mode, checkpoint, target) if continuation else output
    metrics = [json.loads(line) for line in (terminal / "metrics.jsonl").read_text().splitlines()]
    completion = read_json(terminal / ("stopped_at_ecp.json" if mode == prefix_change.MODE and macro == 270 else "completion.json"))
    if mode == CONDITIONAL_MODE and run.get("source_resume") is not None:
        parent_checkpoint = Path(run["parent_checkpoint"])
        parent_run = read_json(parent_checkpoint.parent.parent / "run_contract.json")
        parent_macro = int(parent_checkpoint.name.removeprefix("macro_"))
        parent_metrics = (parent_checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()
        current_metrics = (output / "metrics.jsonl").read_text().splitlines()
        owned_parent = (parent_checkpoint.parent.parent.parent.resolve()
                        == (root / mode / "train/attempts").resolve())
        fixed450 = continuation and parent_checkpoint.resolve() == CONDITIONAL_PARENT_CHECKPOINT.resolve()
        if (not (owned_parent or fixed450)
                or not complete_checkpoint(parent_checkpoint) or parent_macro >= macro
                or not conditional_resume_compatible(parent_run, run)
                or len(parent_metrics) < parent_macro
                or current_metrics[:parent_macro] != parent_metrics[:parent_macro]):
            raise ValueError("conditional readout lost its complete actual parent or copied event history")
    if (any(actual != wanted for actual, wanted in facts) or not valid_metrics(metrics, mode=mode, updates=target)
            or completion.get("updates") != target
            or completion.get("checkpoint") != str(terminal / "checkpoints" / f"macro_{target:08d}")):
        raise ValueError("joint source loss/ECP/optimizer/sampler/completion changed")
    return run


def valid_metrics(metrics: list, *, mode: str = MODE, updates: int = 450) -> bool:
    if len(metrics) != updates or [row["update"] for row in metrics] != list(range(1, updates + 1)):
        return False
    loss = "full" if mode in (CONDITIONAL_MODE, prefix_change.MODE) else LOSS
    for row in metrics:
        if (row.get("mode") != mode or row.get("loss_variant") != loss
                or row.get("queries") != 112 or len(row.get("jobs", ())) != 4):
            return False
        if any(job.get("loss_variant") != loss
               or (bool(job.get("public_query_reuse")) != (loss == LOSS))
               for job in row["jobs"]):
            return False
    return True
