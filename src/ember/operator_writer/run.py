"""Formal T/U cross-episode FM training with sealed ECP recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.lora import copy_task_lora_state_, task_lora_state_dict, validate_lora_state
from ember.pi05_eval_contract import git_state, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import (initialize_deferred_process_group, initialize_distributed,
                                      load_policy, seed_everything)
from ember.source_sft.control import clamped_lr_multiplier
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.replay import sum_writer_gradients
from ember.writer.runtime import autocast
from ember.writer.task_execution import condition_assignment

from .credit import gradient_groups, one_job
from .data import (CHECKPOINTS, CONTINUATION_CHECKPOINTS, CONTINUATION_UPDATES,
                   CONTINUATION1350_CHECKPOINTS, CONTINUATION1350_UPDATES,
                   CONTINUATION1800_CHECKPOINTS, CONTINUATION1800_UPDATES,
                   PILOT_CHECKPOINTS, PILOT_UPDATES,
                   CONTINUATION2340_CHECKPOINTS, CONTINUATION2340_UPDATES,
                   CONTINUATION2790_CHECKPOINTS, CONTINUATION2790_UPDATES,
                   TASKS, UPDATES, FormalData)
from .model import OperatorReadWrite
from .native import read_native_video
from . import change_clock


REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "configs/operator_read_write_v1/learning_spec.json"
CHANGE_CLOCK_SPEC_PATH = REPO / "configs/operator_read_write_v1" / change_clock.SPEC_NAME
CONTINUATION_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation900_spec.json"
CONTINUATION1350_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation1350_spec.json"
CONTINUATION1800_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation1800_spec.json"
PILOT_SPEC_PATH = REPO / "configs/operator_read_write_v1/public_function_pilot_spec.json"
CONTINUATION2340_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation2340_spec.json"
CONTINUATION2790_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation2790_spec.json"
PILOT_ROOT = Path("/data1/user/ymdai/ember_runs/operator_public_function_pilot_20260929")
CONTINUATION2340_ROOT = Path(
    "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340")
CONTINUATION2790_ROOT = Path(
    "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2790")
CONTINUATION900_ROOT = Path(
    "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900")
CONTINUATION1350_ROOT = Path(
    "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation1350")
SEALED_ROOT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1")
SEALED_SPEC_PATH = Path("/data1/user/ymdai/projects/EMBER-operator-stage1-formal"
                        "/configs/operator_read_write_v1/learning_spec.json")
SCHEMA = "ember_operator_read_write_formal_run_v1"
STAGE = "operator_read_write_learning"
OPERATOR_CONTRACT = {
    "rank": 128, "alpha": 128, "targets": 38, "identity_seed": 20260721,
    "module_seed": 7, "probe_seed": 1729, "probe_shape": [50, 32],
    "teaching_camera": "dual", "frame_stride": 5, "frame_chunk": 8,
    "source_frozen": True,
    "mode_T": "actual execution A is also teaching key",
    "mode_U": "independent S cloned from A is teaching key only",
    "value_width": 256, "memory_dtype": "float32", "memory_step": 1.0,
    "additional_loss": False,
}
OPTIMIZATION_CONTRACT = {
    "seed": 7, "lr": 0.0003, "betas": [0.9, 0.95], "eps": 1e-8,
    "weight_decay": 0.0001, "grad_clip": 1.0, "warmup_updates": 150,
    "decay_updates": 1200, "floor_lr": 1e-5,
}
EVENT_CONTRACT = {
    "schema_version": "ember_operator_read_write_events_v2", "seed": 20260928,
    "task_ids": list(TASKS), "task_permutation_seed": [20260928, 0],
    "teacher_permutation_seed": [20260928, 1], "query_seed": [20260928, 2],
    "teacher_pool": list(range(30)), "query_demo_pool": [0, 49],
    "queries_per_task": 28, "tasks_per_update": 4, "query_action_offset": 1,
    "logical_queries_per_macro": 112, "same_events_both_modes": True,
}
EXECUTION_CONTRACT = {
    "modes": ["T", "U"], "updates_per_mode": UPDATES, "queries_per_mode": 30240,
    "world_size": 2, "initial_policy_microbatch": 28,
    "oom_only_policy_microbatches": [14, 7], "oom_only_frame_chunk": 4,
    "checkpoints": list(CHECKPOINTS), "only_selected_checkpoint": 270,
}
CONTINUATION_EVENTS = {**EVENT_CONTRACT,
                       "schema_version": "ember_operator_read_write_events_v3",
                       "teacher_round2_permutation_seed": [20260928, 1, "task", 1],
                       "teacher_round_visits": 50, "teacher_demo_pool": list(range(50))}
CONTINUATION_EVENTS.pop("teacher_pool")
CONTINUATION_EXECUTION = {**EXECUTION_CONTRACT, "updates_per_mode": CONTINUATION_UPDATES,
                          "queries_per_mode": CONTINUATION_UPDATES * 112,
                          "world_sizes": [2, 3, 4], "checkpoints": list(CONTINUATION_CHECKPOINTS),
                          "only_selected_checkpoints": [450, 900]}
CONTINUATION_EXECUTION.pop("world_size")
CONTINUATION_EXECUTION.pop("only_selected_checkpoint")
CONTINUATION1350_EVENTS = {**CONTINUATION_EVENTS,
                           "schema_version": "ember_operator_read_write_events_v4",
                           "teacher_round3_permutation_seed": [20260928, 1, "task", 2]}
CONTINUATION1350_EXECUTION = {**CONTINUATION_EXECUTION,
                              "updates_per_mode": CONTINUATION1350_UPDATES,
                              "queries_per_mode": CONTINUATION1350_UPDATES * 112,
                              "checkpoints": list(CONTINUATION1350_CHECKPOINTS),
                              "only_selected_checkpoints": [1080, 1350]}
CONTINUATION1800_EVENTS = {**CONTINUATION1350_EVENTS,
                           "schema_version": "ember_operator_read_write_events_v5",
                           "teacher_round4_permutation_seed": [20260928, 1, "task", 3]}
CONTINUATION1800_EXECUTION = {**CONTINUATION1350_EXECUTION,
                              "modes": ["T"], "updates_per_mode": CONTINUATION1800_UPDATES,
                              "queries_per_mode": CONTINUATION1800_UPDATES * 112,
                              "checkpoints": list(CONTINUATION1800_CHECKPOINTS),
                              "only_selected_checkpoints": [1710, 1800]}
PILOT_EVENTS = {**CONTINUATION1800_EVENTS,
                "schema_version": "ember_operator_read_write_events_v6",
                "teacher_round5_permutation_seed": [20260928, 1, "task", 4]}
PILOT_EXECUTION = {**CONTINUATION1800_EXECUTION,
                   "updates_per_mode": PILOT_UPDATES, "queries_per_mode": PILOT_UPDATES * 112,
                   "checkpoints": list(PILOT_CHECKPOINTS), "only_selected_checkpoints": [1890]}
CONTINUATION2340_EVENTS = {**PILOT_EVENTS,
                           "schema_version": "ember_operator_read_write_events_v7",
                           "teacher_round6_permutation_seed": [20260928, 1, "task", 5]}
CONTINUATION2340_EXECUTION = {**PILOT_EXECUTION,
                              "updates_per_mode": CONTINUATION2340_UPDATES,
                              "queries_per_mode": CONTINUATION2340_UPDATES * 112,
                              "checkpoints": list(CONTINUATION2340_CHECKPOINTS),
                              "only_selected_checkpoints": list(CONTINUATION2340_CHECKPOINTS)}
CONTINUATION2790_EVENTS = {**CONTINUATION2340_EVENTS,
                           "schema_version": "ember_operator_read_write_events_v8",
                           "teacher_round7_permutation_seed": [20260928, 1, "task", 6]}
CONTINUATION2790_EXECUTION = {**CONTINUATION2340_EXECUTION,
                              "updates_per_mode": CONTINUATION2790_UPDATES,
                              "queries_per_mode": CONTINUATION2790_UPDATES * 112,
                              "checkpoints": list(CONTINUATION2790_CHECKPOINTS),
                              "only_selected_checkpoints": list(CONTINUATION2790_CHECKPOINTS)}
PILOT_ARMS = {"control": "full", "public_aux": "full_plus_public_beta"}
PILOT_CONTRACT = {"arms": list(PILOT_ARMS), "loss_variants": PILOT_ARMS,
                  "public_loss_coefficient": 1.0,
                  "query_reuse": "same_112_query_action_tau_noise",
                  "public_credit": "direct_A_B0_only"}


def specification(path: Path = SPEC_PATH) -> dict:
    path = path.resolve()
    if path == CHANGE_CLOCK_SPEC_PATH:
        spec = read_json(path)
        if spec != change_clock.expected_spec(specification(SPEC_PATH)):
            raise ValueError("change-clock fresh learning contract changed")
        return spec
    if path == CONTINUATION2790_SPEC_PATH:
        spec, base = read_json(path), specification(CONTINUATION2340_SPEC_PATH)
        expected = {**base, "task": "operator_read_write_continuation_2790_20260929",
                    "design": "docs/designs/operator_read_write_learning_design.md#21",
                    "run_root": str(CONTINUATION2790_ROOT),
                    "events": CONTINUATION2790_EVENTS, "execution": CONTINUATION2790_EXECUTION,
                    "evaluation": {**base["evaluation"],
                                   "bank_macros": list(CONTINUATION2790_CHECKPOINTS)},
                    "continuation": {
                        "parent_run_root": str(CONTINUATION2340_ROOT), "parent_arm": "T",
                        "parent_macro": 2340,
                        "parent_training_git": "e2afbfd7c997e3f792921600608efa2fa3c1b25a",
                        "parent_event_schema": CONTINUATION2340_EVENTS["schema_version"],
                        "parent_loss_variant": "full",
                        "sampler_migration": "append_teacher_round_6_v7_to_v8_at_2340"},
                    "budget": {"new_gpu_hours_expected": 14.5, "new_gpu_hours_hard": 18,
                               "peak_new_gib": 64}}
        if spec != expected:
            raise ValueError("operator 2790 continuation contract changed")
        return spec
    if path == CONTINUATION2340_SPEC_PATH:
        spec, base = read_json(path), specification(PILOT_SPEC_PATH)
        expected = {**base, "task": "operator_read_write_continuation_2340_20260929",
                    "design": "docs/designs/operator_read_write_learning_design.md#20",
                    "run_root": str(CONTINUATION2340_ROOT),
                    "events": CONTINUATION2340_EVENTS, "execution": CONTINUATION2340_EXECUTION,
                    "evaluation": {**base["evaluation"],
                                   "bank_macros": list(CONTINUATION2340_CHECKPOINTS)},
                    "continuation": {
                        "parent_run_root": str(PILOT_ROOT), "parent_arm": "control",
                        "parent_macro": 1890,
                        "parent_training_git": "9801641d0967e163d91474ff92e6fb6520be1084",
                        "parent_event_schema": PILOT_EVENTS["schema_version"],
                        "parent_loss_variant": "full",
                        "sampler_migration": "append_teacher_round_5_v6_to_v7_at_1890"},
                    "budget": {"new_gpu_hours_expected": 14, "new_gpu_hours_hard": 30,
                               "peak_new_gib": 128}}
        expected.pop("pilot")
        if spec != expected:
            raise ValueError("operator 2340 continuation contract changed")
        return spec
    if path == PILOT_SPEC_PATH:
        spec, base = read_json(path), specification(CONTINUATION1800_SPEC_PATH)
        expected = {**base, "task": "operator_public_function_pilot_20260929",
                    "design": "docs/designs/operator_read_write_learning_design.md#18",
                    "run_root": str(PILOT_ROOT), "events": PILOT_EVENTS,
                    "execution": PILOT_EXECUTION, "pilot": PILOT_CONTRACT,
                    "evaluation": {**base["evaluation"], "bank_macros": [1890]},
                    "continuation": {
                        "parent_run_root": base["run_root"], "parent_macro": 1800,
                        "parent_training_git": "fcc23cd15cc475530c385e354670efee6bacfa12",
                        "parent_event_schema": CONTINUATION1800_EVENTS["schema_version"],
                        "sampler_migration": "append_teacher_round_4_v5_to_v6_at_1800"},
                    "budget": {"new_gpu_hours_expected": 6, "new_gpu_hours_hard": 8,
                               "peak_new_gib": 24}}
        if spec != expected:
            raise ValueError("operator public function pilot contract changed")
        return spec
    if path == CONTINUATION1800_SPEC_PATH:
        spec, base = read_json(path), specification(CONTINUATION1350_SPEC_PATH)
        expected = {**base, "design": "docs/designs/operator_read_write_learning_design.md#16",
                    "run_root": "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation1800",
                    "events": CONTINUATION1800_EVENTS, "execution": CONTINUATION1800_EXECUTION,
                    "evaluation": {**base["evaluation"], "bank_macros": [1710, 1800]},
                    "continuation": {
                        "parent_run_root": str(CONTINUATION1350_ROOT), "parent_macro": 1350,
                        "parent_training_git": "14bac4cdd6c27eee06f5574317da8257834e3884",
                        "parent_event_schema": CONTINUATION1350_EVENTS["schema_version"],
                        "sampler_migration": "append_teacher_round_3_v4_to_v5_at_1350"},
                    "budget": {"new_gpu_hours_expected": 10.2, "new_gpu_hours_hard": 14,
                               "peak_new_gib": 32}}
        if spec != expected:
            raise ValueError("operator 1800 continuation contract changed")
        return spec
    if path == CONTINUATION1350_SPEC_PATH:
        spec, base = read_json(path), specification(CONTINUATION_SPEC_PATH)
        expected = {**base, "design": "docs/designs/operator_read_write_learning_design.md#15",
                    "run_root": "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation1350",
                    "events": CONTINUATION1350_EVENTS, "execution": CONTINUATION1350_EXECUTION,
                    "evaluation": {**base["evaluation"], "bank_macros": [1080, 1350]},
                    "continuation": {
                        "parent_run_root": str(CONTINUATION900_ROOT), "parent_macro": 900,
                        "parent_training_git": "81846ed35933222b14ac693a0b760268ecff7f17",
                        "parent_event_schema": CONTINUATION_EVENTS["schema_version"],
                        "sampler_migration": "append_teacher_round_2_v3_to_v4_at_900"},
                    "budget": {"new_gpu_hours_expected": 17.2, "new_gpu_hours_hard": 24,
                               "peak_new_gib": 52}}
        if spec != expected:
            raise ValueError("operator 1350 continuation contract changed")
        return spec
    if path == CONTINUATION_SPEC_PATH:
        spec, base = read_json(path), specification()
        expected = {**base, "design": "docs/designs/operator_read_write_learning_design.md#11",
                    "run_root": "/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation900",
                    "events": CONTINUATION_EVENTS, "execution": CONTINUATION_EXECUTION,
                    "evaluation": {**base["evaluation"], "bank_macros": [450, 900]},
                    "continuation": {
                        "parent_run_root": str(SEALED_ROOT), "parent_macro": 270,
                        "parent_training_git": "784febbff32d991e53b9e5c6ba9f74683890425e",
                        "parent_event_schema": EVENT_CONTRACT["schema_version"],
                        "sampler_migration": "teacher_pool_0_29_is_visit_index_not_demo_pool"},
                    "budget": {"new_gpu_hours_expected": 22.1, "new_gpu_hours_hard": 30,
                               "peak_new_gib": 56, "throughput_profile_gpu_hours_max": 0.5}}
        expected["evaluation"].pop("bank_macro")
        if ({key: value for key, value in spec.items() if key != "budget"}
                != {key: value for key, value in expected.items() if key != "budget"}):
            raise ValueError("operator 900 continuation contract changed")
        return spec
    if path != SPEC_PATH:
        raise ValueError("unregistered operator specification path")
    spec = read_json(path)
    expected = (
        (spec.get("schema_version"), "ember_operator_read_write_learning_v1"),
        (spec.get("task"), "operator_read_write_learning_20260928"),
        (spec.get("operator"), OPERATOR_CONTRACT),
        (spec.get("optimization"), OPTIMIZATION_CONTRACT),
        (spec.get("events"), EVENT_CONTRACT),
        (spec.get("execution"), EXECUTION_CONTRACT),
        (spec["evaluation"].get("task_ids"), [3, 6, 11, 16, 23, 26, 31, 39]),
        (spec["evaluation"].get("video_schedule_seed"), 7),
        (spec["evaluation"].get("scenes"),
         "/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes"),
    )
    if any(actual != wanted for actual, wanted in expected):
        raise ValueError("formal operator read/write contract changed")
    return spec


def frozen_git(*, continuation: bool = False, change_clock_pilot: bool = False) -> dict:
    state = git_state(REPO)
    if state["branch"] or state["dirty_paths"]:
        raise ValueError("GPU calculation requires clean detached frozen source")
    refs = subprocess.run(["git", "branch", "-r", "--contains", state["commit"]], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.splitlines()
    allowed = (("origin/codex/operator-change-clock", "origin/main") if change_clock_pilot else
               ("origin/codex/demonstration-transfer", "origin/main")
               if continuation else ("origin/main",))
    pushed_ref = next((ref for ref in allowed if any(row.strip() == ref for row in refs)), None)
    if pushed_ref is None:
        raise ValueError("formal frozen code commit must be pushed to its registered source ref")
    return {"commit": state["commit"], "branch": "", "dirty_paths": [],
            "pushed_ref": pushed_ref}


def gather(value, world: int):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def resume_contract_compatible(parent: dict, current: dict, *, allow_topology_change: bool = False) -> bool:
    """Only registered packing and, for the clock arm, physical topology may change."""
    mutable = ("microbatch", "frame_chunk", "topology") if allow_topology_change else ("microbatch", "frame_chunk")
    return (packing_compatible(parent, current)
            and {k: v for k, v in parent.items() if k not in mutable}
            == {k: v for k, v in current.items() if k not in mutable})


def packing_compatible(parent: dict, current: dict) -> bool:
    return ((parent.get("microbatch"), current.get("microbatch")) in {
            (28, 28), (28, 14), (28, 7), (14, 14), (14, 7), (7, 7)}
            and (parent.get("frame_chunk"), current.get("frame_chunk")) in {
                (8, 8), (8, 4), (4, 4)})


def complete_checkpoint(path: Path) -> bool:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    macro = manifest.get("next_macro")
    world = manifest.get("world_size")
    allowed = ((macro in CHECKPOINTS and world in (1, 2, 3, 4))
               or (macro in (*CONTINUATION_CHECKPOINTS, *CONTINUATION1350_CHECKPOINTS,
                             *CONTINUATION1800_CHECKPOINTS, *PILOT_CHECKPOINTS,
                             *CONTINUATION2340_CHECKPOINTS, *CONTINUATION2790_CHECKPOINTS)
                   and world in (2, 3, 4)))
    expected_files = ({"ecp.safetensors", "trainer_state.pt"}
                      | {f"rank_{rank:02d}_state.pt" for rank in range(world)}) if allowed else set()
    return (manifest.get("stage") == STAGE and manifest.get("run_contract_schema") == SCHEMA
            and allowed and set(files) == expected_files
            and all((path / name).is_file() and (path / name).stat().st_size == row.get("bytes")
                    for name, row in files.items()))


def validate_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    root = Path(spec["run_root"])
    event_schema = spec.get("events", {}).get("schema_version")
    if event_schema in (CONTINUATION1350_EVENTS["schema_version"],
                        CONTINUATION1800_EVENTS["schema_version"],
                        PILOT_EVENTS["schema_version"],
                        CONTINUATION2340_EVENTS["schema_version"],
                        CONTINUATION2790_EVENTS["schema_version"]):
        _validate_late_continuation_attempt(spec, args, contract, output)
        return
    if spec.get("execution", {}).get("updates_per_mode") == CONTINUATION_UPDATES:
        _validate_continuation_attempt(spec, args, contract, output)
        return
    if args.resume:
        parent = args.resume.resolve().parent.parent
        attempts = (root / args.mode / "train" / "attempts").resolve()
        if (parent.parent != attempts or parent == output.resolve()
                or args.resume.name not in {f"macro_{step:08d}" for step in CHECKPOINTS}):
            raise ValueError("resume requires this arm's registered ECP90/180/270")
        if not resume_contract_compatible(read_json(parent / "run_contract.json"), contract,
                                          allow_topology_change=spec.get("task") == change_clock.TASK):
            raise ValueError("source, parameter, sampler, numerical or physical topology changed")
        if not complete_checkpoint(args.resume):
            raise ValueError("requested same-arm ECP is incomplete")
        latest = max((int(path.name.split("_")[-1]) for path in attempts.glob(
            "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
        if int(args.resume.name.split("_")[-1]) != latest:
            raise ValueError("resume must use the latest complete same-arm ECP")
    if (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("formal attempt output already exists")


def _validate_continuation_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    if args.resume is None:
        raise ValueError("900 continuation requires a complete parent ECP")
    checkpoint = args.resume.resolve()
    macro = int(checkpoint.name.removeprefix("macro_")) if re.fullmatch(r"macro_[0-9]{8}", checkpoint.name) else -1
    if macro not in (270, *CONTINUATION_CHECKPOINTS[:-1]) or not complete_checkpoint(checkpoint):
        raise ValueError("900 continuation requires a registered complete ECP270..810")
    parent = checkpoint.parent.parent
    if output.resolve() == parent or (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("continuation attempt output already exists")
    attempts = Path(spec["run_root"]) / args.mode / "train/attempts"
    latest = max((int(path.name.split("_")[-1]) for path in attempts.glob(
        "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
    if (macro == 270 and latest != -1) or (macro != 270 and macro != latest):
        raise ValueError("resume requires the latest complete same-arm continuation ECP")
    if macro == 270:
        expected = SEALED_ROOT / args.mode / "train/attempts/fresh/checkpoints/macro_00000270"
        if checkpoint != expected:
            raise ValueError("continuation parent is not the sealed same-arm 270 ECP")
        from .bank import inspect_training_source

        old = inspect_training_source(read_json(SEALED_SPEC_PATH), checkpoint, args.mode,
                                      sealed_evaluation=True)
        if (old.get("events") != EVENT_CONTRACT or old.get("optimizer") != OPTIMIZATION_CONTRACT
                or old.get("source") != contract["source"] or old.get("lora") != contract["lora"]
                or old.get("operator") != contract["operator"]
                or any(old.get(key) != contract.get(key) for key in (
                    "trainable_names", "source_trainable", "information_wall"))):
            raise ValueError("sealed 270 source, parameters or optimization changed")
    else:
        if parent.parent.resolve() != attempts.resolve():
            raise ValueError("continuation ECP is outside this arm's registered attempts")
        old = read_json(parent / "run_contract.json")
        fixed = ("schema_version", "stage", "git", "spec", "mode", "source", "lora", "operator",
                 "events", "optimizer", "sampler", "trainable_names", "source_trainable",
                 "information_wall", "continuation")
        if any(old.get(key) != contract.get(key) for key in fixed):
            raise ValueError("continuation scientific source or frozen Git changed")
        if not packing_compatible(old, contract):
            raise ValueError("continuation packing changed outside registered OOM transitions")


def _validate_late_continuation_attempt(spec: dict, args, contract: dict, output: Path) -> None:
    """One registered parent migration and same-window ECP resume owner."""
    parent_macro = int(spec["continuation"]["parent_macro"])
    checkpoints = tuple(spec["execution"]["checkpoints"])
    pilot = parent_macro == 1800
    arm = getattr(args, "pilot_arm", None) if pilot else args.mode
    if (parent_macro not in (900, 1350, 1800, 1890, 2340)
            or args.mode not in spec["execution"]["modes"]
            or pilot and (arm not in PILOT_ARMS or contract.get("pilot_arm") != arm
                          or contract.get("loss_variant") != PILOT_ARMS[arm])
            or parent_macro in (1890, 2340) and
            (arm != "T" or contract.get("loss_variant") != "full" or
             spec["continuation"].get("parent_arm") != ("control" if parent_macro == 1890 else "T"))):
        raise ValueError("unregistered continuation parent or arm")
    if args.resume is None:
        raise ValueError("continuation requires a complete same-arm ECP")
    checkpoint = args.resume.resolve()
    macro = int(checkpoint.name.removeprefix("macro_")) if re.fullmatch(r"macro_[0-9]{8}", checkpoint.name) else -1
    if macro not in (parent_macro, *checkpoints[:-1]) or not complete_checkpoint(checkpoint):
        raise ValueError("continuation requires a registered complete parent/intermediate ECP")
    parent = checkpoint.parent.parent
    if output.resolve() == parent or (output / "run_contract.json").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("continuation attempt output already exists")
    attempts = Path(spec["run_root"]) / arm / "train/attempts"
    if output.parent.resolve() != attempts.resolve():
        raise ValueError("continuation attempt is outside its registered arm")
    latest = max((int(path.name.split("_")[-1]) for path in attempts.glob(
        "*/checkpoints/macro_*") if complete_checkpoint(path)), default=-1)
    if (macro == parent_macro and latest != -1) or (macro != parent_macro and macro != latest):
        raise ValueError("resume requires the latest complete same-arm continuation ECP")
    if macro == parent_macro:
        expected = (Path(spec["continuation"]["parent_run_root"])
                    / spec["continuation"].get("parent_arm", args.mode)
                    / "train/attempts/continuation/checkpoints" / f"macro_{parent_macro:08d}")
        if checkpoint != expected:
            raise ValueError("continuation parent is not the original same-arm ECP")
        from .bank import inspect_training_source

        old_spec = (CONTINUATION_SPEC_PATH if parent_macro == 900 else
                    CONTINUATION1350_SPEC_PATH if parent_macro == 1350 else
                    CONTINUATION1800_SPEC_PATH if parent_macro == 1800 else
                    PILOT_SPEC_PATH if parent_macro == 1890 else CONTINUATION2340_SPEC_PATH)
        old = inspect_training_source(specification(old_spec), checkpoint, args.mode,
                                      sealed_evaluation=True)
        fixed = ("schema_version", "stage", "mode", "source", "lora", "operator", "optimizer",
                 "trainable_names", "source_trainable", "information_wall")
        if any(old.get(key) != contract.get(key) for key in fixed):
            raise ValueError("sealed parent scientific source or optimizer changed")
        if parent_macro in (1890, 2340) and (
                old.get("loss_variant") != "full"
                or parent_macro == 1890 and old.get("pilot_arm") != "control"
                or parent_macro == 2340 and old.get("pilot_arm") is not None
                or old.get("git", {}).get("commit")
                != spec["continuation"]["parent_training_git"]):
            raise ValueError("sealed parent arm, loss or Git changed")
    else:
        if parent.parent.resolve() != attempts.resolve():
            raise ValueError("continuation ECP is outside this arm's registered attempts")
        old = read_json(parent / "run_contract.json")
        fixed = ("schema_version", "stage", "git", "spec", "mode", "source", "lora", "operator",
                 "events", "optimizer", "sampler", "trainable_names", "source_trainable",
                 "information_wall", "continuation")
        if pilot:
            fixed += ("pilot_arm", "loss_variant", "pilot")
        elif parent_macro in (1890, 2340):
            fixed += ("loss_variant",)
        if any(old.get(key) != contract.get(key) for key in fixed):
            raise ValueError("continuation scientific source or frozen Git changed")
        if not packing_compatible(old, contract):
            raise ValueError("continuation packing changed outside registered OOM transitions")


@dataclass
class Runtime:
    policy: torch.nn.Module
    writer: OperatorReadWrite
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: object
    source: dict
    device: torch.device
    identity: dict[str, torch.Tensor]
    identity_resets: int = 0

    def restore_identity(self) -> None:
        copy_task_lora_state_(self.policy, self.identity, self.lora)
        self.identity_resets += 1

    def physical_delta(self) -> float:
        current = task_lora_state_dict(self.policy)
        return float(torch.stack([(current[name].float() - expected.float()).norm()
                                  for name, expected in self.identity.items()]).norm())

    def compile(self, condition: tuple, *, frame_chunk: int = 8,
                retain_native: bool = False) -> tuple[dict, dict | None]:
        self.restore_identity()
        with autocast(self.device):
            x, h = read_native_video(self.policy, self.writer.public_state(), self.writer.probe,
                                     condition, self.writer.names, frame_chunk=frame_chunk)
            if retain_native:
                if h.requires_grad:
                    h.retain_grad()
                for value in x.values():
                    if value.requires_grad:
                        value.retain_grad()
            state = self.writer(x, h)
        validate_lora_state(state, self.lora)
        return state, ({"x": x, "h": h} if retain_native else None)


def build_runtime(asset_root: Path, spec: dict, device: torch.device, mode: str, *,
                  evaluation: bool = False) -> Runtime:
    source_config = spec["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                       evaluation_mode="formal")
    tokenizer_path = asset_root / source_config["tokenizer"]
    stats = read_json(asset_root / source_config["normalization"])["stats"]
    if evaluation:
        from ember.pi05_eval.worker_setup import load_policy as load_eval_policy

        recipe = read_json(asset_root / "configs/pi05_target_evaluation_v1.json")
        policy, processor, _ = load_eval_policy(
            Path(source["model_path"]), stats, tokenizer_path, recipe["policy"])
    else:
        policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
        processor = Pi05LiberoProcessor(stats, tokenizer_path, 200, str(device))
    base = load_pi05_lora_contract(asset_root / source_config["lora_contract"])
    lora = derive_pi05_lora_rank(base, rank=128)
    identity = prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    writer = OperatorReadWrite(lora, identity, mode).to(device)
    if any(p.requires_grad for p in policy.parameters()):
        raise ValueError("physical source parameters must be frozen")
    return Runtime(policy, writer, Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device)),
                   processor, lora, source, device, identity)


def optimizer_for(writer: OperatorReadWrite, spec: dict):
    opt = spec["optimization"]
    parameters = tuple(p for p in writer.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                  eps=opt["eps"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: clamped_lr_multiplier(step, warmup=opt["warmup_updates"],
                                                      decay=opt["decay_updates"], peak=opt["lr"],
                                                      floor=opt["floor_lr"]))
    return optimizer, scheduler, parameters




@dataclass
class Session:
    spec: dict
    context: object
    data: FormalData
    runtime: Runtime
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameters: tuple[torch.nn.Parameter, ...]
    output: Path
    contract: dict
    mode: str
    microbatch: int
    frame_chunk: int


def prepare_train(spec: dict, args) -> Session:
    continuation = spec["execution"]["updates_per_mode"] in (CONTINUATION_UPDATES,
                                                                CONTINUATION1350_UPDATES,
                                                                CONTINUATION1800_UPDATES,
                                                                PILOT_UPDATES,
                                                                CONTINUATION2340_UPDATES,
                                                                CONTINUATION2790_UPDATES)
    clock_pilot = spec["task"] == change_clock.TASK
    git = frozen_git(continuation=continuation, change_clock_pilot=clock_pilot)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    allowed_worlds = spec["execution"].get("world_sizes", [spec["execution"].get("world_size")])
    if context.world_size not in allowed_worlds or os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise ValueError("operator formal train/resume topology or NCCL contract changed")
    torch.set_num_threads(args.cpu_threads)
    seed_everything(7, context)
    data = FormalData(args.asset_root, spec)
    runtime = build_runtime(args.asset_root, spec, context.device, args.mode)
    runtime.writer.train()
    seed_everything(7, context)
    optimizer, scheduler, parameters = optimizer_for(runtime.writer, spec)
    root = Path(spec["run_root"])
    pilot = spec["execution"]["updates_per_mode"] == PILOT_UPDATES
    next_window = spec["execution"]["updates_per_mode"] in (
        CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES)
    output = root / (args.pilot_arm if pilot else args.mode) / "train" / "attempts" / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    initialize_deferred_process_group(context, rendezvous_root=output)
    local = {"rank": context.rank, "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
             "numa_node": context.numa_node, "cpu_affinity": list(context.cpu_affinity or ())}
    topology = {"host": socket.gethostname(), "world_size": context.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
                "ranks": gather(local, context.world_size)}
    contract = {"schema_version": SCHEMA, "stage": STAGE, "git": git,
                "spec": str(CHANGE_CLOCK_SPEC_PATH if clock_pilot else
                            CONTINUATION2790_SPEC_PATH if spec["execution"]["updates_per_mode"] == CONTINUATION2790_UPDATES else
                            CONTINUATION2340_SPEC_PATH if next_window else
                            PILOT_SPEC_PATH if pilot else
                            CONTINUATION1800_SPEC_PATH if spec["execution"]["updates_per_mode"] == CONTINUATION1800_UPDATES
                            else CONTINUATION1350_SPEC_PATH if spec["execution"]["updates_per_mode"] == CONTINUATION1350_UPDATES
                            else CONTINUATION_SPEC_PATH if continuation else SPEC_PATH),
                "mode": args.mode, "source": runtime.source,
                "lora": runtime.lora.to_dict(), "operator": spec["operator"],
                "events": spec["events"], "optimizer": spec["optimization"],
                "topology": topology, "microbatch": args.microbatch, "frame_chunk": args.frame_chunk,
                "sampler": {key: value for key, value in data.sampler_state().items() if key != "next_step"},
                "trainable_names": [name for name, p in runtime.writer.named_parameters() if p.requires_grad],
                "source_trainable": sum(p.numel() for p in runtime.policy.parameters() if p.requires_grad),
                "information_wall": "teacher exact language + dual RGB only; independent query own RGB/state/action FM"}
    if continuation:
        contract["continuation"] = spec["continuation"]
        contract["parent_checkpoint"] = str(args.resume.resolve())
        if pilot:
            contract.update(pilot_arm=args.pilot_arm, loss_variant=PILOT_ARMS[args.pilot_arm],
                            pilot=spec["pilot"])
        elif next_window:
            contract["loss_variant"] = "full"
        if spec["execution"]["updates_per_mode"] in (CONTINUATION1350_UPDATES,
                                                       CONTINUATION1800_UPDATES,
                                                       CONTINUATION2340_UPDATES,
                                                       CONTINUATION2790_UPDATES):
            contract["stop_after_macro"] = args.stop_after_macro
    error = None
    try:
        if context.is_main:
            validate_attempt(spec, args, contract, output)
            write_json_atomic(output / "run_contract.json", contract)
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, context.world_size) if value]
    if failures:
        raise RuntimeError(f"run contract failure: {failures}")
    return Session(spec, context, data, runtime, optimizer, scheduler, parameters,
                   output, contract, args.mode, args.microbatch, args.frame_chunk)


def restore(session: Session, checkpoint: Path) -> tuple[int, int]:
    restored, result, error = {}, None, None
    try:
        updates_target = session.spec["execution"]["updates_per_mode"]
        continuation = updates_target in (CONTINUATION_UPDATES, CONTINUATION1350_UPDATES,
                                          CONTINUATION1800_UPDATES, PILOT_UPDATES,
                                          CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES)
        updates, rows = load_ecp_checkpoint(
            checkpoint=checkpoint, stage=STAGE, context=session.context,
            model=session.runtime.writer, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA,
            restored_state=restored, allow_world_size_change=(continuation or session.mode == change_clock.MODE))
        migration = session.data.restore(restored["sampler_state"],
                                         migrate_sealed_270=updates_target == CONTINUATION_UPDATES and updates == 270,
                                         migrate_continuation_900=updates_target == CONTINUATION1350_UPDATES and updates == 900,
                                         migrate_continuation_1350=updates_target == CONTINUATION1800_UPDATES and updates == 1350,
                                         migrate_continuation_1800=updates_target == PILOT_UPDATES and updates == 1800,
                                         migrate_pilot_1890=updates_target == CONTINUATION2340_UPDATES and updates == 1890,
                                         migrate_continuation_2340=updates_target == CONTINUATION2790_UPDATES and updates == 2340)
        valid_checkpoints = ((2340, *CONTINUATION2790_CHECKPOINTS)
                             if updates_target == CONTINUATION2790_UPDATES else
                             (1890, *CONTINUATION2340_CHECKPOINTS)
                             if updates_target == CONTINUATION2340_UPDATES else
                             (1800, *PILOT_CHECKPOINTS) if updates_target == PILOT_UPDATES else
                             (1350, *CONTINUATION1800_CHECKPOINTS)
                             if updates_target == CONTINUATION1800_UPDATES else
                             (900, *CONTINUATION1350_CHECKPOINTS)
                             if updates_target == CONTINUATION1350_UPDATES else
                             (270, *CONTINUATION_CHECKPOINTS) if continuation else CHECKPOINTS)
        if (updates not in valid_checkpoints or rows != updates or session.scheduler.last_epoch != updates
                or restored["training_state"] != (
                    {"updates": updates, "mode": session.mode,
                     "pilot_arm": "control", "loss_variant": "full"}
                    if updates_target == CONTINUATION2340_UPDATES and updates == 1890 else
                    {"updates": updates, "mode": session.mode, "loss_variant": "full"}
                    if updates_target in (CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES) else
                    {"updates": updates, "mode": session.mode}
                    if updates_target != PILOT_UPDATES or updates == 1800 else
                    {"updates": updates, "mode": session.mode,
                     "pilot_arm": session.contract["pilot_arm"],
                     "loss_variant": session.contract["loss_variant"]})):
            raise ValueError("formal ECP optimizer/scheduler/sampler cursor changed")
        rows_from_parent = (checkpoint.parent.parent / "metrics.jsonl").read_text().splitlines()
        prefix = rows_from_parent[:rows]
        if (len(prefix) != rows or [json.loads(row)["update"] for row in prefix] != list(range(1, rows + 1))
                or any(json.loads(row)["mode"] != session.mode for row in prefix)):
            raise ValueError("ECP metrics history lacks a complete consecutive same-arm prefix")
        if session.context.is_main:
            (session.output / "metrics.jsonl").write_text("\n".join(prefix) + "\n")
            if continuation or session.mode == change_clock.MODE:
                parent_world = int(read_json(checkpoint / "checkpoint_manifest.json")["world_size"])
                write_json_atomic(session.output / "resume_provenance.json", {
                    "checkpoint": str(checkpoint), "parent_git": read_json(
                        checkpoint.parent.parent / "run_contract.json")["git"],
                    "old_world_size": parent_world, "new_world_size": session.context.world_size,
                    "restored_rank_rng": list(range(min(parent_world, session.context.world_size))),
                    "fresh_rank_rng": {str(rank): {"seed_function": "seed_everything(7, context)",
                                                   "rank_seed": 7 + rank}
                                       for rank in range(parent_world, session.context.world_size)},
                    "sampler_migration": migration, "metrics_prefix_rows": rows,
                    "topology": session.contract["topology"]})
        result = updates, rows
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, session.context.world_size) if value]
    if failures:
        raise RuntimeError(f"formal ECP restoration failed on a rank: {failures}")
    return result


def update(session: Session, updates: int, rows: int) -> tuple[int, int]:
    started = time.perf_counter()
    jobs = [session.data.event(updates, task) for task in session.data.tasks_for_step(updates)]
    costs = {index: session.data.videos.frame_counts(job["task"], job["teacher_demo"])[1]
             for index, job in enumerate(jobs)}
    world = session.context.world_size
    assigned = condition_assignment(tuple(costs), costs, world_size=world)
    session.optimizer.zero_grad(set_to_none=True)
    local, error = [], None
    try:
        for index in assigned[session.context.rank]:
            local.append(one_job(session.runtime, session.data, jobs[index],
                                 session.microbatch, session.frame_chunk,
                                 session.contract.get("loss_variant", "full")))
    except Exception:
        error = traceback.format_exc()
    failures = [value for value in gather(error, world) if value]
    if failures:
        raise RuntimeError(f"operator macro job failed on a rank: {failures}")
    sum_writer_gradients(session.parameters, world_size=world)
    gradients = gradient_groups(session.runtime.writer)
    norm = float(torch.nn.utils.clip_grad_norm_(
        session.parameters, session.spec["optimization"]["grad_clip"], error_if_nonfinite=True))
    lr = session.optimizer.param_groups[0]["lr"]
    session.optimizer.step()
    session.scheduler.step()
    torch.cuda.synchronize(session.context.device)
    updates += 1
    session.data.next_step = updates
    packets = gather(local, world)
    memory = gather({"rank": session.context.rank,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated(session.context.device) / 2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved(session.context.device) / 2**30}, world)
    if session.context.is_main:
        append_jsonl(session.output / "metrics.jsonl", {
            "update": updates, "mode": session.mode, "queries": 112,
            **({"loss_variant": "full"} if session.data.updates in (
                CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES) else {}),
            **({"pilot_arm": session.contract["pilot_arm"],
                "loss_variant": session.contract["loss_variant"]}
               if "pilot_arm" in session.contract else {}),
            "jobs": [row for packet in packets for row in packet],
            "lr_applied": lr, "lr_next": session.scheduler.get_last_lr()[0],
            "grad_norms_before_clip": gradients, "total_grad_norm": norm,
            "rank_memory": memory, "seconds": time.perf_counter() - started})
    rows += 1
    if updates in session.data.checkpoints:
        save_ecp_checkpoint(
            output_dir=session.output, macro=updates, stage=STAGE, context=session.context,
            model=session.runtime.writer, optimizer=session.optimizer,
            scheduler=session.scheduler, run_contract_schema=SCHEMA, metrics_rows=rows,
            sampler_state=session.data.sampler_state(),
            training_state={"updates": updates, "mode": session.mode,
                            **({"loss_variant": "full"} if session.data.updates in (
                                CONTINUATION2340_UPDATES, CONTINUATION2790_UPDATES) else {}),
                            **({"pilot_arm": session.contract["pilot_arm"],
                                "loss_variant": session.contract["loss_variant"]}
                               if "pilot_arm" in session.contract else {})})
    return updates, rows


def validate_train_request(spec: dict, args) -> None:
    if spec["task"] == change_clock.TASK:
        if (args.mode != change_clock.MODE or not args.attempt
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.attempt) is None
                or getattr(args, "pilot_arm", None) is not None
                or args.microbatch not in (28, 14, 7) or args.frame_chunk not in (8, 4)
                or args.stop_after_macro not in (None, *CHECKPOINTS[:-1])
                or (args.resume is None) != (args.attempt == "fresh")):
            raise ValueError("change-clock learning requires fresh or same-arm ECP resume")
        return
    if (spec["execution"]["updates_per_mode"] != CONTINUATION2790_UPDATES
            or spec["events"] != CONTINUATION2790_EVENTS
            or spec.get("continuation", {}).get("parent_arm") != "T"
            or spec.get("pilot") is not None):
        raise ValueError("new formal operator training requires the T2790 continuation spec")
    if (args.mode != "T" or not args.attempt
            or getattr(args, "pilot_arm", None) is not None
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.attempt) is None
            or args.resume is None or args.attempt == "fresh"
            or args.microbatch not in (28, 14, 7) or args.frame_chunk not in (8, 4)):
        raise ValueError("operator train requires registered T attempt and ECP policy")
    if (args.stop_after_macro is not None
            and args.stop_after_macro not in CONTINUATION2790_CHECKPOINTS[:-1]):
        raise ValueError("controlled stop must be a complete intermediate 2790 ECP boundary")


def train(spec: dict, args) -> None:
    validate_train_request(spec, args)
    session = prepare_train(spec, args)
    try:
        updates, rows = restore(session, args.resume) if args.resume else (0, 0)
        start_updates = updates
        started = time.perf_counter()
        target = session.data.updates
        if updates >= target:
            raise ValueError("completed operator checkpoint cannot start a new training attempt")
        if args.stop_after_macro is not None and args.stop_after_macro <= updates:
            raise ValueError("controlled stop is not after the resumed ECP")
        while updates < target:
            updates, rows = update(session, updates, rows)
            if updates in spec["execution"]["checkpoints"][:-1]:
                requested = ((session.output / "stop_at_next_ecp.request").exists()
                             if session.context.is_main else False)
                stop_at_ecp = any(gather(requested or updates == args.stop_after_macro,
                                         session.context.world_size))
                if stop_at_ecp:
                    if session.context.is_main:
                        write_json_atomic(session.output / "stopped_at_ecp.json", {
                            "schema_version": SCHEMA, "mode": session.mode, "updates": updates,
                            "metrics_rows": rows,
                            "checkpoint": str(session.output / "checkpoints"
                                              / f"macro_{updates:08d}"),
                            "reason": ("registered stop-after-macro" if updates == args.stop_after_macro
                                       else "stop_at_next_ecp.request"),
                            "next_resume_from_this_ecp": True})
                    return
        if session.context.is_main:
            write_json_atomic(session.output / "completion.json", {
                "schema_version": SCHEMA, "mode": session.mode, "updates": updates,
                **({"pilot_arm": session.contract["pilot_arm"],
                    "loss_variant": session.contract["loss_variant"]}
                   if "pilot_arm" in session.contract else {}),
                "actual_segment_updates": updates - start_updates,
                "actual_segment_queries": (updates - start_updates) * 112,
                "metrics_rows": rows, "seconds": time.perf_counter() - started,
                "resumed_from": str(args.resume) if args.resume else None,
                "checkpoint": str(session.output / "checkpoints" / f"macro_{target:08d}"),
                "formal_checkpoint_complete": True,
                "scientific_qualification": False})
    finally:
        session.data.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def audit(spec: dict, asset_root: Path) -> dict:
    data = FormalData(asset_root, spec, query_labels=False)
    try:
        visits = {task: [] for task in TASKS}
        query_total = 0
        for step in range(data.updates):
            jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
            if len(jobs) != 4 or len({row["task"] for row in jobs}) != 4:
                raise ValueError("macro lost equal four-task allocation")
            for row in jobs:
                if (len(row["queries"]) != 28 or any(q["demo"] == row["teacher_demo"]
                                                     for q in row["queries"])
                        or len({q["demo"] for q in row["queries"]}) != 28
                        or any(not 0 <= q["frame"] < data.tasks[row["task"]].episode_lengths[q["demo"]] - 1
                               for q in row["queries"])):
                    raise ValueError("event lost cross-episode FM")
                visits[row["task"]].append(row["teacher_demo"])
                query_total += 28
        count = data.updates // 9
        expected = {task: [] for task in TASKS}
        for task in TASKS:
            for round_index in range((count + 49) // 50):
                seed = ([20260928, 1, task] if round_index == 0 else
                        [20260928, 1, task, round_index])
                length = min(50, count - 50 * round_index)
                expected[task] += list(np.random.default_rng(
                    np.random.SeedSequence(seed)).permutation(50)[:length])
        if visits != expected or query_total != data.updates * 112:
            raise ValueError("36-task teacher rounds or cross-episode queries changed")
        return {"schema_version": spec["events"]["schema_version"],
                "modes": spec["execution"]["modes"], "updates_per_mode": data.updates,
                "conditions_per_mode": sum(len(rows) for rows in visits.values()),
                "queries_per_mode": query_total, "visits_per_task": count,
                "teacher_order": visits, "query_offset": 1}
    finally:
        data.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "train"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("T", "U", change_clock.MODE))
    parser.add_argument("--pilot-arm", choices=tuple(PILOT_ARMS))
    parser.add_argument("--attempt", type=str)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--frame-chunk", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--stop-after-macro", type=int)
    parser.add_argument("--spec", type=Path, default=CONTINUATION2790_SPEC_PATH)
    args = parser.parse_args()
    spec = specification(args.spec)
    if args.phase == "train":
        train(spec, args)
    else:
        print(json.dumps(audit(spec, args.asset_root), sort_keys=True))


if __name__ == "__main__":
    main()
