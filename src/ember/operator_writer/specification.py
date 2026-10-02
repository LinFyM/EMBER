"""Sealed operator study specifications, separate from the shared training loop."""
from __future__ import annotations
from pathlib import Path
from ember.pi05_source_checkpoint import read_json
from .data import (CHECKPOINTS, CONTINUATION_CHECKPOINTS, CONTINUATION_UPDATES,
                   CONTINUATION1350_CHECKPOINTS, CONTINUATION1350_UPDATES,
                   CONTINUATION1800_CHECKPOINTS, CONTINUATION1800_UPDATES,
                   PILOT_CHECKPOINTS, PILOT_UPDATES,
                   CONTINUATION2340_CHECKPOINTS, CONTINUATION2340_UPDATES,
                   CONTINUATION2790_CHECKPOINTS, CONTINUATION2790_UPDATES, TASKS, UPDATES)
from . import change_clock, joint_training, support_diversity, prefix_change

REPO = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO / "configs/operator_read_write_v1/learning_spec.json"
CHANGE_CLOCK_SPEC_PATH = REPO / "configs/operator_read_write_v1" / change_clock.SPEC_NAME
CHANGE_CLOCK_CONTINUATION_SPEC_PATH = CHANGE_CLOCK_SPEC_PATH.with_name(change_clock.CONTINUATION_SPEC_NAME)
CONTINUATION_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation900_spec.json"
CONTINUATION1350_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation1350_spec.json"
CONTINUATION1800_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation1800_spec.json"
PILOT_SPEC_PATH = REPO / "configs/operator_read_write_v1/public_function_pilot_spec.json"
CONTINUATION2340_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation2340_spec.json"
CONTINUATION2790_SPEC_PATH = REPO / "configs/operator_read_write_v1/continuation2790_spec.json"
JOINT_SPEC_PATH = REPO / "configs/operator_read_write_v1" / joint_training.SPEC_NAME
CONTEXT_SPEC_PATH = JOINT_SPEC_PATH.with_name(joint_training.CONTEXT_SPEC_NAME)
CONTEXT_CONTINUATION_SPEC_PATH = JOINT_SPEC_PATH.with_name(joint_training.CONTEXT_CONTINUATION_SPEC_NAME)
SELF_READ_SPEC_PATH = JOINT_SPEC_PATH.with_name(joint_training.SELF_READ_SPEC_NAME)
CONDITIONAL_SPEC_PATH = JOINT_SPEC_PATH.with_name(joint_training.CONDITIONAL_SPEC_NAME)
CONDITIONAL_CONTINUATION_SPEC_PATH = JOINT_SPEC_PATH.with_name(joint_training.CONDITIONAL_CONTINUATION_SPEC_NAME)
SUPPORT_DIVERSITY_SPEC_PATH = JOINT_SPEC_PATH.with_name(support_diversity.SPEC_NAME)
PREFIX_CHANGE_SPEC_PATH = JOINT_SPEC_PATH.with_name(prefix_change.SPEC_NAME)
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
    if path == PREFIX_CHANGE_SPEC_PATH:
        spec = read_json(path)
        if spec != prefix_change.expected_spec(specification(JOINT_SPEC_PATH)):
            raise ValueError("native prefix change fresh450 contract changed")
        return spec
    if path == SUPPORT_DIVERSITY_SPEC_PATH:
        spec = read_json(path)
        if spec != support_diversity.expected_spec(specification(CONDITIONAL_SPEC_PATH)):
            raise ValueError("registered support diversity contract changed")
        return spec
    if path == CONDITIONAL_CONTINUATION_SPEC_PATH:
        spec = read_json(path)
        if spec != joint_training.expected_conditional_continuation_spec(specification(CONDITIONAL_SPEC_PATH)):
            raise ValueError("conditional read/write continuation900 contract changed")
        return spec
    if path == CONDITIONAL_SPEC_PATH:
        spec = read_json(path)
        if spec != joint_training.expected_conditional_spec(specification(SELF_READ_SPEC_PATH)):
            raise ValueError("conditional read/write fresh450 contract changed")
        return spec
    if path == SELF_READ_SPEC_PATH:
        spec = read_json(path)
        if spec != joint_training.expected_self_read_spec(specification(CONTEXT_SPEC_PATH)):
            raise ValueError("shared self-conditioned native fresh450 contract changed")
        return spec
    if path == CONTEXT_CONTINUATION_SPEC_PATH:
        spec = read_json(path)
        if spec != joint_training.expected_context_continuation_spec(specification(CONTEXT_SPEC_PATH)):
            raise ValueError("context Value continuation900 contract changed")
        return spec
    if path == CONTEXT_SPEC_PATH:
        spec = read_json(path)
        if spec != joint_training.expected_context_spec(specification(SPEC_PATH), CONTINUATION_EVENTS):
            raise ValueError("context Value fresh450 contract changed")
        return spec
    if path == JOINT_SPEC_PATH:
        spec = read_json(path)
        if spec != joint_training.expected_spec(specification(SPEC_PATH), CONTINUATION_EVENTS):
            raise ValueError("joint fresh450 contract changed")
        return spec
    if path == CHANGE_CLOCK_CONTINUATION_SPEC_PATH:
        spec = read_json(path)
        if spec != change_clock.expected_continuation_spec(
                specification(CHANGE_CLOCK_SPEC_PATH), CONTINUATION_EVENTS):
            raise ValueError("change-clock 450 continuation contract changed")
        return spec
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



def specification_path(spec: dict) -> Path:
    """The current consumer spec path; actual training provenance stays in its run."""
    paths = (PREFIX_CHANGE_SPEC_PATH, SUPPORT_DIVERSITY_SPEC_PATH, SPEC_PATH, CHANGE_CLOCK_SPEC_PATH, CHANGE_CLOCK_CONTINUATION_SPEC_PATH,
             CONTINUATION_SPEC_PATH, CONTINUATION1350_SPEC_PATH, CONTINUATION1800_SPEC_PATH,
             PILOT_SPEC_PATH, CONTINUATION2340_SPEC_PATH, CONTINUATION2790_SPEC_PATH, JOINT_SPEC_PATH, CONTEXT_SPEC_PATH, CONTEXT_CONTINUATION_SPEC_PATH, SELF_READ_SPEC_PATH, CONDITIONAL_SPEC_PATH, CONDITIONAL_CONTINUATION_SPEC_PATH)
    for path in paths:
        if read_json(path)["run_root"] == spec["run_root"]:
            return path
    raise ValueError("unregistered operator run root")
