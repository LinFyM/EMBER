"""Configuration and ownership contract for direct-functional EBSRI S2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.joint_program_primal.runtime import (
    R5_SHARED_FUNCTIONAL_CHART,
    SCORER_INTERACTION_ONLY,
)
from ember.pi05_source_checkpoint import read_json


BANK_SET_SHARED_SCHEMA = "ember_ecp_event_bank_set_shared_direct_functional_v1"
BANK_SET_SHARED_RUN_SCHEMA = (
    "ember_ecp_event_bank_set_shared_direct_functional_run_v1"
)
BANK_SET_SHARED_STAGE = "g3_event_bank_set_s2_fixed_route_shared_loto"
BANK_SET_SHARED_STATUS = "preregistered_fixed_route_shared_direct_functional"
BANK_SET_SHARED_TRAIN_PANEL = "panel_a_only_gradient_tasks_exact_cross_episode_task_cycle"
BANK_SET_SHARED_LOSS = "exact_panel_a_functional_vjp_raw_correct_wrong_neutralization_hinge_then_panel_b_gate"

BANK_SET_SHARED_GRADIENT_META_TASKS = (8, 9, 32, 52)
BANK_SET_SHARED_GRADIENT_TARGET_TASKS = (72, 73, 75, 94)
BANK_SET_SHARED_GRADIENT_TASKS = (*BANK_SET_SHARED_GRADIENT_META_TASKS, *BANK_SET_SHARED_GRADIENT_TARGET_TASKS)
BANK_SET_SHARED_HELD_META_TASKS = (1,)
BANK_SET_SHARED_HELD_TARGET_TASKS = (93,)
BANK_SET_SHARED_HELD_TASKS = (*BANK_SET_SHARED_HELD_META_TASKS, *BANK_SET_SHARED_HELD_TARGET_TASKS)
BANK_SET_SHARED_TASKS = tuple(sorted((*BANK_SET_SHARED_GRADIENT_TASKS, *BANK_SET_SHARED_HELD_TASKS)))
BANK_SET_SHARED_FORBIDDEN_TASKS = (2, 74)
BANK_SET_SHARED_ARMS = (
    "correct_fit0",
    "correct_fit1",
    "correct_held",
    "wrong_fit0",
    "wrong_fit1",
)
BANK_SET_SHARED_ARM_SCHEDULE = (
    "wrong_fit0",
    "correct_fit0",
    "wrong_fit0",
    "correct_fit1",
)
BANK_SET_SHARED_WRONG_TASK_BY_TASK = {
    8: 9,
    9: 32,
    32: 52,
    52: 8,
    72: 73,
    73: 75,
    75: 94,
    94: 72,
}
BANK_SET_SHARED_EVALUATION_WRONG_TASK_BY_TASK = {1: 8, 93: 94}
BANK_SET_SHARED_DIRECT_FUNCTIONAL = {
    "panel": "a",
    "correct_objective": "generated_raw_flow_loss",
    "correct_backward_mass": 1.0,
    "wrong_objective": "max(carrier_raw_flow_loss-generated_raw_flow_loss,0)",
    "wrong_backward_mass": 0.5,
    "inactive_wrong_leaf_gradients": "explicit_zero",
    "panel_visit_schedule": "task_appearance_cursor_floor_div_4_mod_16",
    "memory_schedule": "no_grad_bank_leaf_vjp_cpu_offload_fresh_bank_replay",
    "task_gradient_combiner": "raw_equal_task_sum_no_normalization_or_mgda",
}
BANK_SET_SHARED_TARGET_CACHE_TEACHER = {
    "updates": 1,
    "learning_rate": 0.02,
    "panel_a_visit": 0,
    "gradient_clip_norm": 1.0,
    "target_authority": (
        "wrong_fit0_one_round_functional_free_delta_suppressive_teacher"
    ),
}


@dataclass(frozen=True)
class FunctionalArmObjective:
    kind: str
    value: float
    benefit_over_carrier: float
    backward_mass: float
    gradient_active: bool


def _task_profile(
    replay: int, group_batch: int, policy_batch: int, frames: tuple[int, int, int]
) -> dict[str, Any]:
    return {
        "replay_frame_chunk_size": replay,
        "interaction_group_batch_size": group_batch,
        "functional_policy_microbatch_size": policy_batch,
        "correct_arm_sampled_frames": {
            "correct_fit0": frames[0],
            "correct_fit1": frames[1],
            "correct_held": frames[2],
        },
    }


# Frozen-cache frame counts are pre-profiled engineering authority.  They are
# used only for queue cost and the one allowed pre-formal profile revision.
BANK_SET_SHARED_TASK_PROFILES: dict[int, dict[str, Any]] = {
    1: _task_profile(4, 16, 8, (51, 44, 43)),
    8: _task_profile(16, 8, 2, (15, 15, 18)),
    9: _task_profile(16, 8, 2, (21, 24, 35)),
    32: _task_profile(16, 8, 2, (37, 44, 35)),
    52: _task_profile(16, 8, 2, (20, 18, 18)),
    72: _task_profile(16, 8, 2, (22, 23, 20)),
    73: _task_profile(16, 8, 2, (30, 30, 29)),
    75: _task_profile(16, 8, 2, (26, 26, 27)),
    93: _task_profile(32, 4, 2, (79, 87, 92)),
    94: _task_profile(32, 4, 2, (67, 52, 51)),
}


class SharedInteractionWriterState(torch.nn.Module):
    """Checkpoint only fresh shared interaction state and exact-resume cursors."""

    def __init__(self, bank_set_interaction: torch.nn.Module) -> None:
        super().__init__()
        self.bank_set_interaction = bank_set_interaction
        reference = next(bank_set_interaction.parameters())
        self.register_buffer(
            "task_arm_cursors",
            torch.zeros(
                len(BANK_SET_SHARED_GRADIENT_TASKS),
                dtype=torch.int64,
                device=reference.device,
            ),
            persistent=True,
        )


def _string_keyed(value: Mapping[int, Any]) -> dict[str, Any]:
    return {str(key): row for key, row in value.items()}


def task_cursor_counts(global_macro: int) -> tuple[int, ...]:
    """Return role-equal 3-of-4 task appearances after ``global_macro``."""

    if global_macro < 0:
        raise ValueError("S2 global macro cannot be negative")
    counts = [0] * len(BANK_SET_SHARED_GRADIENT_TASKS)
    for macro in range(global_macro):
        start = macro % 4
        for role_start in (0, 4):
            for offset in range(3):
                counts[role_start + (start + offset) % 4] += 1
    return tuple(counts)


def task_panel_a_visit(task_cursor: int, panel_visits: int) -> int:
    """Cycle Panel-A after one task completes its four-arm schedule."""

    if task_cursor < 0 or panel_visits <= 0:
        raise ValueError("S2 task cursor or Panel-A visit count is invalid")
    return (task_cursor // len(BANK_SET_SHARED_ARM_SCHEDULE)) % panel_visits


def functional_arm_objective(
    arm_name: str,
    *,
    generated_loss: float,
    carrier_loss: float,
    correct_backward_mass: float,
    wrong_backward_mass: float,
) -> FunctionalArmObjective:
    """Return the raw-unit objective and signed Writer backward mass."""

    if min(correct_backward_mass, wrong_backward_mass) <= 0:
        raise ValueError("S2 direct-functional backward mass must be positive")
    benefit = float(carrier_loss) - float(generated_loss)
    if arm_name in {"correct_fit0", "correct_fit1"}:
        return FunctionalArmObjective(
            "raw_flow_loss", float(generated_loss), benefit,
            float(correct_backward_mass), True,
        )
    if arm_name != "wrong_fit0":
        raise ValueError("S2 direct-functional objective received a zero-gradient arm")
    return FunctionalArmObjective(
        "raw_unit_neutralization_hinge", max(benefit, 0.0), benefit,
        -float(wrong_backward_mass), benefit > 0.0,
    )


def is_bank_set_shared_config(config: Mapping[str, Any]) -> bool:
    """Return whether ``config`` claims the sealed S2 schema."""

    return config.get("schema_version") == BANK_SET_SHARED_SCHEMA


def _config_valid(config: Mapping[str, Any]) -> bool:
    shared = config.get("shared_training", {})
    model = config.get("model", {})
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    optimizer = optimization.get("joint", {}).get("optimizer", {})
    evaluation = config.get("evaluation", {})
    gate = config.get("gate", {})
    wall = config.get("information_wall", {})
    split = config.get("task_split", {})
    cache = config.get("frozen_condition_cache_authority", {})
    s1_gate = config.get("authorities", {}).get("required_s1_gate", {})
    profiles = shared.get("task_profiles", {})
    return all(
        (
            is_bank_set_shared_config(config),
            config.get("stage") == BANK_SET_SHARED_STAGE,
            config.get("status") == BANK_SET_SHARED_STATUS,
            cache.get("config_schema") == "ember_ecp_joint_program_primal_j2_v1",
            cache.get("config_bytes") == 6017,
            s1_gate.get("aggregate_schema")
            == "ember_ecp_event_bank_set_tasklocal_aggregate_v1",
            s1_gate.get("required_gate") == "pass",
            isinstance(s1_gate.get("path"), str),
            split.get("gradient_meta")
            == list(BANK_SET_SHARED_GRADIENT_META_TASKS),
            split.get("gradient_target")
            == list(BANK_SET_SHARED_GRADIENT_TARGET_TASKS),
            split.get("held_interaction_meta")
            == list(BANK_SET_SHARED_HELD_META_TASKS),
            split.get("held_interaction_target")
            == list(BANK_SET_SHARED_HELD_TARGET_TASKS),
            split.get("selection_uses_outcomes") is False,
            shared.get("gradient_meta_task_ids")
            == list(BANK_SET_SHARED_GRADIENT_META_TASKS),
            shared.get("gradient_target_task_ids")
            == list(BANK_SET_SHARED_GRADIENT_TARGET_TASKS),
            shared.get("gradient_task_ids") == list(BANK_SET_SHARED_GRADIENT_TASKS),
            shared.get("held_interaction_meta_task_ids")
            == list(BANK_SET_SHARED_HELD_META_TASKS),
            shared.get("held_interaction_target_task_ids")
            == list(BANK_SET_SHARED_HELD_TARGET_TASKS),
            shared.get("held_interaction_task_ids")
            == list(BANK_SET_SHARED_HELD_TASKS),
            shared.get("wrong_task_by_task")
            == _string_keyed(BANK_SET_SHARED_WRONG_TASK_BY_TASK),
            shared.get("evaluation_wrong_task_by_task")
            == _string_keyed(BANK_SET_SHARED_EVALUATION_WRONG_TASK_BY_TASK),
            shared.get("arm_schedule") == list(BANK_SET_SHARED_ARM_SCHEDULE),
            shared.get("gradient_arms") == [
                "correct_fit0",
                "correct_fit1",
                "wrong_fit0",
            ],
            shared.get("zero_gradient_arms")
            == ["correct_held", "wrong_fit1", "panel_b"],
            profiles == _string_keyed(BANK_SET_SHARED_TASK_PROFILES),
            shared.get("task_weighting")
            == "role_equal_then_task_equal_global_optimizer_step",
            model.get("program_source")
            == "fixed_nontrainable_128d_orthogonal_task_token",
            model.get("primal_scorer_initialization") == R5_SHARED_FUNCTIONAL_CHART,
            model.get("primal_scorer_trainable_partition")
            == SCORER_INTERACTION_ONLY,
            model.get("interaction_initialization")
            == "fresh_event_conditioned_bank_set_interaction_zero_correction",
            model.get("trainable") == ["EventConditionedBankSetInteraction"],
            model.get("generated_adapter")
            == "one_complete_38_target_rank12_plus_rank4_rank16",
            model.get("deployment_candidate") is False,
            data.get("functional_panel_train") == BANK_SET_SHARED_TRAIN_PANEL,
            data.get("video_action_cross_episode") is True,
            optimization.get("loss") == BANK_SET_SHARED_LOSS,
            optimization.get("result_or_action_gradient_calls_per_optimizer_step")
            == 6,
            optimization.get("direct_functional") == BANK_SET_SHARED_DIRECT_FUNCTIONAL,
            optimization.get("joint", {}).get("warmup_optimizer_steps") == 10,
            optimization.get("joint", {}).get("effective_optimizer_steps") == 100,
            optimization.get("joint", {}).get("checkpoint_effective_steps")
            == [60, 100],
            optimization.get("joint", {}).get("global_tasks_per_optimizer_step")
            == 6,
            optimizer.get("peak_lr") == 0.0001,
            optimizer.get("decay_lr") == 0.000001,
            evaluation.get("checkpoint_optimizer_steps") == [70, 110],
            evaluation.get("arms") == list(BANK_SET_SHARED_ARMS),
            evaluation.get("functional_panel") == "panel_b",
            evaluation.get("panel_visits") == 16,
            evaluation.get("job_unit") == "task_arm_checkpoint",
            evaluation.get("real_bank_lifetime") == "one_job_only_then_release",
            evaluation.get("target_cache")
            == "small_cpu_effective_targets_and_family_denominators_only",
            evaluation.get("target_cache_scope")
            == "diagnostics_and_gate_only_never_training",
            evaluation.get("target_cache_wrong_free_delta_teacher")
            == BANK_SET_SHARED_TARGET_CACHE_TEACHER,
            gate.get("correct_fit_median_minimum") == 0.85,
            gate.get("same_task_held_median_minimum") == 0.80,
            gate.get("wrong_median_maximum") == 0.25,
            gate.get("margin_median_minimum") == 0.50,
            gate.get("held_interaction_each_correct_better_than_wrong") is True,
            gate.get("held_to_gradient_correct_fit_minimum") == 0.85,
            gate.get("maximum_later_correct_fit_drop") == 0.05,
            gate.get("roles_required") == ["meta", "target"],
            wall.get("forbidden_task_ids") == list(BANK_SET_SHARED_FORBIDDEN_TASKS),
            wall.get("forbidden_task_reads") == 0,
            wall.get("held_interaction_task_backward_calls") == 0,
            wall.get("same_task_held_backward_calls") == 0,
            wall.get("wrong_fit1_backward_calls") == 0,
            wall.get("result_or_action_gradient_calls_per_optimizer_step") == 6,
            wall.get("result_or_action_gradient_scope")
            == "gradient_tasks_panel_a_only",
            wall.get("panel_b_backward_calls") == 0,
            wall.get("action_meta_installed") is False,
            wall.get("shuffled_or_reversed_use") is False,
            wall.get("single_complete_rank16") is True,
            config.get("privileged_critic") is None,
        )
    )


def load_bank_set_shared_config(path: Path) -> dict[str, Any]:
    """Load the sealed S2 config and reject any task/Gate authority drift."""

    config = read_json(path.resolve())
    if not _config_valid(config):
        raise ValueError("unsupported EBSRI S2 shared LOTO config")
    return config


def _world_topology(value: Any, world_size: int) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) != world_size
        or sorted(int(row.get("rank", -1)) for row in value)
        != list(range(world_size))
    ):
        raise ValueError("S2 training world topology changed")
    return [dict(row) for row in value]


def _optional_cursors(value: Any, step: int) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        rows = [int(value[str(task)]) for task in BANK_SET_SHARED_GRADIENT_TASKS]
    else:
        rows = list(map(int, value))
    if rows != list(task_cursor_counts(step)):
        raise ValueError("S2 checkpoint task cursors changed")
    return rows


def checkpoint_authority(
    *, config_path: Path, compiler_run: Path, checkpoint: Path
) -> dict[str, Any]:
    """Validate one adjacent formal checkpoint without loading its model tensors."""

    config_path = config_path.resolve()
    compiler_run = compiler_run.resolve()
    checkpoint = checkpoint.resolve()
    step = checkpoint_macro(checkpoint)
    config = load_bank_set_shared_config(config_path)
    if step not in set(map(int, config["evaluation"]["checkpoint_optimizer_steps"])):
        raise ValueError("S2 evaluator checkpoint step changed")
    contract = read_json(compiler_run / "run_contract.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    world_size = int(manifest.get("world_size", -1))
    topology = _world_topology(contract.get("world_topology"), world_size)
    if manifest.get("world_topology") is not None:
        if _world_topology(manifest["world_topology"], world_size) != topology:
            raise ValueError("S2 manifest and run topology disagree")
    _optional_cursors(manifest.get("task_cursors"), step)
    expected_files = {
        "ecp.safetensors", "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    files = manifest.get("files", {})
    if (
        checkpoint.parent.parent != compiler_run
        or contract.get("schema_version") != BANK_SET_SHARED_RUN_SCHEMA
        or contract.get("stage") != BANK_SET_SHARED_STAGE
        or contract.get("phase") != "shared_loto"
        or contract.get("mode") != "formal"
        or contract.get("config", {}).get("path") != str(config_path)
        or int(contract.get("config", {}).get("bytes", -1))
        != config_path.stat().st_size
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != BANK_SET_SHARED_STAGE
        or int(manifest.get("next_macro", -1)) != step
        or manifest.get("run_contract_schema") != BANK_SET_SHARED_RUN_SCHEMA
        or set(files) != expected_files
    ):
        raise ValueError("S2 shared checkpoint authority changed")
    for name, record in files.items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"S2 checkpoint file changed: {name}")
    with safe_open(checkpoint / "ecp.safetensors", framework="pt", device="cpu") as data:
        if "task_arm_cursors" not in data.keys():
            raise ValueError("S2 checkpoint omitted task-arm cursors")
        cursors = list(map(int, data.get_tensor("task_arm_cursors").tolist()))
    if cursors != list(task_cursor_counts(step)):
        raise ValueError("S2 checkpoint tensor cursors changed")
    return {
        "optimizer_step": step, "path": str(checkpoint),
        "training_commit": str(contract.get("git", {}).get("commit", "")),
        "world_size": world_size, "world_topology": topology,
        "task_cursors": cursors,
        "tensor_bytes": int(files["ecp.safetensors"]["bytes"]),
    }


def bank_set_shared_parameter_ownership(
    program: torch.nn.Module,
    compiler: torch.nn.Module,
) -> tuple[torch.nn.Module, tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    """Freeze R5/Program and expose only one fresh shared interaction module."""

    program.requires_grad_(False).eval()
    compiler.requires_grad_(False).eval()
    interaction = compiler.bank_set_interaction
    interaction.requires_grad_(True).train()
    writer = SharedInteractionWriterState(interaction)
    named = {
        name: parameter
        for name, parameter in writer.named_parameters()
        if parameter.requires_grad
    }
    allowed_roots = (
        "bank_set_interaction.set_encoder",
        "bank_set_interaction.input_candidate",
        "bank_set_interaction.output_candidate",
        "bank_set_interaction.input_condition",
        "bank_set_interaction.output_condition",
    )
    unexpected = sorted(
        name
        for name in named
        if not any(name.startswith(f"{root}.") for root in allowed_roots)
    )
    missing = [
        root
        for root in allowed_roots
        if not any(name.startswith(f"{root}.") for name in named)
    ]
    if unexpected or missing or not named:
        raise ValueError(
            "shared bank-set trainable inventory changed: "
            f"unexpected={unexpected}, missing={missing}"
        )
    trainable = tuple(named.values())
    frozen = tuple(
        parameter
        for root in (program, compiler)
        for parameter in root.parameters()
        if not parameter.requires_grad
    )
    if any(parameter.requires_grad for parameter in frozen):
        raise ValueError("shared bank-set frozen ownership changed")
    return writer, trainable, frozen


def writer_trainable_inventory(writer: torch.nn.Module) -> dict[str, Any]:
    """Describe the state that formal checkpoints must retain."""

    named = [
        (name, parameter)
        for name, parameter in writer.named_parameters()
        if parameter.requires_grad
    ]
    cursor = getattr(writer, "task_arm_cursors", None)
    if cursor is None or cursor.shape != (len(BANK_SET_SHARED_GRADIENT_TASKS),):
        raise ValueError("shared bank-set task-arm cursor inventory changed")
    return {
        "writer_trainable_parameter_names": [name for name, _ in named],
        "writer_trainable_parameter_count": sum(value.numel() for _, value in named),
        "persistent_buffer_names": ["task_arm_cursors"],
        "task_arm_cursor_task_order": list(BANK_SET_SHARED_GRADIENT_TASKS),
        "task_arm_cursors": [int(value) for value in cursor.detach().cpu().tolist()],
        "descriptor_authority": (
            "frozen_program_native_query_kappa_base_score_metadata_event_assignment"
        ),
        "fresh_interaction_shared_across_tasks": True,
    }
