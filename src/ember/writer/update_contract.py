"""Launch-contract fields owned by the AS-Writer update topology."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ember.pi05_source_checkpoint import DistributedContext
from ember.writer.model import WriterModelError


def _raw_full24_gradient_contract(
    *, context: DistributedContext, tasks_per_rank: int, global_tasks: int
) -> dict[str, Any]:
    distributed = context.world_size > 1
    return {
        "optimizer_gradient_accumulation": False,
        "loss_reduction": "mean_within_each_task_then_equal_mean_across_all_tasks",
        "task_gradient_collection": (
            f"{tasks_per_rank}_rank_local_per_task_writer_gradients_without_"
            "ddp_backward"
        ),
        "task_gradients_per_rank_per_macro": tasks_per_rank,
        "global_task_gradients_per_macro": global_tasks,
        "distributed_full_task_gradient_matrix_materialized": False,
        "gradient_task_id_allgathers_per_macro": 1 if distributed else 0,
        "gradient_composition": "exact_raw_equal_weight_full24_mean_without_projection",
        "gradient_projection": "none",
        "gradient_gram_exchange": (
            "bounded_parameter_chunk_allgathers_with_per_chunk_cuda_completion_"
            "for_exact_raw_mean_full24_and_module_block_grams"
        ),
        "gradient_gram_chunk_elements": 1_048_576,
        "gradient_gram_chunk_allgathers_per_macro": (
            "runtime_enumerated_from_parameter_block_layout" if distributed else 0
        ),
        "gradient_gram_chunk_cuda_synchronizations_per_macro": (
            "one_per_runtime_enumerated_chunk_allgather" if distributed else 0
        ),
        "single_video_gradient_direction_sketch": (
            "fixed_countsketch_32_per_task_per_parameter_block"
        ),
        "diagnostic_tensor_allgathers_per_macro": 1 if distributed else 0,
        "ddp_no_sync_microtasks_per_macro": 0,
        "ddp_gradient_synchronizations_per_macro": 0,
    }


def serial4_gradient_contract(
    *, context: DistributedContext, tasks_per_rank: int, global_tasks: int
) -> dict[str, Any]:
    """Describe selected4 Grams without presenting 25% as model quality."""

    distributed = context.world_size > 1
    return {
        "optimizer_gradient_accumulation": False,
        "loss_reduction": (
            "mean_within_each_task_then_equal_raw_mean_across_selected4"
        ),
        "task_gradient_collection": (
            "one_rank_local_per_task_writer_gradient_without_ddp_backward"
        ),
        "task_gradients_per_rank_per_optimizer_update": tasks_per_rank,
        "global_task_gradients_per_optimizer_update": global_tasks,
        "distributed_selected_task_gradient_matrix_materialized": False,
        "gradient_task_id_allgathers_per_optimizer_update": (
            1 if distributed else 0
        ),
        "gradient_composition": (
            "exact_raw_equal_weight_selected4_mean_without_projection"
        ),
        "gradient_projection": "none",
        "gradient_gram_shape_per_optimizer_update": [global_tasks, global_tasks],
        "orthogonal_equal_norm_mean_to_task_energy_reference": 0.25,
        "energy_ratio_interpretation": (
            "selected4_manipulation_check_only_not_scientific_success"
        ),
        "gradient_gram_exchange": (
            "bounded_parameter_chunk_allgathers_with_per_chunk_cuda_completion_"
            "for_exact_raw_mean_selected4_and_module_block_4x4_grams"
        ),
        "gradient_gram_chunk_elements": 1_048_576,
        "gradient_gram_chunk_allgathers_per_optimizer_update": (
            "runtime_enumerated_from_parameter_block_layout" if distributed else 0
        ),
        "gradient_gram_chunk_cuda_synchronizations_per_optimizer_update": (
            "one_per_runtime_enumerated_chunk_allgather" if distributed else 0
        ),
        "single_video_gradient_direction_sketch": (
            "fixed_countsketch_32_per_task_per_parameter_block"
        ),
        "diagnostic_tensor_allgathers_per_optimizer_update": (
            1 if distributed else 0
        ),
        "ddp_no_sync_microtasks_per_optimizer_update": 0,
        "ddp_gradient_synchronizations_per_optimizer_update": 0,
    }


def _axis_key(serial4: bool, value: str) -> str:
    return f"{value}_per_optimizer_update" if serial4 else f"{value}_per_macro"


def checkpoint_state_family(config: Mapping[str, Any]) -> str:
    """Return the fail-closed optimizer/data/RNG checkpoint family."""

    training = config["conditioning_training"]
    topology = str(training["update_topology"])
    target_owned_factor = (
        config.get("writer", {}).get("architecture")
        == "pi05_target_owned_factor_program_v1"
    )
    task_query_keyed = (
        training.get("policy_randomness_scheme")
        == "task_query_keyed_stateless_policy_cpu_cuda_v2"
    )
    if topology == "task_complete_all_tasks":
        if target_owned_factor and task_query_keyed:
            return "target_owned_factor_task_query_keyed_rawfull24_v1"
        return (
            "cvadr_task_query_keyed_rawfull24_v2"
            if task_query_keyed
            else "cvadr_legacy_full24_v1"
        )
    if topology == "serial4_exposure_matched_six_phase_task_cycle":
        return "cvadr_legacy_serial4_v1"
    if (
        topology
        == "cycle_normalized_randomized_group4_six_phase_task_cycle"
        and task_query_keyed
    ):
        return "cvadr_cycle_normalized_randomized_group4_v2"
    raise WriterModelError("unsupported AS-Writer checkpoint state family")


def _update_topology_contract(
    config: Mapping[str, Any],
    context: DistributedContext,
    total_steps: int,
    tasks_per_rank: int,
    global_tasks: int,
) -> tuple[bool, dict[str, Any]]:
    training = config["conditioning_training"]
    update_topology = str(training["update_topology"])
    serial4 = update_topology in {
        "serial4_exposure_matched_six_phase_task_cycle",
        "cycle_normalized_randomized_group4_six_phase_task_cycle",
    }
    randomized_group4 = (
        update_topology
        == "cycle_normalized_randomized_group4_six_phase_task_cycle"
    )
    topology = (
        {
            "optimizer_update_axis": (
                "cycle_normalized_randomized_group4_raw_mean_optimizer_update"
                if randomized_group4
                else "serial4_raw_mean_optimizer_update"
            ),
            "task_cycle_axis": "six_optimizer_updates_cover_full24_once",
            "optimizer_updates_per_task_cycle": 6,
            "total_task_cycles": total_steps // 6,
            "task_visit": "zero_based_task_cycle",
            "task_cycle_phase": "optimizer_update_modulo_6",
            "tasks_per_rank_per_optimizer_update": tasks_per_rank,
            "global_tasks_per_optimizer_update": global_tasks,
            "task_assignment": (
                "randomized_latin_group4_without_video_cost_input_phase_"
                "balanced_over_six_cycle_superblocks_with_complementary_tail"
                if randomized_group4
                else (
                    "reuse_full24_cost_balanced_rank_rotation_per_task_cycle_then_"
                    "select_one_long_first_rank_column_per_phase"
                )
            ),
            **serial4_gradient_contract(
                context=context,
                tasks_per_rank=tasks_per_rank,
                global_tasks=global_tasks,
            ),
            "adamw_updates_per_optimizer_update": 1,
            "scheduler_updates_per_task_cycle": 1,
            "scheduler_update_cadence": (
                "after_phase5_only_with_logical_cycle_lr_and_physical_lr_div_6"
                if randomized_group4
                else (
                    "after_phase5_only_lr_at_update_u_equals_full24_lr_floor_u_div_6"
                )
            ),
            **(
                {
                    "optimizer_cycle_normalization": dict(
                        config["optimization"]["cycle_normalization"]
                    ),
                    "phase_cost_assignment_input": "none",
                    "rank_local_long_first": "single_task_trivial_order",
                }
                if randomized_group4
                else {}
            ),
            "checkpoint_axis": "completed_optimizer_update",
        }
        if serial4
        else {
            "macro_step_axis": "raw_mean_full_task_optimizer_update",
            "tasks_per_rank_per_optimizer_update": tasks_per_rank,
            "global_tasks_per_optimizer_update": global_tasks,
            "task_assignment": (
                "selected_video_frame_cost_balanced_groups_rotated_across_"
                "physical_ranks_longest_task_first_within_each_rank"
            ),
            **_raw_full24_gradient_contract(
                context=context,
                tasks_per_rank=tasks_per_rank,
                global_tasks=global_tasks,
            ),
            "adamw_updates_per_macro": 1,
        }
    )
    return serial4, topology


def _policy_microbatch_contract(
    config: Mapping[str, Any], batch_size: int
) -> dict[str, int]:
    microbatch_size = int(
        config["optimization"].get(
            "functional_policy_microbatch_size", batch_size
        )
    )
    return {
        "functional_policy_microbatch_size": microbatch_size,
        "physical_policy_forwards_per_task": (
            batch_size + microbatch_size - 1
        )
        // microbatch_size,
    }


def build_update_runtime_contract(
    *,
    config: Mapping[str, Any],
    context: DistributedContext,
    video_data: Mapping[str, Any],
    total_steps: int,
    stop_step: int,
    batch_size: int,
    batch_cycle: Sequence[int],
    checkpoint_steps: Sequence[int],
    num_workers: int,
    rank_topology: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training = config["conditioning_training"]
    tasks_per_rank = int(training["tasks_per_rank_per_optimizer_update"])
    global_tasks = context.world_size * tasks_per_rank
    videos_per_visit = int(training["teacher_videos_per_task_visit"])
    serial4, topology = _update_topology_contract(
        config,
        context,
        total_steps,
        tasks_per_rank,
        global_tasks,
    )
    return {
        "world_size": context.world_size,
        "one_policy_cuda_process_per_rank": True,
        "extra_cuda_roles_on_any_rank": 0,
        "process_group_initialization": (
            "out_of_band_all_rank_cuda_ready_rendezvous_then_nccl_"
            "before_first_distributed_collective"
        ),
        "nccl_transport": "bci_a40_shm_with_direct_p2p_disabled",
        "ddp_object": "rank_synchronized_shared_writer_without_ddp_backward",
        "checkpoint_state_family": checkpoint_state_family(config),
        **topology,
        "task_video_cost_sha256": video_data["sampled_frame_cost_sha256"],
        "action_query_batch_size_per_task": batch_size,
        **_policy_microbatch_contract(config, batch_size),
        _axis_key(serial4, "action_query_batch_size_per_rank"): (
            tasks_per_rank * batch_size
        ),
        "per_rank_unique_action_query_cycle": list(batch_cycle),
        "teacher_videos_per_task_visit": videos_per_visit,
        _axis_key(serial4, "writer_video_conditions_per_rank"): (
            tasks_per_rank * videos_per_visit
        ),
        "actions_per_video_condition": batch_size,
        "action_video_assignment": "all_actions_share_single_video_lora",
        _axis_key(serial4, "logical_pairs_per_rank"): tasks_per_rank * batch_size,
        _axis_key(serial4, "global_policy_samples"): global_tasks * batch_size,
        _axis_key(serial4, "local_policy_functional_forwards"): tasks_per_rank,
        _axis_key(serial4, "global_policy_functional_forwards"): global_tasks,
        _axis_key(serial4, "writer_conditions_per_rank"): (
            tasks_per_rank * videos_per_visit
        ),
        "total_steps": total_steps,
        "selected_stop_step": stop_step,
        "checkpoint_steps": list(checkpoint_steps),
        "num_workers_per_rank": num_workers,
        "rank_topology": list(rank_topology),
    }
