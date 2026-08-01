"""Launch-contract fields owned by the AS-Writer update topology."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ember.pi05_source_checkpoint import DistributedContext


def _raw_full24_gradient_contract(
    *, context: DistributedContext, tasks_per_rank: int, global_tasks: int
) -> dict[str, Any]:
    distributed = context.world_size > 1
    return {
        "optimizer_gradient_accumulation": False,
        "loss_reduction": "mean_within_each_task_then_equal_mean_across_all_tasks",
        "task_gradient_collection": "six_rank_local_per_task_writer_gradients_without_ddp_backward",
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
    serial4 = (
        training["update_topology"]
        == "serial4_exposure_matched_six_phase_task_cycle"
    )
    topology = (
        {
            "optimizer_update_axis": "serial4_raw_mean_optimizer_update",
            "task_cycle_axis": "six_optimizer_updates_cover_full24_once",
            "optimizer_updates_per_task_cycle": 6,
            "total_task_cycles": total_steps // 6,
            "task_visit": "zero_based_task_cycle",
            "task_cycle_phase": "optimizer_update_modulo_6",
            "tasks_per_rank_per_optimizer_update": tasks_per_rank,
            "global_tasks_per_optimizer_update": global_tasks,
            "task_assignment": (
                "reuse_full24_cost_balanced_rank_rotation_per_task_cycle_then_"
                "select_one_long_first_rank_column_per_phase"
            ),
            **serial4_gradient_contract(
                context=context,
                tasks_per_rank=tasks_per_rank,
                global_tasks=global_tasks,
            ),
            "adamw_updates_per_optimizer_update": 1,
            "scheduler_updates_per_task_cycle": 1,
            "scheduler_update_cadence": (
                "after_phase5_only_lr_at_update_u_equals_full24_lr_floor_u_div_6"
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
    return {
        "world_size": context.world_size,
        "one_policy_cuda_process_per_rank": True,
        "extra_cuda_roles_on_any_rank": 0,
        "ddp_object": "rank_synchronized_shared_writer_without_ddp_backward",
        **topology,
        "task_video_cost_sha256": video_data["sampled_frame_cost_sha256"],
        "action_query_batch_size_per_task": batch_size,
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
