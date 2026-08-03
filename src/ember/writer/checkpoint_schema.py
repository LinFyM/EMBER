"""Fail-closed schema registry for incompatible AS-Writer checkpoint families."""

from __future__ import annotations

from ember.writer.model import WriterModelError


AS_WRITER_CHECKPOINT_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_full24_checkpoint_v1"
)
AS_WRITER_TRAINER_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_full24_trainer_state_v1"
)
AS_WRITER_RANK_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_full24_rank_state_v1"
)
AS_WRITER_SERIAL4_CHECKPOINT_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_serial4_exposurematched_checkpoint_v1"
)
AS_WRITER_SERIAL4_TRAINER_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_serial4_exposurematched_trainer_state_v1"
)
AS_WRITER_SERIAL4_RANK_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_serial4_exposurematched_rank_state_v1"
)
AS_WRITER_TASK_QUERY_RAW_CHECKPOINT_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_task_query_rawfull24_checkpoint_v2"
)
AS_WRITER_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_task_query_rawfull24_trainer_state_v2"
)
AS_WRITER_TASK_QUERY_RAW_RANK_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_task_query_rawfull24_rank_state_v2"
)
AS_WRITER_CYCLE_NORMALIZED_GROUP4_CHECKPOINT_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_cycle_normalized_group4_checkpoint_v2"
)
AS_WRITER_CYCLE_NORMALIZED_GROUP4_TRAINER_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_cycle_normalized_group4_trainer_state_v2"
)
AS_WRITER_CYCLE_NORMALIZED_GROUP4_RANK_STATE_SCHEMA = (
    "ember_pi05_contextual_value_dual_read_cycle_normalized_group4_rank_state_v2"
)
TARGET_BOUND_ROLE_RAW_CHECKPOINT_SCHEMA = (
    "ember_pi05_target_bound_role_program_rawfull24_checkpoint_v1"
)
TARGET_BOUND_ROLE_RAW_TRAINER_STATE_SCHEMA = (
    "ember_pi05_target_bound_role_program_rawfull24_trainer_state_v1"
)
TARGET_BOUND_ROLE_RAW_RANK_STATE_SCHEMA = (
    "ember_pi05_target_bound_role_program_rawfull24_rank_state_v1"
)
TARGET_BOUND_ROLE_TASK_QUERY_RAW_CHECKPOINT_SCHEMA = (
    "ember_pi05_target_bound_role_program_task_query_rawfull24_checkpoint_v1"
)
TARGET_BOUND_ROLE_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA = (
    "ember_pi05_target_bound_role_program_task_query_rawfull24_trainer_state_v1"
)
TARGET_BOUND_ROLE_TASK_QUERY_RAW_RANK_STATE_SCHEMA = (
    "ember_pi05_target_bound_role_program_task_query_rawfull24_rank_state_v1"
)
TARGET_BOUND_ROLE_GROUP4_CHECKPOINT_SCHEMA = (
    "ember_pi05_target_bound_role_program_cycle_normalized_group4_checkpoint_v1"
)
TARGET_BOUND_ROLE_GROUP4_TRAINER_STATE_SCHEMA = (
    "ember_pi05_target_bound_role_program_cycle_normalized_group4_trainer_state_v1"
)
TARGET_BOUND_ROLE_GROUP4_RANK_STATE_SCHEMA = (
    "ember_pi05_target_bound_role_program_cycle_normalized_group4_rank_state_v1"
)
SEMANTIC_FACTOR_BASIS_RAW_CHECKPOINT_SCHEMA = (
    "ember_pi05_semantic_factor_basis_rawfull24_checkpoint_v1"
)
SEMANTIC_FACTOR_BASIS_RAW_TRAINER_STATE_SCHEMA = (
    "ember_pi05_semantic_factor_basis_rawfull24_trainer_state_v1"
)
SEMANTIC_FACTOR_BASIS_RAW_RANK_STATE_SCHEMA = (
    "ember_pi05_semantic_factor_basis_rawfull24_rank_state_v1"
)
SEMANTIC_FACTOR_BASIS_TASK_QUERY_RAW_CHECKPOINT_SCHEMA = (
    "ember_pi05_semantic_factor_basis_task_query_rawfull24_checkpoint_v1"
)
SEMANTIC_FACTOR_BASIS_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA = (
    "ember_pi05_semantic_factor_basis_task_query_rawfull24_trainer_state_v1"
)
SEMANTIC_FACTOR_BASIS_TASK_QUERY_RAW_RANK_STATE_SCHEMA = (
    "ember_pi05_semantic_factor_basis_task_query_rawfull24_rank_state_v1"
)
SEMANTIC_DIRECTION_STORE_TASK_QUERY_RAW_CHECKPOINT_SCHEMA = (
    "ember_pi05_semantic_direction_store_task_query_rawfull24_checkpoint_v1"
)
SEMANTIC_DIRECTION_STORE_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA = (
    "ember_pi05_semantic_direction_store_task_query_rawfull24_trainer_state_v1"
)
SEMANTIC_DIRECTION_STORE_TASK_QUERY_RAW_RANK_STATE_SCHEMA = (
    "ember_pi05_semantic_direction_store_task_query_rawfull24_rank_state_v1"
)


_FAMILIES = {
    "cvadr_legacy_full24_v1": (
        1,
        AS_WRITER_CHECKPOINT_SCHEMA,
        AS_WRITER_TRAINER_STATE_SCHEMA,
        AS_WRITER_RANK_STATE_SCHEMA,
    ),
    "cvadr_legacy_serial4_v1": (
        6,
        AS_WRITER_SERIAL4_CHECKPOINT_SCHEMA,
        AS_WRITER_SERIAL4_TRAINER_STATE_SCHEMA,
        AS_WRITER_SERIAL4_RANK_STATE_SCHEMA,
    ),
    "cvadr_task_query_keyed_rawfull24_v2": (
        1,
        AS_WRITER_TASK_QUERY_RAW_CHECKPOINT_SCHEMA,
        AS_WRITER_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA,
        AS_WRITER_TASK_QUERY_RAW_RANK_STATE_SCHEMA,
    ),
    "cvadr_cycle_normalized_randomized_group4_v2": (
        6,
        AS_WRITER_CYCLE_NORMALIZED_GROUP4_CHECKPOINT_SCHEMA,
        AS_WRITER_CYCLE_NORMALIZED_GROUP4_TRAINER_STATE_SCHEMA,
        AS_WRITER_CYCLE_NORMALIZED_GROUP4_RANK_STATE_SCHEMA,
    ),
    "target_bound_role_rawfull24_v1": (
        1,
        TARGET_BOUND_ROLE_RAW_CHECKPOINT_SCHEMA,
        TARGET_BOUND_ROLE_RAW_TRAINER_STATE_SCHEMA,
        TARGET_BOUND_ROLE_RAW_RANK_STATE_SCHEMA,
    ),
    "target_bound_role_task_query_keyed_rawfull24_v1": (
        1,
        TARGET_BOUND_ROLE_TASK_QUERY_RAW_CHECKPOINT_SCHEMA,
        TARGET_BOUND_ROLE_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA,
        TARGET_BOUND_ROLE_TASK_QUERY_RAW_RANK_STATE_SCHEMA,
    ),
    "target_bound_role_cycle_normalized_randomized_group4_v1": (
        6,
        TARGET_BOUND_ROLE_GROUP4_CHECKPOINT_SCHEMA,
        TARGET_BOUND_ROLE_GROUP4_TRAINER_STATE_SCHEMA,
        TARGET_BOUND_ROLE_GROUP4_RANK_STATE_SCHEMA,
    ),
    "semantic_factor_basis_rawfull24_v1": (
        1,
        SEMANTIC_FACTOR_BASIS_RAW_CHECKPOINT_SCHEMA,
        SEMANTIC_FACTOR_BASIS_RAW_TRAINER_STATE_SCHEMA,
        SEMANTIC_FACTOR_BASIS_RAW_RANK_STATE_SCHEMA,
    ),
    "semantic_factor_basis_task_query_keyed_rawfull24_v1": (
        1,
        SEMANTIC_FACTOR_BASIS_TASK_QUERY_RAW_CHECKPOINT_SCHEMA,
        SEMANTIC_FACTOR_BASIS_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA,
        SEMANTIC_FACTOR_BASIS_TASK_QUERY_RAW_RANK_STATE_SCHEMA,
    ),
    "semantic_direction_store_task_query_keyed_rawfull24_v1": (
        1,
        SEMANTIC_DIRECTION_STORE_TASK_QUERY_RAW_CHECKPOINT_SCHEMA,
        SEMANTIC_DIRECTION_STORE_TASK_QUERY_RAW_TRAINER_STATE_SCHEMA,
        SEMANTIC_DIRECTION_STORE_TASK_QUERY_RAW_RANK_STATE_SCHEMA,
    ),
}


def state_schemas(
    optimizer_updates_per_task_cycle: int,
    checkpoint_state_family: str | None = None,
) -> tuple[str, str, str]:
    family = checkpoint_state_family or (
        "cvadr_legacy_full24_v1"
        if optimizer_updates_per_task_cycle == 1
        else "cvadr_legacy_serial4_v1"
    )
    registered = _FAMILIES.get(family)
    if (
        registered is None
        or registered[0] != optimizer_updates_per_task_cycle
    ):
        raise WriterModelError("unsupported AS-Writer checkpoint task cycle")
    return registered[1:]
