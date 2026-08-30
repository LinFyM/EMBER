"""Runtime and launch contract for joint Program--primal functional training."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ember.ecp.bank_conditioning import primal_capacity
from ember.ecp.bank_conditioning.consensus import FitConsensusTeacherStore
from ember.ecp.bank_conditioning.frozen_condition_cache import (
    FROZEN_CONDITION_CACHE_SCHEMA,
    FrozenMappingConditionCache,
    frozen_condition_cache_authority,
)
from ember.ecp.bank_conditioning.mapping import (
    MappingCondition,
    SharedCompilerMappingSplit,
    load_mapping_split,
)
from ember.ecp.checkpoint import load_ecp_checkpoint
from ember.ecp.contracts import TargetOwner, build_target_owners
from ember.ecp.natural_program import NaturalProgramModel
from ember.ecp.natural_program_data import (
    NaturalProgramTask,
    load_natural_program_tasks,
)
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    SharedCompilerRankAssets,
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
    load_shared_scale_prior,
)
from ember.ecp.shared_compiler_native_teacher import NativeTeacherStore
from ember.ecp.stage0_training import (
    stage0_source_authority,
    tokenize_stage0_languages,
)
from ember.ecp.native_factors import native_capture_modes
from ember.ecp.joint_program_primal.routing_initialization import (
    FunctionalCodeTarget,
    R5_SHARED_FUNCTIONAL_CHART,
    R9_STABLE_CONTENT,
    load_functional_code_targets,
    load_passed_r5_primal_scorer,
    load_r9_stable_writer,
)
from ember.ecp.joint_program_primal.raw_stage0 import RAW_STAGE0_PROGRAM_INPUT
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_config,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[4]
J2_SCHEMA = "ember_ecp_joint_program_primal_j3_v1"
J2_RUN_SCHEMA = "ember_ecp_joint_program_primal_run_v2"
J2_STAGE = "j3_counterfactual_functional_routing"
CHART_RECONNECT_SCHEMA = "ember_ecp_natural_program_chart_reconnect_r6_v1"
CHART_RECONNECT_RUN_SCHEMA = "ember_ecp_natural_program_chart_reconnect_run_v1"
CHART_RECONNECT_STAGE = "g3_natural_program_functional_chart_reconnect"
FUNCTIONAL_CHART_ACQUISITION_SCHEMA = "ember_ecp_functional_code_chart_acquisition_r7_v1"
FUNCTIONAL_CHART_ACQUISITION_RUN_SCHEMA = "ember_ecp_functional_code_chart_acquisition_run_v1"
FUNCTIONAL_CHART_ACQUISITION_STAGE = "g3_fit_only_functional_code_chart_acquisition"
FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA = "ember_ecp_functional_code_stable_chart_joint_r9_v1"
FUNCTIONAL_CODE_STABLE_JOINT_RUN_SCHEMA = (
    "ember_ecp_functional_code_stable_chart_joint_run_v1"
)
FUNCTIONAL_CODE_STABLE_JOINT_STAGE = (
    "g3_fit_only_functional_code_stable_chart_joint_acquisition"
)
FUNCTIONAL_REFINEMENT_SCHEMA = "ember_ecp_r9_initialized_functional_refinement_r10_v1"
FUNCTIONAL_REFINEMENT_RUN_SCHEMA = "ember_ecp_r9_initialized_functional_refinement_run_v1"
FUNCTIONAL_REFINEMENT_STAGE = "g3_r9_initialized_functional_refinement"
RAW_STAGE0_SUFFICIENCY_SCHEMA = "ember_ecp_raw_stage0_sufficiency_r11_v1"
RAW_STAGE0_SUFFICIENCY_RUN_SCHEMA = "ember_ecp_raw_stage0_sufficiency_run_v1"
RAW_STAGE0_SUFFICIENCY_STAGE = "g3_raw_stage0_sufficiency_diagnostic"
FRESH_SCORER = "fresh"
SCORER_ALL_PARAMETERS = "all"
SCORER_NATIVE_HEADS_ONLY = "native_heads_only"
SCORER_FEATURE_CHART_ONLY = "feature_chart_only"


@dataclass(frozen=True)
class FunctionalPanelVisit:
    action_demos: tuple[int, ...]
    action_frames: tuple[int, ...]
    policy_rng_seed: int
    flow_loss: float


@dataclass(frozen=True)
class FunctionalPanelAuthority:
    task_id: int
    role: str
    panel_a: tuple[FunctionalPanelVisit, ...]
    panel_b: tuple[FunctionalPanelVisit, ...]
    program_video_demos: tuple[int, ...]
    path: Path


@dataclass(frozen=True)
class JointTaskConditions:
    fit_views: tuple[MappingCondition, MappingCondition]
    held_video: MappingCondition


class JointWriterState(torch.nn.Module):
    """Checkpoint only trainable Writer modules, never source/Stage0 weights."""

    def __init__(
        self,
        program: NaturalProgramModel,
        compiler: SharedNativeFactorCompiler,
    ) -> None:
        super().__init__()
        self.language_reader = program.language_reader
        self.scene_reader = program.scene_reader
        self.process_fusion = program.process_fusion
        self.aligner = program.aligner
        self.primal_scorer = compiler.primal_scorer


@dataclass
class JointProgramPrimalRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    base_config: dict[str, Any]
    context: DistributedContext
    tasks: tuple[NaturalProgramTask, ...]
    task_by_id: dict[int, NaturalProgramTask]
    mapping_split: SharedCompilerMappingSplit
    task_conditions: dict[int, JointTaskConditions]
    panels: dict[int, FunctionalPanelAuthority]
    video_store: RawTeacherVideoStore
    query_dataset: FunctionalQueryDataset
    query_processor: Pi05LiberoProcessor
    panel_batch_cache: dict[tuple[int, str, int], dict[str, Any]]
    counterfactual_margin_scales: dict[int, float]
    positive_control_files: tuple[Path, ...]
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    policy: torch.nn.Module
    program: NaturalProgramModel
    compiler: SharedNativeFactorCompiler
    writer_state: JointWriterState
    owners: tuple[TargetOwner, ...]
    ranks: SharedCompilerRankAssets
    rank4_contract: Any
    native_teachers: NativeTeacherStore
    consensus_teachers: FitConsensusTeacherStore
    condition_cache: FrozenMappingConditionCache
    query_points: int
    trainable_parameters: tuple[torch.nn.Parameter, ...]
    frozen_parameters: tuple[torch.nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_presence: tuple[bool, ...] | None
    optimizer_steps: int
    stop_after_step: int
    checkpoint_steps: tuple[int, ...]
    metrics_rows: int
    primal_scorer_initialization: dict[str, Any]
    functional_code_targets: dict[int, FunctionalCodeTarget]
    functional_code_authority: dict[str, Any]
    run_contract: dict[str, Any]

    def close(self) -> None:
        self.video_store.close()
        self.query_dataset.close()


@dataclass(frozen=True)
class _AuthorityAssets:
    selected_tasks: tuple[NaturalProgramTask, ...]
    task_by_id: dict[int, NaturalProgramTask]
    panels: dict[int, FunctionalPanelAuthority]
    mapping_split: SharedCompilerMappingSplit
    task_conditions: dict[int, JointTaskConditions]
    expected_checkpoint: Path
    source: dict[str, Any]
    source_config: dict[str, Any]
    counterfactual_margin_scales: dict[int, float]
    positive_control_files: tuple[Path, ...]


@dataclass(frozen=True)
class _ModelAssets:
    policy: torch.nn.Module
    ranks: SharedCompilerRankAssets
    owners: tuple[TargetOwner, ...]
    rank4_contract: Any
    program: NaturalProgramModel
    compiler: SharedNativeFactorCompiler
    writer_state: JointWriterState
    trainable: tuple[torch.nn.Parameter, ...]
    frozen: tuple[torch.nn.Parameter, ...]
    native_teachers: NativeTeacherStore
    consensus_teachers: FitConsensusTeacherStore
    primal_scorer_initialization: dict[str, Any]
    functional_code_targets: dict[int, FunctionalCodeTarget]
    functional_code_authority: dict[str, Any]


@dataclass(frozen=True)
class _DataAssets:
    video_store: RawTeacherVideoStore
    query_dataset: FunctionalQueryDataset
    query_processor: Pi05LiberoProcessor
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    condition_cache: FrozenMappingConditionCache
    query_points: int


@dataclass(frozen=True)
class _OptimizerCursor:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    checkpoints: tuple[int, ...]
    stop: int
    optimizer_steps: int
    metrics_rows: int


def is_chart_reconnect_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == CHART_RECONNECT_SCHEMA


def is_functional_chart_acquisition_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") in {
        FUNCTIONAL_CHART_ACQUISITION_SCHEMA,
        FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA,
    }


def is_r5_chart_config(config: Mapping[str, Any]) -> bool:
    return (
        is_chart_reconnect_config(config)
        or is_functional_chart_acquisition_config(config)
        or config.get("schema_version") in {
            FUNCTIONAL_REFINEMENT_SCHEMA,
            RAW_STAGE0_SUFFICIENCY_SCHEMA,
            primal_capacity.BANK_INTERACTION_CONTROL_SCHEMA,
        }
    )


def is_raw_stage0_sufficiency_config(config: Mapping[str, Any]) -> bool:
    return config.get("schema_version") == RAW_STAGE0_SUFFICIENCY_SCHEMA


def joint_run_schema(config: Mapping[str, Any]) -> str:
    if is_chart_reconnect_config(config):
        return CHART_RECONNECT_RUN_SCHEMA
    if config.get("schema_version") == FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA:
        return FUNCTIONAL_CODE_STABLE_JOINT_RUN_SCHEMA
    if config.get("schema_version") == FUNCTIONAL_REFINEMENT_SCHEMA:
        return FUNCTIONAL_REFINEMENT_RUN_SCHEMA
    if is_raw_stage0_sufficiency_config(config):
        return RAW_STAGE0_SUFFICIENCY_RUN_SCHEMA
    if is_functional_chart_acquisition_config(config):
        return FUNCTIONAL_CHART_ACQUISITION_RUN_SCHEMA
    return J2_RUN_SCHEMA


def joint_stage(config: Mapping[str, Any]) -> str:
    if is_chart_reconnect_config(config):
        return CHART_RECONNECT_STAGE
    if config.get("schema_version") == FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA:
        return FUNCTIONAL_CODE_STABLE_JOINT_STAGE
    if config.get("schema_version") == FUNCTIONAL_REFINEMENT_SCHEMA:
        return FUNCTIONAL_REFINEMENT_STAGE
    if is_raw_stage0_sufficiency_config(config):
        return RAW_STAGE0_SUFFICIENCY_STAGE
    if is_functional_chart_acquisition_config(config):
        return FUNCTIONAL_CHART_ACQUISITION_STAGE
    return J2_STAGE


def load_joint_program_primal_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    split = config.get("task_split", {})
    data = config.get("data", {})
    joint = config.get("optimization", {}).get("joint", {})
    counterfactual = joint.get("counterfactual", {})
    cache_authority = config.get("frozen_condition_cache_authority", {})
    tasks = tuple(
        map(
            int,
            (
                *split.get("gradient_meta", ()),
                *split.get("gradient_target", ()),
                *split.get("true_task_held_meta", ()),
                *split.get("true_task_held_target", ()),
            ),
        )
    )
    schema = config.get("schema_version")
    wall = config.get("information_wall", {})
    model = config.get("model", {})
    authorities = config.get("authorities", {})
    common_valid = all(
        (
            schema
            in {
                J2_SCHEMA,
                CHART_RECONNECT_SCHEMA,
                FUNCTIONAL_CHART_ACQUISITION_SCHEMA,
                FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA,
                FUNCTIONAL_REFINEMENT_SCHEMA,
                RAW_STAGE0_SUFFICIENCY_SCHEMA, primal_capacity.BANK_INTERACTION_CONTROL_SCHEMA,
            },
            len(tasks) == len(set(tasks)) == 12,
            split.get("gradient_meta") == [1, 8, 9, 32, 52],
            split.get("gradient_target") == [72, 73, 75, 93, 94],
            split.get("true_task_held_meta") == [2],
            split.get("true_task_held_target") == [74],
            set(map(int, authorities.get("functional_panel_records", {})))
            == set(tasks),
            data.get("K") == 1,
            data.get("fit_video_views_per_task") == 2,
            data.get("panel_visits") == 16,
            data.get("rows_per_visit") == 16,
            joint.get("warmup_optimizer_steps") == 10,
            joint.get("effective_optimizer_steps") == 100,
            joint.get("checkpoint_effective_steps") == [60, 100],
            joint.get("global_tasks_per_optimizer_step") == 6,
            joint.get("video_views_per_task") == 2,
            cache_authority.get("config_schema")
            == "ember_ecp_joint_program_primal_j2_v1",
            cache_authority.get("config_bytes") == 6017,
            wall.get("action_meta_installed") is False,
            wall.get("shuffled_or_reversed_use") is False,
        )
    )
    counterfactual_valid = all(
        (
            schema == J2_SCHEMA,
            config.get("status")
            == "active_counterfactual_functional_routing_qualification",
            model.get("primal_scorer_initialization") == FRESH_SCORER,
            model.get("primal_scorer_trainable_partition", SCORER_ALL_PARAMETERS)
            == SCORER_ALL_PARAMETERS,
            config.get("optimization", {}).get("loss")
            == "correct_flow_plus_paired_counterfactual_functional_margin",
            counterfactual.get("arm_schedule")
            == "alternate_wrong_program_wrong_bank",
            counterfactual.get("negative_pairing") == "same_role_cyclic_next",
            counterfactual.get("negative_views_per_task") == 1,
            counterfactual.get("normalized_margin") == 0.1,
            counterfactual.get("weight") == 1.0,
            counterfactual.get("margin_scale")
            == "formal_positive_control_fit_panel_a_mean_benefit",
            isinstance(authorities.get("positive_control_root"), str),
        )
    )
    reconnect_valid = all(
        (
            schema == CHART_RECONNECT_SCHEMA,
            config.get("status")
            == "active_natural_program_chart_reconnect_qualification",
            model.get("program_initialization")
            == "c1493a1_macro20_model_tensors",
            model.get("primal_scorer_initialization")
            == R5_SHARED_FUNCTIONAL_CHART,
            model.get("primal_scorer_trainable_partition")
            == SCORER_NATIVE_HEADS_ONLY,
            config.get("optimization", {}).get("loss")
            == "generated_rank16_cross_episode_pi05_flow_only",
            "counterfactual" not in joint,
            isinstance(authorities.get("r5_primal_scorer_checkpoint"), str),
            isinstance(authorities.get("r5_gate_aggregate"), str),
            wall.get("functional_chart_initialization_training_only") is True,
            wall.get("primal_scorer_feature_chart_frozen") is True,
            wall.get("fixed_routing_token_deployment_input") is False,
        )
    )
    acquisition_valid = all(
        (
            schema == FUNCTIONAL_CHART_ACQUISITION_SCHEMA,
            config.get("status")
            == "active_fit_only_functional_code_chart_acquisition",
            model.get("program_initialization")
            == "c1493a1_macro20_model_tensors",
            model.get("primal_scorer_initialization")
            == R5_SHARED_FUNCTIONAL_CHART,
            model.get("primal_scorer_trainable_partition")
            == SCORER_FEATURE_CHART_ONLY,
            config.get("optimization", {}).get("loss")
            == "fit_only_functional_code_outer_direction_only",
            "counterfactual" not in joint,
            isinstance(authorities.get("positive_control_root"), str),
            wall.get("functional_code_labels_training_only") is True,
            wall.get("native_heads_frozen") is True,
            wall.get("fixed_routing_token_deployment_input") is False,
        )
    )
    joint_acquisition_valid = all(
        (
            schema == FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA,
            config.get("status")
            == "active_fit_only_functional_code_stable_chart_joint_acquisition",
            model.get("program_initialization")
            == "c1493a1_macro20_model_tensors",
            model.get("primal_scorer_initialization")
            == R5_SHARED_FUNCTIONAL_CHART,
            model.get("primal_scorer_trainable_partition")
            == SCORER_ALL_PARAMETERS,
            config.get("optimization", {}).get("loss")
            == "fit_only_functional_code_outer_direction_only",
            "counterfactual" not in joint,
            isinstance(authorities.get("r5_primal_scorer_checkpoint"), str),
            isinstance(authorities.get("r5_gate_aggregate"), str),
            isinstance(authorities.get("positive_control_root"), str),
            wall.get("functional_code_labels_training_only") is True,
            wall.get("native_heads_trainable") is True,
            wall.get("absolute_outer_code_target_anchors_moving_heads") is True,
            wall.get("stable_r5_shared_chart_initialization") is True,
            wall.get("fixed_routing_token_deployment_input") is False,
        )
    )
    refinement_valid = all(
        (
            schema == FUNCTIONAL_REFINEMENT_SCHEMA,
            config.get("status")
            == "active_r9_initialized_functional_refinement",
            model.get("program_initialization") == R9_STABLE_CONTENT,
            model.get("primal_scorer_initialization") == R9_STABLE_CONTENT,
            model.get("primal_scorer_trainable_partition")
            == SCORER_NATIVE_HEADS_ONLY,
            config.get("optimization", {}).get("loss")
            == "generated_rank16_cross_episode_pi05_flow_only",
            "counterfactual" not in joint,
            isinstance(authorities.get("r9_writer_checkpoint"), str),
            isinstance(authorities.get("r9_gate_aggregate"), str),
            wall.get("r9_writer_initialization_training_only") is True,
            wall.get("primal_scorer_feature_chart_frozen") is True,
            wall.get("outer_code_loss_active") is False,
            wall.get("fixed_routing_token_deployment_input") is False,
        )
    )
    raw_stage0_valid = all(
        (
            schema == RAW_STAGE0_SUFFICIENCY_SCHEMA,
            config.get("status") == "active_raw_stage0_sufficiency_diagnostic",
            model.get("program_initialization") == R9_STABLE_CONTENT,
            model.get("program_input") == RAW_STAGE0_PROGRAM_INPUT,
            model.get("primal_scorer_initialization") == R9_STABLE_CONTENT,
            model.get("primal_scorer_trainable_partition")
            == SCORER_NATIVE_HEADS_ONLY,
            config.get("optimization", {}).get("loss")
            == "generated_rank16_cross_episode_pi05_flow_only",
            "counterfactual" not in joint,
            isinstance(authorities.get("r9_writer_checkpoint"), str),
            isinstance(authorities.get("r9_gate_aggregate"), str),
            wall.get("diagnostic_only") is True,
            wall.get("deployment_writer") is False,
            wall.get("r9_writer_initialization_training_only") is True,
            wall.get("natural_program_process_fusion_active") is False,
            wall.get("canonical_alignment_active") is False,
            wall.get("primal_scorer_feature_chart_frozen") is True,
            wall.get("outer_code_loss_active") is False,
            wall.get("fixed_routing_token_deployment_input") is False,
        )
    )
    if not common_valid or not (
        counterfactual_valid
        or reconnect_valid
        or acquisition_valid
        or joint_acquisition_valid
        or refinement_valid
        or raw_stage0_valid or primal_capacity.bank_interaction_control_config_valid(config)
    ):
        raise ValueError("unsupported joint Program-primal functional config")
    return config


def _counterfactual_margin_scales(
    config: Mapping[str, Any], *, asset_root: Path
) -> tuple[dict[int, float], tuple[Path, ...]]:
    """Load training-fit positive-control benefits used only as hinge margins."""

    root = (
        asset_root / str(config["authorities"]["positive_control_root"])
    ).resolve()
    aggregate_path = root / "aggregate.json"
    aggregate = read_json(aggregate_path)
    if (
        aggregate.get("schema_version")
        != "ember_ecp_j2_functional_positive_control_aggregate_v1"
        or aggregate.get("overall_gate")
        != "pass_after_runtime_microbatch_correction"
    ):
        raise ValueError("J3 positive-control aggregate authority changed")
    gradient_tasks = tuple(
        map(
            int,
            (
                *config["task_split"]["gradient_meta"],
                *config["task_split"]["gradient_target"],
            ),
        )
    )
    if {int(row["task"]) for row in aggregate.get("tasks", ())} != set(
        gradient_tasks
    ):
        raise ValueError("J3 positive-control task coverage changed")
    scales: dict[int, float] = {}
    files = [aggregate_path]
    for task in gradient_tasks:
        path = root / f"task_{task:03d}" / "result.json"
        row = read_json(path)
        fit = row.get("evaluation", {}).get("fit_videos", ())
        benefits = [
            float(value.get("panel_a", {}).get("benefit_over_carrier", float("nan")))
            for value in fit
        ]
        if (
            row.get("schema_version")
            != "ember_ecp_j2_functional_positive_control_task_v1"
            or row.get("status") != "complete"
            or int(row.get("task", -1)) != task
            or len(benefits) != 2
            or not all(math.isfinite(value) and value > 0 for value in benefits)
        ):
            raise ValueError(f"J3 positive-control fit authority changed for task {task}")
        scales[task] = sum(benefits) / len(benefits)
        files.append(path)
    return scales, tuple(files)


def _tasks(
    base: Mapping[str, Any], data_root: Path, asset_root: Path
) -> tuple[NaturalProgramTask, ...]:
    fold = base["fold"]
    return load_natural_program_tasks(
        meta_protocol_path=authority_path(base, "meta_protocol", asset_root=asset_root),
        source_manifest_path=authority_path(
            base, "source_manifest", asset_root=asset_root
        ),
        target_manifest_path=authority_path(
            base, "target_manifest", asset_root=asset_root
        ),
        data_root=data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )


def _panel_visit(row: Mapping[str, Any]) -> FunctionalPanelVisit:
    demos = tuple(map(int, row.get("action_demos", ())))
    frames = tuple(map(int, row.get("action_frames", ())))
    seed = int(row.get("policy_rng_seed", -1))
    flow_loss = float(row.get("flow_loss", float("nan")))
    if (
        len(demos) != len(frames)
        or len(demos) != 16
        or seed < 0
        or not math.isfinite(flow_loss)
        or flow_loss <= 0
    ):
        raise ValueError("J2 functional panel visit changed")
    return FunctionalPanelVisit(demos, frames, seed, flow_loss)


def _load_panels(
    config: Mapping[str, Any], *, asset_root: Path
) -> dict[int, FunctionalPanelAuthority]:
    output = {}
    for task_key, relative in config["authorities"]["functional_panel_records"].items():
        task_id = int(task_key)
        path = (asset_root / str(relative)).resolve()
        row = read_json(path)
        panel_a = tuple(_panel_visit(value) for value in row.get("panel_a_visits", ()))
        panel_b = tuple(_panel_visit(value) for value in row.get("panel_b_visits", ()))
        videos = tuple(map(int, row.get("program_video_demos", ())))
        a_demos = {demo for visit in panel_a for demo in visit.action_demos}
        b_demos = {demo for visit in panel_b for demo in visit.action_demos}
        if (
            int(row.get("task", -1)) != task_id
            or row.get("role") not in {"meta_fit", "target_fit"}
            or len(panel_a) != 16
            or len(panel_b) != 16
            or int(row.get("logical_rows_per_panel", -1)) != 256
            or not videos
            or a_demos.intersection(b_demos)
            or a_demos.intersection(videos)
            or b_demos.intersection(videos)
            or row.get("episode_sets_pairwise_disjoint") is not True
        ):
            raise ValueError("J2 functional panel authority changed")
        output[task_id] = FunctionalPanelAuthority(
            task_id=task_id,
            role=str(row["role"]),
            panel_a=panel_a,
            panel_b=panel_b,
            program_video_demos=videos,
            path=path,
        )
    return output


def _task_conditions(
    config: Mapping[str, Any], split: SharedCompilerMappingSplit
) -> dict[int, JointTaskConditions]:
    gradient = tuple(
        map(
            int,
            (
                *config["task_split"]["gradient_meta"],
                *config["task_split"]["gradient_target"],
            ),
        )
    )
    output = {}
    for task_id in gradient:
        fit = split.fit_by_task[task_id]
        held = split.video_held_by_task[task_id]
        if len(fit) < 2 or len(held) != 1:
            raise ValueError("J2 mapping video split changed")
        output[task_id] = JointTaskConditions(
            fit_views=(fit[0], fit[1]), held_video=held[0]
        )
    return output


def _joint_parameter_ownership(
    program: NaturalProgramModel,
    compiler: SharedNativeFactorCompiler,
    *,
    scorer_partition: str = SCORER_ALL_PARAMETERS,
    raw_stage0_input: bool = False,
) -> tuple[JointWriterState, tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    program.requires_grad_(False).eval()
    compiler.requires_grad_(False).eval()
    writer = JointWriterState(program, compiler)
    program_modules = [program.language_reader, program.scene_reader]
    if not raw_stage0_input:
        program_modules.extend((program.process_fusion, program.aligner))
    for module in program_modules:
        module.requires_grad_(True).train()
    if scorer_partition == SCORER_ALL_PARAMETERS:
        compiler.primal_scorer.requires_grad_(True).train()
    elif scorer_partition == SCORER_NATIVE_HEADS_ONLY:
        compiler.primal_scorer.input_primal_heads.requires_grad_(True).train()
        compiler.primal_scorer.output_primal_heads.requires_grad_(True).train()
    elif scorer_partition == SCORER_FEATURE_CHART_ONLY:
        compiler.primal_scorer.requires_grad_(True).train()
        compiler.primal_scorer.input_primal_heads.requires_grad_(False).eval()
        compiler.primal_scorer.output_primal_heads.requires_grad_(False).eval()
    else:
        raise ValueError("unsupported joint primal-scorer partition")
    program.encoder.requires_grad_(False).eval()
    program.decoder.requires_grad_(False).eval()
    compiler.scale_head.requires_grad_(False).eval()
    trainable = tuple(
        parameter for parameter in writer.parameters() if parameter.requires_grad
    )
    roots = (program, compiler)
    frozen = tuple(
        parameter
        for root in roots
        for parameter in root.parameters()
        if not parameter.requires_grad
    )
    if not trainable or len(set(map(id, trainable))) != len(trainable):
        raise ValueError("J2 trainable parameter ownership changed")
    return writer, trainable, frozen


def _optimizer(
    parameters: tuple[torch.nn.Parameter, ...], config: Mapping[str, Any]
) -> torch.optim.AdamW:
    cell = config["optimization"]["joint"]["optimizer"]
    return torch.optim.AdamW(
        parameters,
        lr=float(cell["peak_lr"]),
        betas=tuple(cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    joint = config["optimization"]["joint"]
    cell = joint["optimizer"]
    warmup = int(joint["warmup_optimizer_steps"])
    total = warmup + int(joint["effective_optimizer_steps"])
    floor = float(cell["decay_lr"]) / float(cell["peak_lr"])

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _inventory(
    policy: torch.nn.Module,
    program: NaturalProgramModel,
    compiler: SharedNativeFactorCompiler,
    owners: tuple[TargetOwner, ...],
) -> dict[str, Any]:
    action_meta = [
        f"{prefix}.{name}:{type(module).__name__}"
        for root, prefix in ((policy, "policy"), (program, "program"))
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    trainable = [
        *(f"program.{name}" for name, value in program.named_parameters() if value.requires_grad),
        *(f"compiler.{name}" for name, value in compiler.named_parameters() if value.requires_grad),
    ]
    forbidden = [
        name
        for name in trainable
        if any(token in name for token in ("task_lookup", "video_lookup", "frame_lookup", "free_logits"))
    ]
    if (
        action_meta
        or any(value.requires_grad for value in policy.parameters())
        or any(value.requires_grad for value in program.encoder.parameters())
        or any(value.requires_grad for value in program.decoder.parameters())
        or any(value.requires_grad for value in compiler.scale_head.parameters())
        or forbidden
        or set(native_capture_modes(policy, owners)) != {"identity_lora_base_layer"}
    ):
        raise ValueError("J2 pure-Native information wall changed")
    return {
        "action_meta_argument": None,
        "install_action_meta_lora": False,
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "source_policy_trainable_parameter_count": 0,
        "native_stage0_trainable_parameter_count": 0,
        "temporal_decoder_trainable_parameter_count": 0,
        "scale_trainable_parameter_count": 0,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": sum(value.numel() for value in (*program.parameters(), *compiler.parameters()) if value.requires_grad),
        "task_video_frame_free_parameter_count": 0,
        "native_capture_modes": list(native_capture_modes(policy, owners)),
    }


def _topology(context: DistributedContext) -> list[dict[str, Any]]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    rows: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rows, local)
    else:
        rows[0] = local
    return rows


def _run_contract(runtime: JointProgramPrimalRuntime) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    contract = {
        "schema_version": joint_run_schema(runtime.config),
        "stage": joint_stage(runtime.config),
        "phase": runtime.args.phase,
        "mode": runtime.args.mode,
        "git": {
            "branch": state["branch"],
            "commit": state["commit"],
            "authority_commit": state["commit"] if runtime.args.mode == "formal" else state["authority_commit"],
        },
        "config": {"path": str(runtime.args.config), "bytes": runtime.args.config.stat().st_size},
        "base_g3_config": {"path": str(runtime.args.base_config), "bytes": runtime.args.base_config.stat().st_size},
        "source_checkpoint": str(runtime.args.checkpoint),
        "tokenizer": str(runtime.args.tokenizer_path),
        "data_root": str(runtime.args.data_root),
        "condition_cache": {
            "root": str(runtime.args.condition_cache_root),
            "schema_version": FROZEN_CONDITION_CACHE_SCHEMA,
            "program_output_cached": False,
            "checkpoint_payload": False,
        },
        "task_split": dict(runtime.config["task_split"]),
        "functional_panels": {
            str(task): {"path": str(panel.path), "bytes": panel.path.stat().st_size}
            for task, panel in runtime.panels.items()
        },
        "primal_scorer_initialization": dict(
            runtime.primal_scorer_initialization
        ),
        "model": dict(runtime.config["model"]),
        "optimization": dict(runtime.config["optimization"]),
        "throughput_gate": dict(runtime.config["throughput_gate"]),
        "information_wall": dict(runtime.config["information_wall"]),
        "inventory": _inventory(runtime.policy, runtime.program, runtime.compiler, runtime.owners),
        "world_topology": _topology(runtime.context),
    }
    if runtime.counterfactual_margin_scales:
        contract["counterfactual_margin_authority"] = {
            "source": "formal_positive_control_fit_panel_a_mean_benefit",
            "files": [
                {"path": str(path), "bytes": path.stat().st_size}
                for path in runtime.positive_control_files
            ],
            "task_scales": {
                str(task): value
                for task, value in runtime.counterfactual_margin_scales.items()
            },
        }
    if runtime.functional_code_authority:
        contract["functional_code_target_authority"] = dict(
            runtime.functional_code_authority
        )
    return contract


def _authority_assets(
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
) -> _AuthorityAssets:
    if context.world_size not in config["profile"]["allowed_world_sizes"]:
        raise ValueError("J2 world size is outside its launch contract")
    if args.mode == "formal":
        state = git_state(REPO_ROOT)
        if (
            not git_state_is_clean_pushed_or_frozen_authority(state)
            or state.get("branch") != ""
            or state.get("upstream") is not None
        ):
            raise ValueError("formal J2 requires clean detached origin/main authority")
    seed_everything(int(config["optimization"]["seed"]), context)
    all_tasks = _tasks(base, args.data_root, args.asset_root)
    task_by_id = {task.authority_id: task for task in all_tasks}
    panels = _load_panels(config, asset_root=args.asset_root)
    if is_r5_chart_config(config):
        margin_scales, positive_control_files = {}, ()
    else:
        margin_scales, positive_control_files = _counterfactual_margin_scales(
            config, asset_root=args.asset_root
        )
    selected_tasks = tuple(task_by_id[task] for task in sorted(panels))
    mapping_split = load_mapping_split(base, asset_root=args.asset_root)
    expected_checkpoint = authority_path(
        base, "source_checkpoint", asset_root=args.asset_root
    )
    expected_tokenizer = authority_path(
        base, "tokenizer", asset_root=args.asset_root
    )
    if (
        args.checkpoint != expected_checkpoint
        or args.source_run != expected_checkpoint.parent.parent
        or args.tokenizer_path != expected_tokenizer
    ):
        raise ValueError("J2 source or tokenizer authority changed")
    return _AuthorityAssets(
        selected_tasks=selected_tasks,
        task_by_id=task_by_id,
        panels=panels,
        mapping_split=mapping_split,
        task_conditions=_task_conditions(config, mapping_split),
        expected_checkpoint=expected_checkpoint,
        source=stage0_source_authority(args),
        source_config=load_config(
            authority_path(base, "source_base_config", asset_root=args.asset_root)
        ),
        counterfactual_margin_scales=margin_scales,
        positive_control_files=positive_control_files,
    )


def _model_assets(
    args: argparse.Namespace,
    context: DistributedContext,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    authority: _AuthorityAssets,
) -> _ModelAssets:
    policy = load_policy(
        Path(authority.source["model_path"]), authority.source_config, context.device
    )
    policy.requires_grad_(False).eval()
    ranks = load_shared_rank_assets(
        base, asset_root=args.asset_root,
        held_global_ids=set(map(int, base["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    rank4_contract = derive_pi05_lora_rank(ranks.contract, rank=4)
    program = build_frozen_g2_program(
        base, asset_root=args.asset_root, owners=owners, device=context.device
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    model = base["model"]
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(model["program_width"]),
        event_slots=int(model["event_slots"]),
        relative_eigenvalue_floor=float(model["relative_eigenvalue_floor"]),
        replay_score_rms=float(model["replay_score_rms"]),
        covariance_frame_chunk=int(model["frame_chunk_size"]),
        inverse_covariance_power=float(config["model"].get("inverse_covariance_power", 1.0)),
        scale_prior_ratio=load_shared_scale_prior(
            base, asset_root=args.asset_root, device=context.device
        ),
    ).to(context.device)
    scorer_initialization = str(config["model"]["primal_scorer_initialization"])
    if scorer_initialization == R5_SHARED_FUNCTIONAL_CHART:
        initialization = load_passed_r5_primal_scorer(
            config, compiler, asset_root=args.asset_root, device=context.device
        )
    elif scorer_initialization == FRESH_SCORER:
        initialization = {
            "kind": FRESH_SCORER,
            "state": "seeded_random",
            "fixed_routing_token_loaded": False,
            "task_lookup_parameters_loaded": False,
        }
    elif scorer_initialization == R9_STABLE_CONTENT:
        initialization = None
    elif config.get("schema_version") == "ember_ecp_routing_token_control_r1_v1":
        initialization = {
            "kind": scorer_initialization,
            "state": "deferred_to_routing_control",
        }
    else:
        raise ValueError("unsupported joint primal-scorer initialization")
    writer_state, trainable, frozen = _joint_parameter_ownership(
        program,
        compiler,
        scorer_partition=str(
            config["model"].get(
                "primal_scorer_trainable_partition", SCORER_ALL_PARAMETERS
            )
        ),
        raw_stage0_input=is_raw_stage0_sufficiency_config(config),
    )
    if scorer_initialization == R9_STABLE_CONTENT:
        initialization = load_r9_stable_writer(
            config,
            writer_state,
            asset_root=args.asset_root,
            device=context.device,
        )
    if initialization is None:
        raise RuntimeError("joint Writer initialization was not resolved")
    teacher_path = authority_path(
        base, "native_teacher_manifest", asset_root=args.asset_root
    )
    teacher_root = read_json(teacher_path)
    native_teachers = NativeTeacherStore(
        teacher_path,
        contract=rank4_contract,
        expected_fit_task_ids=set(map(int, teacher_root["coverage"]["task_ids"])),
        expected_full_fit_task_ids=set(
            map(int, teacher_root["fit_authority_task_ids"])
        ),
        device=context.device,
    )
    functional_code_targets: dict[int, FunctionalCodeTarget] = {}
    functional_code_authority: dict[str, Any] = {}
    if is_functional_chart_acquisition_config(config):
        gradient_tasks = tuple(
            map(
                int,
                (
                    *config["task_split"]["gradient_meta"],
                    *config["task_split"]["gradient_target"],
                ),
            )
        )
        functional_code_targets, functional_code_authority = (
            load_functional_code_targets(
                config,
                asset_root=args.asset_root,
                task_ids=gradient_tasks,
                owners=owners,
                device=context.device,
            )
        )
    return _ModelAssets(
        policy=policy,
        ranks=ranks,
        owners=owners,
        rank4_contract=rank4_contract,
        program=program,
        compiler=compiler,
        writer_state=writer_state,
        trainable=trainable,
        frozen=frozen,
        native_teachers=native_teachers,
        consensus_teachers=FitConsensusTeacherStore(
            native_teachers, authority.mapping_split, rank4_contract
        ),
        primal_scorer_initialization=initialization,
        functional_code_targets=functional_code_targets,
        functional_code_authority=functional_code_authority,
    )


def _data_assets(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    context: DistributedContext,
    authority: _AuthorityAssets,
    model: _ModelAssets,
) -> _DataAssets:
    task_authorities = tuple(
        task.writer_authority() for task in authority.selected_tasks
    )
    video_store = RawTeacherVideoStore(
        task_authorities,
        frame_stride=int(base["data"]["frame_stride"]),
        max_open_files=8,
    )
    query_dataset = FunctionalQueryDataset(
        task_authorities,
        demo_indices=range(50),
        action_chunk_size=int(authority.source_config["features"]["chunk_size"]),
        max_open_files_per_worker=8,
    )
    query_processor = Pi05LiberoProcessor(
        load_stats(
            authority.source_config,
            authority.source_config["data"]["active_task_ids"],
        ),
        args.tokenizer_path,
        int(authority.source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    language_tokens = tokenize_stage0_languages(
        authority.selected_tasks,
        tokenizer_path=args.tokenizer_path,
        max_length=int(authority.source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    cache_authority = frozen_condition_cache_authority(
        config_schema=str(config["frozen_condition_cache_authority"]["config_schema"]),
        config_bytes=int(config["frozen_condition_cache_authority"]["config_bytes"]),
        source_checkpoint=authority.expected_checkpoint,
        g2_program_checkpoint=authority_path(
            base, "g2_program_checkpoint", asset_root=args.asset_root
        ),
        native_observer_checkpoint=authority_path(
            base, "native_observer_checkpoint", asset_root=args.asset_root
        ),
        frame_stride=int(base["data"]["frame_stride"]),
        owners=model.owners,
    )
    return _DataAssets(
        video_store=video_store,
        query_dataset=query_dataset,
        query_processor=query_processor,
        language_tokens=language_tokens,
        condition_cache=FrozenMappingConditionCache(
            args.condition_cache_root,
            owners=model.owners,
            operator=model.compiler.bank_operator,
            authority=cache_authority,
            cache_program=False,
        ),
        query_points=int(
            read_json(
                authority_path(base, "g2_config", asset_root=args.asset_root)
            )["data"]["query_points"]
        ),
    )


def _optimizer_cursor(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    writer_state: JointWriterState,
    trainable: tuple[torch.nn.Parameter, ...],
) -> _OptimizerCursor:
    optimizer = _optimizer(trainable, config)
    scheduler = _scheduler(optimizer, config)
    joint = config["optimization"]["joint"]
    warmup = int(joint["warmup_optimizer_steps"])
    effective = int(joint["effective_optimizer_steps"])
    checkpoints = tuple(
        warmup + int(value) for value in joint["checkpoint_effective_steps"]
    )
    stop = int(
        args.stop_after_step
        or (1 if args.mode == "profile" else warmup + effective)
    )
    allowed_stops = {1} if args.mode == "profile" else set(checkpoints)
    if stop not in allowed_stops:
        raise ValueError("J2 stop step is not pre-registered")
    optimizer_steps = 0
    metrics_rows = 0
    if args.resume is not None:
        optimizer_steps, expected_rows = load_ecp_checkpoint(
            checkpoint=args.resume,
            stage=joint_stage(config),
            context=context,
            model=writer_state,
            optimizer=optimizer,
            scheduler=scheduler,
            run_contract_schema=joint_run_schema(config),
        )
        if context.is_main:
            metrics_rows = reconcile_metrics(
                args.output_dir / "metrics.jsonl",
                optimizer_steps,
                expected_rows,
                cursor_key="optimizer_step",
            )
    return _OptimizerCursor(
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoints=checkpoints,
        stop=stop,
        optimizer_steps=optimizer_steps,
        metrics_rows=metrics_rows,
    )


def prepare_joint_program_primal_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> JointProgramPrimalRuntime:
    config = load_joint_program_primal_config(args.config)
    if (
        primal_capacity.is_bank_interaction_control_config(config)
        and args.phase != "positive-control"
    ):
        raise ValueError("bank-interaction control cannot train a shared Writer")
    base_path = (args.asset_root / config["authorities"]["base_g3_config"]).resolve()
    if args.base_config != base_path:
        raise ValueError("J2 base G3 config authority changed")
    base = load_shared_compiler_config(base_path)
    authority = _authority_assets(args, context, config, base)
    model = _model_assets(args, context, config, base, authority)
    data = _data_assets(args, config, base, context, authority, model)
    initialize_deferred_process_group(context, rendezvous_root=args.output_dir.parent)
    if context.world_size > 1:
        for value in model.writer_state.state_dict().values():
            dist.broadcast(value, src=0)
    cursor = _optimizer_cursor(
        args, config, context, model.writer_state, model.trainable
    )
    runtime = JointProgramPrimalRuntime(
        args=args,
        config=config,
        base_config=base,
        context=context,
        tasks=authority.selected_tasks,
        task_by_id=authority.task_by_id,
        mapping_split=authority.mapping_split,
        task_conditions=authority.task_conditions,
        panels=authority.panels,
        video_store=data.video_store,
        query_dataset=data.query_dataset,
        query_processor=data.query_processor,
        panel_batch_cache={},
        counterfactual_margin_scales=authority.counterfactual_margin_scales,
        positive_control_files=authority.positive_control_files,
        language_tokens=data.language_tokens,
        policy=model.policy,
        program=model.program,
        compiler=model.compiler,
        writer_state=model.writer_state,
        owners=model.owners,
        ranks=model.ranks,
        rank4_contract=model.rank4_contract,
        native_teachers=model.native_teachers,
        consensus_teachers=model.consensus_teachers,
        condition_cache=data.condition_cache,
        query_points=data.query_points,
        trainable_parameters=model.trainable,
        frozen_parameters=model.frozen,
        optimizer=cursor.optimizer,
        scheduler=cursor.scheduler,
        gradient_presence=None,
        optimizer_steps=cursor.optimizer_steps,
        stop_after_step=cursor.stop,
        checkpoint_steps=cursor.checkpoints,
        metrics_rows=cursor.metrics_rows,
        primal_scorer_initialization=model.primal_scorer_initialization,
        functional_code_targets=model.functional_code_targets,
        functional_code_authority=model.functional_code_authority,
        run_contract={},
    )
    runtime.run_contract = _run_contract(runtime)
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / "run_contract.json", runtime.run_contract)
    torch.cuda.reset_peak_memory_stats(context.device)
    return runtime
