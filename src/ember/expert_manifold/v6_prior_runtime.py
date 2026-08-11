"""Assets, random-reset environments, and exact-resume Reward-Credit runtime."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    ExpertTask,
    load_task_expert_config,
    load_train_tasks,
)
from ember.expert_manifold.v6_prior import (
    V6PriorOwnership,
    V6PriorWarmStart,
    freeze_v6_prior_writer,
    load_v6_prior_warm_start_,
)
from ember.expert_manifold.v6_prior_checkpoint import load_v6_prior_checkpoint
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    authority_path,
    load_v6_prior_config,
    runtime_for_mode,
)
from ember.expert_manifold.v6_prior_run_contract import (
    build_run_contract,
    checkpoint_contract,
    cursor_contract,
    publish_contract,
    residual_git_state,
)
from ember.lora import LoRAContract
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import DistributedContext, read_json
from ember.pi05_source_contract import reconcile_metrics
from ember.pi05_source_setup import (
    initialize_deferred_process_group,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.reward.protocol import RewardTask, SUITE_HORIZONS
from ember.reward.rollout import RandomResetEnvironmentPool
from ember.writer.architecture import LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.condition_update import (
    FrozenV6ConditionResidualWriter,
    ProgramReconciliationState,
    validate_frozen_v6_residual_writer,
)
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.topology import visible_physical_cuda_index


_RESUME_NAME = re.compile(r"macro_([0-9]{8})")


@dataclass(frozen=True)
class RuntimeSegment:
    total_macros: int
    checkpoint_macros: tuple[int, ...]
    start_macro: int
    stop_macro: int
    schedule_origin: int
    continuation_gate_evidence: Mapping[str, Any] | None

    @property
    def schedule_start_macro(self) -> int:
        return self.schedule_origin + self.start_macro

    @property
    def schedule_stop_macro(self) -> int:
        return self.schedule_origin + self.stop_macro


@dataclass
class V6PriorRuntime:
    args: argparse.Namespace
    context: DistributedContext
    config: dict[str, Any]
    segment: RuntimeSegment
    source: dict[str, Any]
    tokenizer: dict[str, Any]
    tasks: tuple[ExpertTask, ...]
    task_by_global_id: dict[int, ExpertTask]
    reward_task_by_global_id: dict[int, RewardTask]
    local_tasks: tuple[ExpertTask, ...]
    video_schedule: TeacherVideoSchedule
    video_store: RawTeacherVideoStore
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    processor: Pi05LiberoProcessor
    policy: torch.nn.Module
    writer: FrozenV6ConditionResidualWriter
    identity_state: dict[str, torch.Tensor]
    reconciliation: ProgramReconciliationState
    env_pool: RandomResetEnvironmentPool
    lora_contract: LoRAContract
    warm_start: V6PriorWarmStart
    ownership: V6PriorOwnership
    run_contract: dict[str, Any]
    checkpoint_contract: dict[str, Any]
    metrics_path: Path
    rank_counters: dict[str, int | float]


def _resume_macro(path: Path | None) -> int:
    if path is None:
        return 0
    match = _RESUME_NAME.fullmatch(path.name)
    if match is None or path.parent.name != "checkpoints":
        raise ExpertManifoldError("Reward-Credit resume path is not a macro checkpoint")
    return int(match.group(1))


def _registered_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


def _formal_decision_evidence(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    start_macro: int,
) -> Mapping[str, Any] | None:
    if args.mode != "formal":
        return None
    raw_decision = config["formal_run"]["decision_evaluation"]
    baseline = _registered_path(str(raw_decision["macro0_reference_root"]))
    macro1 = _registered_path(str(raw_decision["macro1_registered_root"]))
    macro2 = _registered_path(str(raw_decision["macro2_registered_root"]))
    controls1 = {
        condition: _registered_path(str(path))
        for condition, path in raw_decision["macro1_control_registered_roots"].items()
    }
    controls2 = {
        condition: _registered_path(str(path))
        for condition, path in raw_decision["macro2_control_registered_roots"].items()
    }
    decision = {
        **raw_decision,
        "macro0_reference_root": str(baseline),
        "macro1_registered_root": str(macro1),
        "macro2_registered_root": str(macro2),
        "macro1_control_registered_roots": {
            condition: str(path) for condition, path in controls1.items()
        },
        "macro2_control_registered_roots": {
            condition: str(path) for condition, path in controls2.items()
        },
    }
    if not baseline.is_dir():
        raise ExpertManifoldError("Reward-Credit baseline strict400 root is missing")
    if start_macro == 0:
        if (
            macro1.exists()
            or macro2.exists()
            or any(path.exists() for path in (*controls1.values(), *controls2.values()))
        ):
            raise ExpertManifoldError(
                "fresh Reward-Credit run requires unused registered evaluation roots"
            )
        return {
            "passed": None,
            "baseline_root": str(baseline),
            "macro1_registered_root": str(macro1),
            "macro2_registered_root": str(macro2),
        }
    if (
        start_macro != 1
        or not macro1.is_dir()
        or macro2.exists()
        or any(path.exists() for path in controls2.values())
    ):
        raise ExpertManifoldError(
            "Reward-Credit continuation root state is inconsistent"
        )
    from ember.pi05_eval.reward_credit_gate import (
        load_reward_credit_decision_evidence,
    )

    evidence = load_reward_credit_decision_evidence(
        macro0_root=baseline,
        macro1_root=macro1,
        resume_checkpoint=args.resume,
        expected_macro0_commit=str(decision["macro0_reference_commit"]),
        expected_macro0_correct=int(decision["macro0_reference_correct"]),
        expected_macro0_breadth=int(decision["macro0_reference_breadth"]),
        expected_current_commit=str(state["commit"]),
        decision_gates=config["formal_run"]["decision_gates"],
    )
    if evidence.get("six_arm_required") is True:
        if not all(path.is_dir() for path in controls1.values()):
            raise ExpertManifoldError(
                "macro1 Reward-Credit score requires its registered six-arm audit"
            )
        from ember.pi05_eval.reward_credit_gate import (
            load_reward_credit_six_arm_evidence,
        )

        evidence = {
            **evidence,
            "six_arm": load_reward_credit_six_arm_evidence(
                correct_root=macro1,
                control_roots=controls1,
                macro=1,
                decision_evaluation=decision,
                decision_gates=config["formal_run"]["decision_gates"],
            ),
        }
    requires_support = bool(
        config["formal_run"]["decision_gates"]["macro2_requires_macro1_support_gate"]
    )
    if requires_support and evidence.get("passed") is not True:
        raise ExpertManifoldError(
            "macro1 strict400 result did not pass Reward-Credit continuation"
        )
    return evidence


def _resolve_segment(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> RuntimeSegment:
    total, checkpoints, schedule_origin = runtime_for_mode(config, args.mode)
    start = _resume_macro(args.resume)
    stop = int(args.stop_after_macro or total)
    selected = (
        config["profile_run"]
        if args.mode == "mechanism-profile"
        else config["formal_run"]
    )
    registered_output = _registered_path(str(selected["registered_output_root"]))
    profile_valid = args.mode != "mechanism-profile" or (
        args.resume is None and start == 0 and stop == total == 1
    )
    formal_valid = args.mode != "formal" or (
        args.stop_after_macro is not None and (start, stop) in {(0, 1), (1, 2)}
    )
    state = residual_git_state(REPO_ROOT)
    if args.resume is None:
        git_valid = state["commit"] == state["authority_commit"]
    else:
        try:
            stored = read_json(args.resume.parent.parent / "run_contract.json")
            resume_commit = stored["git"]["commit"]
        except Exception as error:
            raise ExpertManifoldError(
                "Reward-Credit resume lacks its original Git authority"
            ) from error
        git_valid = (
            isinstance(resume_commit, str)
            and state["commit"] == resume_commit
            and state.get("authority_contains_commit") is True
        )
    valid = (
        context.world_size == int(selected["expected_world_size"]) == 6
        and 24 // context.world_size == int(selected["tasks_per_rank"]) == 4
        and args.num_workers == int(selected["num_workers_per_rank"]) == 0
        and args.output_dir == registered_output
        and 0 <= start < stop <= total
        and profile_valid
        and formal_valid
        and not state["dirty_paths"]
        and git_valid
    )
    if not valid:
        raise ExpertManifoldError(
            "Reward-Credit runtime differs from its registered frozen segment"
        )
    decision = _formal_decision_evidence(args, config, state, start_macro=start)
    return RuntimeSegment(
        total_macros=total,
        checkpoint_macros=checkpoints,
        start_macro=start,
        stop_macro=stop,
        schedule_origin=schedule_origin,
        continuation_gate_evidence=decision,
    )


def _validate_collective_environment(context: DistributedContext) -> None:
    if context.world_size <= 1:
        return
    expected = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
    }
    if {name: os.environ.get(name) for name in expected} != expected:
        raise ExpertManifoldError("Reward-Credit collective environment changed")


def _configure_egl(context: DistributedContext) -> None:
    expected = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(visible_physical_cuda_index(context.local_rank)),
    }
    for name, value in expected.items():
        observed = os.environ.get(name)
        if observed not in {None, value}:
            raise ExpertManifoldError(f"Reward-Credit {name} mapping changed")
        os.environ[name] = value


def _load_source(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    return authorities, source, tokenizer


def _local_rank_tasks(
    tasks: Sequence[ExpertTask],
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[ExpertTask, ...]:
    rows = config["data"]["rank_task_ordinals"]
    try:
        valid_shape = (
            context.world_size == 6
            and isinstance(rows, list)
            and len(rows) == 6
            and all(isinstance(row, list) and len(row) == 4 for row in rows)
        )
        ordinals = [[int(value) for value in row] for row in rows]
    except (TypeError, ValueError):
        valid_shape = False
        ordinals = []
    if not valid_shape or sorted(value for row in ordinals for value in row) != list(
        range(24)
    ):
        raise ExpertManifoldError("Reward-Credit rank assignment is invalid")
    by_ordinal = {task.ordinal: task for task in tasks}
    local = tuple(by_ordinal[value] for value in ordinals[context.rank])
    if len(local) != 4 or len({task.suite for task in local}) != 4:
        raise ExpertManifoldError(
            "Reward-Credit rank assignment is not one task per suite"
        )
    return local


def _build_tasks(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[
    tuple[ExpertTask, ...],
    dict[int, RewardTask],
    tuple[ExpertTask, ...],
    TeacherVideoSchedule,
]:
    expert_config = load_task_expert_config(
        authority_path(config, "task_expert_config")
    )
    tasks = load_train_tasks(expert_config, args.data_root)
    if len(tasks) != 24 or [task.ordinal for task in tasks] != list(range(24)):
        raise ExpertManifoldError("Reward-Credit train24 identity changed")
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = {
        int(row["global_task_id"]): row
        for row in manifest.get("tasks", [])
        if row.get("split_role") == "train"
    }
    if len(rows) != 24:
        raise ExpertManifoldError("Reward-Credit target manifest lost train24")
    reward_tasks: dict[int, RewardTask] = {}
    for task in tasks:
        row = rows.get(task.global_task_id)
        if (
            not isinstance(row, Mapping)
            or row.get("suite") != task.suite
            or int(row.get("task_id", -1)) != task.task_id
            or row.get("language") != task.language
            or Path(str(row.get("hdf5", {}).get("relative_path", ""))).name
            != task.authority.path.name
        ):
            raise ExpertManifoldError("Reward-Credit HDF5 and task manifest disagree")
        bddl = row["bddl"]
        reward_tasks[task.global_task_id] = RewardTask(
            suite=task.suite,
            task_id=task.task_id,
            global_task_id=task.global_task_id,
            split_role="train",
            language=task.language,
            problem_folder=str(row["problem_folder"]),
            bddl_file=str(bddl["filename"]),
            bddl_bytes=int(bddl["bytes"]),
            bddl_sha256=None,
            horizon=SUITE_HORIZONS[task.suite],
        )
    local = _local_rank_tasks(tasks, config, context)
    first, last = map(int, config["data"]["demo_indices"])
    schedule = TeacherVideoSchedule(
        task_ids=tuple(task.global_task_id for task in tasks),
        demo_indices=tuple(range(first, last + 1)),
        seed=int(config["data"]["teacher_video_seed"]),
        videos_per_visit=1,
    )
    return tasks, reward_tasks, local, schedule


def _build_policy_writer(
    *,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> tuple[
    torch.nn.Module,
    FrozenV6ConditionResidualWriter,
    dict[str, torch.Tensor],
    LoRAContract,
    V6PriorWarmStart,
    V6PriorOwnership,
]:
    policy = load_policy(Path(str(source["model_path"])), source_config, context.device)
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    if hasattr(policy, "config"):
        policy.config.gradient_checkpointing = False
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    template = prepare_frozen_writer_policy(policy, lora)
    identity = {name: value.detach().clone() for name, value in template.items()}
    bridge = policy.model.paligemma_with_expert
    writer_config = {
        name: value
        for name, value in config["writer"].items()
        if name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
    }
    base = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        **writer_config,
    )
    base_checkpoint = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve()
    warm_start = load_v6_prior_warm_start_(base, base_checkpoint)
    if any(
        not torch.equal(
            value.detach().cpu(), identity[name].detach().cpu().to(value.dtype)
        )
        for name, value in base.template_state().items()
    ):
        raise ExpertManifoldError("historical v6 load changed physical identity")
    ownership = freeze_v6_prior_writer(base)
    feature = config["condition_feature"]
    writer = FrozenV6ConditionResidualWriter(
        base,
        feature_width=int(feature["feature_width"]),
        feature_seed=int(feature["projection_seed"]),
    ).to(context.device)
    validate_frozen_v6_residual_writer(writer, require_zero_memory=True)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ExpertManifoldError("Reward-Credit source policy is not frozen")
    return policy, writer, identity, lora, warm_start, ownership


def _build_language_inputs(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source_config: Mapping[str, Any],
    tasks: Sequence[ExpertTask],
) -> tuple[
    RawTeacherVideoStore,
    Pi05LiberoProcessor,
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    max_length = int(source_config["features"]["tokenizer_max_length"])
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path,
        max_length,
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path, max_length, str(context.device)
    )
    language = {task.global_task_id: tokenizer((task.language,)) for task in tasks}
    store = RawTeacherVideoStore(
        [task.authority for task in tasks],
        frame_stride=int(config["writer"]["frame_stride"]),
        max_open_files=4,
    )
    return store, processor, language


def _prepare_libero_paths(
    args: argparse.Namespace, context: DistributedContext
) -> dict[str, str]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = prepare_libero_config(args.output_dir / "libero_config")
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    paths = payload[0]
    if not isinstance(paths, Mapping) or paths.get("error"):
        raise ExpertManifoldError(
            f"Reward-Credit LIBERO path preparation failed: {paths}"
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (args.output_dir / "libero_config").resolve()
    )
    return {str(name): str(value) for name, value in paths.items()}


def _reconcile_metrics_cursor(
    path: Path,
    *,
    context: DistributedContext,
    expected_rows: int,
) -> int:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = {
                "rows": reconcile_metrics(
                    path,
                    expected_rows,
                    expected_rows,
                    cursor_key="macro",
                )
            }
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    result = payload[0]
    if (
        not isinstance(result, Mapping)
        or result.get("error")
        or type(result.get("rows")) is not int
    ):
        raise ExpertManifoldError(
            f"Reward-Credit metrics differ from resume cursor: {result}"
        )
    return int(result["rows"])


def _restore_resume(
    runtime_args: argparse.Namespace,
    config: Mapping[str, Any],
    segment: RuntimeSegment,
    context: DistributedContext,
    writer: FrozenV6ConditionResidualWriter,
    reconciliation: ProgramReconciliationState,
    checkpoint_contract_value: Mapping[str, Any],
) -> dict[str, Any] | None:
    if runtime_args.resume is None:
        return None
    loaded, rows, interaction_cursor = load_v6_prior_checkpoint(
        checkpoint=runtime_args.resume,
        memory=writer.program_memory,
        reconciliation=reconciliation,
        context=context,
        expected_cursor_contract=cursor_contract(config, segment.start_macro),
        expected_checkpoint_contract=checkpoint_contract_value,
    )
    if loaded != segment.start_macro or rows != segment.start_macro:
        raise ExpertManifoldError("Reward-Credit resume cursor changed")
    validate_frozen_v6_residual_writer(writer, require_zero_memory=False)
    return interaction_cursor


def _rank_counters(
    restored_cursor: Mapping[str, Any] | None,
) -> dict[str, int | float]:
    if restored_cursor is None:
        return {
            "rollouts": 0,
            "environment_actions": 0,
            "successes": 0,
            "reward_sum": 0.0,
        }
    return {
        "rollouts": int(restored_cursor["rollouts"]),
        "environment_actions": int(restored_cursor["environment_actions"]),
        "successes": int(restored_cursor["successes"]),
        "reward_sum": float(restored_cursor["reward_sum"]),
    }


def _prepare_runtime(
    args: argparse.Namespace, context: DistributedContext
) -> V6PriorRuntime:
    raise ExpertManifoldError(
        "Reward-Credit training is retired on the active HEAD"
    )
