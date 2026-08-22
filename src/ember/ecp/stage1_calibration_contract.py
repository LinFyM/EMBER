"""Data, scheduling, and artifact authority for MDCO structured calibration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.stage1_config import stage1_repo_authority
from ember.ecp.stage1_support import PolicySupportPanel, PolicySupportTask
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_stats
from ember.reward.protocol import RewardTask, SUITE_HORIZONS


STRUCTURED_CALIBRATION_SCHEMA = "ember_ecp_stage1_structured_calibration_v1"
STRUCTURED_CALIBRATION_FILE = "structured_calibration.json"


@dataclass(frozen=True)
class StructuredCalibrationState:
    pending: bool
    assignments: tuple[tuple[int, ...], ...]
    reward_tasks: Mapping[int, RewardTask]
    processor: Pi05LiberoProcessor | None
    panel_requests: frozenset[tuple[int, int]]


def calibration_task_count(config: Mapping[str, Any], *, mode: str) -> int:
    cell = config["structured_calibration"]
    return int(cell["task_count"] if mode == "formal" else cell["profile_task_count"])


def structured_calibration_requested(
    args: Any,
    config: Mapping[str, Any],
    *,
    start_task_visits: int,
    stop_after_task_visits: int,
) -> bool:
    if args.mode == "profile":
        return bool(args.profile_structured_calibration)
    trigger = int(config["structured_calibration"]["after_task_visits"])
    return start_task_visits < trigger <= stop_after_task_visits


def build_calibration_assignments(
    tasks: Sequence[Any],
    *,
    world_size: int,
    frame_stride: int,
    task_count: int,
) -> tuple[tuple[int, ...], ...]:
    fit = tuple(task for task in tasks if task.fold_role == "fit")
    if task_count <= 0 or task_count > len(fit) or world_size <= 0:
        raise ValueError("invalid MDCO structured-calibration task ownership")
    selected = fit[:task_count]
    bins: list[list[int]] = [[] for _ in range(world_size)]
    costs = [0.0] * world_size

    def cost(task: Any) -> float:
        sampled_frames = sum(
            (int(length) - 1) // frame_stride + 1
            for length in task.episode_lengths
        ) / len(task.episode_lengths)
        return float(SUITE_HORIZONS[task.suite]) + sampled_frames

    for task in sorted(selected, key=lambda value: (-cost(value), value.ordinal)):
        rank = min(range(world_size), key=lambda value: (costs[value], value))
        bins[rank].append(int(task.ordinal))
        costs[rank] += cost(task)
    assignments = tuple(tuple(sorted(values)) for values in bins)
    flattened = sorted(ordinal for values in assignments for ordinal in values)
    if flattened != [int(task.ordinal) for task in selected]:
        raise ValueError("MDCO structured-calibration assignment lost a fit task")
    return assignments


def successful_panel_for_visit(
    task: PolicySupportTask, visit: int
) -> PolicySupportPanel:
    successful = tuple(panel for panel in task.panels if panel.kind == "successful")
    if not successful:
        raise ValueError("structured calibration task has no successful panel")
    return successful[visit % len(successful)]


def calibration_panel_requests(
    *,
    support_bank: Any,
    task_ordinals: Sequence[int],
    visit: int,
) -> frozenset[tuple[int, int]]:
    return frozenset(
        (
            int(ordinal),
            int(successful_panel_for_visit(support_bank.task(ordinal), visit).panel_id),
        )
        for ordinal in task_ordinals
    )


def load_stage1_reward_tasks(
    config: Mapping[str, Any], tasks: Sequence[Any]
) -> dict[int, RewardTask]:
    source = read_json(stage1_repo_authority(config, "source_overlap_audit"))
    target = read_json(stage1_repo_authority(config, "target_manifest"))
    source_rows = {
        int(row["task_id"]): dict(row)
        for row in source.get("source_tasks", ())
        if row.get("decision") == "active"
    }
    target_rows = {
        int(row["global_task_id"]): dict(row) for row in target.get("tasks", ())
    }
    if (
        source.get("schema_version") != "ember_pi05_source_overlap_v1"
        or target.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or len(source_rows) != 71
    ):
        raise ValueError("MDCO calibration BDDL authority changed")
    result: dict[int, RewardTask] = {}
    for task in tasks:
        if task.fold_role != "fit":
            continue
        if task.domain == "libero90_nonheld":
            row = source_rows.get(int(task.task_id))
            bddl = {} if row is None else row
            bddl_file = bddl.get("bddl_file")
            bddl_bytes = bddl.get("bddl_bytes")
        else:
            row = target_rows.get(int(task.global_task_id))
            bddl = {} if row is None else row.get("bddl", {})
            bddl_file = bddl.get("filename")
            bddl_bytes = bddl.get("bytes")
        if (
            row is None
            or str(row.get("suite")) != task.suite
            or int(row.get("task_id", -1)) != task.task_id
            or str(row.get("language")) != task.language
        ):
            raise ValueError(f"MDCO calibration task authority changed: {task.ordinal}")
        result[task.ordinal] = RewardTask(
            suite=task.suite,
            task_id=task.task_id,
            global_task_id=task.ordinal,
            split_role="train",
            language=task.language,
            problem_folder=str(row["problem_folder"]),
            bddl_file=str(bddl_file),
            bddl_bytes=int(bddl_bytes),
            bddl_sha256=None,
            horizon=SUITE_HORIZONS[task.suite],
        )
    expected = {int(task.ordinal) for task in tasks if task.fold_role == "fit"}
    if set(result) != expected:
        raise ValueError("MDCO calibration reward tasks do not cover fit90")
    return result


def plan_structured_calibration(
    args: Any,
    config: Mapping[str, Any],
    tasks: Sequence[Any],
    context: Any,
    *,
    start_task_visits: int,
    stop_after_task_visits: int,
) -> StructuredCalibrationState:
    pending = structured_calibration_requested(
        args,
        config,
        start_task_visits=start_task_visits,
        stop_after_task_visits=stop_after_task_visits,
    )
    assignments = (
        build_calibration_assignments(
            tasks,
            world_size=context.world_size,
            frame_stride=int(config["data"]["frame_stride"]),
            task_count=calibration_task_count(config, mode=args.mode),
        )
        if pending
        else tuple(() for _ in range(context.world_size))
    )
    return StructuredCalibrationState(
        pending=pending,
        assignments=assignments,
        reward_tasks={},
        processor=None,
        panel_requests=frozenset(),
    )


def prepare_structured_calibration(
    state: StructuredCalibrationState,
    *,
    config: Mapping[str, Any],
    tasks: Sequence[Any],
    source_config: Mapping[str, Any],
    tokenizer_path: Path,
    context: Any,
    support_bank: Any,
) -> StructuredCalibrationState:
    if not state.pending:
        return state
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    local_ordinals = state.assignments[context.rank]
    return replace(
        state,
        reward_tasks=load_stage1_reward_tasks(config, tasks),
        processor=processor,
        panel_requests=calibration_panel_requests(
            support_bank=support_bank,
            task_ordinals=local_ordinals,
            visit=int(config["structured_calibration"]["support_visit"]),
        ),
    )


def validate_structured_calibration(
    output_dir: Path, *, checkpoint_task_visits: int
) -> dict[str, Any]:
    value = read_json(output_dir / STRUCTURED_CALIBRATION_FILE)
    records = tuple(value.get("tasks", ()))
    assignments = tuple(
        tuple(int(item) for item in row) for row in value.get("assignments", ())
    )
    if (
        value.get("schema_version") != STRUCTURED_CALIBRATION_SCHEMA
        or value.get("status") != "complete_fit90_structured_calibration"
        or value.get("mode") != "formal"
        or int(value.get("applied_after_task_visits", -1)) != 540
        or checkpoint_task_visits < 540
        or int(value.get("task_count", -1)) != 90
        or value.get("task_weight") != "equal"
        or value.get("global_16d_estimator") is not False
        or int(value.get("optimizer_updates", -1)) != 1
        or sorted(item for row in assignments for item in row) != list(range(90))
        or [int(row.get("task_ordinal", -1)) for row in records] != list(range(90))
        or value.get("information_wall", {}).get("held5_reward_reads") != 0
        or value.get("information_wall", {}).get("validation_reward_reads") != 0
        or value.get("information_wall", {}).get("test_reward_reads") != 0
    ):
        raise ValueError("MDCO structured calibration authority changed")
    return dict(value)
