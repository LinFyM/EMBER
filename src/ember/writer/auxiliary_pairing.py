"""Opt-in config and event preflight for the matched auxiliary-pairing run."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


REFERENCE_CONFIG = "configs/libero_24_8_8_coverage_v1/writer.json"
DESIGN = "docs/writer_auxiliary_pairing_design.md"
DECLARATION = {
    "kind": "coverage_auxiliary_pairing_fresh_v1",
    "reference_config": REFERENCE_CONFIG,
    "reference_commit": "7f62c7b70e608e0dc97d06cb7004a698fd5b5fea",
    "initialization": "fresh",
    "sole_variable": "data.teaching_episode",
}
PREFLIGHT_SCHEMA = "ember_auxiliary_pairing_preflight_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _expected_candidate(reference: Mapping[str, Any]) -> dict[str, Any]:
    expected = deepcopy(dict(reference))
    expected["data"]["teaching_episode"] = "cross_episode"
    expected["design"] = DESIGN
    expected["experiment"] = DECLARATION
    return expected


def require_matched_config(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    """Permit exactly the registered fresh cross-episode auxiliary variant."""
    if (reference.get("data", {}).get("teaching_episode") != "same_video"
            or reference.get("data", {}).get("maximum_updates") is not None
            or reference.get("training_control", {}).get("validation_interval") != 200):
        raise ValueError("auxiliary pairing requires the registered dynamic same-video reference")
    if dict(candidate) != _expected_candidate(reference):
        raise ValueError("auxiliary pairing may change only its declaration, design record, and teaching episode")


def declared_dynamic_episode(config: Mapping[str, Any], repository_root: Path) -> str:
    """Keep the historical dynamic default closed unless this exact variant opts in."""
    episode = config.get("data", {}).get("teaching_episode")
    if episode == "same_video":
        if "experiment" in config:
            raise ValueError("same-video dynamic training cannot carry an auxiliary-pairing declaration")
        return episode
    if episode != "cross_episode":
        raise ValueError("dynamic Writer teaching episode is unregistered")
    require_matched_config(config, _read_json(repository_root / REFERENCE_CONFIG))
    return episode


def _event_map(data: Any, updates: int) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, int], dict[str, Any]]]:
    if type(updates) is not int or updates <= 0 or updates > len(data._groups):
        raise ValueError("event preflight needs a positive registered update prefix")
    groups, events = [], {}
    for step, group in enumerate(data._groups[:updates], 1):
        group_events = tuple(data._events[index] for index in group)
        if len(group_events) != 4 or len({event["task"] for event in group_events}) != 4:
            raise ValueError("every preflight update must retain four distinct tasks")
        groups.append(tuple(event["task"] for event in group_events))
        for event in group_events:
            key = (step, event["task"])
            if key in events:
                raise ValueError("event preflight generated a duplicate task update")
            events[key] = event
    return tuple(groups), events


def _load_events(
    asset_root: Path, data_config: Mapping[str, Any], updates: int,
) -> tuple[tuple[tuple[int, ...], ...], dict[tuple[int, int], dict[str, Any]], dict[int, tuple[int, ...]]]:
    from ember.writer.learning_data import WriterTrainingData

    data = WriterTrainingData(asset_root, data_config, camera_view="agentview", planned_updates=updates)
    try:
        groups, events = _event_map(data, updates)
        lengths = {task: tuple(item.episode_lengths) for task, item in data.tasks.items()}
        return groups, events, lengths
    finally:
        data.close()


def _require_matching_main_event(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    without_teaching = lambda event: {key: value for key, value in event.items() if key != "teaching"}
    if without_teaching(left) != without_teaching(right):
        raise ValueError("teacher or main-query event differs from the reference")
    old, new = left["teaching"], right["teaching"]
    if old["policy_rng_seed"] != new["policy_rng_seed"]:
        raise ValueError("auxiliary noise seed differs from the reference")
    if old["policy_random_batch_size"] != new["policy_random_batch_size"]:
        raise ValueError("auxiliary noise batch differs from the reference")


def _cross_episode_payload(teaching: Mapping[str, Any], teacher: int) -> tuple[int, list[int]]:
    demos, frames = teaching.get("action_demos"), teaching.get("action_frames")
    if not isinstance(demos, list) or not isinstance(frames, list) or len(demos) != 7 or len(frames) != 7:
        raise ValueError("auxiliary event must contain seven matched episode queries")
    if len(set(demos)) != 1 or type(demos[0]) is not int:
        raise ValueError("auxiliary queries must share one explicit episode")
    demo = demos[0]
    if demo not in range(46) or demo == teacher or teaching.get("episode_relation") != "cross_episode":
        raise ValueError("auxiliary episode must be a legal same-task non-teacher episode")
    return demo, frames


def _validate_teaching_frames(
    teaching: Mapping[str, Any], *, frames: list[int], task: int, demo: int,
    lengths: Mapping[int, tuple[int, ...]],
) -> bool:
    length = lengths[task][demo]
    legal = range(0, length - 5, 5)
    replacement = len(legal) < 7
    if teaching.get("sampling_with_replacement") is not replacement:
        raise ValueError("auxiliary replacement declaration differs from its legal frame pool")
    if any(type(frame) is not int or frame not in legal for frame in frames):
        raise ValueError("auxiliary query lacks five real future actions")
    if teaching.get("action_start_indices") != [frame + 1 for frame in frames]:
        raise ValueError("auxiliary action alignment changed")
    if not replacement and len(set(frames)) != 7:
        raise ValueError("auxiliary frames unexpectedly use replacement")
    return replacement


def _validate_cross_episode_teaching(
    teaching: Mapping[str, Any], *, teacher: int, task: int, lengths: Mapping[int, tuple[int, ...]],
) -> bool:
    demo, frames = _cross_episode_payload(teaching, teacher)
    return _validate_teaching_frames(teaching, frames=frames, task=task, demo=demo, lengths=lengths)


def audit_events(
    asset_root: Path, reference: Mapping[str, Any], candidate: Mapping[str, Any], *, updates: int = 1800,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    """Replay the production sampler on real metadata and check its actual event contract."""
    require_matched_config(candidate, reference)
    before_groups, before, before_lengths = _load_events(asset_root, reference["data"], updates)
    after_groups, after, after_lengths = _load_events(asset_root, candidate["data"], updates)
    if before_groups != after_groups or before_lengths != after_lengths or set(before) != set(after):
        raise ValueError("candidate changed the registered task/event metadata")
    replacements = 0
    for key, left in before.items():
        right = after[key]
        _require_matching_main_event(left, right)
        replacements += _validate_cross_episode_teaching(
            right["teaching"], teacher=right["teacher_demo"], task=right["task"], lengths=after_lengths,
        )
    return ({"updates": updates, "conditions": len(before), "teacher_main_events_equal": True,
             "auxiliary_noise_equal": True, "auxiliary_teacher_excluded": True,
             "future_five_actions_valid": True, "auxiliary_replacement_conditions": replacements,
             "basis": "production_sampler_replay_from_real_task_metadata"}, before)


def _require_exposure_matches(row: Mapping[str, Any], event: Mapping[str, Any]) -> None:
    if row.get("video_demos") != [event["teacher_demo"]]:
        raise ValueError("recorded teacher differs from deterministic replay")
    fields = ("occurrence", "query_seed", "action_demos", "action_frames", "action_start_indices", "policy_rng_seed")
    if any(row.get(field) != event[field] for field in fields):
        raise ValueError("recorded main event differs from deterministic replay")
    teaching = event["teaching"]
    if row.get("teaching_policy_rng_seed") != teaching["policy_rng_seed"]:
        raise ValueError("recorded auxiliary noise differs from deterministic replay")
    if row.get("teaching_action_demos") != teaching["action_demos"]:
        raise ValueError("recorded auxiliary episodes differ from deterministic replay")
    if row.get("teaching_action_frames") != teaching["action_frames"]:
        raise ValueError("recorded auxiliary frames differ from deterministic replay")


def audit_actual_exposures(path: Path, expected: Mapping[tuple[int, int], Mapping[str, Any]]) -> int:
    """Bind the deterministic replay to all archived same-video exposures in scope."""
    seen, last_step = set(), max(step for step, _ in expected)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            step, task = row.get("step"), row.get("task")
            if type(step) is not int or type(task) is not int:
                raise ValueError("reference exposure row lacks a valid step/task key")
            if step > last_step:
                continue
            key = (step, task)
            if key not in expected or key in seen:
                raise ValueError("reference exposures contain an unexpected or duplicate task update")
            _require_exposure_matches(row, expected[key])
            seen.add(key)
    if seen != set(expected):
        raise ValueError("reference exposures do not cover every required update")
    return len(seen)


def run_preflight(
    *, asset_root: Path, reference_exposures: Path, candidate: Mapping[str, Any], reference: Mapping[str, Any],
    updates: int,
) -> dict[str, Any]:
    """Run the only new CPU gate; source/model execution remains untouched."""
    audit, expected = audit_events(asset_root, reference, candidate, updates=updates)
    audit["actual_reference_exposures_checked"] = True
    audit["actual_reference_rows"] = audit_actual_exposures(reference_exposures, expected)
    source = asset_root / candidate["source"]["checkpoint"]
    if not source.is_dir():
        raise FileNotFoundError(f"canonical Source checkpoint is unavailable: {source}")
    return {"schema_version": PREFLIGHT_SCHEMA, "experiment": DECLARATION, "event_alignment": audit,
            "only_scientific_difference": "data.teaching_episode", "model_or_loss_code_changed": False,
            "source_checkpoint_present": True, "gpu_execution_tested": False, "resources_checked": False,
            "training_started": False,
            "next": "Use the existing coverage controller after formal resource preflight."}
