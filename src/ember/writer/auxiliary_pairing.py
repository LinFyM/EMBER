"""Opt-in contract and CPU decisions for the matched cross-episode Writer run.

No policy execution, optimizer changes, or environment launches occur here.
The existing sampler, loss implementation and early-stopping rules remain the
source of truth; this module binds the one additional registered experiment.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from ember.early_stopping import validation_decision


REFERENCE_CONFIG = "configs/libero_24_8_8_coverage_v1/writer.json"
REFERENCE_COMMIT = "7f62c7b70e608e0dc97d06cb7004a698fd5b5fea"
DECLARATION = {
    "kind": "coverage_auxiliary_pairing_fresh_v1",
    "reference_config": REFERENCE_CONFIG,
    "reference_commit": REFERENCE_COMMIT,
    "initialization": "fresh",
    "sole_variable": "data.teaching_episode",
}
INTERVAL = 200
ORIGINAL_BEST = 117
MTBC_BEST = 155
# Only documentation/provenance can differ in addition to the registered leaf.
_METADATA_KEYS = frozenset({"design", "evidence", "experiment", "status"})


def _scientific_view(config: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy({key: value for key, value in config.items() if key not in _METADATA_KEYS})


def require_matched_config(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    """Reject unregistered changes, including low-LR or parent continuations."""
    if candidate.get("experiment") != DECLARATION:
        raise ValueError("cross-episode coverage requires its explicit fresh experiment declaration")
    if (reference.get("data", {}).get("teaching_episode") != "same_video"
            or reference.get("data", {}).get("maximum_updates") is not None
            or not reference.get("training_control")):
        raise ValueError("auxiliary-pairing reference must be the original dynamic same-video config")
    if any(key in candidate for key in ("continuation", "phase_continuation")):
        raise ValueError("this experiment permits fresh training or its own exact-resume only")
    expected = _scientific_view(reference)
    expected["data"]["teaching_episode"] = "cross_episode"
    if _scientific_view(candidate) != expected:
        raise ValueError("auxiliary-pairing scientific config may change only data.teaching_episode")
    before = reference.get("evidence", {}).get("qualification", {})
    after = candidate.get("evidence", {}).get("qualification", {})
    keys = ("K", "conditions", "seed", "selection_mode", "state_count", "tasks", "video_pool")
    if any(after.get(key) != before.get(key) for key in keys):
        raise ValueError("auxiliary-pairing validation panel or video schedule changed")
    if candidate.get("evidence", {}).get("test_use") is not False:
        raise ValueError("this experiment does not authorize Test")
    profile = candidate.get("evidence", {}).get("profile_registration", {})
    original_profile = reference.get("evidence", {}).get("profile_registration", {})
    for key in ("world_size", "native_frame_chunk", "policy_microbatches"):
        if profile.get(key) != original_profile.get(key):
            raise ValueError(f"matched reference physical profile changed: {key}")
    diagnostic = candidate.get("evidence", {}).get("supervised_validation", {})
    if diagnostic != reference.get("evidence", {}).get("supervised_validation", {}):
        raise ValueError("do not change the registered frozen training diagnostic policy")


def declared_dynamic_episode(config: Mapping[str, Any], repository_root: Path) -> str:
    """Keep historical defaults; opt in only to the fully matched new variant."""
    if "experiment" not in config:
        return "same_video"
    with (repository_root / REFERENCE_CONFIG).open(encoding="utf-8") as handle:
        reference = json.load(handle)
    require_matched_config(config, reference)
    return "cross_episode"


def _metadata_sampler(data_config: Mapping[str, Any], lengths: Mapping[int, Sequence[int]],
                      updates: int, sampler_type=None):
    """Replay existing metadata-only event methods without opening HDF5 arrays."""
    if sampler_type is None:
        from ember.writer.learning_data import WriterTrainingData
        sampler_type = WriterTrainingData
    if type(updates) is not int or updates <= 0:
        raise ValueError("metadata replay needs a positive update count")
    obj = sampler_type.__new__(sampler_type)
    obj.config = deepcopy(dict(data_config))
    obj._validate_config()
    obj.seed = data_config["seed"]
    obj.sampler_seed = data_config["sampler_seed"]
    obj.teacher_video_seed = data_config["teacher_video_seed"]
    obj.task_ids = tuple(sorted(data_config["task_ids"]))
    if len(obj.task_ids) != 36 or set(obj.task_ids) != set(lengths):
        raise ValueError("metadata must cover exactly the registered 36 Train tasks")
    if any(len(lengths[t]) != 50 or any(type(n) is not int or n < 6 for n in lengths[t])
           for t in obj.task_ids):
        raise ValueError("each task needs 50 valid episode lengths")
    obj.tasks = {task: SimpleNamespace(episode_lengths=tuple(lengths[task])) for task in obj.task_ids}
    obj.dynamic, obj.maximum_updates = True, updates
    obj.round_updates = len(obj.task_ids) // 4
    obj.rounds = (updates + obj.round_updates - 1) // obj.round_updates
    obj.generated_updates = obj.rounds * obj.round_updates
    obj.video_pool = tuple(data_config["video_demos"])
    obj.action_pool = tuple(data_config["action_demos"])
    obj.diagnostic_pool = tuple(data_config["diagnostic_action_demos"])
    obj.held_video_pool = tuple(data_config["held_video_demos"])
    obj._groups, obj._events = obj._build_groups(), obj._build_events()
    return obj


def audit_events(reference: Mapping[str, Any], candidate: Mapping[str, Any],
                 lengths: Mapping[int, Sequence[int]], *, updates: int = 1800,
                 sampler_type=None) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    """Compare real generator outputs; return reference events for log checking."""
    require_matched_config(candidate, reference)
    old = _metadata_sampler(reference["data"], lengths, updates, sampler_type)
    new = _metadata_sampler(candidate["data"], lengths, updates, sampler_type)
    if old._groups != new._groups:
        raise ValueError("registered logical task batches differ")
    planned = {}
    replacements = 0
    for step, group in enumerate(old._groups[:updates], 1):
        for index in group:
            left, right = old._events[index], new._events[index]
            if {k: v for k, v in left.items() if k != "teaching"} != {
                k: v for k, v in right.items() if k != "teaching"
            }:
                raise ValueError(f"teacher or main-query event differs at update {step}")
            old_aux, aux = left["teaching"], right["teaching"]
            if (aux["policy_rng_seed"] != old_aux["policy_rng_seed"]
                    or aux["policy_random_batch_size"] != old_aux["policy_random_batch_size"]):
                raise ValueError("auxiliary noise stream changed")
            demos, frames = aux["action_demos"], aux["action_frames"]
            if (len(demos) != 7 or len(frames) != 7 or len(set(demos)) != 1
                    or demos[0] not in new.action_pool or demos[0] == right["teacher_demo"]
                    or aux["episode_relation"] != "cross_episode"):
                raise ValueError("auxiliary queries must use one legal same-task non-teacher episode")
            length = lengths[right["task"]][demos[0]]
            if any(type(p) is not int or p < 0 or p % 5 or p + 5 >= length for p in frames):
                raise ValueError("auxiliary query lacks five real future actions")
            if aux["action_start_indices"] != [p + 1 for p in frames]:
                raise ValueError("auxiliary action alignment changed")
            legal_count = len(range(0, length - 5, 5))
            if bool(aux["sampling_with_replacement"]) != (legal_count < 7):
                raise ValueError("auxiliary replacement declaration changed")
            if legal_count >= 7 and len(set(frames)) != 7:
                raise ValueError("auxiliary frames unexpectedly sampled with replacement")
            replacements += int(aux["sampling_with_replacement"])
            planned[(step, left["task"])] = left
    return ({"updates": updates, "conditions": 4 * updates,
             "teacher_main_events_equal": True, "auxiliary_noise_equal": True,
             "auxiliary_teacher_excluded": True, "future_five_actions_valid": True,
             "auxiliary_replacement_conditions": replacements,
             "basis": "canonical_sampler_replay_from_manifest_metadata",
             "actual_reference_exposures_checked": False}, planned)


def audit_actual_exposures(path: Path, expected: Mapping[tuple[int, int], Mapping[str, Any]]) -> int:
    """Verify stored reference events as well as reconstructed deterministic plans."""
    seen = set()
    last_step = max(step for step, _ in expected)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["step"] > last_step:
                continue
            key = (row["step"], row["task"])
            if key not in expected or key in seen:
                raise ValueError("reference exposure rows contain unexpected/duplicate task updates")
            event = expected[key]
            if row["video_demos"] != [event["teacher_demo"]]:
                raise ValueError("recorded teacher differs from deterministic replay")
            for field in ("occurrence", "query_seed", "action_demos", "action_frames",
                          "action_start_indices", "policy_rng_seed"):
                if row[field] != event[field]:
                    raise ValueError(f"recorded reference event differs at {key}: {field}")
            if (row["teaching_policy_rng_seed"] != event["teaching"]["policy_rng_seed"]
                    or row["teaching_action_demos"] != event["teaching"]["action_demos"]
                    or row["teaching_action_frames"] != event["teaching"]["action_frames"]):
                raise ValueError("recorded same-video auxiliary event differs from replay")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("reference exposure log does not cover the required updates")
    return len(seen)


def auxiliary_validation_decision(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reuse scientific rules, then apply the registered unattended resource stop."""
    decision = validation_decision(history, interval=INTERVAL)
    scores = [row["successes"] for row in history]
    # First occurrence of the final maximum is its last *strict* record update.
    no_record_nodes = len(scores) - scores.index(max(scores)) - 1
    decision["nodes_without_strict_new_best"] = no_record_nodes
    decision["resource_review"] = False
    if not decision["stop"] and no_record_nodes >= 6:
        decision.update(stop=True, reason="resource_review_no_progress", resource_review=True)
    return decision


def measurement_arms(best_correct: int) -> list[str]:
    if type(best_correct) is not int or not 0 <= best_correct <= 400:
        raise ValueError("correct successes must be an integer in [0, 400]")
    if best_correct <= ORIGINAL_BEST:
        return []
    arms = ["same_task_other", "cross_suite_wrong"]
    if best_correct > MTBC_BEST:
        arms += ["shuffled", "reversed"]
    return arms


def selection_plan(history: Sequence[Mapping[str, Any]],
                   other_successes: Mapping[int, int] | None = None) -> dict[str, Any]:
    """Only complete stopped histories select; controls never choose checkpoints."""
    decision = auxiliary_validation_decision(history)
    if not decision["stop"]:
        return {"status": "continue_training", "next_stop_after_step": history[-1]["step"] + INTERVAL,
                "decision": decision, "selected_macro": None, "controls_required": []}
    tied = decision["tied_best_steps"]
    supplied = dict(other_successes or {})
    if (set(supplied) - set(tied) or any(type(k) is not int or type(v) is not int or not 0 <= v <= 400
                                       for k, v in supplied.items())):
        raise ValueError("other scores must be valid results for exactly the tied correct candidates")
    missing = [step for step in tied if step not in supplied] if len(tied) > 1 else []
    if missing:
        return {"status": "awaiting_tied_other400", "decision": decision,
                "required_other_steps": missing, "selected_macro": None, "controls_required": []}
    selected = min(tied, key=lambda step: (-supplied[step], step)) if len(tied) > 1 else tied[0]
    return {"status": "selected", "decision": decision, "selected_macro": selected,
            "correct_successes": decision["best_successes"],
            "other_successes": {str(k): supplied[k] for k in sorted(supplied)},
            "controls_required": measurement_arms(decision["best_successes"]),
            "replace_original_selected": decision["best_successes"] > ORIGINAL_BEST,
            "test_allowed": False, "ft_allowed": False, "rl_allowed": False}
