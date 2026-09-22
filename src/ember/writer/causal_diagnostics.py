"""Bounded Core/Procedure and auxiliary-gradient diagnostics for frozen Writers.

This module owns scientific measurements only.  It deliberately reuses the
canonical Writer decoder, functional credit, AdamW implementation, and rollout
adapter rather than adding a second training or evaluation path.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch

from ember.writer.data import FunctionalQueryDataset
from ember.writer.runtime import autocast
from ember.writer.stability_diagnostics import (
    LoadedWriter,
    combine_gradients,
    gradient_dot,
    optimizer_step,
    probe_one,
    snapshot_parent,
    restore_parent,
    writer_condition_gradient,
    writer_frozen_policy,
)


DIAGNOSTIC_SCHEMA = "ember_writer_causal_diagnostics_v1"
PATH_TASKS = (5, 7, 12, 14, 20, 25, 34, 37)
PATH_DONORS = {5: 7, 7: 5, 12: 14, 14: 12, 20: 25, 25: 20, 34: 37, 37: 34}
PATH_ARMS = ("CC", "CW", "WC", "WW", "CO")
PATH_ARM_SOURCES = {
    "CC": ("correct", "correct"),
    "CW": ("correct", "wrong"),
    "WC": ("wrong", "correct"),
    "WW": ("wrong", "wrong"),
    "CO": ("correct", "other"),
}
PATH_TRAJECTORY_CONDITIONS = ((5, 0), (37, 1))
GRADIENT_GROUPS = (
    "text_meta",
    "vl_meta",
    "action_meta",
    "semantic_core",
    "procedure",
    "compiler",
    "factor_heads",
)


@dataclass(frozen=True)
class PathCohort:
    label: str
    teacher_demo: int
    query_demo: int
    other_demo: int
    init_state_id: int


PATH_COHORTS = (
    PathCohort("A", 46, 47, 48, 0),
    PathCohort("B", 48, 49, 46, 1),
)


@dataclass(frozen=True)
class PathProbeRequest:
    task: int
    cohort: PathCohort
    fraction: float
    query_position: int
    noise_seed: int
    raw: Mapping[str, Any]


@dataclass
class EncodedPath:
    core: torch.Tensor
    valid_core: torch.Tensor
    procedure: torch.Tensor
    positions: torch.Tensor
    valid_frames: torch.Tensor
    trace: Mapping[str, torch.Tensor]


@dataclass
class CompiledPath:
    task: int
    suite: str
    suite_task_id: int
    cohort: PathCohort
    arm: str
    state: Mapping[str, torch.Tensor]
    core_source: EncodedPath
    procedure_source: EncodedPath
    fused: Mapping[str, torch.Tensor]
    cc_reconstruction_max_abs: float | None


def _stable_seed(*values: Any) -> int:
    text = ":".join(map(str, values)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") & ((1 << 63) - 1)


def require_path_task_contract(data) -> None:
    task_ids = tuple(int(value) for value in data.task_ids)
    if not set(PATH_TASKS) <= set(task_ids) or set(PATH_DONORS) != set(PATH_TASKS):
        raise ValueError("registered causal path tasks are absent from the current train authority")
    if any(data.tasks[task].suite != data.tasks[PATH_DONORS[task]].suite for task in PATH_TASKS):
        raise ValueError("causal path donor is outside its target suite")


class CausalRawInputCache:
    """Bounded raw-video cache keyed separately from the target-language condition."""

    def __init__(self, loaded: LoadedWriter, data, *, byte_limit: int = 1 << 30) -> None:
        if byte_limit <= 0:
            raise ValueError("causal raw-video cache needs a positive byte limit")
        self.loaded, self.data, self.byte_limit = loaded, data, int(byte_limit)
        self._entries: OrderedDict[tuple[int, int], tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], int]] = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def _video(self, task: int, demo: int) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        key = (int(task), int(demo))
        if key in self._entries:
            self.hits += 1
            frames, indices, size = self._entries.pop(key)
            self._entries[key] = (frames, indices, size)
            return frames, indices
        self.misses += 1
        frames, indices = self.data.load_videos(key[0], (key[1],))
        size = sum(value.numel() * value.element_size() for value in (*frames, *indices))
        if size <= self.byte_limit:
            while self._entries and self.bytes + size > self.byte_limit:
                self.bytes -= self._entries.popitem(last=False)[1][2]
            self._entries[key] = (frames, indices, size)
            self.bytes += size
        return frames, indices

    def condition(self, *, language_task: int, video_task: int, demo: int) -> tuple:
        if language_task not in self.data.tasks or video_task not in self.data.tasks:
            raise ValueError("causal path condition references an unknown train task")
        frames, indices = self._video(video_task, demo)
        return self.loaded.runtime.prepare(
            frames,
            indices,
            self.data.tasks[language_task].authority.language,
        )


def _encode(loaded: LoadedWriter, cache: CausalRawInputCache, *, language_task: int, video_task: int, demo: int) -> EncodedPath:
    condition = cache.condition(language_task=language_task, video_task=video_task, demo=demo)
    with torch.no_grad(), autocast(loaded.runtime.device):
        encoded, trace = loaded.runtime.state.writer.encode_task(
            loaded.runtime.policy,
            *condition,
            return_trace=True,
        )
    core, valid_core, procedure, positions, valid_frames, _attention = encoded
    return EncodedPath(core, valid_core, procedure, positions, valid_frames, trace)


def _state_max_abs(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if set(left) != set(right):
        raise ValueError("LoRA state keys differ in causal reconstruction check")
    return max(float((left[name].float() - right[name].float()).abs().max()) for name in left)


def select_path_sources(available: Mapping[str, EncodedPath], arm: str) -> tuple[EncodedPath, EncodedPath]:
    """Select complete Core and Procedure bundles; their masks/positions never split off."""
    if arm not in PATH_ARM_SOURCES or not set(PATH_ARM_SOURCES[arm]) <= set(available):
        raise ValueError("causal path source set is incomplete")
    core_label, procedure_label = PATH_ARM_SOURCES[arm]
    return available[core_label], available[procedure_label]


def compile_path(
    loaded: LoadedWriter,
    cache: CausalRawInputCache,
    *,
    task: int,
    cohort: PathCohort,
    arm: str,
    check_cc: bool = False,
    encodings: dict[tuple[int, int, int], EncodedPath] | None = None,
) -> CompiledPath:
    """Compile one arm while keeping exact language and interface masks coupled."""
    if arm not in PATH_ARM_SOURCES or task not in PATH_DONORS:
        raise ValueError("unknown registered causal path arm or task")
    core_label, procedure_label = PATH_ARM_SOURCES[arm]
    encoded = encodings if encodings is not None else {}

    def source(label: str) -> EncodedPath:
        video_task, demo = {
            "correct": (task, cohort.teacher_demo),
            "wrong": (PATH_DONORS[task], cohort.teacher_demo),
            "other": (task, cohort.other_demo),
        }[label]
        key = (task, video_task, demo)
        if key not in encoded:
            encoded[key] = _encode(
                loaded,
                cache,
                language_task=task,
                video_task=video_task,
                demo=demo,
            )
        return encoded[key]

    available = {label: source(label) for label in {core_label, procedure_label}}
    core_source, procedure_source = select_path_sources(available, arm)
    with torch.no_grad(), autocast(loaded.runtime.device):
        state, fused = loaded.runtime.state.writer.compile_encoded_task(
            core_source.core,
            core_source.valid_core,
            procedure_source.procedure,
            procedure_source.positions,
            procedure_source.valid_frames,
        )
        direct = loaded.runtime.compile(
            cache.condition(language_task=task, video_task=task, demo=cohort.teacher_demo)
        ) if check_cc and arm == "CC" else None
    return CompiledPath(
        task=task,
        suite=str(cache.data.tasks[task].suite),
        suite_task_id=int(cache.data.tasks[task].suite_task_id),
        cohort=cohort,
        arm=arm,
        state=state,
        core_source=core_source,
        procedure_source=procedure_source,
        fused=fused,
        cc_reconstruction_max_abs=None if direct is None else _state_max_abs(state, direct),
    )


def compile_path_panel(
    loaded: LoadedWriter,
    cache: CausalRawInputCache,
    *,
    arms: Sequence[str] = PATH_ARMS,
    check_cc: bool = False,
) -> dict[tuple[int, str, str], CompiledPath]:
    """Compile all registered path arms, reusing only frozen raw-video encodings."""
    if not arms or any(arm not in PATH_ARM_SOURCES for arm in arms):
        raise ValueError("causal path panel contains an unknown arm")
    panel, encodings = {}, {}
    for task in PATH_TASKS:
        for cohort in PATH_COHORTS:
            for arm in arms:
                panel[(task, cohort.label, arm)] = compile_path(
                    loaded,
                    cache,
                    task=task,
                    cohort=cohort,
                    arm=arm,
                    check_cc=check_cc,
                    encodings=encodings,
                )
    expected = len(PATH_TASKS) * len(PATH_COHORTS) * len(arms)
    if len(panel) != expected:
        raise ValueError("causal path panel key coverage changed")
    return panel


def rollout_path_states(
    panel: Mapping[tuple[int, str, str], CompiledPath],
    *,
    arm: str,
    asset: str,
) -> tuple[dict[str, Mapping[str, torch.Tensor]], dict[str, dict[str, Any]]]:
    """Adapt a precompiled arm to the existing official train-side rollout panel."""
    if arm not in PATH_ARM_SOURCES:
        raise ValueError("unknown causal rollout arm")
    states, evidence = {}, {}
    for task in PATH_TASKS:
        for cohort in PATH_COHORTS:
            compiled = panel[(task, cohort.label, arm)]
            key = f"{compiled.suite}:{compiled.suite_task_id}:{cohort.init_state_id}"
            states[key] = compiled.state
            evidence[key] = {
                "schema_version": DIAGNOSTIC_SCHEMA,
                "asset": asset,
                "arm": arm,
                "global_task_id": task,
                "suite": compiled.suite,
                "task_id": compiled.suite_task_id,
                "init_state_id": cohort.init_state_id,
                "teacher_demo": cohort.teacher_demo,
                "other_demo": cohort.other_demo,
                "donor_global_task_id": PATH_DONORS[task],
                "outcome_dependence": False,
                "checkpoint_selection_use": False,
                "test_use": False,
            }
    if len(states) != 16 or len(evidence) != 16:
        raise ValueError("causal rollout panel must contain eight tasks by two states")
    return states, evidence


def registered_path_probes(data) -> list[PathProbeRequest]:
    """Materialize the fixed 32 target-language action queries once per process."""
    require_path_task_contract(data)
    dataset = FunctionalQueryDataset(
        tuple(task.authority for task in data.tasks.values()),
        demo_indices=(47, 49),
        action_chunk_size=50,
        action_start_offset=1,
    )
    requests = []
    try:
        for task in PATH_TASKS:
            for cohort in PATH_COHORTS:
                rows = dataset.task_episode_rows[task][cohort.query_demo]
                for fraction in (0.25, 0.75):
                    position = min(len(rows) - 1, int(round((len(rows) - 1) * fraction)))
                    requests.append(PathProbeRequest(
                        task=task,
                        cohort=cohort,
                        fraction=fraction,
                        query_position=position,
                        noise_seed=_stable_seed("writer-causal-path", task, cohort.label, fraction),
                        raw=dataset[rows[position]],
                    ))
    finally:
        dataset.close()
    if len(requests) != 32:
        raise ValueError("causal path probe registration changed")
    return requests


def _tensor_delta(left: torch.Tensor, right: torch.Tensor, *, prefix: str) -> dict[str, Any]:
    left_norm = float(left.detach().float().norm())
    right_norm = float(right.detach().float().norm())
    result: dict[str, Any] = {
        f"{prefix}_norm": left_norm,
        f"{prefix}_reference_norm": right_norm,
        f"{prefix}_shape": list(left.shape),
        f"{prefix}_reference_shape": list(right.shape),
        f"{prefix}_shape_match": tuple(left.shape) == tuple(right.shape),
        f"{prefix}_relative_l2": None,
    }
    if left.shape == right.shape:
        result[f"{prefix}_relative_l2"] = float((left.float() - right.float()).norm() / max(right_norm, 1e-12))
    return result


def _state_delta(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], suffix: str, label: str) -> dict[str, Any]:
    names = tuple(sorted(name for name in left if suffix in name))
    if not names or names != tuple(sorted(name for name in right if suffix in name)):
        raise ValueError("causal LoRA factor state changed")
    difference = sum(float((left[name].float() - right[name].float()).square().sum()) for name in names)
    reference = sum(float(right[name].float().square().sum()) for name in names)
    return {
        f"lora_{label}_relative_l2": math.sqrt(difference) / max(math.sqrt(reference), 1e-12),
        f"lora_{label}_norm": math.sqrt(sum(float(left[name].float().square().sum()) for name in names)),
    }


def path_trace_delta(candidate: CompiledPath, reference: CompiledPath) -> dict[str, Any]:
    """Summarize observed paths; shape mismatches are recorded rather than coerced."""
    fused_left = torch.cat([value.detach().float().flatten() for value in candidate.fused.values()])
    fused_right = torch.cat([value.detach().float().flatten() for value in reference.fused.values()])
    result = {
        **_tensor_delta(candidate.core_source.trace["frame_evidence"], reference.core_source.trace["frame_evidence"], prefix="e"),
        **_tensor_delta(candidate.procedure_source.trace["horizon"], reference.procedure_source.trace["horizon"], prefix="h"),
        **_tensor_delta(candidate.core_source.core, reference.core_source.core, prefix="core"),
        **_tensor_delta(candidate.procedure_source.procedure, reference.procedure_source.procedure, prefix="procedure"),
        **_tensor_delta(fused_left, fused_right, prefix="fused"),
        **_state_delta(candidate.state, reference.state, "lora_A", "a"),
        **_state_delta(candidate.state, reference.state, "lora_B", "b"),
    }
    return result


def _action_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> float:
    left = torch.tensor(candidate["inference_first5_actions"], dtype=torch.float32)
    right = torch.tensor(reference["inference_first5_actions"], dtype=torch.float32)
    if left.shape != right.shape:
        raise ValueError("causal path action probe shapes differ")
    return float((left - right).square().mean())


def probe_path(
    loaded: LoadedWriter,
    compiled: CompiledPath,
    request: PathProbeRequest,
    *,
    reference: CompiledPath,
    reference_metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (compiled.task, compiled.cohort) != (request.task, request.cohort):
        raise ValueError("path probe does not match its compiled task condition")
    metric = probe_one(writer_frozen_policy(loaded, compiled.state), request.raw, seed=request.noise_seed)
    action_delta = 0.0 if reference_metric is None else _action_delta(metric, reference_metric)
    return {
        "asset": loaded.name,
        "arm": compiled.arm,
        "global_task_id": request.task,
        "suite": compiled.suite,
        "suite_task_id": compiled.suite_task_id,
        "cohort": request.cohort.label,
        "teacher_demo": request.cohort.teacher_demo,
        "query_demo": request.cohort.query_demo,
        "other_demo": request.cohort.other_demo,
        "donor_global_task_id": PATH_DONORS[request.task],
        "query_fraction": request.fraction,
        "query_position": request.query_position,
        "noise_seed": request.noise_seed,
        "cc_reconstruction_max_abs": compiled.cc_reconstruction_max_abs,
        "action_delta_mse_vs_cc": action_delta,
        **metric,
        **path_trace_delta(compiled, reference),
    }


def causal_parameter_group(name: str) -> str:
    """Return one mutually exclusive owner for every trainable Writer parameter."""
    markers = (
        ("text_meta", "text_meta_lora."),
        ("vl_meta", "vl_meta_lora."),
        ("action_meta", "action_meta_lora."),
        ("procedure", "procedure."),
        ("compiler", "compiler."),
        ("factor_heads", "factor_heads."),
    )
    matches = [label for label, marker in markers if name.startswith(marker) or f".{marker}" in name]
    if len(matches) > 1:
        raise ValueError(f"causal parameter belongs to multiple groups: {name}")
    if matches:
        return matches[0]
    if (name.startswith("semantic_encoder.") or name.startswith("semantic_core.")
            or ".semantic_encoder." in name or ".semantic_core." in name):
        return "semantic_core"
    raise ValueError(f"causal parameter lacks a registered group: {name}")


def validate_causal_parameter_groups(loaded: LoadedWriter) -> dict[str, int]:
    parameters = tuple(loaded.runtime.state.parameters())
    optimizer_parameters = tuple(parameter for group in loaded.optimizer.param_groups for parameter in group["params"])
    identities = tuple(id(parameter) for parameter in optimizer_parameters)
    if (len(parameters) != len(optimizer_parameters)
            or len(identities) != len(set(identities))
            or any(left is not right for left, right in zip(parameters, optimizer_parameters, strict=True))):
        raise ValueError("causal optimizer parameter identities are not a one-to-one Writer mapping")
    groups = {label: 0 for label in GRADIENT_GROUPS}
    for name in loaded.parameter_names:
        groups[causal_parameter_group(name)] += 1
    if not all(groups.values()):
        raise ValueError("causal Writer parameter group is empty")
    return groups


def _zero_gradient_like(loaded: LoadedWriter) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
                 for parameter in loaded.runtime.state.parameters())


def _add_gradient(destination: list[torch.Tensor], source: Sequence[torch.Tensor], scale: float = 1.0) -> None:
    if len(destination) != len(source):
        raise ValueError("causal gradients have different parameter counts")
    for current, value in zip(destination, source, strict=True):
        current.add_(value, alpha=float(scale))


def _scaled_gradient(source: Sequence[torch.Tensor], scale: float) -> tuple[torch.Tensor, ...]:
    return tuple(value.mul(float(scale)) for value in source)


def _mean_gradient(values: Sequence[Sequence[torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    if not values:
        raise ValueError("cannot average an empty causal gradient collection")
    result = [torch.zeros_like(value) for value in values[0]]
    for value in values:
        _add_gradient(result, value)
    return _scaled_gradient(result, 1.0 / len(values))


def _gradient_norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.float().square().sum()) for value in values))


def _gradient_group_stats(
    names: Sequence[str],
    q: Sequence[torch.Tensor],
    a: Sequence[torch.Tensor],
    joint: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    accum = {group: {"q": 0.0, "a": 0.0, "joint": 0.0, "dot": 0.0, "count": 0} for group in GRADIENT_GROUPS}
    for name, q_value, a_value, joint_value in zip(names, q, a, joint, strict=True):
        group = causal_parameter_group(name)
        cell = accum[group]
        cell["q"] += float(q_value.float().square().sum())
        cell["a"] += float(a_value.float().square().sum())
        cell["joint"] += float(joint_value.float().square().sum())
        cell["dot"] += float(q_value.float().flatten().dot(a_value.float().flatten()))
        cell["count"] += 1
    rows = []
    for group in GRADIENT_GROUPS:
        cell = accum[group]
        q_norm, a_norm = math.sqrt(cell["q"]), math.sqrt(cell["a"])
        rows.append({
            "record": "group",
            "group": group,
            "parameter_count": cell["count"],
            "q_grad_norm": q_norm,
            "a_grad_norm": a_norm,
            "joint_grad_norm": math.sqrt(cell["joint"]),
            "q_a_dot": cell["dot"],
            "q_a_cosine": cell["dot"] / max(q_norm * a_norm, 1e-20),
        })
    return rows


def measure_draw_gradients(loaded: LoadedWriter, data, cache, draw: Mapping[str, Any]) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]
]:
    """Get a native mixed-precision J VJP and its Q/A decomposition for one draw.

    The production Writer first adds the main and weighted auxiliary LoRA
    cotangents, then casts that joint cotangent into the native Writer VJP.
    Replaying the two Writer VJPs separately changes that BF16 rounding point.
    Therefore ``A`` is defined at Writer-parameter space as ``J - Q`` rather
    than as an independently rounded teaching-only VJP.
    """
    q, q_metrics = writer_condition_gradient(loaded, data, cache, draw, component="main")
    joint, joint_metrics = writer_condition_gradient(loaded, data, cache, draw, component="joint")
    if len(q) != len(joint):
        raise ValueError("causal native joint and main parameter counts differ")
    a = tuple(joint_value - q_value for q_value, joint_value in zip(q, joint, strict=True))
    return q, a, joint, {
        "main_loss": float(q_metrics["main_loss"]),
        "auxiliary_loss": float(joint_metrics["teaching_loss"]),
        "main_grad_norm": _gradient_norm(q),
        "auxiliary_grad_norm": _gradient_norm(a),
        "joint_grad_norm": _gradient_norm(joint),
    }


def measure_batch_gradients(loaded: LoadedWriter, data, cache, draws: Sequence[Mapping[str, Any]]) -> tuple[
    tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, float]
]:
    if len(draws) != 4:
        raise ValueError("causal microtrain update requires four registered tasks")
    values = [measure_draw_gradients(loaded, data, cache, draw) for draw in draws]
    q = _mean_gradient([value[0] for value in values])
    a = _mean_gradient([value[1] for value in values])
    joint = _mean_gradient([value[2] for value in values])
    metrics = {
        key: sum(value[3][key] for value in values) / len(values)
        for key in values[0][3]
    }
    return q, a, joint, metrics


def gradient_linearity_residual(
    q: Sequence[torch.Tensor], a: Sequence[torch.Tensor], joint: Sequence[torch.Tensor]
) -> float:
    """Check the native parameter-space Q + (J - Q) decomposition."""
    if len(q) != len(a) or len(q) != len(joint):
        raise ValueError("causal native gradient parameter counts differ")
    return max(float((q_value + a_value - joint_value).abs().max())
               for q_value, a_value, joint_value in zip(q, a, joint, strict=True))


def _collect_gradient_batches(loaded: LoadedWriter, data, cache, batches) -> tuple[list[Any], list[dict[str, float]], list[dict[str, Any]]]:
    per_batch: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = []
    batch_losses: list[dict[str, float]] = []
    rows: list[dict[str, Any]] = []
    for batch_index, draws in enumerate(batches, start=1):
        q_values, a_values, joint_values, losses = [], [], [], []
        for draw in draws:
            q, a, joint, metrics = measure_draw_gradients(loaded, data, cache, draw)
            q_values.append(q)
            a_values.append(a)
            joint_values.append(joint)
            losses.append(metrics)
        q_batch = _mean_gradient(q_values)
        a_batch = _mean_gradient(a_values)
        joint_batch = _mean_gradient(joint_values)
        per_batch.append((q_batch, a_batch, joint_batch))
        batch_losses.append({key: sum(value[key] for value in losses) / len(losses) for key in losses[0]})
        rows.extend({
            "record": "draw",
            "window": "B4" if batch_index == 1 else "B36_component",
            "batch_index": batch_index,
            "global_task_id": int(draw["task"]),
            "occurrence": int(draw["occurrence"]),
            "teacher_demo": int(draw["video_demos"][0]),
            **metrics,
        } for draw, metrics in zip(draws, losses, strict=True))
    return per_batch, batch_losses, rows


def _window_aggregates(per_batch, batch_losses) -> dict[str, dict[str, Any]]:
    b4_q, b4_a, b4_joint = per_batch[0]
    b36_q = _mean_gradient([item[0] for item in per_batch])
    b36_a = _mean_gradient([item[1] for item in per_batch])
    b36_joint = _mean_gradient([item[2] for item in per_batch])
    return {
        "B4": {"q": b4_q, "a": b4_a, "joint": b4_joint, "draw_count": 4, "losses": batch_losses[0]},
        "B36": {
            "q": b36_q,
            "a": b36_a,
            "joint": b36_joint,
            "draw_count": 36,
            "losses": {key: sum(value[key] for value in batch_losses) / len(batch_losses) for key in batch_losses[0]},
        },
    }


def _gradient_window_rows(loaded: LoadedWriter, windows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for window, value in windows.items():
        losses = {key: metric for key, metric in value["losses"].items() if key != "joint_grad_norm"}
        for row in _gradient_group_stats(loaded.parameter_names, value["q"], value["a"], value["joint"]):
            rows.append({"window": window, "batch_index": None, **row, **losses})
    return rows


def _batch_gram_rows(per_batch) -> list[dict[str, Any]]:
    rows = []
    for left_index, (_left_q, _left_a, left_joint) in enumerate(per_batch, start=1):
        for right_index, (_right_q, _right_a, right_joint) in enumerate(per_batch[:left_index], start=1):
            dot = gradient_dot(left_joint, right_joint)
            rows.append({
                "record": "batch_gram",
                "window": "B36",
                "batch_index": left_index,
                "other_batch_index": right_index,
                "component": "joint",
                "dot": dot,
                "cosine": dot / max(_gradient_norm(left_joint) * _gradient_norm(right_joint), 1e-20),
            })
    return rows


def measure_gradient_windows(loaded: LoadedWriter, data, cache, batches: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]]
]:
    """Measure B4/B36 once per registered draw and return aggregate vectors plus rows."""
    if len(batches) != 9 or any(len(batch) != 4 for batch in batches):
        raise ValueError("causal B36 requires nine complete four-task batches")
    per_batch, batch_losses, rows = _collect_gradient_batches(loaded, data, cache, batches)
    windows = _window_aggregates(per_batch, batch_losses)
    rows.extend(_gradient_window_rows(loaded, windows))
    rows.extend(_batch_gram_rows(per_batch))
    return windows, rows


def _parameter_snapshot(loaded: LoadedWriter) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().float().cpu().clone() for parameter in loaded.runtime.state.parameters())


def _group_displacements(loaded: LoadedWriter, before: Sequence[torch.Tensor]) -> dict[str, float]:
    sums = {group: 0.0 for group in GRADIENT_GROUPS}
    for name, parameter, prior in zip(loaded.parameter_names, loaded.runtime.state.parameters(), before, strict=True):
        sums[causal_parameter_group(name)] += float((parameter.detach().float().cpu() - prior).square().sum())
    return {group: math.sqrt(value) for group, value in sums.items()}


def _set_scaled_displacement(loaded: LoadedWriter, before: Sequence[torch.Tensor], scale: float) -> None:
    for parameter, prior in zip(loaded.runtime.state.parameters(), before, strict=True):
        displacement = parameter.detach().float().cpu() - prior
        parameter.data.copy_((prior + float(scale) * displacement).to(parameter))


def apply_virtual_candidate(
    loaded: LoadedWriter,
    *,
    q: Sequence[torch.Tensor],
    a: Sequence[torch.Tensor],
    joint: Sequence[torch.Tensor] | None = None,
    candidate: str,
    lr: float,
) -> dict[str, Any]:
    """Leave ``loaded`` at J/Q/M/Z, preserving the required AdamW semantics."""
    if candidate not in {"J", "Q", "M", "Z"}:
        raise ValueError("unknown causal virtual candidate")
    if joint is None:
        joint = tuple(q_value + a_value for q_value, a_value in zip(q, a, strict=True))
    if len(q) != len(a) or len(q) != len(joint):
        raise ValueError("causal candidate gradient parameter counts differ")
    parent = snapshot_parent(loaded)
    before = _parameter_snapshot(loaded)
    if candidate == "Q":
        result = optimizer_step(loaded, gradients=(q,), weights=(1.0,), lr=lr)
        alpha = None
    elif candidate == "Z":
        result = optimizer_step(loaded, gradients=(_zero_gradient_like(loaded),), weights=(1.0,), lr=lr)
        alpha = None
    elif candidate == "J":
        result = optimizer_step(loaded, gradients=(joint,), weights=(1.0,), lr=lr)
        alpha = None
    else:
        q_result = optimizer_step(loaded, gradients=(q,), weights=(1.0,), lr=lr)
        q_norm = float(q_result["parameter_displacement_norm"])
        restore_parent(loaded, parent)
        before = _parameter_snapshot(loaded)
        result = optimizer_step(loaded, gradients=(joint,), weights=(1.0,), lr=lr)
        joint_norm = float(result["parameter_displacement_norm"])
        if joint_norm <= 0:
            raise ValueError("causal M candidate has zero joint AdamW displacement")
        alpha = q_norm / joint_norm
        _set_scaled_displacement(loaded, before, alpha)
        result = {**result, "q_parameter_displacement_norm": q_norm}
    return {
        "candidate": candidate,
        "lr": float(lr),
        "scale_alpha": alpha,
        **result,
        **{f"displacement_{group}": value for group, value in _group_displacements(loaded, before).items()},
    }
