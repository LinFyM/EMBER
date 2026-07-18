"""Pure protocol and statistics for the action-hidden RGB diagnostic."""

from __future__ import annotations

import copy
import hashlib
import tomllib
from pathlib import Path
from typing import Any

import numpy as np


class VideoInformationProbeError(RuntimeError):
    """Raised when the frozen RGB-only diagnostic contract is violated."""


EXPECTED_CONDITIONS = [
    "ordered_full",
    "reversed",
    "shuffled",
    "first_frame",
    "last_frame",
    "static_temporal_median",
    "drop_last_20_percent",
]


def _validate_demo_partition(spec: dict[str, Any]) -> None:
    support = spec.get("support_demo_indices")
    query = spec.get("query_demo_indices")
    reserved = spec.get("reserved_demo_indices")
    if support != list(range(24)) or query != list(range(24, 48)):
        raise VideoInformationProbeError("support/query demonstration split changed")
    if reserved != [48, 49]:
        raise VideoInformationProbeError("reserved demonstration rows changed")
    if set(support) & set(query) or set(support + query) & set(reserved):
        raise VideoInformationProbeError("demonstration partitions overlap")


def _validate_encoder(spec: dict[str, Any]) -> None:
    encoder = spec.get("encoder", {})
    expected = {
        "contract_key": "smolvlm_constructor_dependency",
        "repo_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        "weight_sha256": "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3",
        "weight_bytes": 2029990624,
        "dtype": "bfloat16",
        "feature": "final_nonpadding_causal_context_hidden_state",
        "task_language_visible": False,
        "task_id_visible": False,
        "trajectory_length_visible": False,
    }
    for field, value in expected.items():
        if encoder.get(field) != value:
            raise VideoInformationProbeError(f"frozen encoder field changed: {field}")
    if not isinstance(encoder.get("prompt"), str) or not encoder["prompt"]:
        raise VideoInformationProbeError("neutral encoder prompt is missing")


def _validate_thresholds(spec: dict[str, Any]) -> None:
    expected = {
        "minimum_ordered_balanced_accuracy": 0.80,
        "minimum_bidirectional_query_pair_fraction": 0.80,
        "minimum_wrong_video_specificity": 0.80,
        "minimum_ordered_vs_first_frame_gap": 0.20,
        "minimum_ordered_vs_static_median_gap": 0.20,
        "minimum_temporal_order_gap": 0.10,
        "last_frame_full_ratio_limit": 0.95,
        "drop_last_20_minimum_retention": 0.90,
    }
    if spec.get("thresholds") != expected:
        raise VideoInformationProbeError("predeclared video thresholds changed")


def _validate_fixed_identity(spec: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "surface": "libero90_source_action_hidden_video_information",
        "task_suite": "libero_90",
        "task_ids": [3, 4],
        "source_task_ids": [3, 4],
        "scene_id": "KITCHEN_SCENE10",
        "camera_dataset": "obs/agentview_rgb",
        "input_height": 128,
        "input_width": 128,
        "input_channels": 3,
        "frame_count": 16,
        "end_exclusion_frames": 1,
        "frame_sampler": "uniform_rint_inclusive",
        "standardized_video_fps": 1,
        "gallery_demo_index": 24,
        "conditions": EXPECTED_CONDITIONS,
        "shuffle_seed": 20260718,
    }
    for field, value in expected.items():
        if spec.get(field) != value:
            raise VideoInformationProbeError(f"frozen video field changed: {field}")
    if spec.get("source_pair_config_sha256") != (
        "5155374dcaf394db6e50ed818d80dd8303764a9e5e010be33039e323044cb2ea"
    ):
        raise VideoInformationProbeError("source pair authority hash changed")
    if spec.get("split_seal_sha256") != (
        "9f5bc62e15e2cb07887e97bc98630a3f527ac6b5e253f41c203cf37459568428"
    ):
        raise VideoInformationProbeError("split seal hash changed")


def _validate_readout_and_resources(spec: dict[str, Any]) -> None:
    _validate_demo_partition(spec)
    _validate_encoder(spec)
    _validate_thresholds(spec)
    readout = spec.get("readout", {})
    if readout.get("kind") != "balanced_binary_dual_ridge":
        raise VideoInformationProbeError("primary readout changed")
    if readout.get("ridge_lambda") != 1.0 or readout.get("hyperparameter_search") is not False:
        raise VideoInformationProbeError("readout fitting contract changed")
    if readout.get("feature_l2_normalization") is not True or readout.get("support_feature_centering") is not True:
        raise VideoInformationProbeError("readout normalization contract changed")
    if readout.get("primary_metric") != "query_balanced_accuracy" or readout.get("confidence_level") != 0.95:
        raise VideoInformationProbeError("readout metric contract changed")
    if readout.get("bootstrap_samples") != 10000 or readout.get("bootstrap_seed") != 20260718:
        raise VideoInformationProbeError("bootstrap contract changed")
    resources = spec.get("resources", {})
    if resources.get("gpu_count") != 1 or resources.get("batch_size") != 48:
        raise VideoInformationProbeError("single-GPU batch contract changed")
    if resources.get("minimum_gpu_headroom_gib") != 10:
        raise VideoInformationProbeError("GPU headroom contract changed")
    if spec["encoder"].get("same_batch_repeat_atol") != 1e-7:
        raise VideoInformationProbeError("encoder repeat tolerance changed")


def validate_video_spec(spec: dict[str, Any]) -> None:
    _validate_fixed_identity(spec)
    _validate_readout_and_resources(spec)
    boundary = spec.get("claim_boundary", {})
    if boundary.get("gate_decision_authorized") is not False or boundary.get("writer_authorized") is not False:
        raise VideoInformationProbeError("diagnostic cannot authorize Gate -1 or Writer")


def load_video_spec(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    validate_video_spec(spec)
    return spec


def load_video_recovery_spec(path: Path, base_config_path: Path) -> dict[str, Any]:
    """Load the sole bounded representation recovery over the frozen base spec."""

    with path.open("rb") as handle:
        recovery = tomllib.load(handle)
    base_sha = hashlib.sha256(base_config_path.read_bytes()).hexdigest()
    expected_flags = {
        "schema_version": 1,
        "surface": "libero90_source_action_hidden_video_representation_recovery1",
        "recovery_class": "representation",
        "base_config_sha256": base_sha,
        "thresholds_unchanged": True,
        "task_demo_split_unchanged": True,
        "conditions_unchanged": True,
        "readout_unchanged": True,
        "model_weights_unchanged": True,
        "batch_size_unchanged": True,
        "held_access_unchanged": True,
    }
    if any(recovery.get(field) != value for field, value in expected_flags.items()):
        raise VideoInformationProbeError("representation recovery invariants changed")
    expected_prior = {
        "status": "source_video_information_not_established",
        "probe_result_sha256": "434522b29602e7bec085364c084176611dc5dbf5e4a1bec9b47be9f6795a5d6e",
        "clip_cache_sha256": "26c29bc69c2bd6ed633aef3ab3f9de3357ece1118c89bd17177cde2c316edfe8",
        "feature_cache_sha256": "8ee768482b82afad09035e6785435a258da7263a2a64176745e78a56d7ee3f83",
        "gallery_manifest_sha256": "53cb2cf6a021a7622915c9c7112e404293802c6c81d68d95ee15d89af8f38ad0",
    }
    if recovery.get("prior_evidence") != expected_prior:
        raise VideoInformationProbeError("prior failure authority changed")
    expected_representation = {
        "feature": "smolvlm_visual_connector_temporal_moments_v1",
        "spatial_pool": "mean_over_64_visual_connector_tokens",
        "per_frame_l2_normalization": True,
        "components": ["mean", "first", "last", "last_minus_first", "linear_time_slope"],
        "time_coordinates": "linspace_minus_one_to_one",
        "feature_dimension": 4800,
        "trainable_parameters": 0,
        "language_prompt_used": False,
    }
    if recovery.get("representation") != expected_representation:
        raise VideoInformationProbeError("temporal representation recovery changed")
    resources = recovery.get("resources", {})
    if resources != {
        "reuse_rgb_clip_cache": True,
        "reuse_gallery_videos": True,
        "batch_size": 48,
        "gpu_count": 1,
        "expected_peak_gpu_reserved_gib": 50,
        "minimum_gpu_headroom_gib": 10,
        "expected_output_gib": 0.03,
        "timeout_seconds": 1800,
    }:
        raise VideoInformationProbeError("representation recovery resources changed")
    boundary = recovery.get("claim_boundary", {})
    if boundary.get("gate_decision_authorized") is not False or boundary.get("writer_authorized") is not False:
        raise VideoInformationProbeError("recovery cannot authorize Gate -1 or Writer")
    spec = copy.deepcopy(load_video_spec(base_config_path))
    spec["schema_version"] = 2
    spec["surface"] = recovery["surface"]
    spec["encoder"]["feature"] = recovery["representation"]["feature"]
    spec["resources"]["expected_output_gib"] = resources["expected_output_gib"]
    spec["recovery"] = recovery
    return spec


def uniform_frame_indices(
    total_frames: int,
    *,
    frame_count: int,
    end_exclusion: int,
    end_fraction: float = 1.0,
) -> np.ndarray:
    """Return fixed-length uniform indices without exposing source trajectory length."""

    if frame_count < 2 or end_exclusion < 0 or not 0 < end_fraction <= 1:
        raise VideoInformationProbeError("invalid frame-sampling contract")
    last = int(np.floor((total_frames - 1 - end_exclusion) * end_fraction))
    if last + 1 < frame_count:
        raise VideoInformationProbeError("trajectory does not contain enough frames")
    indices = np.rint(np.linspace(0, last, frame_count)).astype(np.int64)
    if len(np.unique(indices)) != frame_count:
        raise VideoInformationProbeError("uniform frame sampler produced duplicates")
    return indices


def _shuffled_clip(clip: np.ndarray, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(len(clip))
    identity = np.arange(len(clip))
    if np.array_equal(permutation, identity) or np.array_equal(permutation, identity[::-1]):
        permutation = np.roll(permutation, 1)
    return clip[permutation]


def derive_clip_condition(
    ordered_clip: np.ndarray,
    condition: str,
    *,
    shuffle_seed: int,
    drop_last_clip: np.ndarray | None = None,
) -> np.ndarray:
    """Derive a fixed-shape control from a single action-hidden RGB clip."""

    clip = np.asarray(ordered_clip)
    if clip.ndim != 4 or clip.shape[-1] != 3 or clip.dtype != np.uint8:
        raise VideoInformationProbeError("clip must be uint8 [frame,height,width,3]")
    if condition == "ordered_full":
        result = clip
    elif condition == "reversed":
        result = clip[::-1]
    elif condition == "shuffled":
        result = _shuffled_clip(clip, shuffle_seed)
    elif condition == "first_frame":
        result = np.repeat(clip[:1], len(clip), axis=0)
    elif condition == "last_frame":
        result = np.repeat(clip[-1:], len(clip), axis=0)
    elif condition == "static_temporal_median":
        static = np.rint(np.median(clip, axis=0)).astype(np.uint8)
        result = np.repeat(static[None], len(clip), axis=0)
    elif condition == "drop_last_20_percent" and drop_last_clip is not None:
        result = np.asarray(drop_last_clip)
    else:
        raise VideoInformationProbeError(f"unsupported or incomplete clip condition: {condition}")
    result = np.ascontiguousarray(result)
    if result.shape != clip.shape or result.dtype != clip.dtype:
        raise VideoInformationProbeError("control changed clip shape or dtype")
    return result


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise VideoInformationProbeError("feature row has invalid norm")
    return features / norms


def temporal_moment_descriptor(frame_features: np.ndarray) -> np.ndarray:
    """Build the fixed mean/endpoint/delta/slope descriptor from frame features."""

    values = np.asarray(frame_features, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise VideoInformationProbeError("frame features must be finite [batch,time,feature]")
    normalized = _normalize_rows(values.reshape(-1, values.shape[-1])).reshape(values.shape)
    coordinates = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float64)
    slope = np.sum(normalized * coordinates[None, :, None], axis=1) / np.sum(coordinates**2)
    components = (
        normalized.mean(axis=1),
        normalized[:, 0],
        normalized[:, -1],
        normalized[:, -1] - normalized[:, 0],
        slope,
    )
    descriptor = np.concatenate(components, axis=1)
    if not np.isfinite(descriptor).all():
        raise VideoInformationProbeError("temporal descriptor is non-finite")
    return descriptor.astype(np.float32)


def fit_frozen_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    task_ids: list[int],
    ridge_lambda: float,
) -> dict[str, Any]:
    """Fit the predeclared balanced binary dual-ridge readout once."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels) or len(task_ids) != 2:
        raise VideoInformationProbeError("invalid support readout arrays")
    if not np.isfinite(features).all() or ridge_lambda <= 0:
        raise VideoInformationProbeError("invalid support features or ridge lambda")
    counts = [int(np.sum(labels == task_id)) for task_id in task_ids]
    if min(counts) < 2 or counts[0] != counts[1] or set(labels.tolist()) != set(task_ids):
        raise VideoInformationProbeError("support labels must be balanced over two tasks")
    normalized = _normalize_rows(features)
    center = normalized.mean(axis=0)
    centered = normalized - center
    targets = np.where(labels == task_ids[0], -1.0, 1.0)
    gram = centered @ centered.T
    dual = np.linalg.solve(gram + ridge_lambda * np.eye(len(gram)), targets)
    weight = centered.T @ dual
    if not np.isfinite(weight).all() or np.linalg.norm(weight) <= 0:
        raise VideoInformationProbeError("ridge readout produced an invalid weight")
    return {
        "task_ids": list(task_ids),
        "ridge_lambda": float(ridge_lambda),
        "center": center,
        "weight": weight,
        "weight_norm": float(np.linalg.norm(weight)),
    }


def score_linear_probe(
    model: dict[str, Any], features: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    task_ids = model["task_ids"]
    if features.ndim != 2 or len(features) != len(labels):
        raise VideoInformationProbeError("invalid query readout arrays")
    if not set(labels.tolist()).issubset(set(task_ids)):
        raise VideoInformationProbeError("query label is outside the frozen source pair")
    scores = (_normalize_rows(features) - model["center"]) @ model["weight"]
    predictions = np.where(scores > 0, task_ids[1], task_ids[0])
    signs = np.where(labels == task_ids[0], -1.0, 1.0)
    correct = predictions == labels
    per_task = {
        str(task_id): float(np.mean(correct[labels == task_id])) for task_id in task_ids
    }
    return {
        "scores": scores,
        "predictions": predictions,
        "correct": correct,
        "signed_margins": scores * signs,
        "per_task_accuracy": per_task,
        "balanced_accuracy": float(np.mean(list(per_task.values()))),
    }


def stratified_accuracy_interval(
    correct: np.ndarray,
    labels: np.ndarray,
    *,
    task_ids: list[int],
    samples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels)
    if len(correct) != len(labels) or samples < 100 or not 0 < confidence_level < 1:
        raise VideoInformationProbeError("invalid bootstrap contract")
    groups = [np.flatnonzero(labels == task_id) for task_id in task_ids]
    if any(len(group) == 0 for group in groups):
        raise VideoInformationProbeError("bootstrap task stratum is empty")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = np.mean(
            [correct[rng.choice(group, len(group), replace=True)].mean() for group in groups]
        )
    alpha = (1 - confidence_level) / 2
    point = float(np.mean([correct[group].mean() for group in groups]))
    return {
        "point_estimate": point,
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1 - alpha)),
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def decide_video_probe(spec: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    thresholds = spec["thresholds"]
    accuracy = {key: metrics[key]["balanced_accuracy"] for key in EXPECTED_CONDITIONS}
    ordered = accuracy["ordered_full"]
    diagnostics = {
        "ordered_vs_first_frame_gap": ordered - accuracy["first_frame"],
        "ordered_vs_static_median_gap": ordered - accuracy["static_temporal_median"],
        "temporal_order_gap": ordered - max(accuracy["reversed"], accuracy["shuffled"]),
        "last_frame_full_ratio": accuracy["last_frame"] / ordered if ordered else float("inf"),
        "drop_last_20_retention": accuracy["drop_last_20_percent"] / ordered if ordered else 0.0,
    }
    content_present = all(
        (
            ordered >= thresholds["minimum_ordered_balanced_accuracy"],
            metrics["bidirectional_query_pair_fraction"]
            >= thresholds["minimum_bidirectional_query_pair_fraction"],
            metrics["wrong_video_specificity"] >= thresholds["minimum_wrong_video_specificity"],
            diagnostics["ordered_vs_first_frame_gap"] >= thresholds["minimum_ordered_vs_first_frame_gap"],
            diagnostics["ordered_vs_static_median_gap"] >= thresholds["minimum_ordered_vs_static_median_gap"],
            diagnostics["last_frame_full_ratio"] < thresholds["last_frame_full_ratio_limit"],
            diagnostics["drop_last_20_retention"] >= thresholds["drop_last_20_minimum_retention"],
        )
    )
    temporal_present = diagnostics["temporal_order_gap"] >= thresholds["minimum_temporal_order_gap"]
    if content_present and temporal_present:
        status = "source_video_content_and_temporal_signal_present"
    elif content_present:
        status = "source_video_content_present_temporal_order_not_established"
    elif ordered < thresholds["minimum_ordered_balanced_accuracy"]:
        status = "source_video_information_not_established"
    else:
        status = "source_video_content_ambiguous_static_or_endpoint_shortcut"
    return {
        "status": status,
        "content_criteria_met": content_present,
        "temporal_order_criterion_met": temporal_present,
        "diagnostics": diagnostics,
        "gate_decision_authorized": False,
        "writer_authorized": False,
    }
