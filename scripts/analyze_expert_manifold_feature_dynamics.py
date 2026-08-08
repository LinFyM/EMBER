#!/usr/bin/env python3
"""Audit temporal video evidence against one formal task-expert bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

from ember.expert_manifold.contract import (
    REPO_ROOT,
    authority_path,
    load_expert_manifold_config,
)
from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.expert_manifold.feature_cache import inspect_feature_cache
from ember.expert_manifold.model import TopologicalLoRAChunkLayout
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ANALYSIS_SCHEMA = "ember_pi05_expert_manifold_feature_dynamics_v1"


def _quantiles(value: torch.Tensor | np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    return torch.nn.functional.cosine_similarity(left, right, dim=1)


def _cosine_matrix(value: torch.Tensor) -> np.ndarray:
    rows = value.reshape(value.shape[0], -1)
    rows = rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(1e-30)
    return (rows @ rows.T).double().numpy()


def _off_diagonal(value: np.ndarray) -> np.ndarray:
    return value[~np.eye(value.shape[0], dtype=bool)]


def _centered_spectrum(value: torch.Tensor) -> dict[str, Any]:
    centered = value - value.mean(dim=0, keepdim=True)
    eigenvalues = np.linalg.eigvalsh((centered @ centered.T).double().numpy())[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normalized = eigenvalues / eigenvalues.sum()
    active = normalized > 0
    return {
        "effective_rank": float(
            np.exp(-(normalized[active] * np.log(normalized[active])).sum())
        ),
        "top1_energy_fraction": float(normalized[0]),
        "top4_energy_fraction": float(normalized[:4].sum()),
        "top8_energy_fraction": float(normalized[:8].sum()),
    }


def _phase_shuffle(features: torch.Tensor) -> torch.Tensor:
    rows = []
    for task in range(features.shape[0]):
        demos = []
        for demo in range(features.shape[1]):
            generator = torch.Generator().manual_seed(20260808 + task * 100 + demo)
            permutation = torch.randperm(features.shape[2], generator=generator)
            demos.append(features[task, demo].index_select(0, permutation))
        rows.append(torch.stack(demos))
    return torch.stack(rows)


def _representation(features: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "dc":
        return features.mean(dim=2)
    if kind == "temporal":
        centered = features - features.mean(dim=2, keepdim=True)
        return centered.flatten(2)
    if kind == "difference":
        return (features[:, :, 1:] - features[:, :, :-1]).flatten(2)
    raise ValueError(f"unsupported feature representation: {kind}")


def _loo_template_scores(
    normal: torch.Tensor, reversed_value: torch.Tensor, shuffled: torch.Tensor
) -> dict[str, Any]:
    correct_scores = []
    reversed_scores = []
    shuffled_scores = []
    same_task_output = []
    for task in range(normal.shape[0]):
        total = normal[task].sum(dim=0, keepdim=True)
        templates = (total - normal[task]) / (normal.shape[1] - 1)
        correct_scores.append(_cosine_rows(normal[task], templates))
        reversed_scores.append(_cosine_rows(reversed_value[task], templates))
        shuffled_scores.append(_cosine_rows(shuffled[task], templates))
        gram = _cosine_matrix(normal[task])
        same_task_output.append(gram[np.triu_indices(normal.shape[1], 1)])
    correct = torch.cat(correct_scores)
    reverse = torch.cat(reversed_scores)
    shuffle = torch.cat(shuffled_scores)
    return {
        "correct_template_cosine": _quantiles(correct),
        "reversed_template_cosine": _quantiles(reverse),
        "phase_shuffled_template_cosine": _quantiles(shuffle),
        "correct_minus_reversed_median": float(
            correct.median() - reverse.median()
        ),
        "correct_minus_phase_shuffled_median": float(
            correct.median() - shuffle.median()
        ),
        "same_task_video_cosine": _quantiles(np.concatenate(same_task_output)),
    }


def _coefficient_predictions(
    task_centroids: torch.Tensor,
    queries: torch.Tensor,
    held_task: int,
) -> np.ndarray:
    keep = np.arange(task_centroids.shape[0]) != held_task
    train_ids = np.flatnonzero(keep)
    train = task_centroids[keep]
    mean = train.mean(dim=0, keepdim=True)
    centered = train - mean
    query = queries / torch.linalg.vector_norm(
        queries, dim=1, keepdim=True
    ).clamp_min(1e-12)
    kernel = centered @ centered.T + 0.1 * torch.eye(centered.shape[0])
    right = centered @ (query - mean).T
    weights = torch.linalg.solve(kernel, right).T.double().numpy()
    base = (1.0 - weights.sum(axis=1, keepdims=True)) / len(train_ids)
    coefficients = np.zeros((len(query), len(task_centroids)), dtype=np.float64)
    coefficients[:, train_ids] = weights + base
    return coefficients


def _target_cosine(
    coefficients: np.ndarray, held_task: int, gram: np.ndarray
) -> np.ndarray:
    numerator = coefficients @ gram[:, held_task]
    prediction_energy = np.einsum(
        "bi,ij,bj->b", coefficients, gram, coefficients
    )
    return numerator / np.sqrt(
        np.maximum(prediction_energy, 1e-30) * gram[held_task, held_task]
    )


def _linear_transfer_proxy(
    normal: torch.Tensor,
    reversed_value: torch.Tensor,
    shuffled: torch.Tensor,
    target_gram: np.ndarray,
    suites: list[str],
) -> dict[str, Any]:
    centroids = normal.mean(dim=1)
    centroids = centroids / torch.linalg.vector_norm(
        centroids, dim=1, keepdim=True
    ).clamp_min(1e-12)
    scores: dict[str, list[float]] = {
        "one_shot_correct": [],
        "three_shot_correct": [],
        "five_shot_correct": [],
        "reversed": [],
        "phase_shuffled": [],
    }
    per_suite = {
        suite: {"correct": [], "reversed": [], "phase_shuffled": []}
        for suite in sorted(set(suites))
    }
    for task in range(normal.shape[0]):
        correct = _target_cosine(
            _coefficient_predictions(centroids, normal[task], task),
            task,
            target_gram,
        )
        reverse = _target_cosine(
            _coefficient_predictions(centroids, reversed_value[task], task),
            task,
            target_gram,
        )
        shuffle = _target_cosine(
            _coefficient_predictions(centroids, shuffled[task], task),
            task,
            target_gram,
        )
        scores["one_shot_correct"].extend(correct)
        scores["reversed"].extend(reverse)
        scores["phase_shuffled"].extend(shuffle)
        per_suite[suites[task]]["correct"].extend(correct)
        per_suite[suites[task]]["reversed"].extend(reverse)
        per_suite[suites[task]]["phase_shuffled"].extend(shuffle)
        for shots, name in ((3, "three_shot_correct"), (5, "five_shot_correct")):
            usable = (normal.shape[1] // shots) * shots
            averaged = normal[task, :usable].reshape(
                -1, shots, normal.shape[-1]
            ).mean(dim=1)
            scores[name].extend(
                _target_cosine(
                    _coefficient_predictions(centroids, averaged, task),
                    task,
                    target_gram,
                )
            )
    result = {name: _quantiles(values) for name, values in scores.items()}
    result["correct_minus_reversed_median"] = float(
        np.median(scores["one_shot_correct"]) - np.median(scores["reversed"])
    )
    result["correct_minus_phase_shuffled_median"] = float(
        np.median(scores["one_shot_correct"])
        - np.median(scores["phase_shuffled"])
    )
    result["per_suite_median"] = {
        suite: {name: float(np.median(values)) for name, values in arms.items()}
        for suite, arms in per_suite.items()
    }
    return result


def _inspect_inputs(
    args: argparse.Namespace, config_path: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        args.source_run.resolve(),
        args.checkpoint.resolve(),
        evaluation_mode="formal",
    )
    manifest = read_json(authority_path(config, "target_data_manifest"))
    train_rows = sorted(
        (row for row in manifest["tasks"] if row["split_role"] == "train"),
        key=lambda row: int(row["global_task_id"]),
    )
    task_keys = tuple((str(row["suite"]), int(row["task_id"])) for row in train_rows)
    expert = inspect_task_expert_bank(
        config_path=config_path,
        bank_root=args.expert_bank_root.resolve(),
        step=args.expert_step,
        source=source,
        task_keys=task_keys,
        evaluation_role="development_train",
        require_formal=True,
    )
    cache = inspect_feature_cache(
        config_path, args.feature_cache_root.resolve(), source=source
    )
    cache_rows = sorted(cache["tasks"], key=lambda row: int(row["task_ordinal"]))
    expert_rows = {int(row["ordinal"]): row for row in expert["tasks"]}
    return cache_rows, expert_rows


def _load_features(cache_rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack(
        [
            load_file(str(row["features"]["path"]), device="cpu")[
                "video_innovation"
            ].float()
            for row in cache_rows
        ]
    )


def _load_targets(
    config: dict[str, Any], expert_rows: dict[int, dict[str, Any]]
) -> tuple[torch.Tensor, torch.Tensor]:
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    template = identity_lora_state(lora)
    layout = TopologicalLoRAChunkLayout(
        lora, chunk_width=int(config["topological_writer"]["chunk_width"])
    )
    targets = []
    b_targets = []
    for ordinal in range(24):
        state = load_file(
            str(Path(expert_rows[ordinal]["checkpoint"]) / "adapter.safetensors"),
            device="cpu",
        )
        targets.append(layout.tokenize(state, template).float().flatten())
        b_targets.append(
            torch.cat(
                [
                    state[name].float().flatten()
                    for name in sorted(state)
                    if name.endswith(".lora_B.default.weight")
                ]
            )
        )
    return torch.stack(targets), torch.stack(b_targets)


def _feature_energy(features: torch.Tensor) -> dict[str, Any]:
    total_energy = features.square().sum(dim=(-2, -1))
    dc_energy = features.mean(dim=2).square().sum(dim=-1) * features.shape[2]
    return {
        "rms": _quantiles(features.square().mean(dim=(-2, -1)).sqrt()),
        "dc_fraction": _quantiles(dc_energy / total_energy),
        "temporal_fraction": _quantiles(1.0 - dc_energy / total_energy),
        "first_difference_over_total": _quantiles(
            (features[:, :, 1:] - features[:, :, :-1])
            .square()
            .sum(dim=(-2, -1))
            / total_energy
        ),
        "reversal_relative_l2": _quantiles(
            torch.linalg.vector_norm(
                (features - features.flip(2)).flatten(2), dim=2
            )
            / torch.linalg.vector_norm(features.flatten(2), dim=2).clamp_min(1e-30)
        ),
    }


def _representation_results(
    features: torch.Tensor, b_target: torch.Tensor, suites: list[str]
) -> dict[str, Any]:
    shuffled_features = _phase_shuffle(features)
    result = {}
    b_gram = (b_target @ b_target.T).double().numpy()
    for kind in ("dc", "temporal", "difference"):
        normal = _representation(features, kind)
        reversed_value = _representation(features.flip(2), kind)
        shuffled = _representation(shuffled_features, kind)
        result[kind] = {
            "template_evidence": _loo_template_scores(
                normal, reversed_value, shuffled
            ),
            "expert_b_transfer_proxy": _linear_transfer_proxy(
                normal,
                reversed_value,
                shuffled,
                b_gram,
                suites,
            ),
        }
    return result


def _target_results(
    features: torch.Tensor, target: torch.Tensor, b_target: torch.Tensor
) -> tuple[dict[str, Any], dict[str, float]]:
    feature_task_geometry = _cosine_matrix(
        _representation(features, "temporal").mean(dim=1)
    )
    raw_target_geometry = _cosine_matrix(target)
    b_target_geometry = _cosine_matrix(b_target)
    upper = np.triu_indices(24, 1)
    raw_alignment = spearmanr(
        feature_task_geometry[upper], raw_target_geometry[upper]
    )
    b_alignment = spearmanr(
        feature_task_geometry[upper], b_target_geometry[upper]
    )
    targets = {
        "raw_mean_energy_fraction": float(
            target.mean(dim=0).square().sum() / target.square().sum(dim=1).mean()
        ),
        "raw_centered_spectrum": _centered_spectrum(target),
        "raw_cross_task_cosine": _quantiles(_off_diagonal(raw_target_geometry)),
        "b_cross_task_cosine": _quantiles(_off_diagonal(b_target_geometry)),
    }
    alignment = {
        "raw_spearman_rho": float(raw_alignment.statistic),
        "raw_spearman_p": float(raw_alignment.pvalue),
        "b_spearman_rho": float(b_alignment.statistic),
        "b_spearman_p": float(b_alignment.pvalue),
    }
    return targets, alignment


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_expert_manifold_config(config_path)
    cache_rows, expert_rows = _inspect_inputs(args, config_path, config)
    features = _load_features(cache_rows)
    target, b_target = _load_targets(config, expert_rows)
    targets, alignment = _target_results(features, target, b_target)
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "config": {"path": str(config_path), "schema": config["schema_version"]},
        "feature_cache": str(args.feature_cache_root.resolve()),
        "expert_bank": str(args.expert_bank_root.resolve()),
        "expert_step": args.expert_step,
        "task_count": 24,
        "video_count": 1200,
        "feature_shape": list(features.shape),
        "feature_energy": _feature_energy(features),
        "expert_targets": targets,
        "temporal_feature_target_alignment": alignment,
        "representations": _representation_results(
            features, b_target, [str(row["suite"]) for row in cache_rows]
        ),
        "phase_shuffle_scope": (
            "deterministic permutation of the sealed 16-phase cache; formal rollout "
            "still shuffles raw sampled frames before complete encoder forward"
        ),
        "information_wall": {
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "validation_video_reads": 0,
            "test_video_reads": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json",
    )
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--expert-bank-root", type=Path, required=True)
    parser.add_argument("--expert-step", type=int, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze(args)
    print(
        json.dumps(
            {
                "event": "complete",
                "expert_step": result["expert_step"],
                "tasks": result["task_count"],
                "videos": result["video_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
