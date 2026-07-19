"""Fixed-query and action-drift metrics shared by Gate 0 selection and report."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from ember.gate_zero_runtime import (
    batch_provenance_keys,
    deterministic_flow_inputs,
    preprocess_smolvla_batch,
)


def evenly_spaced_anchor_indices(frame_count: int, anchor_count: int) -> list[int]:
    """Return the frozen inclusive round-linspace anchor schedule."""

    if frame_count < anchor_count:
        raise ValueError("demonstration has fewer frames than required unique anchors")
    if anchor_count < 2:
        raise ValueError("anchor count must include distinct first and last frames")
    indices = [round(index * (frame_count - 1) / (anchor_count - 1)) for index in range(anchor_count)]
    if indices[0] != 0 or indices[-1] != frame_count - 1 or len(set(indices)) != anchor_count:
        raise ValueError("anchor schedule is not inclusive and unique")
    return indices


def anchor_flat_indices(
    frame_index: Sequence[tuple[int, int, int]], *, anchor_count: int
) -> list[int]:
    """Choose frozen anchors within each task/demo while preserving demo order."""

    groups: dict[tuple[int, int], list[int]] = {}
    for flat_index, (task_id, demo_index, _) in enumerate(frame_index):
        groups.setdefault((task_id, demo_index), []).append(flat_index)
    if not groups:
        raise ValueError("query frame index is empty")
    selected: list[int] = []
    for key in sorted(groups):
        group = groups[key]
        selected.extend(group[index] for index in evenly_spaced_anchor_indices(len(group), anchor_count))
    return selected


def unit_variance_mean_action_kl(
    adapted_actions: torch.Tensor, base_actions: torch.Tensor
) -> float:
    """Mean scalar KL for equal unit-variance Gaussians with different means."""

    if adapted_actions.shape != base_actions.shape or adapted_actions.numel() == 0:
        raise ValueError("action mean tensors changed shape or are empty")
    if not torch.isfinite(adapted_actions).all() or not torch.isfinite(base_actions).all():
        raise ValueError("action mean tensors are non-finite")
    return float(0.5 * (adapted_actions.to(torch.float32) - base_actions.to(torch.float32)).square().mean())


def summarize_action_chunk_errors(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    row_keys: Sequence[str],
    *,
    time_partition_count: int = 4,
) -> dict[str, Any]:
    """Summarize normalized action-chunk MSE without losing query identity."""

    if (
        predicted_actions.shape != target_actions.shape
        or predicted_actions.ndim != 3
        or predicted_actions.numel() == 0
    ):
        raise ValueError("predicted and target action chunk shape changed")
    if len(row_keys) != len(predicted_actions) or len(set(row_keys)) != len(row_keys):
        raise ValueError("action chunk row keys changed or are duplicated")
    if not torch.isfinite(predicted_actions).all() or not torch.isfinite(target_actions).all():
        raise ValueError("action chunks are non-finite")
    chunk_size = predicted_actions.shape[1]
    if time_partition_count <= 0 or time_partition_count > chunk_size:
        raise ValueError("action chunk time partition count is invalid")
    episodes: list[str] = []
    for key in row_keys:
        match = re.fullmatch(r"task\d+/demo(\d+)/frame\d+", key)
        if match is None:
            raise ValueError("action chunk row key format changed")
        episodes.append(match.group(1))
    squared = (
        predicted_actions.to(device="cpu", dtype=torch.float64)
        - target_actions.to(device="cpu", dtype=torch.float64)
    ).square()
    by_row = squared.mean(dim=(1, 2))
    by_episode: dict[str, float] = {}
    for episode in sorted(set(episodes), key=int):
        selected = torch.tensor([value == episode for value in episodes])
        by_episode[episode] = float(squared[selected].mean())
    bounds = [index * chunk_size // time_partition_count for index in range(time_partition_count)]
    bounds.append(chunk_size)
    partitions = [[start, stop] for start, stop in zip(bounds[:-1], bounds[1:], strict=True)]
    return {
        "sample_count": len(row_keys),
        "row_keys_sha256": _row_digest(row_keys),
        "action_chunk_size": chunk_size,
        "action_dimension": predicted_actions.shape[2],
        "mean_squared_error": float(squared.mean()),
        "by_row_mse": {key: float(value) for key, value in zip(row_keys, by_row, strict=True)},
        "by_episode_mse": by_episode,
        "by_action_dimension_mse": [float(value) for value in squared.mean(dim=(0, 1))],
        "time_partitions": partitions,
        "by_time_partition_mse": [float(squared[:, start:stop].mean()) for start, stop in partitions],
    }


def _owner(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _row_digest(keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8") + b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedQueryReference:
    query_flow_mse: float
    query_sample_count: int
    query_row_keys_sha256: str
    anchor_actions: torch.Tensor
    anchor_row_keys_sha256: str
    anchor_count: int


class FixedQueryEvaluator:
    """Reuse fixed query/anchor loaders across every selection candidate."""

    def __init__(
        self,
        dataset: Any,
        *,
        preprocessor: Any,
        batch_size: int,
        num_workers: int,
        anchor_count_per_demo: int,
        action_chunk_size: int,
        fixed_noise_seed: int,
        fixed_time_seed: int,
        inference_noise_seed: int,
    ) -> None:
        if batch_size <= 0 or num_workers < 0 or action_chunk_size <= 0:
            raise ValueError("fixed-query loader parameters are invalid")
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.action_chunk_size = action_chunk_size
        self.fixed_noise_seed = fixed_noise_seed
        self.fixed_time_seed = fixed_time_seed
        self.inference_noise_seed = inference_noise_seed
        loader_kwargs: dict[str, Any] = {}
        if num_workers:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        generator = torch.Generator(device="cpu").manual_seed(fixed_noise_seed)
        self.query_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=generator,
            **loader_kwargs,
        )
        anchors = anchor_flat_indices(
            dataset.frame_index, anchor_count=anchor_count_per_demo
        )
        self.anchor_loader = DataLoader(
            Subset(dataset, anchors),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            generator=torch.Generator(device="cpu").manual_seed(inference_noise_seed),
            **loader_kwargs,
        )

    @staticmethod
    def _preserve_rng() -> dict[str, Any]:
        from lerobot.utils.random_utils import get_rng_state

        return get_rng_state()

    @staticmethod
    def _restore_rng(state: dict[str, Any]) -> None:
        from lerobot.utils.random_utils import set_rng_state

        set_rng_state(state)

    def _query_losses(self, model: Any) -> tuple[float, int, str]:
        owner = _owner(model)
        device = next(owner.parameters()).device
        values: list[torch.Tensor] = []
        row_keys: list[str] = []
        with torch.inference_mode():
            for raw_batch in self.query_loader:
                keys = batch_provenance_keys(raw_batch)
                batch = preprocess_smolvla_batch(
                    raw_batch, self.preprocessor, list(owner.config.image_features)
                )
                noise, flow_time = deterministic_flow_inputs(
                    keys,
                    action_shape=(self.action_chunk_size, owner.config.max_action_dim),
                    noise_seed=self.fixed_noise_seed,
                    time_seed=self.fixed_time_seed,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses, _ = model.forward(
                        batch, noise=noise, time=flow_time, reduction="none"
                    )
                if losses.ndim != 1 or len(losses) != len(keys) or not torch.isfinite(losses).all():
                    raise ValueError("fixed query returned invalid per-sample losses")
                values.append(losses.detach().to(device="cpu", dtype=torch.float64))
                row_keys.extend(keys)
        merged = torch.cat(values)
        return float(merged.mean()), int(merged.numel()), _row_digest(row_keys)

    def _anchor_actions(self, model: Any) -> tuple[torch.Tensor, str]:
        owner = _owner(model)
        device = next(owner.parameters()).device
        values: list[torch.Tensor] = []
        row_keys: list[str] = []
        with torch.inference_mode():
            for raw_batch in self.anchor_loader:
                keys = batch_provenance_keys(raw_batch)
                batch = preprocess_smolvla_batch(
                    raw_batch, self.preprocessor, list(owner.config.image_features)
                )
                noise, _ = deterministic_flow_inputs(
                    keys,
                    action_shape=(self.action_chunk_size, owner.config.max_action_dim),
                    noise_seed=self.inference_noise_seed,
                    time_seed=self.inference_noise_seed + 1,
                    device=device,
                )
                owner.reset()
                actions = owner.predict_action_chunk(batch, noise=noise)
                if actions.ndim != 3 or len(actions) != len(keys) or not torch.isfinite(actions).all():
                    raise ValueError("fixed anchor inference returned invalid actions")
                values.append(actions.detach().to(device="cpu", dtype=torch.float32))
                row_keys.extend(keys)
        return torch.cat(values), _row_digest(row_keys)

    def _anchor_action_error_summary(self, model: Any) -> dict[str, Any]:
        owner = _owner(model)
        device = next(owner.parameters()).device
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        row_keys: list[str] = []
        with torch.inference_mode():
            for raw_batch in self.anchor_loader:
                keys = batch_provenance_keys(raw_batch)
                batch = preprocess_smolvla_batch(
                    raw_batch, self.preprocessor, list(owner.config.image_features)
                )
                noise, _ = deterministic_flow_inputs(
                    keys,
                    action_shape=(self.action_chunk_size, owner.config.max_action_dim),
                    noise_seed=self.inference_noise_seed,
                    time_seed=self.inference_noise_seed + 1,
                    device=device,
                )
                owner.reset()
                actions = owner.predict_action_chunk(batch, noise=noise)
                target = batch.get("action")
                if (
                    not isinstance(target, torch.Tensor)
                    or actions.shape != target.shape
                    or actions.ndim != 3
                ):
                    raise ValueError("fixed anchor predicted/target action shape changed")
                predictions.append(actions.detach().to(device="cpu", dtype=torch.float32))
                targets.append(target.detach().to(device="cpu", dtype=torch.float32))
                row_keys.extend(keys)
        return summarize_action_chunk_errors(
            torch.cat(predictions), torch.cat(targets), row_keys
        )

    def evaluate_action_chunk_errors(self, model: Any) -> dict[str, Any]:
        """Evaluate generated actions against fixed source-query demonstrations."""

        rng = self._preserve_rng()
        was_training = model.training
        try:
            model.eval()
            return self._anchor_action_error_summary(model)
        finally:
            model.train(was_training)
            self._restore_rng(rng)

    def capture_base_reference(self, model: Any) -> FixedQueryReference:
        rng = self._preserve_rng()
        was_training = model.training
        try:
            model.eval()
            query_mse, sample_count, query_digest = self._query_losses(model)
            actions, anchor_digest = self._anchor_actions(model)
            return FixedQueryReference(
                query_flow_mse=query_mse,
                query_sample_count=sample_count,
                query_row_keys_sha256=query_digest,
                anchor_actions=actions,
                anchor_row_keys_sha256=anchor_digest,
                anchor_count=len(actions),
            )
        finally:
            model.train(was_training)
            self._restore_rng(rng)

    def evaluate_candidate(
        self, model: Any, reference: FixedQueryReference, *, step: int
    ) -> dict[str, Any]:
        rng = self._preserve_rng()
        was_training = model.training
        try:
            model.eval()
            query_mse, sample_count, query_digest = self._query_losses(model)
            actions, anchor_digest = self._anchor_actions(model)
        finally:
            model.train(was_training)
            self._restore_rng(rng)
        if (
            sample_count != reference.query_sample_count
            or query_digest != reference.query_row_keys_sha256
            or anchor_digest != reference.anchor_row_keys_sha256
            or len(actions) != reference.anchor_count
        ):
            raise ValueError("candidate query/anchor authority changed")
        return {
            "step": step,
            "query_flow_mse": query_mse,
            "base_query_flow_mse": reference.query_flow_mse,
            "query_sample_count": sample_count,
            "query_row_keys_sha256": query_digest,
            "action_drift_proxy": unit_variance_mean_action_kl(
                actions, reference.anchor_actions
            ),
            "anchor_count": len(actions),
            "anchor_row_keys_sha256": anchor_digest,
        }

    def close(self) -> None:
        for loader in (self.query_loader, self.anchor_loader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
        self.dataset.close()
