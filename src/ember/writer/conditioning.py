"""Sealed Writer-v2 conditioning schedules and paired functional objective."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.writer.model import WriterModelError


def conditioning_cycle(config: Mapping[str, Any]) -> tuple[str, ...]:
    cycle = tuple(str(value) for value in config["conditioning_training"]["step_cycle"])
    allowed = {
        ("normal",),
        (
            "normal",
            "full_language_contrast",
            "generic_language_contrast",
        ),
    }
    if cycle not in allowed:
        raise WriterModelError("AS-Writer conditioning cycle changed")
    return cycle


def batch_size_cycle(batch_size: int, config: Mapping[str, Any]) -> tuple[int, ...]:
    cycle = conditioning_cycle(config)
    if batch_size <= 0:
        raise WriterModelError("AS-Writer batch must be positive")
    if cycle == ("normal",):
        return (batch_size,)
    if batch_size < 2 or batch_size % 2:
        raise WriterModelError("AS-Writer contrast requires an even per-rank batch")
    fraction = float(config["conditioning_training"]["contrast_query_fraction"])
    contrast = int(batch_size * fraction)
    if fraction != 0.5 or contrast * 2 != batch_size:
        raise WriterModelError("AS-Writer contrast must use paired half batches")
    return tuple(
        batch_size if mode == "normal" else contrast
        for mode in cycle
    )


def pack_writer_conditions(
    language: torch.Tensor,
    generic_language: torch.Tensor,
    correct_video: torch.Tensor,
    wrong_video: torch.Tensor | None,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if language.ndim != 2 or generic_language.ndim != 2 or correct_video.ndim != 3:
        raise WriterModelError("AS-Writer condition tensors changed shape")
    if mode == "normal":
        languages, videos = language, correct_video
        language_lengths = (language.shape[0],)
        video_lengths = (correct_video.shape[0],)
    elif mode in {"full_language_contrast", "generic_language_contrast"}:
        if wrong_video is None or wrong_video.ndim != 3:
            raise WriterModelError("contrast step lost its wrong teaching video")
        selected = language if mode == "full_language_contrast" else generic_language
        languages = torch.cat((selected, selected), dim=0)
        videos = torch.cat((correct_video, wrong_video), dim=0)
        language_lengths = (selected.shape[0], selected.shape[0])
        video_lengths = (correct_video.shape[0], wrong_video.shape[0])
    else:
        raise WriterModelError(f"unsupported AS-Writer conditioning mode: {mode}")
    language_offsets = torch.tensor(
        [0, *torch.tensor(language_lengths).cumsum(0).tolist()], dtype=torch.int64
    )
    video_offsets = torch.tensor(
        [0, *torch.tensor(video_lengths).cumsum(0).tolist()], dtype=torch.int64
    )
    return languages, videos, language_offsets, video_offsets


def adapter_state_at(
    generated: Mapping[str, torch.Tensor], index: int, count: int
) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in generated.items():
        if count == 1 and value.ndim == 2 and index == 0:
            result[name] = value
        elif count > 1 and value.ndim == 3 and value.shape[0] == count:
            result[name] = value[index]
        else:
            raise WriterModelError("AS-Writer generated the wrong adapter batch")
    return result


def matching_objective(
    losses: tuple[torch.Tensor, torch.Tensor],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    correct, wrong = losses
    correct_weight = float(config["contrast_correct_loss_weight"])
    matching_weight = float(config["matching_loss_weight"])
    margin = float(config["matching_margin"])
    temperature = float(config["matching_temperature"])
    z = (margin + correct - wrong) / temperature
    probability = torch.sigmoid(z)
    objective = (
        correct_weight * correct
        + matching_weight * temperature * torch.nn.functional.softplus(z)
    )
    coefficients = (
        torch.as_tensor(correct_weight, device=correct.device)
        + matching_weight * probability,
        -matching_weight * probability,
    )
    return objective, coefficients, probability


def same_torch_rng(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        torch.equal(left["torch_cpu"], right["torch_cpu"])
        and torch.equal(left["torch_cuda"], right["torch_cuda"])
    )
