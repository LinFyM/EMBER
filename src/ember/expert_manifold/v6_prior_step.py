"""One-task generation and gradient composition for the v6-prior Writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.effective_objective import EffectiveAuxiliaryGradients
from ember.expert_manifold.v6_prior import (
    V6PriorDynamicAnchor,
    counterfactual_frame_order,
)
from ember.writer.data import RawTeacherVideo
from ember.writer.model import CompleteLoRAWriter


@dataclass(frozen=True)
class GeneratedCounterfactualPair:
    correct: Mapping[str, torch.Tensor]
    counterfactual: Mapping[str, torch.Tensor]
    correct_anchor: Mapping[str, torch.Tensor]
    counterfactual_anchor: Mapping[str, torch.Tensor]
    correct_raw_frames: int
    correct_sampled_frames: int
    counterfactual_raw_frames: int
    counterfactual_sampled_frames: int
    correct_comparison: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class ParameterGradientComponents:
    positive: tuple[torch.Tensor, ...]
    projection: tuple[torch.Tensor, ...]
    ranking: tuple[torch.Tensor, ...]
    distillation: tuple[torch.Tensor, ...] | None = None


def _video_tensors(
    video: RawTeacherVideo,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = torch.from_numpy(video.frames).to(device, non_blocking=True)
    indices = torch.from_numpy(video.frame_indices).to(device, non_blocking=True)
    offsets = torch.tensor(
        (0, int(frames.shape[0])),
        dtype=torch.long,
        device="cpu",
    )
    return frames, indices, offsets


def _decode_frozen_references(
    writer: CompleteLoRAWriter,
    correct_memories: torch.Tensor,
    negative_memories: torch.Tensor,
    dynamic_anchor: V6PriorDynamicAnchor,
    comparison_decoder: V6PriorDynamicAnchor | None,
) -> tuple[
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor] | None,
]:
    correct_anchor = writer.decode_memories(
        correct_memories,
        compiler=dynamic_anchor.compiler,
        factor_heads=dynamic_anchor.factor_heads,
    )
    counterfactual_anchor = writer.decode_memories(
        negative_memories,
        compiler=dynamic_anchor.compiler,
        factor_heads=dynamic_anchor.factor_heads,
    )
    comparison = None
    if comparison_decoder is not None:
        comparison = writer.decode_memories(
            correct_memories,
            compiler=comparison_decoder.compiler,
            factor_heads=comparison_decoder.factor_heads,
        )
    return correct_anchor, counterfactual_anchor, comparison


def generate_counterfactual_pair(
    *,
    writer: CompleteLoRAWriter,
    dynamic_anchor: V6PriorDynamicAnchor,
    comparison_decoder: V6PriorDynamicAnchor | None = None,
    policy: torch.nn.Module,
    correct_video: RawTeacherVideo,
    counterfactual_video: RawTeacherVideo | None,
    language_tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    kind: str,
    counterfactual_seed: int,
    task_ordinal: int,
    task_visit: int,
    teacher_demo: int,
    device: torch.device,
) -> GeneratedCounterfactualPair:
    """Generate correct and one bounded negative from exact current language."""

    if (kind == "wrong") != (counterfactual_video is not None):
        raise ExpertManifoldError("counterfactual video ownership changed")
    tokens, mask, task_span = language_tokens
    correct_frames, correct_indices, correct_offsets = _video_tensors(
        correct_video, device
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        evidence = writer.encode_video_evidence(
            policy,
            correct_frames,
            correct_offsets,
            tokens,
            mask,
            task_span,
        )
        correct_memories = writer.build_memories(evidence, correct_indices)
        if kind == "wrong":
            if counterfactual_video is None:
                raise ExpertManifoldError("wrong-video negative is missing")
            negative_frames, negative_indices, negative_offsets = _video_tensors(
                counterfactual_video, device
            )
            negative_evidence = writer.encode_video_evidence(
                policy,
                negative_frames,
                negative_offsets,
                tokens,
                mask,
                task_span,
            )
            negative_memories = writer.build_memories(
                negative_evidence,
                negative_indices,
            )
        else:
            frame_order = counterfactual_frame_order(
                kind,
                evidence.offsets,
                seed=counterfactual_seed,
                task_ordinal=task_ordinal,
                task_visit=task_visit,
                teacher_demo=teacher_demo,
                device=device,
            )
            if frame_order is None:
                raise ExpertManifoldError("ordered negative lost its frame order")
            negative_memories = writer.build_memories(
                evidence,
                correct_indices,
                frame_order=frame_order,
            )
            negative_indices = correct_indices
        correct_anchor, counterfactual_anchor, correct_comparison = (
            _decode_frozen_references(
                writer,
                correct_memories,
                negative_memories,
                dynamic_anchor,
                comparison_decoder,
            )
        )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        correct = writer.decode_memories(correct_memories)
        counterfactual = writer.decode_memories(negative_memories)
    states = (correct, counterfactual, correct_anchor, counterfactual_anchor)
    if correct_comparison is not None:
        states = (*states, correct_comparison)
    if any(set(state) != set(correct) for state in states):
        raise ExpertManifoldError("counterfactual LoRA topology changed")
    if any(
        value.requires_grad
        for state in (correct_anchor, counterfactual_anchor, correct_comparison or {})
        for value in state.values()
    ):
        raise ExpertManifoldError("condition-local v6 anchor gained gradients")
    negative_video_value = counterfactual_video or correct_video
    return GeneratedCounterfactualPair(
        correct=correct,
        counterfactual=counterfactual,
        correct_anchor=correct_anchor,
        counterfactual_anchor=counterfactual_anchor,
        correct_comparison=correct_comparison,
        correct_raw_frames=int(correct_video.raw_frame_count),
        correct_sampled_frames=int(correct_indices.numel()),
        counterfactual_raw_frames=int(negative_video_value.raw_frame_count),
        counterfactual_sampled_frames=int(negative_indices.numel()),
    )


def merged_output_gradients(
    *,
    pair: GeneratedCounterfactualPair,
    functional: Mapping[str, torch.Tensor],
    auxiliary: EffectiveAuxiliaryGradients,
    projection_weight: float,
    ranking_weight: float,
    task_scale: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Merge three objectives at generated LoRA outputs for one backward pass."""

    names = tuple(pair.correct)
    if (
        set(pair.counterfactual) != set(names)
        or set(functional) != set(names)
        or not 0 <= projection_weight <= 1
        or not 0 <= ranking_weight <= 1
        or not 0 < task_scale <= 1
    ):
        raise ExpertManifoldError("invalid v6-prior output-gradient merge")
    correct_gradients = tuple(
        (
            functional[name]
            + projection_weight * auxiliary.correct_projection[name]
            + ranking_weight * auxiliary.correct_ranking[name]
        )
        * task_scale
        for name in names
    )
    negative_gradients = tuple(
        (
            projection_weight * auxiliary.counterfactual_projection[name]
            + ranking_weight * auxiliary.counterfactual_ranking[name]
        )
        * task_scale
        for name in names
    )
    gradients = (*correct_gradients, *negative_gradients)
    outputs = tuple(pair.correct[name] for name in names) + tuple(
        pair.counterfactual[name] for name in names
    )
    return outputs, gradients


def parameter_gradient_components(
    *,
    pair: GeneratedCounterfactualPair,
    functional: Mapping[str, torch.Tensor],
    distillation: Mapping[str, torch.Tensor] | None = None,
    auxiliary: EffectiveAuxiliaryGradients,
    parameters: tuple[torch.nn.Parameter, ...],
    completion_only: bool = False,
) -> ParameterGradientComponents:
    """Measure unweighted compiler/head gradient vectors on one graph."""

    names = tuple(pair.correct)
    if not parameters or set(functional) != set(names):
        raise ExpertManifoldError("invalid v6-prior gradient-profile request")
    correct = tuple(pair.correct[name] for name in names)
    negative = tuple(pair.counterfactual[name] for name in names)
    positive = torch.autograd.grad(
        correct,
        parameters,
        grad_outputs=tuple(functional[name] for name in names),
        retain_graph=True,
    )
    distillation_gradient = (
        torch.autograd.grad(
            correct,
            parameters,
            grad_outputs=tuple(distillation[name] for name in names),
            retain_graph=True,
        )
        if distillation is not None
        else None
    )
    if completion_only:
        projection = torch.autograd.grad(
            correct,
            parameters,
            grad_outputs=tuple(auxiliary.correct_completion[name] for name in names),
            retain_graph=True,
        )
    else:
        projection = torch.autograd.grad(
            (*correct, *negative),
            parameters,
            grad_outputs=(
                *tuple(auxiliary.correct_projection[name] for name in names),
                *tuple(auxiliary.counterfactual_projection[name] for name in names),
            ),
            retain_graph=True,
        )
    ranking = torch.autograd.grad(
        (*correct, *negative),
        parameters,
        grad_outputs=(
            *tuple(auxiliary.correct_ranking[name] for name in names),
            *tuple(auxiliary.counterfactual_ranking[name] for name in names),
        ),
    )
    return ParameterGradientComponents(
        positive=tuple(value.detach() for value in positive),
        projection=tuple(value.detach() for value in projection),
        ranking=tuple(value.detach() for value in ranking),
        distillation=(
            tuple(value.detach() for value in distillation_gradient)
            if distillation_gradient is not None
            else None
        ),
    )
