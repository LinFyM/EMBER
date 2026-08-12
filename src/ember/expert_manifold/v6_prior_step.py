"""One ordered-video Program leaf and matched counterfactual construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import counterfactual_frame_order
from ember.writer.condition_update import FrozenV6ConditionResidualWriter
from ember.writer.data import RawTeacherVideo


@dataclass(frozen=True)
class GeneratedViewGraph:
    """One complete one-shot Writer graph and its zero-RHS counterfactual key."""

    correct_lora: Mapping[str, torch.Tensor]
    program_leaf: torch.Tensor
    program_input_before: torch.Tensor
    base_program_slots: torch.Tensor
    residual_before: torch.Tensor
    correct_feature: torch.Tensor
    negative_feature: torch.Tensor
    correct_raw_frames: int
    correct_sampled_frames: int
    negative_raw_frames: int
    negative_sampled_frames: int


def _video_tensors(
    video: RawTeacherVideo,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = torch.from_numpy(video.frames).to(device, non_blocking=True)
    indices = torch.from_numpy(video.frame_indices).to(device, non_blocking=True)
    offsets = torch.tensor((0, frames.shape[0]), dtype=torch.long, device="cpu")
    return frames, indices, offsets


def generate_view_graph(
    *,
    writer: FrozenV6ConditionResidualWriter,
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
) -> GeneratedViewGraph:
    """Build one ordered correct Program and its matched negative feature."""

    if (kind == "wrong") != (counterfactual_video is not None):
        raise ExpertManifoldError("counterfactual video ownership changed")
    tokens, mask, task_span = language_tokens
    correct_frames, correct_indices, correct_offsets = _video_tensors(
        correct_video, device
    )
    base = writer.base_writer
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        correct_evidence = base.encode_video_evidence(
            policy,
            correct_frames,
            correct_offsets,
            tokens,
            mask,
            task_span,
        )
        correct_memories = base.build_memories(correct_evidence, correct_indices)
        base_slots = base.compile_slots(correct_memories)

        if kind == "wrong":
            if counterfactual_video is None:
                raise ExpertManifoldError("wrong-video negative is missing")
            negative_frames, negative_indices, negative_offsets = _video_tensors(
                counterfactual_video, device
            )
            # Target-task language is unchanged; only RGB evidence is replaced.
            correct_feature, negative_feature = writer.paired_condition_features(
                policy,
                correct_frames,
                correct_offsets,
                tokens,
                mask,
                task_span,
                negative_frames=negative_frames,
                negative_offsets=negative_offsets,
            )
        else:
            frame_order = counterfactual_frame_order(
                kind,
                correct_evidence.offsets,
                seed=counterfactual_seed,
                task_ordinal=task_ordinal,
                task_visit=task_visit,
                teacher_demo=teacher_demo,
                device=device,
            )
            if frame_order is None:
                raise ExpertManifoldError("ordered negative lost its frame order")
            negative_indices = correct_indices
            correct_feature, negative_feature = writer.paired_condition_features(
                policy,
                correct_frames,
                correct_offsets,
                tokens,
                mask,
                task_span,
                frame_order=frame_order,
            )
        stored_residual = writer.program_memory(correct_feature)
        stored_program = base_slots + stored_residual.to(dtype=base_slots.dtype)

    # The complete Program is the sole differentiable Writer leaf.
    program_leaf = stored_program.detach().to(dtype=torch.float32).requires_grad_(True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        correct_lora = base.decode_slots(program_leaf)
    if (
        program_leaf.shape != (1, 320, base.program_width)
        or correct_feature.shape != negative_feature.shape
        or correct_feature.shape[0] != 1
    ):
        raise ExpertManifoldError("paired-video Program graph changed topology")
    negative_source = counterfactual_video or correct_video
    return GeneratedViewGraph(
        correct_lora=correct_lora,
        program_leaf=program_leaf,
        program_input_before=stored_program.detach(),
        base_program_slots=base_slots.detach(),
        residual_before=stored_residual.detach(),
        correct_feature=correct_feature[0],
        negative_feature=negative_feature[0],
        correct_raw_frames=int(correct_video.raw_frame_count),
        correct_sampled_frames=int(correct_indices.numel()),
        negative_raw_frames=int(negative_source.raw_frame_count),
        negative_sampled_frames=int(negative_indices.numel()),
    )


def program_cotangent(
    graph: GeneratedViewGraph,
    lora_gradients: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Transport one view's LoRA VJP to its complete Program leaf."""

    names = tuple(graph.correct_lora)
    if set(lora_gradients) != set(names):
        raise ExpertManifoldError("functional LoRA cotangent topology changed")
    gradient = torch.autograd.grad(
        tuple(graph.correct_lora[name] for name in names),
        graph.program_leaf,
        grad_outputs=tuple(lora_gradients[name] for name in names),
    )[0]
    if gradient.shape != graph.program_leaf.shape:
        raise ExpertManifoldError("functional loss did not reach the complete Program")
    return gradient[0].detach().to(dtype=torch.float32)
