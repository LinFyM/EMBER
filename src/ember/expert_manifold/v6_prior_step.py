"""One-task Program leaf and counterfactual feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import counterfactual_frame_order
from ember.writer.condition_update import FrozenV6ConditionResidualWriter
from ember.writer.data import RawTeacherVideo


@dataclass(frozen=True)
class GeneratedConditionGraph:
    """Correct functional graph plus one action-free counterfactual key."""

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


def generate_condition_graph(
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
) -> GeneratedConditionGraph:
    """Encode correct once and build a negative feature without policy actions."""

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
            # The exact target-task tokens remain unchanged. Only the video is
            # replaced, so the information wall cannot leak wrong-task language.
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

    # The Program itself is the only differentiable leaf.  The historical v6
    # graph, fixed feature, and manual memory are all outside autograd ownership.
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
        raise ExpertManifoldError("condition Program graph changed topology")
    negative_source = counterfactual_video or correct_video
    return GeneratedConditionGraph(
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


def decode_candidate_program(
    graph: GeneratedConditionGraph,
    *,
    writer: FrozenV6ConditionResidualWriter,
    motion: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Reproduce the exact post-write Program arithmetic, then decode once."""

    if (
        graph.base_program_slots.shape != graph.residual_before.shape
        or graph.base_program_slots.shape != graph.program_input_before.shape
        or graph.base_program_slots.shape[0] != 1
        or motion.shape != graph.residual_before.shape[1:]
        or motion.dtype != torch.float32
        or motion.device != graph.residual_before.device
    ):
        raise ExpertManifoldError("PCUG candidate Program topology changed")
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        candidate_program = graph.base_program_slots + (
            graph.residual_before + motion.unsqueeze(0)
        ).to(dtype=graph.base_program_slots.dtype)
        candidate_program = candidate_program.to(dtype=torch.float32)
        candidate_lora = writer.base_writer.decode_slots(candidate_program)
    return candidate_program.detach(), {
        name: value.detach() for name, value in candidate_lora.items()
    }


def program_cotangent(
    graph: GeneratedConditionGraph,
    lora_gradients: Mapping[str, torch.Tensor],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Transport one task-local LoRA VJP to its complete Program leaf."""

    return _transport_program_cotangent(
        graph.correct_lora,
        graph.program_leaf,
        lora_gradients,
        retain_graph=retain_graph,
    )


def _transport_program_cotangent(
    lora_state: Mapping[str, torch.Tensor],
    program_leaf: torch.Tensor,
    lora_gradients: Mapping[str, torch.Tensor],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    names = tuple(lora_state)
    if set(lora_gradients) != set(names):
        raise ExpertManifoldError("functional LoRA cotangent topology changed")
    gradient = torch.autograd.grad(
        tuple(lora_state[name] for name in names),
        program_leaf,
        grad_outputs=tuple(lora_gradients[name] for name in names),
        retain_graph=retain_graph,
    )[0]
    if gradient.shape != program_leaf.shape:
        raise ExpertManifoldError("functional loss did not reach the complete Program")
    # The caller owns task-local aggregation.  No rank/task/world-size scaling
    # is permitted here.
    return gradient[0].detach().to(dtype=torch.float32)


def redecoded_program_cotangent(
    *,
    writer: FrozenV6ConditionResidualWriter,
    program_value: torch.Tensor,
    lora_gradients: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Replay only the compiler to transport a delayed LoRA cotangent."""

    program_leaf = program_value.detach().to(dtype=torch.float32).requires_grad_(True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        lora_state = writer.base_writer.decode_slots(program_leaf)
    return _transport_program_cotangent(
        lora_state,
        program_leaf,
        lora_gradients,
    )
