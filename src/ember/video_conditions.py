"""Frame-level controls for causal evaluation of teaching videos."""

from __future__ import annotations

from dataclasses import dataclass

import torch


TEMPORAL_PROCESS_VIDEO_CONDITIONS = frozenset(
    {
        "reversed",
        "shuffled",
        "shuffled_keep_first",
        "first_frame_only",
        "final_frame_only",
        "first_final",
        "endpoints_middle_shuffled",
        "monotone_sparse",
        "static_first_repeated",
    }
)


@dataclass(frozen=True)
class FrameControl:
    """Content order and displayed source-time positions for one video control."""

    content: torch.Tensor
    positions: torch.Tensor


def shuffled_frame_permutation(
    frame_count: int,
    order_seed: int,
    *,
    keep_first: bool,
) -> torch.Tensor:
    if frame_count <= 0 or order_seed < 0:
        raise ValueError("invalid frame permutation request")
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    permutation = torch.randperm(frame_count, generator=generator)
    if keep_first:
        permutation = torch.cat(
            (torch.zeros(1, dtype=permutation.dtype), permutation[permutation != 0])
        )
    return permutation


def _endpoints_middle_shuffled(frame_count: int, order_seed: int) -> torch.Tensor:
    if frame_count <= 2:
        return torch.arange(frame_count)
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    middle = torch.randperm(frame_count - 2, generator=generator) + 1
    return torch.cat((torch.tensor([0]), middle, torch.tensor([frame_count - 1])))


def _monotone_sparse(frame_count: int) -> torch.Tensor:
    selected = torch.arange(0, frame_count, 2)
    if selected[-1].item() != frame_count - 1:
        selected = torch.cat((selected, torch.tensor([frame_count - 1])))
    return selected


def frame_control(
    frame_count: int,
    *,
    condition: str,
    order_seed: int,
) -> FrameControl:
    """Build a real-frame control while retaining meaningful time positions."""

    if frame_count <= 0 or order_seed < 0:
        raise ValueError("invalid process-control request")
    natural = torch.arange(frame_count)
    if condition in {"correct", "same_task_other", "cross_suite_wrong"}:
        content = positions = natural
    elif condition == "reversed":
        content, positions = natural.flip(0), natural
    elif condition in {"shuffled", "shuffled_keep_first"}:
        content = shuffled_frame_permutation(
            frame_count,
            order_seed,
            keep_first=condition == "shuffled_keep_first",
        )
        positions = natural
    elif condition == "endpoints_middle_shuffled":
        content = _endpoints_middle_shuffled(frame_count, order_seed)
        positions = natural
    elif condition == "first_frame_only":
        content = positions = natural[:1]
    elif condition == "final_frame_only":
        content = positions = natural[-1:]
    elif condition == "first_final":
        content = positions = natural if frame_count == 1 else natural[[0, -1]]
    elif condition == "monotone_sparse":
        content = positions = _monotone_sparse(frame_count)
    elif condition == "static_first_repeated":
        content = torch.zeros(frame_count, dtype=torch.long)
        positions = natural
    else:
        raise ValueError(f"unsupported process-control condition: {condition}")
    return FrameControl(content=content, positions=positions)
