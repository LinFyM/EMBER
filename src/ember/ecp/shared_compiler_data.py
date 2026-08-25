"""Frozen G2 Program and real native-bank assembly for G3."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ember.ecp.contracts import TargetOwner
from ember.ecp.g1_video import prepare_native_video_readout
from ember.ecp.natural_program import NaturalProgram, NaturalProgramModel
from ember.ecp.natural_program_data import PackedNaturalProgramCondition
from ember.ecp.shared_compiler import SharedCompilerVideo


@dataclass(frozen=True)
class SharedCompilerCondition:
    program: NaturalProgram
    videos: tuple[SharedCompilerVideo, ...]
    metrics: dict[str, object]


def _ordinary(value: torch.Tensor) -> torch.Tensor:
    """Detach frozen Pass-A evidence as a normal FP32 autograd input."""

    return value.detach().float().clone()


def prepare_shared_compiler_condition(
    *,
    policy: torch.nn.Module,
    program_model: NaturalProgramModel,
    owners: tuple[TargetOwner, ...],
    packed: PackedNaturalProgramCondition,
    language_tokens: torch.Tensor,
    language_mask: torch.Tensor,
    chunk_size: int,
) -> SharedCompilerCondition:
    """Run frozen Pass A once, then expose each video's chunked native Pass B."""

    boundaries = tuple(map(int, packed.video_offsets.detach().cpu().tolist()))
    set_boundaries = tuple(
        map(int, packed.video_set_offsets.detach().cpu().tolist())
    )
    if (
        set_boundaries != (0, len(boundaries) - 1)
        or len(boundaries) not in (2, 3, 5)
        or packed.frame_condition_ids.shape != (packed.frames.shape[0],)
        or torch.count_nonzero(packed.frame_condition_ids).item() != 0
        or language_tokens.shape[0] != 1
        or language_mask.shape != language_tokens.shape
        or chunk_size <= 0
    ):
        raise ValueError("G3 compiler condition is not one K={1,2,4} task")
    device = packed.frames.device
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        output = program_model(
            policy=policy,
            frames=packed.frames,
            frame_indices=packed.frame_indices,
            raw_frame_counts=packed.raw_frame_counts,
            video_offsets=packed.video_offsets,
            video_set_offsets=packed.video_set_offsets,
            frame_condition_ids=packed.frame_condition_ids,
            language_tokens=language_tokens,
            language_mask=language_mask,
            query_times=packed.query_times,
        )
    program = NaturalProgram(
        p_lang=_ordinary(output.program.p_lang[0]),
        p_scene=_ordinary(output.program.p_scene[0]),
        p_process=_ordinary(output.program.p_process[0]),
        rho=_ordinary(output.program.rho[0]),
        tau=_ordinary(output.program.tau[0]),
        sigma=_ordinary(output.program.sigma[0]),
    )
    videos = []
    for video, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        count = stop - start
        raw_count = int(packed.raw_frame_counts[video])
        positions = packed.frame_indices[start:stop].float() / max(raw_count - 1, 1)
        canonical = _ordinary(output.canonical_assignment[video, :count])
        native = prepare_native_video_readout(
            policy=policy,
            encoder=program_model.encoder,
            owners=owners,
            frames=packed.frames[start:stop],
            tokens=language_tokens,
            masks=language_mask,
            process=_ordinary(output.local_process[video]),
            posterior=canonical,
            chunk_size=chunk_size,
        )
        videos.append(
            SharedCompilerVideo(
                native=native,
                canonical_assignment=canonical,
                frame_positions=_ordinary(positions),
                local_scene=_ordinary(output.local_scene[video]),
                local_process=_ordinary(output.local_process[video]),
                local_presence=_ordinary(output.local_presence[video]),
                local_tau=_ordinary(output.local_tau[video]),
                local_sigma=_ordinary(output.local_sigma[video]),
            )
        )
    return SharedCompilerCondition(
        program=program,
        videos=tuple(videos),
        metrics={
            **packed.metrics,
            "canonical_active_events": int(
                (program.rho > 0.2).sum().detach().cpu()
            ),
        },
    )
