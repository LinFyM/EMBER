"""Matched raw frozen-Stage0 input for the G3 sufficiency diagnostic."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ember.ecp.natural_program import NaturalProgram, NaturalProgramModel


RAW_STAGE0_PROGRAM_INPUT = "raw_frozen_stage0_evidence"


@dataclass(frozen=True)
class RawStage0ProgramOutput:
    """Batched raw view retained only for existing training diagnostics."""

    program: NaturalProgram


def prepare_raw_stage0_primal_condition(
    *,
    program_model: NaturalProgramModel,
    condition: Any,
    query_times: torch.Tensor,
) -> tuple[NaturalProgram, RawStage0ProgramOutput]:
    """Expose frozen Stage0 evidence without Natural Program process compression.

    The returned ``NaturalProgram`` is only the fixed-shape carrier consumed by
    the unchanged primal scorer.  Its dynamic fields come directly from the two
    frozen Stage0 probes; no process fusion, canonical aligner, or learned
    video aggregation is executed.  K=1 makes the local event order canonical.
    """

    evidence = getattr(condition, "evidence", None)
    videos = getattr(condition, "videos", ())
    events = int(program_model.event_slots)
    owners = int(program_model.owners)
    width = int(program_model.width)
    if (
        evidence is None
        or len(videos) != 1
        or query_times.ndim != 2
        or evidence.language_embeddings.ndim != 3
        or evidence.language_embeddings.shape[0] != 1
        or evidence.language_mask.shape != evidence.language_embeddings.shape[:2]
        or evidence.patch_states.ndim != 4
        or evidence.patch_states.shape[0] != 1
        or evidence.patch_states.shape[-1] != width
        or evidence.frame_mask.shape != evidence.patch_states.shape[:2]
        or evidence.process.shape != (2, 1, events, owners, width)
        or evidence.uncertainty.shape != (2, 1, events, owners, width)
        or evidence.presence.shape != (2, 1, events)
        or evidence.state_posterior.shape[:2] != (2, 1)
        or evidence.state_posterior.shape[-1] != events
        or evidence.video_offsets.numel() != 2
        or evidence.video_set_offsets.detach().cpu().tolist() != [0, 1]
    ):
        raise ValueError("raw Stage0 sufficiency input changed its K1 schema")

    device = evidence.language_embeddings.device
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        # These are dimension-matched shared readers, not a task/frame lookup.
        # They expose the exact language embedding and first/final patch relation
        # to the same scorer width used by the Natural Program arm.
        p_lang = program_model.language_reader(
            evidence.language_embeddings, evidence.language_mask
        )
        video_condition_ids = evidence.frame_condition_ids.index_select(
            0, evidence.video_offsets[:-1].to(evidence.frame_condition_ids.device)
        )
        p_scene = program_model.scene_reader(
            evidence.patch_states,
            evidence.frame_mask,
            p_lang,
            video_condition_ids,
        )

    process = evidence.process.float()
    mean_process = process.mean(0)
    sigma = (
        evidence.uncertainty.float().square().mean(0)
        + (process - mean_process[None]).square().mean(0)
    ).clamp_min(1e-6).sqrt()
    rho = evidence.presence.float().mean(0)
    positions = program_model._padded_positions(
        evidence.frame_indices,
        evidence.raw_frame_counts,
        evidence.video_offsets,
        evidence.frame_mask,
    )
    tau = program_model._temporal_moments(
        evidence.state_posterior.float(), positions, evidence.frame_mask
    )
    batched = NaturalProgram(
        p_lang=p_lang.float(),
        p_scene=p_scene.float(),
        p_process=mean_process,
        rho=rho,
        tau=tau,
        sigma=sigma,
    )
    program = NaturalProgram(
        p_lang=batched.p_lang[0],
        p_scene=batched.p_scene[0],
        p_process=batched.p_process[0],
        rho=batched.rho[0],
        tau=batched.tau[0],
        sigma=batched.sigma[0],
    )
    return program, RawStage0ProgramOutput(program=batched)
