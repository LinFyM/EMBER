"""Frozen privileged Program coordinates for compiler-only Stage 1 training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from ember.ecp.program import ECPProgram
from ember.ecp.stage1_data import pack_stage1_videos


def fixed_program_ordinals(
    config: Mapping[str, Any], *, mode: str
) -> tuple[int, ...]:
    fit = tuple(int(value) for value in config["roles"]["fit_task_ordinals"])
    return fit[:1] if mode == "profile" else fit


def capture_fixed_q_pi_programs(
    *,
    mode: str,
    config: Mapping[str, Any],
    context: Any,
    inputs: Any,
    policy: torch.nn.Module,
    observer: Any,
    model: torch.nn.Module,
    support_bank: Any,
    language_tokens: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, ECPProgram]:
    """Capture one q_pi coordinate per fit task; none is trainable or deployed."""

    ordinals = fixed_program_ordinals(config, mode=mode)
    owners = int(config["model"]["target_owners"])
    events = int(config["model"]["event_slots"])
    width = int(config["model"]["program_width"])
    count = len(ordinals)
    tensors = {
        "language": torch.zeros(count, owners, width, device=context.device),
        "scene": torch.zeros(count, owners, width, device=context.device),
        "process": torch.zeros(count, events, owners, width, device=context.device),
        "presence": torch.zeros(count, events, device=context.device),
        "uncertainty": torch.zeros(
            count, events, owners, width, device=context.device
        ),
    }
    coverage = torch.zeros(count, device=context.device)
    visit = int(config["fixed_program_coordinate"]["video_visit"])
    expert = policy.model.paligemma_with_expert.gemma_expert.model
    try:
        for index, ordinal in enumerate(ordinals):
            if index % context.world_size != context.rank:
                continue
            packed = pack_stage1_videos(
                store=inputs.video_store,
                ordinal=ordinal,
                visit=visit,
                seed=int(config["data"]["pair_seed"]),
                k=int(config["data"]["visible_videos_per_visit"]),
                device=context.device,
            )
            tokens, mask = language_tokens[ordinal]
            with torch.no_grad(), observer.action_meta.installed(expert):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    encoded = observer.model.encoder(
                        policy=policy,
                        frames=packed.frames,
                        video_offsets=packed.video_offsets,
                        frame_condition_ids=packed.frame_condition_ids,
                        language_tokens=tokens,
                        language_mask=mask,
                    )
                    evidence = inputs.evidence_bank.evidence(
                        ordinal, support_bank.task(ordinal)
                    )
                    anchors = model.visible_program(
                        encoded, packed.video_group_ids, group_count=1
                    )
                    program = model.policy_teacher(anchors, evidence).program
            for name, target in tensors.items():
                target[index].copy_(getattr(program, name)[0].float())
            coverage[index] = 1.0
    finally:
        inputs.video_store.close()
    if context.world_size > 1:
        for value in (*tensors.values(), coverage):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if not bool(torch.all(coverage == 1.0)):
        raise RuntimeError("fixed q_pi Program capture did not cover every fit task")
    return {
        ordinal: ECPProgram(
            language=tensors["language"][index : index + 1],
            scene=tensors["scene"][index : index + 1],
            process=tensors["process"][index : index + 1],
            presence=tensors["presence"][index : index + 1],
            uncertainty=tensors["uncertainty"][index : index + 1],
        )
        for index, ordinal in enumerate(ordinals)
    }
