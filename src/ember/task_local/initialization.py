"""Create one immutable task-local LoRA initialization per task/seed/arm."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import torch

from ember.lora import (
    LoRAContract,
    initialize_identity_lora_,
    lora_state_sha256,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.reward.protocol import RewardProtocolError
from ember.task_local.contract import TaskLocalUnit, cohort_video_demo


class FrozenWriterGenerator(Protocol):
    arm: str
    writer_state_sha256: str

    def generate(
        self, *, global_task_id: int, demo_index: int
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]: ...


@torch.no_grad()
def prepare_unit_initialization(
    *,
    policy: torch.nn.Module,
    lora_contract: LoRAContract,
    config: Mapping[str, Any],
    unit: TaskLocalUnit,
    source_checkpoint_manifest_sha256: str,
    as_writer: FrozenWriterGenerator | None,
    rl_writer: FrozenWriterGenerator | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generate once; callers immediately publish the returned immutable bundle."""

    if unit.arm == "identity":
        initialize_identity_lora_(policy, lora_contract)
        state = task_lora_state_dict(policy, clone=True)
        writer_evidence: Mapping[str, Any] = {}
        teacher_used = False
        demo: int | None = None
        writer_sha: str | None = None
    else:
        generator = as_writer if unit.arm == "as_writer" else rl_writer
        if generator is None or generator.arm != unit.arm:
            raise RewardProtocolError(
                f"task-local {unit.arm} initialization artifact is unavailable"
            )
        demo = cohort_video_demo(config, unit)
        generated, writer_evidence = generator.generate(
            global_task_id=unit.global_task_id, demo_index=demo
        )
        validate_lora_state(generated, lora_contract)
        state = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in generated.items()
        }
        teacher_used = True
        writer_sha = generator.writer_state_sha256
    evidence = {
        "arm": unit.arm,
        "global_task_id": unit.global_task_id,
        "suite": unit.suite,
        "task_id": unit.task_id,
        "adaptation_seed": unit.adaptation_seed,
        "teacher_video_used": teacher_used,
        "teacher_demo_index": demo,
        "cohort_video_schedule_excludes_arm": True,
        "writer_state_sha256": writer_sha,
        "writer_evidence": dict(writer_evidence),
        "source_checkpoint_manifest_sha256": source_checkpoint_manifest_sha256,
        "stacked_source_sft": False,
        "initial_lora_state_sha256": lora_state_sha256(state),
    }
    return state, evidence
