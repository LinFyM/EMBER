"""Training-only successful task-expert behavior authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.expert_manifold.contract import (
    authority_path,
    load_task_expert_config,
)
from ember.lora import validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.reward.protocol import RewardTask
from ember.writer.errors import WriterModelError


def pad_expert_lora_to_public_rank(
    state: Mapping[str, torch.Tensor],
    *,
    expert_contract: Any,
    public_contract: Any,
) -> dict[str, torch.Tensor]:
    """Zero-pad rank16 expert factors into the scale-one rank32 policy."""

    validate_lora_state(state, expert_contract)
    if (
        int(public_contract.rank) != 2 * int(expert_contract.rank)
        or int(public_contract.alpha) != int(public_contract.rank)
        or int(expert_contract.alpha) != int(expert_contract.rank)
        or tuple(public_contract.targets) != tuple(expert_contract.targets)
    ):
        raise WriterModelError("successful expert public-rank topology changed")
    result: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if name.endswith(".lora_A.default.weight"):
            result[name] = torch.cat((value, torch.zeros_like(value)), dim=0)
        elif name.endswith(".lora_B.default.weight"):
            result[name] = torch.cat((value, torch.zeros_like(value)), dim=1)
        else:
            raise WriterModelError("successful expert state contains a non-LoRA tensor")
    validate_lora_state(result, public_contract)
    return result


def load_successful_expert_bank(
    *,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    tasks: Sequence[RewardTask],
    public_contract: Any,
) -> tuple[dict[str, Any], dict[int, Mapping[str, torch.Tensor]]]:
    """Inspect once and load the complete train24 expert bank on CPU."""

    teacher = config["privileged_teacher"]
    config_path = Path(str(config["resolved_task_expert_config"]))
    bank_root = Path(str(config["resolved_task_expert_bank_root"]))
    step = int(teacher["step"])
    evidence = inspect_task_expert_bank(
        config_path=config_path,
        bank_root=bank_root,
        step=step,
        source=source,
        task_keys=tuple((task.suite, task.task_id) for task in tasks),
        evaluation_role="development_train",
        require_formal=True,
    )
    expert_authority = load_task_expert_config(config_path)
    expert_config = load_pi05_lora_contract(
        authority_path(expert_authority, "lora_contract")
    )
    if int(expert_config.rank) != int(teacher["rank"]):
        raise WriterModelError("successful expert rank authority changed")
    states: dict[int, Mapping[str, torch.Tensor]] = {}
    for row in evidence["tasks"]:
        global_task_id = int(row["global_task_id"])
        checkpoint = Path(str(row["checkpoint"]))
        state = load_file(str(checkpoint / "adapter.safetensors"), device="cpu")
        states[global_task_id] = pad_expert_lora_to_public_rank(
            state,
            expert_contract=expert_config,
            public_contract=public_contract,
        )
    expected = {task.global_task_id for task in tasks}
    if set(states) != expected:
        raise WriterModelError("successful expert bank lost train24 coverage")
    return evidence, states
