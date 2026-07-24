from __future__ import annotations

import copy

import torch

from ember.batched_lora import BatchedLoRAInference
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    copy_task_lora_state_,
    inject_task_lora,
    task_lora_state_dict,
)


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(5, 3, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("proj", 5, 3),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=7,
    )


def test_per_sample_batched_lora_matches_materialized_policy() -> None:
    torch.manual_seed(11)
    contract = _contract()
    policy = inject_task_lora(_TinyPolicy(), contract)
    identity = copy.deepcopy(task_lora_state_dict(policy, clone=True))
    states = []
    for _ in range(3):
        states.append(
            {
                name: torch.randn_like(value)
                for name, value in identity.items()
            }
        )
    value = torch.randn(3, 4, 5)

    sequential = []
    for index, state in enumerate(states):
        copy_task_lora_state_(policy, state, contract)
        sequential.append(policy(value[index : index + 1]))
    sequential_value = torch.cat(sequential, dim=0)

    copy_task_lora_state_(policy, identity, contract)
    batched = BatchedLoRAInference(policy, contract)
    with batched.activate(states):
        batched_value = policy(value)
    batched.close()

    torch.testing.assert_close(batched_value, sequential_value)
