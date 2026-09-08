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


def test_bfloat16_base_matches_physical_fp32_lora_at_rounding_boundary() -> None:
    contract = _contract()
    policy = inject_task_lora(_TinyPolicy(), contract)
    policy.proj.base_layer.bfloat16()
    with torch.no_grad():
        policy.proj.base_layer.weight.zero_()
        policy.proj.base_layer.weight[:, 0] = 1
    identity = task_lora_state_dict(policy, clone=True)
    state = {name: torch.zeros_like(value) for name, value in identity.items()}
    a, b = state.values()
    a[0, 0] = 1
    b[:, 0] = 0.00391
    value = torch.tensor([[[1, 0, 0, 0, 0]]], dtype=torch.bfloat16)
    copy_task_lora_state_(policy, state, contract)
    expected = policy(value)
    assert bool((expected != 1).all())
    copy_task_lora_state_(policy, identity, contract)
    batched = BatchedLoRAInference(policy, contract)
    with batched.activate([state]):
        actual = policy(value)
    batched.close()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
