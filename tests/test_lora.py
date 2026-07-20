from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ember.lora import (
    LORA_B_SUFFIX,
    LoRAContractError,
    LoRATarget,
    SmolVLALoRAContract,
    copy_task_lora_state_,
    derive_smolvla_targets,
    functional_lora_call,
    inject_task_lora,
    load_lora_contract,
    task_lora_state_dict,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 4, bias=False)
        self.v_proj = torch.nn.Linear(3, 2, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(16)])


class _VLMWithExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_expert = _Expert()


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = _VLMWithExpert()
        self.state_proj = torch.nn.Linear(3, 4, bias=False)
        self.action_in_proj = torch.nn.Linear(3, 4, bias=False)
        self.action_out_proj = torch.nn.Linear(4, 3, bias=False)
        self.action_time_mlp_in = torch.nn.Linear(3, 4, bias=False)
        self.action_time_mlp_out = torch.nn.Linear(4, 4, bias=False)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model.state_proj(value)


def _small_contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("model.state_proj", 3, 4),),
        rank=2,
        alpha=1,
        dropout=0.0,
        identity_seed=17,
    )


def test_sealed_contract_has_exact_canonical_capacity() -> None:
    contract = load_lora_contract(REPO_ROOT / "configs/smolvla_lora_v1.json")
    assert len(contract.targets) == 37
    assert contract.state_tensor_count == 74
    assert contract.parameter_count == 1_485_312
    assert contract.rank == 32
    assert contract.alpha == 16
    assert contract.dropout == 0.0


def test_derivation_requires_all_37_named_linears() -> None:
    policy = _Policy()
    targets = derive_smolvla_targets(policy)
    assert len(targets) == 37
    assert targets[0].name.endswith("layers.0.self_attn.q_proj")
    assert targets[-1].name == "model.action_time_mlp_out"
    policy.model.vlm_with_expert.lm_expert.layers[3].self_attn.q_proj = torch.nn.Identity()
    with pytest.raises(LoRAContractError, match="not a Linear"):
        derive_smolvla_targets(policy)


def test_injected_identity_is_only_trainable_state_and_is_repeatable() -> None:
    contract = _small_contract()
    first = inject_task_lora(_Policy(), contract)
    second = inject_task_lora(_Policy(), contract)
    first_state = task_lora_state_dict(first, clone=True)
    second_state = task_lora_state_dict(second, clone=True)
    assert set(first_state) == set(second_state)
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in first_state.items()
        if name.endswith(LORA_B_SUFFIX)
    )
    assert {name for name, value in first.named_parameters() if value.requires_grad} == set(
        first_state
    )


def test_functional_state_has_gradient_and_copy_is_exact() -> None:
    contract = _small_contract()
    policy = inject_task_lora(_Policy(), contract)
    state = task_lora_state_dict(policy, clone=True)
    generated = {name: value.requires_grad_() for name, value in state.items()}
    b_name = next(name for name in generated if name.endswith(LORA_B_SUFFIX))
    generated[b_name] = torch.ones_like(generated[b_name], requires_grad=True)
    output = functional_lora_call(policy, generated, contract, torch.ones(2, 3))
    output.sum().backward()
    assert generated[b_name].grad is not None

    zero = {name: torch.zeros_like(value) for name, value in state.items()}
    copy_task_lora_state_(policy, zero, contract)
    assert all(torch.count_nonzero(value) == 0 for value in task_lora_state_dict(policy).values())
