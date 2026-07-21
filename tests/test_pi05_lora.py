from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ember.lora import (
    LoRAContractError,
    LoRATarget,
    canonical_contract_sha256,
    copy_task_lora_state_,
    functional_lora_call,
)
from ember.pi05_lora import (
    Pi05LoRAContract,
    derive_pi05_targets,
    load_pi05_lora_contract,
    pi05_target_names,
)
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(1024, 2048, bias=False, device="meta")
        self.v_proj = torch.nn.Linear(1024, 256, bias=False, device="meta")


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _ExpertModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(18)])


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ExpertModel()


class _PaliGemmaWithExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gemma_expert = _Expert()


class _Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = _PaliGemmaWithExpert()
        self.action_in_proj = torch.nn.Linear(32, 1024, bias=False, device="meta")
        self.action_out_proj = torch.nn.Linear(1024, 32, bias=False, device="meta")


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()


def _tiny_contract() -> Pi05LoRAContract:
    sealed = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    return Pi05LoRAContract(
        targets=(LoRATarget("low", 3, 4), LoRATarget("high", 3, 4)),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=sealed.identity_seed,
        foundation_repository=sealed.foundation_repository,
        foundation_revision=sealed.foundation_revision,
        foundation_weights_sha256=sealed.foundation_weights_sha256,
        foundation_config_sha256=sealed.foundation_config_sha256,
        source_base_config_sha256=sealed.source_base_config_sha256,
        recipe_sha256=sealed.recipe_sha256,
    )


class _MixedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.low = torch.nn.Linear(3, 4, bias=False).to(torch.bfloat16)
        self.high = torch.nn.Linear(3, 4, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        value = batch["value"]
        combined = self.low(value.to(torch.bfloat16)).float() + self.high(value.float())
        loss = combined.square().mean()
        return loss, {"loss": float(loss.detach())}


def _writer(template: dict[str, torch.Tensor]) -> CompleteLoRAWriter:
    return CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        vision_feature_dim=7,
        language_feature_dim=5,
        hidden_dim=12,
        attention_heads=3,
        temporal_chunk_size=4,
        chunk_memory_tokens=2,
        episode_memory_tokens=2,
        task_memory_tokens=2,
        decoder_hidden_dim=10,
    )


def test_sealed_pi05_contract_has_exact_topology_and_capacity() -> None:
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    assert tuple(target.name for target in contract.targets) == pi05_target_names()
    assert len(contract.targets) == 38
    assert contract.state_tensor_count == 76
    assert contract.parameter_count == 1_287_168
    assert contract.rank == contract.alpha == 16
    assert contract.dropout == 0.0
    assert len(canonical_contract_sha256(contract)) == 64


def test_pi05_derivation_requires_all_exact_named_linears() -> None:
    policy = _Policy()
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    assert derive_pi05_targets(policy) == contract.targets
    policy.model.paligemma_with_expert.gemma_expert.model.layers[4].self_attn.v_proj = (
        torch.nn.Identity()
    )
    with pytest.raises(LoRAContractError, match="not a Linear"):
        derive_pi05_targets(policy)


def test_writer_preserves_mixed_lora_dtypes_and_matches_materialized_policy() -> None:
    torch.manual_seed(7)
    policy = _MixedPolicy()
    template = prepare_frozen_writer_policy(policy, _tiny_contract())
    assert {value.dtype for value in template.values()} == {torch.bfloat16, torch.float32}
    writer = _writer(template)
    with torch.no_grad():
        for head in writer.heads.values():
            head[-1].bias.fill_(0.01)
    generated = writer(
        torch.randn(3, 5),
        torch.randn(7, 7),
        torch.tensor([0, 7]),
    )
    assert {name: value.dtype for name, value in generated.items()} == {
        name: value.dtype for name, value in template.items()
    }

    materialized = copy.deepcopy(policy)
    copy_task_lora_state_(materialized, generated, _tiny_contract())
    batch = {"value": torch.randn(6, 3)}
    functional_loss, _ = functional_lora_call(
        policy, generated, _tiny_contract(), batch
    )
    materialized_loss, _ = materialized(batch)
    torch.testing.assert_close(functional_loss, materialized_loss, rtol=0.0, atol=0.0)
