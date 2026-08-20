from dataclasses import dataclass

import torch

from ember.functional_adaptation.objectives import (
    effective_update_exact_loss,
    effective_update_probe_loss,
    effective_update_probes,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget


@dataclass(frozen=True)
class TinyContract:
    targets: tuple[LoRATarget, ...] = (LoRATarget("model.q_proj", 5, 4),)
    rank: int = 2
    alpha: int = 2
    dropout: float = 0.0
    identity_seed: int = 7

    @property
    def parameter_count(self) -> int:
        return 18

    @property
    def state_tensor_count(self) -> int:
        return 2

    def to_dict(self) -> dict:
        return {}


def _state(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "model.q_proj" + LORA_A_SUFFIX: a,
        "model.q_proj" + LORA_B_SUFFIX: b,
    }


def test_effective_update_probe_loss_is_gauge_invariant() -> None:
    contract = TinyContract()
    probes = effective_update_probes(contract, probe_count=3, seed=11, device="cpu")
    a = torch.randn(2, 5)
    b = torch.randn(4, 2)
    target = _state(a, b)
    equivalent = _state(2.0 * a, 0.5 * b)

    loss = effective_update_probe_loss(equivalent, target, contract, probes)

    assert loss.item() < 1e-12


def test_effective_update_probe_loss_backpropagates_from_identity() -> None:
    contract = TinyContract()
    probes = effective_update_probes(contract, probe_count=3, seed=11, device="cpu")
    a = torch.randn(2, 5)
    target = _state(a, torch.ones(4, 2))
    candidate_b = torch.zeros(4, 2, requires_grad=True)
    candidate = _state(a, candidate_b)

    loss = effective_update_probe_loss(candidate, target, contract, probes)
    loss.backward()

    assert loss.item() > 0.9
    assert candidate_b.grad is not None
    assert torch.count_nonzero(candidate_b.grad)


def test_effective_update_exact_loss_matches_dense_ba_and_backpropagates() -> None:
    contract = TinyContract()
    target_a = torch.randn(2, 5)
    target_b = torch.randn(4, 2)
    candidate_a = torch.randn(2, 5)
    candidate_b = torch.randn(4, 2, requires_grad=True)
    target = _state(target_a, target_b)
    candidate = _state(candidate_a, candidate_b)

    loss = effective_update_exact_loss(candidate, target, contract)
    expected = (candidate_b @ candidate_a - target_b @ target_a).square().sum()
    expected = expected / (target_b @ target_a).square().sum()
    loss.backward()

    assert torch.allclose(loss, expected, rtol=1e-5, atol=1e-6)
    assert candidate_b.grad is not None
    assert torch.count_nonzero(candidate_b.grad)
