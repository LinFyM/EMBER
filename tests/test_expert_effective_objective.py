from __future__ import annotations

import torch

from ember.expert_manifold.effective_objective import (
    effective_alignment,
    effective_auxiliary_output_gradients,
    effective_counterfactual_ranking_loss,
    effective_expert_loss,
)
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
)


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(
            LoRATarget("first", in_features=5, out_features=4),
            LoRATarget("second", in_features=3, out_features=6),
        ),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=7,
    )


def _state(contract: SmolVLALoRAContract, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for target in contract.targets:
        result[target.name + LORA_A_SUFFIX] = torch.randn(
            contract.rank, target.in_features, generator=generator
        )
        result[target.name + LORA_B_SUFFIX] = torch.randn(
            target.out_features, contract.rank, generator=generator
        )
    return result


def _flat_effective(
    state: dict[str, torch.Tensor], contract: SmolVLALoRAContract
) -> torch.Tensor:
    return torch.cat(
        [
            (
                state[target.name + LORA_B_SUFFIX]
                @ state[target.name + LORA_A_SUFFIX]
            ).flatten()
            for target in contract.targets
        ]
    )


def test_effective_alignment_matches_explicit_ba_matrices() -> None:
    contract = _contract()
    generated = _state(contract, 11)
    target = _state(contract, 13)
    observed = effective_alignment(generated, target, contract)
    left = _flat_effective(generated, contract)
    right = _flat_effective(target, contract)
    assert torch.allclose(observed.generated_norm, torch.linalg.vector_norm(left))
    assert torch.allclose(observed.target_norm, torch.linalg.vector_norm(right))
    assert torch.allclose(observed.inner_product, torch.dot(left, right))
    assert torch.allclose(
        observed.cosine,
        torch.nn.functional.cosine_similarity(left[None], right[None])[0],
    )


def test_effective_objective_is_invariant_to_invertible_lora_gauge() -> None:
    contract = _contract()
    generated = _state(contract, 17)
    target = _state(contract, 19)
    transformed = {}
    gauge = torch.tensor([[1.4, 0.2], [-0.3, 0.9]])
    inverse = torch.linalg.inv(gauge)
    for item in contract.targets:
        a_name = item.name + LORA_A_SUFFIX
        b_name = item.name + LORA_B_SUFFIX
        transformed[a_name] = gauge @ generated[a_name]
        transformed[b_name] = generated[b_name] @ inverse
    original = effective_expert_loss(
        generated, target, contract, norm_weight=0.25, smooth_l1_beta=0.5
    )
    changed = effective_expert_loss(
        transformed, target, contract, norm_weight=0.25, smooth_l1_beta=0.5
    )
    assert torch.allclose(original.total, changed.total, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        original.alignment.cosine,
        changed.alignment.cosine,
        atol=2e-6,
        rtol=2e-6,
    )


def test_effective_losses_have_finite_generated_lora_gradients() -> None:
    contract = _contract()
    correct = {
        name: value.requires_grad_() for name, value in _state(contract, 23).items()
    }
    negative = {
        name: value.requires_grad_() for name, value in _state(contract, 29).items()
    }
    target = _state(contract, 31)
    expert = effective_expert_loss(
        correct, target, contract, norm_weight=0.25, smooth_l1_beta=0.5
    )
    ranking = effective_counterfactual_ranking_loss(
        correct,
        negative,
        target,
        contract,
        required_margin=0.2,
        temperature=0.1,
    )
    (expert.total + ranking.loss).backward()
    gradients = [value.grad for value in (*correct.values(), *negative.values())]
    assert all(value is not None for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients if value is not None)
    assert sum(int(torch.count_nonzero(value)) for value in gradients if value is not None) > 0


def test_identical_correct_and_negative_have_zero_ranking_advantage() -> None:
    contract = _contract()
    correct = _state(contract, 37)
    target = _state(contract, 41)
    ranking = effective_counterfactual_ranking_loss(
        correct,
        {name: value.clone() for name, value in correct.items()},
        target,
        contract,
        required_margin=0.2,
        temperature=0.1,
    )
    assert torch.equal(ranking.margin, torch.zeros_like(ranking.margin))
    assert ranking.loss > 0


def test_effective_alignment_supports_generated_batches_and_shared_target() -> None:
    contract = _contract()
    first = _state(contract, 43)
    second = _state(contract, 47)
    batched = {
        name: torch.stack((first[name], second[name])) for name in first
    }
    target = _state(contract, 53)
    observed = effective_alignment(batched, target, contract)
    expected = torch.stack(
        [effective_alignment(value, target, contract).cosine for value in (first, second)]
    )
    assert observed.cosine.shape == (2,)
    assert torch.allclose(observed.cosine, expected)


def test_auxiliary_output_gradients_preserve_exact_parameter_chain_rule() -> None:
    contract = _contract()
    correct_parameters = {
        name: value.requires_grad_() for name, value in _state(contract, 59).items()
    }
    negative_parameters = {
        name: value.requires_grad_() for name, value in _state(contract, 61).items()
    }
    correct = {name: value * 1.1 for name, value in correct_parameters.items()}
    negative = {name: value * 0.9 for name, value in negative_parameters.items()}
    target = _state(contract, 67)
    output = effective_auxiliary_output_gradients(
        correct,
        negative,
        target,
        contract,
        norm_weight=0.25,
        smooth_l1_beta=0.5,
        required_margin=0.2,
        temperature=0.1,
    )
    names = tuple(correct)
    torch.autograd.backward(
        tuple(correct[name] for name in names)
        + tuple(negative[name] for name in names),
        tuple(
            output.correct_expert[name] + output.correct_ranking[name]
            for name in names
        )
        + tuple(output.counterfactual_ranking[name] for name in names),
    )
    observed = tuple(
        value.grad.detach().clone()
        for value in (*correct_parameters.values(), *negative_parameters.values())
    )

    direct_correct_parameters = {
        name: value.detach().clone().requires_grad_()
        for name, value in correct_parameters.items()
    }
    direct_negative_parameters = {
        name: value.detach().clone().requires_grad_()
        for name, value in negative_parameters.items()
    }
    direct_correct = {
        name: value * 1.1 for name, value in direct_correct_parameters.items()
    }
    direct_negative = {
        name: value * 0.9 for name, value in direct_negative_parameters.items()
    }
    expert = effective_expert_loss(
        direct_correct,
        target,
        contract,
        norm_weight=0.25,
        smooth_l1_beta=0.5,
    )
    ranking = effective_counterfactual_ranking_loss(
        direct_correct,
        direct_negative,
        target,
        contract,
        required_margin=0.2,
        temperature=0.1,
    )
    expected = torch.autograd.grad(
        expert.total + ranking.loss,
        (*direct_correct_parameters.values(), *direct_negative_parameters.values()),
    )
    assert all(
        torch.allclose(left, right, atol=1e-6, rtol=1e-5)
        for left, right in zip(observed, expected, strict=True)
    )
