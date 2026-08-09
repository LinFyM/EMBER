from __future__ import annotations

import torch

from ember.expert_manifold.effective_objective import (
    effective_alignment,
    effective_auxiliary_output_gradients,
    effective_condition_local_tangent_tube_loss,
    effective_counterfactual_ranking_loss,
    effective_projection_loss,
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


def _dense_contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("only", in_features=3, out_features=2),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=11,
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


def _matrix_state(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "only" + LORA_A_SUFFIX: matrix,
        "only" + LORA_B_SUFFIX: torch.eye(
            2,
            dtype=matrix.dtype,
            device=matrix.device,
            requires_grad=matrix.requires_grad,
        ),
    }


def _gauge_state(
    state: dict[str, torch.Tensor],
    contract: SmolVLALoRAContract,
    gauges: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    changed = {}
    for target, gauge in zip(contract.targets, gauges, strict=True):
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        changed[a_name] = torch.linalg.solve(gauge, state[a_name])
        changed[b_name] = state[b_name] @ gauge
    return changed


def test_effective_alignment_and_projection_match_explicit_ba_matrices() -> None:
    contract = _contract()
    generated = _state(contract, 11)
    target = _state(contract, 13)
    observed = effective_alignment(generated, target, contract)
    left = _flat_effective(generated, contract)
    right = _flat_effective(target, contract)
    expected_coefficient = torch.dot(left, right) / right.square().sum()
    assert torch.allclose(observed.generated_norm, torch.linalg.vector_norm(left))
    assert torch.allclose(observed.target_norm, torch.linalg.vector_norm(right))
    assert torch.allclose(observed.inner_product, torch.dot(left, right))
    assert torch.allclose(
        observed.cosine,
        torch.nn.functional.cosine_similarity(left[None], right[None])[0],
    )
    assert torch.allclose(
        observed.projection_coefficient,
        expected_coefficient,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        observed.per_target_inner_product.sum(),
        observed.inner_product,
    )
    assert torch.allclose(
        observed.per_target_target_norm_sq.sum(),
        observed.target_norm.square(),
    )


def test_projection_is_invariant_to_independent_generated_and_expert_gauges() -> None:
    contract = _contract()
    generated = _state(contract, 17)
    target = _state(contract, 19)
    generated_changed = {}
    target_changed = {}
    generated_gauge = torch.tensor([[1.4, 0.2], [-0.3, 0.9]])
    target_gauge = torch.tensor([[0.8, -0.4], [0.1, 1.3]])
    for item in contract.targets:
        a_name = item.name + LORA_A_SUFFIX
        b_name = item.name + LORA_B_SUFFIX
        generated_changed[a_name] = torch.linalg.solve(
            generated_gauge, generated[a_name]
        )
        generated_changed[b_name] = generated[b_name] @ generated_gauge
        target_changed[a_name] = torch.linalg.solve(target_gauge, target[a_name])
        target_changed[b_name] = target[b_name] @ target_gauge
    original = effective_projection_loss(
        generated, target, contract, smooth_l1_beta=0.5
    )
    changed = effective_projection_loss(
        generated_changed, target_changed, contract, smooth_l1_beta=0.5
    )
    assert torch.allclose(original.total, changed.total, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        original.coefficient,
        changed.coefficient,
        atol=2e-6,
        rtol=2e-6,
    )


def test_projection_only_completes_expert_component() -> None:
    contract = _dense_contract()
    expert = torch.tensor([[1.2, -0.7, 0.4], [-0.3, 0.8, 1.1]])
    candidate = torch.tensor([[0.2, 1.0, -0.5], [0.6, -0.4, 0.3]])
    orthogonal = candidate - (
        (candidate * expert).sum() / expert.square().sum()
    ) * expert
    target = _matrix_state(expert)
    losses = []
    for orthogonal_scale in (0.0, 2.0, 20.0):
        generated = _matrix_state(0.8 * expert + orthogonal_scale * orthogonal)
        projection = effective_projection_loss(
            generated, target, contract, smooth_l1_beta=0.5
        )
        assert torch.allclose(
            projection.coefficient,
            torch.tensor(0.8),
            atol=2e-6,
            rtol=2e-6,
        )
        losses.append(projection.total)
    assert all(torch.allclose(losses[0], item, atol=2e-6, rtol=2e-6) for item in losses)


def test_projection_direct_effective_gradient_is_parallel_to_expert() -> None:
    contract = _dense_contract()
    expert = torch.tensor([[1.2, -0.7, 0.4], [-0.3, 0.8, 1.1]])
    generated_matrix = (0.8 * expert).requires_grad_()
    generated = _matrix_state(generated_matrix)
    target = _matrix_state(expert)
    loss = effective_projection_loss(
        generated, target, contract, smooth_l1_beta=0.5
    )
    (gradient,) = torch.autograd.grad(loss.total, generated_matrix)
    expected = ((0.8 - 1.0) / 0.5) * expert / expert.square().sum()
    assert torch.allclose(gradient, expected, atol=2e-6, rtol=2e-6)
    residual = gradient - (gradient * expert).sum() / expert.square().sum() * expert
    assert torch.linalg.vector_norm(residual) < 2e-6


def test_coefficient_ranking_uses_language_task_expert_and_correct_signs() -> None:
    contract = _dense_contract()
    expert = torch.tensor([[1.2, -0.7, 0.4], [-0.3, 0.8, 1.1]])
    correct_matrix = (0.8 * expert).requires_grad_()
    negative_matrix = (0.7 * expert).requires_grad_()
    ranking = effective_counterfactual_ranking_loss(
        _matrix_state(correct_matrix),
        _matrix_state(negative_matrix),
        _matrix_state(expert),
        contract,
        required_margin=0.2,
        temperature=0.1,
    )
    expected = torch.nn.functional.softplus(torch.tensor(1.0)) * 0.1
    assert torch.allclose(ranking.margin, torch.tensor(0.1), atol=2e-6)
    assert torch.allclose(ranking.loss, expected, atol=2e-6)
    correct_gradient, negative_gradient = torch.autograd.grad(
        ranking.loss, (correct_matrix, negative_matrix)
    )
    assert (correct_gradient * expert).sum() < 0
    assert (negative_gradient * expert).sum() > 0

    unrelated = -2.0 * expert
    unchanged = effective_counterfactual_ranking_loss(
        _matrix_state(correct_matrix.detach()),
        _matrix_state(negative_matrix.detach()),
        _matrix_state(expert),
        contract,
        required_margin=0.2,
        temperature=0.1,
    )
    wrong_task_target = effective_counterfactual_ranking_loss(
        _matrix_state(correct_matrix.detach()),
        _matrix_state(negative_matrix.detach()),
        _matrix_state(unrelated),
        contract,
        required_margin=0.2,
        temperature=0.1,
    )
    assert torch.allclose(unchanged.margin, ranking.margin)
    assert not torch.allclose(wrong_task_target.margin, ranking.margin)


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
        [
            effective_alignment(value, target, contract).projection_coefficient
            for value in (first, second)
        ]
    )
    assert observed.projection_coefficient.shape == (2,)
    assert observed.per_target_inner_product.shape == (2, 2)
    assert torch.allclose(observed.projection_coefficient, expected)


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
    correct_anchor = _state(contract, 71)
    negative_anchor = _state(contract, 73)
    output = effective_auxiliary_output_gradients(
        correct,
        negative,
        correct_anchor,
        negative_anchor,
        target,
        contract,
        smooth_l1_beta=0.5,
        required_margin=0.2,
        temperature=0.1,
    )
    names = tuple(correct)
    torch.autograd.backward(
        tuple(correct[name] for name in names)
        + tuple(negative[name] for name in names),
        tuple(
            output.correct_projection[name] + output.correct_ranking[name]
            for name in names
        )
        + tuple(
            output.counterfactual_projection[name]
            + output.counterfactual_ranking[name]
            for name in names
        ),
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
    projection = effective_condition_local_tangent_tube_loss(
        direct_correct,
        direct_negative,
        correct_anchor,
        negative_anchor,
        target,
        contract,
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
        projection.total + ranking.loss,
        (*direct_correct_parameters.values(), *direct_negative_parameters.values()),
    )
    assert all(
        torch.allclose(left, right, atol=1e-6, rtol=1e-5)
        for left, right in zip(observed, expected, strict=True)
    )


def test_tangent_three_state_low_rank_matches_dense_global_oracle() -> None:
    contract = _contract()
    correct = _state(contract, 79)
    negative = _state(contract, 83)
    correct_anchor = _state(contract, 89)
    negative_anchor = _state(contract, 97)
    expert = _state(contract, 101)
    observed = effective_condition_local_tangent_tube_loss(
        correct,
        negative,
        correct_anchor,
        negative_anchor,
        expert,
        contract,
        smooth_l1_beta=0.5,
    )

    expert_dense = _flat_effective(expert, contract)
    denominator = expert_dense.square().sum()
    for state, anchor, tangent in (
        (correct, correct_anchor, observed.correct),
        (negative, negative_anchor, observed.counterfactual),
    ):
        delta = _flat_effective(state, contract) - _flat_effective(anchor, contract)
        projection = torch.dot(delta, expert_dense)
        expected_d = projection / denominator
        expected_orthogonal_sq = (
            delta.square().sum() - projection.square() / denominator
        ).clamp_min(0)
        assert torch.allclose(tangent.delta_norm, delta.norm(), atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            tangent.directional_coefficient,
            expected_d,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            tangent.orthogonal_delta_norm.square(),
            expected_orthogonal_sq,
            atol=5e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            tangent.loss,
            expected_orthogonal_sq / denominator,
            atol=5e-5,
            rtol=2e-5,
        )


def test_tangent_tube_is_independently_gauge_invariant() -> None:
    contract = _contract()
    states = (
        _state(contract, 103),
        _state(contract, 107),
        _state(contract, 109),
        _state(contract, 113),
        _state(contract, 127),
    )
    original = effective_condition_local_tangent_tube_loss(
        *states,
        contract,
        smooth_l1_beta=0.5,
    )
    gauges = tuple(
        (
            torch.tensor([[1.1 + 0.03 * index, 0.2], [-0.1, 0.9]]),
            torch.tensor([[0.8, -0.15], [0.05, 1.2 + 0.02 * index]]),
        )
        for index in range(len(states))
    )
    changed_states = tuple(
        _gauge_state(state, contract, state_gauges)
        for state, state_gauges in zip(states, gauges, strict=True)
    )
    changed = effective_condition_local_tangent_tube_loss(
        *changed_states,
        contract,
        smooth_l1_beta=0.5,
    )
    for left, right in (
        (original.total, changed.total),
        (original.tube, changed.tube),
        (
            original.correct.directional_coefficient,
            changed.correct.directional_coefficient,
        ),
        (
            original.correct.orthogonal_delta_norm,
            changed.correct.orthogonal_delta_norm,
        ),
        (
            original.counterfactual.directional_coefficient,
            changed.counterfactual.directional_coefficient,
        ),
        (
            original.counterfactual.orthogonal_delta_norm,
            changed.counterfactual.orthogonal_delta_norm,
        ),
    ):
        assert torch.allclose(left, right, atol=2e-4, rtol=2e-5)


def test_tangent_macro0_anchor_is_zero_and_preserves_ecp_projection_gradient() -> None:
    contract = _dense_contract()
    expert = torch.tensor([[1.2, -0.7, 0.4], [-0.3, 0.8, 1.1]])
    correct_matrix = torch.tensor(
        [[0.4, 0.2, -0.1], [0.3, -0.5, 0.7]], requires_grad=True
    )
    negative_matrix = torch.tensor(
        [[-0.2, 0.5, 0.1], [0.6, 0.3, -0.4]], requires_grad=True
    )
    correct = _matrix_state(correct_matrix)
    negative = _matrix_state(negative_matrix)
    correct_anchor = _matrix_state(correct_matrix.detach().clone())
    negative_anchor = _matrix_state(negative_matrix.detach().clone())
    target = _matrix_state(expert)
    auxiliary = effective_auxiliary_output_gradients(
        correct,
        negative,
        correct_anchor,
        negative_anchor,
        target,
        contract,
        smooth_l1_beta=0.5,
        required_margin=0.2,
        temperature=0.1,
    )
    legacy = effective_projection_loss(
        correct,
        target,
        contract,
        smooth_l1_beta=0.5,
    )
    (legacy_gradient,) = torch.autograd.grad(legacy.total, correct_matrix)
    assert torch.equal(auxiliary.projection.tube, torch.zeros_like(auxiliary.projection.tube))
    assert torch.equal(
        auxiliary.projection.correct.orthogonal_delta_norm,
        torch.zeros_like(auxiliary.projection.correct.orthogonal_delta_norm),
    )
    assert torch.equal(
        auxiliary.projection.counterfactual.orthogonal_delta_norm,
        torch.zeros_like(auxiliary.projection.counterfactual.orthogonal_delta_norm),
    )
    assert torch.allclose(
        auxiliary.correct_projection["only" + LORA_A_SUFFIX],
        legacy_gradient,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.count_nonzero(
        auxiliary.counterfactual_projection["only" + LORA_A_SUFFIX]
    ) == 0


def test_parallel_and_orthogonal_tangent_geometry_have_exact_roles() -> None:
    contract = _dense_contract()
    expert = torch.tensor(
        [[1.2, -0.7, 0.4], [-0.3, 0.8, 1.1]], dtype=torch.float64
    )
    candidate = torch.tensor(
        [[0.2, 1.0, -0.5], [0.6, -0.4, 0.3]], dtype=torch.float64
    )
    orthogonal = candidate - (
        (candidate * expert).sum() / expert.square().sum()
    ) * expert
    anchor_matrix = 0.3 * expert
    parallel_matrix = (anchor_matrix + 0.4 * expert).requires_grad_()
    parallel = effective_condition_local_tangent_tube_loss(
        _matrix_state(parallel_matrix),
        _matrix_state(parallel_matrix.detach()),
        _matrix_state(anchor_matrix),
        _matrix_state(parallel_matrix.detach()),
        _matrix_state(expert),
        contract,
        smooth_l1_beta=0.5,
    )
    (parallel_gradient,) = torch.autograd.grad(
        parallel.correct.loss,
        parallel_matrix,
    )
    assert parallel.correct.orthogonal_delta_norm < 2e-7
    assert torch.linalg.vector_norm(parallel_gradient) < 2e-7

    orthogonal_matrix = (
        anchor_matrix + 0.4 * expert + 0.25 * orthogonal
    ).requires_grad_()
    changed = effective_condition_local_tangent_tube_loss(
        _matrix_state(orthogonal_matrix),
        _matrix_state(parallel_matrix.detach()),
        _matrix_state(anchor_matrix),
        _matrix_state(parallel_matrix.detach()),
        _matrix_state(expert),
        contract,
        smooth_l1_beta=0.5,
    )
    (orthogonal_gradient,) = torch.autograd.grad(
        changed.correct.loss,
        orthogonal_matrix,
    )
    assert torch.allclose(
        changed.correct.orthogonal_delta_norm.square(),
        (0.25 * orthogonal).square().sum().float(),
        atol=2e-6,
        rtol=2e-6,
    )
    assert abs(float((orthogonal_gradient * expert).sum())) < 2e-6
    assert float((orthogonal_gradient * orthogonal).sum()) > 0


def test_two_arm_tube_is_mean_and_halves_each_arm_gradient() -> None:
    contract = _dense_contract()
    expert = torch.tensor([[1.2, -0.7, 0.4], [-0.3, 0.8, 1.1]])
    correct_matrix = torch.tensor(
        [[0.4, 0.2, -0.1], [0.3, -0.5, 0.7]], requires_grad=True
    )
    negative_matrix = torch.tensor(
        [[-0.2, 0.5, 0.1], [0.6, 0.3, -0.4]], requires_grad=True
    )
    value = effective_condition_local_tangent_tube_loss(
        _matrix_state(correct_matrix),
        _matrix_state(negative_matrix),
        _matrix_state(torch.zeros_like(correct_matrix)),
        _matrix_state(torch.zeros_like(negative_matrix)),
        _matrix_state(expert),
        contract,
        smooth_l1_beta=0.5,
    )
    correct_direct, negative_direct = torch.autograd.grad(
        (value.correct.loss, value.counterfactual.loss),
        (correct_matrix, negative_matrix),
        grad_outputs=(torch.ones(()), torch.ones(())),
        retain_graph=True,
    )
    correct_mean, negative_mean = torch.autograd.grad(
        value.tube,
        (correct_matrix, negative_matrix),
    )
    assert torch.allclose(
        value.tube,
        0.5 * (value.correct.loss + value.counterfactual.loss),
    )
    assert torch.allclose(correct_mean, 0.5 * correct_direct)
    assert torch.allclose(negative_mean, 0.5 * negative_direct)
