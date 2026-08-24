from __future__ import annotations

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.g1_objective import (
    family_balanced_sensitivity_weights,
    global_member_effect_loss,
    low_rank_distance_squared,
    sensitivity_normalized_update_losses,
)
from ember.ecp.policy_effects import ExecutionPolicyPrefix, PolicyEffectResponse
from ember.ecp.g1_objective import G1EffectBank, VerifiedMemberObjective
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
)


def test_low_rank_distance_matches_explicit_matrix_distance() -> None:
    generator = torch.Generator().manual_seed(41)
    a = torch.randn(4, 7, generator=generator)
    b = torch.randn(5, 4, generator=generator)
    reference_a = torch.randn(4, 7, generator=generator)
    reference_b = torch.randn(5, 4, generator=generator)

    observed = low_rank_distance_squared(a, b, reference_a, reference_b)
    expected = (b @ a - reference_b @ reference_a).square().sum()

    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)


def test_sensitivity_weights_balance_the_four_target_families() -> None:
    owners = (
        TargetOwner(0, "q0", TargetFamily.Q, 0, 3, 4),
        TargetOwner(1, "q1", TargetFamily.Q, 1, 3, 4),
        TargetOwner(2, "v0", TargetFamily.V, 0, 3, 4),
        TargetOwner(3, "ain", TargetFamily.ACTION_IN, None, 3, 4),
        TargetOwner(4, "aout", TargetFamily.ACTION_OUT, None, 3, 4),
    )
    sensitivity = torch.tensor([[0.0, 8.0, 2.0, 4.0, 16.0], [5.0, 1.0, 0.0, 0.0, 3.0]])

    weights = family_balanced_sensitivity_weights(sensitivity, owners)

    assert torch.allclose(weights.sum(1), torch.ones(2))
    for family in TargetFamily:
        indices = [
            index for index, owner in enumerate(owners) if owner.family is family
        ]
        assert torch.allclose(weights[:, indices].sum(1), torch.full((2,), 0.25))
    assert torch.all(weights > 0)


def test_sensitivity_normalized_update_uses_effective_low_rank_matrices() -> None:
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("one", 3, 2), LoRATarget("two", 2, 4)),
        rank=16,
        alpha=16,
        dropout=0.0,
        identity_seed=9,
    )
    candidate: dict[str, torch.Tensor] = {}
    reference: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(73)
    expected = 0.0
    scales = torch.tensor([0.5, 2.0])
    weights = torch.tensor([[0.25, 0.75]])
    for index, target in enumerate(contract.targets):
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        candidate[a_name] = torch.randn(4, target.in_features, generator=generator)
        candidate[b_name] = torch.randn(target.out_features, 4, generator=generator)
        reference[a_name] = torch.randn(4, target.in_features, generator=generator)
        reference[b_name] = torch.randn(target.out_features, 4, generator=generator)
        delta = (
            candidate[b_name] @ candidate[a_name]
            - reference[b_name] @ reference[a_name]
        )
        expected += float(weights[0, index]) * (
            delta.square().mean() / scales[index].square()
        )

    observed = sensitivity_normalized_update_losses(
        candidate_state=candidate,
        reference_states=(reference,),
        contract=contract,
        s_ref=scales,
        sensitivity_weights=weights,
    )

    assert torch.allclose(observed, torch.tensor([expected]), atol=2e-5, rtol=2e-5)


def test_global_member_effect_is_set_valued_and_reliability_weighted() -> None:
    empty_response = PolicyEffectResponse(
        owner=torch.empty(0), flow=torch.empty(0), action=torch.empty(0)
    )
    bank = G1EffectBank(
        prefix=ExecutionPolicyPrefix(torch.empty(0), torch.empty(0)),
        suffix_noise=torch.empty(0),
        category_ids=torch.empty(0, dtype=torch.long),
        stage_ids=torch.empty(0, dtype=torch.long),
        progress=torch.empty(0),
        source=empty_response,
        carrier=empty_response,
        members=empty_response,
        member_reliability=torch.tensor([1.0, 3.0, 1.0]),
        member_names=("latest", "independent", "earliest"),
        anchors=(),
    )
    objective = VerifiedMemberObjective(
        bank=bank,
        validity=torch.empty(0),
        scales=empty_response,
        reliability=torch.tensor([0.2, 0.6, 0.2]),
        temperature=0.25,
    )
    losses = torch.tensor([2.0, 0.1, 1.5])

    global_loss, responsibilities = global_member_effect_loss(losses, objective)

    assert int(responsibilities.argmax()) == 1
    assert torch.allclose(responsibilities.sum(), torch.tensor(1.0))
    assert float(losses.min()) <= float(global_loss)
    assert float(global_loss) < float((objective.reliability * losses).sum())
