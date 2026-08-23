import torch

from ember.ecp.policy_effects import ExecutionPolicyPrefix, PolicyEffectResponse
from ember.ecp.stage1_equivalence import (
    Stage1EffectBank,
    equal_time_progress_strata,
)
from ember.ecp.stage1_realization import (
    RealizationConfig,
    fixed_a_relative_distance,
    fixed_a_state,
    project_expert_onto_rank_reserved_residual,
    solve_fixed_a_particle_effects,
)
from ember.lora import LoRATarget, SmolVLALoRAContract


def _contract():
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=1, out_features=1),),
        rank=1,
        alpha=1,
        dropout=0.0,
        identity_seed=1,
    )
    carrier = {
        "tiny.lora_A.default.weight": torch.ones(1, 1),
        "tiny.lora_B.default.weight": torch.full((1, 1), 0.1),
    }
    return contract, carrier


def _response(value: torch.Tensor, states: int) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=value.reshape(1, 1, 1, 1).expand(states, 38, 4, 128),
        flow=value.reshape(1, 1, 1, 1).expand(states, 10, 50, 32),
        action=value.reshape(1, 1, 1, 1).expand(states, 10, 50, 7),
    )


def _bank() -> Stage1EffectBank:
    states = 48
    source = _response(torch.tensor(0.0), states)
    carrier = _response(torch.tensor(0.1), states)
    members = PolicyEffectResponse(
        owner=torch.stack(
            [_response(torch.tensor(value), states).owner for value in (0.4, 0.6, 0.8)]
        ),
        flow=torch.stack(
            [_response(torch.tensor(value), states).flow for value in (0.4, 0.6, 0.8)]
        ),
        action=torch.stack(
            [_response(torch.tensor(value), states).action for value in (0.4, 0.6, 0.8)]
        ),
    )
    return Stage1EffectBank(
        prefix=ExecutionPolicyPrefix(
            embeddings=torch.zeros(states, 1, 1),
            padding=torch.ones(states, 1, dtype=torch.bool),
        ),
        suffix_noise=torch.zeros(states, 50, 32),
        category_ids=torch.tensor([0] * 8 + [1] * 24 + [2] * 8 + [3] * 8),
        stage_ids=torch.tensor([0] * 8 + list(range(8)) * 3 + list(range(8)) * 2),
        progress=torch.tensor([0.0] * 8 + [value / 7 for value in range(8)] * 5),
        source=source,
        carrier=carrier,
        members=members,
        member_reliability=torch.ones(3),
    )


def test_time_progress_strata_are_ordered_and_unique() -> None:
    progress = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    selected = equal_time_progress_strata(progress)
    assert len(selected) == 8
    assert selected == tuple(sorted(set(selected)))
    assert selected[0] == 0
    assert selected[-1] == 9


def test_fixed_a_update_has_no_factor_cross_term() -> None:
    contract, carrier = _contract()
    delta = {"tiny.lora_B.default.weight": torch.full((1, 1), 0.05)}
    state = fixed_a_state(carrier, delta, contract)
    assert state["tiny.lora_A.default.weight"] is carrier["tiny.lora_A.default.weight"]
    assert torch.equal(state["tiny.lora_B.default.weight"], torch.full((1, 1), 0.15))
    assert torch.allclose(
        fixed_a_relative_distance(delta, carrier, contract), torch.tensor(0.25)
    )


def test_rank_reserved_projection_recovers_a_mobile_rank_one_correction() -> None:
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=3, out_features=2),),
        rank=3,
        alpha=3,
        dropout=0.0,
        identity_seed=1,
    )
    carrier = {
        "tiny.lora_A.default.weight": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "tiny.lora_B.default.weight": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
    }
    expert = {
        "tiny.lora_A.default.weight": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "tiny.lora_B.default.weight": torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 1.0, 6.0]]
        ),
    }
    projected, metrics = project_expert_onto_rank_reserved_residual(
        carrier=carrier, expert=expert, contract=contract, carrier_rank=2
    )
    expert_update = (
        expert["tiny.lora_B.default.weight"]
        @ expert["tiny.lora_A.default.weight"]
    )
    actual = (
        projected["tiny.lora_B.default.weight"]
        @ projected["tiny.lora_A.default.weight"]
    )
    assert torch.allclose(actual, expert_update, atol=1e-5)
    assert metrics[0].carrier_rank == 2
    assert metrics[0].residual_rank == 1
    assert metrics[0].residual_energy < 1e-8


def test_rank_reserved_projection_matches_the_best_truncated_correction() -> None:
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=3, out_features=2),),
        rank=3,
        alpha=3,
        dropout=0.0,
        identity_seed=1,
    )
    identity = torch.eye(3)
    carrier = {
        "tiny.lora_A.default.weight": identity,
        "tiny.lora_B.default.weight": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
    }
    expert = {
        "tiny.lora_A.default.weight": identity,
        "tiny.lora_B.default.weight": torch.tensor(
            [[1.0, 2.0, 0.0], [0.0, 1.0, 6.0]]
        ),
    }
    projected, metrics = project_expert_onto_rank_reserved_residual(
        carrier=carrier, expert=expert, contract=contract, carrier_rank=2
    )
    actual = (
        projected["tiny.lora_B.default.weight"]
        @ projected["tiny.lora_A.default.weight"]
    )
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 6.0]])
    assert torch.allclose(actual, expected, atol=1e-5)
    assert abs(metrics[0].residual_energy - 4.0) < 1e-5


def test_fixed_solver_reduces_particle_equivalence_error() -> None:
    contract, carrier = _contract()
    bank = _bank()

    def response(state, indices):
        value = (
            state["tiny.lora_B.default.weight"] @ state["tiny.lora_A.default.weight"]
        ).mean()
        return _response(value, int(indices.numel()))

    _, history, final = solve_fixed_a_particle_effects(
        carrier=carrier,
        bank=bank,
        contract=contract,
        response=response,
        config=RealizationConfig(
            owner_weight=1.0,
            flow_weight=0.0,
            action_weight=0.0,
            microbatch_size=8,
        ),
    )
    assert final.total < history[0].snapshot.total
