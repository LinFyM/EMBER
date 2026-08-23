import torch

from ember.ecp.effect_path_calibration import (
    build_verified_member_objective,
    build_verified_member_validity,
    global_particle_loss,
    verified_member_losses,
)
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectParticles,
    PolicyEffectResponse,
)
from ember.ecp.stage1_equivalence import (
    Stage1EffectBank,
    equal_time_progress_strata,
)
from ember.ecp.stage1_objective import RealizationConfig
from ember.ecp.stage1_parameterization import (
    interpolate_rank_reserved_endpoint,
    project_expert_onto_rank_reserved_residual,
    rank_reserved_relative_distance,
    rank_reserved_state,
)
from ember.ecp.stage1_realization import (
    solve_effective_update_particle_effects,
)
from ember.lora import LoRATarget, SmolVLALoRAContract


def _contract():
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=1, out_features=1),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=1,
    )
    carrier = {
        "tiny.lora_A.default.weight": torch.tensor([[1.0], [2.0]]),
        "tiny.lora_B.default.weight": torch.tensor([[0.1, 0.0]]),
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


def test_effect_particles_preserve_probe_axis_before_averaging() -> None:
    particles = PolicyEffectParticles(
        owner=torch.tensor([[[1.0], [3.0]]]),
        flow=torch.tensor([[[2.0], [6.0]]]),
        action=torch.tensor([[[4.0], [8.0]]]),
    )
    response = particles.mean_response()
    assert particles.owner.shape[1] == 2
    assert response.owner.item() == 2.0
    assert response.flow.item() == 4.0
    assert response.action.item() == 6.0


def _solver_contract():
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=8, out_features=8),),
        rank=6,
        alpha=6,
        dropout=0.0,
        identity_seed=1,
    )
    carrier_a = torch.zeros(6, 8)
    carrier_b = torch.zeros(8, 6)
    carrier_a[0] = 1.0
    carrier_a[1, 0] = 1.0
    carrier_a[2:, :4] = torch.eye(4)
    carrier_b[:, 0] = 0.1
    carrier = {
        "tiny.lora_A.default.weight": carrier_a,
        "tiny.lora_B.default.weight": carrier_b,
    }
    return contract, carrier


def test_time_progress_strata_are_ordered_and_unique() -> None:
    progress = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    selected = equal_time_progress_strata(progress)
    assert len(selected) == 8
    assert selected == tuple(sorted(set(selected)))
    assert selected[0] == 0
    assert selected[-1] == 9


def test_rank_reserved_update_is_exactly_effective_additive() -> None:
    contract, carrier = _contract()
    residual = {
        "tiny.lora_A.default.weight": torch.tensor([[2.0]]),
        "tiny.lora_B.default.weight": torch.tensor([[0.05]]),
    }
    state = rank_reserved_state(carrier, residual, contract, carrier_rank=1)
    candidate = (
        state["tiny.lora_B.default.weight"]
        @ state["tiny.lora_A.default.weight"]
    )
    carrier_update = (
        carrier["tiny.lora_B.default.weight"][:, :1]
        @ carrier["tiny.lora_A.default.weight"][:1]
    )
    residual_update = (
        residual["tiny.lora_B.default.weight"]
        @ residual["tiny.lora_A.default.weight"]
    )
    assert torch.equal(candidate, carrier_update + residual_update)
    assert torch.allclose(
        rank_reserved_relative_distance(
            residual, carrier, contract, carrier_rank=1
        ),
        torch.tensor(1.0),
    )


def test_rank_reserved_path_scales_the_effective_correction() -> None:
    contract, carrier = _contract()
    endpoint = {
        "tiny.lora_A.default.weight": torch.tensor([[1.0], [4.0]]),
        "tiny.lora_B.default.weight": torch.tensor([[0.1, 0.05]]),
    }
    state = interpolate_rank_reserved_endpoint(
        carrier=carrier,
        endpoint=endpoint,
        alpha=0.25,
        contract=contract,
        carrier_rank=1,
    )
    effective = state["tiny.lora_B.default.weight"] @ state[
        "tiny.lora_A.default.weight"
    ]
    carrier_effective = carrier["tiny.lora_B.default.weight"] @ carrier[
        "tiny.lora_A.default.weight"
    ]
    endpoint_effective = endpoint["tiny.lora_B.default.weight"] @ endpoint[
        "tiny.lora_A.default.weight"
    ]
    assert torch.allclose(
        effective - carrier_effective,
        0.25 * (endpoint_effective - carrier_effective),
    )


def test_verified_objective_keeps_one_member_identity() -> None:
    anchors = [
        {"category": "initial", "init_state_id": index} for index in range(8)
    ]
    anchors.extend(
        {"category": "successful", "generator": member}
        for member in ("latest", "independent", "earliest")
        for _ in range(8)
    )
    anchors.extend({"category": "candidate"} for _ in range(8))
    anchors.extend({"category": "recovery"} for _ in range(8))
    names = ("latest", "independent", "earliest")
    validity = build_verified_member_validity(
        anchors=anchors,
        member_names=names,
        initial_success={
            "latest": {index: True for index in range(8)},
            "independent": {},
            "earliest": {0: True},
        },
    )
    assert validity.sum(1).tolist() == [16, 8, 9]
    bank = _bank()
    config = RealizationConfig()
    objective = build_verified_member_objective(bank, validity, config)
    losses = verified_member_losses(
        _response(torch.tensor(0.4), 48), bank, objective, config
    )
    _, responsibilities = global_particle_loss(losses, objective)
    assert int(responsibilities.argmax()) == 0


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


def test_effective_update_solver_reduces_error_inside_the_trust_region() -> None:
    contract, carrier = _solver_contract()
    bank = _bank()

    def response(state, indices):
        value = (
            state["tiny.lora_B.default.weight"] @ state["tiny.lora_A.default.weight"]
        ).mean()
        return _response(value, int(indices.numel()))

    outcome = solve_effective_update_particle_effects(
        carrier=carrier,
        bank=bank,
        contract=contract,
        response=response,
        carrier_rank=2,
        config=RealizationConfig(
            owner_weight=1.0,
            flow_weight=0.0,
            action_weight=0.0,
            microbatch_size=8,
        ),
    )
    assert outcome.final.total < outcome.initial.total
    assert outcome.initial_state_is_exact_carrier
    assert outcome.initial_directional_derivative < 0.0
    assert outcome.initial.trust_distance == 0.0
    assert 0 < len(outcome.history) <= 9
    assert outcome.history[0].phase == "matrix_free_initial_sketch"
    assert all(row.directional_derivative < 0 for row in outcome.history)
    assert all(row.after.total < row.before.total for row in outcome.history)
    assert outcome.vjp_evaluations <= 12
    assert outcome.final.trust_distance <= 1.5
    assert outcome.objective_gap_recovery > 0.0


def test_effective_update_solver_is_invariant_to_carrier_factor_gauge() -> None:
    contract, carrier = _solver_contract()
    bank = _bank()

    def response(state, indices):
        value = (
            state["tiny.lora_B.default.weight"]
            @ state["tiny.lora_A.default.weight"]
        ).mean()
        return _response(value, int(indices.numel()))

    transform = torch.tensor([[2.0, 0.5], [0.0, 0.5]])
    inverse = torch.linalg.inv(transform)
    gauged = {name: value.clone() for name, value in carrier.items()}
    gauged["tiny.lora_A.default.weight"][:2] = (
        transform @ carrier["tiny.lora_A.default.weight"][:2]
    )
    gauged["tiny.lora_B.default.weight"][:, :2] = (
        carrier["tiny.lora_B.default.weight"][:, :2] @ inverse
    )
    config = RealizationConfig(
        owner_weight=1.0,
        flow_weight=0.0,
        action_weight=0.0,
        microbatch_size=8,
    )
    original = solve_effective_update_particle_effects(
        carrier=carrier,
        bank=bank,
        contract=contract,
        response=response,
        carrier_rank=2,
        config=config,
    )
    transformed = solve_effective_update_particle_effects(
        carrier=gauged,
        bank=bank,
        contract=contract,
        response=response,
        carrier_rank=2,
        config=config,
    )
    original_update = (
        original.state["tiny.lora_B.default.weight"]
        @ original.state["tiny.lora_A.default.weight"]
    )
    transformed_update = (
        transformed.state["tiny.lora_B.default.weight"]
        @ transformed.state["tiny.lora_A.default.weight"]
    )
    assert torch.allclose(original_update, transformed_update, atol=1e-5)
    assert abs(original.final.total - transformed.final.total) < 1e-6
