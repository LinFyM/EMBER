from pathlib import Path

import pytest
import torch

from ember.ecp.bank_conditioning.operator import BankConditioningError
from ember.ecp.bank_conditioning.program_bank_interaction import (
    ProgramBankContext,
    ProgramBankInteractionScorer,
)
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.joint_program_primal.routing_control import (
    INTERACTION_BASE_SCORE_FEATURE,
    load_routing_control_config,
)


def test_v4_contract_activates_base_score_feature() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_routing_control_config(
        root / "configs/pi05_ecp_program_bank_candidate_interaction_v4.json"
    )
    assert config["model"]["interaction_base_score_feature"] == (
        INTERACTION_BASE_SCORE_FEATURE
    )


def test_base_score_feature_matches_b1_score_and_is_translation_invariant() -> None:
    scorer = ProgramBankInteractionScorer(
        (TargetOwner(0, "q", TargetFamily.Q, 0, 4, 16),),
        program_width=8,
        event_slots=4,
        replay_score_rms=0.02,
    )
    generator = torch.Generator().manual_seed(20260831)
    values = torch.randn(5, 2, 50, 4, generator=generator)
    mean = values.reshape(-1, 4).mean(0)
    query = torch.randn(4, 4, generator=generator, requires_grad=True)
    centered = values - mean
    observed = scorer._base_score_feature(query, centered)
    expected = torch.einsum("rd,...d->r...", query.detach(), centered) / 0.02

    assert observed.shape == (4, 4, 5, 2, 50, 1)
    torch.testing.assert_close(observed[:, 0, ..., 0], expected)
    torch.testing.assert_close(observed[:, 0], observed[:, -1])
    torch.testing.assert_close(
        scorer._base_score_feature(
            query,
            (values + 7.0) - (mean + 7.0),
        ),
        observed,
        rtol=2e-4,
        atol=2e-4,
    )
    assert query.grad is None


def test_vector_interaction_is_family_side_shared_zero_initialized_and_live() -> None:
    owners = (
        TargetOwner(0, "q0", TargetFamily.Q, 0, 64, 256),
        TargetOwner(1, "q1", TargetFamily.Q, 1, 64, 256),
        TargetOwner(2, "v", TargetFamily.V, 0, 64, 32),
        TargetOwner(3, "action_in", TargetFamily.ACTION_IN, None, 32, 64),
        TargetOwner(4, "action_out", TargetFamily.ACTION_OUT, None, 64, 32),
    )
    scorer = ProgramBankInteractionScorer(
        owners,
        program_width=8,
        event_slots=2,
        replay_score_rms=0.02,
    )
    assert scorer._correction_feature_width == 45
    assert set(scorer.native_query_projection["input"]) == {
        family.value for family in TargetFamily
    }
    for side in ("input", "output"):
        for family in TargetFamily:
            query = scorer.native_query_projection[side][family.value].weight
            candidate = scorer.native_key_projection[side][family.value].weight
            torch.testing.assert_close(query, candidate, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                query @ query.T,
                torch.eye(32),
                rtol=2e-5,
                atol=2e-5,
            )

    generator = torch.Generator().manual_seed(20260831)
    values = torch.randn(2, 2, 50, 64, generator=generator)
    context = ProgramBankContext(
        canonical_assignment=torch.eye(2),
        frame_positions=torch.tensor([0.0, 1.0]),
        local_scene=torch.randn(5, 8, generator=generator),
        local_process=torch.randn(2, 5, 8, generator=generator),
        local_presence=torch.ones(2),
        local_tau=torch.randn(2, 2, generator=generator),
        local_sigma=torch.randn(2, 5, 8, generator=generator),
    )
    arguments = {
        "target": 0,
        "program_event_state": torch.randn(4, 2, 8, generator=generator),
        "native_event_query": torch.randn(
            4, 2, 64, generator=generator, requires_grad=True
        ),
        "event_weights": torch.softmax(torch.randn(4, 2, generator=generator), -1),
        "base_query": torch.randn(4, 64, generator=generator),
        "values": values,
        "native_mean": values.reshape(-1, 64).mean(0),
        "context": context,
    }
    captured = []
    handle = scorer.correction[TargetFamily.Q.value].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )
    zero = scorer.input_logit_corrections(**arguments)
    handle.remove()
    assert zero.shape == (4, 2, 2, 2, 50)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert captured[0].shape[-1] == 45

    with torch.no_grad():
        scorer.correction[TargetFamily.Q.value][-1].weight.normal_(std=0.1)
    changed = scorer.input_logit_corrections(**arguments)
    changed.square().mean().backward()
    assert bool(torch.count_nonzero(changed))
    torch.testing.assert_close(changed[:, 0], -changed[:, 1])
    for projection in (
        scorer.native_query_projection["input"][TargetFamily.Q.value],
        scorer.native_key_projection["input"][TargetFamily.Q.value],
    ):
        assert projection.weight.grad is not None
        assert bool(torch.isfinite(projection.weight.grad).all())
        assert bool(torch.count_nonzero(projection.weight.grad))


def test_vector_interaction_rejects_family_native_width_drift() -> None:
    with pytest.raises(BankConditioningError, match="native width changed"):
        ProgramBankInteractionScorer(
            (
                TargetOwner(0, "q0", TargetFamily.Q, 0, 4, 16),
                TargetOwner(1, "q1", TargetFamily.Q, 1, 5, 16),
            ),
            program_width=8,
            event_slots=4,
            replay_score_rms=0.02,
        )
