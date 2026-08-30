from pathlib import Path

import torch

from ember.ecp.bank_conditioning.program_bank_interaction import (
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
