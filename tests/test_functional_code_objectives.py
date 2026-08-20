from __future__ import annotations

import pytest
import torch

from ember.functional_adaptation.inference import FunctionalCodePosterior
from ember.functional_adaptation.objectives import functional_code_inference_loss


def test_confidence_loss_uses_autocast_safe_logits() -> None:
    logits = torch.zeros((1, 1), requires_grad=True)
    code = torch.zeros((1, 2))
    posterior = FunctionalCodePosterior(
        language_code=code,
        video_code=code,
        posterior_delta=code,
        posterior_confidence_logits=logits,
        posterior_confidence=torch.full((1, 1), 0.9),
        combined_code=code,
        per_video_program=torch.zeros((1, 1, 2)),
        per_video_summary=torch.zeros((1, 2)),
        video_condition_ids=torch.zeros(1, dtype=torch.long),
        action_phase_predictions=torch.zeros((1, 1, 7)),
    )
    weights = {
        "combined_code": 0.0,
        "language_code": 0.0,
        "video_code": 0.0,
        "correct_confidence": 1.0,
        "control_confidence": 0.0,
        "control_update": 0.0,
        "same_task_consistency": 0.0,
        "action_alignment": 0.0,
    }

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = functional_code_inference_loss(posterior, code, weights=weights)
    loss.total.backward()

    assert loss.correct_confidence.item() == pytest.approx(0.69314718)
    assert logits.grad is not None
    assert logits.grad.item() == pytest.approx(-0.5)
