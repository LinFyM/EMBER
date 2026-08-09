from __future__ import annotations

import torch

from ember.writer.condition_update import (
    FixedTemporalConditionFeature,
    FrozenV6ConditionResidualWriter,
    ProgramResidualMemory,
    apply_program_residual_delta_with_evidence_,
    counterfactual_null_program_delta,
)
from ember.writer.model import WriterVideoEvidence


class _FrozenSlotDecoder(torch.nn.Module):
    program_width = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)

    def compile_slots(self, memories: torch.Tensor) -> torch.Tensor:
        return memories

    def decode_slots(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded = self.projection(slots)
        return {
            "target.lora_A.default.weight": decoded[:, :160].clone(),
            "target.lora_B.default.weight": decoded[:, 160:].clone(),
        }


def _evidence(frame_values: torch.Tensor, text: torch.Tensor) -> WriterVideoEvidence:
    frames, tokens, width = frame_values.shape
    assert text.shape == (tokens, width)
    return WriterVideoEvidence(
        text_queries=text[None],
        frame_evidence=frame_values,
        grounded_evidence=frame_values.clone(),
        interactions=torch.zeros(frames, width),
        valid_task_tokens=torch.ones(1, tokens, dtype=torch.bool),
        offsets=(0, frames),
    )


def test_fixed_temporal_feature_is_zero_preserving_and_reads_real_order() -> None:
    encoder = FixedTemporalConditionFeature(
        program_width=3,
        feature_width=5,
        initialization_seed=17,
    )
    text = torch.tensor([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]])
    zero = _evidence(text[None].expand(4, -1, -1).clone(), text)
    indices = torch.tensor([0, 5, 10, 15], dtype=torch.long)
    zero_feature = encoder(zero, indices)
    assert torch.equal(zero_feature, torch.zeros_like(zero_feature))

    innovation = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]],
            [[-1.0, 0.5, 0.0], [-0.5, 0.25, 0.0]],
        ]
    )
    evidence = _evidence(text[None] + innovation, text)
    natural = encoder(evidence, indices)
    reversed_feature = encoder(
        evidence,
        indices,
        frame_order=torch.tensor([3, 2, 1, 0], dtype=torch.long),
    )
    shuffled = encoder(
        evidence,
        indices,
        frame_order=torch.tensor([0, 2, 1, 3], dtype=torch.long),
    )
    assert natural.shape == (1, 5)
    torch.testing.assert_close(
        natural.square().sum(dim=1), torch.ones(1), rtol=1e-6, atol=1e-6
    )
    assert not torch.equal(natural, reversed_feature)
    assert not torch.equal(natural, shuffled)
    assert not tuple(encoder.parameters())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_feature = encoder(evidence, indices)
        memory = ProgramResidualMemory(
            feature_width=5,
            program_slots=3,
            program_width=4,
        )
        autocast_read = memory(autocast_feature)
    assert autocast_feature.dtype == torch.float32
    assert autocast_read.dtype == torch.float32


def test_counterfactual_null_update_moves_correct_and_preserves_negative_rows() -> None:
    correct_features = torch.eye(4, dtype=torch.float32)[:2]
    negative_features = torch.eye(4, dtype=torch.float32)[2:]
    cotangents = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    delta, summary = counterfactual_null_program_delta(
        correct_features,
        negative_features,
        cotangents,
        step_size=1.0,
        relative_damping=0.01,
    )
    full_features = torch.cat((correct_features, negative_features))
    predicted = (full_features @ delta.flatten(1)).reshape(4, 3, 4)
    torch.testing.assert_close(
        predicted[:2],
        -cotangents / 1.01,
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.equal(predicted[2:], torch.zeros_like(predicted[2:]))
    assert summary.feature_rank == 4
    assert summary.predicted_correct_motion_rms > 0
    assert summary.predicted_negative_motion_rms == 0
    assert summary.predicted_negative_to_correct_ratio == 0
    assert delta.dtype == torch.float32

    memory = ProgramResidualMemory(
        feature_width=4,
        program_slots=3,
        program_width=4,
    )
    assert torch.equal(memory(full_features), torch.zeros(4, 3, 4))
    application = apply_program_residual_delta_with_evidence_(
        memory,
        delta,
        full_features,
    )
    torch.testing.assert_close(memory(full_features), predicted)
    assert application.predicted_observed_max_abs == 0
    assert application.predicted_observed_relative_rms == 0


def test_zero_correct_motion_uses_finite_gate_failure_value() -> None:
    features = torch.eye(4, dtype=torch.float32)
    _, summary = counterfactual_null_program_delta(
        features[:2],
        features[2:],
        torch.zeros(2, 3, 4),
        step_size=1.0,
        relative_damping=0.01,
    )
    assert torch.isfinite(
        torch.tensor(summary.predicted_negative_to_correct_ratio)
    )
    assert summary.predicted_negative_to_correct_ratio > 1e30


def test_zero_residual_is_exact_and_one_decoder_moves_both_lora_factors() -> None:
    base = _FrozenSlotDecoder()
    with torch.no_grad():
        base.projection.weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 2.0]]))
    writer = FrozenV6ConditionResidualWriter(
        base,  # type: ignore[arg-type]
        feature_width=3,
        feature_seed=23,
    )
    slots = torch.randn(2, 320, 2)
    features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    baseline = base.decode_slots(slots)
    step0 = base.decode_slots(writer.condition_slots(slots, features))
    assert all(torch.equal(baseline[name], step0[name]) for name in baseline)
    assert all(not parameter.requires_grad for parameter in writer.parameters())

    with torch.no_grad():
        writer.program_memory.value.normal_(std=0.02)
    changed = base.decode_slots(writer.condition_slots(slots, features))
    assert all(not torch.equal(step0[name], changed[name]) for name in step0)
