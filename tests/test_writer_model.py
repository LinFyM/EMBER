from __future__ import annotations

import pytest
import torch

from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


def _state() -> dict[str, torch.Tensor]:
    return {
        "base.block.q_proj.lora_A.default.weight": torch.randn(2, 3),
        "base.block.q_proj.lora_B.default.weight": torch.zeros(4, 2),
        "base.block.v_proj.lora_A.default.weight": torch.randn(2, 3),
        "base.block.v_proj.lora_B.default.weight": torch.zeros(5, 2),
    }


def _model(
    conditioned_query_fusion: str = "condition_only_v2",
) -> CompleteLoRAWriter:
    state = _state()
    return CompleteLoRAWriter(
        build_lora_tensor_specs(state),
        template_state=state,
        vision_feature_dim=7,
        vision_spatial_tokens=4,
        language_feature_dim=5,
        hidden_dim=12,
        attention_heads=3,
        temporal_chunk_size=4,
        chunk_memory_tokens=2,
        episode_memory_tokens=2,
        language_memory_tokens=2,
        task_memory_tokens=2,
        decoder_hidden_dim=10,
        conditioned_query_fusion=conditioned_query_fusion,
    )


def test_writer_generates_every_template_tensor() -> None:
    model = _model()
    result = model(
        torch.zeros(3, 5),
        torch.zeros(9, 4, 7),
        torch.tensor([0, 9]),
    )
    assert set(result) == set(_state())
    for name, expected in _state().items():
        assert result[name].shape == expected.shape
        assert torch.isfinite(result[name]).all()


def test_writer_accepts_variable_videos_and_batches_one_video_per_condition() -> None:
    model = _model()
    language = torch.zeros(2, 5)
    one = model(language, torch.zeros(1, 4, 7), torch.tensor([0, 1]))
    longer = model(language, torch.zeros(17, 4, 7), torch.tensor([0, 17]))
    assert set(one) == set(longer)
    batched = model(
        torch.zeros(4, 5),
        torch.zeros(3, 4, 7),
        torch.tensor([0, 1, 3]),
        language_offsets=torch.tensor([0, 2, 4]),
    )
    assert all(value.shape[0] == 2 for value in batched.values())
    with pytest.raises(RuntimeError, match="condition batches differ"):
        model(
            torch.zeros(4, 5),
            torch.zeros(3, 4, 7),
            torch.tensor([0, 1, 3]),
            language_offsets=torch.tensor([0, 4]),
        )


def test_v3_query_identity_is_memory_gated_and_bias_free() -> None:
    model = _model("memory_gated_query_v3")
    assert set(model.task_encoder.condition_gates) == {
        "chunk",
        "episode",
        "language",
        "task",
    }
    assert model.parameter_attention.in_proj_bias is None
    assert model.parameter_attention.out_proj.bias is None
    assert model.parameter_ffn[1].bias is None
    assert model.parameter_ffn[3].bias is None
    assert model.task_encoder.chunk_attention.in_proj_bias is None
    assert model.task_encoder.vision_projection[1].bias is None
    assert model.parameter_gate.bias is None

    result = model(
        torch.zeros(3, 5),
        torch.zeros(9, 4, 7),
        torch.tensor([0, 9]),
    )
    for name, buffer_name in model._template_buffers.items():
        expected = getattr(model, buffer_name)
        assert torch.equal(result[name], expected)
