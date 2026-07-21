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


def _model() -> CompleteLoRAWriter:
    state = _state()
    return CompleteLoRAWriter(
        build_lora_tensor_specs(state),
        template_state=state,
        vision_feature_dim=7,
        language_feature_dim=5,
        hidden_dim=12,
        attention_heads=3,
        temporal_chunk_size=4,
        chunk_memory_tokens=2,
        episode_memory_tokens=2,
        task_memory_tokens=2,
        decoder_hidden_dim=10,
    )


def test_writer_generates_every_template_tensor() -> None:
    model = _model()
    result = model(
        torch.zeros(3, 5),
        torch.zeros(9, 7),
        torch.tensor([0, 9]),
    )
    assert set(result) == set(_state())
    for name, expected in _state().items():
        assert result[name].shape == expected.shape
        assert torch.isfinite(result[name]).all()


def test_writer_accepts_one_variable_length_video_and_rejects_multiple() -> None:
    model = _model()
    language = torch.zeros(2, 5)
    one = model(language, torch.zeros(1, 7), torch.tensor([0, 1]))
    longer = model(language, torch.zeros(17, 7), torch.tensor([0, 17]))
    assert set(one) == set(longer)
    with pytest.raises(RuntimeError, match="exactly one non-empty teaching video"):
        model(language, torch.zeros(3, 7), torch.tensor([0, 1, 3]))
