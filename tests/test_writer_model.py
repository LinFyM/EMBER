from __future__ import annotations

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
        torch.tensor([0, 4, 9]),
    )
    assert set(result) == set(_state())
    for name, expected in _state().items():
        assert result[name].shape == expected.shape
        assert torch.isfinite(result[name]).all()


def test_writer_accepts_one_short_or_fifty_variable_episodes() -> None:
    model = _model()
    language = torch.zeros(2, 5)
    one = model(language, torch.zeros(1, 7), torch.tensor([0, 1]))
    lengths = [(index % 5) + 1 for index in range(50)]
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    many = model(language, torch.zeros(sum(lengths), 7), offsets)
    assert set(one) == set(many)
