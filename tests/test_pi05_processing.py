from __future__ import annotations

import pytest
import torch

from ember.pi05_processing import Pi05ForecastPrefixTokenizer


class _TokenizerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, prompt: str, *, add_bos: bool) -> list[int]:
        self.calls.append((prompt, add_bos))
        return [1, 2, 3] if add_bos else [4, 5]


def _tokenizer(max_length: int = 16) -> Pi05ForecastPrefixTokenizer:
    tokenizer = object.__new__(Pi05ForecastPrefixTokenizer)
    tokenizer._tokenizer = _TokenizerStub()  # type: ignore[attr-defined]
    tokenizer._max_length = max_length  # type: ignore[attr-defined]
    tokenizer._device = "cpu"  # type: ignore[attr-defined]
    return tokenizer


def test_forecast_prompt_preserves_native_layout_with_eight_virtual_slots() -> None:
    tokenizer = _tokenizer()
    tokens, masks, positions = tokenizer(["pick_up bowl"])
    assert tokenizer._tokenizer.calls == [  # type: ignore[attr-defined]
        ("Task: pick up bowl, State: ", True),
        (";\nAction: ", False),
    ]
    assert positions.tolist() == [[3, 4, 5, 6, 7, 8, 9, 10]]
    assert tokens.shape == masks.shape == (1, 16)
    assert int(masks.sum()) == 13
    assert torch.equal(tokens[0, :3], torch.tensor([1, 2, 3]))
    assert torch.equal(tokens[0, 11:13], torch.tensor([4, 5]))


def test_forecast_prompt_fails_closed_instead_of_truncating() -> None:
    tokenizer = _tokenizer(max_length=12)
    with pytest.raises(ValueError, match="exceeds the sealed tokenizer length"):
        tokenizer(["pick up bowl"])
