from __future__ import annotations

import pytest
import torch

from ember.pi05_processing import (
    PI05_STATE_ANCHOR_TEXT,
    PI05_STATE_ANCHOR_TOKEN_IDS,
    Pi05ForecastPrefixTokenizer,
)


class _TokenizerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, prompt: str, *, add_bos: bool) -> list[int]:
        self.calls.append((prompt, add_bos))
        if prompt == PI05_STATE_ANCHOR_TEXT and not add_bos:
            return list(PI05_STATE_ANCHOR_TOKEN_IDS)
        return [1, 2, 3] if add_bos else [4, 5]


def _tokenizer(max_length: int = 40) -> Pi05ForecastPrefixTokenizer:
    tokenizer = object.__new__(Pi05ForecastPrefixTokenizer)
    tokenizer._tokenizer = _TokenizerStub()  # type: ignore[attr-defined]
    tokenizer._max_length = max_length  # type: ignore[attr-defined]
    tokenizer._device = "cpu"  # type: ignore[attr-defined]
    return tokenizer


def test_forecast_prompt_preserves_native_layout_with_32_state_tokens() -> None:
    tokenizer = _tokenizer()
    tokens, masks, positions = tokenizer(["pick_up bowl"])
    assert tokenizer._tokenizer.calls == [  # type: ignore[attr-defined]
        (PI05_STATE_ANCHOR_TEXT, False),
        ("Task: pick up bowl, State:", True),
        (";\nAction: ", False),
    ]
    assert positions.tolist() == [list(range(3, 35))]
    assert tokens.shape == masks.shape == (1, 40)
    assert int(masks.sum()) == 37
    assert torch.equal(tokens[0, :3], torch.tensor([1, 2, 3]))
    assert torch.equal(
        tokens[0, 3:35],
        torch.tensor(PI05_STATE_ANCHOR_TOKEN_IDS),
    )
    assert torch.equal(tokens[0, 35:37], torch.tensor([4, 5]))


def test_forecast_prompt_fails_closed_instead_of_truncating() -> None:
    tokenizer = _tokenizer(max_length=32)
    with pytest.raises(ValueError, match="exceeds the sealed tokenizer length"):
        tokenizer(["pick up bowl"])
