from __future__ import annotations

import pytest
import torch

from ember.pi05_processing import Pi05TeacherPrefixTokenizer


class _TokenizerStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, prompt: str, *, add_bos: bool) -> list[int]:
        self.calls.append((prompt, add_bos))
        return [1, 2, 3, 4, 5]


def _tokenizer(max_length: int = 8) -> Pi05TeacherPrefixTokenizer:
    tokenizer = object.__new__(Pi05TeacherPrefixTokenizer)
    tokenizer._tokenizer = _TokenizerStub()  # type: ignore[attr-defined]
    tokenizer._max_length = max_length  # type: ignore[attr-defined]
    tokenizer._device = "cpu"  # type: ignore[attr-defined]
    return tokenizer


def test_teacher_prompt_is_state_free_and_preserves_action_suffix() -> None:
    tokenizer = _tokenizer()
    tokens, masks = tokenizer(["pick_up bowl"])
    assert tokenizer._tokenizer.calls == [  # type: ignore[attr-defined]
        ("Task: pick up bowl;\nAction: ", True),
    ]
    assert tokens.shape == masks.shape == (1, 8)
    assert int(masks.sum()) == 5
    assert torch.equal(tokens[0, :5], torch.tensor([1, 2, 3, 4, 5]))


def test_teacher_prompt_fails_closed_instead_of_truncating() -> None:
    tokenizer = _tokenizer(max_length=4)
    with pytest.raises(ValueError, match="exceeds the sealed tokenizer length"):
        tokenizer(["pick up bowl"])
