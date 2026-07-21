from __future__ import annotations

import pytest
import torch

from ember.pi05_processing import Pi05PureLanguageTokenizer


class _TokenizerStub:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, prompts: list[str], *, add_bos: bool) -> list[list[int]]:
        assert add_bos is True
        self.prompts = prompts
        return [[1, 2, 3], [4, 5]]


def test_writer_language_prompt_contains_no_state_or_other_privileged_fields() -> None:
    tokenizer = object.__new__(Pi05PureLanguageTokenizer)
    tokenizer._tokenizer = _TokenizerStub()  # type: ignore[attr-defined]
    tokenizer._max_length = 4  # type: ignore[attr-defined]
    tokenizer._device = "cpu"  # type: ignore[attr-defined]
    tokens, masks = tokenizer(["pick_up bowl", "open drawer\n"])

    assert tokenizer._tokenizer.prompts == [  # type: ignore[attr-defined]
        "Task: pick up bowl\n",
        "Task: open drawer\n",
    ]
    assert all(
        forbidden not in "".join(tokenizer._tokenizer.prompts).lower()  # type: ignore[attr-defined]
        for forbidden in ("state:", "action:", "reward", "terminal", "proprio")
    )
    assert tokens.tolist() == [[1, 2, 3, 0], [4, 5, 0, 0]]
    assert torch.equal(
        masks,
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
    )


def test_writer_language_prompt_fails_closed_instead_of_truncating() -> None:
    tokenizer = object.__new__(Pi05PureLanguageTokenizer)
    tokenizer._tokenizer = _TokenizerStub()  # type: ignore[attr-defined]
    tokenizer._max_length = 2  # type: ignore[attr-defined]
    tokenizer._device = "cpu"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="exceeds the sealed tokenizer length"):
        tokenizer(["pick up bowl", "open drawer"])
