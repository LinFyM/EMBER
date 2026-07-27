from __future__ import annotations

import pytest
import torch

from ember.pi05_processing import Pi05TeacherPrefixTokenizer


class _Piece:
    def __init__(self, piece_id: int, begin: int, end: int) -> None:
        self.id = piece_id
        self.begin = begin
        self.end = end


class _Proto:
    def __init__(self, pieces: list[_Piece]) -> None:
        self.pieces = pieces


class _TokenizerStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @staticmethod
    def bos_id() -> int:
        return 1

    def encode_as_immutable_proto(self, prompt: str) -> _Proto:
        self.calls.append(prompt)
        return _Proto(
            [
                _Piece(2, 0, 5),
                _Piece(3, 5, 10),
                _Piece(4, 10, 13),
                _Piece(5, 13, 18),
                _Piece(6, 18, len(prompt)),
            ]
        )


def _tokenizer(max_length: int = 8) -> Pi05TeacherPrefixTokenizer:
    tokenizer = object.__new__(Pi05TeacherPrefixTokenizer)
    tokenizer._tokenizer = _TokenizerStub()  # type: ignore[attr-defined]
    tokenizer._max_length = max_length  # type: ignore[attr-defined]
    tokenizer._device = "cpu"  # type: ignore[attr-defined]
    return tokenizer


def test_teacher_prompt_is_state_free_and_preserves_action_suffix() -> None:
    tokenizer = _tokenizer()
    tokens, masks, task_spans = tokenizer(["pick_up bowl"])
    assert tokenizer._tokenizer.calls == [  # type: ignore[attr-defined]
        "Task: pick up bowl;\nAction: ",
    ]
    assert tokens.shape == masks.shape == task_spans.shape == (1, 8)
    assert int(masks.sum()) == 6
    assert torch.equal(tokens[0, :6], torch.tensor([1, 2, 3, 4, 5, 6]))
    assert torch.equal(
        task_spans[0, :6],
        torch.tensor([False, False, True, True, True, False]),
    )


def test_teacher_prompt_fails_closed_instead_of_truncating() -> None:
    tokenizer = _tokenizer(max_length=5)
    with pytest.raises(ValueError, match="exceeds the sealed tokenizer length"):
        tokenizer(["pick up bowl"])
