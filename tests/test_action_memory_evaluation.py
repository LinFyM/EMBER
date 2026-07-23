from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ember.writer.inference as writer_inference
from ember.pi05_source_checkpoint import canonical_hash
from ember.writer.inference import (
    FrozenWriterTaskAdapter,
    WRITER_ADAPTER_SCHEMA,
    _task_video_mapping,
)


def _generic_writer_adapter() -> dict:
    keys = tuple(
        (suite, 0)
        for suite in (
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        )
    )
    roles = {key: "train" for key in keys}
    mapping = list(_task_video_mapping(keys, roles, "generic_correct"))
    return {
        "schema_version": WRITER_ADAPTER_SCHEMA,
        "kind": "as_writer",
        "writer_method": "as_writer",
        "arm": "as_writer_generic_correct_video",
        "video_condition": "generic_correct",
        "writer_language_condition": "generic_neutral",
        "checkpoint": {
            "cursor": 12,
            "cursor_axis": "optimizer_step",
            "manifest_file_sha256": "3" * 64,
            "writer_state_sha256": "4" * 64,
        },
        "lora_contract_sha256": "5" * 64,
        "video_schedule": {"seed": 7, "demo_count": 50},
        "task_video_mapping_sha256": canonical_hash(mapping),
        "task_video_mapping": mapping,
        "pairing_sha256": "6" * 64,
    }


def test_generic_writer_condition_uses_neutral_language_with_raw_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = SimpleNamespace(
        frames=np.zeros((5, 3, 4, 4), dtype=np.uint8),
        frame_indices=np.arange(5, dtype=np.int64),
    )
    observed: dict[str, object] = {}

    class Store:
        def load(self, *_args):
            return teacher

    class Tokenizer:
        def __call__(self, tasks):
            observed["writer_language"] = tasks[0]
            return torch.full((1, 3), 7), torch.ones(1, 3, dtype=torch.bool)

    class Writer:
        def __call__(
            self,
            _frames,
            _indices,
            _offsets,
            language,
            _mask,
            *,
            policy,
        ):
            observed["tokens"] = language.detach().cpu()
            observed["policy"] = policy
            return {"adapter": torch.zeros(1)}

    frozen = object.__new__(FrozenWriterTaskAdapter)
    frozen.store = Store()
    frozen.tokenizer = Tokenizer()
    frozen.generic_language = "perform the demonstrated task"
    frozen.language_by_id = {0: "real task"}
    frozen.writer = Writer()
    frozen.lora_contract = object()
    frozen.identity_state = {"adapter": torch.zeros(1)}
    frozen.policy = object()
    frozen.device = torch.device("cpu")
    frozen.evaluation_adapter = _generic_writer_adapter()
    monkeypatch.setattr(writer_inference, "validate_lora_state", lambda *_args: None)
    monkeypatch.setattr(
        writer_inference,
        "lora_state_sha256",
        lambda _state: "7" * 64,
    )
    monkeypatch.setattr(
        writer_inference,
        "copy_task_lora_state_",
        lambda *_args: None,
    )
    frozen.prepare_episode(
        suite="libero_spatial",
        task_id=0,
        init_state_id=0,
    )
    assert observed["writer_language"] == "perform the demonstrated task"
    torch.testing.assert_close(observed["tokens"], torch.full((1, 3), 7))
    assert observed["policy"] is frozen.policy
