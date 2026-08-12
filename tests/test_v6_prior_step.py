from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_step import generate_view_graph, program_cotangent
from ember.writer.data import RawTeacherVideo


class _ProgramMemory(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], 320, 2, dtype=torch.float32)


class _BaseWriter(torch.nn.Module):
    program_width = 2

    def __init__(self) -> None:
        super().__init__()
        self.encoder_frames: list[torch.Tensor] = []
        self.encoder_tokens: list[torch.Tensor] = []

    def encode_video_evidence(
        self, _policy, frames, offsets, tokens, _mask, _span
    ) -> SimpleNamespace:
        self.encoder_frames.append(frames.clone())
        self.encoder_tokens.append(tokens.clone())
        return SimpleNamespace(offsets=(0, int(offsets[-1])), frames=frames)

    def build_memories(self, evidence, _indices):
        return evidence

    def compile_slots(self, _memories) -> torch.Tensor:
        return torch.zeros(1, 320, 2, dtype=torch.bfloat16)

    def decode_slots(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "target.lora_A.default.weight": slots[0, 0:1] * 2,
            "target.lora_B.default.weight": slots[0, 1:2] * 3,
        }


class _Writer(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(base_writer=_BaseWriter(), program_memory=_ProgramMemory())
        self.orders: list[torch.Tensor | None] = []
        self.tokens: list[torch.Tensor] = []
        self.frames: list[torch.Tensor] = []

    def paired_condition_features(
        self,
        _policy,
        frames,
        _offsets,
        tokens,
        _mask,
        _span,
        *,
        negative_frames=None,
        negative_offsets=None,
        frame_order=None,
    ):
        self.orders.append(None if frame_order is None else frame_order.clone())
        self.tokens.append(tokens.clone())
        self.frames.append(frames.clone())
        if negative_frames is not None:
            assert negative_offsets is not None and frame_order is None
            negative = negative_frames
            self.frames.append(negative.clone())
        else:
            assert negative_offsets is None and frame_order is not None
            negative = frames.index_select(0, frame_order.cpu())

        def feature(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(1, 256, dtype=torch.float32)
            result[0, : value.shape[0]] = value.float().mean((1, 2, 3))
            return torch.nn.functional.normalize(result, dim=1)

        return feature(frames), feature(negative)


def _video(offset: int = 0) -> RawTeacherVideo:
    frames = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2) + offset
    return RawTeacherVideo(
        frames=frames,
        frame_indices=np.array([0, 5, 10, 15], dtype=np.int64),
        raw_frame_count=16,
    )


def _language() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[11, 12, 13]], dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[False, True, True]], dtype=torch.bool),
    )


def _graph(kind: str, negative=None):
    writer = _Writer()
    graph = generate_view_graph(
        writer=writer,  # type: ignore[arg-type]
        policy=torch.nn.Identity(),
        correct_video=_video(),
        counterfactual_video=negative,
        language_tokens=_language(),
        kind=kind,
        counterfactual_seed=17,
        task_ordinal=2,
        task_visit=3,
        teacher_demo=4,
        device=torch.device("cpu"),
    )
    return writer, graph


def test_ordered_view_builds_one_complete_program_and_reuses_real_frames() -> None:
    writer, graph = _graph("reversed")
    assert len(writer.base_writer.encoder_frames) == 1
    assert torch.equal(writer.orders[0], torch.tensor([3, 2, 1, 0]))
    assert graph.correct_feature.shape == graph.negative_feature.shape == (256,)
    assert graph.correct_sampled_frames == graph.negative_sampled_frames == 4
    assert graph.program_leaf.shape == (1, 320, 2)
    assert graph.program_leaf.dtype == torch.float32
    assert graph.program_leaf.requires_grad
    assert graph.base_program_slots.dtype == torch.bfloat16
    assert graph.residual_before.dtype == torch.float32


def test_wrong_view_keeps_target_language_and_uses_external_rgb_only() -> None:
    writer, graph = _graph("wrong", _video(25))
    assert writer.orders == [None]
    assert len(writer.frames) == 2
    assert all(torch.equal(tokens, _language()[0]) for tokens in writer.tokens)
    assert not torch.equal(graph.correct_feature, graph.negative_feature)


def test_counterfactual_video_ownership_is_fail_closed() -> None:
    with pytest.raises(ExpertManifoldError, match="ownership"):
        _graph("wrong")
    with pytest.raises(ExpertManifoldError, match="ownership"):
        _graph("reversed", _video(1))


def test_complete_lora_cotangent_reaches_the_program_leaf() -> None:
    _, graph = _graph("reversed")
    gradients = {name: torch.ones_like(value) for name, value in graph.correct_lora.items()}
    observed = program_cotangent(graph, gradients)
    expected = torch.zeros((320, 2), dtype=torch.float32)
    expected[0].fill_(2)
    expected[1].fill_(3)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_missing_lora_gradient_is_rejected() -> None:
    _, graph = _graph("reversed")
    with pytest.raises(ExpertManifoldError, match="topology"):
        program_cotangent(graph, {})
