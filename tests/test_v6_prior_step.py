from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    generate_condition_graph,
    program_cotangent,
    redecoded_program_cotangent,
)
from ember.writer.data import RawTeacherVideo


class _ProgramMemory(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], 320, 2, dtype=torch.float32)


class _BaseWriter(torch.nn.Module):
    program_width = 2

    def __init__(self) -> None:
        super().__init__()
        self.encoder_tokens: list[torch.Tensor] = []
        self.encoder_frames: list[torch.Tensor] = []

    def encode_video_evidence(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        offsets: torch.Tensor,
        tokens: torch.Tensor,
        _mask: torch.Tensor,
        _span: torch.Tensor,
    ) -> SimpleNamespace:
        self.encoder_tokens.append(tokens.clone())
        self.encoder_frames.append(frames.clone())
        return SimpleNamespace(offsets=(0, int(offsets[-1])), frames=frames)

    def build_memories(
        self, evidence: SimpleNamespace, _indices: torch.Tensor
    ) -> SimpleNamespace:
        return evidence

    def compile_slots(self, _memories: SimpleNamespace) -> torch.Tensor:
        return torch.zeros(1, 320, 2, dtype=torch.bfloat16)

    def decode_slots(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "target.lora_A.default.weight": slots[0, 0:1],
            "target.lora_B.default.weight": slots[0, 1:2],
        }


class _Writer(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            base_writer=_BaseWriter(),
            program_memory=_ProgramMemory(),
        )
        self.condition_orders: list[torch.Tensor | None] = []
        self.condition_tokens: list[torch.Tensor] = []
        self.condition_frames: list[torch.Tensor] = []

    def paired_condition_features(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        _offsets: torch.Tensor,
        tokens: torch.Tensor,
        _mask: torch.Tensor,
        _span: torch.Tensor,
        *,
        negative_frames: torch.Tensor | None = None,
        negative_offsets: torch.Tensor | None = None,
        frame_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.condition_orders.append(
            None if frame_order is None else frame_order.clone()
        )
        self.condition_tokens.append(tokens.clone())
        self.condition_frames.append(frames.clone())
        if negative_frames is not None:
            assert negative_offsets is not None and frame_order is None
            self.condition_frames.append(negative_frames.clone())
            negative_ordered = negative_frames
        else:
            assert negative_offsets is None and frame_order is not None
            negative_ordered = frames.index_select(0, frame_order.cpu())

        def feature(value: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(1, 256, dtype=torch.float32)
            result[0, : value.shape[0]] = value.float().mean((1, 2, 3))
            return torch.nn.functional.normalize(result, dim=1)

        return feature(frames), feature(negative_ordered)


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


def test_ordered_negative_reuses_one_video_encode_and_keeps_sampled_ordinals() -> None:
    writer = _Writer()
    graph = generate_condition_graph(
        writer=writer,  # type: ignore[arg-type]
        policy=torch.nn.Identity(),
        correct_video=_video(),
        counterfactual_video=None,
        language_tokens=_language(),
        kind="reversed",
        counterfactual_seed=17,
        task_ordinal=2,
        task_visit=3,
        teacher_demo=4,
        device=torch.device("cpu"),
    )
    assert len(writer.base_writer.encoder_frames) == 1
    assert torch.equal(
        writer.condition_orders[0],
        torch.tensor([3, 2, 1, 0]),
    )
    assert graph.correct_feature.shape == graph.negative_feature.shape == (256,)
    assert graph.program_leaf.dtype == torch.float32
    assert graph.program_leaf.requires_grad
    assert graph.program_input_before.dtype == torch.bfloat16


def test_wrong_video_keeps_exact_target_language_and_has_no_action_policy_forward() -> (
    None
):
    writer = _Writer()
    target_tokens = _language()
    graph = generate_condition_graph(
        writer=writer,  # type: ignore[arg-type]
        policy=torch.nn.Identity(),
        correct_video=_video(),
        counterfactual_video=_video(50),
        language_tokens=target_tokens,
        kind="wrong",
        counterfactual_seed=19,
        task_ordinal=1,
        task_visit=2,
        teacher_demo=3,
        device=torch.device("cpu"),
    )
    assert len(writer.base_writer.encoder_tokens) == 1
    assert len(writer.condition_tokens) == 1
    assert all(
        torch.equal(tokens, target_tokens[0]) for tokens in writer.condition_tokens
    )
    assert not torch.equal(
        writer.condition_frames[0],
        writer.condition_frames[1],
    )
    assert graph.negative_raw_frames == 16
    assert graph.negative_sampled_frames == 4


def test_program_cotangent_transports_complete_a_b_vjp_without_hidden_scaling() -> None:
    leaf = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).requires_grad_(True)
    outputs = {
        "target.lora_A.default.weight": 2.0 * leaf,
        "target.lora_B.default.weight": leaf.square(),
    }
    graph = GeneratedConditionGraph(
        correct_lora=outputs,
        program_leaf=leaf,
        program_input_before=leaf.detach(),
        correct_feature=torch.zeros(256),
        negative_feature=torch.zeros(256),
        correct_raw_frames=1,
        correct_sampled_frames=1,
        negative_raw_frames=1,
        negative_sampled_frames=1,
    )
    gradients = {
        "target.lora_A.default.weight": torch.full_like(leaf, 3.0),
        "target.lora_B.default.weight": torch.full_like(leaf, 4.0),
    }
    observed = program_cotangent(graph, gradients)
    torch.testing.assert_close(observed, (6.0 + 8.0 * leaf.detach())[0])
    with pytest.raises(ExpertManifoldError, match="topology changed"):
        program_cotangent(
            graph,
            {"target.lora_A.default.weight": torch.ones_like(leaf)},
        )


def test_redecoded_program_cotangent_matches_the_original_compiler_graph() -> None:
    writer = _Writer()
    leaf = torch.randn(1, 320, 2, generator=torch.Generator().manual_seed(41))
    leaf.requires_grad_(True)
    state = writer.base_writer.decode_slots(leaf)
    gradients = {
        name: torch.randn_like(value, generator=torch.Generator().manual_seed(index))
        for index, (name, value) in enumerate(state.items(), start=43)
    }
    graph = GeneratedConditionGraph(
        correct_lora=state,
        program_leaf=leaf,
        program_input_before=leaf.detach(),
        correct_feature=torch.zeros(256),
        negative_feature=torch.zeros(256),
        correct_raw_frames=1,
        correct_sampled_frames=1,
        negative_raw_frames=1,
        negative_sampled_frames=1,
    )
    expected = program_cotangent(graph, gradients)
    observed = redecoded_program_cotangent(
        writer=writer,  # type: ignore[arg-type]
        program_value=leaf,
        lora_gradients=gradients,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
