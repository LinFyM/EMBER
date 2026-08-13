"""Shared fixtures for the v6 Dynamic Slot-Set Writer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ember.writer.model import CompleteLoRAWriter


def _template() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu").manual_seed(13)
    for layer in range(18):
        prefix = (
            "model.paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.self_attn."
        )
        for projection, output_width in (("q_proj", 2048), ("v_proj", 256)):
            state[prefix + projection + ".lora_A.default.weight"] = torch.randn(
                16, 1024, generator=generator
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output_width, 16
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(
            16, input_width, generator=generator
        )
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, 16)
    return state


@dataclass(frozen=True)
class _Evidence:
    frames: torch.Tensor
    offsets: tuple[int, ...]
    language_tokens: torch.Tensor


@dataclass(frozen=True)
class _Memories:
    evidence: _Evidence
    frame_indices: torch.Tensor


class _FakeV6Base(torch.nn.Module):
    program_width = 256
    PUBLIC_LORA_RANK = 16

    def __init__(self, template: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._template = template
        self.factor = torch.nn.Linear(256, 1, bias=False)
        torch.nn.init.normal_(self.factor.weight, std=0.01)

    def template_state(self) -> dict[str, torch.Tensor]:
        return self._template

    def encode_video_evidence(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _task_span_mask: torch.Tensor,
    ) -> _Evidence:
        return _Evidence(
            frames=frames,
            offsets=tuple(int(value) for value in video_offsets.tolist()),
            language_tokens=language_tokens,
        )

    @staticmethod
    def build_memories(
        evidence: _Evidence, frame_indices: torch.Tensor
    ) -> _Memories:
        return _Memories(evidence=evidence, frame_indices=frame_indices)

    def compile_slots(self, memories: _Memories) -> torch.Tensor:
        rows = []
        slot = torch.arange(320, device=memories.frame_indices.device).float()
        width = torch.arange(256, device=memories.frame_indices.device).float()
        for video, (left, right) in enumerate(
            zip(memories.evidence.offsets, memories.evidence.offsets[1:])
        ):
            frame_value = memories.evidence.frames[left:right].float().mean(
                dim=(1, 2, 3)
            )
            ordinal = torch.arange(
                1, right - left + 1, device=frame_value.device
            ).float()
            ordered = (frame_value * ordinal).sum() / ordinal.sum()
            language = memories.evidence.language_tokens[video].float().mean()
            rows.append(
                ordered
                + language
                + slot[:, None] * 1e-3
                + width[None] * 1e-4
            )
        return torch.stack(rows)

    def decode_slots(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        scalars = self.factor(slots[:, :76]).squeeze(-1)
        result = {}
        for index, (name, template) in enumerate(self._template.items()):
            value = template[None] + scalars[:, index, None, None]
            result[name] = value[0] if slots.shape[0] == 1 else value
        return result


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    template = _template()
    return CompleteLoRAWriter(_FakeV6Base(template)), template


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(8 * 3 * 4 * 4, dtype=torch.uint8).reshape(8, 3, 4, 4)
    frame_indices = torch.tensor([0, 5, 10, 0, 5, 0, 5, 10])
    video_offsets = torch.tensor([0, 3, 5, 8], dtype=torch.long)
    condition_video_offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    tokens = torch.tensor(
        [[1, 10, 11, 12, 13, 0], [1, 20, 21, 22, 23, 24]], dtype=torch.long
    )
    masks = tokens.ne(0)
    spans = torch.tensor(
        [
            [False, False, True, True, False, False],
            [False, True, True, True, True, False],
        ]
    )
    return (
        frames,
        frame_indices,
        video_offsets,
        condition_video_offsets,
        tokens,
        masks,
        spans,
    )
