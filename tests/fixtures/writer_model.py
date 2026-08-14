"""Shared fixtures for the v6 Semantic-Core set Writer."""

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
    core: torch.Tensor
    valid_core: torch.Tensor
    procedure: torch.Tensor
    positions: torch.Tensor
    valid_procedure: torch.Tensor


class _FakeCompiler(torch.nn.Module):
    def routing(self, batch: int) -> torch.Tensor:
        slot = torch.arange(320).float()[:, None] * 1e-3
        width = torch.arange(256).float()[None] * 1e-4
        return (slot + width)[None].expand(batch, -1, -1)

    @staticmethod
    def read_core_slots(
        routing: torch.Tensor,
        core: torch.Tensor,
        valid_core: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_core[..., None]
        mean = core.masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
        return routing + mean[:, None]

    @staticmethod
    def normalize_core_slots(core_slots: torch.Tensor) -> torch.Tensor:
        return core_slots

    @staticmethod
    def read_procedure_slots(
        routing: torch.Tensor,
        normalized_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (positions + 1).to(procedure.dtype) * valid_procedure
        summary = (procedure * weights[..., None]).sum(1)
        summary = summary / weights.sum(1, keepdim=True).clamp_min(1)
        return routing + normalized_core + summary[:, None], procedure

    @staticmethod
    def fuse_readouts(
        _routing: torch.Tensor,
        normalized_core: torch.Tensor,
        procedure_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(procedure_slots)
        return normalized_core + procedure_slots, zeros, zeros


class _FakeV6Base(torch.nn.Module):
    program_width = 256
    PUBLIC_LORA_RANK = 16

    def __init__(self, template: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._template = template
        self.factor = torch.nn.Linear(256, 1, bias=False)
        torch.nn.init.normal_(self.factor.weight, std=0.01)
        self.compiler = _FakeCompiler()

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
        videos = len(evidence.offsets) - 1
        lengths = [right - left for left, right in zip(evidence.offsets, evidence.offsets[1:])]
        maximum = max(lengths)
        core = torch.zeros(videos, 2, 256)
        procedure = torch.zeros(videos, maximum, 256)
        positions = torch.zeros(videos, maximum, dtype=torch.long)
        valid = torch.zeros(videos, maximum, dtype=torch.bool)
        width = torch.arange(256).float() * 1e-4
        for video, (left, right) in enumerate(
            zip(evidence.offsets, evidence.offsets[1:])
        ):
            values = evidence.frames[left:right].float().mean(dim=(1, 2, 3))
            language = evidence.language_tokens[video].float().mean()
            core[video] = values.mean() + language + width
            procedure[video, : right - left] = values[:, None] + width
            positions[video, : right - left] = frame_indices[left:right]
            valid[video, : right - left] = True
        return _Memories(
            core=core,
            valid_core=torch.ones(videos, 2, dtype=torch.bool),
            procedure=procedure,
            positions=positions,
            valid_procedure=valid,
        )

    def compile_slots(self, memories: _Memories) -> torch.Tensor:
        routing = self.compiler.routing(memories.core.shape[0])
        core = self.compiler.read_core_slots(
            routing, memories.core, memories.valid_core
        )
        normalized = self.compiler.normalize_core_slots(core)
        procedure, _ = self.compiler.read_procedure_slots(
            routing,
            normalized,
            memories.procedure,
            memories.positions,
            memories.valid_procedure,
        )
        return self.compiler.fuse_readouts(routing, normalized, procedure)[0]

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
