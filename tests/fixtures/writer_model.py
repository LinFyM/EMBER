"""Shared rank-8 Dynamic-K Writer fixtures."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.writer.backbone_memory import BackboneMemoryOutput
from ember.writer.lora_mapper import build_lora_tensor_specs
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
                8, 1024, generator=generator
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output_width, 8
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(
            8, input_width, generator=generator
        )
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, 8)
    return state


class _FakeBackboneMemory(torch.nn.Module):
    def forward(
        self,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> BackboneMemoryOutput:
        language = language_tokens.float().mean(dim=1)
        frame = frames.float().mean(dim=(1, 2, 3))
        content = frame + language.index_select(0, frame_condition_ids)
        layer = torch.arange(18, device=frames.device).float()[None, :, None, None]
        rank = torch.arange(8, device=frames.device).float()[None, None, :, None]
        memory = content[:, None, None, None] + 0.01 * layer + 0.001 * rank
        memory = memory.expand(-1, -1, -1, 1024).clone()
        maximum = int(task_span_mask.sum(dim=1).max())
        return BackboneMemoryOutput(
            layer_memory=memory,
            probe_hidden=memory.new_zeros(frames.shape[0], 50, 1024),
            task_hidden=memory.new_zeros(frames.shape[0], maximum, 2048),
            valid_task_tokens=torch.ones(
                frames.shape[0], maximum, dtype=torch.bool, device=frames.device
            ),
        )


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    template = _template()
    bridge = SimpleNamespace(
        paligemma=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[None] * 18))
        ),
        gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=[None] * 18)),
    )
    model = CompleteLoRAWriter.__new__(CompleteLoRAWriter)
    torch.nn.Module.__init__(model)
    model.tensor_specs = build_lora_tensor_specs(template)
    model.backbone_memory = _FakeBackboneMemory()
    from ember.writer.memory_program import DynamicKMemoryProgram
    from ember.writer.lora_mapper import CompleteLoRAMapper

    model.memory_program = DynamicKMemoryProgram()
    model.lora_mapper = CompleteLoRAMapper(
        model.tensor_specs,
        template_state=template,
        program_width=256,
        mapper_width=1024,
        dynamic_a=False,
    )
    return model, template


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
