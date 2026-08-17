"""Shared CPU fixtures for the Layer-Matched Memory Program Compiler."""

from __future__ import annotations

import torch

from ember.expert_manifold.legacy_v6_model import build_lora_tensor_specs
from ember.writer.backbone_memory import LayerMatchedVideoEncoding
from ember.writer.model import CompleteLoRAWriter
from ember.writer.parameter_grid import (
    AddressPreservingVideoSet,
    LayerMatchedMemoryProgramCompiler,
    LayerRankMemoryReader,
)
from ember.writer.temporal import (
    CausalProcedureEncoder,
    LanguageSemanticCore,
    TaskGroundedVisualTransitionFusion,
)


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


class _FakeSemanticEncoder(torch.nn.Module):
    pass


class _FakeBackboneMemoryEncoder(torch.nn.Module):
    """Emit differentiable frame evidence and exact 18x16 addressed memories."""

    def forward(
        self,
        _semantic: torch.nn.Module,
        _policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        _language_mask: torch.Tensor,
        _task_span_mask: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> LayerMatchedVideoEncoding:
        frame = frames.float().mean(dim=(1, 2, 3)).div(255.0)
        language = language_tokens.float().mean(dim=1).div(32.0)
        frame_language = language.index_select(0, frame_condition_ids)
        width = torch.arange(256, dtype=torch.float32)[None].mul_(1e-3)
        token = torch.arange(2, dtype=torch.float32)[None, :, None].mul_(0.02)
        text = language[:, None, None] + token + width[:, None]
        evidence = frame[:, None, None] + frame_language[:, None, None]
        evidence = evidence + token + width[:, None]
        grounded = evidence + frame.square()[:, None, None] * 0.1
        interaction = frame[:, None] * (1.0 + width)
        interaction = interaction + frame_language[:, None] * 0.1
        layer = torch.arange(18, dtype=torch.float32)[None, :, None, None]
        layer = layer.mul_(0.01)
        context = 1.0 + frame[:, None, None, None] * 0.2
        context = context + frame_language[:, None, None, None] * 0.05 + layer
        rank_memory = memory_tokens[None, None] * context
        rank_address = torch.arange(16, dtype=torch.float32)[None, None, :, None]
        rank_memory = rank_memory + rank_address * frame[:, None, None, None] * 1e-3
        return LayerMatchedVideoEncoding(
            text_queries=text,
            frame_evidence=evidence,
            grounded_evidence=grounded,
            interactions=interaction,
            valid_task_tokens=torch.ones(
                language_tokens.shape[0], 2, dtype=torch.bool
            ),
            layer_memory=rank_memory,
        )


def _model() -> tuple[CompleteLoRAWriter, dict[str, torch.Tensor]]:
    template = _template()
    model = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        semantic_encoder=_FakeSemanticEncoder(),
        semantic_core=LanguageSemanticCore(width=256, heads=8, blocks=1),
        visual_transition=TaskGroundedVisualTransitionFusion(width=256, heads=8),
        procedure=CausalProcedureEncoder(width=256, heads=8, blocks=1),
        backbone_memory_encoder=_FakeBackboneMemoryEncoder(),
        memory_reader=LayerRankMemoryReader(heads=8, initialization_seed=7),
        video_set=AddressPreservingVideoSet(),
        compiler=LayerMatchedMemoryProgramCompiler(
            heads=8,
            blocks=1,
            max_relative_correction=0.5,
            initialization_seed=7,
        ),
        factor_hidden_width=32,
        initialization_seed=7,
    )
    return model, template


def _open_factor_heads(model: CompleteLoRAWriter) -> None:
    generator = torch.Generator(device="cpu").manual_seed(41)
    with torch.no_grad():
        for head in model.factor_heads.values():
            head.network[-1].weight.normal_(std=0.01, generator=generator)


def _inputs() -> tuple[torch.Tensor, ...]:
    frames = torch.arange(8 * 3 * 4 * 4, dtype=torch.int64)
    frames = frames.remainder(251).to(torch.uint8).reshape(8, 3, 4, 4)
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
