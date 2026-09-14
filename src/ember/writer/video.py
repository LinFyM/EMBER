"""Changes in a task-grounded video yield action cotangents for a fixed source.

Static semantics and native action knowledge condition attention and memory
gates. Only actual changes enter the process Values consumed by the q head.
The source compiler owns the fixed, differentiable q-to-LoRA relationship.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract
from ember.writer.attention import Attention, RotaryBlock, feed_forward
from ember.writer.factor import SharedSourceOutlet


SCHEMA = "video_conditioned_writer_v9"
ARCHITECTURE = "source_pullback_process_lora_v2"


def require_architecture_identity(config: Mapping[str, object]) -> None:
    if (config.get("schema") != SCHEMA or config.get("architecture") != ARCHITECTURE
            or config.get("process_mode") != "ordered"):
        raise ValueError("incompatible video Writer identity; use its frozen runtime")


@dataclass(frozen=True)
class VideoWriterConfig:
    width: int = 256
    heads: int = 8
    horizon: int = 50
    native_width: int = 1024
    language_width: int = 2048
    blocks: int = 2
    query_chunk: int = 32
    activation_checkpoint: bool = True
    process_mode: str = "ordered"
    memory_timescale: float = 64.0
    action_width: int = 7
    schema: str = SCHEMA
    architecture: str = ARCHITECTURE

    def __post_init__(self) -> None:
        require_architecture_identity(vars(self))
        dimensions = (self.width, self.heads, self.horizon, self.native_width, self.language_width,
                      self.blocks, self.query_chunk, self.action_width)
        if min(dimensions) <= 0 or self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("positive dimensions and even RoPE width per head are required")
        if not math.isfinite(self.memory_timescale) or self.memory_timescale < 2:
            raise ValueError("process memory requires a finite timescale of at least two frames")

    def to_dict(self) -> dict:
        return asdict(self)


class _ChangeMemory(nn.Module):
    """Conditioned retention of real changes, with no static-content injection."""

    def __init__(self, width: int, timescale: float) -> None:
        super().__init__()
        self.condition_norm = nn.LayerNorm(width)
        self.gates = nn.Linear(2 * width, 2 * width)
        with torch.no_grad():
            # Cover short and long initial lifetimes. Gate weights still
            # respond to the current semantic condition and process memory.
            lifetimes = torch.linspace(2., timescale, width)
            self.gates.bias[:width].copy_((lifetimes - 1).log())
            self.gates.bias[width:].zero_()

    def forward(self, states: Tensor, *, reverse: bool = False) -> Tensor:
        ordered = states.flip(0) if reverse else states
        condition = self.condition_norm(ordered)
        memory = torch.zeros_like(ordered[0], dtype=torch.float32)
        previous, values = ordered[0].float(), []
        for index in range(len(ordered)):
            current = ordered[index].float()
            delta = current - previous
            gate_input = torch.cat((condition[index], F.layer_norm(memory, memory.shape[-1:])), -1)
            retain, accept = self.gates(gate_input).float().sigmoid().chunk(2, -1)
            memory = retain * memory + accept * delta
            values.append(memory)
            previous = current
        result = torch.stack(values)
        return result.flip(0) if reverse else result


class _VideoEncoder(nn.Module):
    """Shared language roles align frames before directional process reading."""

    def __init__(self, config: VideoWriterConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.visual_projection = nn.Linear(config.language_width, width, bias=False)
        self.task_norm, self.patch_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.patch_read = Attention(width, config.heads)
        self.native_projection = nn.Linear(config.native_width, width, bias=False)
        self.native_norm, self.role_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.horizon_embedding = nn.Parameter(torch.randn(config.horizon, width) * .02)
        self.horizon_read = Attention(width, config.heads)
        self.fusion = feed_forward(width, 2 * width)
        self.language_blocks = nn.ModuleList([
            RotaryBlock(width, config.heads, causal=False) for _ in range(config.blocks)
        ])
        self.state_norm = nn.LayerNorm(width)
        self.past = _ChangeMemory(width, config.memory_timescale)
        self.future = _ChangeMemory(width, config.memory_timescale)

    def _ground(self, visual: Tensor, valid: Tensor, task_mask: Tensor, language: Tensor) -> Tensor:
        image_mask = valid & ~task_mask
        if not image_mask.any(-1).all():
            raise ValueError("every frame requires real valid image patches")
        projected = self.visual_projection(visual)
        tasks = projected[task_mask].reshape(len(visual), len(language), self.config.width)
        # Shared exact-language roles query each frame without a video clock.
        query = self.task_norm(tasks + self.visual_projection(language)[None])
        return tasks + self.patch_read(query, self.patch_norm(projected), projected,
                                       image_mask[:, None, None, :])

    def forward(self, response: Tensor, visual: Tensor, valid: Tensor, task_mask: Tensor,
                language: Tensor, language_positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        grounded = self._ground(visual, valid, task_mask, language)
        native = self.native_norm(self.native_projection(response))
        # All native horizon positions survive until this actual learned read.
        read = self.horizon_read(self.role_norm(grounded),
                                 native + self.horizon_embedding[None], native)
        semantic = grounded + self.fusion(torch.cat((grounded, read), -1))
        for block in self.language_blocks:
            semantic = block(semantic, language_positions)
        semantic = self.state_norm(semantic)
        process = torch.cat((self.past(semantic), self.future(semantic, reverse=True)), 1)
        return process, semantic, native


class VideoConditionedWriter(nn.Module):
    """Predict action cotangents and share a two-sided native LoRA outlet."""

    def __init__(self, contract: LoRAContract, config: VideoWriterConfig = VideoWriterConfig()) -> None:
        super().__init__()
        if contract.alpha != contract.rank:
            raise ValueError("source pullback compilation requires alpha=rank")
        self.contract, self.config = contract, config
        width = config.width
        self.encoder = _VideoEncoder(config)
        self.action_condition = nn.Linear(config.action_width, width, bias=False)
        self.query_norm, self.key_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.direction_keys = nn.Parameter(torch.randn(2, width) * .02)
        self.process_read = Attention(width, config.heads, zero_value=True)
        self.process_ffn = nn.Sequential(nn.Linear(width, 4 * width, bias=False), nn.GELU(),
                                         nn.Linear(4 * width, width, bias=False))
        self.q_head = nn.Linear(width, config.action_width, bias=False)
        nn.init.zeros_(self.q_head.weight)
        self.outlet = SharedSourceOutlet(contract)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def compiler_parameters(self) -> Iterator[nn.Parameter]:
        # The compiler partition includes q reading and the shared outlet.
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("encoder."))

    def _validate_video(self, response, indices, visual, valid, task_mask, count) -> None:
        if response.ndim != 3 or not len(response) or response.shape[1:] != (self.config.horizon, self.config.native_width):
            raise ValueError("native response must retain every frame and complete horizon")
        if indices.shape != response.shape[:1] or not torch.isfinite(indices).all():
            raise ValueError("one finite original frame index is required per frame")
        if len(indices) > 1 and not (indices[1:] > indices[:-1]).all():
            raise ValueError("ordered video frame indices must be strictly increasing")
        if visual.ndim != 3 or visual.shape[0] != len(response) or visual.shape[-1] != self.config.language_width:
            raise ValueError("visual evidence must be [T,P,language_width]")
        if (valid.shape != visual.shape[:2] or task_mask.shape != valid.shape or (task_mask & ~valid).any()
                or not (task_mask.sum(-1) == count).all()):
            raise ValueError("every frame must retain exactly its valid contextual task-token span")

    def encode(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
               language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
               visual_task_masks: Sequence[Tensor]) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        if not (len(responses) == len(frame_indices) == len(visual_tokens)
                == len(visual_masks) == len(visual_task_masks) == 1):
            raise ValueError("the registered source pullback Writer requires one complete K1 video")
        if (language_embeddings.ndim != 2 or language_embeddings.shape[-1] != self.config.language_width
                or language_mask.shape != language_embeddings.shape[:1] or not language_mask.bool().any()):
            raise ValueError("exact language requires nonempty valid native embeddings")
        positions = language_mask.bool().nonzero().flatten()
        language = language_embeddings[language_mask.bool()]
        response, indices, visual = responses[0], frame_indices[0], visual_tokens[0]
        valid, task_mask = visual_masks[0].bool(), visual_task_masks[0].bool()
        self._validate_video(response, indices, visual, valid, task_mask, len(positions))
        args = (response, visual, valid, task_mask, language, positions.to(response.device))
        if self.config.activation_checkpoint and torch.is_grad_enabled():
            encoded = checkpoint(self.encoder, *args, use_reentrant=False)
        else:
            encoded = self.encoder(*args)
        return (encoded,)

    def action_cotangents(self, videos, predictions: Tensor) -> Tensor:
        if len(videos) != 1:
            raise ValueError("one complete K1 process must form one LoRA")
        process, semantic, native = videos[0]
        if (predictions.shape != (*native.shape[:2], self.config.action_width) or predictions.requires_grad
                or process.shape != (len(native), 2 * semantic.shape[1], self.config.width)):
            raise ValueError("process codes require frozen matching native action coordinates")
        directions = self.direction_keys[:, None].expand(-1, semantic.shape[1], -1).flatten(0, 1)
        keys = self.key_norm(semantic.repeat(1, 2, 1) + process + directions[None])
        values = F.layer_norm(process, process.shape[-1:])
        native_actions = predictions.to(native.device, dtype=native.dtype, non_blocking=True)
        query = self.query_norm(native + self.encoder.horizon_embedding[None]
                                + self.action_condition(native_actions))
        outputs = []
        for start in range(0, len(native), self.config.query_chunk):
            stop = start + self.config.query_chunk
            content = self.process_read(query[start:stop], keys[start:stop], values[start:stop])
            content = content + self.process_ffn(F.layer_norm(content, content.shape[-1:]))
            outputs.append(self.q_head(content).float())
        return torch.cat(outputs)

    def raw_factors(self, videos, native_inputs) -> dict[str, Tensor]:
        """Compile once; training retains this small A0/B0 for outlet replay."""
        return native_inputs.compile(self.action_cotangents(videos, native_inputs.predictions))

    def decode(self, videos, native_inputs) -> dict[str, Tensor]:
        return self.outlet(self.raw_factors(videos, native_inputs))

    def forward(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
                language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
                visual_task_masks: Sequence[Tensor], *, native_inputs) -> dict[str, Tensor]:
        return self.decode(self.encode(responses, frame_indices, language_embeddings, language_mask,
                                       visual_tokens, visual_masks, visual_task_masks), native_inputs)
