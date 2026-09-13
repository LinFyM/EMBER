"""Task-grounded local process evidence compiled once into a corrective LoRA."""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract
from ember.writer.attention import Attention, RotaryBlock, feed_forward
from ember.writer.factor import LocalFieldLoRADecoder


SCHEMA = "video_conditioned_writer_v7"
ARCHITECTURE = "local_correction_field_lora_v1"


def require_architecture_identity(config: Mapping[str, object]) -> None:
    if (config.get("schema") != SCHEMA or config.get("architecture") != ARCHITECTURE
            or config.get("process_mode") not in ("ordered", "frame_set")):
        raise ValueError("incompatible video Writer identity; use its frozen runtime")


@dataclass(frozen=True)
class VideoWriterConfig:
    width: int = 256
    heads: int = 8
    horizon: int = 50
    native_width: int = 1024
    language_width: int = 2048
    blocks: int = 2
    factor_width: int = 256
    field_unit: float = 5.823084826577233e-6
    query_chunk: int = 32
    activation_checkpoint: bool = True
    process_mode: str = "ordered"
    schema: str = SCHEMA
    architecture: str = ARCHITECTURE

    def __post_init__(self) -> None:
        require_architecture_identity(vars(self))
        dimensions = (self.width, self.heads, self.horizon, self.native_width, self.language_width,
                      self.blocks, self.factor_width, self.query_chunk)
        if min(dimensions) <= 0 or self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("positive dimensions and even RoPE width per head are required")
        if not math.isfinite(self.field_unit) or self.field_unit <= 0:
            raise ValueError("the local field requires a fixed positive physical unit")

    def to_dict(self) -> dict:
        return asdict(self)


class _VideoEncoder(nn.Module):
    """Preserve current native positions while interpreting their full video context."""

    def __init__(self, config: VideoWriterConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.visual_projection = nn.Linear(config.language_width, width, bias=False)
        self.task_norm, self.patch_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.patch_read = Attention(width, config.heads)
        self.context_norm, self.context_query_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.context_read = Attention(width, config.heads)
        self.native_projection = nn.Linear(config.native_width, width, bias=False)
        self.horizon_norm, self.horizon_query_norm = nn.LayerNorm(width), nn.LayerNorm(2 * width)
        self.horizon_query = nn.Linear(2 * width, width)
        self.horizon_embedding = nn.Parameter(torch.randn(config.horizon, width) * .02)
        self.horizon_read = Attention(width, config.heads)
        self.fusion = feed_forward(width, 3 * width)
        self.language_blocks = nn.ModuleList([
            RotaryBlock(width, config.heads, causal=False) for _ in range(config.blocks)
        ])
        self.temporal_blocks = nn.ModuleList([
            RotaryBlock(width, config.heads, causal=False) for _ in range(config.blocks)
        ])
        self.local_query_norm, self.role_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.local_read = Attention(width, config.heads)
        self.local_fusion = feed_forward(width, 2 * width)

    def _ground(self, visual: Tensor, valid: Tensor, task_mask: Tensor, count: int) -> Tensor:
        image_mask = valid & ~task_mask
        if not image_mask.any(-1).all():
            raise ValueError("every frame requires real valid image patches")
        projected = self.visual_projection(visual)
        tasks = projected[task_mask].reshape(len(visual), count, self.config.width)
        return tasks + self.patch_read(self.task_norm(tasks), self.patch_norm(projected), projected,
                                       image_mask[:, None, None, :])

    def _native_read(self, grounded: Tensor, context: Tensor, values: Tensor) -> Tensor:
        keys = (values + self.horizon_embedding[None]).flatten(0, 1)[None]
        # Every frame and all H positions survive until the actual full-video read.
        key, value = self.horizon_read.project_memory(keys, values.flatten(0, 1)[None])
        outputs = []
        for start in range(0, len(grounded), self.config.query_chunk):
            current = grounded[start:start + self.config.query_chunk]
            whole = context[None].expand_as(current)
            query = self.horizon_query(self.horizon_query_norm(torch.cat((current, whole), -1)))
            read = self.horizon_read.read_projected(query.flatten(0, 1)[None], key, value)[0]
            read = read.reshape_as(current)
            outputs.append(current + self.fusion(torch.cat((current, whole, read), -1)))
        return torch.cat(outputs)

    def forward(self, response: Tensor, visual: Tensor, valid: Tensor, task_mask: Tensor,
                language: Tensor, language_positions: Tensor, frame_indices: Tensor) -> tuple[Tensor, Tensor]:
        grounded = self._ground(visual, valid, task_mask, len(language))
        query = self.context_query_norm(self.visual_projection(language))[None]
        memory = self.context_norm(grounded.flatten(0, 1))[None]
        context = self.context_read(query, memory, memory)[0]
        native = self.horizon_norm(self.native_projection(response))
        content = self._native_read(grounded, context, native)
        time = frame_indices.float() / 5 if self.config.process_mode == "ordered" else torch.zeros_like(frame_indices)
        for language_block, temporal in zip(self.language_blocks, self.temporal_blocks, strict=True):
            content = language_block(content, language_positions)
            content = temporal(content.transpose(0, 1), time).transpose(0, 1)
        roles = self.role_norm(content)
        # Each current (t,h) queries its own frame's roles after their actual
        # bidirectional process interpretation; local native state stays live.
        interpreted = self.local_read(self.local_query_norm(native + self.horizon_embedding[None]), roles, roles)
        local = native + self.local_fusion(torch.cat((native, interpreted), -1))
        return local, roles


class VideoConditionedWriter(nn.Module):
    """Legal video-only acquisition; source coordinates are fixed read-only values."""

    def __init__(self, contract: LoRAContract, config: VideoWriterConfig = VideoWriterConfig()) -> None:
        super().__init__()
        if contract.alpha != contract.rank:
            raise ValueError("local field contraction requires alpha=rank")
        self.contract, self.config = contract, config
        width = config.width
        self.encoder = _VideoEncoder(config)
        self.target_queries = nn.Parameter(torch.randn(len(contract.targets), width) * .02)
        self.rank_queries = nn.Parameter(torch.randn(contract.rank, width) * .02)
        self.semantic_norm, self.query_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.parameter_read = Attention(width, config.heads)
        self.code_norm, self.code_ffn = nn.LayerNorm(width), feed_forward(width)
        self.decoder = LocalFieldLoRADecoder(contract, width, config.factor_width, config.field_unit,
            chunk_size=config.query_chunk, activation_checkpoint=config.activation_checkpoint)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def compiler_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("encoder."))

    def _validate_video(self, response, indices, visual, valid, task_mask, count) -> None:
        if response.ndim != 3 or not len(response) or response.shape[1:] != (self.config.horizon, self.config.native_width):
            raise ValueError("native response must retain every frame and complete horizon")
        if indices.shape != response.shape[:1] or not torch.isfinite(indices).all():
            raise ValueError("one finite original frame index is required per frame")
        if (self.config.process_mode == "ordered" and len(indices) > 1
                and not (indices[1:] > indices[:-1]).all()):
            raise ValueError("ordered video frame indices must be strictly increasing")
        if visual.ndim != 3 or visual.shape[0] != len(response) or visual.shape[-1] != self.config.language_width:
            raise ValueError("visual evidence must be [T,P,language_width]")
        if (valid.shape != visual.shape[:2] or task_mask.shape != valid.shape or (task_mask & ~valid).any()
                or not (task_mask.sum(-1) == count).all()):
            raise ValueError("every frame must retain exactly its valid contextual task-token span")

    def encode(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
               language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
               visual_task_masks: Sequence[Tensor]) -> tuple[tuple[Tensor, Tensor], ...]:
        if not (len(responses) == len(frame_indices) == len(visual_tokens)
                == len(visual_masks) == len(visual_task_masks) == 1):
            raise ValueError("the registered local field Writer requires one complete K1 video")
        if (language_embeddings.ndim != 2 or language_embeddings.shape[-1] != self.config.language_width
                or language_mask.shape != language_embeddings.shape[:1] or not language_mask.bool().any()):
            raise ValueError("exact language requires nonempty valid native embeddings")
        positions = language_mask.bool().nonzero().flatten()
        language = language_embeddings[language_mask.bool()]
        response, indices, visual = responses[0], frame_indices[0], visual_tokens[0]
        valid, task_mask = visual_masks[0].bool(), visual_task_masks[0].bool()
        self._validate_video(response, indices, visual, valid, task_mask, len(positions))
        args = (response, visual, valid, task_mask, language, positions.to(response.device), indices.to(response.device))
        if self.config.activation_checkpoint and torch.is_grad_enabled():
            encoded = checkpoint(self.encoder, *args, use_reentrant=False)
        else:
            encoded = self.encoder(*args)
        return (encoded,)

    def _compile(self, videos, native_inputs, field_indices=None):
        if len(videos) != 1 or len(native_inputs) != 1:
            raise ValueError("one complete K1 field must form one LoRA")
        local, roles = videos[0]
        if (roles.ndim != 3 or roles.shape[0] != len(local) or roles.shape[1] <= 0
                or roles.shape[2] != self.config.width or local.shape[1:] != (self.config.horizon, self.config.width)):
            raise ValueError("local fields and language roles must retain their complete native positions")
        memory = self.semantic_norm(roles.flatten(0, 1))[None]
        query = (self.target_queries[:, None] + self.rank_queries[None]).flatten(0, 1)
        code = self.parameter_read(self.query_norm(query)[None], memory, memory)[0]
        code = code + self.code_ffn(self.code_norm(code))
        return self.decoder(code.unflatten(0, (len(self.contract.targets), self.contract.rank)),
                            local, native_inputs[0], field_indices)

    def decode(self, videos, native_inputs) -> dict[str, Tensor]:
        return self._compile(videos, native_inputs)[0]

    def decode_with_fields(self, videos, native_inputs, field_indices: Tensor):
        return self._compile(videos, native_inputs, field_indices)

    def forward(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
                language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
                visual_task_masks: Sequence[Tensor], *, native_inputs) -> dict[str, Tensor]:
        return self.decode(self.encode(responses, frame_indices, language_embeddings, language_mask,
                                       visual_tokens, visual_masks, visual_task_masks), native_inputs)
