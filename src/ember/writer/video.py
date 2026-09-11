"""Exact task-token video evidence compiled once into one complete task LoRA."""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract
from ember.writer.attention import Attention, CompilerBlock, RotaryBlock, feed_forward, position_encoding
from ember.writer.native_factor import NativeFactorLoRADecoder


SCHEMA = "video_conditioned_writer_v2"
ARCHITECTURE = "task_token_adjacent_full_h_v1"


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
    compiler_blocks: int = 2
    factor_width: int = 256
    edge_chunk: int = 32
    activation_checkpoint: bool = True
    process_mode: str = "ordered"
    schema: str = SCHEMA
    architecture: str = ARCHITECTURE

    def __post_init__(self) -> None:
        require_architecture_identity(vars(self))
        dimensions = (self.width, self.heads, self.horizon, self.native_width, self.language_width,
                      self.blocks, self.compiler_blocks, self.factor_width, self.edge_chunk)
        if min(dimensions) <= 0 or self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("positive dimensions and even RoPE width per head are required")


class _VideoEncoder(nn.Module):
    """Own every parameter used to produce E[T,L,d], independently per video."""

    def __init__(self, config: VideoWriterConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.visual_projection = nn.Linear(config.language_width, width, bias=False)
        self.task_norm, self.patch_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.patch_read = Attention(width, config.heads)
        self.native_projection = nn.Linear(config.native_width, width)
        self.horizon_norm, self.pair_norm = nn.LayerNorm(width), nn.LayerNorm(2 * width)
        self.pair_query = nn.Linear(2 * width, width)
        self.horizon_embedding = nn.Parameter(torch.randn(config.horizon, width) * 0.02)
        self.endpoint_embedding = nn.Parameter(torch.randn(2, width) * 0.02)
        self.endpoint_time = nn.Linear(width, width, bias=False)
        self.horizon_read = Attention(width, config.heads)
        self.fusion = feed_forward(width, 3 * width)
        self.language_blocks = nn.ModuleList([
            RotaryBlock(width, config.heads, causal=False) for _ in range(config.blocks)
        ])
        self.temporal_blocks = nn.ModuleList([
            RotaryBlock(width, config.heads, causal=config.process_mode == "ordered")
            for _ in range(config.blocks)
        ])

    def _ground(self, visual: Tensor, valid: Tensor, task_mask: Tensor, token_count: int) -> Tensor:
        image_mask = valid & ~task_mask
        if not image_mask.any(-1).all():
            raise ValueError("every frame requires real valid image patches")
        projected = self.visual_projection(visual)
        tasks = projected[task_mask].reshape(len(visual), token_count, self.config.width)
        grounded = self.patch_read(self.task_norm(tasks), self.patch_norm(projected), projected,
                                   image_mask[:, None, None, :])
        return tasks + grounded

    def _pair_read(self, grounded: Tensor, native: Tensor, times: Tensor) -> Tensor:
        count, width = len(grounded), self.config.width
        current = torch.arange(count, device=grounded.device)
        previous = (current - 1).clamp_min(0) if self.config.process_mode == "ordered" else current
        values = self.horizon_norm(self.native_projection(native))
        contents = []
        for start in range(0, count, self.config.edge_chunk):
            now, before = current[start:start + self.config.edge_chunk], previous[start:start + self.config.edge_chunk]
            z_before, z_now = grounded[before], grounded[now]
            query = self.pair_query(self.pair_norm(torch.cat((z_before, z_now), -1)))
            # Both COMPLETE H endpoints remain available until this task-conditioned read.
            pair_values = torch.stack((values[before], values[now]), 1)
            keys = pair_values + self.horizon_embedding[None, None, :, :]
            if self.config.process_mode == "ordered":
                offsets = torch.stack((times[before] - times[now], torch.zeros_like(times[now])), 1) / 5
                route = self.endpoint_time(position_encoding(offsets, width, values.dtype))
                keys = keys + (self.endpoint_embedding[None, :, :] + route)[:, :, None, :]
            read = self.horizon_read(query, keys.flatten(1, 2), pair_values.flatten(1, 2))
            contents.append(z_now + self.fusion(torch.cat((z_before, z_now, read), -1)))
        return torch.cat(contents)

    def forward(self, response: Tensor, indices: Tensor, visual: Tensor, valid: Tensor,
                task_mask: Tensor, positions: Tensor) -> Tensor:
        times = indices.to(device=response.device, dtype=torch.float32)
        grounded = self._ground(visual, valid, task_mask, len(positions))
        content = self._pair_read(grounded, response, times)
        temporal_positions = times / 5 if self.config.process_mode == "ordered" else torch.zeros_like(times)
        for language, temporal in zip(self.language_blocks, self.temporal_blocks, strict=True):
            content = language(content, positions)
            content = temporal(content.transpose(0, 1), temporal_positions).transpose(0, 1)
        return content


class VideoConditionedWriter(nn.Module):
    """Deployment inputs are native video evidence and exact language, with no source-policy owner."""

    def __init__(self, contract: LoRAContract, config: VideoWriterConfig = VideoWriterConfig()) -> None:
        super().__init__()
        self.contract, self.config = contract, config
        width = config.width
        self.encoder = _VideoEncoder(config)
        self.target_queries = nn.Parameter(torch.randn(len(contract.targets), width) * 0.02)
        self.rank_queries = nn.Parameter(torch.randn(contract.rank, width) * 0.02)
        self.time_projection = nn.Linear(width, width, bias=False)
        self.compiler = nn.ModuleList([CompilerBlock(width, config.heads) for _ in range(config.compiler_blocks)])
        self.decoder = NativeFactorLoRADecoder(contract, width, config.factor_width)

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        return self.encoder.parameters()

    def compiler_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("encoder."))

    def _validate_video(self, response: Tensor, indices: Tensor, visual: Tensor, valid: Tensor,
                        task_mask: Tensor, token_count: int) -> None:
        expected = (self.config.horizon, self.config.native_width)
        if response.ndim != 3 or not len(response) or response.shape[1:] != expected:
            raise ValueError(f"native response must have shape [T,{expected}], T > 0")
        if indices.shape != response.shape[:1] or not torch.isfinite(indices).all():
            raise ValueError("one finite real frame index is required per frame")
        if (self.config.process_mode == "ordered" and len(indices) > 1
                and not (indices[1:] > indices[:-1]).all()):
            raise ValueError("ordered video frame indices must be strictly increasing")
        if (visual.ndim != 3 or visual.shape[0] != len(response)
                or visual.shape[-1] != self.config.language_width):
            raise ValueError("visual evidence must have shape [T,P,language_width]")
        if (valid.shape != visual.shape[:2] or task_mask.shape != valid.shape
                or (task_mask & ~valid).any() or not (task_mask.sum(-1) == token_count).all()):
            raise ValueError("every frame must retain exactly its valid contextual task-token span")

    def encode(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
               language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
               visual_task_masks: Sequence[Tensor]) -> tuple[Tensor, ...]:
        if not responses or not (len(responses) == len(frame_indices) == len(visual_tokens)
                                 == len(visual_masks) == len(visual_task_masks)):
            raise ValueError("one or more videos need matching native responses, times and visual evidence")
        if (language_embeddings.ndim != 2 or language_embeddings.shape[-1] != self.config.language_width
                or language_mask.shape != language_embeddings.shape[:1] or not language_mask.bool().any()):
            raise ValueError("exact language needs embeddings [L,language_width] and a nonempty valid mask")
        # Content comes from the contextual native task spans, not a second language-only path.
        positions = language_mask.bool().nonzero().flatten()
        videos = []
        for response, indices, visual, valid, task_mask in zip(
                responses, frame_indices, visual_tokens, visual_masks, visual_task_masks, strict=True):
            valid, task_mask = valid.bool(), task_mask.bool()
            self._validate_video(response, indices, visual, valid, task_mask, len(positions))
            args = (response, indices, visual, valid, task_mask, positions.to(response.device))
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                videos.append(checkpoint(self.encoder, *args, use_reentrant=False))
            else:
                videos.append(self.encoder(*args))
        return tuple(videos)

    def memory(self, videos: Sequence[Tensor], frame_indices: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        if not videos or len(videos) != len(frame_indices):
            raise ValueError("a condition needs one or more encoded videos and matching frame indices")
        memories, routes, priors = [], [], []
        for video, indices in zip(videos, frame_indices, strict=True):
            if (video.ndim != 3 or min(video.shape[:2]) <= 0 or video.shape[-1] != self.config.width
                    or indices.shape != video.shape[:1]):
                raise ValueError("video evidence must have shape [T,L,d] with matching frame indices")
            memory = video.flatten(0, 1)
            if self.config.process_mode == "ordered":
                time = position_encoding(indices.to(video.device) / 5, self.config.width, video.dtype)
                routing = self.time_projection(time)[:, None, :].expand_as(video).flatten(0, 1)
            else:
                routing = torch.zeros_like(memory)
            memories.append(memory)
            routes.append(routing)
            priors.append(video.new_full((len(memory),), -math.log(len(memory))))
        return torch.cat(memories), torch.cat(routes), torch.cat(priors)[None, :]

    def decode(self, videos: Sequence[Tensor], frame_indices: Sequence[Tensor]) -> dict[str, Tensor]:
        memory, routing, prior = self.memory(videos, frame_indices)
        query = (self.target_queries[:, None, :] + self.rank_queries[None, :, :]).flatten(0, 1)
        for block in self.compiler:
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                query = checkpoint(block, query, memory, routing, prior, use_reentrant=False)
            else:
                query = block(query, memory, routing, prior)
        return self.decoder(query.unflatten(0, (len(self.contract.targets), self.contract.rank)))

    def forward(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
                language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
                visual_task_masks: Sequence[Tensor]) -> dict[str, Tensor]:
        videos = self.encode(responses, frame_indices, language_embeddings, language_mask,
                             visual_tokens, visual_masks, visual_task_masks)
        return self.decode(videos, frame_indices)
