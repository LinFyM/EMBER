"""Pure native-response -> layered process -> one complete task LoRA graph."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract
from ember.writer.coordinate import CoordinateLoRADecoder
from ember.writer.relation import LocalRelationBlock


@dataclass(frozen=True)
class LayeredWriterConfig:
    width: int = 256
    heads: int = 8
    layers: int = 18
    horizon: int = 50
    native_width: int = 1024
    language_width: int = 2048
    blocks: int = 4
    radius: int = 4
    compiler_blocks: int = 2
    coordinate_width: int = 64
    coordinate_readout: str = "target_rank"
    edge_chunk: int = 16
    coordinate_chunk: int = 256
    activation_checkpoint: bool = True

    def __post_init__(self) -> None:
        positive = (self.width, self.heads, self.layers, self.horizon, self.native_width,
                    self.language_width, self.coordinate_width, self.edge_chunk, self.coordinate_chunk)
        if min(positive) <= 0 or self.width % self.heads:
            raise ValueError("Writer dimensions must be positive and width divisible by heads")
        if min(self.blocks, self.radius) < 0 or self.compiler_blocks <= 0:
            raise ValueError("invalid relation radius/block or compiler block count")
        if self.coordinate_readout != "target_rank":
            raise ValueError("canonical coordinate readout must be target_rank; incompatible models require fresh training")


class _Attention(nn.Module):
    """Standard projected multihead read with broadcastable batch dimensions."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)

    def _heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.head_width)).transpose(-3, -2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
        result = F.scaled_dot_product_attention(
            self._heads(self.query(query)), self._heads(self.key(key)),
            self._heads(self.value(value)), attn_mask=mask,
        )
        return self.output(result.transpose(-3, -2).flatten(-2))


class _CompilerBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.cross_norm, self.self_norm, self.ffn_norm = (nn.LayerNorm(width) for _ in range(3))
        self.memory_norm = nn.LayerNorm(width)
        self.cross, self.self_attention = _Attention(width, heads), _Attention(width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, query: Tensor, memory: Tensor, routing: Tensor, prior: Tensor) -> Tensor:
        values = self.memory_norm(memory)
        query = query + self.cross(self.cross_norm(query), values + routing, values, prior)
        normalized = self.self_norm(query)
        query = query + self.self_attention(normalized, normalized, normalized)
        return query + self.ffn(self.ffn_norm(query))


class LayeredRelationWriter(nn.Module):
    """Owns no native policy; receives only responses, real times, and language."""

    def __init__(self, contract: LoRAContract, config: LayeredWriterConfig = LayeredWriterConfig()) -> None:
        super().__init__()
        self.contract, self.config = contract, config
        width = config.width
        self.language_input = nn.Linear(config.language_width, width)
        self.language_query = nn.Parameter(torch.empty(1, width))
        self.language_read = _Attention(width, config.heads)
        self.input_projection = nn.Linear(config.native_width, width)
        self.layer_embedding = nn.Parameter(torch.empty(config.layers, width))
        self.relation_blocks = nn.ModuleList([
            LocalRelationBlock(width, config.heads, config.horizon, config.radius,
                               config.edge_chunk, config.activation_checkpoint)
            for _ in range(config.blocks)
        ])
        self.read_norm = nn.LayerNorm(width)
        self.read_language = nn.Linear(width, width)
        self.read_layers = nn.Parameter(torch.empty(config.layers, width))
        self.horizon_read = _Attention(width, config.heads)
        self.target_queries = nn.Parameter(torch.empty(len(contract.targets), width))
        self.rank_queries = nn.Parameter(torch.empty(contract.rank, width))
        self.query_language = nn.Linear(width, width)
        self.memory_layers = nn.Parameter(torch.empty(config.layers, width))
        self.time_projection = nn.Linear(width, width, bias=False)
        self.compiler = nn.ModuleList([_CompilerBlock(width, config.heads) for _ in range(config.compiler_blocks)])
        self.decoder = CoordinateLoRADecoder(contract, width, config.coordinate_width,
                                             config.coordinate_chunk, config.activation_checkpoint)
        for parameter in (self.language_query, self.layer_embedding, self.read_layers,
                          self.target_queries, self.rank_queries, self.memory_layers):
            nn.init.normal_(parameter, std=0.02)
        frequencies = torch.exp(-math.log(10000) * torch.arange(0, width, 2) / width)
        self.register_buffer("time_frequencies", frequencies, persistent=False)

    def encode_language(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.config.language_width:
            raise ValueError("language embeddings must have shape [tokens, language_width]")
        if mask.shape != embeddings.shape[:1] or not mask.bool().any():
            raise ValueError("language mask must include at least one valid token")
        tokens = self.language_input(embeddings)
        return self.language_read(self.language_query, tokens, tokens, mask.bool()[None, :]).squeeze(0)

    def encode_video(self, responses: Tensor, frame_indices: Tensor, language: Tensor) -> Tensor:
        """Return [T,J,d] local E; language is the masked language-read vector."""
        expected = (self.config.layers, self.config.horizon, self.config.native_width)
        if responses.ndim != 4 or responses.shape[1:] != expected or not len(responses):
            raise ValueError(f"native responses must have shape [T,{expected}], T > 0")
        if frame_indices.shape != responses.shape[:1] or not torch.isfinite(frame_indices).all():
            raise ValueError("one finite real frame index is required per response")
        if len(frame_indices) > 1 and not (frame_indices[1:] > frame_indices[:-1]).all():
            raise ValueError("video frame indices must be strictly increasing")
        states = self.input_projection(responses) + self.layer_embedding[None, :, None, :]
        for block in self.relation_blocks:
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                states = checkpoint(block, states, frame_indices, use_reentrant=False)
            else:
                states = block(states, frame_indices)
        query = (self.read_language(language) + self.read_layers)[:, None, :]
        normalized = self.read_norm(states)
        return self.horizon_read(query, normalized, normalized).squeeze(-2)

    def _memory(self, videos: Sequence[Tensor], frame_indices: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        memories, routes, priors = [], [], []
        for video, indices in zip(videos, frame_indices, strict=True):
            if video.ndim != 3 or video.shape[1:] != (self.config.layers, self.config.width) or not len(video):
                raise ValueError("compiled video memories must have shape [T,J,d], T > 0")
            if indices.shape != video.shape[:1]:
                raise ValueError("memory time indices do not match video length")
            phase = (indices.to(video.dtype)[:, None] / 5) * self.time_frequencies.to(video.dtype)
            time = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[:, :self.config.width]
            route = self.time_projection(time)[:, None, :] + self.memory_layers[None, :, :]
            memories.append(video.flatten(0, 1))
            routes.append(route.flatten(0, 1))
            priors.append(video.new_full((video.shape[0] * video.shape[1],), -math.log(video.shape[0] * video.shape[1])))
        return torch.cat(memories), torch.cat(routes), torch.cat(priors)[None, :]

    def compile(self, videos: Sequence[Tensor], frame_indices: Sequence[Tensor], language: Tensor) -> Tensor:
        """Return paired [target,rank,d] codes from the whole video set."""
        if not videos or len(videos) != len(frame_indices):
            raise ValueError("a condition needs one or more videos with matching time arrays")
        memory, routing, prior = self._memory(videos, frame_indices)
        query = self.target_queries[:, None, :] + self.rank_queries[None, :, :] + self.query_language(language)
        query = query.flatten(0, 1)
        for block in self.compiler:
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                query = checkpoint(block, query, memory, routing, prior, use_reentrant=False)
            else:
                query = block(query, memory, routing, prior)
        return query.unflatten(0, (len(self.contract.targets), self.contract.rank))

    def forward(
        self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor],
        language_embeddings: Tensor, language_mask: Tensor,
    ) -> dict[str, Tensor]:
        if not responses or len(responses) != len(frame_indices):
            raise ValueError("a condition needs one or more response videos with matching time arrays")
        language = self.encode_language(language_embeddings, language_mask)
        videos = [self.encode_video(response, indices, language)
                  for response, indices in zip(responses, frame_indices, strict=True)]
        return self.decoder(self.compile(videos, frame_indices, language))
