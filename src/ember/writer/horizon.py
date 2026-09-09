"""Final native horizon + real visual evidence -> one complete task LoRA."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract
from ember.writer.attention import Attention, CompilerBlock, RotaryBlock, position_encoding
from ember.writer.native_factor import NativeFactorLoRADecoder
from ember.writer.relation import LocalRelationBlock


COMPILER_LANGUAGE_MODE = "first_query_only_v1"
PROCESS_LANGUAGE_SOURCE = "frame_contextual_task_tokens_v1"
BACKEND_CONDITIONING = ("all", "local_only", "none")


def require_architecture_identity(config: Mapping[str, object]) -> None:
    if (config.get("compiler_language_mode") != COMPILER_LANGUAGE_MODE
            or config.get("process_language_source") != PROCESS_LANGUAGE_SOURCE
            or config.get("backend_conditioning") not in BACKEND_CONDITIONING):
        raise ValueError("Writer architecture identity is missing or incompatible; use its frozen runtime")


@dataclass(frozen=True)
class HorizonWriterConfig:
    width: int = 256
    heads: int = 8
    horizon: int = 50
    native_width: int = 1024
    language_width: int = 2048
    blocks: int = 4
    radius: int = 4
    compiler_blocks: int = 2
    factor_width: int = 256
    edge_chunk: int = 8
    activation_checkpoint: bool = True
    compiler_language_mode: str = COMPILER_LANGUAGE_MODE
    process_language_source: str = PROCESS_LANGUAGE_SOURCE
    backend_conditioning: str = "all"

    def __post_init__(self) -> None:
        require_architecture_identity({"compiler_language_mode": self.compiler_language_mode,
                                       "process_language_source": self.process_language_source,
                                       "backend_conditioning": self.backend_conditioning})
        positive = (self.width, self.heads, self.horizon, self.native_width, self.language_width,
                    self.blocks, self.radius, self.compiler_blocks, self.factor_width, self.edge_chunk)
        if min(positive) <= 0 or self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("Writer dimensions must be positive; RoPE requires even width per head")


class HorizonWriteback(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.local_norm, self.process_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.local, self.process = nn.Linear(width, 2 * width), nn.Linear(width, 2 * width)
        self.output = nn.Linear(2 * width, width)

    def forward(self, states: Tensor, process: Tensor) -> Tensor:
        hidden = self.local(self.local_norm(states)) + self.process(self.process_norm(process))[:, None, :]
        return states + self.output(F.gelu(hidden))


class HorizonProcessGroup(nn.Module):
    """One synchronous local group, H read, past+self temporal block and optional writeback."""

    def __init__(self, config: HorizonWriterConfig, *, writeback: bool) -> None:
        super().__init__()
        width = config.width
        self.local = LocalRelationBlock(width, config.heads, config.horizon, config.radius,
                                        config.edge_chunk, config.activation_checkpoint, config.language_width)
        self.read_norm, self.read_language = nn.LayerNorm(width), nn.Linear(width, width)
        self.horizon_read = Attention(width, config.heads)
        self.temporal = RotaryBlock(width, config.heads, causal=True)
        self.writeback = HorizonWriteback(width) if writeback else None
        self.backend_conditioning = config.backend_conditioning

    def forward(self, states: Tensor, times: Tensor, language: Tensor, visual_tokens: Tensor,
                visual_mask: Tensor, horizon_embedding: Tensor) -> tuple[Tensor, Tensor]:
        local_condition = language if self.backend_conditioning != "none" else torch.zeros_like(language)
        states = self.local(states, times, local_condition, visual_tokens, visual_mask, horizon_embedding)
        normalized = self.read_norm(states)
        read_condition = language if self.backend_conditioning == "all" else torch.zeros_like(language)
        query = self.read_language(read_condition)[:, None, :]
        readout = self.horizon_read(query, normalized, normalized).squeeze(-2)
        process = self.temporal(readout, times / 5)
        return (self.writeback(states, process) if self.writeback is not None else states), process


class HorizonRelationWriter(nn.Module):
    """Owns no native policy and accepts no teacher state/actions/task identifiers."""

    def __init__(self, contract: LoRAContract, config: HorizonWriterConfig = HorizonWriterConfig()) -> None:
        super().__init__()
        self.contract, self.config = contract, config
        width = config.width
        self.language_input = nn.Linear(config.language_width, width)
        self.language_query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.language_read = Attention(width, config.heads)
        self.input_projection = nn.Linear(config.native_width, width)
        self.horizon_embedding = nn.Parameter(torch.randn(config.horizon, width) * 0.02)
        self.process_groups = nn.ModuleList([
            HorizonProcessGroup(config, writeback=index < config.blocks - 1) for index in range(config.blocks)
        ])
        self.target_queries = nn.Parameter(torch.randn(len(contract.targets), width) * 0.02)
        self.rank_queries = nn.Parameter(torch.randn(contract.rank, width) * 0.02)
        self.query_language = nn.Linear(width, width)
        self.time_projection = nn.Linear(width, width, bias=False)
        self.compiler = nn.ModuleList([CompilerBlock(width, config.heads) for _ in range(config.compiler_blocks)])
        self.decoder = NativeFactorLoRADecoder(contract, width, config.factor_width)

    def encode_language(self, embeddings: Tensor, mask: Tensor, *, positions: Tensor | None = None) -> Tensor:
        if embeddings.ndim not in (2, 3) or embeddings.shape[-1] != self.config.language_width:
            raise ValueError("language embeddings must have shape [..., tokens, language_width]")
        if mask.shape != embeddings.shape[:-1] or not mask.bool().any(-1).all():
            raise ValueError("language mask must include at least one valid token")
        tokens = self.language_input(embeddings)
        if positions is None:
            positions = torch.arange(tokens.shape[-2], device=tokens.device)
        if positions.shape != tokens.shape[-2:-1]:
            raise ValueError("language positions must identify each input token")
        keys = tokens + position_encoding(positions, self.config.width, tokens.dtype)
        return self.language_read(self.language_query, keys, tokens, mask.bool()[..., None, None, :]).squeeze(-2)

    def contextual_language(self, visual_tokens: Tensor, task_mask: Tensor, language_mask: Tensor) -> Tensor:
        """Read each frame's exact native task span, with the same reader and text positions."""
        positions = language_mask.bool().nonzero().flatten()
        if (visual_tokens.ndim != 3 or task_mask.shape != visual_tokens.shape[:2]
                or not len(positions) or not (task_mask.bool().sum(-1) == len(positions)).all()):
            raise ValueError("each frame must retain exactly its contextual task tokens")
        tokens = visual_tokens[task_mask.bool()].reshape(len(visual_tokens), len(positions), -1)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return self.encode_language(tokens, mask, positions=positions)

    def encode_video(self, responses: Tensor, frame_indices: Tensor, language: Tensor,
                     visual_tokens: Tensor, visual_mask: Tensor) -> Tensor:
        """Encode a whole ordered video, preserving prefix dependence; return P4[T,d]."""
        expected = (self.config.horizon, self.config.native_width)
        if responses.ndim != 3 or responses.shape[1:] != expected or not len(responses):
            raise ValueError(f"native responses must have shape [T,{expected}], T > 0")
        if frame_indices.shape != responses.shape[:1] or not torch.isfinite(frame_indices).all():
            raise ValueError("one finite real frame index is required per response")
        if len(frame_indices) > 1 and not (frame_indices[1:] > frame_indices[:-1]).all():
            raise ValueError("video frame indices must be strictly increasing")
        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != len(responses) or visual_tokens.shape[-1] != self.config.language_width:
            raise ValueError("visual evidence must have shape [T,tokens,language_width]")
        if visual_mask.shape != visual_tokens.shape[:2] or not visual_mask.bool().any(-1).all():
            raise ValueError("every frame needs at least one valid visual evidence token")
        if language.shape != (len(responses), self.config.width):
            raise ValueError("process language must be a separate current-frame condition")
        states = self.input_projection(responses) + self.horizon_embedding
        times = frame_indices.to(device=states.device, dtype=torch.float32)
        for group in self.process_groups:
            args = (states, times, language, visual_tokens, visual_mask.bool(), self.horizon_embedding)
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                states, process = checkpoint(group, *args, use_reentrant=False)
            else:
                states, process = group(*args)
        return process

    def _memory(self, videos: Sequence[Tensor], frame_indices: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        memories, routes, priors = [], [], []
        for video, indices in zip(videos, frame_indices, strict=True):
            if video.ndim != 2 or video.shape[1] != self.config.width or not len(video):
                raise ValueError("compiled video memories must have shape [T,d], T > 0")
            if indices.shape != video.shape[:1]:
                raise ValueError("memory time indices do not match video length")
            time = position_encoding(indices.to(video.device) / 5, self.config.width, video.dtype)
            memories.append(video)
            routes.append(self.time_projection(time))
            priors.append(video.new_full((len(video),), -math.log(len(video))))
        return torch.cat(memories), torch.cat(routes), torch.cat(priors)[None, :]

    def compile(self, videos: Sequence[Tensor], frame_indices: Sequence[Tensor], language: Tensor) -> Tensor:
        if not videos or len(videos) != len(frame_indices):
            raise ValueError("a condition needs one or more videos with matching time arrays")
        memory, routing, prior = self._memory(videos, frame_indices)
        query = (self.target_queries[:, None, :] + self.rank_queries[None, :, :]).flatten(0, 1)
        language_route = self.query_language(language)
        if self.config.backend_conditioning != "all":
            language_route = torch.zeros_like(language_route)
        for index, block in enumerate(self.compiler):
            # Language guides the first lookup; task-conditioned residual
            # content arrives through the real video values.
            args = (query, memory, routing, prior, language_route if index == 0 else None)
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                query = checkpoint(block, *args, use_reentrant=False)
            else:
                query = block(*args)
        return query.unflatten(0, (len(self.contract.targets), self.contract.rank))

    def forward(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
                language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
                visual_task_masks: Sequence[Tensor]) -> dict[str, Tensor]:
        if not responses or not (len(responses) == len(frame_indices) == len(visual_tokens)
                                 == len(visual_masks) == len(visual_task_masks)):
            raise ValueError("a condition needs one or more response videos with matching times and visual evidence")
        language = self.encode_language(language_embeddings, language_mask)
        videos = []
        for response, indices, visual, mask, task_mask in zip(
                responses, frame_indices, visual_tokens, visual_masks, visual_task_masks, strict=True):
            if task_mask.shape != mask.shape or (task_mask.bool() & ~mask.bool()).any():
                raise ValueError("contextual task tokens must be valid visual evidence")
            contextual = self.contextual_language(visual, task_mask, language_mask)
            videos.append(self.encode_video(response, indices, contextual, visual, mask))
        return self.decoder(self.compile(videos, frame_indices, language))
