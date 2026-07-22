"""Condition-only language/video memory for the complete-LoRA Writer."""

from __future__ import annotations

import math

import torch


class VariableEpisodeInputError(ValueError):
    """Raised when a batched one-video Writer input is structurally invalid."""


def _sinusoidal_positions(
    start: int, length: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    positions = torch.arange(start, start + length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 2))
    )[None]
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(
        positions * frequencies[:, : encoding[:, 1::2].shape[1]]
    )
    return encoding.to(dtype=dtype)


def _validated_offsets(
    offsets: torch.Tensor, total: int, *, label: str
) -> tuple[int, ...]:
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise VariableEpisodeInputError(f"Writer {label} offsets are invalid")
    values = tuple(
        int(value)
        for value in offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if (
        values[0] != 0
        or values[-1] != total
        or any(right <= left for left, right in zip(values, values[1:]))
    ):
        raise VariableEpisodeInputError(f"Writer {label} offsets are invalid")
    return values


class VariableEpisodeTaskEncoder(torch.nn.Module):
    """Encode a batch where every condition contains one language and one video.

    PI05 spatial tokens are preserved inside each frame. Learned queries only
    address conditional memories; they are never added to the returned values,
    so there is no query-only route to a task-independent adapter.
    """

    def __init__(
        self,
        *,
        vision_feature_dim: int,
        vision_spatial_tokens: int,
        language_feature_dim: int,
        hidden_dim: int,
        attention_heads: int,
        temporal_chunk_size: int,
        chunk_memory_tokens: int,
        episode_memory_tokens: int,
        language_memory_tokens: int,
        task_memory_tokens: int,
    ) -> None:
        super().__init__()
        values = (
            vision_feature_dim,
            vision_spatial_tokens,
            language_feature_dim,
            hidden_dim,
            attention_heads,
            temporal_chunk_size,
            chunk_memory_tokens,
            episode_memory_tokens,
            language_memory_tokens,
            task_memory_tokens,
        )
        if any(value <= 0 for value in values) or hidden_dim % attention_heads:
            raise VariableEpisodeInputError("invalid variable-video Writer dimensions")
        self.vision_feature_dim = vision_feature_dim
        self.vision_spatial_tokens = vision_spatial_tokens
        self.language_feature_dim = language_feature_dim
        self.hidden_dim = hidden_dim
        self.temporal_chunk_size = temporal_chunk_size

        self.vision_projection = self._projection(vision_feature_dim, hidden_dim)
        self.language_projection = self._projection(language_feature_dim, hidden_dim)
        self.chunk_queries = self._queries(chunk_memory_tokens, hidden_dim)
        self.episode_queries = self._queries(episode_memory_tokens, hidden_dim)
        self.language_queries = self._queries(language_memory_tokens, hidden_dim)
        self.task_queries = self._queries(task_memory_tokens, hidden_dim)

        self.chunk_attention = self._attention(hidden_dim, attention_heads)
        self.episode_attention = self._attention(hidden_dim, attention_heads)
        self.language_attention = self._attention(hidden_dim, attention_heads)
        self.task_attention = self._attention(hidden_dim, attention_heads)
        self.chunk_norm = torch.nn.LayerNorm(hidden_dim)
        self.episode_norm = torch.nn.LayerNorm(hidden_dim)
        self.language_norm = torch.nn.LayerNorm(hidden_dim)
        self.task_norm = torch.nn.LayerNorm(hidden_dim)
        self.chunk_ffn = self._ffn(hidden_dim)
        self.episode_ffn = self._ffn(hidden_dim)
        self.language_ffn = self._ffn(hidden_dim)
        self.task_ffn = self._ffn(hidden_dim)

    @staticmethod
    def _projection(input_dim: int, hidden_dim: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _queries(count: int, hidden_dim: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.randn(count, hidden_dim) * 0.02)

    @staticmethod
    def _attention(hidden_dim: int, heads: int) -> torch.nn.MultiheadAttention:
        return torch.nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True, dropout=0.0
        )

    @staticmethod
    def _ffn(width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width * 4),
            torch.nn.GELU(),
            torch.nn.Linear(width * 4, width),
        )

    @staticmethod
    def _conditional_attention(
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention: torch.nn.MultiheadAttention,
        norm: torch.nn.LayerNorm,
        ffn: torch.nn.Module,
    ) -> torch.Tensor:
        attended, _ = attention(
            norm(queries)[None], norm(memory)[None], norm(memory)[None], need_weights=False
        )
        result = attended[0]
        return result + ffn(result)

    def _encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        if (
            frames.ndim != 3
            or frames.shape[0] < 1
            or frames.shape[1] != self.vision_spatial_tokens
            or frames.shape[2] != self.vision_feature_dim
        ):
            raise VariableEpisodeInputError("Writer video feature tensor has wrong shape")
        memories: list[torch.Tensor] = []
        spatial = _sinusoidal_positions(
            0,
            self.vision_spatial_tokens,
            self.hidden_dim,
            device=frames.device,
            dtype=torch.float32,
        )
        for start in range(0, frames.shape[0], self.temporal_chunk_size):
            stop = min(start + self.temporal_chunk_size, frames.shape[0])
            tokens = self.vision_projection(frames[start:stop].to(torch.float32))
            temporal = _sinusoidal_positions(
                start,
                stop - start,
                self.hidden_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + temporal[:, None] + spatial.to(tokens)[None]
            memories.append(
                self._conditional_attention(
                    self.chunk_queries,
                    tokens.flatten(0, 1),
                    self.chunk_attention,
                    self.chunk_norm,
                    self.chunk_ffn,
                )
            )
        return self._conditional_attention(
            self.episode_queries,
            torch.cat(memories, dim=0),
            self.episode_attention,
            self.episode_norm,
            self.episode_ffn,
        )

    def _encode_language(self, tokens: torch.Tensor) -> torch.Tensor:
        if (
            tokens.ndim != 2
            or tokens.shape[0] < 1
            or tokens.shape[1] != self.language_feature_dim
        ):
            raise VariableEpisodeInputError("Writer language feature tensor has wrong shape")
        memory = self.language_projection(tokens.to(torch.float32))
        memory = memory + _sinusoidal_positions(
            0,
            memory.shape[0],
            self.hidden_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        return self._conditional_attention(
            self.language_queries,
            memory,
            self.language_attention,
            self.language_norm,
            self.language_ffn,
        )

    def forward(
        self,
        language_tokens: torch.Tensor,
        video_features: torch.Tensor,
        language_offsets: torch.Tensor,
        video_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if language_tokens.ndim != 2 or video_features.ndim != 3:
            raise VariableEpisodeInputError("Writer language/video task input has wrong shape")
        language = _validated_offsets(
            language_offsets, language_tokens.shape[0], label="language"
        )
        video = _validated_offsets(video_offsets, video_features.shape[0], label="video")
        if len(language) != len(video):
            raise VariableEpisodeInputError("Writer condition batch sizes differ")
        tasks = []
        for language_span, video_span in zip(
            zip(language, language[1:]), zip(video, video[1:]), strict=True
        ):
            left_l, right_l = language_span
            left_v, right_v = video_span
            memory = torch.cat(
                (
                    self._encode_language(language_tokens[left_l:right_l]),
                    self._encode_video(video_features[left_v:right_v]),
                ),
                dim=0,
            )
            tasks.append(
                self._conditional_attention(
                    self.task_queries,
                    memory,
                    self.task_attention,
                    self.task_norm,
                    self.task_ffn,
                )
            )
        return torch.stack(tasks, dim=0)
