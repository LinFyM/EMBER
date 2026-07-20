"""Variable-length language/video task encoder for the direct LoRA Writer."""

from __future__ import annotations

import math

import torch


class VariableEpisodeInputError(ValueError):
    """Raised when a language/video task input is structurally invalid."""


def _sinusoidal_positions(
    start: int, length: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return absolute positions without a learned maximum sequence length."""

    positions = torch.arange(start, start + length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 2))
    )[None]
    encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class VariableEpisodeTaskEncoder(torch.nn.Module):
    """Encode full language and any finite set of variable-length videos.

    Frames are never temporally averaged or truncated. Long episodes are
    hierarchically resampled in bounded-size chunks, then all episode memories
    are aggregated as an unordered demonstration set. The same parameters and
    interface therefore accept one short episode or many long episodes.
    """

    def __init__(
        self,
        *,
        vision_feature_dim: int,
        language_feature_dim: int,
        hidden_dim: int,
        attention_heads: int,
        temporal_chunk_size: int,
        chunk_memory_tokens: int,
        episode_memory_tokens: int,
        task_memory_tokens: int,
    ) -> None:
        super().__init__()
        values = (
            vision_feature_dim,
            language_feature_dim,
            hidden_dim,
            attention_heads,
            temporal_chunk_size,
            chunk_memory_tokens,
            episode_memory_tokens,
            task_memory_tokens,
        )
        if any(value <= 0 for value in values) or hidden_dim % attention_heads:
            raise VariableEpisodeInputError("invalid variable-video Writer dimensions")
        self.vision_feature_dim = vision_feature_dim
        self.language_feature_dim = language_feature_dim
        self.hidden_dim = hidden_dim
        self.temporal_chunk_size = temporal_chunk_size

        self.vision_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(vision_feature_dim),
            torch.nn.Linear(vision_feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.language_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(language_feature_dim),
            torch.nn.Linear(language_feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.chunk_queries = torch.nn.Parameter(torch.randn(chunk_memory_tokens, hidden_dim) * 0.02)
        self.episode_queries = torch.nn.Parameter(
            torch.randn(episode_memory_tokens, hidden_dim) * 0.02
        )
        self.task_queries = torch.nn.Parameter(torch.randn(task_memory_tokens, hidden_dim) * 0.02)
        self.language_modality = torch.nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.video_modality = torch.nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.chunk_attention = torch.nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True, dropout=0.0
        )
        self.episode_attention = torch.nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True, dropout=0.0
        )
        self.task_attention = torch.nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True, dropout=0.0
        )
        self.chunk_norm = torch.nn.LayerNorm(hidden_dim)
        self.episode_norm = torch.nn.LayerNorm(hidden_dim)
        self.task_norm = torch.nn.LayerNorm(hidden_dim)
        self.chunk_ffn = self._ffn(hidden_dim)
        self.episode_ffn = self._ffn(hidden_dim)
        self.task_ffn = self._ffn(hidden_dim)

    @staticmethod
    def _ffn(width: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width * 4),
            torch.nn.GELU(),
            torch.nn.Linear(width * 4, width),
        )

    @staticmethod
    def _cross_attention(
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention: torch.nn.MultiheadAttention,
        norm: torch.nn.LayerNorm,
        ffn: torch.nn.Module,
    ) -> torch.Tensor:
        attended, _ = attention(
            norm(queries)[None], norm(memory)[None], norm(memory)[None], need_weights=False
        )
        result = queries + attended[0]
        return result + ffn(result)

    def _encode_episode(self, frames: torch.Tensor) -> torch.Tensor:
        if (
            frames.ndim != 2
            or frames.shape[0] < 1
            or frames.shape[1] != self.vision_feature_dim
        ):
            raise VariableEpisodeInputError("Writer episode feature tensor has wrong shape")
        chunk_memories: list[torch.Tensor] = []
        for start in range(0, frames.shape[0], self.temporal_chunk_size):
            stop = min(start + self.temporal_chunk_size, frames.shape[0])
            tokens = self.vision_projection(frames[start:stop].to(torch.float32))
            tokens = tokens + _sinusoidal_positions(
                start,
                stop - start,
                self.hidden_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            chunk_memories.append(
                self._cross_attention(
                    self.chunk_queries,
                    tokens,
                    self.chunk_attention,
                    self.chunk_norm,
                    self.chunk_ffn,
                )
            )
        chunks = torch.cat(chunk_memories, dim=0)
        return self._cross_attention(
            self.episode_queries,
            chunks,
            self.episode_attention,
            self.episode_norm,
            self.episode_ffn,
        )

    def forward(
        self,
        language_tokens: torch.Tensor,
        video_features: torch.Tensor,
        episode_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if (
            language_tokens.ndim != 2
            or language_tokens.shape[0] < 1
            or language_tokens.shape[1] != self.language_feature_dim
            or video_features.ndim != 2
            or video_features.shape[1] != self.vision_feature_dim
            or episode_offsets.ndim != 1
            or episode_offsets.numel() < 2
        ):
            raise VariableEpisodeInputError("Writer language/video task input has wrong shape")
        offsets = episode_offsets.detach().to(device="cpu", dtype=torch.int64).tolist()
        if offsets[0] != 0 or offsets[-1] != video_features.shape[0] or any(
            right <= left for left, right in zip(offsets, offsets[1:])
        ):
            raise VariableEpisodeInputError("Writer episode offsets are invalid")
        episode_memory = torch.cat(
            [self._encode_episode(video_features[left:right]) for left, right in zip(offsets, offsets[1:])],
            dim=0,
        )
        language_memory = self.language_projection(language_tokens.to(torch.float32))
        memory = torch.cat(
            (
                language_memory + self.language_modality,
                episode_memory + self.video_modality,
            ),
            dim=0,
        )
        return self._cross_attention(
            self.task_queries,
            memory,
            self.task_attention,
            self.task_norm,
            self.task_ffn,
        )
