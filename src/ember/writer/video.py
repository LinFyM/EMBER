"""Task-grounded semantic paths compiled once into complete conditional LoRAs."""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LoRAContract
from ember.writer.attention import Attention, RotaryBlock, feed_forward
from ember.writer.factor import FactorLoRADecoder


SCHEMA = "video_conditioned_writer_v6"
ARCHITECTURE = "semantic_state_path_lora_v1"


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
    state_width: int = 32
    query_chunk: int = 32
    activation_checkpoint: bool = True
    process_mode: str = "ordered"
    schema: str = SCHEMA
    architecture: str = ARCHITECTURE

    def __post_init__(self) -> None:
        require_architecture_identity(vars(self))
        dimensions = (self.width, self.heads, self.horizon, self.native_width, self.language_width,
                      self.blocks, self.factor_width, self.state_width, self.query_chunk)
        if min(dimensions) <= 0 or self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("positive dimensions and even RoPE width per head are required")

    def to_dict(self) -> dict:
        return asdict(self)


def path_statistics(states: Tensor, process_mode: str) -> Tensor:
    """Second-order log-signature or matched full-set second moments, [L,d(d+1)/2]."""
    if states.ndim != 3 or min(states.shape) <= 0:
        raise ValueError("semantic states require nonempty [T,L,d]")
    with torch.autocast(states.device.type, enabled=False):
        values = states.float()
        dimension = values.shape[-1]
        if process_mode == "ordered":
            differences = values[1:] - values[:-1]
            preceding = differences.cumsum(0) - differences
            integral = torch.einsum("tli,tlj->lij", preceding, differences)
            area = .5 * (integral - integral.transpose(-1, -2))
            rows, columns = torch.triu_indices(dimension, dimension, 1, device=states.device)
            return torch.cat((values[-1] - values[0], area[:, rows, columns]), -1)
        if process_mode == "frame_set":
            moments = torch.einsum("tli,tlj->lij", values, values) / len(values)
            rows, columns = torch.triu_indices(dimension, dimension, device=states.device)
            return moments[:, rows, columns]
    raise ValueError("unknown semantic path aggregation")


class _VideoEncoder(nn.Module):
    """Read complete native evidence, then form one semantic path per language role."""

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
        self.state_norm, self.summary_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.state_projection = nn.Linear(width, config.state_width)
        self.summary_read = Attention(width, config.heads)

    def _ground(self, visual: Tensor, valid: Tensor, task_mask: Tensor, count: int) -> Tensor:
        image_mask = valid & ~task_mask
        if not image_mask.any(-1).all():
            raise ValueError("every frame requires real valid image patches")
        projected = self.visual_projection(visual)
        tasks = projected[task_mask].reshape(len(visual), count, self.config.width)
        return tasks + self.patch_read(self.task_norm(tasks), self.patch_norm(projected), projected,
                                       image_mask[:, None, None, :])

    def _native_read(self, grounded: Tensor, context: Tensor, response: Tensor) -> Tensor:
        values = self.horizon_norm(self.native_projection(response))
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
                language: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        grounded = self._ground(visual, valid, task_mask, len(language))
        query = self.context_query_norm(self.visual_projection(language))[None]
        memory = self.context_norm(grounded.flatten(0, 1))[None]
        context = self.context_read(query, memory, memory)[0]
        content = self._native_read(grounded, context, response)
        # Both models use the same full, bidirectional, permutation-equivariant
        # state interpretation. Only the final path/set statistic uses order.
        no_time = content.new_zeros(len(content))
        for language_block, temporal in zip(self.language_blocks, self.temporal_blocks, strict=True):
            content = language_block(content, positions)
            content = temporal(content.transpose(0, 1), no_time).transpose(0, 1)
        states = self.state_projection(self.state_norm(content)).tanh()
        memory = self.summary_norm(content.flatten(0, 1))[None]
        summary = self.summary_read(query, memory, memory)[0]
        return summary, path_statistics(states, self.config.process_mode)


class VideoConditionedWriter(nn.Module):
    """No source-policy owner, privileged teacher fields, or execution-time state."""

    def __init__(self, contract: LoRAContract, config: VideoWriterConfig = VideoWriterConfig()) -> None:
        super().__init__()
        self.contract, self.config = contract, config
        width, state_width = config.width, config.state_width
        self.encoder = _VideoEncoder(config)
        statistics = state_width * (state_width + 1) // 2
        self.process_norm = nn.LayerNorm(statistics)
        self.process_projection = nn.Linear(statistics, width)
        self.process_gain, self.process_value = nn.Linear(width, width), nn.Linear(width, width)
        self.target_queries = nn.Parameter(torch.randn(len(contract.targets), width) * .02)
        self.rank_queries = nn.Parameter(torch.randn(contract.rank, width) * .02)
        self.semantic_norm, self.query_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.parameter_read = Attention(width, config.heads)
        self.code_norm, self.code_ffn = nn.LayerNorm(width), feed_forward(width)
        self.decoder = FactorLoRADecoder(contract, width, config.factor_width)

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
        if not responses or not (len(responses) == len(frame_indices) == len(visual_tokens)
                                 == len(visual_masks) == len(visual_task_masks)):
            raise ValueError("every video requires matching native, visual, and frame evidence")
        if (language_embeddings.ndim != 2 or language_embeddings.shape[-1] != self.config.language_width
                or language_mask.shape != language_embeddings.shape[:1] or not language_mask.bool().any()):
            raise ValueError("exact language requires nonempty valid native embeddings")
        positions = language_mask.bool().nonzero().flatten()
        language = language_embeddings[language_mask.bool()]
        videos = []
        for response, indices, visual, valid, task_mask in zip(
                responses, frame_indices, visual_tokens, visual_masks, visual_task_masks, strict=True):
            valid, task_mask = valid.bool(), task_mask.bool()
            self._validate_video(response, indices, visual, valid, task_mask, len(positions))
            args = (response, visual, valid, task_mask, language, positions.to(response.device))
            if self.config.activation_checkpoint and torch.is_grad_enabled():
                videos.append(checkpoint(self.encoder, *args, use_reentrant=False))
            else:
                videos.append(self.encoder(*args))
        return tuple(videos)

    def decode(self, videos: Sequence[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
        if not videos:
            raise ValueError("parameter compilation requires actual video knowledge")
        semantics, contents, priors = [], [], []
        statistics = self.config.state_width * (self.config.state_width + 1) // 2
        for semantic, path in videos:
            if (semantic.ndim != 2 or semantic.shape[0] <= 0 or semantic.shape[1] != self.config.width
                    or path.shape != (len(semantic), statistics)):
                raise ValueError("each video needs matched semantic roles and path statistics")
            semantic = self.semantic_norm(semantic)
            process = self.process_projection(self.process_norm(path))
            content = semantic * (1 + self.process_gain(process).tanh()) + self.process_value(process)
            semantics.append(semantic)
            contents.append(content)
            priors.append(semantic.new_full((len(semantic),), -math.log(len(semantic))))
        query = (self.target_queries[:, None] + self.rank_queries[None]).flatten(0, 1)
        # Learned addresses only enter Q; the actual parameter Value comes from
        # paired semantic conditions and their observed path.
        code = self.parameter_read(self.query_norm(query)[None], torch.cat(semantics)[None],
                                   torch.cat(contents)[None], torch.cat(priors)[None])[0]
        code = code + self.code_ffn(self.code_norm(code))
        return self.decoder(code.unflatten(0, (len(self.contract.targets), self.contract.rank)))

    def forward(self, responses: Sequence[Tensor], frame_indices: Sequence[Tensor], language_embeddings: Tensor,
                language_mask: Tensor, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
                visual_task_masks: Sequence[Tensor]) -> dict[str, Tensor]:
        return self.decode(self.encode(responses, frame_indices, language_embeddings, language_mask,
                                       visual_tokens, visual_masks, visual_task_masks))
