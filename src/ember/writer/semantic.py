"""Task-token visual semantics and their explicit consumption of video changes."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ember.writer.attention import Attention, RotaryBlock, feed_forward, position_encoding


class VisualSemanticEncoder(nn.Module):
    """Per-token frame-set reading: video order never enters this branch."""

    def __init__(self, width: int, heads: int, native_width: int) -> None:
        super().__init__()
        self.width = width
        self.projection = nn.Linear(native_width, width, bias=False)
        self.task_norm, self.patch_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.patch_read = Attention(width, heads)
        self.set_query_norm, self.set_key_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.frame_read = Attention(width, heads)
        self.language_blocks = nn.ModuleList([RotaryBlock(width, heads, causal=False) for _ in range(2)])

    def frame_tokens(self, visual: Tensor, valid: Tensor, task_mask: Tensor, token_count: int) -> Tensor:
        if (visual.ndim != 3 or valid.shape != visual.shape[:2] or task_mask.shape != valid.shape
                or (task_mask & ~valid).any() or not (task_mask.sum(-1) == token_count).all()):
            raise ValueError("semantic reading needs the aligned exact native task-token span")
        image_mask = valid & ~task_mask
        if not image_mask.any(-1).all():
            raise ValueError("semantic reading requires real valid image evidence in every frame")
        projected = self.projection(visual)
        tasks = projected[task_mask].reshape(len(visual), token_count, self.width)
        grounded = self.patch_read(self.task_norm(tasks), self.patch_norm(projected), projected,
                                   image_mask[:, None, None, :])
        return tasks + grounded

    def forward(self, visual_tokens: Sequence[Tensor], visual_masks: Sequence[Tensor],
                task_masks: Sequence[Tensor], language_embeddings: Tensor, language_mask: Tensor) -> Tensor:
        positions = language_mask.bool().nonzero().flatten()
        if not len(positions) or len(visual_tokens) != len(visual_masks) or len(visual_tokens) != len(task_masks):
            raise ValueError("semantic video collection and language span differ")
        contents, priors = [], []
        for visual, valid, task_mask in zip(visual_tokens, visual_masks, task_masks, strict=True):
            contents.append(self.frame_tokens(visual, valid.bool(), task_mask.bool(), len(positions)))
            priors.append(visual.new_full((len(visual),), -math.log(len(visual))))
        # L is a batch axis: each exact task token reads its matching evidence
        # over every frame, before tokens interact along the language axis.
        memory = torch.cat(contents).transpose(0, 1)
        query = self.projection(language_embeddings[language_mask.bool()])[:, None, :]
        prior = torch.cat(priors)[None, :]
        semantic = self.frame_read(self.set_query_norm(query), self.set_key_norm(memory), memory, prior).squeeze(1)
        for block in self.language_blocks:
            semantic = block(semantic, positions)
        return semantic


class SemanticProcessCompiler(nn.Module):
    """A semantic slot consumes video changes and produces one policy code."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.identity_norm, self.semantic_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.semantic_read = Attention(width, heads)
        self.identity_query, self.semantic_query = nn.Linear(width, width), nn.Linear(width, width)
        self.process_key_norm = nn.LayerNorm(width)
        self.time_projection = nn.Linear(width, width, bias=False)
        self.process_read = Attention(width, heads)
        # Centered zero content must stay zero through the Value/output path.
        self.process_read.value = nn.Linear(width, width, bias=False)
        self.process_read.output = nn.Linear(width, width, bias=False)
        self.process_norm = nn.LayerNorm(width)
        self.fusion = feed_forward(width, 2 * width)
        self.self_norm, self.ffn_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attention, self.ffn = Attention(width, heads), feed_forward(width)

    def process_memory(self, videos: Sequence[Tensor], indices: Sequence[Tensor], *, ordered: bool):
        keys, values, priors = [], [], []
        for content, times in zip(videos, indices, strict=True):
            if content.ndim != 2 or not len(content) or times.shape != content.shape[:1]:
                raise ValueError("process memory needs one index for each real frame")
            key = self.process_key_norm(content)
            if ordered:
                key = key + self.time_projection(position_encoding(times.to(content.device) / 5,
                                                                   self.width, content.dtype))
            keys.append(key)
            values.append(content - content.mean(0, keepdim=True))
            priors.append(content.new_full((len(content),), -math.log(len(content))))
        return torch.cat(keys), torch.cat(values), torch.cat(priors)[None, :]

    def forward(self, identities: Tensor, semantic: Tensor, videos: Sequence[Tensor],
                indices: Sequence[Tensor], *, ordered: bool) -> Tensor:
        identity = self.identity_norm(identities)
        slots = self.semantic_read(identity, self.semantic_norm(semantic), semantic)
        query = self.identity_query(identity) + self.semantic_query(self.semantic_norm(slots))
        key, value, prior = self.process_memory(videos, indices, ordered=ordered)
        changes = self.process_read(query, key, value, prior)
        code = self.fusion(torch.cat((self.semantic_norm(slots), self.process_norm(changes)), -1))
        normalized = self.self_norm(code)
        code = code + self.self_attention(normalized, normalized, normalized)
        return code + self.ffn(self.ffn_norm(code))
