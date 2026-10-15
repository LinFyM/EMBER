"""Fresh image-only ordered video Reader into the fixed64 control coordinates."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ember.proposal_writer.model import attention, positions


class ImageReader(nn.Module):
    """Eight slots/frame, four ordered layers, then one global query and zero head."""
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(2048, 256)
        self.spatial = nn.Embedding(512, 256)
        self.frame_queries = nn.Parameter(torch.randn(8, 256) / 16)
        self.frame_attention = attention()
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0., batch_first=True, norm_first=True)
        self.video_time = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(256)
        self.global_query = nn.Parameter(torch.randn(1, 256) / 16)
        self.global_attention = attention()
        self.output_norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 64)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images, indices, *, checkpoint_layers=False):
        if (images.ndim != 3 or images.shape[1:] != (512, 2048) or not len(images)
                or indices.ndim != 1 or len(images) != len(indices)
                or indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Reader requires complete synchronized image-prefix frames and original integer indices')
        # Native BF16 features stay BF16 under autocast; CPU callers may use FP32 weights.
        if not torch.is_autocast_enabled(images.device.type):images = images.to(self.image.weight.dtype)
        memory = self.image(images) + self.spatial.weight
        query = self.frame_queries[None].expand(len(images), -1, -1)
        slots = query + self.frame_attention(query, memory, memory, need_weights=False)[0]
        ordered = (slots + positions(indices.to(images.device))[:, None]).flatten(0, 1)[None]
        if checkpoint_layers and torch.is_grad_enabled():
            for layer in self.video_time.layers:
                ordered = checkpoint(layer, ordered, use_reentrant=False, preserve_rng_state=False)
        else:
            ordered = self.video_time(ordered)
        ordered = self.norm(ordered)
        query = self.global_query[None]
        pooled = query + self.global_attention(query, ordered, ordered, need_weights=False)[0]
        return self.head(self.output_norm(pooled))[0, 0]
