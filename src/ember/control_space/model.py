"""Shared video/actual-history context and native functional pressure only."""
from __future__ import annotations

import torch
from torch import nn


class NativeOperatorContext(nn.Module):
    """One complete ordered M and selected actual H produce M local cotangents.

    The module receives no language, task identity, behavior code, source weight
    arrays, reference parameters or query labels. Frozen evidence can stay on
    CPU until this forward; every learned read and pressure parameter belongs to
    the single shared compiler.
    """
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.image = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 256))
            self.hidden = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256))
            self.spatial = nn.Embedding(512, 256)
            self.pre_post = nn.Embedding(2, 256)
            self.flow_call = nn.Embedding(2, 256)
            self.action_position = nn.Embedding(50, 256)
            self.facts = nn.Linear(63, 256)
            self.event_query = nn.Parameter(torch.randn(1, 256) / 16)
            self.event_attention = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0.)
            self.video_attention = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0.)
            layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.,
                                               batch_first=True, norm_first=True)
            self.history_time = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(256)
            self.pressure = nn.Sequential(nn.Linear(291, 256), nn.SiLU(), nn.Linear(256, 35))
            nn.init.normal_(self.pressure[-1].weight, mean=0., std=1e-3)
            nn.init.zeros_(self.pressure[-1].bias)

    def forward(self, M, images, hidden, facts, F0):
        count = len(images)
        if (not 1 <= count <= 16 or images.shape != (count, 2, 512, 2048)
                or hidden.shape != (count, 2, 50, 1024) or facts.shape != (count, 63)
                or F0.shape != (count, 5, 7)):
            raise ValueError('native context requires selected actual image/hidden/facts/F0 shapes')
        if (M.ndim != 3 or M.shape[0] != 1 or M.shape[2] != 256
                or M.shape[1] < 8 or M.shape[1] % 8):
            raise ValueError('native context requires the complete ordered [1,frames*8,256] memory')
        device, dtype = self.event_query.device, self.event_query.dtype
        M, images, hidden, facts, F0 = (v.to(device=device, dtype=dtype)
                                       for v in (M, images, hidden, facts, F0))
        pixels = (self.image(images) + self.spatial.weight[None, None]
                  + self.pre_post.weight[None, :, None])
        actual = (self.hidden(hidden) + self.action_position.weight[None, None]
                  + self.flow_call.weight[None, :, None])
        tokens = torch.cat((pixels.flatten(1, 2), actual.flatten(1, 2)), 1)
        query = self.event_query[None] + self.facts(facts)[:, None]
        events = (query + self.event_attention(query, tokens, tokens, need_weights=False)[0])[:, 0]
        events = events[None] + self.video_attention(events[None], M, M, need_weights=False)[0]
        context = self.norm(self.history_time(events))[0]
        return self.pressure(torch.cat((context, F0.flatten(1)), -1)).reshape(count, 5, 7).float()
