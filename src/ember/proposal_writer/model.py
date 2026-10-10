"""Full-coordinate hierarchical flow fields and independent decision Readers."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


def positions(index, width=256):
    frequency = torch.exp(torch.arange(0, width, 2, device=index.device).float() * (-math.log(10000) / width))
    angle = index.float()[..., None] * frequency
    return torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)


def attention():
    return nn.MultiheadAttention(256, 4, batch_first=True, dropout=0.)


class FactorLayout(nn.Module):
    """Original A rows/B columns, including masks; no gauge/rank reordering."""
    def __init__(self, contract, initial, scales=None):
        super().__init__()
        self.parts, self.names, self.factor_rank = [], [], contract.rank
        target, rank, side, coordinate, valid, scale_blocks = [], [], [], [], [], []
        cursor = 0
        for i, item in enumerate(contract.targets):
            for j, suffix in enumerate((LORA_A_SUFFIX, LORA_B_SUFFIX)):
                name = item.name + suffix
                width = item.in_features if j == 0 else item.out_features
                blocks = math.ceil(width / 64)
                self.parts.append((name, j, width, blocks, cursor, cursor + contract.rank * blocks))
                self.names.append(name)
                mask = (torch.arange(blocks * 64).reshape(blocks, 64) < width).repeat(contract.rank, 1)
                valid.append(mask)
                target.append(torch.full((contract.rank * blocks,), i))
                rank.append(torch.arange(contract.rank).repeat_interleave(blocks))
                side.append(torch.full((contract.rank * blocks,), j))
                coordinate.append(torch.arange(blocks).repeat(contract.rank))
                scale_blocks.append(torch.full((contract.rank * blocks, 1), 1. if scales is None else scales[name]))
                cursor += contract.rank * blocks
        for name, chunks in [('target', target), ('rank', rank), ('side', side), ('coordinate', coordinate),
                             ('valid', valid), ('scale', scale_blocks)]:
            self.register_buffer(name, torch.cat(chunks))
        self.register_buffer('group', self.target * contract.rank + self.rank)
        self.count = int(self.valid.sum())
        if (cursor, self.count, int(self.valid.numel() - self.count)) != (10064, 643584, 512):
            raise ValueError('full rank8 block layout differs from registered coordinates')
        self.register_buffer('center', self.pack(initial))

    def pack(self, state):
        blocks = []
        if set(state) != set(self.names):
            raise ValueError('complete task state missing/extra factors')
        for name, side, width, n, _, _ in self.parts:
            value = state[name].float()
            value = value if side == 0 else value.T
            if value.shape != (self.factor_rank, width):
                raise ValueError('factor shape differs from original coordinates')
            blocks.append(torch.nn.functional.pad(value, (0, n * 64 - width)).reshape(-1, 64))
        return torch.cat(blocks)

    def unpack(self, blocks):
        if blocks.shape != self.valid.shape:
            raise ValueError('complete coordinate output has wrong shape')
        result = {}
        for name, side, width, n, start, stop in self.parts:
            value = blocks[start:stop].reshape(self.factor_rank, n * 64)[:, :width]
            result[name] = (value if side == 0 else value.T).contiguous()
        return result


class BlockLayer(nn.Module):
    """Bidirectional block→rank→target→rank→block communication."""
    def __init__(self):
        super().__init__()
        self.local_norm, self.rank_norm, self.target_norm = (nn.LayerNorm(256) for _ in range(3))
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
        self.rank_query = nn.Parameter(torch.randn(256) / 16)
        self.target_query = nn.Parameter(torch.randn(256) / 16)
        self.rank_attention, self.target_attention, self.memory_attention = (attention() for _ in range(3))
        self.broadcast = nn.Linear(512, 256)

    def forward(self, blocks, ranks, targets, group, memory):
        blocks = blocks + self.ff(self.local_norm(blocks))
        # Query-weighted pooling within each actual target/rank (A and B together).
        scores = (self.local_norm(blocks) * self.rank_query).sum(-1).float() / 16
        maxima = scores.new_full((ranks.numel() // 256,), -torch.inf).scatter_reduce_(0, group, scores, reduce='amax', include_self=True)
        weights = (scores - maxima[group]).exp()
        sums = weights.new_zeros(ranks.numel() // 256).index_add_(0, group, weights)
        pooled = blocks.new_zeros((ranks.numel() // 256, 256)).index_add_(0, group, blocks * (weights / sums[group])[:, None].to(blocks))
        ranks = ranks + pooled.reshape_as(ranks)
        q = self.rank_norm(ranks)
        ranks = ranks + self.rank_attention(q, q, q, need_weights=False)[0]
        score = (self.rank_norm(ranks) * self.target_query).sum(-1).float() / 16
        targets = targets + (score.softmax(-1)[..., None].to(ranks) * ranks).sum(1)
        q = self.target_norm(targets)[None]
        targets = targets + self.target_attention(q, q, q, need_weights=False)[0][0]
        if memory.numel():
            targets = targets + self.memory_attention(self.target_norm(targets)[None], memory[None],
                                                      memory[None], need_weights=False)[0][0]
        ranks = ranks + targets[:, None]
        context = torch.cat((ranks.flatten(0, 1)[group], targets[group // ranks.shape[1]]), -1)
        return blocks + self.broadcast(context), ranks, targets


class FactorEncoder(nn.Module):
    """Two-layer static factor summaries without H/video/flow dependence."""
    def __init__(self, layout, layers=2, *, flow=False):
        super().__init__()
        object.__setattr__(self, 'layout', layout)
        self.flow = flow
        self.input = nn.Linear(192 if flow else 128, 256)
        self.target_address, self.rank_address = nn.Embedding(38, 256), nn.Embedding(layout.factor_rank, 256)
        self.side_address, self.block_address = nn.Embedding(2, 256), nn.Embedding(32, 256)
        self.layers = nn.ModuleList(BlockLayer() for _ in range(layers))
        self.norm = nn.LayerNorm(256)
        if flow:
            self.output = nn.Linear(256, 64)

    def forward(self, parent, *, x=None, time=None, memory=None, checkpoint_layers=True):
        layout = self.layout
        valid = layout.valid.to(parent)
        fields = (parent, valid) if x is None else (x, parent, valid)
        blocks = self.input(torch.cat(fields, -1))
        blocks = blocks + self.target_address(layout.target) + self.rank_address(layout.rank)
        blocks = blocks + self.side_address(layout.side) + self.block_address(layout.coordinate)
        if time is not None:
            blocks = blocks + positions(torch.as_tensor(time, device=blocks.device))
        ranks = self.rank_address.weight[None].expand(38, -1, -1)
        targets = self.target_address.weight
        memory = blocks.new_empty(0, 256) if memory is None else memory
        for layer in self.layers:
            args = (blocks, ranks, targets, layout.group, memory)
            blocks, ranks, targets = (checkpoint(layer, *args, use_reentrant=False, preserve_rng_state=False)
                                      if checkpoint_layers and torch.is_grad_enabled() else layer(*args))
        return self.output(self.norm(blocks)) * valid if self.flow else self.norm(targets)


class ContextReader(nn.Module):
    """Learned ordered teaching and actual-history read; no ID feature route."""
    def __init__(self):
        super().__init__()
        self.z, self.h, self.language = nn.Linear(2048, 256), nn.Linear(1024, 256), nn.Linear(2048, 256)
        self.spatial = nn.Embedding(512, 256)
        self.pre_post = nn.Embedding(2, 256)
        self.flow_call = nn.Embedding(2, 256)
        self.action_position = nn.Embedding(50, 256)
        self.frame_queries = nn.Parameter(torch.randn(8, 256) / 16)
        self.frame_attention, self.event_attention, self.correspondence = (attention() for _ in range(3))
        layer = lambda: nn.TransformerEncoderLayer(256, 4, 1024, dropout=0., batch_first=True, norm_first=True)
        self.video_time = nn.TransformerEncoder(layer(), 4, enable_nested_tensor=False)
        self.history_time = nn.TransformerEncoder(layer(), 4, enable_nested_tensor=False)
        self.facts = nn.Linear(16 + 35 + 5 + 5 + 2, 256)
        self.event_query, self.decision_query, self.empty = (nn.Parameter(torch.randn(1, 256) / 16) for _ in range(3))
        self.norm = nn.LayerNorm(256)

    def teaching(self, features):
        z, h, language = features['z'], features['h'], features['language']
        tokens = self.language(language[features['language_mask']].float())
        memory = torch.cat((self.z(z.float()) + self.spatial.weight,
                            self.h(h.float()), tokens[None].expand(len(z), -1, -1)), 1)
        q = self.frame_queries[None].expand(len(z), -1, -1)
        slots = q + self.frame_attention(q, memory, memory, need_weights=False)[0]
        slots = slots + positions(features['indices'])[:, None]
        ordered = self.video_time(slots.flatten(0, 1)[None])[0]
        tokens = tokens + self.correspondence(tokens[None], ordered[None], ordered[None], need_weights=False)[0][0]
        return torch.cat((ordered, tokens))

    def forward(self, features, history, parameter_summaries):
        video = self.teaching(features)
        events = []
        for i, row in enumerate(history):
            images = row['images'].to(video.device).float()  # frozen raw image embeddings, pre/post
            actual_hidden = row['hidden'].to(video.device).float()
            p = parameter_summaries[row['parameter_ref']]
            pixels = self.z(images) + self.spatial.weight[None] + self.pre_post.weight[:, None]
            hidden = self.h(actual_hidden) + self.action_position.weight[None] + self.flow_call.weight[:, None]
            memory = torch.cat((pixels.flatten(0, 1), hidden.flatten(0, 1), p))
            facts = torch.cat((row['proprio'].flatten(), row['actions'].flatten(), row['executed'].float(),
                               row['feedback'], torch.tensor([row['episode'], row['step']], dtype=torch.float32)))
            q = self.event_query + self.facts(facts.to(video.device))[None] + positions(torch.tensor(i, device=video.device))
            event = q + self.event_attention(q[None], memory[None], memory[None], need_weights=False)[0][0]
            event = event + self.correspondence(event[None], video[None], video[None], need_weights=False)[0][0]
            events.append(event)
        history_memory = self.empty if not events else self.history_time(torch.cat(events)[None])[0]
        q = self.decision_query
        decision = q + self.correspondence(q[None], video[None], video[None], need_weights=False)[0][0]
        return self.norm(torch.cat((video, history_memory, decision)))


class Generator(nn.Module):
    """ψ owns meta, independent context and full-coordinate velocity field."""
    def __init__(self, layout, meta):
        super().__init__()
        self.layout, self.meta = layout, meta
        self.reader, self.static = ContextReader(), FactorEncoder(layout)
        self.field = FactorEncoder(layout, 6, flow=True)

    def context(self, features, history, states):
        summaries = {key: self.static((self.layout.pack(value).to(self.layout.scale.device) - self.layout.center) / self.layout.scale)
                     for key, value in states.items()}
        return self.reader(features, history, summaries)

    def forward(self, x, time, parent, features, history, states):
        memory = self.context(features, history, states)
        return self.field((parent - self.layout.center) / self.layout.scale, x=x, time=time, memory=memory)

    @torch.no_grad()
    def generate(self, parent, features, history, states, *, noise_seed):
        packed = self.layout.pack(parent).to(self.layout.scale.device)
        xi = torch.randn(self.layout.valid.shape, generator=torch.Generator().manual_seed(noise_seed))
        x = xi.to(packed.device) * self.layout.valid
        memory = self.context(features, history, states)
        for step in range(16):
            x = x + self.field((packed - self.layout.center) / self.layout.scale, x=x, time=step / 16, memory=memory,
                               checkpoint_layers=False).float() / 16
        return self.layout.unpack(packed + self.layout.scale * x)


class Actor(nn.Module):
    """θ owns every downstream Reader and all three categorical heads."""
    def __init__(self, layout):
        super().__init__()
        object.__setattr__(self, 'layout', layout)
        self.static, self.reader = FactorEncoder(layout), ContextReader()
        self.candidate_query, self.stop_query = (nn.Parameter(torch.randn(1, 256) / 16) for _ in range(2))
        self.candidate_attention, self.response_attention, self.set_attention = (attention() for _ in range(3))
        self.responses = nn.Linear(70, 256)  # absolute prediction, optional parent-relative difference
        self.budget = nn.Linear(4, 256)
        self.practiced = nn.Embedding(2, 256)
        self.types = nn.Embedding(3, 256)
        self.norm = nn.LayerNorm(256)
        self.heads = nn.ModuleList(nn.Linear(256, 1) for _ in range(3))

    def forward(self, kind, features, history, states, candidates, responses, budget):
        type_id = {'parent': 0, 'practice': 1, 'final': 2}[kind]
        summaries = {key: self.static((self.layout.pack(value).to(self.layout.scale.device) - self.layout.center) / self.layout.scale)
                     for key, value in states.items()}
        memory = self.reader(features, history, summaries)
        context = self.budget(torch.as_tensor(budget, device=memory.device).float()) + self.types.weight[type_id]
        tokens = []
        for candidate in candidates:
            q = self.stop_query if candidate == 'STOP' else self.candidate_query
            q = q + context
            if candidate != 'STOP':
                practiced = candidate in {row['parameter_ref'] for row in history} or candidate == 'MT300'
                q = q + self.practiced.weight[int(practiced)]
            p = memory if candidate == 'STOP' else torch.cat((memory, summaries[candidate]))
            q = q + self.candidate_attention(q[None], p[None], p[None], need_weights=False)[0][0]
            if candidate in responses:
                response = self.responses(responses[candidate].to(memory.device).float())
                response = response + positions(torch.arange(len(response), device=memory.device))
                response = response + self.reader.correspondence(response[None], memory[None], memory[None],
                                                                  need_weights=False)[0][0]
                q = q + self.response_attention(q[None], response[None], response[None], need_weights=False)[0][0]
            tokens.append(q)
        values = torch.cat(tokens)[None]
        values = values + self.set_attention(values, values, values, need_weights=False)[0]
        return self.heads[type_id](self.norm(values[0])).squeeze(-1)
