"""RGB physical prediction, shared relation encoding, and native-horizon read.

Only predicted physical fields connect Phi to Omega in deployment. The optional
frame-valid field belongs to F's separate GT encoder and never to G predictions.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .attention import (HEADS, HEAD_WIDTH, WIDTH, CrossAttentionBlock, TemporalBlock,
                        check_frames, feed_forward, merge_heads, presence_attention,
                        split_heads)

ENTITIES = 33
PHYSICAL_SHAPES = {"p": (ENTITIES, 3), "R": (ENTITIES, 3, 3),
                   "semantic": (ENTITIES, 64), "presence": (ENTITIES,),
                   "joint_type": (ENTITIES, 3), "joint_q": (ENTITIES, 2),
                   "hand_q": (2,)}
NODE_FIELDS = 3 + 64 + 3 + 2 + 2 + 1
EDGE_FIELDS = 3 + 9 + 2 * NODE_FIELDS


def check_physical(physical: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
    prefix = physical["p"].shape[:-2]
    for name, suffix in PHYSICAL_SHAPES.items():
        if physical[name].shape != (*prefix, *suffix):
            raise ValueError(f"physical field {name} must have shape {(*prefix, *suffix)}")
    return prefix


def rotation_6d(value: torch.Tensor) -> torch.Tensor:
    first, second = value.float().split(3, dim=-1)
    first = F.normalize(first, dim=-1, eps=1e-6)
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-6)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), -1)


class VisualRelations(nn.Module):
    """Phi: full native H to one hand and 32 anonymous physical trajectories."""
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(1024, WIDTH)
        self.queries = nn.Parameter(torch.randn(ENTITIES, WIDTH) * 0.02)
        self.cross = nn.ModuleList(CrossAttentionBlock() for _ in range(2))
        self.temporal = nn.ModuleList(TemporalBlock() for _ in range(2))
        self.output_norm = nn.LayerNorm(WIDTH)
        self.heads = nn.ModuleDict({name: nn.Linear(WIDTH, width) for name, width in
            (("p", 3), ("R", 6), ("semantic", 64), ("presence", 1),
             ("joint_type", 3), ("joint_q", 2))})
        self.fingers = nn.Linear(WIDTH, 2)

    def forward(self, h: torch.Tensor, frame_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if h.ndim != 3 or h.shape[1:] != (50, 1024):
            raise ValueError("Phi requires the complete real N×50×1024 native grid")
        indices = check_frames(frame_indices, len(h), h.device)
        normal = h.float() * torch.rsqrt(h.float().square().mean(-1, keepdim=True) + 1e-6)
        memory = self.input(normal)
        slots = self.queries[None].expand(len(h), -1, -1)
        for layer in self.cross:
            slots = layer(slots, memory)
        slots = slots.transpose(0, 1)
        for layer in self.temporal:
            slots = layer(slots, indices)
        slots = self.output_norm(slots.transpose(0, 1))
        fields = {name: head(slots).float() for name, head in self.heads.items()}
        fields["R"] = rotation_6d(fields["R"])
        fields["semantic"] = F.normalize(fields["semantic"], dim=-1, eps=1e-6)
        fields["presence"] = fields["presence"].squeeze(-1).sigmoid()
        fields["joint_type"] = fields["joint_type"].softmax(-1)
        fields["hand_q"] = self.fingers(slots[:, 0]).float()
        return fields


def spatial_fields(physical: Mapping[str, torch.Tensor]):
    """Construct only explicit physical node/edge fields; no slot identities."""
    prefix = check_physical(physical)
    valid = physical.get("valid")
    fields = {name: physical[name].float() for name in PHYSICAL_SHAPES}
    if valid is not None:
        if valid.shape != prefix:
            raise ValueError("F geometry validity must have one flag per physical frame")
        fields = {name: torch.where(valid.bool().reshape(*prefix, *([1] * len(suffix))),
                                    fields[name], torch.zeros_like(fields[name]))
                  for name, suffix in PHYSICAL_SHAPES.items()}
    # Exact-zero absent entities have no geometry (GT padding, or absent input).
    # Predicted sigmoid presence stays continuous; no GT count enters G.
    for name in ("p", "R", "semantic", "joint_type", "joint_q"):
        exists = (fields["presence"] > 0).reshape(
            *prefix, ENTITIES, *([1] * (len(PHYSICAL_SHAPES[name]) - 1)))
        fields[name] = torch.where(exists, fields[name], torch.zeros_like(fields[name]))
    p, rotation = fields["p"], fields["R"]
    hand = torch.zeros((*prefix, ENTITIES, 1), device=p.device, dtype=p.dtype)
    hand[..., 0, :] = 1
    fingers = fields["hand_q"][..., None, :] / 0.04 * hand
    scale = p.new_tensor((0.25, math.pi))
    node = torch.cat((-rotation[..., 2, :], fields["semantic"], fields["joint_type"],
                      fields["joint_q"] / scale, fingers, hand), -1)
    relative_p = torch.einsum("...iab,...ijb->...ija", rotation.transpose(-1, -2),
                              p.unsqueeze(-3) - p.unsqueeze(-2)) / 0.25
    relative_r = torch.einsum("...iab,...jbc->...ijac", rotation.transpose(-1, -2),
                              rotation).flatten(-2)
    source = node.unsqueeze(-2).expand(*prefix, ENTITIES, ENTITIES, NODE_FIELDS)
    destination = node.unsqueeze(-3).expand_as(source)
    edge = torch.cat((relative_p, relative_r, source, destination), -1)
    return node, edge, fields["presence"], valid


class SpatialRelationBlock(nn.Module):
    """Shared entity operator, with relation edges in both bias and Value."""
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.edge = nn.Sequential(nn.Linear(EDGE_FIELDS, WIDTH), nn.GELU(),
                                  nn.Linear(WIDTH, WIDTH))
        self.edge_bias = nn.Linear(WIDTH, HEADS)
        self.edge_value = nn.Linear(WIDTH, WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)
        self.ffn_norm = nn.LayerNorm(WIDTH)
        self.ffn = feed_forward()

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor,
                presence: torch.Tensor) -> torch.Tensor:
        normal = self.norm(nodes)
        q, k = split_heads(self.q(normal)), split_heads(self.k(normal))
        relation = self.edge(edges)
        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(HEAD_WIDTH)
        scores = scores + self.edge_bias(relation).movedim(-1, -3).float()
        alpha = presence_attention(scores, presence)
        values = self.v(normal).float().reshape(*nodes.shape[:-1], HEADS, HEAD_WIDTH)
        edge_values = self.edge_value(relation).float().reshape(
            *relation.shape[:-1], HEADS, HEAD_WIDTH)
        values = (values.unsqueeze(-4) + edge_values) * presence[..., None, :, None, None]
        attended = torch.einsum("...hij,...ijhd->...ihd", alpha, values).flatten(-2)
        nodes = nodes + self.out(attended)
        return nodes + self.ffn(self.ffn_norm(nodes))


class RelationEncoder(nn.Module):
    """Omega, or a completely independent instance with GT input for F."""
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(NODE_FIELDS, WIDTH), nn.GELU(),
                                   nn.Linear(WIDTH, WIDTH))
        self.relations = nn.ModuleList(SpatialRelationBlock() for _ in range(2))
        self.temporal = nn.ModuleList(TemporalBlock() for _ in range(2))

    def spatial(self, physical: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Any leading query batch dimensions, with no temporal interpretation."""
        fields, edges, presence, valid = spatial_fields(physical)
        nodes = self.input(fields)
        for layer in self.relations:
            nodes = layer(nodes, edges, presence)
        return nodes if valid is None else torch.where(valid[..., None, None].bool(),
                                                       nodes, torch.zeros_like(nodes))

    def forward(self, physical: Mapping[str, torch.Tensor],
                frame_indices: torch.Tensor) -> torch.Tensor:
        if physical["p"].ndim != 3:
            raise ValueError("temporal relation encoding requires one N-frame video")
        indices = check_frames(frame_indices, len(physical["p"]), physical["p"].device)
        valid = physical.get("valid")
        if valid is not None and not bool(valid.any()):
            raise ValueError("F teacher geometry needs at least one valid frame")
        nodes = self.spatial(physical).transpose(0, 1)
        for layer in self.temporal:
            nodes = layer(nodes, indices, key_valid=valid)
        nodes = nodes.transpose(0, 1)
        return nodes if valid is None else torch.where(valid[:, None, None].bool(),
                                                       nodes, torch.zeros_like(nodes))


class RelationalRead(nn.Module):
    """Native H selects entity context and its same-selection backward change."""
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(1024, WIDTH)
        self.key = nn.Linear(WIDTH, WIDTH)
        self.context = nn.Linear(WIDTH, 1024)
        self.dynamic = nn.Linear(WIDTH, 1024, bias=False)

    def forward(self, h: torch.Tensor, w: torch.Tensor,
                physical: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 3 or h.shape[1:] != (50, 1024) or w.shape != (len(h), ENTITIES, WIDTH):
            raise ValueError("relational read needs complete H and one 33-entity grid per frame")
        presence = physical["presence"]
        if presence.shape != (len(h), ENTITIES):
            raise ValueError("relational read requires predicted entity presence")
        q, k = split_heads(self.query(h.float())), split_heads(self.key(w))
        alpha = presence_attention(q.float() @ k.float().transpose(-1, -2)
                                   / math.sqrt(HEAD_WIDTH), presence)
        difference = torch.cat((torch.zeros_like(w[:1]), w[1:] - w[:-1]), 0)
        values = split_heads(w.float() * presence[..., None])
        deltas = split_heads(difference.float() * presence[..., None])
        c = self.context(merge_heads(alpha @ values))
        d = self.dynamic(merge_heads(alpha @ deltas))
        return c, d
