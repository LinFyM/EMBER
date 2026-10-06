"""Independent train-only relationship feedback; GT never enters the Writer."""
from __future__ import annotations

import torch
from torch import nn


def hand_relations(physical):
    p, rotation = physical['p'].float(), physical['R'].float()
    hand = rotation[..., :1, :, :].transpose(-1, -2)
    displacement = (hand @ (p - p[..., :1, :]).unsqueeze(-1)).squeeze(-1) / .25
    relative_rotation = hand @ rotation
    return displacement, relative_rotation


class FeedbackFunction(nn.Module):
    """Shared state–relationship function with a zero-initial seven-dimensional edit."""
    def __init__(self):
        super().__init__()
        from .representation import RelationEncoder
        from .attention import CrossAttentionBlock
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261006)
            self.encoder = RelationEncoder()
            self.value = nn.Sequential(nn.Linear(528, 256), nn.GELU(), nn.Linear(256, 256))
            self.query = nn.Linear(1024, 256)
            self.layers = nn.ModuleList(CrossAttentionBlock() for _ in range(2))
            self.out = nn.Linear(256, 7)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def _single_teacher(self, teacher, current, h0, v0, indices):
        # Missing terminal GT and physical padding are masked, never fake poses.
        observed = teacher['valid'].bool()
        if not bool(observed.any()):
            raise ValueError('F requires observed nonterminal teacher relationships')
        teacher = {key: value[observed] for key, value in teacher.items()}
        indices = indices[observed]
        w = self.encoder(teacher, indices)
        current_nodes = self.encoder.spatial(current)
        count, frames, entities = len(h0), len(w), w.shape[-2]
        r_a, q_a = hand_relations(teacher)
        r_b, q_b = hand_relations(current)
        relative_q = q_b[:, None].transpose(-1, -2) @ q_a[None]
        joint = (teacher['joint_q'][None] - current['joint_q'][:, None]).float()
        joint = joint / joint.new_tensor([.25, torch.pi])
        finger = (teacher['hand_q'][None] - current['hand_q'][:, None]).float() / .04
        values = self.value(torch.cat((
            w[None].expand(count, -1, -1, -1),
            current_nodes[:, None].expand(-1, frames, -1, -1),
            r_a[None] - r_b[:, None], relative_q.flatten(-2), joint,
            finger[:, :, None].expand(-1, -1, entities, -1)), -1))
        presence = teacher['presence'].float()
        valid = teacher['valid'].bool()
        weights = presence * valid[:, None]
        values = values * weights[None, :, :, None]
        memory = values.flatten(1, 2)
        keys = (weights > 0).flatten()[None].expand(count, -1)
        if not bool(keys.any(-1).all()):
            raise ValueError('F teacher has no observed valid relationship')
        query = self.query(h0.float())
        for layer in self.layers:
            query = layer(query, memory, key_valid=keys)
        residual = self.out(query).float()
        return torch.cat((v0[..., :7].float() + residual, v0[..., 7:].float()), -1)

    def forward(self, teacher, current, h0, v0, frame_indices):
        if h0.shape[1:] != (50, 1024) or v0.shape != (len(h0), 50, 32):
            raise ValueError('F requires actual complete source noisy-query H0/v0')
        if teacher['p'].ndim == 3:
            return self._single_teacher(teacher, current, h0, v0, frame_indices)
        if teacher['p'].ndim != 4 or len(teacher['p']) != len(h0):
            raise ValueError('F padded teaching batch changed')
        outputs = []
        for index in range(len(h0)):
            a = {key: value[index] for key, value in teacher.items()}
            b = {key: value[index:index+1] for key, value in current.items()}
            outputs.append(self._single_teacher(a, b, h0[index:index+1], v0[index:index+1], frame_indices[index]))
        return torch.cat(outputs)
