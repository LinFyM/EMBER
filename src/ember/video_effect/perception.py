"""Phi: full frozen image patches and exact language to ordered relations.

Only this reader is trained here. Anonymous assignment and all physical labels
are loss-side data; no registry, object-focus list, count or task ID is a model
input. The caller averages four complete condition losses per logical update.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

WIDTH, HEADS, HEAD_WIDTH, ENTITY_SLOTS = 256, 8, 32, 32
PREDICTION_SHAPES = {
    'hand_p': (3,), 'hand_R': (3, 3), 'hand_q': (2,),
    'entity_p': (32, 3), 'entity_R': (32, 3, 3), 'semantic': (32, 64),
    'presence': (32,), 'relevance': (32,), 'joint_type': (32, 3),
    'joint_q': (32, 2), 'presence_logits': (32,), 'relevance_logits': (32,),
    'joint_type_logits': (32, 3),
}


def _ffn() -> nn.Sequential:
    return nn.Sequential(nn.Linear(WIDTH, 1024), nn.GELU(), nn.Linear(1024, WIDTH))


def rotation_6d(value: torch.Tensor) -> torch.Tensor:
    """Retained Gram-Schmidt convention: first two vectors become columns."""
    first, second = value.float().split(3, dim=-1)
    first = F.normalize(first, dim=-1, eps=1e-6)
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-6)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)


def _sines(position: torch.Tensor, width: int) -> torch.Tensor:
    frequency = torch.exp(-math.log(10000.) * torch.arange(0, width, 2).float() / width)
    phase = position.float()[:, None] * frequency[None]
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def _patch_positions() -> torch.Tensor:
    grid = torch.arange(16)
    row, column = torch.meshgrid(grid, grid, indexing='ij')
    spatial = torch.cat((_sines(row.flatten(), 128), _sines(column.flatten(), 128)), -1)
    return (spatial[None] + _sines(torch.arange(2), WIDTH)[:, None]).reshape(512, WIDTH)


class _CrossBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_norm, self.memory_norm = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, HEADS, dropout=0., batch_first=True)
        self.ffn_norm, self.ffn = nn.LayerNorm(WIDTH), _ffn()

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                padding: torch.Tensor) -> torch.Tensor:
        normal = self.memory_norm(memory)
        update = self.attention(self.query_norm(query), normal, normal,
                                key_padding_mask=padding, need_weights=False)[0]
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class _TimeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm, self.qkv = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 3 * WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)
        self.ffn_norm, self.ffn = nn.LayerNorm(WIDTH), _ffn()
        self.register_buffer('frequency', torch.exp(-math.log(10000.) *
                             torch.arange(0, HEAD_WIDTH, 2).float() / HEAD_WIDTH))

    def _rotate(self, value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        phase = (indices.float() / 5)[:, None] * self.frequency[None]
        cosine, sine = phase.cos().to(value.dtype)[None, None], phase.sin().to(value.dtype)[None, None]
        pairs = value.reshape(*value.shape[:-1], HEAD_WIDTH // 2, 2)
        first, second = pairs.unbind(-1)
        return torch.stack((first * cosine - second * sine,
                            first * sine + second * cosine), -1).flatten(-2)

    def forward(self, slots: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, time, _ = slots.shape
        q, k, v = self.qkv(self.norm(slots)).reshape(batch, time, 3, HEADS, HEAD_WIDTH).unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        attended = F.scaled_dot_product_attention(self._rotate(q, indices),
                                                  self._rotate(k, indices), v,
                                                  dropout_p=0., is_causal=False)
        slots = slots + self.output(attended.transpose(1, 2).reshape(batch, time, WIDTH))
        return slots + self.ffn(self.ffn_norm(slots))


class RelationPerception(nn.Module):
    """One unpadded ordered video per call; image tokens are camera-major 16×16."""
    def __init__(self) -> None:
        super().__init__()
        self.patch_projection, self.language_projection = nn.Linear(2048, WIDTH), nn.Linear(2048, WIDTH)
        self.register_buffer('patch_positions', _patch_positions())
        self.queries = nn.Parameter(torch.randn(33, WIDTH) * .02)
        self.cross = nn.ModuleList(_CrossBlock() for _ in range(2))
        self.temporal = nn.ModuleList(_TimeBlock() for _ in range(2))
        self.output_norm = nn.LayerNorm(WIDTH)
        self.hand_heads = nn.ModuleDict({name: nn.Linear(WIDTH, size)
                                        for name, size in (('p', 3), ('R6', 6), ('q', 2))})
        self.entity_heads = nn.ModuleDict({name: nn.Linear(WIDTH, size) for name, size in
            (('p', 3), ('R6', 6), ('semantic', 64), ('presence', 1),
             ('joint_type', 3), ('joint_q', 2), ('relevance', 1))})

    def forward(self, patches: torch.Tensor, language_embeddings: torch.Tensor,
                language_mask: torch.Tensor, frame_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[1:] != (512, 2048) or len(patches) == 0:
            raise ValueError('Phi requires complete real T×512×2048 source image patches')
        if language_embeddings.ndim != 2 or language_embeddings.shape[-1] != 2048:
            raise ValueError('Phi requires exact-language token embeddings N×2048')
        language_mask = language_mask.to(device=patches.device, dtype=torch.bool)
        if language_mask.shape != language_embeddings.shape[:1] or not bool(language_mask.any()):
            raise ValueError('Phi requires a nonempty valid exact-language token mask')
        indices = frame_indices.to(device=patches.device)
        if (indices.shape != (len(patches),) or indices.dtype not in (torch.int32, torch.int64)
                or bool((indices < 0).any()) or bool((indices[1:] <= indices[:-1]).any())):
            raise ValueError('Phi time positions are ordered actual integer frame indices')
        image = self.patch_projection(patches.detach()) + self.patch_positions
        legal_language = language_embeddings.detach().to(device=patches.device)
        legal_language = torch.where(language_mask[:, None], legal_language, torch.zeros_like(legal_language))
        language = self.language_projection(legal_language)[None].expand(len(patches), -1, -1)
        memory = torch.cat((image, language), dim=1)
        padding = torch.cat((torch.zeros(512, device=patches.device, dtype=torch.bool), ~language_mask))
        padding = padding[None].expand(len(patches), -1)
        slots = self.queries[None].expand(len(patches), -1, -1)
        for block in self.cross:
            slots = block(slots, memory, padding)
        slots = slots.transpose(0, 1)
        for block in self.temporal:
            slots = block(slots, indices)
        slots = self.output_norm(slots.transpose(0, 1))
        hand = {name: head(slots[:, 0]).float() for name, head in self.hand_heads.items()}
        entities = {name: head(slots[:, 1:]).float() for name, head in self.entity_heads.items()}
        return {'hand_p': hand['p'], 'hand_R': rotation_6d(hand['R6']), 'hand_q': hand['q'],
                'entity_p': entities['p'], 'entity_R': rotation_6d(entities['R6']),
                'semantic': F.normalize(entities['semantic'], dim=-1, eps=1e-6),
                'presence': entities['presence'].squeeze(-1).sigmoid(),
                'relevance': entities['relevance'].squeeze(-1).sigmoid(),
                'joint_type': entities['joint_type'].softmax(-1), 'joint_q': entities['joint_q'],
                'presence_logits': entities['presence'].squeeze(-1),
                'relevance_logits': entities['relevance'].squeeze(-1),
                'joint_type_logits': entities['joint_type'], 'frame_indices': indices}


def _mean(value: torch.Tensor) -> torch.Tensor | None:
    return value.mean() if value.numel() else None


def _entity_mask(gt: Mapping, field: str) -> torch.Tensor:
    return gt[f'{field}_mask'].bool() & gt['valid'][:, None].bool() & gt['presence'].bool()


@torch.no_grad()
def _matching(pred: Mapping, gt: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    """One whole-video Hungarian solve; labels affect only this loss indexing."""
    target = (gt['presence'].bool() & gt['valid'][:, None].bool()).any(0).nonzero().flatten()
    if not len(target):
        raise ValueError('shared supervision has no valid physical entity')
    cost = pred['entity_p'].new_zeros((32, len(target)), dtype=torch.float32)
    for field, key in (('p', 'entity_p'), ('R', 'entity_R'), ('semantic', 'semantic')):
        mask = _entity_mask(gt, key if field in ('p', 'R') else field)[:, target]
        truth = gt[key][:, target].float()
        truth = torch.where(mask.reshape(*mask.shape, *([1] * (truth.ndim - 2))),
                            truth, torch.zeros_like(truth))
        guessed = pred[key].float()
        if field == 'p':
            error = F.smooth_l1_loss(guessed[:, :, None].expand(-1, -1, len(target), -1) / .25,
                                     truth[:, None].expand(-1, 32, -1, -1) / .25,
                                     reduction='none', beta=1.).mean(-1)
        elif field == 'R':
            error = .25 * (guessed[:, :, None] - truth[:, None]).square().sum((-1, -2))
        else:
            error = 1 - (F.normalize(guessed, dim=-1)[:, :, None] *
                         F.normalize(truth, dim=-1)[:, None]).sum(-1).clamp(-1, 1)
        cost += torch.where(mask[:, None], error, torch.zeros_like(error)).sum(0) / mask.sum(0).clamp_min(1)[None]
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(rows, device=cost.device, dtype=torch.long),
            target[torch.as_tensor(columns, device=cost.device, dtype=torch.long)])


def _geometry(pred: Mapping, gt: Mapping, field: str, rows: torch.Tensor,
              columns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    guessed = torch.cat((pred[f'hand_{field}'][:, None], pred[f'entity_{field}'][:, rows]), 1).float()
    truth = torch.cat((gt[f'hand_{field}'][:, None], gt[f'entity_{field}'][:, columns]), 1).float()
    mask = torch.cat((gt[f'hand_{field}_mask'][:, None].bool() & gt['valid'][:, None].bool(),
                      _entity_mask(gt, f'entity_{field}')[:, columns]), 1)
    return guessed, truth, mask


def _regression(guessed: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor,
                *, scale=1., rotation=False) -> torch.Tensor | None:
    guessed, truth = guessed[mask].float(), truth[mask].float()
    if rotation:
        return _mean(.25 * (guessed - truth).square().sum((-1, -2)))
    return _mean(F.smooth_l1_loss(guessed / scale, truth / scale, reduction='none', beta=1.))


def _change(guessed: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor,
            *, scale=1., rotation=False) -> torch.Tensor | None:
    adjacent = mask[1:] & mask[:-1]
    return _regression(guessed[1:][adjacent] - guessed[:-1][adjacent],
                       truth[1:][adjacent] - truth[:-1][adjacent],
                       torch.ones_like(adjacent[adjacent]), scale=scale, rotation=rotation)


def _balanced_bce(logits: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor | None:
    available = []
    for positive in (True, False):
        selected = mask & ((target > .5) == positive)
        if bool(selected.any()):
            available.append(F.binary_cross_entropy_with_logits(logits.float()[selected],
                                                               target.float()[selected]))
    return torch.stack(available).mean() if available else None


def loss(prediction: Mapping[str, torch.Tensor], label: Mapping[str, torch.Tensor]
         ) -> tuple[torch.Tensor, dict]:
    """Eight fixed 1/8 groups; masked OOI contributes differentiable zero.

    Each available subgroup averages its valid entity/frame elements. No task
    normalization, count input, masked-value imputation or trajectory fitting.
    Metrics report an unavailable OOI group as None, never as zero error.
    """
    pred = prediction
    time, device = len(pred['entity_p']), pred['entity_p'].device
    for key, shape in PREDICTION_SHAPES.items():
        if pred[key].shape != (time, *shape):
            raise ValueError(f'perception prediction shape differs: {key}')
    gt = {key: value.detach().to(device=device) for key, value in label.items()}
    if (gt['valid'].shape != (time,) or not bool(gt['valid'].any())
            or not torch.equal(pred['frame_indices'], gt['frame_indices'])):
        raise ValueError('perception loss needs the same real, labeled video frame indices')
    rows, columns = _matching(pred, gt)
    p, truth_p, p_mask = _geometry(pred, gt, 'p', rows, columns)
    R, truth_R, R_mask = _geometry(pred, gt, 'R', rows, columns)
    semantic_mask = _entity_mask(gt, 'semantic')[:, columns]
    semantic = 1 - (F.normalize(pred['semantic'][:, rows][semantic_mask].float(), dim=-1) *
                    F.normalize(gt['semantic'][:, columns][semantic_mask].float(), dim=-1)).sum(-1).clamp(-1, 1)
    exists = gt['presence'][:, columns].bool() & gt['valid'][:, None].bool()
    joint_mask = _entity_mask(gt, 'joint_type')[:, columns]
    joint_type = _mean(F.cross_entropy(pred['joint_type_logits'][:, rows][joint_mask].float(),
                                      gt['joint_type'][:, columns][joint_mask].argmax(-1), reduction='none'))
    q_mask = gt['joint_q_mask'][:, columns].bool() & exists[:, :, None]
    q, truth_q = pred['joint_q'][:, rows].float(), gt['joint_q'][:, columns].float()
    q_scale = q.new_tensor((.25, math.pi))
    joint_q = _regression(q / q_scale, truth_q / q_scale, q_mask)
    hand_mask = gt['hand_q_mask'].bool() & gt['valid'][:, None].bool()
    presence_target = torch.zeros_like(pred['presence'])
    presence_target[:, rows] = gt['presence'][:, columns].float()
    presence_mask = gt['valid'][:, None].bool().expand_as(presence_target)
    relevance_target = torch.zeros_like(pred['relevance'])
    relevance_target[:, rows] = gt['relevance'][:, columns].float()
    relevance_mask = torch.zeros_like(presence_mask)
    relevance_mask[:, rows] = gt['relevance_mask'][:, columns].bool()
    # Empty slots have negative OOI supervision only in unmasked conditions.
    if bool(gt['relevance_group_mask'].item()):
        relevance_mask = presence_mask
    joint_terms = [value for value in (joint_type, joint_q) if value is not None]
    motion = {'position': _change(p, truth_p, p_mask, scale=.25),
              'rotation': _change(R, truth_R, R_mask, rotation=True),
              'joint_q': _change(q / q_scale, truth_q / q_scale, q_mask),
              'hand_q': _change(pred['hand_q'], gt['hand_q'], hand_mask, scale=.04)}
    motion_terms = [value for value in motion.values() if value is not None]
    groups = {'position': _regression(p, truth_p, p_mask, scale=.25),
              'rotation': _regression(R, truth_R, R_mask, rotation=True), 'semantic': _mean(semantic),
              'presence': _balanced_bce(pred['presence_logits'], presence_target, presence_mask),
              'joint': torch.stack(joint_terms).mean() if joint_terms else None,
              'hand_q': _regression(pred['hand_q'], gt['hand_q'], hand_mask, scale=.04),
              'relevance': _balanced_bce(pred['relevance_logits'], relevance_target, relevance_mask),
              'motion': torch.stack(motion_terms).mean() if motion_terms else None}
    if any(value is None for key, value in groups.items() if key != 'relevance'):
        raise ValueError('registered video lacks a required geometry/motion supervision group')
    terms = [value if value is not None else pred['relevance_logits'].sum() * 0
             for value in groups.values()]
    total = torch.stack(terms).sum() / 8
    extra = {'joint_type': joint_type, 'joint_q': joint_q,
             **{f'motion_{key}': value for key, value in motion.items()}}
    available = {key: value for key, value in {**groups, **extra}.items() if value is not None}
    values = torch.stack([total.detach(), *(value.detach() for value in available.values())]).cpu().tolist()
    metrics = {key: None for key in (*groups, *extra)}
    metrics.update(zip(available, values[1:]))
    assignment = rows.new_full((32,), -1)
    assignment[rows] = columns
    metrics.update(loss=values[0], group_weight=.125, valid_groups=sum(v is not None for v in groups.values()),
                   relevance_group_mask=bool(gt['relevance_group_mask'].item()),
                   matched_entities=len(rows), valid_frames=int(gt['valid'].sum().item()),
                   assignment=assignment.cpu().tolist())
    return total, metrics
