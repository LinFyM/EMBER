"""Shared actual-parameter and experience-conditioned complete LoRA editing.

Raw teacher/execution evidence and incoming behavior factors stop gradient.
Learned parameter/evidence encoding, teaching reads and edits share one graph.
The runtime owns evidence provenance, interaction budgets and stopping; this
module has no prescribed number of practices or teaching reads.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX

WIDTH, RANK = 256, 128
TEACHER_KEYS = {"phi", "hidden", "language", "indices"}
EXPERIENCE_SHAPES = {
    "images": (2, 512, 2048), "hidden": (2, 50, 1024),
    "proprio": (2, 8), "actions": (5, 7), "executed": (5,),
    "feedback": (4,), "episode": (), "step": (),
}


def position_code(indices: torch.Tensor, width: int = WIDTH) -> torch.Tensor:
    frequency = torch.exp(torch.arange(0, width, 2, device=indices.device).float()
                          * (-math.log(10000.0) / width))
    phase = indices.float()[..., None] * frequency
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


def patch_code() -> torch.Tensor:
    # Each of the two native cameras has a 16x16 patch lattice.
    index = torch.arange(256).repeat(2)
    return torch.cat((position_code(index // 16, 128),
                      position_code(index % 16, 128)), -1)


def recompute(function, *args):
    if torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
    return function(*args)


class Attention(nn.Module):
    """Pre-normalized attention with SDPA rather than a retained score matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.query_norm, self.memory_norm = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.query, self.key = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)
        self.value, self.out = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        def heads(x):
            return x.reshape(x.shape[0], -1, 8, 32).transpose(1, 2)
        query, memory = self.query_norm(query), self.memory_norm(memory)
        attended = F.scaled_dot_product_attention(
            heads(self.query(query)), heads(self.key(memory)), heads(self.value(memory)),
            dropout_p=0.0)
        return self.out(attended.transpose(1, 2).reshape(query.shape))


class TemporalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = Attention()
        self.feedforward = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 4 * WIDTH),
                                         nn.SiLU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(tokens, tokens)
        return tokens + self.feedforward(tokens)


class AxialBlock(nn.Module):
    """Preserve actual target/rank axes while reading the condition memory."""

    def __init__(self) -> None:
        super().__init__()
        self.read, self.rank, self.target = Attention(), Attention(), Attention()
        self.feedforward = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 4 * WIDTH),
                                         nn.SiLU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = q + self.read(q.reshape(1, -1, WIDTH), memory[None]).reshape(q.shape)
        q = q + self.rank(q, q)
        across_targets = q.transpose(0, 1)
        q = q + self.target(across_targets, across_targets).transpose(0, 1)
        return q + self.feedforward(q)


class ExperienceEncoder(nn.Module):
    """All real decisions enter facts; pooling never filters failed decisions."""

    def __init__(self, chunk: int) -> None:
        super().__init__()
        self.chunk = chunk
        self.image, self.hidden = nn.Linear(2048, WIDTH), nn.Linear(1024, WIDTH)
        self.numeric = nn.Sequential(nn.Linear(60, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.fact_query = nn.Parameter(torch.randn(1, WIDTH) * 0.02)
        self.slots = nn.Parameter(torch.randn(16, WIDTH) * 0.02)
        self.camera = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.observation_phase = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.flow_time = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.fact_read, self.pool = Attention(), Attention()
        self.temporal = nn.ModuleList(TemporalBlock() for _ in range(2))
        self.register_buffer("patch_position", patch_code(), persistent=False)
        self.register_buffer("horizon_position", position_code(torch.arange(50)), persistent=False)

    def _facts(self, images: torch.Tensor, hidden: torch.Tensor,
               numeric: torch.Tensor) -> torch.Tensor:
        dtype, device = self.fact_query.dtype, self.fact_query.device
        images = self.image(images.detach().to(device=device, dtype=dtype))
        images = (images + self.patch_position + self.camera.repeat_interleave(256, 0)
                  + self.observation_phase[:, None] + self.modality[0])
        # Hidden axis 1 is actual flow time (1, .1), not pre/post observation.
        hidden = self.hidden(hidden.detach().to(device=device, dtype=dtype))
        hidden = hidden + self.horizon_position + self.flow_time[:, None] + self.modality[1]
        memory = torch.cat((images.flatten(1, 2), hidden.flatten(1, 2)), 1)
        query = self.fact_query[None] + self.numeric(numeric)[:, None]
        return (query + self.fact_read(query, memory))[:, 0]

    def forward(self, experience: dict[str, torch.Tensor]) -> torch.Tensor:
        if not experience:
            return self.slots
        if set(experience) != set(EXPERIENCE_SHAPES):
            raise ValueError("experience must contain only the registered actual-execution fields")
        count = experience["images"].shape[0]
        for name, tail in EXPERIENCE_SHAPES.items():
            if experience[name].shape != (count, *tail):
                raise ValueError(f"experience {name} must retain its complete registered shape")
        if experience["executed"].dtype != torch.bool:
            raise ValueError("executed is the boolean actual-action mask")
        if count == 0:
            return self.slots
        device, dtype = self.slots.device, self.slots.dtype
        episode = experience["episode"].detach().to(device=device)
        step = experience["step"].detach().to(device=device)
        if episode.is_floating_point() or step.is_floating_point():
            raise ValueError("episode and environment-step identities must be integers")
        ordered = ((episode[1:] > episode[:-1])
                   | ((episode[1:] == episode[:-1]) & (step[1:] > step[:-1])))
        if not bool(ordered.all()) or bool((episode < 0).any()) or bool((step < 0).any()):
            raise ValueError("experience must retain increasing actual episode/time identity")
        mask = experience["executed"].detach().to(device=device)
        actions = experience["actions"].detach().to(device=device, dtype=dtype)
        numeric = torch.cat((experience["proprio"].detach().to(device=device, dtype=dtype).flatten(1),
                             torch.where(mask[..., None], actions, 0).flatten(1), mask.to(dtype),
                             experience["feedback"].detach().to(device=device, dtype=dtype)), -1)
        facts = torch.cat([recompute(self._facts, experience["images"][i:i + self.chunk],
                                     experience["hidden"][i:i + self.chunk], numeric[i:i + self.chunk])
                           for i in range(0, count, self.chunk)])
        positions = torch.cat((position_code(episode, 128), position_code(step, 128)), -1)
        facts = (facts + positions.to(dtype=dtype))[None]
        for block in self.temporal:
            facts = recompute(block, facts)
        return self.slots + self.pool(self.slots[None], facts)[0]


class TeachingReader(nn.Module):
    """Actual-parameter/experience queries precede eight-slot spatial compression."""

    def __init__(self, chunk: int) -> None:
        super().__init__()
        self.chunk = chunk
        self.image, self.hidden = nn.Linear(2048, WIDTH), nn.Linear(1024, WIDTH)
        self.language = nn.Linear(2048, WIDTH)
        self.spatial_queries = nn.Parameter(torch.randn(8, WIDTH) * 0.02)
        self.camera = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.endpoints = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.experience_read, self.state_read, self.spatial = Attention(), Attention(), Attention()
        self.temporal = nn.ModuleList(TemporalBlock() for _ in range(4))
        self.register_buffer("patch_position", patch_code(), persistent=False)
        self.register_buffer("horizon_position", position_code(torch.arange(50)), persistent=False)

    def _spatial(self, phi: torch.Tensor, hidden: torch.Tensor,
                 queries: torch.Tensor) -> torch.Tensor:
        device, dtype = self.spatial_queries.device, self.spatial_queries.dtype
        phi = self.image(phi.detach().to(device=device, dtype=dtype))
        phi = phi + self.patch_position + self.camera.repeat_interleave(256, 0) + self.modality[0]
        hidden = self.hidden(hidden.detach().to(device=device, dtype=dtype))
        hidden = hidden + self.horizon_position + self.modality[1]
        memory = torch.cat((phi, hidden), 1)
        return queries + self.spatial(queries, memory)

    def forward(self, teacher: dict[str, torch.Tensor], facts: torch.Tensor,
                q: torch.Tensor) -> torch.Tensor:
        if set(teacher) != TEACHER_KEYS:
            raise ValueError("teacher input is only Phi, full hidden, exact language and real frame indices")
        count = teacher["phi"].shape[0]
        if (count < 1 or teacher["phi"].shape != (count, 512, 2048)
                or teacher["hidden"].shape != (count, 50, 1024)
                or teacher["language"].shape != (2048,) or teacher["indices"].shape != (count,)):
            raise ValueError("teacher requires the complete ordered native feature grid")
        device, dtype = self.spatial_queries.device, self.spatial_queries.dtype
        indices = teacher["indices"].detach().to(device=device)
        if indices.is_floating_point() or bool((indices < 0).any()) or not bool((indices[1:] > indices[:-1]).all()):
            raise ValueError("teacher indices must be the increasing real stride/end-frame identities")
        language = self.language(teacher["language"].detach().to(device=device, dtype=dtype))
        queries = self.spatial_queries + language
        queries = queries + self.experience_read(queries[None], facts[None])[0]
        queries = queries + self.state_read(queries[None], q.reshape(1, -1, WIDTH))[0]
        # Both boundaries are represented even when a video has one actual frame.
        ordinal = torch.arange(count, device=device)
        boundaries = torch.stack((ordinal == 0, ordinal == count - 1), -1).to(dtype)
        endpoints = boundaries @ self.endpoints
        queries = queries[None] + (position_code(indices).to(dtype=dtype) + endpoints)[:, None]
        slots = torch.cat([recompute(self._spatial, teacher["phi"][i:i + self.chunk],
                                     teacher["hidden"][i:i + self.chunk], queries[i:i + self.chunk])
                           for i in range(0, count, self.chunk)])
        tokens = slots.reshape(1, -1, WIDTH)
        for block in self.temporal:
            tokens = recompute(block, tokens)
        return torch.cat((language[None], tokens[0]), 0)


class ParameterEncoder(nn.Module):
    """Encode actual A rows/B columns, sharing projections only within each role."""

    def __init__(self, mt_state: Mapping[str, torch.Tensor], target_names) -> None:
        super().__init__()
        self.keys = tuple((name + LORA_A_SUFFIX, name + LORA_B_SUFFIX) for name in target_names)
        self.shapes, widths, scales = {}, (set(), set()), []
        for keys in self.keys:
            for role, key in enumerate(keys):
                matrix = mt_state[key]
                rank_axis, width_axis = (0, 1) if role == 0 else (1, 0)
                if (matrix.ndim != 2 or not matrix.is_floating_point()
                        or matrix.shape[rank_axis] != RANK or matrix.shape[width_axis] < 1):
                    raise ValueError(f"actual A/B factors require rank128 and positive width: {key}")
                self.shapes[key] = tuple(matrix.shape)
                widths[role].add(matrix.shape[width_axis])
                scales.append(matrix.detach().to(device="cpu", dtype=torch.float32)
                              .square().mean().sqrt().clamp_min(1e-6))
        self.a = nn.ModuleDict({str(width): nn.Linear(width, 128) for width in sorted(widths[0])})
        self.b = nn.ModuleDict({str(width): nn.Linear(width, 128) for width in sorted(widths[1])})
        self.target_identity = nn.Parameter(torch.randn(len(self.keys), WIDTH) * 0.02)
        self.rank_identity = nn.Parameter(torch.randn(RANK, WIDTH) * 0.02)
        self.norm = nn.LayerNorm(WIDTH)
        self.register_buffer("rms", torch.stack(scales).reshape(len(self.keys), 2))

    def forward(self, incoming: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if set(incoming) != self.shapes.keys():
            raise ValueError("incoming state must contain exactly the complete registered A/B factors")
        device, dtype = self.target_identity.device, self.target_identity.dtype
        tokens = []
        for target, keys in enumerate(self.keys):
            roles = []
            for role, key in enumerate(keys):
                matrix = incoming[key]
                if tuple(matrix.shape) != self.shapes[key] or not matrix.is_floating_point():
                    raise ValueError(f"incoming factor changed its actual registered A/B shape: {key}")
                rows = matrix.detach().to(device=device, dtype=dtype)
                rows = (rows if role == 0 else rows.T) / self.rms[target, role]
                projection = (self.a if role == 0 else self.b)[str(rows.shape[1])]
                roles.append(projection(rows))
            tokens.append(torch.cat(roles, -1))
        return self.norm(torch.stack(tokens) + self.target_identity[:, None] + self.rank_identity[None])


class ExperienceCompiler(nn.Module):
    """Stateless shared editor: actual complete A/B plus teaching and real facts."""

    def __init__(self, mt_state: dict[str, torch.Tensor], target_names,
                 seed: int = 20261009, decoder_chunk: int = 4096, *,
                 frame_chunk: int = 8, experience_chunk: int = 16) -> None:
        super().__init__()
        from .decoder import CoordinateDecoder

        self.target_names = tuple(target_names)
        if not self.target_names or len(set(self.target_names)) != len(self.target_names):
            raise ValueError("parameter targets must be a nonempty unique sequence")
        if frame_chunk < 1 or experience_chunk < 1:
            raise ValueError("physical frame/experience chunks must be positive")
        # All construction is CPU; this scope leaves caller CPU/CUDA RNG intact.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.reader, self.encoder = TeachingReader(frame_chunk), ExperienceEncoder(experience_chunk)
            self.parameter_encoder = ParameterEncoder(mt_state, self.target_names)
            self.editor = nn.ModuleList(AxialBlock() for _ in range(2))
            self.update_gate, self.update_out = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)
            nn.init.constant_(self.update_gate.bias, -2.0)
            nn.init.zeros_(self.update_out.weight)
            nn.init.zeros_(self.update_out.bias)
            self.decoder = CoordinateDecoder(mt_state, self.target_names, seed=seed, chunk_rows=decoder_chunk)

    def delta(self, incoming_state: Mapping[str, torch.Tensor], teacher: dict[str, torch.Tensor],
              experience: dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.parameter_encoder(incoming_state)
        facts = self.encoder(experience)
        memory = torch.cat((self.reader(teacher, facts, q), facts), 0)
        for block in self.editor:
            q = recompute(block, q, memory)
        return torch.sigmoid(self.update_gate(q)) * self.update_out(q)

    def forward(self, incoming_state: Mapping[str, torch.Tensor], teacher: dict[str, torch.Tensor],
                experience: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decoder(incoming_state, self.delta(incoming_state, teacher, experience))
