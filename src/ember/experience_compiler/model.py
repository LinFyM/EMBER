"""Ordered evidence context and a learned energy on actual control responses.

Raw teaching and execution facts stop gradient. Current policy actions enter
only the small energy MLP; the runtime owns native policy derivatives, complete
factor updates, evidence provenance and interaction stopping.
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

    def encode(self, experience: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ordered decision facts and slots pooling every real decision."""
        if not experience:
            return self.slots.new_empty((0, WIDTH)), self.slots
        if set(experience) != set(EXPERIENCE_SHAPES):
            raise ValueError("experience must contain only the registered actual-execution fields")
        count = experience["images"].shape[0]
        for name, tail in EXPERIENCE_SHAPES.items():
            if experience[name].shape != (count, *tail):
                raise ValueError(f"experience {name} must retain its complete registered shape")
        if experience["executed"].dtype != torch.bool:
            raise ValueError("executed is the boolean actual-action mask")
        if count == 0:
            return self.slots.new_empty((0, WIDTH)), self.slots
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
        return facts[0], self.slots + self.pool(self.slots[None], facts)[0]

    def forward(self, experience: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(experience)[1]


class TeachingReader(nn.Module):
    """Language and real experience change queries before spatial compression."""

    def __init__(self, chunk: int) -> None:
        super().__init__()
        self.chunk = chunk
        self.image, self.hidden = nn.Linear(2048, WIDTH), nn.Linear(1024, WIDTH)
        self.language = nn.Linear(2048, WIDTH)
        self.spatial_queries = nn.Parameter(torch.randn(8, WIDTH) * 0.02)
        self.camera = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.endpoints = nn.Parameter(torch.randn(2, WIDTH) * 0.02)
        self.experience_read, self.spatial = Attention(), Attention()
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

    def forward(self, teacher: dict[str, torch.Tensor], facts: torch.Tensor) -> torch.Tensor:
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


class ExperienceCompiler(nn.Module):
    """Shared functional criterion; no raw parameter tokens or factor decoder."""

    def __init__(self, mt_state: Mapping[str, torch.Tensor], target_names,
                 seed: int = 20261010, *,
                 frame_chunk: int = 8, experience_chunk: int = 16) -> None:
        super().__init__()
        self.target_names = tuple(target_names)
        if not self.target_names or len(set(self.target_names)) != len(self.target_names):
            raise ValueError("parameter targets must be a nonempty unique sequence")
        if frame_chunk < 1 or experience_chunk < 1:
            raise ValueError("physical frame/experience chunks must be positive")
        self.keys = tuple(name + suffix for name in self.target_names
                          for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX))
        self.shapes, scales = self._factor_units(mt_state)
        self.register_buffer("rms", scales)
        # All construction is CPU; this scope leaves caller CPU/CUDA RNG intact.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(seed)
            self.reader, self.encoder = TeachingReader(frame_chunk), ExperienceEncoder(experience_chunk)
            self.support_read = Attention()
            self.action_projection = nn.Linear(35, 128)
            self.energy = nn.Sequential(nn.Linear(384, WIDTH), nn.SiLU(),
                                        nn.Linear(WIDTH, 128), nn.SiLU(),
                                        nn.Linear(128, 1, bias=False))
            nn.init.zeros_(self.energy[-1].weight)

    def _factor_units(self, mt_state):
        if set(mt_state) != set(self.keys):
            raise ValueError("MT state must contain exactly the complete registered A/B factors")
        shapes, scales = {}, []
        for index, key in enumerate(self.keys):
            matrix = mt_state[key]
            rank_axis, width_axis = (0, 1) if index % 2 == 0 else (1, 0)
            if (matrix.ndim != 2 or not matrix.is_floating_point()
                    or matrix.shape[rank_axis] != RANK or matrix.shape[width_axis] < 1):
                raise ValueError(f"actual A/B factors require rank128 and positive width: {key}")
            shapes[key] = tuple(matrix.shape)
            scales.append(matrix.detach().to(device="cpu", dtype=torch.float32)
                          .square().mean().sqrt().clamp_min(1e-6))
        units = torch.stack(scales)
        if not torch.isfinite(units).all():
            raise ValueError("fixed MT RMS units must be finite")
        return shapes, units

    def precondition(self, cotangent: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Multiply all actual factor cotangents by fixed MT RMS squared in FP32."""
        if set(cotangent) != set(self.keys):
            raise ValueError("cotangent must contain exactly the complete registered A/B factors")
        result = {}
        for index, key in enumerate(self.keys):
            value = cotangent[key]
            if tuple(value.shape) != self.shapes[key] or not value.is_floating_point():
                raise ValueError(f"cotangent changed its actual registered A/B shape: {key}")
            scale = self.rms[index].to(device=value.device, dtype=torch.float32)
            result[key] = value.float() * scale.square()
        return result

    def context(self, teacher: dict[str, torch.Tensor], experience: dict[str, torch.Tensor],
                support_indices: torch.Tensor) -> torch.Tensor:
        """Read the entire legal condition for selected, distinct real decisions."""
        if (support_indices.ndim != 1 or support_indices.dtype != torch.long
                or not 1 <= support_indices.numel() <= 16):
            raise ValueError("support indices must be one to sixteen actual long decision indices")
        decisions, facts = self.encoder.encode(experience)
        indices = support_indices.detach().to(device=decisions.device)
        if (bool((indices < 0).any()) or bool((indices >= len(decisions)).any())
                or len(indices.unique()) != len(indices)):
            raise ValueError("support indices require distinct existing actual decisions")
        memory = torch.cat((self.reader(teacher, facts), facts), 0)
        queries = decisions.index_select(0, indices)
        return queries + self.support_read(queries[None], memory[None])[0]

    def action_cotangent(self, context: torch.Tensor, actions: torch.Tensor, *,
                         create_graph: bool) -> torch.Tensor:
        """Return each ∂C_i/∂a_i; the native derivative owner supplies 1/M.

        Actions are independent leaves, so no native policy or evidence graph is
        traversed here. Only this small MLP participates in mixed differentiation.
        """
        count = context.shape[0]
        if (context.shape != (count, WIDTH) or not 1 <= count <= 16
                or actions.shape != (count, 5, 7)
                or not context.is_floating_point() or not actions.is_floating_point()):
            raise ValueError("energy requires one to sixteen contexts256 and actual actions5x7")
        dtype, device = self.action_projection.weight.dtype, self.action_projection.weight.device
        with torch.enable_grad():
            points = actions.detach().to(device=device, dtype=dtype).requires_grad_()
            facts = context.to(device=device, dtype=dtype) if create_graph else context.detach().to(device=device, dtype=dtype)
            energies = self.energy(torch.cat((facts, self.action_projection(points.flatten(1))), -1))
            return torch.autograd.grad(energies.sum(), points, create_graph=create_graph)[0]
