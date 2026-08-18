"""Layer-matched temporal memory, video-set consensus, and axial M2P."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ember.writer.errors import WriterModelError
from ember.writer.temporal import CoreSlotReader, RMSNorm, _apply_rope


EXPERT_LAYERS = 18
PUBLIC_RANK = 16
PARAMETER_GROUPS = EXPERT_LAYERS + 2
PROGRAM_WIDTH = 256
MEMORY_WIDTH = 1024


def _split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, width = value.shape
    if width % heads:
        raise WriterModelError("LMMPC attention width changed")
    return value.reshape(batch, tokens, heads, width // heads).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    return value.transpose(1, 2).reshape(batch, tokens, heads * width)


class LayerRankMemoryReader(torch.nn.Module):
    """Use task Core to read ordered Procedure into native layer/rank Values."""

    def __init__(self, *, heads: int, initialization_seed: int) -> None:
        super().__init__()
        if heads <= 0 or PROGRAM_WIDTH % heads:
            raise WriterModelError("invalid Procedure-to-memory attention")
        self.heads = int(heads)
        self.memory_norm = RMSNorm(MEMORY_WIDTH)
        self.memory_projection = torch.nn.Linear(
            MEMORY_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.core_reader = CoreSlotReader(width=PROGRAM_WIDTH, heads=heads)
        self.core_query_norm = RMSNorm(PROGRAM_WIDTH)
        self.procedure_norm = RMSNorm(PROGRAM_WIDTH)
        self.query = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        self.address_norm = RMSNorm(PROGRAM_WIDTH)
        self.key = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        self.output = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 0x4D52)
            self.layer_identity = torch.nn.Parameter(
                torch.empty(EXPERT_LAYERS, PROGRAM_WIDTH)
            )
            self.rank_identity = torch.nn.Parameter(
                torch.empty(PUBLIC_RANK, PROGRAM_WIDTH)
            )
            torch.nn.init.normal_(self.layer_identity, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.rank_identity, mean=0.0, std=0.02)

    def _read_video(
        self,
        memory: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        address: torch.Tensor,
        core_slots: torch.Tensor,
    ) -> torch.Tensor:
        cells = EXPERT_LAYERS * PUBLIC_RANK
        if (
            memory.ndim != 4
            or memory.shape[1:] != (
                EXPERT_LAYERS,
                PUBLIC_RANK,
                MEMORY_WIDTH,
            )
            or procedure.shape != (memory.shape[0], PROGRAM_WIDTH)
            or positions.shape != (memory.shape[0],)
            or positions.dtype != torch.long
            or positions.device != memory.device
            or address.shape != (cells, PROGRAM_WIDTH)
            or core_slots.shape != (cells, PROGRAM_WIDTH)
            or memory.shape[0] <= 0
        ):
            raise WriterModelError("invalid layer/rank memory video")
        length = memory.shape[0]
        projected = self.memory_projection(self.memory_norm(memory))
        relative = projected - projected[:1]
        dynamic_value = relative - relative.mean(dim=0, keepdim=True)
        query_key = address + self.core_query_norm(core_slots)
        query = self.query(query_key)[:, None]
        key = self.key(self.procedure_norm(procedure))[None].expand(
            cells, -1, -1
        )
        value = dynamic_value.permute(1, 2, 0, 3).reshape(
            cells, length, PROGRAM_WIDTH
        )
        query_heads = _apply_rope(
            _split_heads(query, self.heads),
            torch.zeros(cells, 1, dtype=torch.long, device=query.device),
        )
        key_heads = _apply_rope(
            _split_heads(key, self.heads),
            positions[None].expand(cells, -1),
        )
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            _split_heads(value, self.heads),
            dropout_p=0.0,
            is_causal=False,
        )
        final = self.output(_merge_heads(attended)[:, 0])
        return final.reshape(
            EXPERT_LAYERS, PUBLIC_RANK, PROGRAM_WIDTH
        )

    def forward(
        self,
        layer_memory: torch.Tensor,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        procedure: torch.Tensor,
        positions: torch.Tensor,
        valid_procedure: torch.Tensor,
        video_bounds: tuple[int, ...],
    ) -> torch.Tensor:
        videos = len(video_bounds) - 1
        if (
            layer_memory.ndim != 4
            or layer_memory.shape[1:]
            != (EXPERT_LAYERS, PUBLIC_RANK, MEMORY_WIDTH)
            or core.ndim != 3
            or core.shape[0] != videos
            or core.shape[-1] != PROGRAM_WIDTH
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or procedure.ndim != 3
            or procedure.shape[0] != videos
            or procedure.shape[-1] != PROGRAM_WIDTH
            or positions.shape != procedure.shape[:2]
            or positions.dtype != torch.long
            or positions.device != procedure.device
            or valid_procedure.shape != procedure.shape[:2]
            or valid_procedure.dtype != torch.bool
            or video_bounds[0] != 0
            or video_bounds[-1] != layer_memory.shape[0]
        ):
            raise WriterModelError("invalid Procedure-to-memory batch")
        cells = EXPERT_LAYERS * PUBLIC_RANK
        address = self.address_norm(
            (
                self.layer_identity[:, None] + self.rank_identity[None]
            ).reshape(cells, PROGRAM_WIDTH)
        )
        core_slots = self.core_reader(
            address[None].expand(videos, -1, -1),
            core,
            valid_core,
        )
        rows = []
        for video, (left, right) in enumerate(
            zip(video_bounds[:-1], video_bounds[1:], strict=True)
        ):
            length = right - left
            if (
                length <= 0
                or length > procedure.shape[1]
                or not bool(valid_procedure[video, :length].all())
                or bool(valid_procedure[video, length:].any())
            ):
                raise WriterModelError("Procedure and memory video lengths differ")
            rows.append(
                self._read_video(
                    layer_memory[left:right],
                    procedure[video, :length],
                    positions[video, :length],
                    address,
                    core_slots[video],
                )
            )
        return torch.stack(rows)


class AddressPreservingVideoSet(torch.nn.Module):
    """Bound a permutation-invariant K-video consensus around its mean."""

    def __init__(self, *, max_relative_correction: float) -> None:
        super().__init__()
        if not 0.0 < max_relative_correction < 1.0:
            raise WriterModelError("invalid K-video set commitment")
        self.max_relative_correction = float(max_relative_correction)
        self.context_gate = torch.nn.Linear(
            2 * PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.centered = torch.nn.Sequential(
            RMSNorm(PROGRAM_WIDTH),
            torch.nn.Linear(PROGRAM_WIDTH, 2 * PROGRAM_WIDTH, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(2 * PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False),
        )
        self.shared_gate = torch.nn.Linear(
            2 * PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.correction = torch.nn.Sequential(
            RMSNorm(PROGRAM_WIDTH),
            torch.nn.Linear(PROGRAM_WIDTH, 2 * PROGRAM_WIDTH, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(2 * PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False),
        )
        self.commitment_logit = torch.nn.Parameter(torch.zeros(()))

    def bounded_commitment(
        self,
        anchor: torch.Tensor,
        proposal: torch.Tensor,
    ) -> torch.Tensor:
        """Keep every per-video mean cell dominant over the set correction."""

        if proposal.shape != anchor.shape:
            raise WriterModelError("K-video set proposal changed the addressed grid")
        anchor32 = anchor.to(torch.float32)
        delta32 = proposal.to(torch.float32) - anchor32
        anchor_rms = anchor32.square().mean(dim=-1, keepdim=True).sqrt()
        delta_rms = delta32.square().mean(dim=-1, keepdim=True).sqrt()
        scale = (anchor_rms / delta_rms.clamp_min(1e-6)).clamp(max=1.0)
        limited = delta32 * scale
        gate = self.max_relative_correction * torch.sigmoid(
            self.commitment_logit
        )
        return anchor + (gate * limited).to(anchor.dtype)

    def _condition(
        self,
        value: torch.Tensor,
        core_summary: torch.Tensor,
        procedure_summary: torch.Tensor,
    ) -> torch.Tensor:
        mean = value.mean(dim=0)
        if value.shape[0] == 1:
            return mean
        centered = value - mean
        context = torch.cat((core_summary, procedure_summary), dim=-1)
        gate = torch.tanh(self.context_gate(context))[:, None, None]
        per_video = centered + gate * self.centered(centered)
        correction = per_video.mean(dim=0)
        shared = torch.cat(
            (core_summary.mean(dim=0), procedure_summary.mean(dim=0)), dim=-1
        )
        shared_gate = torch.tanh(self.shared_gate(shared))
        proposal = mean + correction + shared_gate * self.correction(correction)
        return self.bounded_commitment(mean, proposal)

    def forward(
        self,
        value: torch.Tensor,
        core_summary: torch.Tensor,
        procedure_summary: torch.Tensor,
        condition_bounds: tuple[int, ...],
    ) -> torch.Tensor:
        videos = value.shape[0]
        if (
            value.ndim != 4
            or value.shape[1:]
            != (EXPERT_LAYERS, PUBLIC_RANK, PROGRAM_WIDTH)
            or core_summary.shape != (videos, PROGRAM_WIDTH)
            or procedure_summary.shape != (videos, PROGRAM_WIDTH)
            or len(condition_bounds) < 2
            or condition_bounds[0] != 0
            or condition_bounds[-1] != videos
            or any(
                right <= left or right - left > 4
                for left, right in zip(
                    condition_bounds[:-1], condition_bounds[1:], strict=True
                )
            )
        ):
            raise WriterModelError("invalid address-preserving video set")
        return torch.stack(
            [
                self._condition(
                    value[left:right],
                    core_summary[left:right],
                    procedure_summary[left:right],
                )
                for left, right in zip(
                    condition_bounds[:-1], condition_bounds[1:], strict=True
                )
            ]
        )


class DynamicCoreFusion(torch.nn.Module):
    """Inject task Core content only through nonzero directed video memory."""

    def __init__(self, *, heads: int, initialization_seed: int) -> None:
        super().__init__()
        if heads <= 0 or PROGRAM_WIDTH % heads:
            raise WriterModelError("invalid dynamic Core fusion")
        self.heads = int(heads)
        self.memory_norm = RMSNorm(PROGRAM_WIDTH)
        self.core_norm = RMSNorm(PROGRAM_WIDTH)
        self.query = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        self.key = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        self.value = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        self.output = torch.nn.Linear(PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False)
        self.dynamic_gate = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.core_projection = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.language_gate = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.action_in = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        self.action_out = torch.nn.Linear(
            PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 0x434F)
            self.layer_identity = torch.nn.Parameter(
                torch.empty(EXPERT_LAYERS, PROGRAM_WIDTH)
            )
            self.rank_identity = torch.nn.Parameter(
                torch.empty(PUBLIC_RANK, PROGRAM_WIDTH)
            )
            torch.nn.init.normal_(self.layer_identity, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.rank_identity, mean=0.0, std=0.02)

    def forward(
        self,
        memory: torch.Tensor,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        language_summary: torch.Tensor,
    ) -> torch.Tensor:
        batch = memory.shape[0]
        if (
            memory.ndim != 4
            or memory.shape[1:]
            != (EXPERT_LAYERS, PUBLIC_RANK, PROGRAM_WIDTH)
            or core.ndim != 3
            or core.shape[0] != batch
            or core.shape[-1] != PROGRAM_WIDTH
            or valid_core.shape != core.shape[:2]
            or valid_core.dtype != torch.bool
            or language_summary.shape != (batch, PROGRAM_WIDTH)
        ):
            raise WriterModelError("invalid dynamic Core batch")
        cells = EXPERT_LAYERS * PUBLIC_RANK
        address = (
            self.layer_identity[:, None] + self.rank_identity[None]
        ).reshape(cells, PROGRAM_WIDTH)
        flat = memory.reshape(batch, cells, PROGRAM_WIDTH)
        query = self.query(self.memory_norm(flat)) + address[None]
        normalized_core = self.core_norm(core)
        attended = F.scaled_dot_product_attention(
            _split_heads(query, self.heads),
            _split_heads(self.key(normalized_core), self.heads),
            _split_heads(self.value(core), self.heads),
            attn_mask=valid_core[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        core_value = self.core_projection(self.output(_merge_heads(attended)))
        dynamic = torch.tanh(self.dynamic_gate(self.memory_norm(flat)))
        language = torch.tanh(self.language_gate(language_summary))[:, None]
        fused = language * (flat + dynamic * core_value)
        fused = fused.reshape(
            batch, EXPERT_LAYERS, PUBLIC_RANK, PROGRAM_WIDTH
        )
        return torch.cat(
            (
                self.action_in(fused[:, :1]),
                fused,
                self.action_out(fused[:, -1:]),
            ),
            dim=1,
        )


class _AddressedAxialBlock(torch.nn.Module):
    """Communicate over policy groups and rank coordinates without static Values."""

    def __init__(self, *, heads: int) -> None:
        super().__init__()
        self.group_norm = RMSNorm(PROGRAM_WIDTH)
        self.group_attention = torch.nn.MultiheadAttention(
            PROGRAM_WIDTH,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.group_ffn_norm = RMSNorm(PROGRAM_WIDTH)
        self.group_ffn = torch.nn.Sequential(
            torch.nn.Linear(PROGRAM_WIDTH, 4 * PROGRAM_WIDTH, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False),
        )
        self.rank_norm = RMSNorm(PROGRAM_WIDTH)
        self.rank_attention = torch.nn.MultiheadAttention(
            PROGRAM_WIDTH,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.rank_ffn_norm = RMSNorm(PROGRAM_WIDTH)
        self.rank_ffn = torch.nn.Sequential(
            torch.nn.Linear(PROGRAM_WIDTH, 4 * PROGRAM_WIDTH, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * PROGRAM_WIDTH, PROGRAM_WIDTH, bias=False),
        )

    def forward(
        self,
        value: torch.Tensor,
        group_identity: torch.Tensor,
        rank_identity: torch.Tensor,
    ) -> torch.Tensor:
        batch = value.shape[0]
        by_group = value.permute(0, 2, 1, 3).reshape(
            batch * PUBLIC_RANK, PARAMETER_GROUPS, PROGRAM_WIDTH
        )
        group_address = (
            group_identity[None, :, :]
            + rank_identity[:, None, :]
        ).repeat(batch, 1, 1)
        addressed = self.group_norm(by_group) + group_address
        attended, _ = self.group_attention(
            addressed, addressed, by_group, need_weights=False
        )
        by_group = by_group + attended
        by_group = by_group + self.group_ffn(self.group_ffn_norm(by_group))
        value = by_group.reshape(
            batch, PUBLIC_RANK, PARAMETER_GROUPS, PROGRAM_WIDTH
        ).permute(0, 2, 1, 3)

        by_rank = value.reshape(
            batch * PARAMETER_GROUPS, PUBLIC_RANK, PROGRAM_WIDTH
        )
        rank_address = (
            group_identity[:, None, :]
            + rank_identity[None, :, :]
        ).repeat(batch, 1, 1)
        addressed = self.rank_norm(by_rank) + rank_address
        attended, _ = self.rank_attention(
            addressed, addressed, by_rank, need_weights=False
        )
        by_rank = by_rank + attended
        by_rank = by_rank + self.rank_ffn(self.rank_ffn_norm(by_rank))
        return by_rank.reshape(
            batch, PARAMETER_GROUPS, PUBLIC_RANK, PROGRAM_WIDTH
        )


class LayerMatchedMemoryProgramCompiler(torch.nn.Module):
    """Fuse Core and commit a bounded axial correction to the addressed grid."""

    def __init__(
        self,
        *,
        heads: int,
        blocks: int,
        max_relative_correction: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            heads <= 0
            or PROGRAM_WIDTH % heads
            or blocks <= 0
            or not 0.0 < max_relative_correction < 1.0
        ):
            raise WriterModelError("invalid layer-matched M2P topology")
        self.max_relative_correction = float(max_relative_correction)
        self.core_fusion = DynamicCoreFusion(
            heads=heads,
            initialization_seed=initialization_seed,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 0x4D3250)
            self.group_identity = torch.nn.Parameter(
                torch.empty(PARAMETER_GROUPS, PROGRAM_WIDTH)
            )
            self.rank_identity = torch.nn.Parameter(
                torch.empty(PUBLIC_RANK, PROGRAM_WIDTH)
            )
            torch.nn.init.normal_(self.group_identity, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.rank_identity, mean=0.0, std=0.02)
        self.blocks = torch.nn.ModuleList(
            _AddressedAxialBlock(heads=heads) for _ in range(blocks)
        )
        self.commitment_logit = torch.nn.Parameter(torch.zeros(()))
        self.output_norm = RMSNorm(PROGRAM_WIDTH)

    def bounded_commitment(
        self,
        anchor: torch.Tensor,
        proposal: torch.Tensor,
    ) -> torch.Tensor:
        """Keep every addressed dynamic cell dominant over its M2P correction."""

        if proposal.shape != anchor.shape:
            raise WriterModelError("M2P proposal changed the addressed grid")
        anchor32 = anchor.to(torch.float32)
        delta32 = proposal.to(torch.float32) - anchor32
        anchor_rms = anchor32.square().mean(dim=-1, keepdim=True).sqrt()
        delta_rms = delta32.square().mean(dim=-1, keepdim=True).sqrt()
        scale = (anchor_rms / delta_rms.clamp_min(1e-6)).clamp(max=1.0)
        limited = delta32 * scale
        gate = self.max_relative_correction * torch.sigmoid(
            self.commitment_logit
        )
        return anchor + (gate * limited).to(anchor.dtype)

    def forward(
        self,
        memory: torch.Tensor,
        core: torch.Tensor,
        valid_core: torch.Tensor,
        language_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.core_fusion(
            memory,
            core,
            valid_core,
            language_summary,
        )
        proposal = fused
        for block in self.blocks:
            proposal = block(
                proposal,
                self.group_identity,
                self.rank_identity,
            )
        committed = self.bounded_commitment(fused, proposal)
        return fused, self.output_norm(committed)
