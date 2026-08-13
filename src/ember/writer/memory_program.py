"""Directed Dynamic-K memory programs and layer/rank communication."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ember.writer.temporal import RMSNorm


class MemoryProgramError(ValueError):
    """Raised when a ragged backbone-memory batch violates its contract."""


@dataclass(frozen=True)
class MemoryProgramDiagnostics:
    """Live intermediate states for representation-to-mapper diagnostics."""

    semantic_addresses: torch.Tensor
    visual_transition_readouts: torch.Tensor
    visual_goal_readouts: torch.Tensor
    video_programs: torch.Tensor
    shared_program: torch.Tensor
    singleton_program: torch.Tensor
    m2p_input: torch.Tensor
    layer_axis_program: torch.Tensor
    consistency_loss: torch.Tensor


def _split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    batch, tokens, width = value.shape
    return value.reshape(batch, tokens, heads, width // heads).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    return value.transpose(1, 2).reshape(batch, tokens, heads * width)


def _apply_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    width = value.shape[-1]
    inverse_frequency = torch.exp(
        torch.arange(0, width, 2, device=value.device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    angles = (
        positions.to(torch.float32)[:, None, :, None]
        * inverse_frequency[None, None, None]
    )
    cosine = torch.cos(angles).to(value.dtype)
    sine = torch.sin(angles).to(value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


class _ZeroPreservingSelfAttention(torch.nn.Module):
    """Route through Q/K while keeping every dynamic value path homogeneous."""

    def __init__(self, *, width: int, heads: int, causal: bool, rotary: bool) -> None:
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.rotary = rotary
        self.qk_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(4 * width, width, bias=False),
        )

    def forward(
        self,
        content: torch.Tensor,
        route: torch.Tensor,
        positions: torch.Tensor | None = None,
        query_address: torch.Tensor | None = None,
    ) -> torch.Tensor:
        addressed = self.qk_norm(content) + route
        if query_address is not None and query_address.shape != content.shape:
            raise MemoryProgramError("query address changed shape")
        query = _split_heads(
            self.query(
                addressed if query_address is None else addressed + query_address
            ),
            self.heads,
        )
        key = _split_heads(self.key(addressed), self.heads)
        if self.rotary:
            if positions is None or positions.shape != content.shape[:2]:
                raise MemoryProgramError("temporal positions changed shape")
            query = _apply_rope(query, positions)
            key = _apply_rope(key, positions)
        elif positions is not None:
            raise MemoryProgramError("non-temporal attention received positions")
        value = _split_heads(self.value(content), self.heads)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=self.causal,
        )
        content = content + self.output(_merge_heads(attended))
        return content + self.ffn(content)


class _ZeroPreservingCrossAttention(torch.nn.Module):
    """Let parameter queries read dynamic content without becoming values."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.memory_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.value = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query_route: torch.Tensor,
        memory: torch.Tensor,
        memory_route: torch.Tensor,
    ) -> torch.Tensor:
        query = _split_heads(self.query(query_route), self.heads)
        key = _split_heads(
            self.key(self.memory_norm(memory) + memory_route), self.heads
        )
        value = _split_heads(self.value(memory), self.heads)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        return self.output(_merge_heads(attended))


class _TaskGroundedVisualReader(torch.nn.Module):
    """Let memory cells read raw task-grounded visual change as Value."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query_norm = RMSNorm(width)
        self.evidence_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        query_content: torch.Tensor,
        query_route: torch.Tensor,
        evidence: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query_content.ndim != 2
            or query_route.ndim != 2
            or query_content.shape != query_route.shape
            or evidence.ndim != 3
            or query_content.shape[-1] != evidence.shape[-1]
            or valid_tokens.shape != evidence.shape[:2]
            or valid_tokens.dtype != torch.bool
        ):
            raise MemoryProgramError("task-grounded visual reader changed shape")
        frames = evidence.shape[0]
        query = _split_heads(
            self.query(self.query_norm(query_content) + query_route)[None].expand(
                frames, -1, -1
            ),
            self.heads,
        )
        key = _split_heads(self.key(self.evidence_norm(evidence)), self.heads)
        value = _split_heads(evidence, self.heads)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid_tokens[:, None, None],
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(_merge_heads(attended))


class DynamicKMemoryProgram(torch.nn.Module):
    """Compile ragged per-frame backbone memories into a ``20 x 8`` program."""

    EXPERT_LAYERS = 18
    RANK_TOKENS = 8
    MEMORY_WIDTH = 1024
    PROGRAM_WIDTH = 256
    POLICY_GROUPS = 20

    def __init__(self, *, heads: int = 8) -> None:
        super().__init__()
        if heads <= 0 or self.PROGRAM_WIDTH % heads:
            raise MemoryProgramError("invalid memory-program attention heads")
        self.dynamic_projection = torch.nn.Linear(
            self.MEMORY_WIDTH, self.PROGRAM_WIDTH, bias=False
        )
        self.semantic_address_norm = RMSNorm(self.MEMORY_WIDTH)
        self.semantic_address_projection = torch.nn.Linear(
            self.MEMORY_WIDTH, self.PROGRAM_WIDTH, bias=False
        )
        self.goal_fusion = torch.nn.Linear(
            2 * self.PROGRAM_WIDTH, self.PROGRAM_WIDTH, bias=False
        )
        self.visual_reader = _TaskGroundedVisualReader(
            width=self.PROGRAM_WIDTH, heads=heads
        )
        self.temporal_blocks = torch.nn.ModuleList(
            _ZeroPreservingSelfAttention(
                width=self.PROGRAM_WIDTH,
                heads=heads,
                causal=True,
                rotary=True,
            )
            for _ in range(2)
        )
        self.set_blocks = torch.nn.ModuleList(
            _ZeroPreservingSelfAttention(
                width=self.PROGRAM_WIDTH,
                heads=heads,
                causal=False,
                rotary=False,
            )
            for _ in range(2)
        )
        self.endpoint_reader = _ZeroPreservingCrossAttention(
            width=self.PROGRAM_WIDTH, heads=heads
        )
        self.layer_axis = _ZeroPreservingSelfAttention(
            width=self.PROGRAM_WIDTH,
            heads=heads,
            causal=False,
            rotary=False,
        )
        self.rank_axis = _ZeroPreservingSelfAttention(
            width=self.PROGRAM_WIDTH,
            heads=heads,
            causal=False,
            rotary=False,
        )
        self.expert_layer_identity = torch.nn.Parameter(
            torch.empty(self.EXPERT_LAYERS, self.PROGRAM_WIDTH)
        )
        self.endpoint_group_identity = torch.nn.Parameter(
            torch.empty(2, self.PROGRAM_WIDTH)
        )
        self.rank_identity = torch.nn.Parameter(
            torch.empty(self.RANK_TOKENS, self.PROGRAM_WIDTH)
        )
        self.endpoint_queries = torch.nn.Parameter(
            torch.empty(2, self.RANK_TOKENS, self.PROGRAM_WIDTH)
        )
        for parameter in (
            self.expert_layer_identity,
            self.endpoint_group_identity,
            self.rank_identity,
            self.endpoint_queries,
        ):
            torch.nn.init.normal_(parameter, mean=0.0, std=0.02)

    def _cell_route(self) -> torch.Tensor:
        return (
            self.expert_layer_identity[:, None] + self.rank_identity[None]
        ).reshape(self.EXPERT_LAYERS * self.RANK_TOKENS, self.PROGRAM_WIDTH)

    @staticmethod
    def _offsets(value: torch.Tensor, *, final: int, name: str) -> tuple[int, ...]:
        if (
            value.device.type != "cpu"
            or value.ndim != 1
            or value.dtype != torch.long
        ):
            raise MemoryProgramError(f"{name} must be a one-dimensional long tensor")
        offsets = tuple(int(item) for item in value.tolist())
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != final
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise MemoryProgramError(f"invalid {name}")
        return offsets

    def _encode_video(
        self,
        memory: torch.Tensor,
        visual_evidence: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacent = torch.zeros_like(memory)
        adjacent[1:] = memory[1:] - memory[:-1]
        goal = memory - memory[:1]
        semantic_address = self.semantic_address_projection(
            self.semantic_address_norm(memory.mean(dim=0))
        )
        visual_transition = torch.zeros_like(visual_evidence)
        visual_transition[1:] = visual_evidence[1:] - visual_evidence[:-1]
        visual_goal = visual_evidence[-1:] - visual_evidence[:1]
        cell_query = semantic_address.reshape(
            self.EXPERT_LAYERS * self.RANK_TOKENS,
            self.PROGRAM_WIDTH,
        )
        cell_route = self._cell_route()
        visual_transition_readout = self.visual_reader(
            cell_query,
            cell_route,
            visual_transition,
            valid_task_tokens,
        ).reshape(
            memory.shape[0],
            self.EXPERT_LAYERS,
            self.RANK_TOKENS,
            self.PROGRAM_WIDTH,
        )
        visual_goal_readout = self.visual_reader(
            cell_query,
            cell_route,
            visual_goal,
            valid_task_tokens[-1:] & valid_task_tokens[:1],
        )[0].reshape(
            self.EXPERT_LAYERS,
            self.RANK_TOKENS,
            self.PROGRAM_WIDTH,
        )
        transition = self.dynamic_projection(adjacent) + visual_transition_readout
        terminal_goal = self.dynamic_projection(goal[-1]) + visual_goal_readout
        frames = memory.shape[0]
        content = transition.permute(1, 2, 0, 3).reshape(
            self.EXPERT_LAYERS * self.RANK_TOKENS,
            frames,
            self.PROGRAM_WIDTH,
        )
        route = self._cell_route()[:, None]
        query_address = semantic_address.reshape(
            self.EXPERT_LAYERS * self.RANK_TOKENS,
            1,
            self.PROGRAM_WIDTH,
        ).expand(-1, frames, -1)
        ordinals = positions[None].expand(content.shape[0], -1)
        for block in self.temporal_blocks:
            content = block(
                content,
                route,
                ordinals,
                query_address=query_address,
            )
        terminal = content[:, -1].reshape(
            self.EXPERT_LAYERS, self.RANK_TOKENS, self.PROGRAM_WIDTH
        )
        return (
            self.goal_fusion(torch.cat((terminal, terminal_goal), dim=-1)),
            semantic_address,
            visual_transition_readout,
            visual_goal_readout,
        )

    @staticmethod
    def _validate_frame_order(
        frame_indices: torch.Tensor,
        video_bounds: tuple[int, ...],
    ) -> None:
        starts = torch.tensor(
            video_bounds[:-1], dtype=torch.long, device=frame_indices.device
        )
        internal = torch.ones(
            frame_indices.shape[0] - 1,
            dtype=torch.bool,
            device=frame_indices.device,
        )
        if len(video_bounds) > 2:
            internal[
                torch.tensor(
                    video_bounds[1:-1],
                    dtype=torch.long,
                    device=frame_indices.device,
                )
                - 1
            ] = False
        invalid_ordinals = (
            (frame_indices.index_select(0, starts) != 0).any()
            | ((frame_indices[1:] <= frame_indices[:-1]) & internal).any()
        )
        if bool(invalid_ordinals):
            raise MemoryProgramError(
                "each video's frame indices must start at zero and increase"
            )

    def _aggregate_set(self, programs: torch.Tensor) -> torch.Tensor:
        shots = programs.shape[0]
        content = programs.permute(1, 2, 0, 3).reshape(
            self.EXPERT_LAYERS * self.RANK_TOKENS,
            shots,
            self.PROGRAM_WIDTH,
        )
        route = self._cell_route()[:, None]
        for block in self.set_blocks:
            content = block(content, route)
        return content.mean(dim=1).reshape(
            self.EXPERT_LAYERS, self.RANK_TOKENS, self.PROGRAM_WIDTH
        )

    def _m2p(
        self, shared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = shared.shape[0]
        endpoint_memory = torch.stack((shared[:, 0], shared[:, -1]), dim=1)
        endpoint_memory = endpoint_memory.reshape(
            2 * batch, self.RANK_TOKENS, self.PROGRAM_WIDTH
        )
        endpoint_route = (
            self.endpoint_queries
            + self.endpoint_group_identity[:, None]
            + self.rank_identity[None]
        )[None].expand(batch, -1, -1, -1).reshape_as(endpoint_memory)
        source_route = torch.stack(
            (self.expert_layer_identity[0], self.expert_layer_identity[-1])
        )[:, None] + self.rank_identity[None]
        source_route = source_route[None].expand(batch, -1, -1, -1).reshape_as(
            endpoint_memory
        )
        endpoints = self.endpoint_reader(
            endpoint_route, endpoint_memory, source_route
        ).reshape(batch, 2, self.RANK_TOKENS, self.PROGRAM_WIDTH)
        m2p_input = torch.cat(
            (endpoints[:, :1], shared, endpoints[:, 1:]), dim=1
        )
        group_identity = torch.cat(
            (
                self.endpoint_group_identity[:1],
                self.expert_layer_identity,
                self.endpoint_group_identity[1:],
            )
        )
        route = group_identity[:, None] + self.rank_identity[None]
        layer_content = m2p_input.permute(0, 2, 1, 3).reshape(
            batch * self.RANK_TOKENS,
            self.POLICY_GROUPS,
            self.PROGRAM_WIDTH,
        )
        layer_route = route.permute(1, 0, 2)[None].expand(batch, -1, -1, -1)
        layer_content = self.layer_axis(
            layer_content,
            layer_route.reshape_as(layer_content),
        )
        layer_program = layer_content.reshape(
            batch,
            self.RANK_TOKENS,
            self.POLICY_GROUPS,
            self.PROGRAM_WIDTH,
        ).permute(0, 2, 1, 3)
        rank_content = layer_program.reshape(
            batch * self.POLICY_GROUPS,
            self.RANK_TOKENS,
            self.PROGRAM_WIDTH,
        )
        rank_route = route[None].expand(batch, -1, -1, -1).reshape_as(rank_content)
        output = self.rank_axis(rank_content, rank_route).reshape_as(layer_program)
        return output, m2p_input, layer_program

    def forward(
        self,
        memory_states: torch.Tensor,
        visual_evidence: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        *,
        singleton_video_index: int = 0,
    ) -> tuple[torch.Tensor, MemoryProgramDiagnostics]:
        """Return the invariant M2P program and its representation diagnostics."""

        expected_tail = (
            self.EXPERT_LAYERS,
            self.RANK_TOKENS,
            self.MEMORY_WIDTH,
        )
        if memory_states.ndim != 4 or memory_states.shape[1:] != expected_tail:
            raise MemoryProgramError("memory states changed shape")
        if (
            visual_evidence.ndim != 3
            or visual_evidence.shape[0] != memory_states.shape[0]
            or visual_evidence.shape[-1] != self.PROGRAM_WIDTH
            or valid_task_tokens.shape != visual_evidence.shape[:2]
            or valid_task_tokens.dtype != torch.bool
        ):
            raise MemoryProgramError("task-grounded visual evidence changed shape")
        if (
            frame_indices.shape != memory_states.shape[:1]
            or frame_indices.dtype != torch.long
        ):
            raise MemoryProgramError("frame indices changed shape")
        video_bounds = self._offsets(
            video_offsets, final=memory_states.shape[0], name="video offsets"
        )
        condition_bounds = self._offsets(
            condition_video_offsets,
            final=len(video_bounds) - 1,
            name="condition video offsets",
        )
        cardinalities = tuple(
            right - left for left, right in zip(condition_bounds, condition_bounds[1:])
        )
        if any(count > 4 for count in cardinalities) or any(
            singleton_video_index not in range(count) for count in cardinalities
        ):
            raise MemoryProgramError("dynamic-K cardinality must stay within 1..4")
        self._validate_frame_order(frame_indices, video_bounds)
        positions = frame_indices.to(device=memory_states.device)
        encoded_videos = tuple(
            self._encode_video(
                memory_states[start:stop],
                visual_evidence[start:stop],
                valid_task_tokens[start:stop],
                positions[start:stop],
            )
            for start, stop in zip(video_bounds, video_bounds[1:])
        )
        video_programs = torch.stack(tuple(item[0] for item in encoded_videos))
        semantic_addresses = torch.stack(tuple(item[1] for item in encoded_videos))
        visual_transition_readouts = torch.cat(
            tuple(item[2] for item in encoded_videos)
        )
        visual_goal_readouts = torch.stack(tuple(item[3] for item in encoded_videos))
        shared_programs = []
        singleton_programs = []
        multi_condition = []
        for start, stop in zip(condition_bounds, condition_bounds[1:]):
            selected = video_programs[start:stop]
            shared = self._aggregate_set(selected)
            singleton = (
                shared
                if stop - start == 1
                else self._aggregate_set(
                    selected[singleton_video_index : singleton_video_index + 1]
                )
            )
            shared_programs.append(shared)
            singleton_programs.append(singleton)
            multi_condition.append(stop - start > 1)
        shared_program = torch.stack(shared_programs)
        singleton_program = torch.stack(singleton_programs)
        multi_mask = torch.tensor(
            multi_condition, dtype=torch.bool, device=shared_program.device
        )
        if bool(multi_mask.any()):
            consistency_loss = F.smooth_l1_loss(
                singleton_program[multi_mask], shared_program[multi_mask].detach()
            )
        else:
            consistency_loss = singleton_program.sum() * 0.0
        output, m2p_input, layer_program = self._m2p(shared_program)
        diagnostics = MemoryProgramDiagnostics(
            semantic_addresses=semantic_addresses,
            visual_transition_readouts=visual_transition_readouts,
            visual_goal_readouts=visual_goal_readouts,
            video_programs=video_programs,
            shared_program=shared_program,
            singleton_program=singleton_program,
            m2p_input=m2p_input,
            layer_axis_program=layer_program,
            consistency_loss=consistency_loss,
        )
        return output, diagnostics
