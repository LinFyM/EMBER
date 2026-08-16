"""Capacity-matched backbone-memory grid for native-zero LoRA-B residuals."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from ember.writer.errors import WriterModelError
from ember.writer.temporal import CausalProcedureEncoder, RMSNorm


NATIVE_B_FAMILIES = ("q_b", "v_b", "action_in_b", "action_out_b")


class _CausalGridReducer(torch.nn.Module):
    """Pool one ordered video with a single causal controller and per-cell weights."""

    LAYERS = 18
    PAYLOAD_TOKENS = 37
    WIDTH = 1024
    CONTROLLER_WIDTH = 256

    def __init__(self) -> None:
        super().__init__()
        self.summary = torch.nn.Linear(
            self.WIDTH, self.CONTROLLER_WIDTH, bias=False
        )
        self.temporal = CausalProcedureEncoder(
            width=self.CONTROLLER_WIDTH,
            heads=8,
            blocks=1,
        )
        self.context_norm = RMSNorm(self.CONTROLLER_WIDTH)
        self.delta_norm = RMSNorm(self.WIDTH)
        self.delta_key = torch.nn.Linear(
            self.WIDTH, self.CONTROLLER_WIDTH, bias=False
        )

    def _video(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if (
            value.ndim != 4
            or value.shape[1:]
            != (self.LAYERS, self.PAYLOAD_TOKENS, self.WIDTH)
            or positions.shape != value.shape[:1]
            or positions.dtype != torch.long
        ):
            raise WriterModelError("invalid ordered parameter-grid video")
        delta = torch.cat(
            (torch.zeros_like(value[:1]), value[1:] - value[:-1]), dim=0
        )
        summary = self.summary(delta.mean(dim=(1, 2)))[None]
        valid = torch.ones(
            1, value.shape[0], dtype=torch.bool, device=value.device
        )
        contextual = self.temporal(summary, positions[None], valid)[0, -1]
        context = self.context_norm(contextual)
        key = self.delta_key(self.delta_norm(delta))
        logits = torch.einsum("tlmw,w->tlm", key, context)
        logits = logits * (self.CONTROLLER_WIDTH**-0.5)
        weights = torch.softmax(logits.to(torch.float32), dim=0).to(value.dtype)
        transition = (weights[..., None] * delta).sum(dim=0)
        return transition + value[-1] - value[0]

    def forward(
        self,
        value: torch.Tensor,
        frame_indices: torch.Tensor,
        video_bounds: tuple[int, ...],
    ) -> torch.Tensor:
        if (
            frame_indices.shape != value.shape[:1]
            or frame_indices.dtype != torch.long
            or frame_indices.device != value.device
            or len(video_bounds) < 2
            or video_bounds[0] != 0
            or video_bounds[-1] != value.shape[0]
            or any(
                right <= left for left, right in zip(video_bounds, video_bounds[1:])
            )
        ):
            raise WriterModelError("invalid parameter-grid video offsets")
        return torch.stack(
            [
                self._video(value[left:right], frame_indices[left:right])
                for left, right in zip(video_bounds[:-1], video_bounds[1:], strict=True)
            ]
        )


class _PayloadAxisMixer(torch.nn.Module):
    """Mix one explicit topology axis while keeping native-width Values."""

    WIDTH = 1024
    ROUTE_WIDTH = 128

    def __init__(self, length: int) -> None:
        super().__init__()
        if length <= 0:
            raise WriterModelError("invalid parameter-grid axis length")
        self.length = int(length)
        self.norm = RMSNorm(self.WIDTH)
        self.query = torch.nn.Linear(self.WIDTH, self.ROUTE_WIDTH, bias=False)
        self.key = torch.nn.Linear(self.WIDTH, self.ROUTE_WIDTH, bias=False)
        self.route = torch.nn.Parameter(torch.empty(length, self.ROUTE_WIDTH))
        torch.nn.init.normal_(self.route, mean=0.0, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (self.length, self.WIDTH):
            raise WriterModelError("parameter-grid axis changed shape")
        normalized = self.norm(value)
        route = self.route[None]
        query = self.query(normalized) + route
        key = self.key(normalized) + route
        logits = torch.einsum("bir,bjr->bij", query, key)
        logits = logits * (self.ROUTE_WIDTH**-0.5)
        weights = torch.softmax(logits.to(torch.float32), dim=-1).to(value.dtype)
        return value + torch.einsum("bij,bje->bie", weights, value)


class _VideoSetMixer(torch.nn.Module):
    """Permutation-equivariant video attention followed by symmetric pooling."""

    WIDTH = 1024
    ROUTE_WIDTH = 128

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(self.WIDTH)
        self.query = torch.nn.Linear(self.WIDTH, self.ROUTE_WIDTH, bias=False)
        self.key = torch.nn.Linear(self.WIDTH, self.ROUTE_WIDTH, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if (
            value.ndim != 3
            or value.shape[-1] != self.WIDTH
            or value.shape[1] not in range(1, 5)
        ):
            raise WriterModelError("invalid CMBG video set")
        normalized = self.norm(value)
        query = self.query(normalized)
        key = self.key(normalized)
        logits = torch.einsum("bir,bjr->bij", query, key)
        logits = logits * (self.ROUTE_WIDTH**-0.5)
        weights = torch.softmax(logits.to(torch.float32), dim=-1).to(value.dtype)
        mixed = value + torch.einsum("bij,bje->bie", weights, value)
        return mixed.mean(dim=1)


class _ParameterGridBranch(torch.nn.Module):
    """Zero-gated memory, directed video pooling, set fusion, and M2P."""

    LAYERS = 18
    PAYLOAD_TOKENS = 37
    WIDTH = 1024

    def __init__(self) -> None:
        super().__init__()
        memory = torch.empty(self.PAYLOAD_TOKENS, self.WIDTH)
        torch.nn.init.normal_(memory, mean=0.0, std=self.WIDTH**-0.5)
        self.memory_tokens = torch.nn.Parameter(memory)
        self.payload_gate = torch.nn.Parameter(
            torch.zeros(self.LAYERS, self.PAYLOAD_TOKENS, self.WIDTH)
        )
        self.temporal = _CausalGridReducer()
        self.video_set = _VideoSetMixer()
        self.layer_axis = _PayloadAxisMixer(self.LAYERS)
        self.token_axis = _PayloadAxisMixer(self.PAYLOAD_TOKENS)

    def _aggregate_condition(self, videos: torch.Tensor) -> torch.Tensor:
        shots = videos.shape[0]
        if shots not in range(1, 5):
            raise WriterModelError("CMBG supports one to four videos")
        cells = videos.permute(1, 2, 0, 3).reshape(
            self.LAYERS * self.PAYLOAD_TOKENS, shots, self.WIDTH
        )
        return self.video_set(cells).reshape(
            self.LAYERS, self.PAYLOAD_TOKENS, self.WIDTH
        )

    def forward(
        self,
        layer_memory: torch.Tensor,
        frame_indices: torch.Tensor,
        video_bounds: tuple[int, ...],
        condition_bounds: tuple[int, ...],
    ) -> torch.Tensor:
        expected = (self.LAYERS, self.PAYLOAD_TOKENS, self.WIDTH)
        if layer_memory.ndim != 4 or layer_memory.shape[1:] != expected:
            raise WriterModelError("backbone-memory parameter grid changed shape")
        frame_grid = layer_memory * self.payload_gate
        video_grid = self.temporal(frame_grid, frame_indices, video_bounds)
        if (
            len(condition_bounds) < 2
            or condition_bounds[0] != 0
            or condition_bounds[-1] != video_grid.shape[0]
            or any(
                right <= left or right - left > 4
                for left, right in zip(
                    condition_bounds[:-1], condition_bounds[1:], strict=True
                )
            )
        ):
            raise WriterModelError("invalid CMBG condition offsets")
        shared = torch.stack(
            [
                self._aggregate_condition(video_grid[left:right])
                for left, right in zip(
                    condition_bounds[:-1], condition_bounds[1:], strict=True
                )
            ]
        )
        batch = shared.shape[0]
        layer_mixed = self.layer_axis(
            shared.permute(0, 2, 1, 3).reshape(
                batch * self.PAYLOAD_TOKENS, self.LAYERS, self.WIDTH
            )
        ).reshape(batch, self.PAYLOAD_TOKENS, self.LAYERS, self.WIDTH)
        layer_mixed = layer_mixed.permute(0, 2, 1, 3)
        return self.token_axis(
            layer_mixed.reshape(
                batch * self.LAYERS, self.PAYLOAD_TOKENS, self.WIDTH
            )
        ).reshape(batch, self.LAYERS, self.PAYLOAD_TOKENS, self.WIDTH)


class CapacityMatchedBackboneMemoryGrid(torch.nn.Module):
    """Emit one backbone-conditioned native-zero B residual bank."""

    LAYERS = 18
    PAYLOAD_TOKENS = 37
    WIDTH = 1024
    PAYLOAD_VALUES = 18 * 37 * 1024
    B_VALUES = 680_448
    UNUSED_VALUES = PAYLOAD_VALUES - B_VALUES

    def __init__(self, *, initialization_seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed) + 0xCA9)
            self.branch = _ParameterGridBranch()

    @staticmethod
    def rows(grid: torch.Tensor) -> Mapping[str, torch.Tensor]:
        expected = (18, 37, 1024)
        if grid.ndim != 4 or grid.shape[1:] != expected:
            raise WriterModelError("capacity-matched output grid changed shape")
        batch = grid.shape[0]
        q_b = grid[:, :, :32].reshape(batch, 18, 16, 2048)
        v_b = grid[:, :, 32:36].reshape(batch, 18, 16, 256)
        action_in_b = grid[:, :16, 36].reshape(batch, 16, 1024)
        action_out_b = grid[:, 16, 36, :512].reshape(batch, 16, 32)
        return {
            "q_b": q_b,
            "v_b": v_b,
            "action_in_b": action_in_b,
            "action_out_b": action_out_b,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        frame_indices: torch.Tensor,
        video_bounds: tuple[int, ...],
        condition_bounds: tuple[int, ...],
    ) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
        delta = self.branch(
            hidden, frame_indices, video_bounds, condition_bounds
        ) / math.sqrt(self.WIDTH)
        return self.rows(delta), delta
