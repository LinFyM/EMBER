"""Complete actual-factor edits with an exact zero-edit parameterization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX


@dataclass(frozen=True)
class _MatrixLayout:
    key: str
    target: int
    width: int
    transpose: bool

    @property
    def blocks(self) -> int:
        return (self.width + 63) // 64

    @property
    def rows(self) -> int:
        return 128 * self.blocks


class CoordinateDecoder(nn.Module):
    """Edit detached actual A rows/B columns using shared nonlinear block heads.

    Only scalar MT RMS units survive initialization. Each random trainable
    64-vector addresses one target/role/rank/block; it contains no MT values.
    Project delta_q once per target/rank, then produce a SiLU difference for
    each block. Its zero point retains incoming factors while its derivative
    with respect to delta_q is live. No separate deployed adapter is added.
    """

    rank = 128
    state_width = 256

    def __init__(self, mt_state: Mapping[str, torch.Tensor],
                 target_names: Sequence[str], *, seed: int = 20261009,
                 chunk_rows: int = 4096) -> None:
        super().__init__()
        self.target_names = tuple(target_names)
        if not self.target_names or len(set(self.target_names)) != len(self.target_names):
            raise ValueError("coordinate decoder requires distinct target names")
        if chunk_rows <= 0:
            raise ValueError("decode chunk_rows must be positive")
        self.chunk_rows = chunk_rows
        self._layouts = self._matrix_layouts(mt_state)
        self.register_buffer("matrix_rms", self._mt_scales(mt_state))
        indices = [(torch.arange(self.rank) + layout.target * self.rank).repeat_interleave(layout.blocks)
                   for layout in self._layouts]
        self.register_buffer("q_indices", torch.cat(indices), persistent=False)
        # Preserve the caller's CPU RNG, without seeding any CUDA generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.input_projection = nn.Linear(64, 512, device="cpu", dtype=torch.float32)
            self.address_projection = nn.Linear(64, 512, bias=False, device="cpu", dtype=torch.float32)
            self.edit_projection = nn.Linear(256, 512, bias=False, device="cpu", dtype=torch.float32)
            self.output_projection = nn.Linear(512, 64, bias=False, device="cpu", dtype=torch.float32)
            self.coordinate_embeddings = nn.Parameter(torch.empty(sum(layout.rows for layout in self._layouts),
                                                                   64, device="cpu", dtype=torch.float32))
            nn.init.normal_(self.coordinate_embeddings, std=0.02)

    def _matrix_layouts(self, mt_state: Mapping[str, torch.Tensor]) -> tuple[_MatrixLayout, ...]:
        result = []
        for target, name in enumerate(self.target_names):
            for suffix, transpose in ((LORA_A_SUFFIX, False), (LORA_B_SUFFIX, True)):
                key = name + suffix
                matrix = mt_state[key]
                rank_axis, width_axis = (1, 0) if transpose else (0, 1)
                if (matrix.ndim != 2 or not matrix.is_floating_point()
                        or matrix.shape[rank_axis] != self.rank or matrix.shape[width_axis] < 1):
                    raise ValueError(f"MT factor must have rank128 and a positive matrix width: {key}")
                result.append(_MatrixLayout(key, target, matrix.shape[width_axis], transpose))
        return tuple(result)

    def _mt_scales(self, mt_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # These mandatory scalar units are the only MT-derived decoder state.
        scales = [mt_state[layout.key].detach().to(device="cpu", dtype=torch.float32)
                  .square().mean().sqrt().clamp_min(1e-6) for layout in self._layouts]
        result = torch.stack(scales)
        if not torch.isfinite(result).all():
            raise ValueError("MT matrix RMS units must be finite")
        return result

    def _incoming_blocks(self, incoming_state, layout, device):
        matrix = incoming_state[layout.key].detach()
        expected = (layout.width, self.rank) if layout.transpose else (self.rank, layout.width)
        if tuple(matrix.shape) != expected or not matrix.is_floating_point():
            raise ValueError(f"incoming factor shape must be {expected}: {layout.key}")
        matrix = matrix.to(device=device)
        if layout.transpose:
            matrix = matrix.T
        return F.pad(matrix, (0, layout.blocks * 64 - layout.width)).reshape(-1, 64)

    def _decode_chunk(self, projected, coordinates, blocks, indices, scale):
        base = self.input_projection(blocks / scale) + self.address_projection(coordinates)
        edit = projected.index_select(0, indices)
        # Both SiLU branches share the exact base, parameters and precision.
        difference = F.silu(base + edit) - F.silu(base)
        return blocks + scale * self.output_projection(difference)

    def forward(self, incoming_state: Mapping[str, torch.Tensor], delta_q: torch.Tensor,
                *, chunk_rows: int | None = None) -> dict[str, torch.Tensor]:
        expected = (len(self.target_names), self.rank, self.state_width)
        if tuple(delta_q.shape) != expected or not delta_q.is_floating_point():
            raise ValueError(f"coordinate delta Q shape must be {expected}, found {tuple(delta_q.shape)}")
        if not torch.isfinite(delta_q).all():
            raise ValueError("coordinate delta Q must be finite")
        if set(incoming_state) != {layout.key for layout in self._layouts}:
            raise ValueError("incoming state must contain exactly the complete target A/B factors")
        rows = self.chunk_rows if chunk_rows is None else chunk_rows
        if rows <= 0:
            raise ValueError("decode chunk_rows must be positive")
        projected = self.edit_projection(delta_q).reshape(-1, 512)
        sizes = [layout.rows for layout in self._layouts]
        result = {}
        for index, (layout, coordinates, indices) in enumerate(zip(
                self._layouts, self.coordinate_embeddings.split(sizes), self.q_indices.split(sizes), strict=True)):
            blocks = self._incoming_blocks(incoming_state, layout, delta_q.device)
            chunks = []
            for address, values, token_ids in zip(coordinates.split(rows), blocks.split(rows),
                                                 indices.split(rows), strict=True):
                inputs = (projected, address, values, token_ids, self.matrix_rms[index])
                chunks.append(checkpoint(self._decode_chunk, *inputs, use_reentrant=False, preserve_rng_state=False)
                              if torch.is_grad_enabled() else self._decode_chunk(*inputs))
            matrix = torch.cat(chunks).reshape(self.rank, layout.blocks * 64)[:, :layout.width]
            result[layout.key] = (matrix.T if layout.transpose else matrix).contiguous()
        return result
