"""Trainable MT coordinates decoded directly into one complete rank128 LoRA."""

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
    width: int
    transpose: bool

    @property
    def blocks(self) -> int:
        return (self.width + 63) // 64

    @property
    def rows(self) -> int:
        return 128 * self.blocks


class CoordinateDecoder(nn.Module):
    """Decode [target, rank, 256] states using shared nonlinear coordinate heads.

    MT is copied into the first half of trainable 128-dimensional coordinates,
    with no retained MT skip or buffer. A coordinates run along input columns;
    B coordinates run along output rows after transposing to [rank, output].
    Every output block uses the same 384 -> 512 -> 64 network. Chunking and
    activation recomputation preserve that network and its complete gradients.
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
        indices = [
            (torch.arange(self.rank, device="cpu") + target * self.rank).repeat_interleave(layout.blocks)
            for target in range(len(self.target_names))
            for layout in self._layouts[2 * target:2 * target + 2]
        ]
        self.register_buffer("q_indices", torch.cat(indices), persistent=False)
        # Seed only the CPU generator: torch.manual_seed would also alter CUDA
        # generators, which are intentionally outside this CPU initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.q_norm = nn.LayerNorm(self.state_width, device="cpu", dtype=torch.float32)
            self.input_projection = nn.Linear(384, 512, device="cpu", dtype=torch.float32)
            self.output_projection = nn.Linear(512, 64, device="cpu", dtype=torch.float32)
            self.coordinate_embeddings = nn.Parameter(self._coordinates(mt_state))
            self._initialize_identity_heads()

    def _matrix_layouts(self, mt_state: Mapping[str, torch.Tensor]) -> tuple[_MatrixLayout, ...]:
        result = []
        for name in self.target_names:
            for suffix, transpose in ((LORA_A_SUFFIX, False), (LORA_B_SUFFIX, True)):
                key = name + suffix
                matrix = mt_state[key]
                rank_axis, width_axis = (1, 0) if transpose else (0, 1)
                if (matrix.ndim != 2 or not matrix.is_floating_point()
                        or matrix.shape[rank_axis] != self.rank or matrix.shape[width_axis] < 1):
                    raise ValueError(f"MT factor must have rank128 and a positive matrix width: {key}")
                result.append(_MatrixLayout(key, matrix.shape[width_axis], transpose))
        return tuple(result)

    def _coordinates(self, mt_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        result = torch.empty(sum(layout.rows for layout in self._layouts), 128,
                             device="cpu", dtype=torch.float32)
        nn.init.normal_(result[:, 64:], std=0.02)
        for layout, coordinates in zip(self._layouts, result.split(
                [layout.rows for layout in self._layouts]), strict=True):
            matrix = mt_state[layout.key].detach().to(device="cpu", dtype=torch.float32)
            if layout.transpose:
                matrix = matrix.T
            blocks = F.pad(matrix, (0, layout.blocks * 64 - layout.width)).reshape(-1, 64)
            coordinates[:, :64].copy_(blocks)
        return result

    @torch.no_grad()
    def _initialize_identity_heads(self) -> None:
        # SiLU(x) - SiLU(-x) = x. Other hidden units retain their ordinary
        # initialization and can acquire output weight on the first update.
        self.input_projection.weight[:128].zero_()
        self.input_projection.bias[:128].zero_()
        identity = torch.eye(64, device="cpu", dtype=torch.float32)
        self.input_projection.weight[:64, 256:320].copy_(identity)
        self.input_projection.weight[64:128, 256:320].copy_(-identity)
        self.output_projection.weight.zero_()
        self.output_projection.weight[:, :64].copy_(identity)
        self.output_projection.weight[:, 64:128].copy_(-identity)
        self.output_projection.bias.zero_()

    def _decode_chunk(self, q: torch.Tensor, coordinates: torch.Tensor,
                      indices: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((q.index_select(0, indices), coordinates), dim=-1)
        return self.output_projection(F.silu(self.input_projection(inputs)))

    def forward(self, q: torch.Tensor, *, chunk_rows: int | None = None) -> dict[str, torch.Tensor]:
        expected = (len(self.target_names), self.rank, self.state_width)
        if tuple(q.shape) != expected:
            raise ValueError(f"coordinate Q shape must be {expected}, found {tuple(q.shape)}")
        rows = self.chunk_rows if chunk_rows is None else chunk_rows
        if rows <= 0:
            raise ValueError("decode chunk_rows must be positive")
        normalized_q = self.q_norm(q).reshape(-1, self.state_width)
        chunks = []
        # split shares one backward assembly instead of scattering a full-size
        # coordinate gradient independently for every sliced chunk.
        for coordinates, indices in zip(self.coordinate_embeddings.split(rows),
                                        self.q_indices.split(rows), strict=True):
            chunks.append(checkpoint(self._decode_chunk, normalized_q, coordinates, indices,
                                     use_reentrant=False, preserve_rng_state=False)
                          if torch.is_grad_enabled()
                          else self._decode_chunk(normalized_q, coordinates, indices))
        decoded = torch.cat(chunks).split([layout.rows for layout in self._layouts])
        result = {}
        for layout, values in zip(self._layouts, decoded, strict=True):
            matrix = values.reshape(self.rank, layout.blocks * 64)[:, :layout.width]
            result[layout.key] = (matrix.T if layout.transpose else matrix).contiguous()
        return result
