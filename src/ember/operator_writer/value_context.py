"""One shared temporal Value context, independently at each native horizon slot."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ValueContext(nn.Module):
    """Bidirectional real-frame attention; no horizon/video aggregation or FFN."""

    def __init__(self) -> None:
        super().__init__()
        self.wh = nn.Linear(1024, 256, bias=False, device="cpu")
        self.q = nn.Linear(256, 256, bias=False, device="cpu")
        self.k = nn.Linear(256, 256, bias=False, device="cpu")
        self.v = nn.Linear(256, 256, bias=False, device="cpu")
        self.out = nn.Linear(256, 256, bias=False, device="cpu")
        self.register_buffer("inv_frequency", 10000. ** (-torch.arange(0, 64, 2).float() / 64),
                             persistent=False)

    @staticmethod
    def _rotate(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(self, h: torch.Tensor, frame_indices) -> torch.Tensor:
        if h.ndim != 3 or h.shape[1:] != (50, 1024) or not len(h) or frame_indices is None:
            raise ValueError("Value context requires actual N-frame native H and frame indices")
        indices = torch.as_tensor(frame_indices)
        if (indices.shape != (len(h),) or indices.dtype == torch.bool
                or indices.is_floating_point() or indices.is_complex()
                or torch.any(indices < 0) or torch.any(indices[1:] <= indices[:-1])):
            raise ValueError("Value context requires ordered real integer frame indices")
        positions = indices.to(device=h.device, dtype=torch.float32) / 5
        normalized = h.float() * torch.rsqrt(h.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        projected = self.wh(normalized)
        def heads(layer):
            return layer(projected).reshape(len(h), 50, 4, 64).permute(1, 2, 0, 3)
        q, k, v = heads(self.q), heads(self.k), heads(self.v)
        angles = positions[:, None] * self.inv_frequency.float()[None, :]
        cosine, sine = angles.cos().to(q.dtype)[None, None], angles.sin().to(q.dtype)[None, None]
        attended = F.scaled_dot_product_attention(self._rotate(q, cosine, sine),
            self._rotate(k, cosine, sine), v, dropout_p=0.0, is_causal=False)
        return self.out(attended.permute(2, 0, 1, 3).reshape(len(h), 50, 256))
