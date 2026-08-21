"""Small-matrix canonicalization and union operations for LoRA factors."""

from __future__ import annotations

import torch


def canonicalize_low_rank_factors(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    output_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the compact SVD gauge without materializing the dense update."""

    squeeze = a.ndim == 2
    a_batch = a[None] if squeeze else a
    b_batch = b[None] if squeeze else b
    if (
        a_batch.ndim != 3
        or b_batch.ndim != 3
        or a_batch.shape[0] != b_batch.shape[0]
        or a_batch.shape[1] != b_batch.shape[2]
    ):
        raise ValueError("low-rank factor shapes changed")
    rank = int(output_rank or a_batch.shape[1])
    if not 0 < rank <= min(a_batch.shape[1], a_batch.shape[2], b_batch.shape[1]):
        raise ValueError("invalid compact low-rank output rank")
    # Autocast would otherwise turn the QR core product back into BF16 before
    # SVD, which CUDA's batched SVD does not support.  The decompositions are
    # tiny (at most 32 x 32), so keep the complete numerical kernel in FP32.
    with torch.autocast(device_type=a_batch.device.type, enabled=False):
        q_b, r_b = torch.linalg.qr(b_batch.float(), mode="reduced")
        q_a, r_a = torch.linalg.qr(
            a_batch.float().transpose(1, 2), mode="reduced"
        )
        core = r_b @ r_a.transpose(1, 2)
        u, singular, vh = torch.linalg.svd(core, full_matrices=False)
        u = u[:, :, :rank]
        singular = singular[:, :rank]
        vh = vh[:, :rank]
        root = singular.clamp_min(0).sqrt()
        canonical_b = (q_b @ u) * root[:, None]
        canonical_a = root[:, :, None] * (vh @ q_a.transpose(1, 2))
    pivots = canonical_b.abs().argmax(dim=1, keepdim=True)
    signs = canonical_b.gather(1, pivots).squeeze(1).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    canonical_b = canonical_b * signs[:, None]
    canonical_a = canonical_a * signs[:, :, None]
    canonical_a = canonical_a.to(a)
    canonical_b = canonical_b.to(b)
    if squeeze:
        return canonical_a[0], canonical_b[0]
    return canonical_a, canonical_b


def merge_low_rank_updates(
    *,
    base_a: torch.Tensor,
    base_b: torch.Tensor,
    residual_a: torch.Tensor,
    residual_b: torch.Tensor,
    output_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-r compression of base BA plus residual BA."""

    residual_a_batch = residual_a[None] if residual_a.ndim == 2 else residual_a
    residual_b_batch = residual_b[None] if residual_b.ndim == 2 else residual_b
    base_a_batch = base_a[None] if base_a.ndim == 2 else base_a
    base_b_batch = base_b[None] if base_b.ndim == 2 else base_b
    batch = residual_a_batch.shape[0]
    if (
        residual_a_batch.ndim != 3
        or residual_b_batch.ndim != 3
        or base_a_batch.ndim != 3
        or base_b_batch.ndim != 3
        or residual_b_batch.shape[0] != batch
        or base_a_batch.shape[0] not in (1, batch)
        or base_b_batch.shape[0] not in (1, batch)
        or base_a_batch.shape[2] != residual_a_batch.shape[2]
        or base_b_batch.shape[1] != residual_b_batch.shape[1]
        or base_a_batch.shape[1] != base_b_batch.shape[2]
        or residual_a_batch.shape[1] != residual_b_batch.shape[2]
    ):
        raise ValueError("low-rank union factor shapes changed")
    base_a_batch = base_a_batch.expand(batch, -1, -1)
    base_b_batch = base_b_batch.expand(batch, -1, -1)
    base_is_zero = bool(
        (base_a_batch.detach().abs().amax() == 0)
        or (base_b_batch.detach().abs().amax() == 0)
    )
    residual_is_zero = bool(
        (residual_a_batch.detach().abs().amax() == 0)
        or (residual_b_batch.detach().abs().amax() == 0)
    )
    if base_is_zero:
        return canonicalize_low_rank_factors(
            residual_a_batch.float(),
            residual_b_batch.float(),
            output_rank=output_rank,
        )
    if residual_is_zero:
        return canonicalize_low_rank_factors(
            base_a_batch.float(), base_b_batch.float(), output_rank=output_rank
        )
    union_a = torch.cat((base_a_batch, residual_a_batch), dim=1).float()
    union_b = torch.cat((base_b_batch, residual_b_batch), dim=2).float()
    return canonicalize_low_rank_factors(
        union_a, union_b, output_rank=output_rank
    )
