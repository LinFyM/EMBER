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


def replace_low_rank_modes(
    *,
    base_a: torch.Tensor,
    base_b: torch.Tensor,
    replacement_a: torch.Tensor,
    replacement_b: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retract each base rank-one mode toward a bounded replacement mode."""

    if (
        base_a.ndim != 2
        or base_b.ndim != 2
        or replacement_a.ndim != 3
        or replacement_b.ndim != 3
        or angles.ndim != 2
        or replacement_a.shape[0] != replacement_b.shape[0]
        or replacement_a.shape[0] != angles.shape[0]
        or base_a.shape[0] != base_b.shape[1]
        or replacement_a.shape[1] != base_a.shape[0]
        or replacement_b.shape[2] != base_a.shape[0]
        or angles.shape[1] != base_a.shape[0]
        or replacement_a.shape[2] != base_a.shape[1]
        or replacement_b.shape[1] != base_b.shape[0]
    ):
        raise ValueError("rank-mode replacement factor shapes changed")
    batch = replacement_a.shape[0]
    with torch.autocast(device_type=replacement_a.device.type, enabled=False):
        base_a_float = base_a.float()
        base_b_float = base_b.float()
        replacement_a_basis, _ = torch.linalg.qr(
            replacement_a.float().transpose(1, 2), mode="reduced"
        )
        replacement_b_basis, _ = torch.linalg.qr(
            replacement_b.float(), mode="reduced"
        )
        # Equal-rank replacement factors preserve the complete base factor
        # energy while preventing raw head amplitude from winning selection.
        replacement_a_scale = (
            base_a_float.square().sum(-1).mean().sqrt()
        )
        replacement_b_scale = (
            base_b_float.square().sum(0).mean().sqrt()
        )
        normalized_a = replacement_a_basis.transpose(1, 2) * replacement_a_scale
        normalized_b = replacement_b_basis * replacement_b_scale
        cosine = angles.float().cos()
        sine = angles.float().sin()
        selected_a = (
            cosine[:, :, None] * base_a_float[None]
            + sine[:, :, None] * normalized_a
        )
        selected_b = (
            cosine[:, None, :] * base_b_float[None]
            + sine[:, None, :] * normalized_b
        )
    if selected_a.shape[0] != batch:
        raise RuntimeError("rank-mode replacement batch changed")
    return selected_a.to(replacement_a), selected_b.to(replacement_b)
