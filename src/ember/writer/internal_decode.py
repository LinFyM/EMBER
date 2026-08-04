"""Diagnostic decoding helpers shared by Writer internal analyses."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.writer.model import CompleteLoRAWriter, WriterModelError


def decode_target_owned_factors(
    writer: CompleteLoRAWriter,
    coordinates: torch.Tensor,
) -> dict[str, Any]:
    """Replay the canonical target-owned decode while retaining stages."""

    head_rows = {name: [] for name in writer.FACTOR_WIDTHS}
    factors: dict[str, torch.Tensor] = {}
    public: dict[str, torch.Tensor] = {}
    for spec, head in zip(writer.tensor_specs, writer.factor_heads, strict=True):
        key, target = writer._decoding[spec.name]
        rows = head(coordinates[:, target])
        head_rows[key].append((target, rows))
        generated = rows.transpose(-1, -2) if spec.transpose_output else rows
        factors[spec.name] = generated
        template = getattr(writer, writer._template_buffers[spec.name])
        public[spec.name] = generated.to(template.dtype) + template[None]
    heads = {}
    head_target_indices = {}
    for key, values in head_rows.items():
        values.sort(key=lambda item: item[0])
        targets = tuple(target for target, _ in values)
        if not values or len(targets) != len(set(targets)):
            raise WriterModelError("factor-head target ownership changed")
        heads[key] = torch.stack([value for _, value in values], dim=1)
        head_target_indices[key] = targets
    return {
        "heads": heads,
        "head_target_indices": head_target_indices,
        "factors": factors,
        "public": public,
    }


def state_row(
    mapping: Mapping[str, torch.Tensor], row: int
) -> dict[str, torch.Tensor]:
    return {name: value[row] for name, value in mapping.items()}


def tensor_row(value: torch.Tensor, index: int = 0) -> torch.Tensor:
    return value[index : index + 1]


def program_row(
    program: Mapping[str, Any],
    index: int,
    *,
    raw: torch.Tensor | None = None,
    memory: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "raw": tensor_row(program["raw"] if raw is None else raw, index),
        "memory": tensor_row(
            program["memory"] if memory is None else memory, index
        ),
        "endpoints": tensor_row(program["endpoints"], index),
        "valid_intervals": tensor_row(program["valid_intervals"], index),
    }
