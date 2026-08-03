"""Diagnostic decoding helpers shared by Writer internal analyses."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.writer.model import CompleteLoRAWriter, WriterModelError


def decode_direction_stores(
    writer: CompleteLoRAWriter,
    coordinates: torch.Tensor,
    task_anchor: torch.Tensor,
) -> dict[str, Any]:
    """Replay the canonical direction-store decode while retaining stages."""

    store_indices, store_weights = writer.direction_router(task_anchor)
    head_rows = {name: [] for name in writer.factor_heads}
    factors: dict[str, torch.Tensor] = {}
    public: dict[str, torch.Tensor] = {}
    for spec in writer.tensor_specs:
        key, target = writer._decoding[spec.name]
        rows = writer.factor_heads[key](
            coordinates[:, target], store_indices, store_weights
        )
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
        "store_indices": store_indices,
        "store_weights": store_weights,
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
