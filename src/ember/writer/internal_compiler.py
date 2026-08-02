"""Exact target/rank role-reader reconstruction for internal analysis."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from ember.writer.internal_metrics import (
    probability_summary,
    relative_metrics,
    routing_centered_energy,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.semantic_program import apply_rope, merge_heads, split_heads


ATTENTION_PARITY_TOLERANCE = 8e-3
ROLE_NAMES = ("A", "E", "D")


def _attention_parity(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    value = relative_metrics(reference, candidate)
    value["tolerance"] = ATTENTION_PARITY_TOLERANCE
    if value["relative_l2"] > ATTENTION_PARITY_TOLERANCE:
        raise WriterModelError(
            f"internal-analysis {name} BF16 SDPA parity failed: {value}"
        )
    return value


def _weights(
    query: torch.Tensor,
    key: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    logits = torch.matmul(query, key.transpose(-1, -2))
    logits = logits / math.sqrt(query.shape[-1])
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    return torch.softmax(logits.float(), dim=-1).to(logits.dtype)


def _target_context(
    writer: CompleteLoRAWriter,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    target_permutation: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    compiler = writer.compiler
    target = compiler.target_identity
    if target_permutation is not None:
        target = target[target_permutation]
    target_query = compiler.target_identity_norm(target)[None].expand(
        core.shape[0], -1, -1
    )
    target_core = compiler.core_reader(
        target_query,
        compiler.core_norm(core),
        core,
        valid_core,
    )
    return target_query, target_core


def _role_inputs(
    compiler: Any,
    memory: torch.Tensor,
    program: Mapping[str, torch.Tensor],
    target_query: torch.Tensor,
    target_core: torch.Tensor,
    rank_permutation: torch.Tensor | None,
) -> dict[str, Any]:
    batch, targets, intervals, roles, width = memory.shape
    rank = compiler.rank_identity
    if rank_permutation is not None:
        rank = rank[rank_permutation]
    rank_identity = compiler.rank_identity_norm(rank)
    role_identity = compiler.role_identity_norm(
        compiler.role_identity
    ).to(memory.dtype)
    target_address = target_query + compiler.core_norm(target_core)
    program_query = (
        target_address[:, :, None, None]
        + rank_identity[None, None, :, None]
        + role_identity[None, None, None]
    )
    flat_query = program_query.permute(0, 1, 3, 2, 4).reshape(
        batch * targets * roles, compiler.rank, width
    )
    flat_memory = memory.permute(0, 1, 3, 2, 4).reshape(
        batch * targets * roles, intervals, width
    )
    valid = program["valid_intervals"][:, None, None].expand(
        batch, targets, roles, intervals
    ).reshape(batch * targets * roles, intervals)
    endpoints = program["endpoints"][:, None, None].expand(
        batch, targets, roles, intervals
    ).reshape(batch * targets * roles, intervals)
    memory_identity = role_identity[None, None, :, None].expand(
        batch, targets, roles, intervals, width
    ).reshape(batch * targets * roles, intervals, width)
    return {
        "batch": batch,
        "targets": targets,
        "intervals": intervals,
        "roles": roles,
        "width": width,
        "program_query": program_query,
        "flat_query": flat_query,
        "flat_memory": flat_memory,
        "valid": valid,
        "endpoints": endpoints,
        "memory_identity": memory_identity,
    }


def _read_roles(compiler: Any, inputs: Mapping[str, Any]) -> torch.Tensor:
    role_read = compiler.program_reader(
        inputs["flat_query"],
        inputs["flat_memory"],
        inputs["valid"],
        memory_qk_identity=inputs["memory_identity"],
        endpoint_positions=inputs["endpoints"],
    )
    return role_read.reshape(
        inputs["batch"],
        inputs["targets"],
        inputs["roles"],
        compiler.rank,
        inputs["width"],
    ).permute(0, 1, 3, 2, 4)


def _core_attention_rebuild(
    compiler: Any,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    target_query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = split_heads(
        compiler.core_reader.query(target_query), compiler.core_reader.heads
    )
    key = split_heads(
        compiler.core_reader.key(compiler.core_norm(core)),
        compiler.core_reader.heads,
    )
    weights = _weights(query, key, valid_core[:, None, None, :])
    rebuilt = compiler.core_reader.output(
        merge_heads(weights @ split_heads(core, compiler.core_reader.heads))
    )
    return weights, rebuilt


def _role_attention_rebuild(
    compiler: Any,
    inputs: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    reader = compiler.program_reader
    query = split_heads(reader.query(inputs["flat_query"]), reader.heads)
    key = split_heads(
        reader.key(
            reader.memory_norm(inputs["flat_memory"])
            + inputs["memory_identity"]
        ),
        reader.heads,
    )
    query_positions = torch.zeros(
        inputs["flat_query"].shape[:2],
        dtype=torch.long,
        device=inputs["flat_memory"].device,
    )
    query = apply_rope(query, query_positions)
    key = apply_rope(key, inputs["endpoints"])
    weights = _weights(query, key, inputs["valid"][:, None, None, :])
    rebuilt = reader.output(
        merge_heads(
            weights @ split_heads(inputs["flat_memory"], reader.heads)
        )
    ).reshape(
        inputs["batch"],
        inputs["targets"],
        inputs["roles"],
        compiler.rank,
        inputs["width"],
    ).permute(0, 1, 3, 2, 4)
    return weights, rebuilt


def _role_attention_summaries(
    compiler: Any,
    inputs: Mapping[str, Any],
    weights: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    batch = int(inputs["batch"])
    targets = int(inputs["targets"])
    intervals = int(inputs["intervals"])
    roles = int(inputs["roles"])
    grid = weights.reshape(
        batch,
        targets,
        roles,
        compiler.program_reader.heads,
        compiler.rank,
        intervals,
    )
    attention: dict[str, Any] = {}
    routing: dict[str, Any] = {}
    for index, name in enumerate(ROLE_NAMES):
        selected = grid[:, :, index]
        flat = selected.reshape(
            batch * targets,
            compiler.program_reader.heads,
            compiler.rank,
            intervals,
        )
        attention[name] = probability_summary(
            flat,
            torch.ones(
                batch * targets,
                compiler.rank,
                dtype=torch.bool,
                device=weights.device,
            ),
        )
        routed = selected.permute(0, 2, 1, 3, 4).reshape(
            batch,
            compiler.program_reader.heads,
            targets * compiler.rank,
            intervals,
        )
        routing[name] = routing_centered_energy(
            routed, targets, compiler.rank
        )
    return attention, routing


def compile_program(
    writer: CompleteLoRAWriter,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    program: Mapping[str, torch.Tensor],
    target_permutation: torch.Tensor | None = None,
    rank_permutation: torch.Tensor | None = None,
    target_context: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Rebuild target Core and three private role reads with softmax metrics."""

    compiler = writer.compiler
    memory = program["memory"]
    if memory.ndim != 5 or memory.shape[3] != len(ROLE_NAMES):
        raise WriterModelError("internal-analysis role Program changed")
    if target_context is None:
        target_query, target_core = _target_context(
            writer, core, valid_core, target_permutation
        )
    else:
        if target_permutation is not None:
            raise WriterModelError("target context and permutation are exclusive")
        target_query, target_core = target_context
    inputs = _role_inputs(
        compiler,
        memory,
        program,
        target_query,
        target_core,
        rank_permutation,
    )
    role_read = _read_roles(compiler, inputs)
    core_broadcast = target_core[:, :, None].expand(
        inputs["batch"],
        inputs["targets"],
        compiler.rank,
        inputs["width"],
    )
    coordinates = torch.cat(
        (
            core_broadcast,
            role_read.reshape(
                inputs["batch"],
                inputs["targets"],
                compiler.rank,
                inputs["roles"] * inputs["width"],
            ),
        ),
        dim=-1,
    )
    core_weights, core_rebuilt = _core_attention_rebuild(
        compiler, core, valid_core, target_query
    )
    role_weights, role_rebuilt = _role_attention_rebuild(compiler, inputs)
    role_attention, routing = _role_attention_summaries(
        compiler, inputs, role_weights
    )
    return {
        "coordinates": coordinates,
        "diagnostic": {
            "target_query": target_query,
            "program_query": inputs["program_query"],
            "core_read": target_core,
            "role_read": role_read,
            "coordinates": coordinates,
        },
        "attention": {
            "core": probability_summary(
                core_weights,
                torch.ones(
                    inputs["batch"],
                    inputs["targets"],
                    dtype=torch.bool,
                    device=core.device,
                ),
            ),
            "role": role_attention,
            "program_target_rank_routing": routing,
        },
        "parity": {
            "core_read": _attention_parity(
                "Core reader", target_core, core_rebuilt
            ),
            "role_read": _attention_parity(
                "role Program reader", role_read, role_rebuilt
            ),
        },
    }
