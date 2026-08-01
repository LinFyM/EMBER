"""Exact CV-ADR Writer-path reconstruction and counterfactual diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ember.lora import lora_state_sha256
from ember.writer.internal_metrics import (
    CONDITIONS,
    attention_summary,
    mapping_metrics,
    probability_summary,
    relative_metrics,
    routing_centered_energy,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.semantic_program import (
    apply_rope,
    apply_two_axis_rope,
    merge_heads,
    split_heads,
)


PARITY_TOLERANCE = 2e-5
# Explicit softmax/weighted-value reconstruction and CUDA BF16 SDPA are two
# mathematically equivalent attention implementations, but they round at
# different points. One BF16 unit roundoff is the narrow numerical contract
# for diagnostics that reconstruct SDPA weights; canonical Writer, compiler,
# and public-LoRA parity retain the strict tolerance above.
ATTENTION_PARITY_TOLERANCE = 8e-3
# Production replay stays on the backend that a fresh training/evaluation
# process uses for Writer generation. The upstream PI05 recursive sampler
# temporarily forces eager attention, so policy probes must restore the prior
# backend before another Writer capture.
REPLAY_TOLERANCE = PARITY_TOLERANCE


def _parity(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    value = relative_metrics(reference, candidate)
    if value["relative_l2"] > PARITY_TOLERANCE:
        raise WriterModelError(f"internal-analysis {name} parity failed: {value}")
    return value


def _attention_parity(
    name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    value = relative_metrics(reference, candidate)
    value["tolerance"] = ATTENTION_PARITY_TOLERANCE
    if value["relative_l2"] > ATTENTION_PARITY_TOLERANCE:
        raise WriterModelError(
            f"internal-analysis {name} BF16 SDPA parity failed: {value}"
        )
    return value


def _weights(query: torch.Tensor, key: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    return torch.softmax(logits.float(), dim=-1).to(logits.dtype)


def _raw_attention(module: Any, args: Sequence[torch.Tensor], kwargs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    content, valid = args
    addressed = module.norm(content) + kwargs["qk_identity"]
    query = apply_rope(split_heads(module.query(addressed), module.heads), kwargs["positions"])
    key = apply_rope(split_heads(module.key(addressed), module.heads), kwargs["positions"])
    safe = valid.clone(); empty = ~safe.any(dim=1); safe[empty, 0] = True
    allowed = safe[:, None, None, :]
    if module.causal:
        allowed = allowed & torch.ones(content.shape[1], content.shape[1], dtype=torch.bool, device=content.device).tril()[None, None]
    weights = _weights(query, key, allowed)
    rebuilt = module.output(merge_heads(weights @ split_heads(content, module.heads))).masked_fill(~valid[..., None], 0)
    return weights, rebuilt


def _program_block(block: Any, content: torch.Tensor, endpoints: torch.Tensor, intervals: torch.Tensor, semantics: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    calls: list[tuple[Any, Sequence[torch.Tensor], Mapping[str, Any]]] = []
    def capture(module: Any, args: Sequence[torch.Tensor], kwargs: Mapping[str, Any]) -> None:
        calls.append((module, args, kwargs))
    handles = [value.register_forward_pre_hook(capture, with_kwargs=True) for value in (block.local_attention, block.temporal_attention)]
    try:
        output = block(content, endpoints, intervals, semantics)
    finally:
        for handle in handles: handle.remove()
    diagnostics = {}
    for name, (module, args, kwargs) in zip(("interval_local", "semantic_column_causal"), calls, strict=True):
        weights, rebuilt = _raw_attention(module, args, kwargs)
        diagnostics[name] = {
            **probability_summary(weights, args[1]),
            "output_parity": _attention_parity(
                f"Program {name} attention", rebuilt, module(*args, **kwargs)
            ),
        }
    return output, diagnostics


def _core_pipeline(writer: CompleteLoRAWriter, query: torch.Tensor, evidence: torch.Tensor, frames: torch.Tensor, tokens: torch.Tensor, carrier: str = "full") -> dict[str, Any]:
    fusion = writer.semantic_core.set_fusion
    pre, observed_attention = fusion(query, evidence, frames, tokens)
    active = frames[:, :, None, None]; counts = frames.sum(1).to(evidence.dtype)[:, None, None]
    mean = evidence.masked_fill(~active, 0).sum(1) / counts
    centered = (evidence - mean[:, None]).masked_fill(~active, 0)
    q = split_heads(fusion.query(fusion.query_norm(query)), fusion.heads)
    b, f, t, w = evidence.shape
    k = fusion.key(fusion.evidence_norm(evidence)).reshape(b, f, t, fusion.heads, w // fusion.heads).permute(0, 3, 1, 2, 4)
    expected_attention = torch.softmax((torch.einsum("bhld,bhtld->bhtl", q, k) / math.sqrt(w // fusion.heads)).masked_fill(~frames[:, None, :, None], torch.finfo(q.dtype).min).float(), dim=2).to(q.dtype)
    expected_attention = expected_attention.masked_fill(~tokens[:, None, None], 0)
    frame_parity = _parity("Core frame attention", observed_attention, expected_attention)
    attended = torch.einsum("bhtl,bhtld->bhld", observed_attention, centered.reshape(b, f, t, fusion.heads, w // fusion.heads).permute(0, 3, 1, 2, 4))
    mean_carrier = fusion.mean(mean).masked_fill(~tokens[..., None], 0)
    residual = fusion.output(merge_heads(attended)).masked_fill(~tokens[..., None], 0)
    _parity("Core carrier", pre, mean_carrier + residual)
    if carrier not in {"full", "no_mean", "no_centered"}: raise WriterModelError("unknown Core carrier ablation")
    value = {"full": pre, "no_mean": residual, "no_centered": mean_carrier}[carrier]
    blocks, attentions = [], []
    for block in writer.semantic_core.blocks:
        normalized = block.attention_norm(value)
        positions = torch.arange(value.shape[1], device=value.device)[None].expand(value.shape[0], -1)
        cq = apply_rope(split_heads(block.query(normalized), block.heads), positions)
        ck = apply_rope(split_heads(block.key(normalized), block.heads), positions)
        attentions.append(probability_summary(_weights(cq, ck, tokens[:, None, None]), tokens))
        value = block(value, tokens); blocks.append(value)
    return {"mean": mean, "centered": centered, "mean_carrier": mean_carrier, "centered_residual": residual, "pre": pre, "blocks": blocks, "final": value, "frame_attention": observed_attention, "attention": {"frame": frame_parity, "token_blocks": attentions}}


def _raw_program(grounded: torch.Tensor, action: torch.Tensor, positions: torch.Tensor, frames: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
    intervals = frames[:, :-1] & frames[:, 1:]; endpoints = positions[:, 1:]
    semantics = torch.cat((torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device), tokens, tokens), 1)
    raw = torch.cat((action[:, :-1, None], grounded[:, 1:], grounded[:, 1:] - grounded[:, :-1]), 2)
    raw = raw.masked_fill(~(intervals[:, :, None] & semantics[:, None])[..., None], 0)
    return raw, endpoints, intervals, semantics


def _program_pipeline(writer: CompleteLoRAWriter, raw: torch.Tensor, endpoints: torch.Tensor, intervals: torch.Tensor, semantics: torch.Tensor) -> dict[str, Any]:
    memory, blocks, attention = raw, [], []
    for block in writer.semantic_program.blocks:
        memory, diagnostic = _program_block(
            block, memory, endpoints, intervals, semantics
        )
        blocks.append(memory); attention.append(diagnostic)
    return {
        "raw": raw,
        "blocks": blocks,
        "memory": memory,
        "endpoints": endpoints,
        "valid_intervals": intervals,
        "valid_semantics": semantics,
        "attention": attention,
    }


def _compile(writer: CompleteLoRAWriter, core: torch.Tensor, valid_core: torch.Tensor, program: Mapping[str, torch.Tensor], target_permutation: torch.Tensor | None = None, rank_permutation: torch.Tensor | None = None) -> dict[str, Any]:
    compiler = writer.compiler; memory = program["memory"]
    b, intervals, columns, width = memory.shape
    target = compiler.target_identity_norm(compiler.target_identity)
    rank = compiler.rank_identity_norm(compiler.rank_identity)
    if target_permutation is not None: target = target.index_select(0, target_permutation)
    if rank_permutation is not None: rank = rank.index_select(0, rank_permutation)
    target_query = target[None].expand(b, -1, -1)
    program_query = (target_query[:, :, None] + rank[None, None]).reshape(b, compiler.target_count * compiler.rank, width)
    if target_permutation is None and rank_permutation is None:
        coordinates, diagnostic = compiler.compile_with_diagnostics(core, valid_core, memory, program["endpoints"], program["valid_intervals"], program["valid_semantics"])
    else:
        core_read = compiler.core_reader(target_query, compiler.core_norm(core), core, valid_core)
        flat_memory = memory.reshape(b, intervals * columns, width)
        valid = (program["valid_intervals"][:, :, None] & program["valid_semantics"][:, None]).reshape(b, -1)
        first = program["endpoints"][:, :, None].expand(b, intervals, columns).reshape(b, -1)
        ordinal = compiler._semantic_ordinals(columns, memory.device); second = ordinal[None, None].expand(b, intervals, columns).reshape(b, -1)
        identity = compiler._type_identity(columns).to(memory.dtype)[None, None].expand(b, intervals, columns, width).reshape_as(flat_memory)
        program_read = compiler.program_reader(program_query, flat_memory, valid, memory_qk_identity=identity, endpoint_positions=first, semantic_positions=second).reshape(b, compiler.target_count, compiler.rank, width)
        coordinates = torch.cat((core_read[:, :, None].expand_as(program_read), program_read), -1)
        diagnostic = {"target_query": target_query, "program_query": program_query, "core_read": core_read, "program_read": program_read, "coordinates": coordinates}
    cq = split_heads(compiler.core_reader.query(target_query), compiler.core_reader.heads)
    ck = split_heads(compiler.core_reader.key(compiler.core_norm(core)), compiler.core_reader.heads)
    core_weights = _weights(cq, ck, valid_core[:, None, None])
    core_rebuilt = compiler.core_reader.output(merge_heads(core_weights @ split_heads(core, compiler.core_reader.heads)))
    flat_memory = memory.reshape(b, intervals * columns, width)
    valid = (program["valid_intervals"][:, :, None] & program["valid_semantics"][:, None]).reshape(b, -1)
    first = program["endpoints"][:, :, None].expand(b, intervals, columns).reshape(b, -1)
    ordinal = compiler._semantic_ordinals(columns, memory.device); second = ordinal[None, None].expand(b, intervals, columns).reshape(b, -1)
    identity = compiler._type_identity(columns).to(memory.dtype)[None, None].expand(b, intervals, columns, width).reshape_as(flat_memory)
    pq = split_heads(compiler.program_reader.query(program_query), compiler.program_reader.heads)
    pk = split_heads(compiler.program_reader.key(compiler.program_reader.memory_norm(flat_memory) + identity), compiler.program_reader.heads)
    zeros = torch.zeros(program_query.shape[:2], dtype=torch.long, device=memory.device)
    pq = apply_two_axis_rope(pq, zeros, zeros); pk = apply_two_axis_rope(pk, first, second)
    program_weights = _weights(pq, pk, valid[:, None, None])
    program_rebuilt = compiler.program_reader.output(merge_heads(program_weights @ split_heads(flat_memory, compiler.program_reader.heads))).reshape_as(diagnostic["program_read"])
    parity = {
        "core_read": _attention_parity(
            "Core reader", diagnostic["core_read"], core_rebuilt
        ),
        "program_read": _attention_parity(
            "Program reader", diagnostic["program_read"], program_rebuilt
        ),
    }
    attention = {"core": probability_summary(core_weights, torch.ones(b, compiler.target_count, dtype=torch.bool, device=core.device)), "program": attention_summary(program_weights, program["valid_intervals"], program["valid_semantics"]), "program_target_rank_routing": routing_centered_energy(program_weights, compiler.target_count, compiler.rank)}
    return {"coordinates": coordinates, "diagnostic": diagnostic, "attention": attention, "parity": parity}


def _decode(writer: CompleteLoRAWriter, coordinates: torch.Tensor) -> dict[str, Any]:
    head_rows: dict[str, list[tuple[int, torch.Tensor]]] = {
        name: [] for name in writer.factor_heads
    }
    factors, public = {}, {}
    for spec in writer.tensor_specs:
        key, target = writer._decoding[spec.name]
        # Match CompleteLoRAWriter.forward exactly: each real target owns one
        # target-sized GEMM. A single [batch, 38, rank, width] GEMM is
        # mathematically equivalent but not BF16 bit-equivalent on CUDA.
        rows = writer.factor_heads[key](coordinates[:, target])
        head_rows[key].append((target, rows))
        generated = rows.transpose(-1, -2) if spec.transpose_output else rows
        factors[spec.name] = generated
        public[spec.name] = generated.to(getattr(writer, writer._template_buffers[spec.name]).dtype) + getattr(writer, writer._template_buffers[spec.name])[None]
    heads, head_target_indices = {}, {}
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


def _pack(value: torch.Tensor, offsets: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    width = value.shape[1:]; maximum = max(b - a for a, b in zip(offsets, offsets[1:]))
    packed = value.new_zeros(len(offsets) - 1, maximum, *width); valid = torch.zeros(packed.shape[:2], dtype=torch.bool, device=value.device)
    for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
        packed[row, : right - left] = value[left:right]; valid[row, : right - left] = True
    return packed, valid


@torch.inference_mode()
def capture_writer(writer: CompleteLoRAWriter, policy: torch.nn.Module, frames: torch.Tensor, indices: torch.Tensor, offsets_tensor: torch.Tensor, tokens: torch.Tensor, masks: torch.Tensor, spans: torch.Tensor) -> dict[str, Any]:
    captured: dict[str, list[Any]] = {
        name: [] for name in ("encoder", "core", "program", "compiler", "raw_action")
    }
    handles = [
        writer.semantic_encoder.register_forward_hook(lambda _m, _a, out: captured["encoder"].append(out)),
        writer.semantic_core.register_forward_hook(lambda _m, _a, out: captured["core"].append(out)),
        writer.semantic_program.register_forward_hook(lambda _m, _a, out: captured["program"].append(out)),
        writer.compiler.register_forward_hook(lambda _m, _a, out: captured["compiler"].append(out)),
        writer.semantic_encoder.interaction_projection.register_forward_pre_hook(lambda _m, args: captured["raw_action"].append(args[0].detach())),
    ]
    try: canonical = writer(frames, indices, offsets_tensor, tokens, masks, spans, policy=policy)
    finally:
        for handle in handles: handle.remove()
    if any(
        len(captured[name]) != 1
        for name in ("encoder", "core", "program", "compiler")
    ) or not captured["raw_action"]:
        raise WriterModelError("internal-analysis Writer hooks changed")
    offsets = writer._validated_offsets(offsets_tensor, frames.shape[0])
    q, x, g, action, valid_tokens = captured["encoder"][0]
    packed_x, packed_g, packed_action, positions, valid_frames = writer._pack_video_program(x, g, action, indices, offsets)
    packed_raw, raw_valid = _pack(torch.cat(captured["raw_action"]), offsets)
    if not torch.equal(raw_valid, valid_frames): raise WriterModelError("native Action capture lost video alignment")
    core = _core_pipeline(writer, q, packed_x, valid_frames, valid_tokens)
    raw = _raw_program(packed_g, packed_action, positions, valid_frames, valid_tokens)
    program = _program_pipeline(writer, *raw)
    observed_core, observed_frame_attention = captured["core"][0]
    observed_program = captured["program"][0]
    observed_coordinates = captured["compiler"][0]
    parity = {
        "core": _parity("Core final", observed_core, core["final"]),
        "frame_attention": _parity(
            "frame attention", observed_frame_attention, core["frame_attention"]
        ),
        "program_memory": _parity(
            "Program memory", observed_program[0], program["memory"]
        ),
    }
    if not all(torch.equal(a, b) for a, b in zip(observed_program[1:], (program["endpoints"], program["valid_intervals"], program["valid_semantics"]), strict=True)):
        raise WriterModelError("Program endpoint/mask parity failed")
    # The production tensors, not a second BF16 attention pass, own every
    # downstream scientific metric.  Keep reconstructed intermediates for
    # diagnostics while anchoring the compiler and counterfactuals to the
    # single canonical Writer forward above.
    core["recomputed_final"] = core["final"]
    core["recomputed_frame_attention"] = core["frame_attention"]
    core["final"] = observed_core
    core["frame_attention"] = observed_frame_attention
    program["recomputed_memory"] = program["memory"]
    program["memory"] = observed_program[0]
    compiled = _compile(writer, core["final"], valid_tokens, program)
    parity["compiler_coordinates"] = _attention_parity(
        "compiler coordinates", observed_coordinates, compiled["coordinates"]
    )
    compiled["recomputed_coordinates"] = compiled["coordinates"]
    compiled["coordinates"] = observed_coordinates
    decoded = _decode(writer, compiled["coordinates"])
    parity["public"] = mapping_metrics(canonical, decoded["public"])
    per_tensor = {
        name: relative_metrics(canonical[name], decoded["public"][name])
        for name in canonical
    }
    worst_name = max(per_tensor, key=lambda name: per_tensor[name]["relative_l2"])
    parity["public"]["max_tensor_relative_l2"] = per_tensor[worst_name]["relative_l2"]
    if parity["public"]["max_tensor_relative_l2"] > PARITY_TOLERANCE:
        raise WriterModelError(
            f"canonical public decode parity failed for {worst_name}: "
            f"{per_tensor[worst_name]}"
        )
    return {"q": q, "m": packed_x - packed_g, "g": packed_g, "x": packed_x, "a_raw": packed_raw, "a": packed_action, "positions": positions, "valid_frames": valid_frames, "valid_tokens": valid_tokens, "core": core, "program": program, "compiled": compiled, "decoded": decoded, "canonical": canonical, "parity": parity}


def _state(mapping: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {name: value[row] for name, value in mapping.items()}


def _variant(writer: CompleteLoRAWriter, core: torch.Tensor, valid_core: torch.Tensor, program: Mapping[str, torch.Tensor], *, target_permutation: torch.Tensor | None = None, rank_permutation: torch.Tensor | None = None, coordinates: torch.Tensor | None = None) -> dict[str, Any]:
    compiled = _compile(writer, core, valid_core, program, target_permutation, rank_permutation) if coordinates is None else {"coordinates": coordinates}
    decoded = _decode(writer, compiled["coordinates"])
    return {"coordinates": compiled["coordinates"][0], "factor": _state(decoded["factors"], 0), "public": _state(decoded["public"], 0), "heads": {name: value[0] for name, value in decoded["heads"].items()}, "attention": compiled.get("attention")}


def _row(value: torch.Tensor, index: int = 0) -> torch.Tensor:
    return value[index : index + 1]


def _program_row(
    program: Mapping[str, torch.Tensor],
    index: int,
    *,
    memory: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "memory": _row(program["memory"] if memory is None else memory, index),
        "endpoints": _row(program["endpoints"], index),
        "valid_intervals": _row(program["valid_intervals"], index),
        "valid_semantics": _row(program["valid_semantics"], index),
    }


def _base_counterfactuals(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    core = captured["core"]["final"]
    valid = captured["valid_tokens"]
    program = captured["program"]
    reference_program = _program_row(program, 0)
    recomputed = _variant(writer, _row(core), _row(valid), reference_program)
    full = _variant(
        writer,
        _row(core),
        _row(valid),
        reference_program,
        coordinates=_row(captured["compiled"]["coordinates"]),
    )
    full["attention"] = recomputed["attention"]
    variants = {"full": full}

    permuted_memory = _row(program["memory"]).clone()
    interval_count = int(_row(program["valid_intervals"]).sum())
    permuted_memory[:, :interval_count] = permuted_memory[
        :, :interval_count
    ].flip(1)
    variants["program_memory/order_permuted"] = _variant(
        writer,
        _row(core),
        _row(valid),
        _program_row(program, 0, memory=permuted_memory),
    )

    coordinate = full["coordinates"]
    core_coordinate = torch.cat(
        (coordinate[..., :256], torch.zeros_like(coordinate[..., 256:])),
        -1,
    )[None]
    program_coordinate = torch.cat(
        (torch.zeros_like(coordinate[..., :256]), coordinate[..., 256:]),
        -1,
    )[None]
    variants["core_only"] = _variant(
        writer,
        _row(core),
        _row(valid),
        reference_program,
        coordinates=core_coordinate,
    )
    variants["program_only"] = _variant(
        writer,
        _row(core),
        _row(valid),
        reference_program,
        coordinates=program_coordinate,
    )
    return variants, full


def _fixed_condition_variants(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
    variants: dict[str, dict[str, Any]],
) -> None:
    core = captured["core"]["final"]
    valid = captured["valid_tokens"]
    program = captured["program"]
    for index, condition in enumerate(CONDITIONS):
        variants[f"fixed_core/{condition}"] = _variant(
            writer,
            _row(core),
            _row(valid),
            _program_row(program, index),
        )
        variants[f"fixed_program/{condition}"] = _variant(
            writer,
            _row(core, index),
            _row(valid, index),
            _program_row(program, 0),
        )


def _program_component_slices(
    raw: torch.Tensor,
) -> dict[str, slice]:
    task_tokens = (raw.shape[2] - 1) // 2
    return {
        "A": slice(0, 1),
        "E": slice(1, 1 + task_tokens),
        "D": slice(1 + task_tokens, None),
    }


def _component_subset_variants(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
    variants: dict[str, dict[str, Any]],
    full: dict[str, Any],
    raw: torch.Tensor,
    slices: Mapping[str, slice],
) -> None:
    core = captured["core"]["final"]
    valid = captured["valid_tokens"]
    program = captured["program"]
    selections = (
        ("A",),
        ("E",),
        ("D",),
        ("A", "E"),
        ("A", "D"),
        ("E", "D"),
        ("A", "E", "D"),
    )
    for names in selections:
        selected = torch.zeros_like(raw)
        for name in names:
            selected[:, :, slices[name]] = raw[:, :, slices[name]]
        label = "+".join(names)
        if names == ("A", "E", "D"):
            if not torch.equal(selected, raw):
                raise WriterModelError(
                    "A+E+D identity reconstruction changed raw Program"
                )
            variants[f"aed/{label}"] = full
            continue
        rebuilt = _program_pipeline(
            writer,
            selected,
            _row(program["endpoints"]),
            _row(program["valid_intervals"]),
            _row(program["valid_semantics"]),
        )
        variants[f"aed/{label}"] = _variant(
            writer,
            _row(core),
            _row(valid),
            rebuilt,
        )


def _component_scale_variants(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
    variants: dict[str, dict[str, Any]],
    full: dict[str, Any],
    raw: torch.Tensor,
    slices: Mapping[str, slice],
) -> None:
    core = captured["core"]["final"]
    valid = captured["valid_tokens"]
    program = captured["program"]
    for name, selected_slice in slices.items():
        for scale in (0.5, 1.0, 2.0):
            scaled = raw.clone()
            scaled[:, :, selected_slice] *= scale
            label = f"scale/{name}/{scale:g}"
            if scale == 1.0:
                if not torch.equal(scaled, raw):
                    raise WriterModelError(
                        f"scale/{name}/1 identity changed raw Program"
                    )
                variants[label] = full
                continue
            rebuilt = _program_pipeline(
                writer,
                scaled,
                _row(program["endpoints"]),
                _row(program["valid_intervals"]),
                _row(program["valid_semantics"]),
            )
            variants[label] = _variant(
                writer,
                _row(core),
                _row(valid),
                rebuilt,
            )


def _core_and_identity_variants(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
    variants: dict[str, dict[str, Any]],
) -> None:
    core = captured["core"]["final"]
    valid = captured["valid_tokens"]
    program = captured["program"]
    for carrier in ("no_mean", "no_centered"):
        changed = _core_pipeline(
            writer,
            _row(captured["q"]),
            _row(captured["x"]),
            _row(captured["valid_frames"]),
            _row(valid),
            carrier,
        )
        variants[f"core_carrier/{carrier}"] = _variant(
            writer,
            changed["final"],
            _row(valid),
            _program_row(program, 0),
        )

    target_permutation = torch.roll(
        torch.arange(writer.compiler.target_count, device=core.device),
        -1,
    )
    rank_permutation = torch.roll(
        torch.arange(writer.compiler.rank, device=core.device),
        -1,
    )
    variants["identity/target"] = _variant(
        writer,
        _row(core),
        _row(valid),
        _program_row(program, 0),
        target_permutation=target_permutation,
    )
    variants["identity/rank"] = _variant(
        writer,
        _row(core),
        _row(valid),
        _program_row(program, 0),
        rank_permutation=rank_permutation,
    )


def _validate_counterfactuals(
    writer: CompleteLoRAWriter,
    program_sha256: str,
    variants: Mapping[str, dict[str, Any]],
    full: dict[str, Any],
) -> None:
    for name in ("aed/A+E+D", "scale/A/1", "scale/E/1", "scale/D/1"):
        error = max(
            relative_metrics(
                full["public"][key],
                variants[name]["public"][key],
            )["relative_l2"]
            for key in full["public"]
        )
        if error > PARITY_TOLERANCE:
            raise WriterModelError(
                f"counterfactual full/scale1 parity failed: {name}"
            )
    if lora_state_sha256(writer.semantic_program.state_dict()) != program_sha256:
        raise WriterModelError(
            "Program-memory counterfactual mutated trained Program state"
        )


@torch.inference_mode()
def counterfactual_states(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Re-run interventions through the one contextual Program memory."""

    program_sha256 = lora_state_sha256(writer.semantic_program.state_dict())
    variants, full = _base_counterfactuals(writer, captured)
    _fixed_condition_variants(writer, captured, variants)
    raw = _row(captured["program"]["raw"])
    slices = _program_component_slices(raw)
    _component_subset_variants(
        writer,
        captured,
        variants,
        full,
        raw,
        slices,
    )
    _component_scale_variants(
        writer,
        captured,
        variants,
        full,
        raw,
        slices,
    )
    _core_and_identity_variants(writer, captured, variants)
    _validate_counterfactuals(writer, program_sha256, variants, full)
    full["program_memory_authority"] = {
        "trained_program_state_sha256": program_sha256,
        "order_permutation": (
            "reverse valid contextual Program intervals while Core, masks, "
            "endpoint positions, and trained weights stay fixed"
        ),
        "key_value_coupling": (
            "the same contextual Program tensor is both normalized K content "
            "and physical V; fixed-key interventions are structurally invalid"
        ),
    }
    return variants
