"""Exact Target-Bound Role Program reconstruction and counterfactuals."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ember.lora import lora_state_sha256
from ember.writer.internal_compiler import compile_program as _compile
from ember.writer.internal_metrics import (
    CONDITIONS,
    mapping_metrics,
    probability_summary,
    relative_metrics,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.semantic_program import apply_rope, merge_heads, split_heads


PARITY_TOLERANCE = 2e-5
ATTENTION_PARITY_TOLERANCE = 8e-3
REPLAY_TOLERANCE = PARITY_TOLERANCE
ROLE_NAMES = ("A", "E", "D")


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
    logits = torch.matmul(query, key.transpose(-1, -2))
    logits = logits / math.sqrt(query.shape[-1])
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    return torch.softmax(logits.float(), dim=-1).to(logits.dtype)


def _raw_attention(
    module: Any,
    content: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor,
    qk_identity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    addressed = module.norm(content) + qk_identity
    query = apply_rope(
        split_heads(module.query(addressed), module.heads), positions
    )
    key = apply_rope(
        split_heads(module.key(addressed), module.heads), positions
    )
    safe = valid.clone()
    empty = ~safe.any(dim=1)
    if bool(empty.any()):
        safe[empty, 0] = True
    allowed = safe[:, None, None, :]
    if module.causal:
        causal = torch.ones(
            content.shape[1],
            content.shape[1],
            dtype=torch.bool,
            device=content.device,
        ).tril()
        allowed = allowed & causal[None, None]
    weights = _weights(query, key, allowed)
    rebuilt = module.output(
        merge_heads(weights @ split_heads(content, module.heads))
    ).masked_fill(~valid[..., None], 0.0)
    return weights, rebuilt


def _core_pipeline(
    writer: CompleteLoRAWriter,
    query: torch.Tensor,
    evidence: torch.Tensor,
    frames: torch.Tensor,
    tokens: torch.Tensor,
    carrier: str = "full",
) -> dict[str, Any]:
    fusion = writer.semantic_core.set_fusion
    pre, observed_attention = fusion(query, evidence, frames, tokens)
    active = frames[:, :, None, None]
    counts = frames.sum(1).to(evidence.dtype)[:, None, None]
    mean = evidence.masked_fill(~active, 0).sum(1) / counts
    centered = (evidence - mean[:, None]).masked_fill(~active, 0)
    q = split_heads(fusion.query(fusion.query_norm(query)), fusion.heads)
    batch, video_frames, task_tokens, width = evidence.shape
    k = fusion.key(fusion.evidence_norm(evidence)).reshape(
        batch,
        video_frames,
        task_tokens,
        fusion.heads,
        width // fusion.heads,
    ).permute(0, 3, 1, 2, 4)
    logits = torch.einsum("bhld,bhtld->bhtl", q, k)
    logits = logits / math.sqrt(width // fusion.heads)
    logits = logits.masked_fill(
        ~frames[:, None, :, None], torch.finfo(q.dtype).min
    )
    expected_attention = torch.softmax(logits.float(), dim=2).to(q.dtype)
    expected_attention = expected_attention.masked_fill(
        ~tokens[:, None, None], 0
    )
    frame_parity = _parity(
        "Core frame attention", observed_attention, expected_attention
    )
    value = centered.reshape(
        batch,
        video_frames,
        task_tokens,
        fusion.heads,
        width // fusion.heads,
    ).permute(0, 3, 1, 2, 4)
    attended = torch.einsum(
        "bhtl,bhtld->bhld", observed_attention, value
    )
    mean_carrier = fusion.mean(mean).masked_fill(~tokens[..., None], 0)
    residual = fusion.output(merge_heads(attended)).masked_fill(
        ~tokens[..., None], 0
    )
    _parity("Core carrier", pre, mean_carrier + residual)
    if carrier not in {"full", "no_mean", "no_centered"}:
        raise WriterModelError("unknown Core carrier ablation")
    content = {
        "full": pre,
        "no_mean": residual,
        "no_centered": mean_carrier,
    }[carrier]
    blocks = []
    for block in writer.semantic_core.blocks:
        content = block(content, tokens)
        blocks.append(content)
    return {
        "mean": mean,
        "centered": centered,
        "mean_carrier": mean_carrier,
        "centered_residual": residual,
        "pre": pre,
        "blocks": blocks,
        "final": content,
        "frame_attention": observed_attention,
        "attention": {"frame": frame_parity},
    }


def _program_temporal_attention(
    block: torch.nn.Module,
    arguments: Sequence[torch.Tensor],
) -> dict[str, Any]:
    content, endpoints, valid, qk_identity = arguments
    batch, targets, intervals, roles, width = content.shape
    sequences = content.permute(0, 1, 3, 2, 4).reshape(
        batch * targets * roles, intervals, width
    )
    positions = endpoints[:, None, None].expand(
        batch, targets, roles, intervals
    ).reshape(batch * targets * roles, intervals)
    active = valid[:, None, None].expand(
        batch, targets, roles, intervals
    ).reshape(batch * targets * roles, intervals)
    identity = qk_identity[:, :, :, None].expand(
        batch, targets, roles, intervals, width
    ).reshape(batch * targets * roles, intervals, width)
    weights, rebuilt = _raw_attention(
        block.temporal_attention,
        sequences,
        active,
        positions,
        identity,
    )
    observed = block.temporal_attention(
        sequences,
        active,
        positions=positions,
        qk_identity=identity,
    )
    return {
        **probability_summary(weights, active),
        "output_parity": _attention_parity(
            "role temporal attention", observed, rebuilt
        ),
    }


def _decode(
    writer: CompleteLoRAWriter,
    coordinates: torch.Tensor,
) -> dict[str, Any]:
    head_rows: dict[str, list[tuple[int, torch.Tensor]]] = {
        name: [] for name in writer.factor_heads
    }
    factors: dict[str, torch.Tensor] = {}
    public: dict[str, torch.Tensor] = {}
    for spec in writer.tensor_specs:
        key, target = writer._decoding[spec.name]
        rows = writer.factor_heads[key](coordinates[:, target])
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


def _pack(
    value: torch.Tensor,
    offsets: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    trailing = value.shape[1:]
    maximum = max(right - left for left, right in zip(offsets, offsets[1:]))
    packed = value.new_zeros(len(offsets) - 1, maximum, *trailing)
    valid = torch.zeros(
        packed.shape[:2], dtype=torch.bool, device=value.device
    )
    for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
        packed[row, : right - left] = value[left:right]
        valid[row, : right - left] = True
    return packed, valid


def _run_with_capture(
    writer: CompleteLoRAWriter,
    policy: torch.nn.Module,
    frames: torch.Tensor,
    indices: torch.Tensor,
    offsets_tensor: torch.Tensor,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    spans: torch.Tensor,
) -> tuple[dict[str, list[Any]], Mapping[str, torch.Tensor]]:
    captured: dict[str, list[Any]] = {
        name: []
        for name in (
            "encoder",
            "core",
            "program",
            "program_input",
            "compiler",
            "raw_action",
            "block_input",
            "block_output",
        )
    }
    handles = [
        writer.semantic_encoder.register_forward_hook(
            lambda _module, _args, output: captured["encoder"].append(output)
        ),
        writer.semantic_core.register_forward_hook(
            lambda _module, _args, output: captured["core"].append(output)
        ),
        writer.semantic_program.register_forward_pre_hook(
            lambda _module, args: captured["program_input"].append(args)
        ),
        writer.semantic_program.register_forward_hook(
            lambda _module, _args, output: captured["program"].append(output)
        ),
        writer.compiler.register_forward_hook(
            lambda _module, _args, output: captured["compiler"].append(output)
        ),
        writer.semantic_encoder.interaction_projection.register_forward_pre_hook(
            lambda _module, args: captured["raw_action"].append(args[0])
        ),
    ]
    for block in writer.semantic_program.blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, args: captured["block_input"].append(args)
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _module, _args, output: captured["block_output"].append(
                    output
                )
            )
        )
    try:
        canonical = writer(
            frames,
            indices,
            offsets_tensor,
            tokens,
            masks,
            spans,
            policy=policy,
        )
    finally:
        for handle in handles:
            handle.remove()
    required = ("encoder", "core", "program", "program_input", "compiler")
    if any(len(captured[name]) != 1 for name in required):
        raise WriterModelError("internal-analysis Writer hooks changed")
    if (
        len(captured["block_input"]) != len(writer.semantic_program.blocks)
        or len(captured["block_output"]) != len(writer.semantic_program.blocks)
        or not captured["raw_action"]
    ):
        raise WriterModelError("internal-analysis Program hooks changed")
    return captured, canonical


@torch.inference_mode()
def capture_writer(
    writer: CompleteLoRAWriter,
    policy: torch.nn.Module,
    frames: torch.Tensor,
    indices: torch.Tensor,
    offsets_tensor: torch.Tensor,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    spans: torch.Tensor,
) -> dict[str, Any]:
    captured, canonical = _run_with_capture(
        writer,
        policy,
        frames,
        indices,
        offsets_tensor,
        tokens,
        masks,
        spans,
    )

    offsets = writer._validated_offsets(offsets_tensor, frames.shape[0])
    query, x, grounded, action, valid_tokens = captured["encoder"][0]
    packed_x, packed_grounded, packed_action, positions, valid_frames = (
        writer._pack_video_program(
            x, grounded, action, indices, offsets
        )
    )
    packed_raw, raw_valid = _pack(
        torch.cat(captured["raw_action"]), offsets
    )
    if not torch.equal(raw_valid, valid_frames):
        raise WriterModelError("native Action capture lost video alignment")
    core = _core_pipeline(
        writer, query, packed_x, valid_frames, valid_tokens
    )
    observed_core, observed_frame_attention = captured["core"][0]
    parity = {
        "core": _parity("Core final", observed_core, core["final"]),
        "frame_attention": _parity(
            "frame attention",
            observed_frame_attention,
            core["frame_attention"],
        ),
    }
    core["recomputed_final"] = core["final"]
    core["recomputed_frame_attention"] = core["frame_attention"]
    core["final"] = observed_core
    core["frame_attention"] = observed_frame_attention

    program_args = captured["program_input"][0]
    target_query, target_core = program_args[5], program_args[6]
    raw_program = captured["block_input"][0][0]
    observed_program = captured["program"][0]
    program = {
        "raw": raw_program,
        "blocks": list(captured["block_output"]),
        "memory": observed_program[0],
        "endpoints": observed_program[1],
        "valid_intervals": observed_program[2],
        "attention": [
            _program_temporal_attention(block, arguments)
            for block, arguments in zip(
                writer.semantic_program.blocks,
                captured["block_input"],
                strict=True,
            )
        ],
    }
    parity["program_memory"] = _parity(
        "Program final",
        captured["block_output"][-1],
        observed_program[0],
    )
    compiled = _compile(
        writer,
        core["final"],
        valid_tokens,
        program,
        target_context=(target_query, target_core),
    )
    parity["compiler_coordinates"] = _attention_parity(
        "compiler coordinates",
        captured["compiler"][0],
        compiled["coordinates"],
    )
    compiled["recomputed_coordinates"] = compiled["coordinates"]
    compiled["coordinates"] = captured["compiler"][0]
    decoded = _decode(writer, compiled["coordinates"])
    parity["public"] = mapping_metrics(canonical, decoded["public"])
    per_tensor = {
        name: relative_metrics(canonical[name], decoded["public"][name])
        for name in canonical
    }
    worst = max(per_tensor, key=lambda name: per_tensor[name]["relative_l2"])
    parity["public"]["max_tensor_relative_l2"] = per_tensor[worst][
        "relative_l2"
    ]
    if parity["public"]["max_tensor_relative_l2"] > PARITY_TOLERANCE:
        raise WriterModelError(
            f"canonical public decode parity failed for {worst}: "
            f"{per_tensor[worst]}"
        )
    return {
        "q": query,
        "m": packed_x - packed_grounded,
        "g": packed_grounded,
        "x": packed_x,
        "a_raw": packed_raw,
        "a": packed_action,
        "positions": positions,
        "valid_frames": valid_frames,
        "valid_tokens": valid_tokens,
        "core": core,
        "program": program,
        "compiled": compiled,
        "decoded": decoded,
        "canonical": canonical,
        "parity": parity,
    }


def _state(
    mapping: Mapping[str, torch.Tensor],
    row: int,
) -> dict[str, torch.Tensor]:
    return {name: value[row] for name, value in mapping.items()}


def _row(value: torch.Tensor, index: int = 0) -> torch.Tensor:
    return value[index : index + 1]


def _program_row(
    program: Mapping[str, Any],
    index: int,
    *,
    raw: torch.Tensor | None = None,
    memory: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return {
        "raw": _row(program["raw"] if raw is None else raw, index),
        "memory": _row(
            program["memory"] if memory is None else memory, index
        ),
        "endpoints": _row(program["endpoints"], index),
        "valid_intervals": _row(program["valid_intervals"], index),
    }


def _variant(
    writer: CompleteLoRAWriter,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    program: Mapping[str, torch.Tensor],
    *,
    target_permutation: torch.Tensor | None = None,
    rank_permutation: torch.Tensor | None = None,
    coordinates: torch.Tensor | None = None,
) -> dict[str, Any]:
    compiled = (
        _compile(
            writer,
            core,
            valid_core,
            program,
            target_permutation,
            rank_permutation,
        )
        if coordinates is None
        else {"coordinates": coordinates}
    )
    decoded = _decode(writer, compiled["coordinates"])
    return {
        "coordinates": compiled["coordinates"][0],
        "factor": _state(decoded["factors"], 0),
        "public": _state(decoded["public"], 0),
        "heads": {
            name: value[0] for name, value in decoded["heads"].items()
        },
        "attention": compiled.get("attention"),
    }


def _contextualize(
    writer: CompleteLoRAWriter,
    raw: torch.Tensor,
    endpoints: torch.Tensor,
    valid: torch.Tensor,
    target_query: torch.Tensor,
    target_core: torch.Tensor,
) -> torch.Tensor:
    roles = writer.semantic_program.role_identity_norm(
        writer.semantic_program.role_identity
    ).to(raw.dtype)
    identity = (
        target_query[:, :, None]
        + writer.semantic_program.core_norm(target_core)[:, :, None]
        + roles[None, None]
    )
    value = raw
    for block in writer.semantic_program.blocks:
        value = block(value, endpoints, valid, identity)
    return value


def _reverse_valid_intervals(
    memory: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    result = memory.clone()
    for row, count in enumerate(valid.sum(dim=1).tolist()):
        result[row, :, : int(count)] = result[
            row, :, : int(count)
        ].flip(1)
    return result


def _coordinate_variants(
    writer: CompleteLoRAWriter,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    base_program: Mapping[str, torch.Tensor],
    full: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    coordinates = full["coordinates"]
    width = writer.program_width
    return {
        "coordinate/core_only": _variant(
            writer,
            _row(core),
            _row(valid_core),
            base_program,
            coordinates=torch.cat(
                (
                    coordinates[..., :width],
                    torch.zeros_like(coordinates[..., width:]),
                ),
                dim=-1,
            )[None],
        ),
        "coordinate/program_only": _variant(
            writer,
            _row(core),
            _row(valid_core),
            base_program,
            coordinates=torch.cat(
                (
                    torch.zeros_like(coordinates[..., :width]),
                    coordinates[..., width:],
                ),
                dim=-1,
            )[None],
        ),
    }


def _program_role_variants(
    writer: CompleteLoRAWriter,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    program: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    memory = _row(program["memory"])
    reversed_memory = _reverse_valid_intervals(
        memory, _row(program["valid_intervals"])
    )
    variants = {
        "program_memory/order_reversed": _variant(
            writer,
            _row(core),
            _row(valid_core),
            _program_row(program, 0, memory=reversed_memory),
        )
    }
    for role, name in enumerate(ROLE_NAMES):
        removed = memory.clone()
        removed[..., role, :] = 0
        variants[f"program_role/remove_{name}"] = _variant(
            writer,
            _row(core),
            _row(valid_core),
            _program_row(program, 0, memory=removed),
        )
        selected = torch.zeros_like(memory)
        selected[..., role, :] = memory[..., role, :]
        variants[f"program_role/{name}_only"] = _variant(
            writer,
            _row(core),
            _row(valid_core),
            _program_row(program, 0, memory=selected),
        )
    return variants


def _program_input_variants(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
    core: torch.Tensor,
    valid_core: torch.Tensor,
    program: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    target_query = _row(captured["compiled"]["diagnostic"]["target_query"])
    target_core = _row(captured["compiled"]["diagnostic"]["core_read"])
    raw = _row(program["raw"])
    variants: dict[str, dict[str, Any]] = {}
    for role, name in enumerate(ROLE_NAMES):
        changed = raw.clone()
        changed[..., role, :] = 0
        contextual = _contextualize(
            writer,
            changed,
            _row(program["endpoints"]),
            _row(program["valid_intervals"]),
            target_query,
            target_core,
        )
        variants[f"program_input/remove_{name}"] = _variant(
            writer,
            _row(core),
            _row(valid_core),
            _program_row(program, 0, raw=changed, memory=contextual),
        )
    zero_action_program = writer.semantic_program(
        _row(captured["g"]),
        torch.zeros_like(_row(captured["a"])),
        _row(captured["positions"]),
        _row(captured["valid_frames"]),
        _row(valid_core),
        target_query,
        target_core,
    )
    variants["action_router/zero"] = _variant(
        writer,
        _row(core),
        _row(valid_core),
        {
            "memory": zero_action_program[0],
            "endpoints": zero_action_program[1],
            "valid_intervals": zero_action_program[2],
        },
    )
    return variants


def _condition_variants(
    writer: CompleteLoRAWriter,
    core: torch.Tensor,
    valid_core: torch.Tensor,
    program: Mapping[str, Any],
    base_program: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(CONDITIONS):
        variants[f"fixed_core/{condition}"] = _variant(
            writer,
            _row(core),
            _row(valid_core),
            _program_row(program, index),
        )
        variants[f"fixed_program/{condition}"] = _variant(
            writer,
            _row(core, index),
            _row(valid_core, index),
            base_program,
        )
    return variants


def _carrier_identity_variants(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
    core: torch.Tensor,
    valid_core: torch.Tensor,
    base_program: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for carrier in ("no_mean", "no_centered"):
        changed = _core_pipeline(
            writer,
            _row(captured["q"]),
            _row(captured["x"]),
            _row(captured["valid_frames"]),
            _row(valid_core),
            carrier,
        )
        variants[f"core_carrier/{carrier}"] = _variant(
            writer, changed["final"], _row(valid_core), base_program
        )
    target_permutation = torch.roll(
        torch.arange(writer.compiler.target_count, device=core.device), -1
    )
    rank_permutation = torch.roll(
        torch.arange(writer.compiler.rank, device=core.device), -1
    )
    variants["identity/target"] = _variant(
        writer,
        _row(core),
        _row(valid_core),
        base_program,
        target_permutation=target_permutation,
    )
    variants["identity/rank"] = _variant(
        writer,
        _row(core),
        _row(valid_core),
        base_program,
        rank_permutation=rank_permutation,
    )
    return variants


@torch.inference_mode()
def counterfactual_states(
    writer: CompleteLoRAWriter,
    captured: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Intervene at Core, role Program, rank read, and identity interfaces."""

    state_sha256 = lora_state_sha256(writer.semantic_program.state_dict())
    core = captured["core"]["final"]
    valid_core = captured["valid_tokens"]
    program = captured["program"]
    base_program = _program_row(program, 0)
    recomputed = _variant(
        writer, _row(core), _row(valid_core), base_program
    )
    full = _variant(
        writer,
        _row(core),
        _row(valid_core),
        base_program,
        coordinates=_row(captured["compiled"]["coordinates"]),
    )
    full["attention"] = recomputed["attention"]
    variants = {"full": full}
    variants.update(
        _coordinate_variants(writer, core, valid_core, base_program, full)
    )
    variants.update(_program_role_variants(writer, core, valid_core, program))
    variants.update(
        _program_input_variants(writer, captured, core, valid_core, program)
    )
    variants.update(
        _condition_variants(
            writer, core, valid_core, program, base_program
        )
    )
    variants.update(
        _carrier_identity_variants(
            writer, captured, core, valid_core, base_program
        )
    )
    if lora_state_sha256(writer.semantic_program.state_dict()) != state_sha256:
        raise WriterModelError("Program counterfactual mutated trained state")
    full["program_memory_authority"] = {
        "trained_program_state_sha256": state_sha256,
        "order_permutation": (
            "reverse each valid contextual role history while Core, endpoint "
            "positions, masks, and trained weights stay fixed"
        ),
        "key_value_coupling": (
            "each Action, Effect, and Change history has a private softmax; "
            "the same contextual role tensor supplies normalized K and raw V"
        ),
    }
    return variants
