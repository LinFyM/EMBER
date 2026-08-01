"""Canonical no-rollout internal analysis for the current AP-ADR AS-Writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from safetensors.torch import load_file

from ember.lora import (
    copy_task_lora_state_, lora_state_sha256, task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_eval_contract import (
    git_state, inspect_source_checkpoint, inspect_tokenizer,
    load_evaluation_authorities, resolve_role_task_keys,
)
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, load_stats
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.as_config import load_writer_config
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority, _camera
from ember.writer.inference import (
    expected_writer_episode_evidence, inspect_as_writer_evaluation,
    writer_shuffled_frame_permutation,
)
from ember.writer.internal_metrics import (
    CONDITIONS, attention_summary, change_retention, effective_ba_error,
    effective_metrics, effective_variance, fixed_sequence, lora_geometry,
    mapping_metrics, probability_summary, rank_gauge_permute, relative_metrics,
    routing_centered_energy, validate_finite_tree, variance_metrics,
)
from ember.writer.internal_results import (
    CONTROL_TIMEOUT, barrier, broadcast, create_control_group, finalize,
    lpt_assignment, record_failure, seal_rank_rows,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.semantic_program import apply_rope, apply_two_axis_rope, merge_heads, split_heads
from ember.writer.validation import _build_models


RUN_SCHEMA = "ember_as_writer_internal_analysis_run_v1"
RESULT_SCHEMA = "ember_as_writer_internal_analysis_v1"
ARCHITECTURE = "pi05_amplitude_preserving_asymmetric_dual_read_v1"
PARITY_TOLERANCE = 2e-5
# Explicit softmax/weighted-value reconstruction and CUDA BF16 SDPA are two
# mathematically equivalent attention implementations, but they round at
# different points.  One BF16 unit roundoff is the narrow numerical contract
# for diagnostics that reconstruct SDPA weights; canonical Writer, compiler,
# and public-LoRA parity retain the strict tolerance above.
ATTENTION_PARITY_TOLERANCE = 8e-3
PROTECTED = (
    "src/ember/writer/model.py", "src/ember/writer/video_program.py",
    "src/ember/writer/semantic_core.py", "src/ember/writer/semantic_program.py",
    "src/ember/writer/program_compiler.py", "src/ember/writer/architecture.py",
    "src/ember/writer/functional.py", "src/ember/lora.py", "src/ember/pi05_lora.py",
    "configs/pi05_as_writer_amplitude_dual_read_full24_decay400_v1.json",
)


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
    value, blocks, attention = raw, [], []
    for block in writer.semantic_program.blocks:
        value, diagnostic = _program_block(block, value, endpoints, intervals, semantics)
        blocks.append(value); attention.append(diagnostic)
    return {"raw": raw, "blocks": blocks, "key": value, "value": raw, "endpoints": endpoints, "valid_intervals": intervals, "valid_semantics": semantics, "attention": attention}


def _compile(writer: CompleteLoRAWriter, core: torch.Tensor, valid_core: torch.Tensor, program: Mapping[str, torch.Tensor], target_permutation: torch.Tensor | None = None, rank_permutation: torch.Tensor | None = None) -> dict[str, Any]:
    compiler = writer.compiler; key = program["key"]; value = program["value"]
    b, intervals, columns, width = key.shape
    target = compiler.target_identity_norm(compiler.target_identity)
    rank = compiler.rank_identity_norm(compiler.rank_identity)
    if target_permutation is not None: target = target.index_select(0, target_permutation)
    if rank_permutation is not None: rank = rank.index_select(0, rank_permutation)
    target_query = target[None].expand(b, -1, -1)
    program_query = (target_query[:, :, None] + rank[None, None]).reshape(b, compiler.target_count * compiler.rank, width)
    if target_permutation is None and rank_permutation is None:
        coordinates, diagnostic = compiler.compile_with_diagnostics(core, valid_core, key, value, program["endpoints"], program["valid_intervals"], program["valid_semantics"])
    else:
        core_read = compiler.core_reader(target_query, compiler.core_norm(core), core, valid_core)
        flat_key = key.reshape(b, intervals * columns, width); flat_value = value.reshape_as(flat_key)
        valid = (program["valid_intervals"][:, :, None] & program["valid_semantics"][:, None]).reshape(b, -1)
        first = program["endpoints"][:, :, None].expand(b, intervals, columns).reshape(b, -1)
        ordinal = compiler._semantic_ordinals(columns, key.device); second = ordinal[None, None].expand(b, intervals, columns).reshape(b, -1)
        identity = compiler._type_identity(columns).to(key.dtype)[None, None].expand(b, intervals, columns, width).reshape_as(flat_key)
        program_read = compiler.program_reader(program_query, compiler.program_norm(flat_key), flat_value, valid, memory_qk_identity=identity, endpoint_positions=first, semantic_positions=second).reshape(b, compiler.target_count, compiler.rank, width)
        coordinates = torch.cat((core_read[:, :, None].expand_as(program_read), program_read), -1)
        diagnostic = {"target_query": target_query, "program_query": program_query, "core_read": core_read, "program_read": program_read, "coordinates": coordinates}
    cq = split_heads(compiler.core_reader.query(target_query), compiler.core_reader.heads)
    ck = split_heads(compiler.core_reader.key(compiler.core_norm(core)), compiler.core_reader.heads)
    core_weights = _weights(cq, ck, valid_core[:, None, None])
    core_rebuilt = compiler.core_reader.output(merge_heads(core_weights @ split_heads(core, compiler.core_reader.heads)))
    flat_key = key.reshape(b, intervals * columns, width); flat_value = value.reshape_as(flat_key)
    valid = (program["valid_intervals"][:, :, None] & program["valid_semantics"][:, None]).reshape(b, -1)
    first = program["endpoints"][:, :, None].expand(b, intervals, columns).reshape(b, -1)
    ordinal = compiler._semantic_ordinals(columns, key.device); second = ordinal[None, None].expand(b, intervals, columns).reshape(b, -1)
    identity = compiler._type_identity(columns).to(key.dtype)[None, None].expand(b, intervals, columns, width).reshape_as(flat_key)
    pq = split_heads(compiler.program_reader.query(program_query), compiler.program_reader.heads)
    pk = split_heads(compiler.program_reader.key(compiler.program_norm(flat_key) + identity), compiler.program_reader.heads)
    zeros = torch.zeros(program_query.shape[:2], dtype=torch.long, device=key.device)
    pq = apply_two_axis_rope(pq, zeros, zeros); pk = apply_two_axis_rope(pk, first, second)
    program_weights = _weights(pq, pk, valid[:, None, None])
    program_rebuilt = compiler.program_reader.output(merge_heads(program_weights @ split_heads(flat_value, compiler.program_reader.heads))).reshape_as(diagnostic["program_read"])
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
    heads = {name: head(coordinates) for name, head in writer.factor_heads.items()}
    factors, public = {}, {}
    for spec in writer.tensor_specs:
        key, target = writer._decoding[spec.name]; rows = heads[key][:, target]
        generated = rows.transpose(-1, -2) if spec.transpose_output else rows
        factors[spec.name] = generated
        public[spec.name] = generated.to(getattr(writer, writer._template_buffers[spec.name]).dtype) + getattr(writer, writer._template_buffers[spec.name])[None]
    return {"heads": heads, "factors": factors, "public": public}


def _pack(value: torch.Tensor, offsets: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    width = value.shape[1:]; maximum = max(b - a for a, b in zip(offsets, offsets[1:]))
    packed = value.new_zeros(len(offsets) - 1, maximum, *width); valid = torch.zeros(packed.shape[:2], dtype=torch.bool, device=value.device)
    for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
        packed[row, : right - left] = value[left:right]; valid[row, : right - left] = True
    return packed, valid


@torch.inference_mode()
def capture_writer(writer: CompleteLoRAWriter, policy: torch.nn.Module, frames: torch.Tensor, indices: torch.Tensor, offsets_tensor: torch.Tensor, tokens: torch.Tensor, masks: torch.Tensor, spans: torch.Tensor) -> dict[str, Any]:
    captured: dict[str, list[Any]] = {name: [] for name in ("encoder", "core", "program", "raw_action")}
    handles = [
        writer.semantic_encoder.register_forward_hook(lambda _m, _a, out: captured["encoder"].append(out)),
        writer.semantic_core.register_forward_hook(lambda _m, _a, out: captured["core"].append(out)),
        writer.semantic_program.register_forward_hook(lambda _m, _a, out: captured["program"].append(out)),
        writer.semantic_encoder.interaction_projection.register_forward_pre_hook(lambda _m, args: captured["raw_action"].append(args[0].detach())),
    ]
    try: canonical = writer(frames, indices, offsets_tensor, tokens, masks, spans, policy=policy)
    finally:
        for handle in handles: handle.remove()
    if any(len(captured[name]) != 1 for name in ("encoder", "core", "program")) or not captured["raw_action"]:
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
    parity = {"core": _parity("Core final", observed_core, core["final"]), "frame_attention": _parity("frame attention", observed_frame_attention, core["frame_attention"]), "program_key": _parity("Program key", observed_program[0], program["key"]), "program_value": _parity("Program value", observed_program[1], program["value"])}
    if not all(torch.equal(a, b) for a, b in zip(observed_program[2:], (program["endpoints"], program["valid_intervals"], program["valid_semantics"]), strict=True)):
        raise WriterModelError("Program endpoint/mask parity failed")
    compiled = _compile(writer, core["final"], valid_tokens, program); decoded = _decode(writer, compiled["coordinates"])
    parity["public"] = mapping_metrics(canonical, decoded["public"])
    parity["public"]["max_tensor_relative_l2"] = max(relative_metrics(canonical[name], decoded["public"][name])["relative_l2"] for name in canonical)
    if parity["public"]["max_tensor_relative_l2"] > PARITY_TOLERANCE: raise WriterModelError("canonical public decode parity failed")
    return {"q": q, "m": packed_x - packed_g, "g": packed_g, "x": packed_x, "a_raw": packed_raw, "a": packed_action, "positions": positions, "valid_frames": valid_frames, "valid_tokens": valid_tokens, "core": core, "program": program, "compiled": compiled, "decoded": decoded, "canonical": canonical, "parity": parity}


def _state(mapping: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {name: value[row] for name, value in mapping.items()}


def _variant(writer: CompleteLoRAWriter, core: torch.Tensor, valid_core: torch.Tensor, program: Mapping[str, torch.Tensor], *, target_permutation: torch.Tensor | None = None, rank_permutation: torch.Tensor | None = None, coordinates: torch.Tensor | None = None) -> dict[str, Any]:
    compiled = _compile(writer, core, valid_core, program, target_permutation, rank_permutation) if coordinates is None else {"coordinates": coordinates}
    decoded = _decode(writer, compiled["coordinates"])
    return {"coordinates": compiled["coordinates"][0], "factor": _state(decoded["factors"], 0), "public": _state(decoded["public"], 0), "heads": {name: value[0] for name, value in decoded["heads"].items()}, "attention": compiled.get("attention")}


@torch.inference_mode()
def counterfactual_states(writer: CompleteLoRAWriter, captured: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Re-run every counterfactual through real Core/Program/readers/factor heads."""

    program_sha256 = lora_state_sha256(writer.semantic_program.state_dict())
    def row(value: torch.Tensor, index: int = 0) -> torch.Tensor: return value[index : index + 1]
    core = captured["core"]["final"]; valid = captured["valid_tokens"]; p = captured["program"]
    def program_row(index: int, *, key: torch.Tensor | None = None, value: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return {"key": row(p["key"] if key is None else key, index), "value": row(p["value"] if value is None else value, index), "endpoints": row(p["endpoints"], index), "valid_intervals": row(p["valid_intervals"], index), "valid_semantics": row(p["valid_semantics"], index)}
    full = _variant(writer, row(core), row(valid), program_row(0)); variants = {"full": full}
    permuted_key = row(p["key"]).clone(); interval_count = int(row(p["valid_intervals"]).sum())
    permuted_key[:, :interval_count] = permuted_key[:, :interval_count].flip(1)
    variants["temporal_keys/order_permuted"] = _variant(writer, row(core), row(valid), program_row(0, key=permuted_key))
    coordinate = full["coordinates"]
    variants["core_only"] = _variant(writer, row(core), row(valid), program_row(0), coordinates=torch.cat((coordinate[..., :256], torch.zeros_like(coordinate[..., 256:])), -1)[None])
    variants["program_only"] = _variant(writer, row(core), row(valid), program_row(0), coordinates=torch.cat((torch.zeros_like(coordinate[..., :256]), coordinate[..., 256:]), -1)[None])
    for index, condition in enumerate(CONDITIONS):
        variants[f"fixed_core/{condition}"] = _variant(writer, row(core), row(valid), program_row(index))
        variants[f"fixed_program/{condition}"] = _variant(writer, row(core, index), row(valid, index), program_row(0))
    raw = row(p["value"]); task_tokens = (raw.shape[2] - 1) // 2
    slices = {"A": slice(0, 1), "E": slice(1, 1 + task_tokens), "D": slice(1 + task_tokens, None)}
    for names in (("A",), ("E",), ("D",), ("A", "E"), ("A", "D"), ("E", "D"), ("A", "E", "D")):
        selected = torch.zeros_like(raw)
        for name in names: selected[:, :, slices[name]] = raw[:, :, slices[name]]
        rebuilt = _program_pipeline(writer, selected, row(p["endpoints"]), row(p["valid_intervals"]), row(p["valid_semantics"]))
        label = "+".join(names); variants[f"aed/{label}"] = _variant(writer, row(core), row(valid), rebuilt)
        fixed_key = {**rebuilt, "key": row(p["key"])}; variants[f"aed_fixed_key/{label}"] = _variant(writer, row(core), row(valid), fixed_key)
    for name, selected_slice in slices.items():
        for scale in (0.5, 1.0, 2.0):
            scaled = raw.clone(); scaled[:, :, selected_slice] *= scale
            rebuilt = _program_pipeline(writer, scaled, row(p["endpoints"]), row(p["valid_intervals"]), row(p["valid_semantics"]))
            variants[f"scale/{name}/{scale:g}"] = _variant(writer, row(core), row(valid), rebuilt)
    for carrier in ("no_mean", "no_centered"):
        changed = _core_pipeline(writer, row(captured["q"]), row(captured["x"]), row(captured["valid_frames"]), row(valid), carrier)
        variants[f"core_carrier/{carrier}"] = _variant(writer, changed["final"], row(valid), program_row(0))
    target_perm = torch.roll(torch.arange(writer.compiler.target_count, device=core.device), -1)
    rank_perm = torch.roll(torch.arange(writer.compiler.rank, device=core.device), -1)
    variants["identity/target"] = _variant(writer, row(core), row(valid), program_row(0), target_permutation=target_perm)
    variants["identity/rank"] = _variant(writer, row(core), row(valid), program_row(0), rank_permutation=rank_perm)
    for name in ("aed/A+E+D", "scale/A/1", "scale/E/1", "scale/D/1"):
        if max(relative_metrics(full["public"][key], variants[name]["public"][key])["relative_l2"] for key in full["public"]) > PARITY_TOLERANCE:
            raise WriterModelError(f"counterfactual full/scale1 parity failed: {name}")
    if lora_state_sha256(writer.semantic_program.state_dict()) != program_sha256:
        raise WriterModelError("temporal-key counterfactual mutated trained Program state")
    full["temporal_key_authority"] = {
        "trained_program_state_sha256": program_sha256,
        "order_permutation": "reverse valid contextual-key intervals only; Core, raw A/E/D, masks, and positions fixed",
        "initialization_keys": {"status": "unsupported", "fail_closed": True, "reason": "Program Linear Q/K/FFN initialization consumes an unsealed global torch RNG sequence; initialization_seed seals identities only"},
    }
    return variants


def _comparison(writer: CompleteLoRAWriter, reference: Mapping[str, Any], candidate: Mapping[str, Any], reference_action: torch.Tensor, candidate_action: torch.Tensor) -> dict[str, Any]:
    return {
        "coordinates": relative_metrics(reference["coordinates"], candidate["coordinates"]),
        "factor_heads": mapping_metrics(reference["heads"], candidate["heads"]),
        "factor": mapping_metrics(reference["factor"], candidate["factor"]),
        "public_a": mapping_metrics(reference["public"], candidate["public"], select="a"),
        "public_b": mapping_metrics(reference["public"], candidate["public"], select="b"),
        "effective_ba": effective_metrics(writer, reference["public"], candidate["public"]),
        "fixed_policy_action": relative_metrics(reference_action, candidate_action),
    }


def _signature(captured: Mapping[str, Any], row: int) -> dict[str, torch.Tensor]:
    vf = captured["valid_frames"][row]; vt = captured["valid_tokens"][row]
    vi = captured["program"]["valid_intervals"][row]; vs = captured["program"]["valid_semantics"][row]
    grid = vf[:, None] & vt[None]; program_grid = vi[:, None] & vs[None]
    result = {
        "q_text": fixed_sequence(captured["q"][row], vt),
        "multimodal_m": fixed_sequence(captured["m"][row], grid),
        "grounded_g": fixed_sequence(captured["g"][row], grid),
        "absolute_x": fixed_sequence(captured["x"][row], grid),
        "raw_action": fixed_sequence(captured["a_raw"][row], vf),
        "action_probe": fixed_sequence(captured["a"][row], vf),
        "core_frame_mean": fixed_sequence(captured["core"]["mean"][row], vt),
        "core_mean_carrier": fixed_sequence(captured["core"]["mean_carrier"][row], vt),
        "core_centered_residual": fixed_sequence(captured["core"]["centered_residual"][row], vt),
        "core_pre": fixed_sequence(captured["core"]["pre"][row], vt),
        "core_final": fixed_sequence(captured["core"]["final"][row], vt),
        "program_raw": fixed_sequence(captured["program"]["value"][row], program_grid),
        "program_key": fixed_sequence(captured["program"]["key"][row], program_grid),
        "core_read": captured["compiled"]["diagnostic"]["core_read"][row].float(),
        "program_read": captured["compiled"]["diagnostic"]["program_read"][row].float(),
        "coordinates": captured["compiled"]["coordinates"][row].float(),
    }
    for index, value in enumerate(captured["core"]["blocks"], 1): result[f"core_block_{index}"] = fixed_sequence(value[row], vt)
    for index, value in enumerate(captured["program"]["blocks"], 1): result[f"program_block_{index}"] = fixed_sequence(value[row], program_grid)
    return result


@torch.inference_mode()
def _paired_diagnostics(writer: CompleteLoRAWriter, captured: Mapping[str, Any], actions: Sequence[torch.Tensor]) -> dict[str, Any]:
    reference = _signature(captured, 0); reference_factor = _state(captured["decoded"]["factors"], 0)
    reference_heads = _state(captured["decoded"]["heads"], 0); reference_public = _state(captured["decoded"]["public"], 0); comparisons, attention = {}, {}
    for row, condition in enumerate(CONDITIONS):
        signature = _signature(captured, row)
        stages = {name: relative_metrics(reference[name], signature[name]) for name in reference}
        heads = _state(captured["decoded"]["heads"], row); factor = _state(captured["decoded"]["factors"], row); public = _state(captured["decoded"]["public"], row)
        effective = effective_metrics(writer, reference_public, public)
        action = relative_metrics(actions[0], actions[row])
        chain = [stages["program_raw"]["relative_l2"], stages["program_key"]["relative_l2"], stages["program_read"]["relative_l2"], effective["relative_l2"], action["relative_l2"]]
        comparisons[condition] = {
            "stages": stages, "factor_heads": mapping_metrics(reference_heads, heads), "factor": mapping_metrics(reference_factor, factor),
            "public_a": mapping_metrics(reference_public, public, select="a"),
            "public_b": mapping_metrics(reference_public, public, select="b"),
            "effective_ba": effective, "fixed_policy_action": action,
            "change_retention": {
                "raw_to_key": change_retention(chain[0], chain[1]),
                "key_to_program_read": change_retention(chain[1], chain[2]),
                "program_read_to_ba": change_retention(chain[2], chain[3]),
                "ba_to_action": change_retention(chain[3], chain[4]),
            },
        }
        vf = captured["valid_frames"][row : row + 1]; vt = captured["valid_tokens"][row : row + 1]
        frame = captured["core"]["frame_attention"][row : row + 1].permute(0, 1, 3, 2)
        program = {name: value[row : row + 1] for name, value in captured["program"].items() if isinstance(value, torch.Tensor)}
        device = captured["core"]["final"].device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            reader = _compile(
                writer,
                captured["core"]["final"][row : row + 1],
                vt,
                program,
            )
        attention[condition] = {"core_frame": probability_summary(frame, vt), "compiler": reader["attention"]}
    return {"comparisons": comparisons, "attention": attention, "program_blocks_five_condition_batch": captured["program"]["attention"]}


def fixed_policy_query(authority: WriterTaskAuthority, processor: Pi05LiberoProcessor, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read only demo0/frame0 observations; never open an actions dataset."""
    with h5py.File(authority.path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        base_array = np.asarray(obs["agentview_rgb"][0]); wrist_array = np.asarray(obs["eye_in_hand_rgb"][0])
        state_array = np.concatenate((np.asarray(obs["ee_states"][0], dtype=np.float32), np.asarray(obs["gripper_states"][0], dtype=np.float32)))
    base = torch.from_numpy(_camera(base_array))[None].to(device, dtype=torch.float32).div_(255)
    wrist = torch.from_numpy(_camera(wrist_array))[None].to(device, dtype=torch.float32).div_(255)
    states = torch.from_numpy(state_array)[None].to(device)
    tokens, masks = processor._tokenize_prompts(states, [authority.language])
    identity = {"demo_index": 0, "frame_index": 0, "observation_only": True, "actions_dataset_opened": False}
    for name, value in (("base", base_array), ("wrist", wrist_array), ("state", state_array)):
        identity[f"{name}_sha256"] = hashlib.sha256(np.ascontiguousarray(value)).hexdigest()
    return {"observation.images.base_0_rgb": base, "observation.images.left_wrist_0_rgb": wrist, OBS_LANGUAGE_TOKENS: tokens, OBS_LANGUAGE_ATTENTION_MASK: masks}, identity


def policy_action(policy: torch.nn.Module, processor: Pi05LiberoProcessor, prepared: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], identity: Mapping[str, torch.Tensor], lora: Any, seed: int, device: torch.device) -> torch.Tensor:
    identity_sha = lora_state_sha256(identity)
    if lora_state_sha256(task_lora_state_dict(policy)) != identity_sha: raise WriterModelError("source policy LoRA was not identity before fixed query")
    copy_task_lora_state_(policy, state, lora)
    noise = torch.randn(1, int(policy.model.config.chunk_size), int(policy.model.config.max_action_dim), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32).to(device)
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = policy.predict_action_chunk(dict(prepared), noise=noise, num_steps=10)
    finally:
        copy_task_lora_state_(policy, identity, lora)
    if lora_state_sha256(task_lora_state_dict(policy)) != identity_sha: raise WriterModelError("source policy LoRA identity restore failed")
    return processor.unnormalize_action(value).detach()


def _condition_capture(task: Mapping[str, Any], reference: int, adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore, tokenizer: Pi05TeacherPrefixTokenizer, policy: torch.nn.Module, writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames, indices, metadata = [], [], []
    for condition in CONDITIONS:
        evidence = expected_writer_episode_evidence(adapters[condition], suite=str(task["suite"]), task_id=int(task["task_id"]), init_state_id=reference, lora_sha256="0" * 64)
        video = store.load(int(evidence["video_global_task_id"]), int(evidence["teacher_demo_index"]))
        value = torch.from_numpy(video.frames).to(device, non_blocking=True); order = torch.arange(value.shape[0])
        if condition == "reversed": order = order.flip(0); value = value.flip(0)
        elif condition == "shuffled":
            order = writer_shuffled_frame_permutation(value.shape[0], int(evidence["teacher_video_order_seed"]), keep_first=False); value = value.index_select(0, order.to(device))
        frames.append(value); indices.append(torch.from_numpy(video.frame_indices).to(device, non_blocking=True))
        metadata.append({"condition": condition, "video_global_task_id": int(evidence["video_global_task_id"]), "teacher_demo_index": int(evidence["teacher_demo_index"]), "sampled_frames": int(value.shape[0]), "raw_frames": int(video.raw_frame_count), "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
    offsets = [0]
    for value in frames: offsets.append(offsets[-1] + value.shape[0])
    language = [str(task["language"])] * len(CONDITIONS); tokens, masks, spans = tokenizer(language)
    copy_task_lora_state_(policy, identity, lora)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        captured = capture_writer(
            writer,
            policy,
            torch.cat(frames),
            torch.cat(indices),
            torch.tensor(offsets, dtype=torch.long, device=device),
            tokens,
            masks,
            spans,
        )
    return captured, metadata


def probe_reference(task: Mapping[str, Any], reference: int, adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore, authority: WriterTaskAuthority, tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor, policy: torch.nn.Module, writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any, device: torch.device, *, replay: bool) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    captured, metadata = _condition_capture(task, reference, adapters, store, tokenizer, policy, writer, identity, lora, device)
    prepared, query_identity = fixed_policy_query(authority, processor, device)
    seed = int.from_bytes(hashlib.sha256(json.dumps(["as_writer_fixed_action", int(task["global_task_id"])], separators=(",", ":")).encode()).digest()[:8], "big") & ((1 << 63) - 1)
    states = [_state(captured["decoded"]["public"], row) for row in range(len(CONDITIONS))]
    for state in states: validate_lora_state(state, lora)
    actions = [policy_action(policy, processor, prepared, state, identity, lora, seed, device) for state in states]
    matched = _paired_diagnostics(writer, captured, actions)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        variants = counterfactual_states(writer, captured)
    variant_actions = {name: policy_action(policy, processor, prepared, value["public"], identity, lora, seed, device) for name, value in variants.items()}
    reference_variant = variants["full"]; reference_action = variant_actions["full"]
    counterfactuals = {name: {"relative_to_full": _comparison(writer, reference_variant, value, reference_action, variant_actions[name]), "geometry": lora_geometry(writer, value["public"]), "compiler_attention": value["attention"]} for name, value in variants.items()}
    temporal_permuted = counterfactuals["temporal_keys/order_permuted"]
    temporal_keys = {**reference_variant["temporal_key_authority"], "trained": {"program_reader_routing": reference_variant["attention"]["program_target_rank_routing"], "effective_ba": lora_geometry(writer, reference_variant["public"]), "fixed_action": relative_metrics(reference_action, reference_action)}, "order_permuted": {"program_reader_routing": variants["temporal_keys/order_permuted"]["attention"]["program_target_rank_routing"], "effective_ba_relative_to_trained": temporal_permuted["relative_to_full"]["effective_ba"], "fixed_action_relative_to_trained": temporal_permuted["relative_to_full"]["fixed_policy_action"]}}
    permutation = torch.roll(torch.arange(writer.PUBLIC_LORA_RANK, device=device), -1)
    gauge, raw_changes = rank_gauge_permute(writer, states[0], permutation); gauge_action = policy_action(policy, processor, prepared, gauge, identity, lora, seed, device)
    gauge_error = effective_ba_error(writer, states[0], gauge)
    if gauge_error["relative_l2"] > PARITY_TOLERANCE: raise WriterModelError("public rank gauge changed effective BA")
    replay_result = {"executed": False}
    if replay:
        repeated, repeated_metadata = _condition_capture(task, reference, adapters, store, tokenizer, policy, writer, identity, lora, device)
        repeated_state = _state(repeated["decoded"]["public"], 0); repeated_action = policy_action(policy, processor, prepared, repeated_state, identity, lora, seed, device)
        state_error = effective_ba_error(writer, states[0], repeated_state); action_error = relative_metrics(actions[0], repeated_action)
        if repeated_metadata != metadata or state_error["relative_l2"] > PARITY_TOLERANCE or action_error["relative_l2"] > PARITY_TOLERANCE: raise WriterModelError("internal-analysis deterministic replay failed")
        replay_result = {"executed": True, "effective_ba": state_error, "fixed_policy_action": action_error}
    row = {
        "global_task_id": int(task["global_task_id"]), "suite": str(task["suite"]), "task_id": int(task["task_id"]), "reference_ordinal": reference,
        "conditions": metadata, **matched, "canonical_parity": captured["parity"],
        "lora_geometry": {condition: lora_geometry(writer, state) for condition, state in zip(CONDITIONS, states, strict=True)},
        "counterfactuals": counterfactuals, "temporal_key_counterfactual": temporal_keys, "rank_gauge": {"permutation": permutation.cpu().tolist(), "raw_changes": raw_changes, "effective_ba_error": gauge_error, "fixed_policy_action_error": relative_metrics(actions[0], gauge_action)},
        "fixed_policy_query": query_identity, "fixed_policy_action_seed": seed, "deterministic_replay": replay_result,
        "information_wall": {"teacher_action_values_read": 0, "teacher_state_values_sent_to_writer": 0, "teacher_reward_or_terminal_values_read": 0, "policy_query_observation_state_sent_to_writer": 0},
    }
    validate_finite_tree(row)
    return row, {name: value.detach().cpu() for name, value in states[0].items()}, actions[0].detach().cpu()


def _provenance(repo: Path, state: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    training_commit = str(training.get("git", {}).get("commit", "")); head = str(state.get("commit", ""))
    if len(training_commit) != 40 or len(head) != 40 or state.get("dirty_paths") or state.get("origin_main") != head:
        raise WriterModelError("analysis Git authority is not clean pushed origin/main")
    if subprocess.run(["git", "merge-base", "--is-ancestor", training_commit, head], cwd=repo).returncode:
        raise WriterModelError("analysis code is not descended from training code")
    changed = subprocess.run(["git", "diff", "--name-only", f"{training_commit}..{head}", "--", *PROTECTED], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    if changed: raise WriterModelError(f"trained AP-ADR topology changed after checkpoint: {changed}")
    return {"analysis_commit": head, "training_commit": training_commit, "training_is_ancestor": True, "protected_paths_unchanged": list(PROTECTED)}


def _task_records(repo: Path, config: Mapping[str, Any], data_root: Path, adapters: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = read_json(repo / str(config["authorities"]["target_data_manifest"]["path"])); validation = set(int(value) for value in target["summary"]["roles"]["validation"])
    needed = {int(item[key]) for adapter in adapters.values() for item in adapter["task_video_mapping"] for key in ("language_global_task_id", "video_global_task_id")}
    tasks, authorities = [], []
    for item in target["tasks"]:
        global_id = int(item["global_task_id"]); path = (data_root / str(item["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root): raise WriterModelError("internal-analysis HDF5 escaped data root")
        if global_id in needed: authorities.append({"task_id": global_id, "language": str(item["language"]), "path": str(path), "expected_bytes": int(item["hdf5"]["bytes"])})
        if global_id in validation: tasks.append({**item, "path": str(path)})
    tasks.sort(key=lambda item: (SUITE_ORDER.index(str(item["suite"])), int(item["task_id"])))
    if len(tasks) != 8 or {item["task_id"] for item in authorities} != needed: raise WriterModelError("internal-analysis task authority changed")
    return tasks, authorities


def _task_costs(tasks: Sequence[Mapping[str, Any]], adapters: Mapping[str, Mapping[str, Any]], references: int) -> dict[int, int]:
    result = {}
    for task in tasks:
        cost = 0
        for reference in range(references):
            for condition in CONDITIONS:
                evidence = expected_writer_episode_evidence(adapters[condition], suite=str(task["suite"]), task_id=int(task["task_id"]), init_state_id=reference, lora_sha256="0" * 64)
                table = adapters[condition]["video_data"]["sampled_frame_counts_by_task"]
                task_table = table.get(str(evidence["video_global_task_id"]), table.get(int(evidence["video_global_task_id"])))
                cost += int(task_table.get(str(evidence["teacher_demo_index"]), task_table.get(int(evidence["teacher_demo_index"]))))
        result[int(task["global_task_id"])] = cost
    return result


def _inspect(args: argparse.Namespace, authorities: Any, task_keys: Sequence[tuple[str, int]]) -> dict[str, Any]:
    if args.output_dir.exists(): raise WriterModelError("internal-analysis output root already exists")
    config = load_writer_config(args.config); training = read_json(args.training_run / "run_contract.json")
    if args.training_run != args.checkpoint.parent.parent or config["writer"]["architecture"] != ARCHITECTURE or training["config_sha256"] != sha256_file(args.config):
        raise WriterModelError("internal-analysis checkpoint/config authority changed")
    state = git_state(args.repo); source = inspect_source_checkpoint(authorities, args.source_run, args.source_checkpoint, evaluation_mode="formal")
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    adapters = {condition: inspect_as_writer_evaluation(config_path=args.config, checkpoint=args.checkpoint, video_data_root=args.data_root, source=source, task_keys=task_keys, video_condition=condition, video_seed=args.video_seed, require_formal=True, video_sampling_mode="without_replacement") for condition in CONDITIONS}
    tasks, task_authorities = _task_records(args.repo, config, args.data_root, adapters)
    costs = _task_costs(tasks, adapters, args.references_per_task); assignment = lpt_assignment(costs)
    checkpoint_hashes = {name: sha256_file(args.checkpoint / name) for name in ("checkpoint_manifest.json", "writer.safetensors")}
    return {"git": state, "provenance": _provenance(args.repo, state, training), "config": config, "training": training, "source": source, "tokenizer": tokenizer, "adapters": adapters, "tasks": tasks, "task_authorities": task_authorities, "task_costs": costs, "assignment": assignment, "checkpoint_hashes": checkpoint_hashes}


def _publish(args: argparse.Namespace, payload: Mapping[str, Any], *, rank: int, world_size: int, group: Any | None) -> None:
    if rank == 0:
        args.output_dir.mkdir(parents=True)
        files = ("scripts/analyze_as_writer.py", "src/ember/writer/internal_analysis.py", "src/ember/writer/internal_metrics.py", "src/ember/writer/internal_results.py")
        write_json_atomic(args.output_dir / "run_contract.json", {
            "schema_version": RUN_SCHEMA, "host": socket.gethostname(), "command": list(os.sys.argv), "git": payload["git"], "provenance": payload["provenance"],
            "analysis_code": {name: sha256_file(args.repo / name) for name in files}, "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
            "training_run": {"path": str(args.training_run), "contract_sha256": canonical_hash(payload["training"])}, "checkpoint": {"path": str(args.checkpoint), **payload["checkpoint_hashes"]},
            "source": payload["source"], "tokenizer": payload["tokenizer"], "conditions": list(CONDITIONS), "adapter_sha256": {name: canonical_hash(value) for name, value in payload["adapters"].items()}, "references_per_task": args.references_per_task,
            "video_seed": args.video_seed, "video_sampling": "without_replacement", "world_size": world_size, "physical_gpu_ids": [4, 5, 6, 7], "task_costs": payload["task_costs"], "task_assignment": payload["assignment"], "task_assignment_sha256": canonical_hash({"costs": payload["task_costs"], "assignment": payload["assignment"]}),
            "distributed_control": {"backend": "gloo", "timeout_seconds": int(CONTROL_TIMEOUT.total_seconds())}, "rollouts": 0, "teacher_action_values_read": 0, "teacher_state_values_sent_to_writer": 0, "fixed_policy_query": "validation HDF5 observation-only demo0/frame0 after Writer LoRA generation",
        })
    barrier(world_size, group)


def _local_rows(args: argparse.Namespace, context: Any, payload: Mapping[str, Any], policy: torch.nn.Module, writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any, store: RawTeacherVideoStore, tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor) -> list[dict[str, Any]]:
    tasks = {int(value["global_task_id"]): value for value in payload["tasks"]}; authorities = {value.task_id: value for value in store.authorities.values()}
    rows = []
    for task_id in payload["assignment"][context.rank]:
        task_rows, states, actions = [], [], []
        for reference in range(args.references_per_task):
            row, state, action = probe_reference(tasks[task_id], reference, payload["adapters"], store, authorities[task_id], tokenizer, processor, policy, writer, identity, lora, context.device, replay=reference == 0)
            task_rows.append(row); states.append(state); actions.append(action)
        variance = {"effective_ba": effective_variance(writer, states), "fixed_policy_action": variance_metrics(actions)}
        for row in task_rows: row["same_task_video_variance"] = variance
        rows.extend(task_rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "training-run", "checkpoint", "source-run", "source-checkpoint", "tokenizer-path", "data-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--references-per-task", type=int, default=1); parser.add_argument("--video-seed", type=int, default=7)
    result = parser.parse_args(argv)
    for name in ("repo", "config", "training_run", "checkpoint", "source_run", "source_checkpoint", "tokenizer_path", "data_root", "output_dir"): setattr(result, name, getattr(result, name).resolve())
    if not 1 <= result.references_per_task <= 50 or result.video_seed != 7: raise WriterModelError("invalid internal-analysis reference schedule")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv); repo = Path(__file__).resolve().parents[3]
    if args.repo != repo or args.config != repo / "configs/pi05_as_writer_amplitude_dual_read_full24_decay400_v1.json": raise WriterModelError("internal-analysis checkout/config is not canonical AP-ADR")
    context = initialize_distributed(); visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if context.world_size != 4 or visible != ["4", "5", "6", "7"]: raise WriterModelError("formal internal analysis requires four ranks on physical GPUs4-7")
    group = create_control_group(context.world_size); authorities = load_evaluation_authorities(repo / "configs/pi05_target_evaluation_v1.json", repo); task_keys = resolve_role_task_keys(authorities.protocol, "validation")
    inspected: Any = None
    if context.is_main:
        try: inspected = _inspect(args, authorities, task_keys)
        except Exception as error: inspected = {"error": repr(error)}
    payload = broadcast(inspected, rank=context.rank, world_size=context.world_size, group=group); _publish(args, payload, rank=context.rank, world_size=context.world_size, group=group)
    policy, writer, lora, identity = _build_models(training=payload["training"], source=payload["source"], context=context)
    writer.load_state_dict(load_file(str(args.checkpoint / "writer.safetensors"), device=str(context.device)), strict=True); writer.eval()
    source_config = authorities.source_base_config; processor = Pi05LiberoProcessor(load_stats(source_config, source_config["data"]["active_task_ids"]), args.tokenizer_path, int(source_config["features"]["tokenizer_max_length"]), str(context.device)); tokenizer = Pi05TeacherPrefixTokenizer(args.tokenizer_path, int(source_config["features"]["tokenizer_max_length"]), str(context.device))
    task_authorities = tuple(WriterTaskAuthority(task_id=int(value["task_id"]), language=str(value["language"]), path=Path(value["path"]), expected_bytes=int(value["expected_bytes"])) for value in payload["task_authorities"]); store = RawTeacherVideoStore(task_authorities, frame_stride=int(payload["config"]["writer"]["frame_stride"]), max_open_files=2); started = time.monotonic()
    try:
        rows = _local_rows(args, context, payload, policy, writer, identity, lora, store, tokenizer, processor)
        if any(sha256_file(args.checkpoint / name) != digest for name, digest in payload["checkpoint_hashes"].items()): raise WriterModelError("analysis mutated checkpoint files")
        for row in rows: row["checkpoint_files_unchanged"] = True
        seal_rank_rows(args.output_dir, rank=context.rank, world_size=context.world_size, assigned_task_ids=payload["assignment"][context.rank], rows=rows, group=group)
        finalize(args.output_dir, rank=context.rank, world_size=context.world_size, references_per_task=args.references_per_task, result_schema=RESULT_SCHEMA, started=started, group=group)
    except Exception as error:
        record_failure(args.output_dir, context.rank, error); raise
    finally: store.close()
    if group is not None: dist.destroy_process_group(group)
    dist.destroy_process_group()
