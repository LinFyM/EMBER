"""Pure metrics and tensor diagnostics for canonical UCP internal analysis."""

from __future__ import annotations

import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.ucp_geometry import (
    aggregate_effective_ba_spectra,
    component_coordinate_geometry,
    effective_ba_spectra,
)
from ember.writer.semantic_program import apply_two_axis_rope, split_heads

CONDITIONS = (
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "shuffled",
    "reversed",
)
STAGES = (
    "q_text",
    "multimodal_m",
    "grounded_g",
    "absolute_x",
    "raw_action",
    "action_probe",
    "initial_program",
    "initial_x",
    "initial_a",
    "initial_d",
    "program_block_1",
    "program_block_2",
    "final_program",
    "final_x",
    "final_a",
    "final_d",
    "coordinates",
)
COUNTERFACTUAL_CONTRACT = {
    "type_ablations": ["full", "x_only", "dynamic_only", "a_only", "d_only"],
    "fixed_x_vary_a_d": (
        "hold correct X and endpoint grid; interpolate A intervals separately; "
        "resample cumulative G-change trajectory then re-difference D so total "
        "change is conserved; preserve same-length A/D exactly"
    ),
    "fixed_a_d_vary_x": (
        "hold correct A/D and its endpoint grid; linearly resample each "
        "condition's X intervals to the correct interval count"
    ),
    "dynamic_scale": {
        "values": [0.5, 1.0, 2.0],
        "scaled_values": "A and D before both Program blocks",
        "fixed_values": "correct-video X",
    },
    "target_identity_permutation": (
        "deterministically permute identities assigned to target query slots; "
        "retain the real 38-target coordinate-to-public-LoRA decode mapping"
    ),
    "rank_gauge_permutation": (
        "apply one deterministic permutation to every public A row and the "
        "matching B column; raw factors change while complete BA stays invariant"
    ),
}


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=check, text=True,
        capture_output=True,
    ).stdout.strip()


def validate_analysis_provenance(
    *, repo: Path, state: Mapping[str, Any], training: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a clean descendant whose trained model/config owners are unchanged."""

    training_commit = str(training.get("git", {}).get("commit", ""))
    head = str(state.get("commit", ""))
    if (
        len(training_commit) != 40
        or len(head) != 40
        or state.get("dirty_paths")
        or state.get("origin_main") != head
    ):
        raise WriterModelError("analysis Git authority is not clean origin/main")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", training_commit, head],
        cwd=repo, text=True, capture_output=True,
    )
    if ancestor.returncode != 0:
        raise WriterModelError("analysis code is not descended from training code")
    protected = (
        "src/ember/writer/model.py",
        "src/ember/writer/video_program.py",
        "src/ember/writer/semantic_program.py",
        "src/ember/writer/program_compiler.py",
        "src/ember/writer/architecture.py",
        "src/ember/writer/functional.py",
        "src/ember/writer/inference.py",
        "src/ember/writer/video_schedule.py",
        "src/ember/writer/validation.py",
        "src/ember/writer/data.py",
        "src/ember/lora.py",
        "src/ember/pi05_lora.py",
        "src/ember/pi05_eval_contract.py",
        "src/ember/pi05_processing.py",
        "src/ember/pi05_source_checkpoint.py",
        "src/ember/pi05_source_setup.py",
        "src/ember/pi05_target_data.py",
        "configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json",
    )
    runtime_compatibility = (
        "src/ember/writer/as_config.py",
        "src/ember/writer/as_contract.py",
        "src/ember/writer/checkpoint.py",
    )
    changed = _git(repo, "diff", "--name-only", f"{training_commit}..{head}", "--", *protected)
    if changed:
        raise WriterModelError("trained UCP model/config changed after checkpoint")
    return {
        "analysis_commit": head,
        "training_commit": training_commit,
        "training_is_ancestor": True,
        "protected_paths_unchanged": list(protected),
        "runtime_compatibility_paths_validated_by_training_contract": list(
            runtime_compatibility
        ),
    }


def relative_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left, right = reference.float().reshape(-1), candidate.float().reshape(-1)
    if left.shape != right.shape or not left.numel():
        raise WriterModelError("paired analysis signatures changed shape")
    left_energy = float(left.square().sum())
    right_energy = float(right.square().sum())
    difference = float((right - left).square().sum())
    dot = float((left * right).sum())
    return {
        "relative_l2": math.sqrt(difference / max(left_energy, 1e-24)),
        "cosine": dot / max(math.sqrt(left_energy * right_energy), 1e-24),
        "reference_rms": math.sqrt(left_energy / left.numel()),
        "candidate_rms": math.sqrt(right_energy / right.numel()),
    }


def mapping_metrics(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    select: str = "all",
) -> dict[str, float]:
    names = sorted(reference)
    if names != sorted(candidate) or select not in {"all", "a", "b"}:
        raise WriterModelError("paired LoRA mapping changed")
    if select != "all":
        marker = f".lora_{select.upper()}.default.weight"
        names = [name for name in names if name.endswith(marker)]
    left_energy = right_energy = difference = dot = count = 0.0
    for name in names:
        left, right = reference[name].float(), candidate[name].float()
        if left.shape != right.shape:
            raise WriterModelError("paired LoRA tensor shape changed")
        left_energy += float(left.square().sum())
        right_energy += float(right.square().sum())
        difference += float((right - left).square().sum())
        dot += float((left * right).sum())
        count += left.numel()
    return {
        "relative_l2": math.sqrt(difference / max(left_energy, 1e-24)),
        "cosine": dot / max(math.sqrt(left_energy * right_energy), 1e-24),
        "reference_rms": math.sqrt(left_energy / max(count, 1.0)),
        "candidate_rms": math.sqrt(right_energy / max(count, 1.0)),
    }


def fixed_sequence(value: torch.Tensor, valid: torch.Tensor, bins: int = 16) -> torch.Tensor:
    selected = value[valid].float()
    if selected.ndim < 2 or selected.shape[0] == 0:
        raise WriterModelError("empty UCP sequence signature")
    flat = selected.reshape(selected.shape[0], -1)
    if flat.shape[0] != bins:
        flat = F.interpolate(flat.T[None], size=bins, mode="linear", align_corners=True)[0].T
    return flat


def pack_flat(
    value: torch.Tensor, offsets: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [right - left for left, right in zip(offsets, offsets[1:])]
    packed = value.new_zeros(len(lengths), max(lengths), *value.shape[1:])
    valid = torch.zeros(
        len(lengths), max(lengths), dtype=torch.bool, device=value.device
    )
    for row, (left, right) in enumerate(zip(offsets, offsets[1:])):
        packed[row, : right - left] = value[left:right]
        valid[row, : right - left] = True
    return packed, valid


def type_ablation(initial: torch.Tensor, kind: str) -> torch.Tensor:
    task_tokens = (initial.shape[2] - 1) // 2
    result = initial.clone()
    if kind == "x_only":
        result[:, :, task_tokens:] = 0
    elif kind == "dynamic_only":
        result[:, :, :task_tokens] = 0
    elif kind == "a_only":
        result[:, :, :task_tokens] = 0
        result[:, :, task_tokens + 1 :] = 0
    elif kind == "d_only":
        result[:, :, : task_tokens + 1] = 0
    else:
        raise WriterModelError("unknown UCP type ablation")
    return result


def program_signature(
    program: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
    *,
    kind: str = "all",
) -> torch.Tensor:
    task_tokens = (program.shape[1] - 1) // 2
    if kind == "x":
        selected = program[:, :task_tokens]
        semantic = valid_semantics[:task_tokens]
    elif kind == "a":
        selected = program[:, task_tokens : task_tokens + 1]
        semantic = valid_semantics[task_tokens : task_tokens + 1]
    elif kind == "d":
        selected = program[:, task_tokens + 1 :]
        semantic = valid_semantics[task_tokens + 1 :]
    elif kind == "all":
        selected, semantic = program, valid_semantics
    else:
        raise WriterModelError("unknown UCP Program value type")
    selected = selected[:, semantic]
    return fixed_sequence(selected, valid_intervals)


def build_initial_program(
    absolute_x: torch.Tensor,
    grounded_g: torch.Tensor,
    action: torch.Tensor,
    frame_positions: torch.Tensor,
    valid_frames: torch.Tensor,
    valid_task_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if absolute_x.shape != grounded_g.shape or action.shape != absolute_x.shape[:2] + absolute_x.shape[-1:]:
        raise WriterModelError("UCP evidence shape changed")
    valid_intervals = valid_frames[:, :-1] & valid_frames[:, 1:]
    endpoint_positions = frame_positions[:, 1:]
    dynamic_d = grounded_g[:, 1:] - grounded_g[:, :-1]
    program = torch.cat((absolute_x[:, :-1], action[:, :-1, None], dynamic_d), dim=2)
    valid_semantics = torch.cat(
        (valid_task_tokens, torch.ones_like(valid_task_tokens[:, :1]), valid_task_tokens), dim=1
    )
    valid = valid_intervals[:, :, None] & valid_semantics[:, None]
    return program.masked_fill(~valid[..., None], 0), endpoint_positions, valid_intervals, valid_semantics


def validate_canonical_program_parity(
    canonical: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ],
    reconstructed: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ],
    *,
    tolerance: float = 2e-5,
) -> dict[str, Any]:
    """Prove manual diagnostics reconstruct the actual semantic_program owner."""

    (
        canonical_final, canonical_endpoints, canonical_intervals,
        canonical_semantics, canonical_coordinates,
    ) = canonical
    final, endpoints, intervals, semantics, coordinates = reconstructed
    final_metrics = relative_metrics(canonical_final, final)
    final_metrics["max_absolute_error"] = float(
        (canonical_final.float() - final.float()).abs().max()
    )
    coordinate_metrics = relative_metrics(canonical_coordinates, coordinates)
    coordinate_metrics["max_absolute_error"] = float(
        (canonical_coordinates.float() - coordinates.float()).abs().max()
    )
    mismatch = {
        "endpoint_mismatch_count": int((canonical_endpoints != endpoints).sum()),
        "valid_interval_mismatch_count": int((canonical_intervals != intervals).sum()),
        "valid_semantic_mismatch_count": int((canonical_semantics != semantics).sum()),
    }
    if (
        max(final_metrics["relative_l2"], coordinate_metrics["relative_l2"])
        > tolerance
        or any(value != 0 for value in mismatch.values())
    ):
        raise WriterModelError("manual UCP reconstruction differs from semantic_program")
    return {
        "final_program": final_metrics,
        "coordinates": coordinate_metrics,
        **mismatch,
        "relative_l2_tolerance": tolerance,
    }


def resample_intervals(value: torch.Tensor, length: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] <= 0 or length <= 0:
        raise WriterModelError("invalid fixed-X dynamic resampling")
    flat = value.float().reshape(value.shape[0], -1)
    if value.shape[0] != length:
        flat = F.interpolate(flat.T[None], size=length, mode="linear", align_corners=True)[0].T
    return flat.reshape(length, *value.shape[1:]).to(value.dtype)


def fixed_stream_counterfactual(
    initial: torch.Tensor,
    valid_intervals: torch.Tensor,
    row: int,
    *,
    fixed: str,
) -> torch.Tensor:
    """Put one resampled condition stream on the correct-video endpoint grid."""

    if initial.ndim != 4 or valid_intervals.shape != initial.shape[:2]:
        raise WriterModelError("invalid UCP fixed-stream counterfactual")
    if not 0 <= row < initial.shape[0] or fixed not in {"x", "a_d"}:
        raise WriterModelError("invalid UCP fixed-stream selector")
    task_tokens = (initial.shape[2] - 1) // 2
    if initial.shape[2] != task_tokens * 2 + 1:
        raise WriterModelError("invalid UCP Program column count")
    correct_count = int(valid_intervals[0].sum())
    condition_count = int(valid_intervals[row].sum())
    if correct_count <= 0 or condition_count <= 0:
        raise WriterModelError("empty UCP fixed-stream counterfactual")
    candidate = initial[0:1].clone()
    if condition_count == correct_count:
        columns = slice(task_tokens, None) if fixed == "x" else slice(0, task_tokens)
        candidate[0, :correct_count, columns] = initial[
            row, :condition_count, columns
        ]
    elif fixed == "a_d":
        candidate[0, :correct_count, :task_tokens] = resample_intervals(
            initial[row, :condition_count, :task_tokens], correct_count,
        )
    else:
        action = initial[row, :condition_count, task_tokens : task_tokens + 1]
        change = initial[row, :condition_count, task_tokens + 1 :]
        trajectory = torch.cat((torch.zeros_like(change[:1]), change.cumsum(0)))
        candidate[0, :correct_count, task_tokens : task_tokens + 1] = (
            resample_intervals(action, correct_count)
        )
        resampled = resample_intervals(trajectory, correct_count + 1)
        candidate[0, :correct_count, task_tokens + 1 :] = (
            resampled[1:] - resampled[:-1]
        )
    return candidate


def reader_attention(
    writer: CompleteLoRAWriter,
    program: torch.Tensor,
    endpoint_positions: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
    target_permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute only the reader probability matrix from its canonical Q/K path."""

    compiler = writer.compiler
    batch, intervals, columns, width = program.shape
    target_identity = compiler.target_identity
    if target_permutation is not None:
        if (
            target_permutation.shape != (compiler.target_count,)
            or target_permutation.dtype != torch.long
            or sorted(target_permutation.cpu().tolist())
            != list(range(compiler.target_count))
        ):
            raise WriterModelError("invalid target identity permutation")
        target_identity = target_identity.index_select(
            0, target_permutation.to(target_identity.device)
        )
    target = compiler.target_identity_norm(target_identity)
    rank = compiler.rank_identity_norm(compiler.rank_identity)
    query_key = (target[:, None] + rank[None]).reshape(1, -1, width).expand(batch, -1, -1)
    flat = program.reshape(batch, intervals * columns, width)
    types = compiler._type_identity(columns).to(program.dtype)
    memory_identity = types[None, None].expand(batch, intervals, columns, width).reshape_as(flat)
    first = endpoint_positions[:, :, None].expand(batch, intervals, columns).reshape(batch, -1)
    second = torch.arange(columns, device=program.device)[None, None].expand(batch, intervals, columns).reshape(batch, -1)
    query = split_heads(compiler.reader.query(query_key), compiler.reader.heads)
    key = split_heads(compiler.reader.key(compiler.program_norm(flat) + memory_identity), compiler.reader.heads)
    zeros = torch.zeros(
        batch, query_key.shape[1], dtype=torch.long, device=program.device
    )
    query = apply_two_axis_rope(query, zeros, zeros)
    key = apply_two_axis_rope(key, first, second)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.shape[-1])
    valid = (valid_intervals[:, :, None] & valid_semantics[:, None]).reshape(batch, -1)
    logits = logits.masked_fill(~valid[:, None, None], float("-inf"))
    return logits.softmax(dim=-1).detach()


def compile_with_target_identity_permutation(
    writer: CompleteLoRAWriter,
    program: torch.Tensor,
    endpoint_positions: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Change only identities assigned to real target slots, never their decode map."""

    compiler = writer.compiler
    if permutation.shape != (compiler.target_count,) or permutation.dtype != torch.long:
        raise WriterModelError("invalid target identity permutation")
    if sorted(permutation.cpu().tolist()) != list(range(compiler.target_count)):
        raise WriterModelError("target identity permutation is not bijective")
    batch, intervals, columns, width = program.shape
    target = compiler.target_identity_norm(
        compiler.target_identity.index_select(0, permutation.to(program.device))
    )
    rank = compiler.rank_identity_norm(compiler.rank_identity)
    query = (target[:, None] + rank[None]).reshape(
        1, compiler.target_count * compiler.rank, width
    ).expand(batch, -1, -1)
    flat = program.reshape(batch, intervals * columns, width)
    valid = (valid_intervals[:, :, None] & valid_semantics[:, None]).reshape(
        batch, intervals * columns
    )
    first = endpoint_positions[:, :, None].expand(
        batch, intervals, columns
    ).reshape(batch, -1)
    second = torch.arange(columns, device=program.device)[None, None].expand(
        batch, intervals, columns
    ).reshape(batch, -1)
    types = compiler._type_identity(columns).to(program.dtype)
    memory_identity = types[None, None].expand(
        batch, intervals, columns, width
    ).reshape_as(flat)
    return compiler.reader(
        query, compiler.program_norm(flat), flat, valid,
        memory_qk_identity=memory_identity,
        first_positions=first, second_positions=second,
    ).reshape(batch, compiler.target_count, compiler.rank, width)


def reader_attention_summary(
    weights: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
) -> dict[str, float]:
    batch, heads, queries, memory = weights.shape
    target_count, rank = 38, 16
    if batch != 1 or queries != target_count * rank:
        raise WriterModelError("UCP reader attention topology changed")
    columns = valid_semantics.shape[1]
    task_tokens = (columns - 1) // 2
    view = weights.reshape(heads, target_count, rank, memory)
    count = int((valid_intervals[:, :, None] & valid_semantics[:, None]).sum())
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1) / max(math.log(count), 1.0)
    by_grid = weights.reshape(1, heads, queries, valid_intervals.shape[1], columns)
    type_mass = by_grid.sum(dim=3).mean(dim=(0, 1, 2))
    target_mean = view.mean(dim=2)
    rank_mean = view.mean(dim=1)
    total_energy = float(view.square().mean())
    return {
        "normalized_entropy_mean": float(entropy.mean()),
        "top_mass_mean": float(weights.max(dim=-1).values.mean()),
        "x_mass": float(type_mass[:task_tokens].sum()),
        "a_mass": float(type_mass[task_tokens]),
        "d_mass": float(type_mass[task_tokens + 1 :].sum()),
        "target_centered_energy_ratio": float(
            (target_mean - target_mean.mean(dim=1, keepdim=True)).square().mean()
        ) / max(total_energy, 1e-24),
        "rank_centered_energy_ratio": float(
            (rank_mean - rank_mean.mean(dim=1, keepdim=True)).square().mean()
        ) / max(total_energy, 1e-24),
    }


def coordinate_summary(value: torch.Tensor) -> dict[str, float]:
    tensor = value.float()
    energy = float(tensor.square().mean())
    target = tensor.mean(dim=1)
    rank = tensor.mean(dim=0)
    return {
        "rms": math.sqrt(energy),
        "target_centered_energy_ratio": float((target - target.mean(dim=0, keepdim=True)).square().mean()) / max(energy, 1e-24),
        "rank_centered_energy_ratio": float((rank - rank.mean(dim=0, keepdim=True)).square().mean()) / max(energy, 1e-24),
    }


def _lora_pairs(writer: CompleteLoRAWriter) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for spec in writer.tensor_specs:
        result[spec.module]["a" if spec.factor_index == 0 else "b"] = spec.name
    if any(set(pair) != {"a", "b"} for pair in result.values()):
        raise WriterModelError("public LoRA A/B pairing changed")
    return dict(result)


def rank_gauge_permute(
    writer: CompleteLoRAWriter,
    state: Mapping[str, torch.Tensor],
    permutation: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, dict[str, float]]]]:
    """Apply the same rank permutation to every public A row and B column."""

    rank = writer.PUBLIC_LORA_RANK
    if (
        permutation.shape != (rank,)
        or permutation.dtype != torch.long
        or sorted(permutation.cpu().tolist()) != list(range(rank))
    ):
        raise WriterModelError("invalid public LoRA rank gauge permutation")
    result = {name: value.detach().clone() for name, value in state.items()}
    changes = {}
    for module, names in _lora_pairs(writer).items():
        a, b = state[names["a"]], state[names["b"]]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise WriterModelError("public LoRA rank axis changed")
        index = permutation.to(a.device)
        result[names["a"]] = a.index_select(0, index)
        result[names["b"]] = b.index_select(1, index.to(b.device))
        changes[module] = {
            "public_a": relative_metrics(a, result[names["a"]]),
            "public_b": relative_metrics(b, result[names["b"]]),
        }
    return result, changes


def _module_kind(module: str) -> str:
    if module.endswith("q_proj"):
        return "q"
    if module.endswith("v_proj"):
        return "v"
    if module.endswith("action_in_proj") or module.endswith("action_out_proj"):
        return "action"
    raise WriterModelError("unknown public LoRA target")


def effective_inner(
    writer: CompleteLoRAWriter,
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    kind: str | None = None,
) -> float:
    total = 0.0
    for module, names in _lora_pairs(writer).items():
        if kind is not None and _module_kind(module) != kind:
            continue
        la, lb = left[names["a"]].float(), left[names["b"]].float()
        ra, rb = right[names["a"]].float(), right[names["b"]].float()
        total += float(((lb.T @ rb) * (la @ ra.T)).sum())
    return total


def effective_metrics(
    writer: CompleteLoRAWriter,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    left = effective_inner(writer, reference, reference)
    right = effective_inner(writer, candidate, candidate)
    dot = effective_inner(writer, reference, candidate)
    return {
        "relative_l2": math.sqrt(max(left + right - 2 * dot, 0.0) / max(left, 1e-24)),
        "cosine": dot / max(math.sqrt(left * right), 1e-24),
        "reference_l2": math.sqrt(max(left, 0.0)),
        "candidate_l2": math.sqrt(max(right, 0.0)),
    }


def effective_ba_error(
    writer: CompleteLoRAWriter,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Direct numerical error between complete public BA functions."""

    difference = reference_energy = 0.0
    maximum = 0.0
    count = 0
    for names in _lora_pairs(writer).values():
        left = reference[names["b"]].float() @ reference[names["a"]].float()
        right = candidate[names["b"]].float() @ candidate[names["a"]].float()
        delta = right - left
        difference += float(delta.square().sum())
        reference_energy += float(left.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        count += delta.numel()
    return {
        "relative_l2": math.sqrt(difference / max(reference_energy, 1e-24)),
        "difference_rms": math.sqrt(difference / max(count, 1)),
        "max_absolute_error": maximum,
    }


def lora_geometry(writer: CompleteLoRAWriter, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rank = writer.PUBLIC_LORA_RANK
    gram = torch.zeros(rank, rank, dtype=torch.float64)
    b_gram = torch.zeros_like(gram)
    kind_energy = {name: 0.0 for name in ("q", "v", "action")}
    kind_module_energy: dict[str, list[float]] = {
        name: [] for name in ("q", "v", "action")
    }
    kind_gram = {name: torch.zeros_like(gram) for name in kind_energy}
    kind_b_gram = {name: torch.zeros_like(gram) for name in kind_energy}
    kind_spectra: dict[str, list[Mapping[str, float | int]]] = {
        name: [] for name in kind_energy
    }
    pairs = _lora_pairs(writer)
    factors = {
        module: (
            state[names["a"]].double().cpu(), state[names["b"]].double().cpu(),
        )
        for module, names in pairs.items()
    }
    target_spectra = effective_ba_spectra(factors)
    for module in pairs:
        a, b = factors[module]
        component = (b.T @ b) * (a @ a.T)
        spectrum = target_spectra[module]
        gram += component
        b_gram += b.T @ b
        energy = float(spectrum["effective_ba_energy"])
        kind = _module_kind(module)
        kind_energy[kind] += energy
        kind_module_energy[kind].append(energy)
        kind_gram[kind] += component
        kind_b_gram[kind] += b.T @ b
        kind_spectra[kind].append(spectrum)
    total_effective = sum(kind_energy.values())
    overall_spectrum = aggregate_effective_ba_spectra(list(target_spectra.values()))
    by_kind_spectrum = {
        kind: aggregate_effective_ba_spectra(values)
        for kind, values in kind_spectra.items()
    }
    component_geometry = component_coordinate_geometry(
        gram, b_gram, [energy for values in kind_module_energy.values() for energy in values],
    )
    by_kind_component = {
        kind: component_coordinate_geometry(
            kind_gram[kind], kind_b_gram[kind], values,
        )
        for kind, values in kind_module_energy.items()
    }
    return {
        **overall_spectrum,
        "q_v_action_energy_ratio": {key: value / max(total_effective, 1e-24) for key, value in kind_energy.items()},
        "effective_ba_spectrum_definition": (
            "reduced-QR singular energy of each real target BA=B@A, followed "
            "by the historical unweighted target mean for rank statistics"
        ),
        "per_target_effective_ba_spectrum": target_spectra,
        "per_kind_effective_ba_spectrum": by_kind_spectrum,
        "rank_coordinate_component_gram": {
            **component_geometry,
            "definition": (
                "cross-target rank-coordinate component Gram; participation and "
                "component/B-column cosine only; never an effective-BA rank spectrum"
            ),
        },
        "per_kind_rank_coordinate_component_gram": by_kind_component,
        "per_target_energy_cv_overall": component_geometry["layer_energy_cv"],
        "per_layer_energy_cv": component_geometry["layer_energy_cv"],
        "per_layer_energy_cv_by_kind": {
            kind: value["layer_energy_cv"]
            for kind, value in by_kind_component.items()
        },
        "cross_layer_effective_ba_cosine": cross_layer_cosine(writer, state),
        "public_a_rms": mapping_metrics(state, state, select="a")["reference_rms"],
        "public_b_rms": mapping_metrics(state, state, select="b")["reference_rms"],
    }


def cross_layer_cosine(writer: CompleteLoRAWriter, state: Mapping[str, torch.Tensor]) -> dict[str, float]:
    pairs = _lora_pairs(writer)
    result = {}
    for kind in ("q", "v"):
        modules = [name for name in sorted(pairs) if _module_kind(name) == kind]
        values = []
        for left_index, left in enumerate(modules):
            for right in modules[left_index + 1 :]:
                left_state = {name: state[name] for name in pairs[left].values()}
                right_state = {name: state[name] for name in pairs[right].values()}
                la, lb = left_state[pairs[left]["a"]].float(), left_state[pairs[left]["b"]].float()
                ra, rb = right_state[pairs[right]["a"]].float(), right_state[pairs[right]["b"]].float()
                dot = float(((lb.T @ rb) * (la @ ra.T)).sum())
                le = float(((lb.T @ lb) * (la @ la.T)).sum())
                re = float(((rb.T @ rb) * (ra @ ra.T)).sum())
                values.append(dot / max(math.sqrt(le * re), 1e-24))
        result[kind] = float(np.mean(values))
    return result


def decode_coordinates(
    writer: CompleteLoRAWriter,
    coordinates: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    factors: dict[str, torch.Tensor] = {}
    public: dict[str, torch.Tensor] = {}
    for spec in writer.tensor_specs:
        key, target = writer._decoding[spec.name]
        rows = writer.factor_heads[key](coordinates[:, target])
        generated = rows.transpose(-1, -2) if spec.transpose_output else rows
        template = getattr(writer, writer._template_buffers[spec.name])
        factors[spec.name] = generated
        public[spec.name] = generated.to(template.dtype) + template[None]
    return factors, public


def split_state(state: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {name: value[row].detach() for name, value in state.items()}


def variance_metrics(values: Sequence[torch.Tensor]) -> dict[str, Any]:
    tensor = torch.stack([value.float().reshape(-1) for value in values])
    sample = float(tensor.square().sum(dim=1).mean())
    mean = float(tensor.mean(dim=0).square().sum())
    if len(values) < 2:
        return {
            "videos": len(values), "estimable": False,
            "sample_energy": sample, "task_mean_energy": mean,
            "centered_variance": None,
            "centered_variance_over_sample_energy": None,
            "centered_variance_over_task_mean_energy": None,
        }
    centered = max(sample - mean, 0.0)
    return {
        "videos": len(values), "estimable": True,
        "centered_variance": centered,
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "centered_variance_over_task_mean_energy": centered / max(mean, 1e-24),
    }


def effective_variance(
    writer: CompleteLoRAWriter, states: Sequence[Mapping[str, torch.Tensor]]
) -> dict[str, Any]:
    size = len(states)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            gram[left, right] = gram[right, left] = effective_inner(
                writer, states[left], states[right]
            )
    sample, mean = float(np.diag(gram).mean()), float(gram.mean())
    if size < 2:
        return {
            "videos": size, "estimable": False,
            "sample_energy": sample, "task_mean_energy": mean,
            "centered_variance": None,
            "centered_variance_over_sample_energy": None,
            "centered_variance_over_task_mean_energy": None,
            "scale_like_video_variance_fraction": None,
            "orthogonal_direction_video_variance_fraction": None,
        }
    centered = max(sample - mean, 0.0)
    row_mean = gram.mean(axis=1)
    delta_energy = np.diag(gram) - 2 * row_mean + mean
    scale_energy = np.square(row_mean - mean) / max(mean, 1e-24)
    mean_scale = float(np.maximum(scale_energy, 0).mean())
    return {
        "videos": size, "estimable": True,
        "centered_variance": centered,
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "centered_variance_over_task_mean_energy": centered / max(mean, 1e-24),
        "scale_like_video_variance_fraction": mean_scale / max(centered, 1e-24),
        "orthogonal_direction_video_variance_fraction": max(
            float(np.maximum(delta_energy - scale_energy, 0).mean()), 0.0
        ) / max(centered, 1e-24),
    }
