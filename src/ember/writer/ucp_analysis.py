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
        "hold correct X and its endpoint grid; linearly resample each "
        "condition's A/D intervals to the correct interval count"
    ),
    "dynamic_scale": {
        "values": [0.5, 1.0, 2.0],
        "scaled_values": "A and D before both Program blocks",
        "fixed_values": "correct-video X",
    },
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
        "src/ember/writer/as_config.py",
        "src/ember/writer/as_contract.py",
        "src/ember/writer/functional.py",
        "src/ember/writer/checkpoint.py",
        "src/ember/writer/inference.py",
        "src/ember/writer/data.py",
        "src/ember/pi05_lora.py",
        "configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json",
    )
    changed = _git(repo, "diff", "--name-only", f"{training_commit}..{head}", "--", *protected)
    if changed:
        raise WriterModelError("trained UCP model/config changed after checkpoint")
    return {
        "analysis_commit": head,
        "training_commit": training_commit,
        "training_is_ancestor": True,
        "protected_paths_unchanged": list(protected),
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


def resample_intervals(value: torch.Tensor, length: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] <= 0 or length <= 0:
        raise WriterModelError("invalid fixed-X dynamic resampling")
    flat = value.float().reshape(value.shape[0], -1)
    if value.shape[0] != length:
        flat = F.interpolate(flat.T[None], size=length, mode="linear", align_corners=True)[0].T
    return flat.reshape(length, *value.shape[1:]).to(value.dtype)


def reader_attention(
    writer: CompleteLoRAWriter,
    program: torch.Tensor,
    endpoint_positions: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
) -> torch.Tensor:
    """Recompute only the reader probability matrix from its canonical Q/K path."""

    compiler = writer.compiler
    batch, intervals, columns, width = program.shape
    target = compiler.target_identity_norm(compiler.target_identity)
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


def lora_geometry(writer: CompleteLoRAWriter, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    rank = writer.PUBLIC_LORA_RANK
    gram = torch.zeros(rank, rank, dtype=torch.float64)
    b_gram = torch.zeros_like(gram)
    module_energy: dict[str, float] = {}
    kind_energy = {name: 0.0 for name in ("q", "v", "action")}
    pairs = _lora_pairs(writer)
    for module, names in pairs.items():
        a, b = state[names["a"]].double().cpu(), state[names["b"]].double().cpu()
        component = (b.T @ b) * (a @ a.T)
        gram += component
        b_gram += b.T @ b
        energy = float(component.sum())
        module_energy[module] = energy
        kind_energy[_module_kind(module)] += energy
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    spectral_total, spectral_top = float(eigen.sum()), float(eigen[0])
    probabilities = eigen / max(spectral_total, 1e-24)
    cumulative = probabilities.cumsum(0)
    diagonal = gram.diag().clamp_min(0)
    component_cosine = gram / torch.sqrt(diagonal[:, None] * diagonal[None]).clamp_min(1e-24)
    b_diagonal = b_gram.diag().clamp_min(0)
    b_cosine = b_gram / torch.sqrt(b_diagonal[:, None] * b_diagonal[None]).clamp_min(1e-24)
    upper = torch.triu(torch.ones(rank, rank, dtype=torch.bool), diagonal=1)
    layer_values = np.asarray(list(module_energy.values()), dtype=np.float64)
    total_effective = sum(kind_energy.values())
    if (
        spectral_total <= 0
        or spectral_top <= 0
        or total_effective <= 0
        or not np.isfinite(layer_values).all()
    ):
        raise WriterModelError("UCP effective LoRA geometry is degenerate")
    return {
        "effective_lora_norm": math.sqrt(max(total_effective, 0.0)),
        "stable_rank": spectral_total / max(spectral_top, 1e-24),
        "entropy_effective_rank": float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())),
        "top_singular_energy": spectral_top / max(spectral_total, 1e-24),
        "rank90": int(torch.searchsorted(cumulative, torch.tensor(.9, dtype=cumulative.dtype))) + 1,
        "rank99": int(torch.searchsorted(cumulative, torch.tensor(.99, dtype=cumulative.dtype))) + 1,
        "coordinate_energy_participation": (diagonal / diagonal.sum().clamp_min(1e-24)).tolist(),
        "active_coordinates_1e6": int((diagonal / diagonal.sum().clamp_min(1e-24) > 1e-6).sum()),
        "component_pair_cosine_mean": float(component_cosine[upper].mean()),
        "component_negative_pair_fraction": float((component_cosine[upper] < 0).float().mean()),
        "b_column_cosine_mean": float(b_cosine[upper].mean()),
        "b_column_negative_fraction": float((b_cosine[upper] < 0).float().mean()),
        "q_v_action_energy_ratio": {key: value / max(total_effective, 1e-24) for key, value in kind_energy.items()},
        "per_layer_energy_cv": float(layer_values.std() / max(layer_values.mean(), 1e-24)),
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


def variance_metrics(values: Sequence[torch.Tensor]) -> dict[str, float]:
    tensor = torch.stack([value.float().reshape(-1) for value in values])
    sample = float(tensor.square().sum(dim=1).mean())
    mean = float(tensor.mean(dim=0).square().sum())
    centered = max(sample - mean, 0.0)
    return {
        "centered_variance": centered,
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "centered_variance_over_task_mean_energy": centered / max(mean, 1e-24),
    }


def effective_variance(
    writer: CompleteLoRAWriter, states: Sequence[Mapping[str, torch.Tensor]]
) -> dict[str, float]:
    size = len(states)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            gram[left, right] = gram[right, left] = effective_inner(
                writer, states[left], states[right]
            )
    sample, mean = float(np.diag(gram).mean()), float(gram.mean())
    centered = max(sample - mean, 0.0)
    row_mean = gram.mean(axis=1)
    delta_energy = np.diag(gram) - 2 * row_mean + mean
    scale_energy = np.square(row_mean - mean) / max(mean, 1e-24)
    mean_scale = float(np.maximum(scale_energy, 0).mean())
    return {
        "videos": size,
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


def _aggregate_scalars(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        key
        for key in set.intersection(*(set(record) for record in records))
        if all(
            isinstance(record[key], (int, float)) and not isinstance(record[key], bool)
            for record in records
        )
    )
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "median": float(np.median([record[key] for record in records])),
            "min": float(np.min([record[key] for record in records])),
            "max": float(np.max([record[key] for record in records])),
        }
        for key in keys
    }


def summarize_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "conditions": {},
        "reader_attention": {},
        "lora_geometry": {},
        "per_task": {},
    }
    metric_stages = (*STAGES, "factor_output", "public_a", "public_b", "effective_ba", "policy_action")
    for condition in CONDITIONS:
        result["conditions"][condition] = {}
        for stage in metric_stages:
            records = [row["comparisons_to_correct"][condition][stage] for row in rows]
            result["conditions"][condition][stage] = {
                metric: {
                    "mean": float(np.mean([record[metric] for record in records])),
                    "median": float(np.median([record[metric] for record in records])),
                    "min": float(np.min([record[metric] for record in records])),
                    "max": float(np.max([record[metric] for record in records])),
                }
                for metric in ("relative_l2", "cosine")
            }
        result["reader_attention"][condition] = _aggregate_scalars(
            [row["reader_attention"][condition] for row in rows]
        )
        result["lora_geometry"][condition] = _aggregate_scalars(
            [row["lora_geometry"][condition] for row in rows]
        )
    for task_id in sorted({int(row["global_task_id"]) for row in rows}):
        selected = [row for row in rows if int(row["global_task_id"]) == task_id]
        key = f"{selected[0]['suite']}:task_{int(selected[0]['task_id']):02d}"
        result["per_task"][key] = {
            "global_task_id": task_id,
            "references": len(selected),
            "conditions": {
                condition: {
                    stage: {
                        "mean_relative_l2": float(np.mean([
                            row["comparisons_to_correct"][condition][stage]["relative_l2"]
                            for row in selected
                        ])),
                        "mean_cosine": float(np.mean([
                            row["comparisons_to_correct"][condition][stage]["cosine"]
                            for row in selected
                        ])),
                    }
                    for stage in metric_stages
                }
                for condition in CONDITIONS
            },
            "same_task_video_variance": selected[0].get("same_task_video_variance"),
            "reader_attention": {
                condition: _aggregate_scalars(
                    [row["reader_attention"][condition] for row in selected]
                )
                for condition in CONDITIONS
            },
            "lora_geometry": {
                condition: _aggregate_scalars(
                    [row["lora_geometry"][condition] for row in selected]
                )
                for condition in CONDITIONS
            },
        }
    result["factor_gauge_caveat"] = (
        "raw factor/public A/B coordinates are gauge-dependent; effective BA and "
        "fixed-query action are primary functional evidence"
    )
    return result
