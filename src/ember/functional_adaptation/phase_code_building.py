"""Build fixed task-consensus codes from the successful phase response bank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ember.functional_adaptation.fingerprint_codes import (
    FunctionalFingerprintCodeSpace,
    whiten_functional_fingerprints,
)
from ember.functional_adaptation.phase_alignment import (
    FunctionalWhitener,
    arc_length_phase_embedding,
    arc_length_phase_indices,
    uniform_time_embedding,
)
from ember.functional_adaptation.phase_decoder_codes import (
    PHASE_DECODER_CODE_SCHEMA,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


CONFIG_SCHEMA = "ember_pi05_train24_phase_aligned_decoder_v1"
SELECTION_SCHEMA = "ember_successful_expert_equivalence_selection_v1"
SHARD_SCHEMA = "ember_successful_expert_equivalence_response_shard_v1"
ANALYSIS_SCHEMA = "ember_successful_expert_equivalence_phase_analysis_v1"
TRANSFORM_SCHEMA = "ember_successful_expert_equivalence_phase_transform_v1"


@dataclass(frozen=True)
class CodeBuildInputs:
    config_path: Path
    config: dict[str, Any]
    repository: dict[str, Any]
    selection_path: Path
    analysis_path: Path
    analysis: dict[str, Any]
    members: tuple[dict[str, Any], ...]
    whitener: FunctionalWhitener


def _authority_path(repo_root: Path, config: Mapping[str, Any], name: str) -> Path:
    value = Path(str(config["authorities"][name]))
    return value if value.is_absolute() else repo_root / value


def _repo_reference(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        parts = resolved.parts
        return str(Path(*parts[parts.index("runs") :])) if "runs" in parts else resolved.name


def _member_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(row["ordinal"]), int(row["expert_step"]), int(row["init_state_id"])


def _load_members(
    selection: Mapping[str, Any], shard_paths: Sequence[Path]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
    count = len(shards)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or count != 6
        or any(row.get("schema_version") != SHARD_SCHEMA for row in shards)
        or {int(row.get("shard_count", -1)) for row in shards} != {count}
        or {int(row.get("shard_index", -1)) for row in shards} != set(range(count))
        or any(row.get("analysis_git") != shards[0].get("analysis_git") for row in shards)
        or any(row.get("panels") != shards[0].get("panels") for row in shards)
    ):
        raise ValueError("phase code response shards changed")
    members = sorted(
        (dict(member) for shard in shards for member in shard.get("members", ())),
        key=_member_key,
    )
    expected = sorted((dict(row) for row in selection.get("rows", ())), key=_member_key)
    if (
        len(members) != 47
        or [_member_key(row) for row in members] != [_member_key(row) for row in expected]
        or any(
            row["action_delta"].ndim != 3
            or tuple(row["action_delta"].shape[1:]) != (50, 7)
            or not torch.isfinite(row["action_delta"]).all()
            for row in members
        )
    ):
        raise ValueError("phase code member bank changed")
    return tuple(members), dict(shards[0]["analysis_git"])


def _load_inputs(config_path: Path, repo_root: Path) -> CodeBuildInputs:
    config_path = config_path.resolve()
    config = read_json(config_path)
    repository = git_state(repo_root)
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "preregistered_before_decoder_optimization"
        or not git_state_is_clean_pushed_or_frozen_authority(repository)
    ):
        raise ValueError("phase code build requires the clean preregistered decoder contract")
    selection_path = _authority_path(repo_root, config, "selection")
    selection = read_json(selection_path)
    shard_paths = tuple(
        repo_root / Path(str(value)) for value in config["authorities"]["response_shards"]
    )
    members, analysis_git = _load_members(selection, shard_paths)
    analysis_path = _authority_path(repo_root, config, "phase_analysis")
    analysis = read_json(analysis_path)
    if (
        analysis.get("schema_version") != ANALYSIS_SCHEMA
        or analysis.get("decision") != "advance_to_phase_aligned_fixed_decoder"
        or analysis.get("held_gate", {}).get("passes") is not True
        or analysis.get("analysis_git") != analysis_git
    ):
        raise ValueError("phase code build lacks a passing representation authority")
    transform = torch.load(
        _authority_path(repo_root, config, "phase_transform"),
        map_location="cpu",
        weights_only=False,
    )
    if (
        transform.get("schema_version") != TRANSFORM_SCHEMA
        or transform.get("analysis_git") != analysis.get("analysis_git")
        or int(transform.get("width", -1)) != 32
    ):
        raise ValueError("phase code functional transform changed")
    whitener = FunctionalWhitener(
        mean=transform["mean"],
        components=transform["components"],
        scales=transform["scales"],
        explained_variance_ratio=float(transform["explained_variance_ratio"]),
    )
    return CodeBuildInputs(
        config_path=config_path,
        config=config,
        repository=repository,
        selection_path=selection_path,
        analysis_path=analysis_path,
        analysis=analysis,
        members=members,
        whitener=whitener,
    )


def _embeddings(
    inputs: CodeBuildInputs,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    phase, uniform, indices = [], [], []
    for row in inputs.members:
        sequence = inputs.whitener.transform(row["action_delta"])
        phase.append(arc_length_phase_embedding(sequence, count=8))
        uniform.append(uniform_time_embedding(sequence, count=8))
        indices.append(tuple(int(value) for value in arc_length_phase_indices(sequence, count=8)))
    return torch.stack(phase).flatten(1).float(), torch.stack(uniform).flatten(1).float(), tuple(indices)


def _task_consensus(
    embeddings: torch.Tensor,
    members: Sequence[Mapping[str, Any]],
    fit_ordinals: Sequence[int],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    rows, global_ids = [], []
    for ordinal in fit_ordinals:
        selected = [index for index, row in enumerate(members) if int(row["ordinal"]) == ordinal]
        ids = {int(members[index]["global_task_id"]) for index in selected}
        if not selected or len(ids) != 1:
            raise ValueError("phase code fit task members changed")
        rows.append(embeddings[selected].mean(dim=0))
        global_ids.append(ids.pop())
    return torch.stack(rows), tuple(global_ids)


def _cosine_summary(
    name: str, values: torch.Tensor, members: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    normalized = torch.nn.functional.normalize(values.double(), dim=1)
    cosine = normalized @ normalized.T
    nearest_panel = cosine.clone()
    nearest_panel.fill_diagonal_(float("-inf"))
    nearest = nearest_panel.argmax(dim=1)
    per_task = []
    for task_id in sorted({int(row["global_task_id"]) for row in members}):
        pair = [index for index, row in enumerate(members) if int(row["global_task_id"]) == task_id]
        if len(pair) != 2:
            continue
        left, right = pair
        per_task.append(
            {
                "global_task_id": task_id,
                "same_task_cosine": float(cosine[left, right]),
                "mutual_cosine_nearest": int(nearest[left]) == right and int(nearest[right]) == left,
            }
        )
    return {
        "family": name,
        "mutual_nearest_tasks": sum(bool(row["mutual_cosine_nearest"]) for row in per_task),
        "per_task": per_task,
        "nearest_indices": [int(value) for value in nearest],
        "cosine": [[float(value) for value in row] for row in cosine],
    }


def _save_codes(
    output_dir: Path,
    phase: torch.Tensor,
    consensus: torch.Tensor,
    space: FunctionalFingerprintCodeSpace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "phase_codes.safetensors"
    save_file(
        {
            "fit_task_codes": space.train_codes.cpu().contiguous(),
            "member_codes": space.held_codes.cpu().contiguous(),
            "fit_task_phase_fingerprints": consensus.cpu().contiguous(),
            "member_phase_fingerprints": phase.cpu().contiguous(),
            "mean": space.mean.cpu().contiguous(),
            "components": space.components.cpu().contiguous(),
            "scales": space.scales.cpu().contiguous(),
        },
        str(path),
    )
    return path


def _result(
    inputs: CodeBuildInputs,
    *,
    codes_path: Path,
    phase: torch.Tensor,
    phase_indices: Sequence[Sequence[int]],
    fit_ordinals: Sequence[int],
    held_ordinals: Sequence[int],
    fit_global_ids: Sequence[int],
    phase_space: FunctionalFingerprintCodeSpace,
    diagnostics: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": PHASE_DECODER_CODE_SCHEMA,
        "formal_authority": True,
        "repository": inputs.repository,
        "config": {"path": _repo_reference(repo_root, inputs.config_path), "bytes": inputs.config_path.stat().st_size},
        "selection": _repo_reference(repo_root, inputs.selection_path),
        "phase_analysis": {
            "path": _repo_reference(repo_root, inputs.analysis_path),
            "bytes": inputs.analysis_path.stat().st_size,
            "decision": inputs.analysis["decision"],
        },
        "construction": {
            "phase_fingerprint_width": int(phase.shape[1]),
            "code_width": int(phase_space.train_codes.shape[1]),
            "fit_surface": "fit19_task_consensus_only_pca_whitening",
            "task_consensus": "mean_successful_member_phase_embedding_not_lora",
            "held_surface": "memberwise_frozen_transform_only",
            "explained_variance_fraction": phase_space.explained_variance_fraction,
            "final_lora_averaging": False,
            "held_code_optimization_steps": 0,
        },
        "fit_tasks": [
            {
                "code_index": index,
                "ordinal": ordinal,
                "global_task_id": fit_global_ids[index],
                "member_code_indices": [member_index for member_index, row in enumerate(inputs.members) if int(row["ordinal"]) == ordinal],
            }
            for index, ordinal in enumerate(fit_ordinals)
        ],
        "members": [
            {
                "code_index": index,
                **{key: row[key] for key in ("suite", "task_id", "global_task_id", "ordinal", "fold_role", "member", "expert_step", "init_state_id")},
                "selected_replan_indices": list(phase_indices[index]),
            }
            for index, row in enumerate(inputs.members)
        ],
        "diagnostics": dict(diagnostics),
        "files": {"phase_codes.safetensors": codes_path.stat().st_size},
        "information_wall": {
            "fit_tasks": len(fit_ordinals),
            "held_transform_only_tasks": len(held_ordinals),
            "validation_use": False,
            "test_use": False,
            "training_gradient_use": False,
            "deployment_task_id_route": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def run(config_path: Path, output_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    inputs = _load_inputs(config_path, repo_root)
    phase, uniform, phase_indices = _embeddings(inputs)
    fit_ordinals = tuple(int(value) for value in inputs.config["roles"]["fit_task_ordinals"])
    held_ordinals = tuple(int(value) for value in inputs.config["roles"]["held_transform_only_task_ordinals"])
    if sorted(fit_ordinals + held_ordinals) != list(range(24)):
        raise ValueError("phase code task roles changed")
    phase_consensus, fit_global_ids = _task_consensus(phase, inputs.members, fit_ordinals)
    uniform_consensus, _ = _task_consensus(uniform, inputs.members, fit_ordinals)
    width = int(inputs.config["representation"]["code_width"])
    phase_space = whiten_functional_fingerprints(phase_consensus, phase, code_width=width)
    uniform_space = whiten_functional_fingerprints(uniform_consensus, uniform, code_width=width)
    held_indices = [index for index, row in enumerate(inputs.members) if int(row["ordinal"]) in held_ordinals]
    held_members = [inputs.members[index] for index in held_indices]
    diagnostics = {
        "selection_note": "PCA16 task consensus was chosen after the phase representation gate as decoder architecture validation, not as held performance selection.",
        "phase": _cosine_summary("held5_phase_task_consensus_pca16", phase_space.held_codes[held_indices], held_members),
        "uniform_time_control": _cosine_summary("held5_uniform_task_consensus_pca16_control", uniform_space.held_codes[held_indices], held_members),
    }
    output_dir = output_dir.resolve()
    codes_path = _save_codes(output_dir, phase, phase_consensus, phase_space)
    result = _result(
        inputs,
        codes_path=codes_path,
        phase=phase,
        phase_indices=phase_indices,
        fit_ordinals=fit_ordinals,
        held_ordinals=held_ordinals,
        fit_global_ids=fit_global_ids,
        phase_space=phase_space,
        diagnostics=diagnostics,
        repo_root=repo_root,
    )
    write_json_atomic(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "phase_held_mutual_nearest_tasks": diagnostics["phase"]["mutual_nearest_tasks"],
                "uniform_held_mutual_nearest_tasks": diagnostics["uniform_time_control"]["mutual_nearest_tasks"],
                "explained_variance_fraction": phase_space.explained_variance_fraction,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return result
