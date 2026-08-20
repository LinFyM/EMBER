"""Frozen task-consensus and memberwise codes for the phase-aligned decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json


PHASE_DECODER_CODE_SCHEMA = "ember_successful_expert_equivalence_phase_codes_v1"


@dataclass(frozen=True)
class PhaseDecoderMember:
    code_index: int
    suite: str
    task_id: int
    global_task_id: int
    ordinal: int
    fold_role: str
    member: str
    expert_step: int
    init_state_id: int
    selected_replan_indices: tuple[int, ...]


@dataclass(frozen=True)
class PhaseDecoderCodeAuthority:
    root: Path
    fit_ordinals: tuple[int, ...]
    held_ordinals: tuple[int, ...]
    fit_task_codes: torch.Tensor
    member_codes: torch.Tensor
    members: tuple[PhaseDecoderMember, ...]

    def member_groups(self, ordinals: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                index
                for index, member in enumerate(self.members)
                if member.ordinal == int(ordinal)
            )
            for ordinal in ordinals
        )


def _member(value: Mapping[str, Any], *, expected_index: int) -> PhaseDecoderMember:
    indices = tuple(int(item) for item in value.get("selected_replan_indices", ()))
    result = PhaseDecoderMember(
        code_index=int(value.get("code_index", -1)),
        suite=str(value.get("suite", "")),
        task_id=int(value.get("task_id", -1)),
        global_task_id=int(value.get("global_task_id", -1)),
        ordinal=int(value.get("ordinal", -1)),
        fold_role=str(value.get("fold_role", "")),
        member=str(value.get("member", "")),
        expert_step=int(value.get("expert_step", -1)),
        init_state_id=int(value.get("init_state_id", -1)),
        selected_replan_indices=indices,
    )
    if (
        result.code_index != expected_index
        or not result.suite
        or min(
            result.task_id,
            result.global_task_id,
            result.ordinal,
            result.expert_step,
            result.init_state_id,
        )
        < 0
        or result.fold_role not in {"fit", "held_transform_only"}
        or result.member not in {"earliest", "latest", "only"}
        or len(indices) != 8
        or any(left >= right for left, right in zip(indices, indices[1:]))
    ):
        raise ValueError("phase decoder member authority changed")
    return result


def _validate_artifact_header(
    result: Mapping[str, Any],
    *,
    config_path: Path,
    codes_path: Path,
) -> None:
    recorded_config = result.get("config", {})
    construction = result.get("construction", {})
    if (
        result.get("schema_version") != PHASE_DECODER_CODE_SCHEMA
        or result.get("formal_authority") is not True
        or result.get("repository", {}).get("dirty_paths") != []
        or Path(str(recorded_config.get("path", ""))).name != config_path.name
        or int(recorded_config.get("bytes", -1)) != config_path.stat().st_size
        or construction.get("fit_surface")
        != "fit19_task_consensus_only_pca_whitening"
        or construction.get("held_surface") != "memberwise_frozen_transform_only"
        or int(construction.get("held_code_optimization_steps", -1)) != 0
        or not codes_path.is_file()
        or codes_path.stat().st_size
        != int(result.get("files", {}).get("phase_codes.safetensors", -1))
    ):
        raise ValueError("phase decoder code artifact changed")


def _load_members_and_roles(
    result: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[PhaseDecoderMember, ...]]:
    fit_ordinals = tuple(int(value) for value in config["roles"]["fit_task_ordinals"])
    held_ordinals = tuple(
        int(value) for value in config["roles"]["held_transform_only_task_ordinals"]
    )
    fit_rows = tuple(result.get("fit_tasks", ()))
    if (
        tuple(int(row.get("ordinal", -1)) for row in fit_rows) != fit_ordinals
        or tuple(int(row.get("code_index", -1)) for row in fit_rows)
        != tuple(range(len(fit_ordinals)))
    ):
        raise ValueError("phase decoder fit task code order changed")
    members = tuple(
        _member(row, expected_index=index)
        for index, row in enumerate(result.get("members", ()))
    )
    fit_roles = {member.ordinal for member in members if member.fold_role == "fit"}
    held_roles = {
        member.ordinal for member in members if member.fold_role == "held_transform_only"
    }
    if len(members) != 47 or fit_roles != set(fit_ordinals) or held_roles != set(held_ordinals):
        raise ValueError("phase decoder member roles changed")
    return fit_ordinals, held_ordinals, members


def load_phase_decoder_code_authority(
    root: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    device: torch.device | str,
) -> PhaseDecoderCodeAuthority:
    """Load one fit19-only code transform with memberwise held projections."""

    resolved = root.resolve()
    result_path = resolved / "result.json"
    codes_path = resolved / "phase_codes.safetensors"
    result = read_json(result_path)
    expected_config = config_path.resolve()
    _validate_artifact_header(result, config_path=expected_config, codes_path=codes_path)
    fit_ordinals, held_ordinals, members = _load_members_and_roles(result, config)
    tensors = load_file(str(codes_path), device=str(device))
    code_width = int(config["representation"]["code_width"])
    fit_task_codes = tensors.get("fit_task_codes")
    member_codes = tensors.get("member_codes")
    if (
        fit_task_codes is None
        or member_codes is None
        or tuple(fit_task_codes.shape) != (len(fit_ordinals), code_width)
        or tuple(member_codes.shape) != (len(members), code_width)
        or not torch.isfinite(fit_task_codes).all()
        or not torch.isfinite(member_codes).all()
    ):
        raise ValueError("phase decoder code tensors changed")
    return PhaseDecoderCodeAuthority(
        root=resolved,
        fit_ordinals=fit_ordinals,
        held_ordinals=held_ordinals,
        fit_task_codes=fit_task_codes,
        member_codes=member_codes,
        members=members,
    )
