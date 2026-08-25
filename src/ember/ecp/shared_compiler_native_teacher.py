"""Offline K1 native-feasible teachers and gauge-invariant G3 losses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


G3_NATIVE_TEACHER_ROOT_SCHEMA = "ember_ecp_g3_k1_native_teacher_root_v1"
G3_NATIVE_TEACHER_TASK_SCHEMA = "ember_ecp_g3_k1_native_teacher_task_v1"
G3_NATIVE_TEACHER_WORKER_SCHEMA = "ember_ecp_g3_k1_native_teacher_worker_v1"
G3_NATIVE_TEACHER_FORMAL_MACROS = 40


class NativeTeacherAuthorityError(ValueError):
    """Raised when a native teacher lookup crosses the fit-only authority."""


@dataclass(frozen=True)
class NativeTeacherFactors:
    """One exact task/video/member native-projected rank-four residual.

    Both ``a`` and ``b`` are pre-scale signed-pooling directions with the rank
    axis first.  ``scales`` is the sole magnitude owner.
    """

    authority_id: int
    video_demo: int
    member_name: str
    a: tuple[torch.Tensor, ...]
    b: tuple[torch.Tensor, ...]
    scales: torch.Tensor
    provenance: Mapping[str, Any]

    @property
    def input_directions(self) -> tuple[torch.Tensor, ...]:
        return self.a

    @property
    def output_directions(self) -> tuple[torch.Tensor, ...]:
        return self.b

    def lora_state(self, contract: LoRAContract) -> dict[str, torch.Tensor]:
        if len(self.a) != len(contract.targets) or len(self.b) != len(
            contract.targets
        ):
            raise NativeTeacherAuthorityError("native teacher target count changed")
        state: dict[str, torch.Tensor] = {}
        for index, (target, a, b) in enumerate(
            zip(contract.targets, self.a, self.b, strict=True)
        ):
            state[target.name + LORA_A_SUFFIX] = a
            state[target.name + LORA_B_SUFFIX] = (
                b * self.scales[index, :, None]
            ).transpose(0, 1)
        validate_lora_state(state, contract)
        return state


@dataclass(frozen=True)
class NativeTeacherLoss:
    """Losses whose two branches have disjoint student gradient owners."""

    total: torch.Tensor
    selection: torch.Tensor
    input_subspace: torch.Tensor
    output_subspace: torch.Tensor
    update_direction: torch.Tensor
    spectrum_scale: torch.Tensor

    def metrics(self) -> dict[str, float]:
        return {
            "native_teacher_total": float(self.total.detach()),
            "native_teacher_selection": float(self.selection.detach()),
            "native_teacher_input_subspace": float(
                self.input_subspace.detach()
            ),
            "native_teacher_output_subspace": float(
                self.output_subspace.detach()
            ),
            "native_teacher_update_direction": float(
                self.update_direction.detach()
            ),
            "native_teacher_spectrum_scale": float(self.spectrum_scale.detach()),
            "native_teacher_selection_owns_directions": 1.0,
            "native_teacher_selection_owns_scales": 0.0,
            "native_teacher_spectrum_owns_directions": 0.0,
            "native_teacher_spectrum_owns_scales": 1.0,
        }


def native_teacher_from_lora_state(
    *,
    authority_id: int,
    video_demo: int,
    member_name: str,
    state: Mapping[str, torch.Tensor],
    scales: torch.Tensor,
    contract: LoRAContract,
    provenance: Mapping[str, Any],
) -> NativeTeacherFactors:
    """Detach the G1 projection result into its offline teacher payload."""

    validate_lora_state(state, contract)
    if contract.rank != 4 or scales.shape != (len(contract.targets), 4):
        raise NativeTeacherAuthorityError("native teacher is not rank four")
    factors = NativeTeacherFactors(
        authority_id=int(authority_id),
        video_demo=int(video_demo),
        member_name=str(member_name),
        a=tuple(
            state[target.name + LORA_A_SUFFIX]
            .detach()
            .float()
            .cpu()
            .contiguous()
            for target in contract.targets
        ),
        b=tuple(
            (
                state[target.name + LORA_B_SUFFIX].detach().float().transpose(0, 1)
                / scales[index].detach().float()[:, None]
            )
            .cpu()
            .contiguous()
            for index, target in enumerate(contract.targets)
        ),
        scales=scales.detach().float().cpu().contiguous(),
        provenance=dict(provenance),
    )
    _validate_teacher(factors, contract)
    return factors


def factor_subspace_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """Chordal row-subspace loss, invariant to any nonsingular rank basis."""

    if student.ndim != 2 or teacher.shape != student.shape:
        raise ValueError("native teacher subspace factors changed shape")
    rank = student.shape[0]
    if rank == 0 or min(student.shape[1], teacher.shape[1]) < rank:
        raise ValueError("native teacher subspace is not full-width")
    student_basis = torch.linalg.qr(student.float().transpose(0, 1), mode="reduced").Q
    teacher_basis = torch.linalg.qr(teacher.float().transpose(0, 1), mode="reduced").Q
    overlap = (student_basis.transpose(0, 1) @ teacher_basis).square().sum()
    return (1.0 - overlap / rank).clamp_min(0.0)


def low_rank_update_direction_loss(
    student_a: torch.Tensor,
    student_b: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Cosine loss between B@A updates without materializing either update."""

    _validate_factor_pair(student_a, student_b)
    _validate_factor_pair(teacher_a, teacher_b)
    if student_a.shape[1] != teacher_a.shape[1] or student_b.shape[0] != teacher_b.shape[0]:
        raise ValueError("native teacher update widths changed")

    def inner(a1: torch.Tensor, b1: torch.Tensor, a2: torch.Tensor, b2: torch.Tensor):
        b_inner = b1.float().transpose(0, 1) @ b2.float()
        a_inner = a1.float() @ a2.float().transpose(0, 1)
        return (b_inner * a_inner).sum()

    dot = inner(student_a, student_b, teacher_a, teacher_b)
    student_norm = inner(student_a, student_b, student_a, student_b).clamp_min(0).sqrt()
    teacher_norm = inner(teacher_a, teacher_b, teacher_a, teacher_b).clamp_min(0).sqrt()
    cosine = dot / (student_norm * teacher_norm).clamp_min(epsilon)
    return 1.0 - cosine.clamp(-1.0, 1.0)


def small_core_singular_values(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Singular values of B@A using only its rank-sized core."""

    _validate_factor_pair(a, b)
    qb, rb = torch.linalg.qr(b.float(), mode="reduced")
    qa, ra = torch.linalg.qr(a.float().transpose(0, 1), mode="reduced")
    del qb, qa
    return torch.linalg.svdvals(rb @ ra.transpose(0, 1))


def small_core_spectrum_loss(
    student_a: torch.Tensor,
    student_b: torch.Tensor,
    teacher_a: torch.Tensor,
    teacher_b: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Smooth log-spectrum scale loss, invariant to low-rank factor gauge."""

    student = small_core_singular_values(student_a, student_b)
    teacher = small_core_singular_values(teacher_a, teacher_b)
    if student.shape != teacher.shape:
        raise ValueError("native teacher spectrum ranks changed")
    return functional.smooth_l1_loss(
        torch.log(student.clamp_min(epsilon)),
        torch.log(teacher.clamp_min(epsilon)),
    )


def family_equal_target_weights(
    owners: Sequence[TargetOwner], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Give every present owner family equal total mass."""

    families = tuple(TargetFamily)
    if {owner.family for owner in owners} != set(families):
        raise ValueError("native teacher supervision lost an owner family")
    weights = torch.zeros(len(owners), device=device, dtype=dtype)
    for family in families:
        indices = [index for index, owner in enumerate(owners) if owner.family == family]
        weights[indices] = 1.0 / (len(families) * len(indices))
    return weights


def native_teacher_supervision_loss(
    *,
    student_a_directions: Sequence[torch.Tensor],
    student_b_directions: Sequence[torch.Tensor],
    student_scales: torch.Tensor,
    teachers: Sequence[NativeTeacherFactors],
    owners: Sequence[TargetOwner],
    member_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    selection_weight: float = 1.0,
    spectrum_weight: float = 1.0,
) -> NativeTeacherLoss:
    """Supervise K1 selection and scale with explicit, disjoint gradient walls.

    ``student_b_directions`` must be the pre-scale signed-pooling output.  Passing
    the already scaled residual would reintroduce a scale path into selection.
    """

    target_count = len(owners)
    member_count = len(teachers)
    teacher_conditions = {
        (teacher.authority_id, teacher.video_demo) for teacher in teachers
    }
    if (
        member_count == 0
        or len(student_a_directions) != target_count
        or len(student_b_directions) != target_count
        or student_scales.shape != (target_count, 4)
        or selection_weight < 0
        or spectrum_weight < 0
        or len(teacher_conditions) != 1
        or len({teacher.member_name for teacher in teachers}) != member_count
    ):
        raise ValueError("native teacher supervision batch changed shape")
    device = student_scales.device
    dtype = student_scales.dtype
    member_weights = _normalized_fixed_weights(
        member_weights,
        count=member_count,
        device=device,
        dtype=dtype,
        name="member",
    )
    if target_weights is None:
        common = family_equal_target_weights(owners, device=device, dtype=dtype)
        target_weights = common[None].expand(member_count, -1)
    else:
        target_weights = target_weights.detach().to(device=device, dtype=dtype)
        if target_weights.ndim == 1:
            target_weights = target_weights[None].expand(member_count, -1)
        _validate_family_weights(target_weights, owners)

    per_member = []
    for teacher in teachers:
        per_member.append(
            _teacher_loss_rows(
                student_a_directions=student_a_directions,
                student_b_directions=student_b_directions,
                student_scales=student_scales,
                teacher=teacher,
                owners=owners,
            )
        )
    components = []
    for component in range(4):
        values = torch.stack(
            [member[component] for member in per_member]
        )
        components.append(
            ((values * target_weights).sum(-1) * member_weights).sum()
        )
    input_subspace, output_subspace, update_direction, spectrum_scale = components
    selection = (input_subspace + output_subspace + update_direction) / 3.0
    total = selection_weight * selection + spectrum_weight * spectrum_scale
    return NativeTeacherLoss(
        total=total,
        selection=selection,
        input_subspace=input_subspace,
        output_subspace=output_subspace,
        update_direction=update_direction,
        spectrum_scale=spectrum_scale,
    )


def _teacher_loss_rows(
    *,
    student_a_directions: Sequence[torch.Tensor],
    student_b_directions: Sequence[torch.Tensor],
    student_scales: torch.Tensor,
    teacher: NativeTeacherFactors,
    owners: Sequence[TargetOwner],
) -> tuple[torch.Tensor, ...]:
    if len(teacher.a) != len(owners) or len(teacher.b) != len(owners):
        raise ValueError("native teacher supervision target count changed")
    rows: list[list[torch.Tensor]] = [[], [], [], []]
    for target, (student_a, student_b_direction, teacher_a, teacher_b) in enumerate(
        zip(
            student_a_directions,
            student_b_directions,
            teacher.a,
            teacher.b,
            strict=True,
        )
    ):
        if student_a.shape != (4, owners[target].in_features) or (
            student_b_direction.shape != (4, owners[target].out_features)
        ):
            raise ValueError("native teacher student factor shape changed")
        teacher_a = teacher_a.detach().to(student_a)
        teacher_b_direction = teacher_b.detach().to(student_b_direction)
        teacher_b = teacher_b_direction.transpose(0, 1) * teacher.scales[
            target
        ].detach().to(student_b_direction)[None]
        selection_b = student_b_direction.transpose(0, 1) * student_scales[
            target
        ].detach()[None]
        scale_b = student_b_direction.detach().transpose(0, 1) * student_scales[
            target
        ][None]
        rows[0].append(factor_subspace_loss(student_a, teacher_a))
        rows[1].append(
            factor_subspace_loss(student_b_direction, teacher_b_direction)
        )
        rows[2].append(
            low_rank_update_direction_loss(
                student_a, selection_b, teacher_a, teacher_b
            )
        )
        rows[3].append(
            small_core_spectrum_loss(
                student_a.detach(), scale_b, teacher_a, teacher_b
            )
        )
    return tuple(torch.stack(row) for row in rows)


class NativeTeacherStore:
    """Lazy, exact fit-task K1 lookup with a zero-read K2/K4 interface."""

    def __init__(
        self,
        root_manifest: Path,
        *,
        contract: LoRAContract,
        expected_fit_task_ids: set[int],
        expected_full_fit_task_ids: set[int],
        device: torch.device | str,
    ) -> None:
        root_manifest = root_manifest.resolve()
        root = read_json(root_manifest)
        records = tuple(root.get("records", ()))
        by_id = {int(row.get("authority_id", -1)): row for row in records}
        if (
            root.get("schema_version") != G3_NATIVE_TEACHER_ROOT_SCHEMA
            or root.get("status") != "complete"
            or int(root.get("formal_macros", -1)) != G3_NATIVE_TEACHER_FORMAL_MACROS
            or int(root.get("task_count", -1)) != len(records)
            or set(by_id) != set(expected_fit_task_ids)
            or set(root.get("fit_authority_task_ids", ()))
            != set(expected_full_fit_task_ids)
            or int(root.get("fit_authority_task_count", -1))
            != len(expected_full_fit_task_ids)
            or int(root.get("K1_covered_task_count", -1))
            != len(expected_fit_task_ids)
            or set(root.get("K1_missing_task_ids", ()))
            != set(expected_full_fit_task_ids) - set(expected_fit_task_ids)
            or root.get("coverage", {}).get("task_ids")
            != sorted(expected_fit_task_ids)
            or root.get("coverage", {}).get("roles") != root.get("roles")
            or root.get("contract") != _contract_record(contract)
            or any(row.get("role") not in {"meta_fit", "target_fit"} for row in records)
        ):
            raise NativeTeacherAuthorityError("native teacher root authority changed")
        self.root_manifest = root_manifest
        self.contract = contract
        self.device = device
        self.records = by_id
        self.cache: dict[int, dict[tuple[int, str], NativeTeacherFactors]] = {}
        self.tensor_reads = 0

    def lookup(
        self,
        *,
        authority_id: int,
        k: int,
        video_demo: int | None = None,
        member_name: str | None = None,
    ) -> NativeTeacherFactors | None:
        if k in (2, 4):
            return None
        if k != 1:
            raise NativeTeacherAuthorityError("native teacher K must be 1, 2, or 4")
        if video_demo is None or member_name is None:
            raise NativeTeacherAuthorityError("K1 native teacher key is incomplete")
        if authority_id not in self.records:
            raise NativeTeacherAuthorityError(
                "K1 native teacher requested a held or schedule-uncovered task"
            )
        if authority_id not in self.cache:
            self.cache[authority_id] = self._load_task(authority_id)
        try:
            return self.cache[authority_id][(int(video_demo), str(member_name))]
        except KeyError as error:
            raise NativeTeacherAuthorityError(
                "K1 native teacher video/member cache miss"
            ) from error

    def lookup_members(
        self,
        *,
        authority_id: int,
        k: int,
        video_demo: int | None,
        member_names: Sequence[str],
    ) -> tuple[NativeTeacherFactors, ...] | None:
        if k in (2, 4):
            return None
        if k != 1:
            raise NativeTeacherAuthorityError("native teacher K must be 1, 2, or 4")
        return tuple(
            self.lookup(
                authority_id=authority_id,
                k=k,
                video_demo=video_demo,
                member_name=member,
            )
            for member in member_names
        )

    def _load_task(self, authority_id: int) -> dict[tuple[int, str], NativeTeacherFactors]:
        record = self.records[authority_id]
        manifest_path = Path(str(record.get("manifest", ""))).resolve()
        if not manifest_path.is_file() or manifest_path.stat().st_size != int(
            record.get("manifest_bytes", -1)
        ):
            raise NativeTeacherAuthorityError("native teacher task manifest changed")
        manifest = read_json(manifest_path)
        tensor_record = manifest.get("tensor_file", {})
        tensor_path = Path(str(tensor_record.get("path", ""))).resolve()
        rows = tuple(manifest.get("teachers", ()))
        if (
            manifest.get("schema_version") != G3_NATIVE_TEACHER_TASK_SCHEMA
            or manifest.get("status") != "complete"
            or int(manifest.get("task", {}).get("authority_id", -1)) != authority_id
            or manifest.get("task", {}).get("role") not in {"meta_fit", "target_fit"}
            or int(manifest.get("teacher_count", -1)) != len(rows)
            or not tensor_path.is_file()
            or tensor_path.stat().st_size != int(tensor_record.get("bytes", -1))
        ):
            raise NativeTeacherAuthorityError("native teacher task authority changed")
        values = load_file(str(tensor_path), device=str(self.device))
        self.tensor_reads += 1
        result: dict[tuple[int, str], NativeTeacherFactors] = {}
        for row in rows:
            prefix = str(row.get("tensor_prefix", ""))
            key = (int(row.get("video_demo", -1)), str(row.get("member_name", "")))
            factors = NativeTeacherFactors(
                authority_id=authority_id,
                video_demo=key[0],
                member_name=key[1],
                a=tuple(
                    values[f"{prefix}.a.{index}"]
                    for index in range(len(self.contract.targets))
                ),
                b=tuple(
                    values[f"{prefix}.b.{index}"]
                    for index in range(len(self.contract.targets))
                ),
                scales=values[f"{prefix}.scales"],
                provenance=dict(row.get("provenance", {})),
            )
            _validate_teacher(factors, self.contract)
            if key in result:
                raise NativeTeacherAuthorityError("duplicate native teacher key")
            result[key] = factors
        if set(values) != {
            name
            for row in rows
            for name in _teacher_tensor_names(str(row["tensor_prefix"]), len(self.contract.targets))
        }:
            raise NativeTeacherAuthorityError("native teacher tensor payload changed")
        return result


def write_native_teacher_task_shard(
    *,
    worker_dir: Path,
    task: Mapping[str, Any],
    teachers: Sequence[NativeTeacherFactors],
    contract: LoRAContract,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write one task-owned shard; workers never share a task path."""

    authority_id = int(task.get("authority_id", -1))
    role = str(task.get("role", ""))
    if role not in {"meta_fit", "target_fit"} or not teachers:
        raise NativeTeacherAuthorityError("native teacher shard is not fit-only")
    for teacher in teachers:
        _validate_teacher(teacher, contract)
        if teacher.authority_id != authority_id:
            raise NativeTeacherAuthorityError("native teacher shard mixed tasks")
    keys = [(row.video_demo, row.member_name) for row in teachers]
    videos = sorted({row[0] for row in keys})
    members = sorted({row[1] for row in keys})
    if len(keys) != len(set(keys)) or set(keys) != {
        (video, member) for video in videos for member in members
    }:
        raise NativeTeacherAuthorityError("native teacher shard is not a full video/member grid")

    worker_dir = worker_dir.resolve()
    task_dir = worker_dir / f"task_{authority_id:03d}"
    partial = worker_dir / f".task_{authority_id:03d}.partial"
    if task_dir.exists() or partial.exists():
        raise NativeTeacherAuthorityError("native teacher task output already exists")
    partial.mkdir(parents=True)
    values: dict[str, torch.Tensor] = {}
    rows = []
    ordered_teachers = sorted(
        teachers, key=lambda row: (row.video_demo, row.member_name)
    )
    for index, teacher in enumerate(ordered_teachers):
        prefix = f"teacher.{index:04d}"
        for target, (a, b) in enumerate(zip(teacher.a, teacher.b, strict=True)):
            values[f"{prefix}.a.{target}"] = a.detach().cpu().contiguous()
            values[f"{prefix}.b.{target}"] = b.detach().cpu().contiguous()
        values[f"{prefix}.scales"] = teacher.scales.detach().cpu().contiguous()
        rows.append(
            {
                "authority_id": authority_id,
                "video_demo": teacher.video_demo,
                "member_name": teacher.member_name,
                "tensor_prefix": prefix,
                "provenance": dict(teacher.provenance),
            }
        )
    tensor_path = partial / "native_teachers.safetensors"
    save_file(
        values,
        str(tensor_path),
        metadata={
            "schema_version": G3_NATIVE_TEACHER_TASK_SCHEMA,
            "provenance": json.dumps(dict(provenance), sort_keys=True),
        },
    )
    manifest = {
        "schema_version": G3_NATIVE_TEACHER_TASK_SCHEMA,
        "status": "complete",
        "task": dict(task),
        "formal_macros": G3_NATIVE_TEACHER_FORMAL_MACROS,
        "video_demos": videos,
        "member_names": members,
        "teacher_count": len(rows),
        "teachers": rows,
        "tensor_file": {
            "path": str((task_dir / tensor_path.name).resolve()),
            "bytes": tensor_path.stat().st_size,
        },
        "provenance": dict(provenance),
        "information_wall": {
            "fit_only": True,
            "K": 1,
            "stored_tensors": ["native_projected_A", "native_projected_B", "scales"],
            "stored_banks_logits_or_weights": False,
        },
    }
    write_json_atomic(partial / "manifest.json", manifest)
    os.replace(partial, task_dir)
    manifest_path = task_dir / "manifest.json"
    return {
        "authority_id": authority_id,
        "role": role,
        "manifest": str(manifest_path.resolve()),
        "manifest_bytes": manifest_path.stat().st_size,
        "video_count": len(videos),
        "member_count": len(members),
        "teacher_count": len(rows),
    }


def publish_native_teacher_root(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    contract: LoRAContract,
    fit_authority_roles: Mapping[int, str],
    provenance: Mapping[str, Any],
) -> Path:
    """Publish the immutable root only after shard aggregation is complete."""

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() or not records:
        raise NativeTeacherAuthorityError("native teacher root publication is unsafe")
    ordered = sorted((dict(row) for row in records), key=lambda row: int(row["authority_id"]))
    task_ids = [int(row["authority_id"]) for row in ordered]
    full_fit_ids = sorted(map(int, fit_authority_roles))
    if len(task_ids) != len(set(task_ids)) or any(
        row.get("role") not in {"meta_fit", "target_fit"} for row in ordered
    ):
        raise NativeTeacherAuthorityError("native teacher root contains duplicate or held tasks")
    if (
        not set(task_ids) <= set(full_fit_ids)
        or any(
            role not in {"meta_fit", "target_fit"}
            for role in fit_authority_roles.values()
        )
        or any(
            row["role"] != fit_authority_roles[int(row["authority_id"])]
            for row in ordered
        )
    ):
        raise NativeTeacherAuthorityError("native teacher root crossed its full fit authority")
    roles = {
        role: sum(row["role"] == role for row in ordered)
        for role in ("meta_fit", "target_fit")
    }
    fit_roles = {
        role: sum(value == role for value in fit_authority_roles.values())
        for role in ("meta_fit", "target_fit")
    }
    write_json_atomic(
        manifest_path,
        {
            "schema_version": G3_NATIVE_TEACHER_ROOT_SCHEMA,
            "status": "complete",
            "formal_macros": G3_NATIVE_TEACHER_FORMAL_MACROS,
            "task_count": len(ordered),
            "fit_authority_task_ids": full_fit_ids,
            "fit_authority_task_count": len(full_fit_ids),
            "fit_authority_roles": fit_roles,
            "K1_covered_task_count": len(task_ids),
            "K1_missing_task_ids": sorted(set(full_fit_ids) - set(task_ids)),
            "video_count": sum(int(row["video_count"]) for row in ordered),
            "member_count": sum(int(row["member_count"]) for row in ordered),
            "teacher_count": sum(int(row["teacher_count"]) for row in ordered),
            "roles": roles,
            "coverage": {
                "task_ids": task_ids,
                "roles": roles,
                "definition": "macro1-40 formal schedule K1-covered fit subset",
            },
            "contract": _contract_record(contract),
            "records": ordered,
            "provenance": dict(provenance),
            "information_wall": {
                "roles": ["meta_fit", "target_fit"],
                "held_task_reads": 0,
                "deployment_use": False,
                "K1_only": True,
                "K2_K4_tensor_reads": 0,
                "action_meta_installed": False,
            },
        },
    )
    return manifest_path


def _validate_factor_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[0] == 0:
        raise ValueError("native teacher low-rank factors changed shape")


def _normalized_fixed_weights(
    values: torch.Tensor | None,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if values is None:
        return torch.full((count,), 1.0 / count, device=device, dtype=dtype)
    values = values.detach().to(device=device, dtype=dtype)
    if values.shape != (count,) or torch.any(values < 0) or not torch.isfinite(values).all():
        raise ValueError(f"native teacher {name} weights changed")
    total = values.sum()
    if not torch.isclose(total, torch.ones((), device=device, dtype=dtype)):
        raise ValueError(f"native teacher {name} weights are not normalized")
    return values


def _validate_family_weights(values: torch.Tensor, owners: Sequence[TargetOwner]) -> None:
    if values.shape[1] != len(owners) or torch.any(values < 0) or not torch.isfinite(values).all():
        raise ValueError("native teacher target weights changed")
    expected = values.new_full((values.shape[0],), 0.25)
    for family in TargetFamily:
        indices = [index for index, owner in enumerate(owners) if owner.family == family]
        if not indices or not torch.allclose(values[:, indices].sum(-1), expected):
            raise ValueError("native teacher target weights lost family equality")


def _validate_teacher(teacher: NativeTeacherFactors, contract: LoRAContract) -> None:
    if len(teacher.a) != len(contract.targets) or len(teacher.b) != len(
        contract.targets
    ):
        raise NativeTeacherAuthorityError("native teacher target count changed")
    tensors = (*teacher.a, *teacher.b, teacher.scales)
    if (
        contract.rank != 4
        or teacher.authority_id < 0
        or not 0 <= teacher.video_demo < 50
        or not teacher.member_name
        or teacher.scales.shape != (len(contract.targets), 4)
        or torch.any(teacher.scales.abs() <= 1e-12)
        or any(
            a.shape != (4, target.in_features)
            or b.shape != (4, target.out_features)
            for target, a, b in zip(
                contract.targets, teacher.a, teacher.b, strict=True
            )
        )
        or any(value.requires_grad or not torch.isfinite(value).all() for value in tensors)
    ):
        raise NativeTeacherAuthorityError("native teacher factor authority changed")
    validate_lora_state(teacher.lora_state(contract), contract)


def _teacher_tensor_names(prefix: str, targets: int) -> set[str]:
    return {
        *(f"{prefix}.a.{index}" for index in range(targets)),
        *(f"{prefix}.b.{index}" for index in range(targets)),
        f"{prefix}.scales",
    }


def _contract_record(contract: LoRAContract) -> dict[str, Any]:
    return {
        "rank": int(contract.rank),
        "targets": [
            {
                "name": target.name,
                "in_features": target.in_features,
                "out_features": target.out_features,
            }
            for target in contract.targets
        ],
    }
