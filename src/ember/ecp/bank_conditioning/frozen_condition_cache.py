"""Node-local frozen Program/native-bank cache for G3 mapping acquisition.

The cache owns no learned state.  It only seals the output of frozen Pass A,
raw X/Y, and the deterministic B0 preparation of one action-hidden video so
repeated P2 student updates can execute canonical fixed-microblock B1 replay. Files are
run-local operational artifacts and never enter a Writer checkpoint.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.bank_conditioning.primal_dual import SpectralNativeCovariance
from ember.ecp.bank_conditioning.primal_dual_runtime import (
    CompactPrimalDualVideo,
    PrimalDualVideoOperator,
)
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import native_output_group_count
from ember.ecp.natural_program import FrozenProgramEvidence, NaturalProgram
from ember.ecp.shared_compiler_data import (
    SharedCompilerCondition,
    cache_shared_compiler_native_replay,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


FROZEN_CONDITION_CACHE_SCHEMA = "ember_ecp_g3_frozen_condition_cache_v4"
_PROGRAM_FIELDS = ("p_lang", "p_scene", "p_process", "rho", "tau", "sigma")
_EVIDENCE_FIELDS = tuple(FrozenProgramEvidence.__dataclass_fields__)


@dataclass(frozen=True)
class FrozenSharedCompilerCondition:
    """One frozen Program, raw native bank, and its B0 spectral operator."""

    program: NaturalProgram | None
    videos: tuple[CompactPrimalDualVideo, ...]
    metrics: dict[str, Any]
    evidence: FrozenProgramEvidence | None = None


@dataclass(frozen=True)
class FrozenConditionCacheResult:
    condition: FrozenSharedCompilerCondition
    hit: bool
    file_bytes: int
    build_seconds: float
    load_seconds: float


def frozen_condition_cache_authority(
    *,
    config_schema: str,
    config_bytes: int,
    source_checkpoint: Path,
    g2_program_checkpoint: Path,
    native_observer_checkpoint: Path,
    frame_stride: int,
    owners: Sequence[TargetOwner],
) -> dict[str, Any]:
    """Build the common train/evaluation identity for frozen cache contents."""

    return {
        "config_schema": config_schema,
        "config_bytes": int(config_bytes),
        "source_checkpoint": str(source_checkpoint),
        "g2_program_checkpoint": str(g2_program_checkpoint),
        "native_observer_checkpoint": str(native_observer_checkpoint),
        "frame_stride": int(frame_stride),
        "owner_shapes": [
            [owner.family.value, owner.in_features, owner.out_features]
            for owner in owners
        ],
    }


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def _operator_tensors(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    operator: SpectralNativeCovariance,
) -> None:
    tensors[f"{prefix}.basis"] = _cpu_tensor(operator.basis)
    tensors[f"{prefix}.eigenvalues"] = _cpu_tensor(operator.eigenvalues)
    tensors[f"{prefix}.eigenvalue_floor"] = _cpu_tensor(
        operator.eigenvalue_floor
    )
    tensors[f"{prefix}.retained_condition"] = _cpu_tensor(
        operator.retained_condition
    )
    tensors[f"{prefix}.retained_trace_fraction"] = _cpu_tensor(
        operator.retained_trace_fraction
    )


def _condition_tensors(
    condition: FrozenSharedCompilerCondition,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if condition.program is not None:
        tensors.update(
            {
                f"program.{name}": _cpu_tensor(getattr(condition.program, name))
                for name in _PROGRAM_FIELDS
            }
        )
    if condition.evidence is not None:
        tensors.update(
            {
                f"evidence.{name}": _cpu_tensor(
                    getattr(condition.evidence, name)
                )
                for name in _EVIDENCE_FIELDS
            }
        )
    for video_index, video in enumerate(condition.videos):
        prefix = f"video.{video_index}"
        tensors[f"{prefix}.frame_measure"] = _cpu_tensor(video.frame_measure)
        for target, value in enumerate(video.input_values):
            tensors[f"{prefix}.input_values.{target}"] = _cpu_tensor(value)
            _operator_tensors(
                tensors,
                f"{prefix}.input_operator.{target}",
                video.input_operators[target],
            )
        for target, values in enumerate(video.output_values):
            tensors[f"{prefix}.output_values.{target}"] = _cpu_tensor(values)
            tensors[f"{prefix}.final_outputs.{target}"] = _cpu_tensor(
                video.final_outputs[target]
            )
            for group, operator in enumerate(video.output_operators[target]):
                _operator_tensors(
                    tensors,
                    f"{prefix}.output_operator.{target}.{group}",
                    operator,
                )
    return tensors


def _load_operator(
    tensors: Mapping[str, torch.Tensor],
    prefix: str,
    *,
    expected_width: int,
) -> SpectralNativeCovariance:
    basis = tensors[f"{prefix}.basis"]
    eigenvalues = tensors[f"{prefix}.eigenvalues"]
    if (
        basis.ndim != 2
        or basis.shape[0] != expected_width
        or eigenvalues.shape != (basis.shape[1],)
        or basis.shape[1] <= 0
    ):
        raise ValueError("frozen condition spectral operator changed")
    return SpectralNativeCovariance(
        basis=basis,
        eigenvalues=eigenvalues,
        native_width=expected_width,
        retained_rank=basis.shape[1],
        eigenvalue_floor=tensors[f"{prefix}.eigenvalue_floor"],
        retained_condition=tensors[f"{prefix}.retained_condition"],
        retained_trace_fraction=tensors[f"{prefix}.retained_trace_fraction"],
    )


def _decode_condition(
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
    owners: Sequence[TargetOwner],
) -> FrozenSharedCompilerCondition:
    video_count = int(metadata["video_count"])
    program = None
    if metadata.get("has_program") == "true":
        program = NaturalProgram(
            **{name: tensors[f"program.{name}"] for name in _PROGRAM_FIELDS}
        )
    evidence = None
    if metadata.get("has_program_evidence") == "true":
        evidence = FrozenProgramEvidence(
            **{
                name: tensors[f"evidence.{name}"]
                for name in _EVIDENCE_FIELDS
            }
        )
    videos = []
    for video_index in range(video_count):
        prefix = f"video.{video_index}"
        input_values = tuple(
            tensors[f"{prefix}.input_values.{target}"]
            for target in range(len(owners))
        )
        output_values = tuple(
            tensors[f"{prefix}.output_values.{target}"]
            for target in range(len(owners))
        )
        final_outputs = tuple(
            tensors[f"{prefix}.final_outputs.{target}"]
            for target in range(len(owners))
        )
        input_operators = tuple(
            _load_operator(
                tensors,
                f"{prefix}.input_operator.{target}",
                expected_width=owner.in_features,
            )
            for target, owner in enumerate(owners)
        )
        output_operators = tuple(
            tuple(
                _load_operator(
                    tensors,
                    f"{prefix}.output_operator.{target}.{group}",
                    expected_width=(
                        owner.out_features // native_output_group_count(owner)
                    ),
                )
                for group in range(native_output_group_count(owner))
            )
            for target, owner in enumerate(owners)
        )
        videos.append(
            CompactPrimalDualVideo(
                frame_measure=tensors[f"{prefix}.frame_measure"],
                input_operators=input_operators,
                output_operators=output_operators,
                input_values=input_values,
                output_values=output_values,
                final_outputs=final_outputs,
            )
        )
    metrics = json.loads(metadata["metrics_json"])
    if not isinstance(metrics, dict):
        raise ValueError("frozen condition metrics changed")
    return FrozenSharedCompilerCondition(
        program=program,
        videos=tuple(videos),
        metrics=metrics,
        evidence=evidence,
    )


def _metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        result = handle.metadata()
    if result is None or result.get("schema_version") != FROZEN_CONDITION_CACHE_SCHEMA:
        raise ValueError("frozen condition cache file schema changed")
    return dict(result)


class FrozenMappingConditionCache:
    """Atomic node-local cache shared by P2 train/evaluation workers."""

    def __init__(
        self,
        root: Path,
        *,
        owners: Sequence[TargetOwner],
        operator: PrimalDualVideoOperator,
        authority: Mapping[str, Any],
        cache_program: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.owners = tuple(owners)
        self.operator = operator
        self.cache_program = bool(cache_program)
        if not self.root.is_absolute() or self.root in (
            Path("/"),
            Path("/tmp"),
            Path("/dev/shm"),
        ):
            raise ValueError("frozen condition cache root is not task-scoped")
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority = {
            "schema_version": FROZEN_CONDITION_CACHE_SCHEMA,
            **dict(authority),
            "checkpoint_payload": False,
            "deployment_input": False,
            "cache_program_output": self.cache_program,
        }
        lock_path = self.root / ".manifest.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            manifest = self.root / "manifest.json"
            if manifest.is_file():
                if read_json(manifest) != self.authority:
                    raise ValueError("frozen condition cache authority changed")
            else:
                write_json_atomic(manifest, self.authority)

    def _path(self, authority_id: int, video_demo: int) -> Path:
        if authority_id < 0 or video_demo < 0:
            raise ValueError("frozen condition cache key changed")
        return self.root / f"task_{authority_id:03d}_video_{video_demo:03d}.safetensors"

    def _freeze(
        self, condition: SharedCompilerCondition
    ) -> FrozenSharedCompilerCondition:
        with self.operator.ieee_matmul(condition.program.p_lang.device):
            cached = cache_shared_compiler_native_replay(condition)
            videos = tuple(
                self.operator.compact(self.operator.prepare(video))
                for video in cached.videos
            )
        return FrozenSharedCompilerCondition(
            program=cached.program if self.cache_program else None,
            videos=videos,
            metrics=dict(cached.metrics),
            evidence=cached.evidence,
        )

    def _save(
        self,
        path: Path,
        condition: FrozenSharedCompilerCondition,
        *,
        authority_id: int,
        video_demo: int,
    ) -> None:
        temporary = path.with_name(
            f".{path.name}.tmp.{os.getpid()}"
        )
        metadata = {
            "schema_version": FROZEN_CONDITION_CACHE_SCHEMA,
            "authority_id": str(authority_id),
            "video_demo": str(video_demo),
            "video_count": str(len(condition.videos)),
            "has_program": str(condition.program is not None).lower(),
            "has_program_evidence": str(condition.evidence is not None).lower(),
            "metrics_json": json.dumps(
                condition.metrics, sort_keys=True, separators=(",", ":")
            ),
        }
        try:
            save_file(_condition_tensors(condition), str(temporary), metadata=metadata)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load(
        self,
        path: Path,
        device: torch.device,
        *,
        authority_id: int,
        video_demo: int,
    ) -> FrozenSharedCompilerCondition:
        metadata = _metadata(path)
        if (
            int(metadata.get("authority_id", -1)) != authority_id
            or int(metadata.get("video_demo", -1)) != video_demo
        ):
            raise ValueError("frozen condition cache pairing changed")
        tensors = load_file(str(path), device=str(device))
        return _decode_condition(tensors, metadata=metadata, owners=self.owners)

    def get_or_build(
        self,
        *,
        authority_id: int,
        video_demo: int,
        device: torch.device,
        builder: Callable[[], SharedCompilerCondition],
        retain: bool = True,
    ) -> FrozenConditionCacheResult:
        path = self._path(authority_id, video_demo)
        if not retain and not path.is_file():
            tick = time.monotonic()
            built = self._freeze(builder())
            return FrozenConditionCacheResult(
                condition=built,
                hit=False,
                file_bytes=0,
                build_seconds=time.monotonic() - tick,
                load_seconds=0.0,
            )
        lock_path = path.with_suffix(".lock")
        built: FrozenSharedCompilerCondition | None = None
        build_seconds = 0.0
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            hit = path.is_file()
            if not hit:
                tick = time.monotonic()
                built = self._freeze(builder())
                self._save(
                    path,
                    built,
                    authority_id=authority_id,
                    video_demo=video_demo,
                )
                build_seconds = time.monotonic() - tick
        load_tick = time.monotonic()
        condition = (
            built
            if built is not None
            else self._load(
                path,
                device,
                authority_id=authority_id,
                video_demo=video_demo,
            )
        )
        load_seconds = 0.0 if built is not None else time.monotonic() - load_tick
        return FrozenConditionCacheResult(
            condition=condition,
            hit=hit,
            file_bytes=path.stat().st_size,
            build_seconds=build_seconds,
            load_seconds=load_seconds,
        )

    def load_program(
        self,
        *,
        authority_id: int,
        video_demo: int,
        device: torch.device,
    ) -> NaturalProgram | None:
        path = self._path(authority_id, video_demo)
        if not path.is_file():
            return None
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device=str(device)) as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("schema_version") != FROZEN_CONDITION_CACHE_SCHEMA
                or int(metadata.get("authority_id", -1)) != authority_id
                or int(metadata.get("video_demo", -1)) != video_demo
            ):
                raise ValueError("frozen condition cache file schema changed")
            if metadata.get("has_program") != "true":
                return None
            values = {name: handle.get_tensor(f"program.{name}") for name in _PROGRAM_FIELDS}
        return NaturalProgram(**values)
