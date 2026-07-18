"""Streaming, partition-scoped LIBERO HDF5 access for Gate 0."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np


class GateZeroDataError(RuntimeError):
    """Raised when an HDF5 source or requested partition violates the seal."""


class GateZeroSurface(StrEnum):
    BASE_FIT = "source_base_fit"
    SUPPORT = "oracle_support"
    QUERY = "functional_query"
    REPORT = "locked_source_report"


@dataclass(frozen=True)
class Hdf5TaskAuthority:
    task_id: int
    language: str
    path: Path
    expected_bytes: int
    expected_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_task_authority(authority: Hdf5TaskAuthority, *, verify_sha256: bool) -> None:
    path = authority.path.resolve()
    if not path.is_file() or path.stat().st_size != authority.expected_bytes:
        raise GateZeroDataError(f"HDF5 authority bytes changed for task {authority.task_id}")
    if verify_sha256:
        if authority.expected_sha256 is None or _sha256(path) != authority.expected_sha256:
            raise GateZeroDataError(f"HDF5 authority SHA256 changed for task {authority.task_id}")


def _json_sha256(path: Path) -> str:
    return _sha256(path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroDataError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise GateZeroDataError(f"invalid {label}: {path}")
    return value


def _verify_report_grant(path: Path | None, spec: dict[str, Any]) -> None:
    if path is None:
        raise GateZeroDataError("locked report requires a selection-freeze grant")
    grant = _load_json(path, "selection-freeze grant")
    expected = {
        "schema_version": 1,
        "status": "oracle_selection_frozen_before_report_access",
        "pilot_name": spec["name"],
        "task_ids": spec["data"]["task_ids"],
        "report_access_authorized": True,
    }
    for key, value in expected.items():
        if grant.get(key) != value:
            raise GateZeroDataError(f"selection-freeze grant has wrong {key}")
    selected = grant.get("selected_adapter_sha256_by_task")
    if not isinstance(selected, dict) or set(selected) != {str(value) for value in spec["data"]["task_ids"]}:
        raise GateZeroDataError("selection-freeze grant lacks selected adapter hashes")
    if not all(isinstance(value, str) and len(value) == 64 for value in selected.values()):
        raise GateZeroDataError("selection-freeze grant has invalid selected adapter hash")


def load_surface_authorities(
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    manifest_path: Path,
    dataset_root: Path,
    surface: GateZeroSurface,
    oracle_task_id: int | None = None,
    report_access_grant: Path | None = None,
) -> tuple[list[Hdf5TaskAuthority], list[int]]:
    """Derive the only legal task/demo authority for a named Gate 0 surface."""

    if _json_sha256(manifest_path) != spec["authority"]["canonical_manifest_sha256"]:
        raise GateZeroDataError("canonical manifest SHA256 changed")
    manifest = _load_json(manifest_path, "canonical manifest")
    records = manifest.get("tasks")
    if not isinstance(records, list):
        raise GateZeroDataError("canonical manifest lacks task records")
    source_ids = phase0["splits"]["source"]
    if surface is GateZeroSurface.REPORT:
        _verify_report_grant(report_access_grant, spec)
    if surface is GateZeroSurface.BASE_FIT:
        if oracle_task_id is not None:
            raise GateZeroDataError("base-fit surface cannot select one oracle_task_id")
        task_ids = source_ids
    else:
        if oracle_task_id not in spec["data"]["task_ids"]:
            raise GateZeroDataError("task-local surface requires one declared oracle_task_id")
        task_ids = [oracle_task_id]
    by_id = {record.get("task_index"): record for record in records if isinstance(record, dict)}
    authorities: list[Hdf5TaskAuthority] = []
    for task_id in task_ids:
        record = by_id.get(task_id)
        if record is None or record.get("split") != "source":
            raise GateZeroDataError(f"task {task_id} is not a sealed source task")
        hdf5 = record.get("hdf5")
        if not isinstance(hdf5, dict):
            raise GateZeroDataError(f"task {task_id} lacks HDF5 authority")
        path = dataset_root / hdf5["filename"]
        authorities.append(
            Hdf5TaskAuthority(
                task_id=task_id,
                language=record["language"],
                path=path,
                expected_bytes=hdf5["bytes"],
                expected_sha256=hdf5["sha256"],
            )
        )
    bounds = spec["access"][surface.value]
    return authorities, list(range(bounds[0], bounds[1] + 1))


def _validate_demo(demo: h5py.Group, label: str) -> int:
    required = {
        "actions": ((7,), np.dtype("float64")),
        "obs/agentview_rgb": ((128, 128, 3), np.dtype("uint8")),
        "obs/eye_in_hand_rgb": ((128, 128, 3), np.dtype("uint8")),
        "obs/ee_states": ((6,), np.dtype("float64")),
        "obs/gripper_states": ((2,), np.dtype("float64")),
    }
    if "actions" not in demo:
        raise GateZeroDataError(f"missing actions in {label}")
    steps = int(demo["actions"].shape[0])
    if steps <= 0 or int(demo.attrs.get("num_samples", -1)) != steps:
        raise GateZeroDataError(f"invalid num_samples in {label}")
    for key, (tail, dtype) in required.items():
        if key not in demo or not isinstance(demo[key], h5py.Dataset):
            raise GateZeroDataError(f"missing {key} in {label}")
        value = demo[key]
        if value.shape != (steps, *tail):
            raise GateZeroDataError(f"invalid {key} shape in {label}")
        if value.dtype != dtype:
            raise GateZeroDataError(f"invalid {key} dtype in {label}")
    return steps


def build_frame_index(
    authorities: Sequence[Hdf5TaskAuthority],
    *,
    demo_indices: Sequence[int],
    verify_sha256: bool = True,
) -> list[tuple[int, int, int]]:
    """Return deterministic ``(task, demo, frame)`` rows for one declared partition."""

    if not authorities or not demo_indices:
        raise GateZeroDataError("authorities and demo indices must be non-empty")
    if len({item.task_id for item in authorities}) != len(authorities):
        raise GateZeroDataError("duplicate HDF5 task authority")
    if len(set(demo_indices)) != len(demo_indices) or any(index < 0 for index in demo_indices):
        raise GateZeroDataError("demo indices must be unique non-negative integers")
    rows: list[tuple[int, int, int]] = []
    for authority in sorted(authorities, key=lambda item: item.task_id):
        verify_task_authority(authority, verify_sha256=verify_sha256)
        with h5py.File(authority.path, "r") as handle:
            if "data" not in handle or not isinstance(handle["data"], h5py.Group):
                raise GateZeroDataError(f"missing data group for task {authority.task_id}")
            data = handle["data"]
            for demo_index in demo_indices:
                name = f"demo_{demo_index}"
                if name not in data or not isinstance(data[name], h5py.Group):
                    raise GateZeroDataError(f"missing {name} for task {authority.task_id}")
                steps = _validate_demo(data[name], f"task {authority.task_id} {name}")
                rows.extend((authority.task_id, demo_index, frame) for frame in range(steps))
    return rows


class SourceHdf5Dataset:
    """Lazy per-process HDF5 reader that never expands its declared demo allowlist."""

    def __init__(
        self,
        authorities: Sequence[Hdf5TaskAuthority],
        *,
        demo_indices: Sequence[int],
        action_chunk_size: int,
        verify_sha256: bool = True,
    ) -> None:
        if action_chunk_size <= 0:
            raise GateZeroDataError("action chunk size must be positive")
        self._authorities = {item.task_id: item for item in authorities}
        self._index = build_frame_index(
            authorities, demo_indices=demo_indices, verify_sha256=verify_sha256
        )
        self.action_chunk_size = action_chunk_size
        self._handles: dict[tuple[int, int], h5py.File] = {}

    @property
    def frame_index(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self, task_id: int) -> h5py.File:
        current_pid = os.getpid()
        stale = [key for key in self._handles if key[0] != current_pid]
        for key in stale:
            try:
                self._handles[key].close()
            finally:
                del self._handles[key]
        key = (current_pid, task_id)
        handle = self._handles.get(key)
        if handle is None:
            handle = h5py.File(self._authorities[task_id].path, "r")
            self._handles[key] = handle
        return handle

    @staticmethod
    def _camera(value: np.ndarray) -> np.ndarray:
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise GateZeroDataError("camera frame changed shape or dtype")
        return np.ascontiguousarray(value[::-1, ::-1].transpose(2, 0, 1))

    def __getitem__(self, item: int) -> dict[str, Any]:
        task_id, demo_index, frame_index = self._index[item]
        demo = self._handle(task_id)["data"][f"demo_{demo_index}"]
        steps = int(demo["actions"].shape[0])
        end = min(frame_index + self.action_chunk_size, steps)
        valid = end - frame_index
        valid_actions = np.asarray(demo["actions"][frame_index:end], dtype=np.float32)
        actions = np.repeat(valid_actions[-1:], self.action_chunk_size, axis=0)
        actions[:valid] = valid_actions
        action_is_pad = np.ones(self.action_chunk_size, dtype=np.bool_)
        action_is_pad[:valid] = False
        ee = np.asarray(demo["obs/ee_states"][frame_index], dtype=np.float32)
        gripper = np.asarray(demo["obs/gripper_states"][frame_index], dtype=np.float32)
        state = np.concatenate((ee, gripper), dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise GateZeroDataError("non-finite or wrong-dimensional state/action sample")
        return {
            "observation.images.camera1": self._camera(
                np.asarray(demo["obs/agentview_rgb"][frame_index])
            ),
            "observation.images.camera2": self._camera(
                np.asarray(demo["obs/eye_in_hand_rgb"][frame_index])
            ),
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
            "task": self._authorities[task_id].language,
            "task_id": task_id,
            "demo_index": demo_index,
            "frame_index": frame_index,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # h5py globals may already be torn down during interpreter shutdown.
            pass


class TaskDemoFrameBatchSampler:
    """Deterministic task→demo→frame-uniform batches with O(1) step resume."""

    def __init__(
        self,
        dataset: SourceHdf5Dataset,
        *,
        micro_batch_size: int,
        optimizer_steps: int,
        gradient_accumulation_steps: int,
        seed: int,
        start_optimizer_step: int = 0,
        rank: int = 0,
        world_size: int = 1,
        global_effective_batch_size: int | None = None,
    ) -> None:
        if (
            micro_batch_size <= 0
            or optimizer_steps <= 0
            or gradient_accumulation_steps <= 0
            or start_optimizer_step < 0
        ):
            raise GateZeroDataError("invalid hierarchical batch sampler bounds")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise GateZeroDataError("invalid distributed sampler rank")
        inferred_global_batch = micro_batch_size * gradient_accumulation_steps * world_size
        if global_effective_batch_size is None:
            global_effective_batch_size = inferred_global_batch
        if global_effective_batch_size != inferred_global_batch:
            raise GateZeroDataError("distributed sampler changes the global effective batch")
        self.micro_batch_size = micro_batch_size
        self.optimizer_steps = optimizer_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.start_optimizer_step = start_optimizer_step
        self.rank = rank
        self.world_size = world_size
        self.global_effective_batch_size = global_effective_batch_size
        nested: dict[int, dict[int, list[int]]] = {}
        for flat_index, (task_id, demo_index, _) in enumerate(dataset.frame_index):
            nested.setdefault(task_id, {}).setdefault(demo_index, []).append(flat_index)
        if not nested:
            raise GateZeroDataError("cannot sample an empty Gate 0 dataset")
        self._nested = nested
        self._tasks = sorted(nested)

    def __len__(self) -> int:
        return self.optimizer_steps * self.gradient_accumulation_steps

    def __iter__(self) -> Iterator[list[int]]:
        stop = self.start_optimizer_step + self.optimizer_steps
        for optimizer_step in range(self.start_optimizer_step, stop):
            for accumulation_step in range(self.gradient_accumulation_steps):
                batch = []
                for local_slot in range(self.micro_batch_size):
                    effective_batch_slot = (
                        accumulation_step * self.world_size * self.micro_batch_size
                        + self.rank * self.micro_batch_size
                        + local_slot
                    )
                    if effective_batch_slot >= self.global_effective_batch_size:
                        raise GateZeroDataError("sampler rank shard escaped the global batch")
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [self.seed, optimizer_step, effective_batch_slot]
                        )
                    )
                    task_id = self._tasks[int(rng.integers(len(self._tasks)))]
                    demos = sorted(self._nested[task_id])
                    demo_id = demos[int(rng.integers(len(demos)))]
                    frames = self._nested[task_id][demo_id]
                    batch.append(frames[int(rng.integers(len(frames)))])
                yield batch
