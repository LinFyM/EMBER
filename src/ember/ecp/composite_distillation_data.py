"""Lazy training data contract for composite-expert on-policy distillation."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ember.ecp.process_meta import ProcessMetaError


DISTILLATION_SHARD_SCHEMA = "ember_ecp_composite_distillation_shard_v1"
DISTILLATION_MANIFEST_SCHEMA = "ember_ecp_composite_distillation_manifest_v1"


@dataclass(frozen=True)
class CompositeDistillationSpec:
    variant_name: str
    language: str
    manifest_path: Path
    manifest_bytes: int
    shard_paths: tuple[Path, ...]
    shard_bytes: tuple[int, ...]
    query_count: int
    initial_adapter_path: Path
    initial_adapter_bytes: int


class CompositeDistillationDataset:
    """Replan-state queries with direct privileged action-chunk labels."""

    def __init__(self, spec: CompositeDistillationSpec, *, task_id: int = 0) -> None:
        self.spec = spec
        self.task_id = int(task_id)
        self._index: list[tuple[int, str, int]] = []
        self._handles: OrderedDict[tuple[int, int], h5py.File] = OrderedDict()
        for shard_index, path in enumerate(spec.shard_paths):
            with h5py.File(path, "r") as handle:
                for group_name in sorted(handle["episodes"]):
                    count = int(handle[f"episodes/{group_name}/state"].shape[0])
                    self._index.extend(
                        (shard_index, group_name, row) for row in range(count)
                    )
        if len(self._index) != spec.query_count:
            raise ProcessMetaError("distillation query index count changed")

    @property
    def task_rows(self) -> dict[int, tuple[int, ...]]:
        return {self.task_id: tuple(range(len(self._index)))}

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self, shard_index: int) -> h5py.File:
        key = (os.getpid(), shard_index)
        if key not in self._handles:
            self._handles[key] = h5py.File(self.spec.shard_paths[shard_index], "r")
            while len(self._handles) > 4:
                _, stale = self._handles.popitem(last=False)
                stale.close()
        self._handles.move_to_end(key)
        return self._handles[key]

    @staticmethod
    def _policy_camera(value: np.ndarray) -> np.ndarray:
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ProcessMetaError("distillation training camera changed")
        return np.ascontiguousarray(value[::-1, ::-1].transpose(2, 0, 1))

    def __getitem__(self, item: int) -> dict[str, Any]:
        shard_index, group_name, row = self._index[item]
        group = self._handle(shard_index)[f"episodes/{group_name}"]
        actions = np.asarray(group["teacher_action_chunks"][row], dtype=np.float32)
        if actions.shape != (50, 7):
            raise ProcessMetaError("distillation training action chunk changed")
        return {
            "observation.images.camera1": self._policy_camera(
                np.asarray(group["camera1"][row])
            ),
            "observation.images.camera2": self._policy_camera(
                np.asarray(group["camera2"][row])
            ),
            "observation.state": np.asarray(group["state"][row], dtype=np.float32),
            "action": actions,
            "action_is_pad": np.zeros(50, dtype=np.bool_),
            "task": self.spec.language,
            "task_id": self.task_id,
            "demo_index": int(group.attrs["state_id"]),
            "frame_index": row,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state


def load_distillation_spec(
    manifest_path: Path,
    *,
    data_root: Path,
    initial_adapter_path: Path,
    initial_adapter_bytes: int,
) -> CompositeDistillationSpec:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != DISTILLATION_MANIFEST_SCHEMA
        or value.get("status") != "completed_one_round_on_policy_phase_distillation"
        or value.get("state_ids") != list(range(50))
        or int(value.get("queries", -1)) <= 0
        or value.get("training_target")
        != "phase_expert_full50_action_chunk_on_composite_occupancy"
    ):
        raise ProcessMetaError("distillation dataset manifest changed")
    paths = tuple(data_root / str(row["path"]) for row in value["shards"])
    sizes = tuple(int(row["bytes"]) for row in value["shards"])
    if any(
        not path.is_file() or path.stat().st_size != size
        for path, size in zip(paths, sizes, strict=True)
    ):
        raise ProcessMetaError("distillation dataset shard changed")
    if (
        not initial_adapter_path.is_file()
        or initial_adapter_path.stat().st_size != initial_adapter_bytes
    ):
        raise ProcessMetaError("distillation initial adapter changed")
    return CompositeDistillationSpec(
        variant_name=str(value["variant_name"]),
        language=str(value["exact_language"]),
        manifest_path=manifest_path,
        manifest_bytes=manifest_path.stat().st_size,
        shard_paths=paths,
        shard_bytes=sizes,
        query_count=int(value["queries"]),
        initial_adapter_path=initial_adapter_path,
        initial_adapter_bytes=initial_adapter_bytes,
    )
