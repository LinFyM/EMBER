"""Training-only teacher-action alignment for non-held meta videos."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch

from ember.writer.data import WriterTaskAuthority, verify_authority


class PrivilegedMetaActionStore:
    """Read action chunks only for explicitly authorized meta-train tasks."""

    def __init__(
        self,
        authorities: Sequence[WriterTaskAuthority],
        *,
        action_q01: Sequence[float],
        action_q99: Sequence[float],
        phase_count: int,
        action_horizon: int = 50,
        max_open_files: int = 8,
    ) -> None:
        self.authorities = {row.task_id: row for row in authorities}
        self.q01 = np.asarray(action_q01, dtype=np.float32)
        self.q99 = np.asarray(action_q99, dtype=np.float32)
        self.phase_count = int(phase_count)
        self.horizon = int(action_horizon)
        self.max_open_files = int(max_open_files)
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        if (
            len(self.authorities) != len(authorities)
            or self.q01.shape != (7,)
            or self.q99.shape != (7,)
            or self.horizon % self.phase_count
            or self.max_open_files <= 0
        ):
            raise ValueError("invalid privileged meta-action authority")
        for authority in authorities:
            verify_authority(authority)

    def _handle(self, task_id: int) -> h5py.File:
        if task_id not in self.authorities:
            raise ValueError("action alignment escaped meta-train authority")
        if task_id in self._handles:
            self._handles.move_to_end(task_id)
        else:
            self._handles[task_id] = h5py.File(
                Path(self.authorities[task_id].path), "r"
            )
            while len(self._handles) > self.max_open_files:
                _, stale = self._handles.popitem(last=False)
                stale.close()
        return self._handles[task_id]

    def phase_targets(
        self,
        *,
        task_id: int,
        demos: Sequence[int],
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = video_offsets.detach().cpu().tolist()
        if len(demos) != len(offsets) - 1:
            raise ValueError("action targets differ from video ownership")
        rows = []
        handle = self._handle(task_id)
        indices = frame_indices.detach().cpu().tolist()
        for demo_index, (start, stop) in zip(demos, zip(offsets, offsets[1:])):
            actions = np.asarray(
                handle[f"data/demo_{int(demo_index)}/actions"], dtype=np.float32
            )
            for frame in indices[start:stop]:
                right = min(int(frame) + self.horizon, actions.shape[0])
                chunk = np.repeat(actions[right - 1 : right], self.horizon, axis=0)
                chunk[: right - int(frame)] = actions[int(frame) : right]
                normalized = (chunk - self.q01) / (self.q99 - self.q01 + 1e-6)
                normalized = normalized * 2.0 - 1.0
                rows.append(
                    normalized.reshape(
                        self.phase_count, self.horizon // self.phase_count, 7
                    ).mean(axis=1)
                )
        return torch.from_numpy(np.stack(rows)).to(device=device, non_blocking=True)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
