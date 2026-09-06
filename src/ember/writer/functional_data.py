"""Read the retained cross-episode functional panels without a retired runtime.

Panel roles are historical metadata, not authorization to produce gradients.
The new experiment must supply its own audited task allowlist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_source_checkpoint import read_json


def resolve_functional_panel_records(
    sources: Sequence[Mapping[str, Any]],
    task_ids: Sequence[int],
    *,
    asset_root: Path,
) -> dict[int, Path]:
    """Resolve completed panel sources for an explicitly selected task list."""

    selected = set(map(int, task_ids))
    if not selected or len(selected) != len(task_ids):
        raise ValueError("functional panel task selection changed")
    records: dict[int, Path] = {}
    for source in sources:
        root = (asset_root / str(source["root"])).resolve()
        completion = root / str(source["completion"])
        if (
            not root.is_dir()
            or not completion.is_file()
            or read_json(completion).get("status") != "complete"
        ):
            raise ValueError("functional panel source is incomplete")
        candidates = tuple(sorted(root.glob("shard_*/task_*.json")))
        if int(source["task_count"]) != len(candidates):
            raise ValueError("functional panel source task count changed")
        for path in candidates:
            task = int(path.stem.removeprefix("task_"))
            if task not in selected:
                continue
            if task in records:
                raise ValueError("functional panel sources overlap")
            records[task] = path
    if set(records) != selected:
        raise ValueError("functional panel source lost a selected task")
    return records


@dataclass(frozen=True)
class FunctionalPanelVisit:
    action_demos: tuple[int, ...]
    action_frames: tuple[int, ...]
    policy_rng_seed: int
    flow_loss: float


@dataclass(frozen=True)
class FunctionalPanelAuthority:
    task_id: int
    role: str
    panel_a: tuple[FunctionalPanelVisit, ...]
    panel_b: tuple[FunctionalPanelVisit, ...]
    program_video_demos: tuple[int, ...]
    path: Path


def _panel_visit(row: Mapping[str, Any]) -> FunctionalPanelVisit:
    demos = tuple(map(int, row.get("action_demos", ())))
    frames = tuple(map(int, row.get("action_frames", ())))
    seed = int(row.get("policy_rng_seed", -1))
    flow_loss = float(row.get("flow_loss", float("nan")))
    if (
        len(demos) != len(frames)
        or len(demos) != 16
        or seed < 0
        or not math.isfinite(flow_loss)
        or flow_loss <= 0
    ):
        raise ValueError("retained functional panel visit changed")
    return FunctionalPanelVisit(demos, frames, seed, flow_loss)


def load_functional_panels(
    records: Mapping[int | str, str | Path], *, asset_root: Path
) -> dict[int, FunctionalPanelAuthority]:
    """Read fit or historical held panels; this never chooses gradient tasks."""

    output = {}
    for task_key, relative in records.items():
        task_id = int(task_key)
        path = (asset_root / str(relative)).resolve()
        row = read_json(path)
        panel_a = tuple(_panel_visit(value) for value in row.get("panel_a_visits", ()))
        panel_b = tuple(_panel_visit(value) for value in row.get("panel_b_visits", ()))
        videos = tuple(map(int, row.get("program_video_demos", ())))
        a_demos = {demo for visit in panel_a for demo in visit.action_demos}
        b_demos = {demo for visit in panel_b for demo in visit.action_demos}
        if (
            int(row.get("task", -1)) != task_id
            or row.get("role") not in {"meta_fit", "target_fit", "meta_held", "target_held"}
            or len(panel_a) != 16
            or len(panel_b) != 16
            or int(row.get("logical_rows_per_panel", -1)) != 256
            or not videos
            or a_demos.intersection(b_demos)
            or a_demos.intersection(videos)
            or b_demos.intersection(videos)
            or row.get("episode_sets_pairwise_disjoint") is not True
        ):
            raise ValueError("retained functional panel authority changed")
        output[task_id] = FunctionalPanelAuthority(
            task_id=task_id,
            role=str(row["role"]),
            panel_a=panel_a,
            panel_b=panel_b,
            program_video_demos=videos,
            path=path,
        )
    return output
