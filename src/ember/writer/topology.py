"""Runtime topology helpers shared by Writer training and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ember.pi05_source_checkpoint import DistributedContext
from ember.writer.model import WriterModelError


def _expand_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return cpus


def cuda_numa_node(device: int) -> int | None:
    """Resolve one visible CUDA device to its Linux NUMA node."""

    properties = torch.cuda.get_device_properties(device)
    pci_address = (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )
    numa_path = Path("/sys/bus/pci/devices") / pci_address / "numa_node"
    try:
        numa_node = int(numa_path.read_text(encoding="utf-8").strip())
        return numa_node if numa_node >= 0 else None
    except (OSError, ValueError):
        return None


def bind_current_process_to_cuda_numa(device: int) -> tuple[int, ...] | None:
    """Constrain this process and its future children to the GPU-local NUMA node."""

    numa_node = cuda_numa_node(device)
    if numa_node is None:
        return None
    try:
        cpus = _expand_cpu_list(
            (Path("/sys/devices/system/node") / f"node{numa_node}" / "cpulist")
            .read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    eligible = cpus.intersection(os.sched_getaffinity(0))
    if not eligible:
        return None
    os.sched_setaffinity(0, eligible)
    return tuple(sorted(eligible))


def validate_task_complete_topology(
    config: Mapping[str, Any],
    context: DistributedContext,
    *,
    expected_world_size: int,
    batch_size: int,
    mode: str,
) -> None:
    """Seal the v6 four-rank, all-task macro-update topology."""

    if context.world_size != expected_world_size:
        raise WriterModelError(
            "AS-Writer training requires exactly "
            f"{expected_world_size} symmetric ranks"
        )
    training = config["conditioning_training"]
    tasks_per_rank = int(training["tasks_per_rank_per_optimizer_update"])
    global_tasks = int(training["global_tasks_per_optimizer_update"])
    invalid = (
        int(training["teacher_videos_per_task_visit"]) != 1
        or tasks_per_rank * context.world_size != global_tasks
        or global_tasks != int(config["data"]["task_count"])
    )
    if invalid:
        raise WriterModelError(
            "AS-Writer macro update must cover every task exactly once"
        )
    if mode == "profile" and batch_size not in {20, 16}:
        raise WriterModelError("v6 profile allows B20 or the single B16 fallback")
