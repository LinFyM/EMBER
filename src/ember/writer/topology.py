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
    """Seal a complete-task-cycle UCP update topology."""

    if context.world_size != expected_world_size:
        raise WriterModelError(
            "AS-Writer training requires exactly "
            f"{expected_world_size} symmetric ranks"
        )
    training = config["conditioning_training"]
    tasks_per_rank = int(training["tasks_per_rank_per_optimizer_update"])
    global_tasks = int(training["global_tasks_per_optimizer_update"])
    task_count = int(config["data"]["task_count"])
    update_topology = str(
        training.get("update_topology", "task_complete_all_tasks")
    )
    updates_per_cycle = int(
        training.get("optimizer_updates_per_task_cycle", 1)
    )
    invalid_common = (
        int(training["teacher_videos_per_task_visit"]) != 1
        or tasks_per_rank * context.world_size != global_tasks
        or global_tasks * updates_per_cycle != task_count
    )
    supported = (
        update_topology == "task_complete_all_tasks"
        and updates_per_cycle == 1
        and global_tasks == task_count
    ) or (
        update_topology in {
            "serial4_exposure_matched_six_phase_task_cycle",
            "cycle_normalized_randomized_group4_six_phase_task_cycle",
        }
        and updates_per_cycle == 6
        and tasks_per_rank == 1
        and global_tasks == 4
    )
    if invalid_common or not supported:
        raise WriterModelError(
            "AS-Writer update topology differs from its declared contract"
        )
    if mode == "profile":
        evidence = config.get("profile_evidence", {})
        candidates = {
            int(evidence[name]["per_task_action_batch_size"])
            for name in ("primary_candidate", "oom_fallback_only")
            if isinstance(evidence.get(name), Mapping)
        }
        if not candidates or batch_size not in candidates:
            raise WriterModelError(
                "UCP profile batch is outside its declared "
                "hardware-friendly candidates"
            )
