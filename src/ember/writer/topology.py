"""Runtime topology helpers shared by Writer training and validation."""

from __future__ import annotations

import os
from pathlib import Path

import torch


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


def bind_current_process_to_cuda_numa(device: int) -> tuple[int, ...] | None:
    """Constrain this process and its future children to the GPU-local NUMA node."""

    properties = torch.cuda.get_device_properties(device)
    pci_address = (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )
    numa_path = Path("/sys/bus/pci/devices") / pci_address / "numa_node"
    try:
        numa_node = int(numa_path.read_text(encoding="utf-8").strip())
        if numa_node < 0:
            return None
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
