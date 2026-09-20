"""Bounded policies for coverage authority and open-ended validation intervals."""
from __future__ import annotations

import math
from typing import Any, Mapping

from ember.writer.errors import WriterModelError


def dynamic_control(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    control = config.get("training_control")
    if control is None:
        return None
    if (control.get("kind") != "validation_early_stopping"
            or control.get("checkpoint_interval") != 25
            or control.get("validation_interval") != 50):
        raise WriterModelError("invalid Source-SFT dynamic training control")
    return control


def checkpoint_declared(contract: Mapping[str, Any], step: int) -> bool:
    control = dynamic_control(contract)
    if control and contract.get("mode") != "profile":
        return step > 0 and step % int(control["checkpoint_interval"]) == 0
    return step in contract.get("runtime", {}).get("checkpoint_steps", ())


def clamped_lr_multiplier(step: int, *, warmup: int, decay: int,
                          peak: float, floor: float) -> float:
    if not 0 <= warmup < decay or not 0 < floor <= peak:
        raise WriterModelError("invalid clamped Source-SFT LR clock or floor")
    if step < warmup:
        return (step + 1) / (warmup + 1)
    phase = min(1.0, max(0.0, (step - warmup) / (decay - warmup)))
    return (floor + (peak - floor) * (1 + math.cos(math.pi * phase)) / 2) / peak


def validate_coverage_manifest(manifest: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    """Validate actual identities and the explicit 24 target + 12 source allowlist."""
    auxiliary = (2, 3, 11, 15, 16, 22, 24, 33, 55, 56, 57, 61)
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    if (protocol.get("version") != "libero_24_8_8_coverage_v1"
            or protocol.get("auxiliary_train") != {
                "suite": "libero_90", "task_ids": list(auxiliary), "global_task_id_offset": 40}):
        raise WriterModelError("Source-SFT coverage protocol or auxiliary allowlist changed")
    expected = {}
    for offset, suite in enumerate(suites):
        roles = protocol["split"]["suites"][suite]
        if {role: len(ids) for role, ids in roles.items()} != {"train": 6, "validation": 2, "test": 2}:
            raise WriterModelError("Source-SFT target suite partition changed")
        for role, ids in roles.items():
            for task in ids:
                key = offset * 10 + int(task)
                if key in expected or not 0 <= task < 10:
                    raise WriterModelError("Source-SFT target partition overlaps")
                expected[key] = (suite, task, role)
    expected.update({40 + task: ("libero_90", task, "train") for task in auxiliary})
    rows = manifest["tasks"]
    actual = {int(row["global_task_id"]): (row["suite"], row["task_id"], row["split_role"]) for row in rows}
    roles = manifest["summary"]["roles"]
    if (len(rows) != 52 or actual != expected
            or any(set(roles[role]) != {task for task, row in expected.items() if row[2] == role}
                   for role in ("train", "validation", "test"))):
        raise WriterModelError("Source-SFT coverage manifest identities differ from protocol")
