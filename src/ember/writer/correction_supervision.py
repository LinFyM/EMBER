"""Training-only source correction targets, compared by actual weight updates."""
from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state


SCHEMA = "native_source_correction_labels_v1"


def validate_correction_config(config):
    expected = {"schema": SCHEMA,
                "label_root": "runs/analysis/native_correction_writer_20260913/correction_labels",
                "weight": .1, "objective": "complete_source_update_relative_frobenius"}
    if config != expected:
        raise ValueError("registered source correction supervision changed")


class CorrectionLabelStore:
    """Data identities resolve labels only here, outside every Writer forward."""

    def __init__(self, asset_root: Path, config, task_ids, demos, contract):
        validate_correction_config(config)
        root = asset_root / config["label_root"]
        registration = json.loads((root / "registration.json").read_text())
        completion = json.loads((root / "completion.json").read_text())
        if (registration["schema"] != SCHEMA or set(registration["tasks"]) != set(task_ids)
                or registration["demos"] != list(demos) or registration["rank"] != contract.rank
                or registration["target_count"] != len(contract.targets)
                or registration["state_contract"] != "state_free"
                or completion["status"] != "complete" or completion["episodes"] != len(task_ids) * len(demos)):
            raise ValueError("incomplete or incompatible source correction labels")
        entries = registration["entries"]
        self.paths = {(row["task"], row["demo"]): asset_root / row["adapter"] for row in entries}
        expected = {(task, demo) for task in task_ids for demo in demos}
        if set(self.paths) != expected or len(entries) != len(expected):
            raise ValueError("source correction labels repeat or omit a training condition")
        self.contract = contract

    def load(self, task, demos, device):
        if len(demos) != 1 or (task, demos[0]) not in self.paths:
            raise ValueError("source correction labels require an authorized K1 training condition")
        target = load_file(str(self.paths[task, demos[0]]), device="cpu")
        validate_lora_state(target, self.contract)
        return {name: value.to(device=device, dtype=torch.float32) for name, value in target.items()}


def _energy(a, b):
    return ((b.transpose(-1, -2) @ b) * (a @ a.transpose(-1, -2))).sum()


def native_update_objective(generated, target, contract):
    """Gauge-invariant ||BA-B*A*||² from small Gram matrices, no dense updates."""
    if set(generated) != set(target):
        raise ValueError("generated and target corrections must cover the same complete LoRA")
    errors, energies = [], []
    first = next(iter(generated.values()))
    with torch.autocast(first.device.type, enabled=False):
        for spec in contract.targets:
            a, b = [generated[spec.name + suffix].float() for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)]
            ta, tb = [target[spec.name + suffix].detach().float() for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)]
            energy = _energy(ta, tb)
            inner = ((b.T @ tb) * (a @ ta.T)).sum()
            errors.append(_energy(a, b) + energy - 2 * inner)
            energies.append(energy)
        energy = torch.stack(energies).sum()
        denominator = torch.where(energy > 0, energy, torch.ones_like(energy))
        value = torch.stack(errors).sum().clamp_min(0) / denominator
    if not torch.isfinite(value):
        raise RuntimeError("source correction objective is nonfinite")
    return value, {"native_update_relative_error": float(value.detach()),
                   "native_update_target_energy": float(energy)}
