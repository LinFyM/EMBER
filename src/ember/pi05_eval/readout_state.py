"""Frozen 37/1 LoRA intervention on the original C correct bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_assets import Pi05EvaluationError


SCHEMA = "ember_readout_realization_masked_states_v1"
GROUPS = ("00", "01", "10", "11")
READOUT = "model.action_out_proj"


def mask_state(state: Mapping[str, torch.Tensor], lora: Any, group: str) -> dict[str, torch.Tensor]:
    """Retain every original A and zero only the B matrices disabled by the group."""
    if group not in GROUPS or len(lora.targets) != 38 or sum(
        target.name == READOUT for target in lora.targets
    ) != 1:
        raise Pi05EvaluationError("readout LoRA group or 38-target topology changed")
    validate_lora_state(state, lora)
    result = dict(state)
    for target in lora.targets:
        enabled = group[1] == "1" if target.name == READOUT else group[0] == "1"
        if not enabled:
            name = target.name + LORA_B_SUFFIX
            result[name] = torch.zeros_like(state[name])
    validate_lora_state(result, lora)
    if any(result[target.name + LORA_A_SUFFIX] is not state[target.name + LORA_A_SUFFIX]
           for target in lora.targets):
        raise Pi05EvaluationError("readout LoRA A changed during masking")
    return result


def derive_states(*, conditions: Mapping[str, Mapping[str, Any]], lora: Any,
                  root: Path, original_manifest: Path) -> dict[str, Any]:
    """Write only 01/10 complete states; 00 and 11 retain source and original assets."""
    root.mkdir(parents=True, exist_ok=False)
    entries: dict[str, Any] = {}
    for condition_id, condition in sorted(conditions.items()):
        original = condition["adapter"]
        source = Path(original["path"])
        if not source.is_file() or source.stat().st_size != int(original["bytes"]):
            raise Pi05EvaluationError(f"original C bank asset changed: {condition_id}")
        state = load_file(str(source), device="cpu")
        validate_lora_state(state, lora)
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()):
            raise Pi05EvaluationError(f"original C bank tensor invalid: {condition_id}")
        outputs = {}
        for group in ("01", "10"):
            derived = mask_state(state, lora, group)
            path = root / f"{condition_id}_{group}.safetensors"
            save_file({name: value.contiguous() for name, value in derived.items()}, str(path))
            outputs[group] = {"path": str(path), "bytes": path.stat().st_size}
        entries[condition_id] = {
            "original": dict(original),
            "global_task_id": int(condition["global_task_id"]),
            "suite": condition["suite"], "task_id": int(condition["task_id"]),
            "teacher_demo_indices": list(condition["teacher_demo_indices"]),
            "derived": outputs,
        }
    manifest = {
        "schema_version": SCHEMA,
        "original_bank_manifest": {"path": str(original_manifest),
                                   "bytes": original_manifest.stat().st_size},
        "readout_target": READOUT, "target_count": len(lora.targets),
        "off_operation": "only disabled LoRA B set to zero; A and scale unchanged",
        "groups": {"00": "original A matrices with every B zero; source function without a copied asset",
                   "01": "only action_out B retained", "10": "only other 37 B retained",
                   "11": "original correct C bank, reused without copy"},
        "conditions": entries,
    }
    with (root / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def load_masked_state(manifest: Mapping[str, Any], *, condition_id: str,
                      group: str, lora: Any) -> dict[str, torch.Tensor]:
    if manifest.get("schema_version") != SCHEMA or group not in ("01", "10"):
        raise Pi05EvaluationError("masked readout bank contract changed")
    record = manifest["conditions"][condition_id]["derived"][group]
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise Pi05EvaluationError("derived readout state asset changed")
    state = load_file(str(path), device="cpu")
    validate_lora_state(state, lora)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
           for value in state.values()):
        raise Pi05EvaluationError("derived readout state invalid")
    return state
