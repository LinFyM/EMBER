"""Complete scheduler-free native-control ECPs and explicit physical migrations."""

from __future__ import annotations

import os
import random
import socket
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.ecp.checkpoint import checkpoint_macro
from ember.pi05_source_checkpoint import (
    DistributedContext, barrier, capture_rng, read_json, restore_rng,
    source_reference_matches, write_json_atomic,
)

from .specification import CHECKPOINTS, MT_WEIGHTS, OPTIMIZATION, SCHEMA, SMOKE_CHECKPOINTS, STUDY_ID


CHECKPOINT_SCHEMA = "ember_native_video_control_checkpoint_v1"


def _validate_origin(origin: Mapping[str, Any], arm: str, macro: int) -> None:
    kind = origin.get("kind")
    allowed = CHECKPOINTS if kind == "formal" else SMOKE_CHECKPOINTS if kind == "smoke" else ()
    lora = origin.get("lora", {})
    if (arm not in ("M", "V") or type(macro) is not int or macro not in allowed
            or (kind == "smoke" and arm != "V") or origin.get("fresh_optimizer") is not True
            or str(origin.get("mt_checkpoint")) != str(MT_WEIGHTS)
            or not isinstance(origin.get("source"), Mapping) or not origin["source"]
            or not isinstance(origin.get("training_git"), Mapping)
            or not origin["training_git"].get("commit")
            or lora.get("target_count") != 38 or lora.get("state_tensor_count") != 76
            or lora.get("adapter", {}).get("rank") != 128
            or lora.get("adapter", {}).get("alpha") != 128):
        raise ValueError("native ECP origin, complete LoRA or registered boundary changed")


def _topology(context: DistributedContext, supplied: Mapping[str, Any] | None) -> dict:
    if supplied is None:
        if context.world_size != 1:
            raise ValueError("multi-rank native ECP needs its complete physical topology")
        supplied = {"host": socket.gethostname(), "world_size": 1,
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "ranks": [{"rank": 0, "local_rank": context.local_rank,
                               "device": str(context.device), "numa_node": context.numa_node,
                               "cpu_affinity": list(context.cpu_affinity or ())}]}
    result = dict(supplied)
    if result.get("world_size") != context.world_size:
        raise ValueError("native ECP topology differs from its current rank count")
    return result


def _capture_rank_rng(context: DistributedContext) -> dict:
    if context.device.type == "cuda":
        return capture_rng(context)
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": None}


def _restore_rank_rng(state: dict, context: DistributedContext) -> None:
    if context.device.type == "cuda":
        restore_rng(state, context)
    else:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])


def _validate_model_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer, macro: int) -> None:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    packed = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
    if (not isinstance(optimizer, torch.optim.AdamW)
            or len(packed) != len(parameters) or {id(value) for value in packed} != {id(value) for value in parameters}
            or any(value.dtype != torch.float32 for value in parameters)
            or any(value.is_floating_point() and value.dtype != torch.float32 for value in model.state_dict().values())):
        raise ValueError("native ECP requires exactly the trainable FP32 model and fresh AdamW")
    for group in optimizer.param_groups:
        if (group["lr"] != OPTIMIZATION["lr"] or tuple(group["betas"]) != tuple(OPTIMIZATION["betas"])
                or group["eps"] != OPTIMIZATION["eps"] or group["weight_decay"] != OPTIMIZATION["weight_decay"]):
            raise ValueError("native AdamW hyperparameters changed")
    if macro == 0:
        if optimizer.state:
            raise ValueError("native initial ECP contains non-fresh optimizer state")
        return
    if set(optimizer.state) != set(parameters):
        raise ValueError("native ECP lacks optimizer state for a learned parameter")
    for parameter in parameters:
        _validate_optimizer_state(parameter, optimizer.state[parameter], macro)


def _validate_optimizer_state(parameter, state, macro):
    if (not {"step", "exp_avg", "exp_avg_sq"} <= state.keys() or int(state["step"].item()) != macro
            or state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape
            or any(value.is_floating_point() and value.dtype != torch.float32
                   for value in state.values() if isinstance(value, torch.Tensor))):
        raise ValueError("native ECP optimizer dtype, shape or update cursor changed")


def save(*, output_dir: Path, macro: int, stage: str, context: DistributedContext,
         model: torch.nn.Module, optimizer: torch.optim.Optimizer, data, origin: Mapping[str, Any],
         metrics_rows: int, topology: Mapping[str, Any] | None = None,
         training_state: Mapping[str, Any] | None = None) -> Path:
    """Atomically publish one model, all AdamW states and every rank RNG."""
    _validate_origin(origin, stage, macro)
    _validate_model_optimizer(model, optimizer, macro)
    physical = _topology(context, topology)
    sampler = data.sampler_state()
    if sampler["next_step"] != macro or metrics_rows != macro:
        raise ValueError("native checkpoint is outside its complete macro/metrics boundary")
    checkpoints = Path(output_dir) / "checkpoints"
    partial, final = checkpoints / f".macro_{macro:08d}.partial", checkpoints / f"macro_{macro:08d}"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if partial.exists() or final.exists():
            raise ValueError(f"native checkpoint already exists: {final}")
        partial.mkdir()
    barrier(context)
    torch.save({"schema_version": CHECKPOINT_SCHEMA, "arm": stage, "rank": context.rank,
                "world_size": context.world_size, "next_macro": macro, "rng": _capture_rank_rng(context)},
               partial / f"rank_{context.rank:02d}_state.pt")
    barrier(context)
    if context.is_main:
        _publish(partial, final, output_dir=Path(output_dir), macro=macro, arm=stage,
                 model=model, optimizer=optimizer, sampler=sampler, origin=origin,
                 metrics_rows=metrics_rows, topology=physical, training_state=training_state)
    barrier(context)
    return final


def _publish(partial: Path, final: Path, *, output_dir: Path, macro: int, arm: str,
             model, optimizer, sampler: dict, origin: Mapping, metrics_rows: int, topology: dict,
             training_state: Mapping | None) -> None:
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              str(partial / "ecp.safetensors"))
    torch.save({"schema_version": CHECKPOINT_SCHEMA, "arm": arm, "next_macro": macro,
                "optimizer": optimizer.state_dict(), "scheduler": None, "scaler": None,
                "metrics_rows": metrics_rows, "sampler_state": sampler, "origin": dict(origin),
                "topology": topology, "training_state": dict(training_state or {})},
               partial / "trainer_state.pt")
    files = {path.name: {"bytes": path.stat().st_size} for path in sorted(partial.iterdir()) if path.is_file()}
    write_json_atomic(partial / "checkpoint_manifest.json", {
        "schema_version": CHECKPOINT_SCHEMA, "study_id": STUDY_ID, "run_contract_schema": SCHEMA,
        "stage": arm, "arm": arm, "next_macro": macro, "step": macro,
        "world_size": topology["world_size"], "topology": topology,
        "fullmodel_FP32": True, "optimizer_dtype": "float32", "scheduler": None, "scaler": None,
        "origin": dict(origin), "source": origin["source"], "MT300_origin": origin["mt_checkpoint"],
        "training_git": origin["training_git"], "lora": origin["lora"],
        "model_file": "ecp.safetensors", "beta_tensor_selector": "common.values.*",
        "controller_tensor_selector": "all", "files": files,
    })
    os.replace(partial, final)
    write_json_atomic(output_dir / "latest_checkpoint.json", {"path": str(final), "macro": macro, "arm": arm})


def inspect_checkpoint(checkpoint: Path, *, arm: str | None = None, terminal: bool = False) -> dict:
    """Inspect only this complete ECP's metadata and files, without model loading."""
    checkpoint = Path(checkpoint)
    macro = checkpoint_macro(checkpoint)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    stage = manifest.get("arm")
    world = int(manifest.get("world_size", -1))
    expected_files = {"ecp.safetensors", "trainer_state.pt", *(f"rank_{rank:02d}_state.pt" for rank in range(world))}
    expected = {"schema_version": CHECKPOINT_SCHEMA, "study_id": STUDY_ID, "run_contract_schema": SCHEMA,
                "stage": stage, "step": macro, "next_macro": macro, "fullmodel_FP32": True,
                "optimizer_dtype": "float32", "scheduler": None, "scaler": None}
    if (any(manifest.get(key) != value for key, value in expected.items())
            or arm is not None and stage != arm or world < 1
            or manifest.get("topology", {}).get("world_size") != world
            or set(manifest.get("files", {})) != expected_files):
        raise ValueError("native checkpoint authority or completeness changed")
    origin = manifest.get("origin", {})
    _validate_origin(origin, stage, macro)
    if (manifest.get("source") != origin["source"] or manifest.get("MT300_origin") != origin["mt_checkpoint"]
            or manifest.get("training_git") != origin["training_git"] or manifest.get("lora") != origin["lora"]
            or (terminal and (macro != 128 or origin["kind"] != "formal"))):
        raise ValueError("native checkpoint source or scientific endpoint changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"native checkpoint file changed: {name}")
    return manifest


def _load_trainer(checkpoint: Path, manifest: dict, expected_origin: Mapping, data) -> dict:
    macro, arm = manifest["next_macro"], manifest["arm"]
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
    if (trainer.get("schema_version") != CHECKPOINT_SCHEMA or trainer.get("arm") != arm
            or trainer.get("next_macro") != macro or trainer.get("metrics_rows") != macro
            or trainer.get("scheduler") is not None or trainer.get("scaler") is not None
            or trainer.get("origin") != manifest["origin"] or trainer.get("topology") != manifest["topology"]):
        raise ValueError("native trainer state or complete macro cursor changed")
    actual = trainer["origin"]
    _validate_origin(expected_origin, arm, macro)
    if (any(actual.get(key) != expected_origin.get(key) for key in ("kind", "mt_checkpoint", "fresh_optimizer", "lora"))
            or not source_reference_matches(actual["source"], expected_origin["source"])):
        raise ValueError("native resume changed its model/data origin")
    sampler = trainer.get("sampler_state", {})
    if sampler.get("next_step") != macro:
        raise ValueError("native sampler is not at the checkpoint macro")
    data.restore(sampler)
    return trainer


def restore(*, checkpoint: Path, stage: str, context: DistributedContext, model: torch.nn.Module,
            optimizer: torch.optim.Optimizer, data, origin: Mapping[str, Any],
            topology: Mapping[str, Any] | None = None, allow_topology_change: bool = False,
            restored_state: dict | None = None) -> tuple[int, int]:
    """Restore logical state; record physical migration and new-rank RNG sources."""
    checkpoint = Path(checkpoint)
    manifest = inspect_checkpoint(checkpoint, arm=stage)
    current_topology = _topology(context, topology)
    changed = manifest["topology"] != current_topology
    if changed and not allow_topology_change:
        raise ValueError("native ECP physical topology changed without explicit migration")
    trainer = _load_trainer(checkpoint, manifest, origin, data)
    ranks = _rank_states(checkpoint, manifest)
    model.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(context.device)), strict=True)
    optimizer.load_state_dict(trainer["optimizer"])
    macro = manifest["next_macro"]
    _validate_model_optimizer(model, optimizer, macro)
    old_world = manifest["world_size"]
    if context.rank < old_world:
        _restore_rank_rng(ranks[context.rank]["rng"], context)
    else:
        from ember.pi05_source_setup import seed_everything
        seed_everything(7, context)
    if restored_state is not None:
        restored_state.update(sampler_state=trainer["sampler_state"], training_state=trainer["training_state"],
                              checkpoint_origin=trainer["origin"], current_training_git=origin["training_git"])
        if changed:
            restored_state["topology_resume"] = {
                "old_topology": manifest["topology"], "new_topology": current_topology,
                "restored_rank_rng": list(range(min(old_world, context.world_size))),
                "fresh_rank_rng": {str(rank): {"source": "seed_everything(7, context)", "seed": 7 + rank}
                                   for rank in range(old_world, context.world_size)},
                "logical_sampler_unchanged": True, "bitwise_exact_claim": False,
            }
    return macro, trainer["metrics_rows"]


def _rank_states(checkpoint: Path, manifest: dict) -> list[dict]:
    states = [torch.load(checkpoint / f"rank_{rank:02d}_state.pt", map_location="cpu", weights_only=False)
              for rank in range(manifest["world_size"])]
    for rank, state in enumerate(states):
        if (state.get("schema_version") != CHECKPOINT_SCHEMA or state.get("arm") != manifest["arm"]
                or state.get("rank") != rank or state.get("world_size") != manifest["world_size"]
                or state.get("next_macro") != manifest["next_macro"]):
            raise ValueError("native ECP rank RNG identity or cursor changed")
    return states
