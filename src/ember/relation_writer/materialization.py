"""Compile the learned RGB relation Writer on the resident condition queue."""
from __future__ import annotations

import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ember.operator_writer.data import FormalData
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.materialization_workers import MaterializationWorkers, _configure_device


def model_weights(checkpoint: Path, model: str, *, device="cpu") -> dict:
    """Read one owner from the canonical joint ECP without copying its peer."""
    if model not in ("G", "F"):
        raise ValueError("relation checkpoint owner must be G or F")
    prefix = model + "."
    with safe_open(str(checkpoint / "ecp.safetensors"), framework="pt", device=str(device)) as reader:
        if {key.split(".", 1)[0] for key in reader.keys()} != {"G", "F"}:
            raise ValueError("joint relation checkpoint lost its independent G/F owners")
        return {key.removeprefix(prefix): reader.get_tensor(key)
                for key in reader.keys() if key.startswith(prefix)}


def write_condition(runtime, data, output: Path, condition: dict, shapes: dict, frame_chunk: int):
    from ember.operator_writer.bank import BANK_SCHEMA, _factor_header
    from .readout import MODE

    condition = dict(condition)
    path = output / f"{condition['condition_id']}.safetensors"
    metadata = {"schema_version": BANK_SCHEMA, "condition_id": condition["condition_id"], "mode": MODE}
    started, reused = time.monotonic(), path.exists()
    if reused:
        raw, sampled = data.videos.frame_counts(condition["global_task_id"], condition["teacher_demo"])
        _factor_header(path, shapes, metadata=metadata)
    else:
        pixels, raw, sampled = data.condition(runtime, condition["global_task_id"], condition["teacher_demo"])
        while True:
            try:
                with torch.no_grad():
                    state, _native = runtime.compile(pixels, frame_chunk=frame_chunk)
                break
            except torch.cuda.OutOfMemoryError:
                if frame_chunk <= 8:
                    raise
                frame_chunk = max(8, frame_chunk // 2)
                torch.cuda.empty_cache()
        factors = {name: value.detach().float().cpu().contiguous() for name, value in state.items()}
        if set(factors) != set(shapes):
            raise ValueError("RGB relation compiler must produce every complete A/B factor")
        temporary = path.with_suffix(".safetensors.tmp")
        save_file(factors, str(temporary), metadata=metadata)
        temporary.replace(path)
    condition.update(factors=file_record(path), raw_frames=raw, sampled_frames=sampled)
    return condition, {"condition_id": condition["condition_id"], "reused": reused,
                       "device": str(runtime.device), "pid": os.getpid(), "frame_chunk": frame_chunk,
                       "seconds": time.monotonic() - started,
                       "peak_reserved_bytes": (torch.cuda.max_memory_reserved(runtime.device)
                                               if runtime.device.type == "cuda" else 0)}


class RelationCompiler:
    """One original source and RGB-only G compiler per physical device."""
    def __init__(self, asset_root, config, device, cpu_threads):
        from .runtime import build_runtime

        _configure_device(device, cpu_threads)
        self.runtime = build_runtime(asset_root, config["spec"], device)
        self.asset_root, self.config, self.data, self.request = asset_root, config, None, None

    def prepare(self, request):
        if request == self.request:
            return
        self.close()
        checkpoint, source, _output, _shapes, _frame_chunk = request
        if self.runtime.source != source:
            raise ValueError("G readout source differs from joint training")
        self.runtime.writer.load_state_dict(model_weights(Path(checkpoint), "G",
                                           device=self.runtime.device), strict=True)
        self.runtime.writer.requires_grad_(False).eval()
        self.runtime.policy.eval()
        self.data = FormalData(self.asset_root, self.config["spec"], query_labels=False,
                               task_ids=self.config["task_ids"], role=self.config["role"])
        self.request = request

    def compile(self, condition):
        _checkpoint, _source, output, shapes, frame_chunk = self.request
        return write_condition(self.runtime, self.data, Path(output), condition, shapes, frame_chunk)

    def close(self):
        if self.data is not None:
            self.data.close()
        self.data, self.request = None, None


def compile_conditions(asset_root, spec, checkpoint, source, output, conditions, shapes,
                       *, devices, frame_chunk, task_ids, role, cpu_threads):
    data = FormalData(asset_root, spec, query_labels=False, task_ids=task_ids, role=role)
    completed, statistics, pending = {}, [], []
    try:
        for condition in conditions:
            if (output / f"{condition['condition_id']}.safetensors").exists():
                reader = type("HeaderReader", (), {"device": torch.device("cpu")})()
                value, stats = write_condition(reader, data, output, condition, shapes, frame_chunk)
                completed[value["condition_id"]] = value
                statistics.append(stats)
            else:
                pending.append(condition)
        pending.sort(key=lambda row: data.videos.frame_counts(row["global_task_id"], row["teacher_demo"])[1],
                     reverse=True)
    finally:
        data.close()
    request = (str(checkpoint), source, str(output), shapes, frame_chunk)
    config = {"spec": spec, "task_ids": task_ids, "role": role}
    if pending:
        with MaterializationWorkers(asset_root=asset_root, config=config, devices=devices,
                                    cpu_threads=cpu_threads, compiler_factory=RelationCompiler) as workers:
            for _job, (value, stats) in workers.compile(request, pending):
                if value["condition_id"] in completed:
                    raise ValueError("relation queue returned a duplicate condition")
                completed[value["condition_id"]] = value
                statistics.append(stats)
    conditions[:] = [completed[row["condition_id"]] for row in conditions]
    write_json_atomic(output / "materialization_execution.json", {
        "devices": [str(device) for device in devices], "native_frame_chunk": frame_chunk,
        "cpu_threads_per_worker": cpu_threads, "conditions": statistics})
