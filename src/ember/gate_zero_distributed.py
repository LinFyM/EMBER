"""Topology and provenance authority for 1/2/4-GPU Gate 0 training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ember.gate_zero_contract import load_gate_zero_contract


class GateZeroDistributedError(RuntimeError):
    """Raised when distributed execution changes the frozen scientific budget."""


@dataclass(frozen=True)
class TrainingTopology:
    world_size: int
    global_effective_batch_size: int
    per_rank_micro_batch_size: int
    gradient_accumulation_steps: int
    data_workers_per_rank: int
    total_data_workers: int
    global_slot_algorithm: str
    flow_input_authority: str
    ddp_static_graph: bool

    def as_manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    process_group_initialized: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroDistributedError(f"{label} changed: {actual!r} != {expected!r}")


def topology_for_world_size(
    spec: dict[str, Any], world_size: int
) -> TrainingTopology:
    distributed = spec["distributed"]
    if world_size not in distributed["allowed_world_sizes"]:
        raise GateZeroDistributedError(f"world size {world_size} is not authorized")
    key = f"world_size_{world_size}"
    try:
        selected = distributed[key]
    except KeyError as error:
        raise GateZeroDistributedError(f"missing topology for world size {world_size}") from error
    topology = TrainingTopology(
        world_size=world_size,
        global_effective_batch_size=distributed["global_effective_batch_size"],
        per_rank_micro_batch_size=selected["per_rank_micro_batch_size"],
        gradient_accumulation_steps=selected["gradient_accumulation_steps"],
        data_workers_per_rank=selected["data_workers_per_rank"],
        total_data_workers=distributed["total_data_workers"],
        global_slot_algorithm=distributed["global_slot_algorithm"],
        flow_input_authority=distributed["flow_input_authority"],
        ddp_static_graph=distributed["ddp_static_graph"],
    )
    if (
        topology.per_rank_micro_batch_size
        * topology.gradient_accumulation_steps
        * topology.world_size
        != topology.global_effective_batch_size
    ):
        raise GateZeroDistributedError("topology changes the global effective batch")
    if topology.data_workers_per_rank * topology.world_size != topology.total_data_workers:
        raise GateZeroDistributedError("topology changes the total data-worker budget")
    return topology


def _validate_topology_authority(
    spec: dict[str, Any], gate_zero_path: Path, phase0_path: Path
) -> None:
    authority = spec.get("authority", {})
    _require_equal(
        authority.get("gate_zero_contract_sha256"),
        _sha256(gate_zero_path),
        "Gate 0 contract SHA256",
    )
    _require_equal(
        authority.get("phase0_contract_sha256"),
        _sha256(phase0_path),
        "Phase 0 contract SHA256",
    )
    if authority.get("single_gpu_reference_is_not_interrupted") is not True:
        raise GateZeroDistributedError("single-GPU reference must remain uninterrupted")


def _validate_distributed_settings(
    spec: dict[str, Any], gate_zero: dict[str, Any]
) -> dict[str, Any]:
    distributed = spec.get("distributed", {})
    _require_equal(distributed.get("allowed_world_sizes"), [1, 2, 4], "allowed world sizes")
    _require_equal(distributed.get("maximum_world_size"), 4, "maximum world size")
    _require_equal(distributed.get("backend"), "nccl", "distributed backend")
    _require_equal(
        distributed.get("global_effective_batch_size"),
        gate_zero["base_fit"]["effective_batch_size"],
        "global effective batch",
    )
    _require_equal(
        distributed.get("global_slot_algorithm"),
        "absolute_optimizer_step_accumulation_rank_local_slot_v1",
        "global slot algorithm",
    )
    _require_equal(
        distributed.get("flow_input_authority"),
        "rank0_global_native_sample_then_contiguous_scatter_v1",
        "flow-input authority",
    )
    _require_equal(
        distributed.get("gradient_aggregation"),
        "ddp_mean_equal_local_batch",
        "gradient aggregation",
    )
    _require_equal(distributed.get("ddp_static_graph"), True, "DDP static graph")
    for required in (
        "rank_zero_checkpoint_only",
        "save_every_rank_rng",
        "resume_requires_same_world_size_and_topology",
        "trackio_rank_zero_only",
    ):
        _require_equal(distributed.get(required), True, required)
    _require_equal(
        distributed.get("checkpoint_manifest_schema_version"),
        3,
        "distributed checkpoint schema",
    )
    _require_equal(
        distributed.get("training_file_hash_authority"),
        "rank0_once_per_process_then_shared_filesystem_load_reuse",
        "training file hash authority",
    )
    if distributed.get("minimum_free_memory_mib", 0) < 10240:
        raise GateZeroDistributedError("distributed memory headroom contract weakened")
    for world_size in distributed["allowed_world_sizes"]:
        topology_for_world_size(spec, world_size)
    return distributed


def _validate_probe_contract(probe: dict[str, Any]) -> None:
    _require_equal(probe.get("candidate_world_sizes"), [1, 2, 4], "probe candidates")
    if (
        probe.get("warmup_optimizer_steps", 0)
        + probe.get("measured_optimizer_steps", 0)
        != probe.get("checkpoint_after_optimizer_steps")
    ):
        raise GateZeroDistributedError("probe step accounting changed")
    for required in (
        "same_global_samples_across_candidates",
        "same_initial_trainable_state",
        "same_optimizer_scheduler",
        "same_global_flow_inputs",
        "retain_full_checkpoint_until_resume_validation",
        "cleanup_validated_probe_checkpoints",
    ):
        _require_equal(probe.get(required), True, f"probe {required}")
    _require_equal(probe.get("record_policy_outcomes"), False, "probe outcome ban")


def _validate_selection_authority(
    selection_authority: dict[str, Any], allowed_world_sizes: list[int]
) -> None:
    status = selection_authority.get("status")
    if status not in {"pending_fixed_1_2_4_probe", "frozen_topology_selection"}:
        raise GateZeroDistributedError("unknown topology selection authority")
    if status == "pending_fixed_1_2_4_probe":
        _require_equal(
            selection_authority.get("formal_distributed_fit_authorized"),
            False,
            "pending distributed-fit authorization",
        )
        _require_equal(selection_authority.get("selected_world_size"), 0, "pending world size")
        _require_equal(selection_authority.get("selection_report_sha256"), "", "pending report")
    else:
        _require_equal(
            selection_authority.get("formal_distributed_fit_authorized"),
            True,
            "distributed-fit authorization",
        )
        if selection_authority.get("selected_world_size") not in allowed_world_sizes:
            raise GateZeroDistributedError("selected world size is not an allowed topology")
        report_sha = selection_authority.get("selection_report_sha256")
        if not isinstance(report_sha, str) or len(report_sha) != 64:
            raise GateZeroDistributedError("invalid topology selection report SHA256")
    _require_equal(
        selection_authority.get("gate_zero_authorized"), False, "topology Gate 0 authority"
    )
    _require_equal(
        selection_authority.get("writer_authorized"), False, "topology Writer authority"
    )


def _validate_selection_contract(
    selection: dict[str, Any], allowed_world_sizes: list[int]
) -> None:
    _require_equal(
        selection.get("require_zero_duplicate_or_missing_global_slots"),
        True,
        "global-slot completeness",
    )
    _require_equal(
        selection.get("require_same_topology_resume_pass"),
        True,
        "same-topology resume requirement",
    )
    if selection.get("require_minimum_free_memory_mib", 0) < 10240:
        raise GateZeroDistributedError("selection memory headroom weakened")
    _require_equal(
        selection.get("scientific_thresholds_may_change"),
        False,
        "scientific thresholds",
    )
    _require_equal(
        selection.get("model_or_data_budget_may_change"),
        False,
        "model or data budget",
    )
    _validate_selection_authority(selection.get("authority", {}), allowed_world_sizes)


def validate_distributed_topology_spec(
    spec: dict[str, Any], gate_zero_path: Path, phase0_path: Path
) -> None:
    if spec.get("schema_version") != 1:
        raise GateZeroDistributedError("unsupported distributed topology schema")
    if spec.get("status") != "predeclared_before_multigpu_probe_outcomes":
        raise GateZeroDistributedError("distributed topology is not predeclared")
    gate_zero = load_gate_zero_contract(gate_zero_path, phase0_path)
    _validate_topology_authority(spec, gate_zero_path, phase0_path)
    distributed = _validate_distributed_settings(spec, gate_zero)
    _validate_probe_contract(spec.get("probe", {}))
    _validate_selection_contract(
        spec.get("selection", {}), distributed["allowed_world_sizes"]
    )
    resources = spec.get("resources", {})
    if resources.get("maximum_concurrent_gpus") != 4:
        raise GateZeroDistributedError("resource contract must retain the four-GPU ceiling")


def load_distributed_topology_spec(
    path: Path, gate_zero_path: Path, phase0_path: Path
) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroDistributedError("invalid distributed topology TOML") from error
    validate_distributed_topology_spec(spec, gate_zero_path, phase0_path)
    return spec


def require_topology_mode_authorization(
    spec: dict[str, Any], *, mode: str, world_size: int
) -> None:
    topology_for_world_size(spec, world_size)
    if mode in {"resume-probe", "topology-probe"}:
        return
    if mode != "train":
        raise GateZeroDistributedError(f"unknown topology mode: {mode}")
    authority = spec["selection"]["authority"]
    if authority["status"] != "frozen_topology_selection":
        raise GateZeroDistributedError("formal distributed fit awaits the fixed 1/2/4 probe")
    if authority["formal_distributed_fit_authorized"] is not True:
        raise GateZeroDistributedError("formal distributed fit is not authorized")
    if authority["selected_world_size"] != world_size:
        raise GateZeroDistributedError("torchrun world size differs from the selected topology")


def global_effective_slots(
    topology: TrainingTopology, *, rank: int, accumulation_step: int
) -> list[int]:
    if not 0 <= rank < topology.world_size:
        raise GateZeroDistributedError("rank is outside the topology")
    if not 0 <= accumulation_step < topology.gradient_accumulation_steps:
        raise GateZeroDistributedError("accumulation step is outside the topology")
    width = topology.per_rank_micro_batch_size
    start = accumulation_step * topology.world_size * width + rank * width
    slots = list(range(start, start + width))
    if slots[-1] >= topology.global_effective_batch_size:
        raise GateZeroDistributedError("rank shard escaped the global effective batch")
    return slots


def merge_rank_provenance(
    topology: TrainingTopology, per_rank_keys: Sequence[Sequence[str]]
) -> dict[str, Any]:
    if len(per_rank_keys) != topology.world_size:
        raise GateZeroDistributedError("provenance rank count changed")
    expected_local = (
        topology.per_rank_micro_batch_size * topology.gradient_accumulation_steps
    )
    if any(len(keys) != expected_local for keys in per_rank_keys):
        raise GateZeroDistributedError("provenance local slot count changed")
    ordered: list[str] = []
    slots: list[int] = []
    width = topology.per_rank_micro_batch_size
    for accumulation_step in range(topology.gradient_accumulation_steps):
        for rank in range(topology.world_size):
            begin = accumulation_step * width
            end = begin + width
            ordered.extend(per_rank_keys[rank][begin:end])
            slots.extend(
                global_effective_slots(
                    topology, rank=rank, accumulation_step=accumulation_step
                )
            )
    if slots != list(range(topology.global_effective_batch_size)):
        raise GateZeroDistributedError("global provenance has duplicate or missing slots")
    digest = hashlib.sha256()
    for key in ordered:
        if not isinstance(key, str):
            raise GateZeroDistributedError("provenance keys must be strings")
        digest.update(key.encode("utf-8") + b"\0")
    return {
        "keys": ordered,
        "sha256": digest.hexdigest(),
        "global_slot_count": len(slots),
        "unique_global_slot_count": len(set(slots)),
    }


def assert_same_topology(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if expected != actual:
        raise GateZeroDistributedError(
            "resume topology changed: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _index_topology_results(
    results: Sequence[dict[str, Any]], expected_worlds: list[int]
) -> dict[int, dict[str, Any]]:
    by_world: dict[int, dict[str, Any]] = {}
    for result in results:
        world_size = result.get("topology", {}).get("world_size")
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise GateZeroDistributedError("topology probe world size is invalid")
        if world_size in by_world:
            raise GateZeroDistributedError("duplicate topology probe result")
        by_world[world_size] = result
    if sorted(by_world) != expected_worlds:
        raise GateZeroDistributedError("topology probe results do not cover 1/2/4 GPUs")
    return by_world


def _validate_cross_topology_authority(
    authority: dict[str, Any], reference: dict[str, Any]
) -> None:
    for key in (
        "initial_model_state_sha256",
        "initial_optimizer_state_sha256",
        "initial_scheduler_state_sha256",
        "row_keys_sha256_by_step",
        "flow_input_sha256_by_step",
        "global_slot_count_by_step",
        "unique_global_slot_count_by_step",
    ):
        if authority.get(key) != reference.get(key):
            raise GateZeroDistributedError(
                f"cross-topology flow/data authority changed: {key}"
            )
    slot_counts = authority["global_slot_count_by_step"]
    unique_counts = authority["unique_global_slot_count_by_step"]
    if not slot_counts or any(value != 64 for value in slot_counts):
        raise GateZeroDistributedError("topology has duplicate or missing global slots")
    if unique_counts != slot_counts:
        raise GateZeroDistributedError("topology has duplicate or missing global slots")


def _candidate_from_probe(
    spec: dict[str, Any],
    *,
    world_size: int,
    result: dict[str, Any],
    reference_authority: dict[str, Any],
) -> dict[str, Any]:
    if result.get("status") != "topology_probe_completed_pending_cross_topology_selection":
        raise GateZeroDistributedError(f"topology {world_size} probe is incomplete")
    if result.get("same_topology_resume", {}).get("all_exact") is not True:
        raise GateZeroDistributedError(f"topology {world_size} resume is not exact")
    authority = result.get("global_authority", {})
    if not isinstance(authority, dict):
        raise GateZeroDistributedError(f"topology {world_size} authority is missing")
    _validate_cross_topology_authority(authority, reference_authority)
    measurement = result.get("measurement", {})
    throughput = measurement.get("global_effective_samples_per_second")
    free_mib = measurement.get("minimum_free_memory_mib")
    if not isinstance(throughput, (int, float)) or isinstance(throughput, bool):
        raise GateZeroDistributedError("topology measurement is invalid")
    if not math.isfinite(throughput) or throughput <= 0:
        raise GateZeroDistributedError("topology measurement is invalid")
    if not isinstance(free_mib, int) or isinstance(free_mib, bool) or free_mib < 0:
        raise GateZeroDistributedError("topology measurement is invalid")
    minimum_free = spec["selection"]["require_minimum_free_memory_mib"]
    return {
        "world_size": world_size,
        "global_effective_samples_per_second": float(throughput),
        "minimum_free_memory_mib": free_mib,
        "headroom_safe": free_mib >= minimum_free,
    }


def _score_topology_candidates(
    spec: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_throughput = candidates["1"]["global_effective_samples_per_second"]
    if not candidates["1"]["headroom_safe"]:
        raise GateZeroDistributedError("single-GPU topology lacks a safe reference")
    eligible: list[dict[str, Any]] = []
    for world_size in spec["probe"]["candidate_world_sizes"]:
        row = candidates[str(world_size)]
        speedup = row["global_effective_samples_per_second"] / baseline_throughput
        efficiency = speedup / world_size
        row["speedup_over_single_gpu"] = speedup
        row["parallel_efficiency"] = efficiency
        if world_size == 1:
            row["eligible_for_single_job"] = row["headroom_safe"]
        else:
            row["eligible_for_single_job"] = (
                row["headroom_safe"]
                and speedup >= spec["selection"]["minimum_speedup_over_single_gpu"]
                and efficiency >= spec["selection"]["minimum_parallel_efficiency"]
            )
        if row["eligible_for_single_job"]:
            eligible.append(row)
    return eligible


def _choose_topology(eligible: list[dict[str, Any]]) -> dict[str, Any]:
    fastest = max(row["global_effective_samples_per_second"] for row in eligible)
    tie_floor = fastest * 0.98
    return min(
        (
            row
            for row in eligible
            if row["global_effective_samples_per_second"] >= tie_floor
        ),
        key=lambda row: row["world_size"],
    )


def select_topology_candidates(
    spec: dict[str, Any], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen efficiency rule after all matched topology probes finish."""

    expected_worlds = spec["probe"]["candidate_world_sizes"]
    by_world = _index_topology_results(results, expected_worlds)
    reference_authority = by_world[1].get("global_authority")
    if not isinstance(reference_authority, dict):
        raise GateZeroDistributedError("single-GPU topology authority is missing")
    candidates = {
        str(world_size): _candidate_from_probe(
            spec,
            world_size=world_size,
            result=by_world[world_size],
            reference_authority=reference_authority,
        )
        for world_size in expected_worlds
    }
    selected = _choose_topology(_score_topology_candidates(spec, candidates))
    selected_world_size = selected["world_size"]
    execution_mode = "ddp" if selected_world_size > 1 else "independent_job_parallelism"
    return {
        "status": "topology_selection_frozen_candidate",
        "execution_mode": execution_mode,
        "selected_world_size": selected_world_size,
        "selected_global_effective_samples_per_second": selected[
            "global_effective_samples_per_second"
        ],
        "candidates": candidates,
        "maximum_concurrent_independent_jobs": 4 // selected_world_size,
        "unused_gpu_capacity_requires_independent_work": 4 % selected_world_size,
        "gate_zero_authorized": False,
        "writer_authorized": False,
    }


def initialize_distributed_context(
    spec: dict[str, Any], *, backend: str | None = None
) -> DistributedContext:
    """Initialize one canonical env:// process group when torchrun selected >1 rank."""

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise GateZeroDistributedError("invalid torchrun rank environment") from error
    topology_for_world_size(spec, world_size)
    if not 0 <= rank < world_size or not 0 <= local_rank < world_size:
        raise GateZeroDistributedError("torchrun rank is outside the selected world")
    selected_backend = backend or spec["distributed"]["backend"]
    initialized_here = False
    if world_size > 1:
        if selected_backend == "nccl":
            if not torch.cuda.is_available():
                raise GateZeroDistributedError("NCCL topology requires CUDA")
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_available():
            raise GateZeroDistributedError("torch.distributed is unavailable")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=selected_backend, init_method="env://")
            initialized_here = True
        if torch.distributed.get_world_size() != world_size or torch.distributed.get_rank() != rank:
            raise GateZeroDistributedError("initialized process group differs from torchrun")
    elif selected_backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=selected_backend if world_size > 1 else None,
        process_group_initialized=initialized_here,
    )


def close_distributed_context(context: DistributedContext) -> None:
    if (
        context.process_group_initialized
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        torch.distributed.barrier()


def gather_rank_rng_states(
    context: DistributedContext,
) -> list[dict[str, torch.Tensor]] | None:
    from lerobot.utils.random_utils import serialize_rng_state

    local = serialize_rng_state()
    if context.world_size == 1:
        return [local]
    gathered: list[dict[str, torch.Tensor] | None] | None = (
        [None] * context.world_size if context.is_primary else None
    )
    torch.distributed.gather_object(local, gathered, dst=0)
    if not context.is_primary:
        return None
    if gathered is None or any(state is None for state in gathered):
        raise GateZeroDistributedError("failed to gather every-rank RNG state")
    return [state for state in gathered if state is not None]


def gather_rank_objects(value: Any, context: DistributedContext) -> list[Any] | None:
    if context.world_size == 1:
        return [value]
    gathered = [None] * context.world_size if context.is_primary else None
    torch.distributed.gather_object(value, gathered, dst=0)
    return gathered


def broadcast_primary_error(
    context: DistributedContext, error: BaseException | str | None
) -> None:
    message = None if error is None else str(error)
    if context.world_size == 1:
        if message is not None:
            raise GateZeroDistributedError(message)
        return
    payload = [message if context.is_primary else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise GateZeroDistributedError(f"rank-0 operation failed: {payload[0]}")


def broadcast_primary_object(context: DistributedContext, value: Any) -> Any:
    if context.world_size == 1:
        return value
    payload = [value if context.is_primary else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def distributed_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.world_size > 1:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        result.div_(context.world_size)
    return result


def distributed_max(value: float, context: DistributedContext, *, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    if context.world_size > 1:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return float(result)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("utf-8") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("utf-8") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def primary_rng_state_sha256(context: DistributedContext) -> str:
    value = None
    if context.is_primary:
        from lerobot.utils.random_utils import serialize_rng_state

        value = tensor_state_sha256(serialize_rng_state())
    result = broadcast_primary_object(context, value)
    if not isinstance(result, str) or len(result) != 64:
        raise GateZeroDistributedError("rank-0 RNG authority is invalid")
    return result


def unwrap_distributed_model(model: torch.nn.Module) -> torch.nn.Module:
    from torch.nn.parallel import DistributedDataParallel

    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_distributed_model(
    model: torch.nn.Module,
    context: DistributedContext,
    *,
    static_graph: bool,
) -> torch.nn.Module:
    if context.world_size == 1:
        return model
    from torch.nn.parallel import DistributedDataParallel

    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "gradient_as_bucket_view": True,
        "static_graph": static_graph,
    }
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None and first_parameter.is_cuda:
        kwargs["device_ids"] = [context.local_rank]
        kwargs["output_device"] = context.local_rank
    return DistributedDataParallel(model, **kwargs)


def native_global_flow_inputs(
    policy: torch.nn.Module,
    topology: TrainingTopology,
    context: DistributedContext,
    *,
    action_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Sample once in single-GPU order on rank 0, then scatter contiguous shards."""

    if context.world_size != topology.world_size:
        raise GateZeroDistributedError("flow-input context differs from topology")
    chunk_size, action_dim = action_shape
    if chunk_size <= 0 or action_dim <= 0:
        raise GateZeroDistributedError("invalid flow-input action shape")
    owner = unwrap_distributed_model(policy)
    model = getattr(owner, "model", None)
    if model is None or not callable(getattr(model, "sample_noise", None)) or not callable(
        getattr(model, "sample_time", None)
    ):
        raise GateZeroDistributedError("policy lacks native flow samplers")
    global_noise: torch.Tensor | None = None
    global_time: torch.Tensor | None = None
    flow_input_sha256: str | None = None
    if context.is_primary:
        global_noise = model.sample_noise(
            (topology.global_effective_batch_size, chunk_size, action_dim), device
        )
        global_time = model.sample_time(topology.global_effective_batch_size, device)
        if global_noise.shape != (
            topology.global_effective_batch_size,
            chunk_size,
            action_dim,
        ) or global_time.shape != (topology.global_effective_batch_size,):
            raise GateZeroDistributedError("native flow sampler changed shape")
        flow_input_sha256 = tensor_state_sha256(
            {"noise": global_noise, "time": global_time}
        )
    flow_input_sha256 = broadcast_primary_object(context, flow_input_sha256)
    if not isinstance(flow_input_sha256, str) or len(flow_input_sha256) != 64:
        raise GateZeroDistributedError("global flow-input digest is invalid")
    local_size = topology.per_rank_micro_batch_size
    if topology.gradient_accumulation_steps != 1:
        raise GateZeroDistributedError("native flow scatter currently requires one accumulation step")
    if context.world_size == 1:
        if global_noise is None or global_time is None:
            raise GateZeroDistributedError("primary flow inputs are missing")
        return global_noise, global_time, flow_input_sha256
    local_noise = torch.empty(
        (local_size, chunk_size, action_dim), dtype=torch.float32, device=device
    )
    local_time = torch.empty((local_size,), dtype=torch.float32, device=device)
    noise_chunks = list(global_noise.split(local_size)) if context.is_primary else None
    time_chunks = list(global_time.split(local_size)) if context.is_primary else None
    torch.distributed.scatter(local_noise, scatter_list=noise_chunks, src=0)
    torch.distributed.scatter(local_time, scatter_list=time_chunks, src=0)
    return local_noise, local_time, flow_input_sha256
