"""Shared model/data/optimizer session for every Gate 0 source-base trainer mode."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ember.eval_artifacts import update_latest_link
from ember.gate_zero_base_runtime import (
    build_base_optimizer,
    build_base_scheduler,
    load_base_training_components,
    make_base_loader,
    optimizer_step,
    validate_base_training_files_authority,
)
from ember.gate_zero_checkpoint import (
    CHECKPOINT_MANIFEST,
    restore_source_base_checkpoint_rng,
    save_source_base_checkpoint,
    validate_source_base_checkpoint,
)
from ember.gate_zero_distributed import (
    DistributedContext,
    TrainingTopology,
    broadcast_primary_error,
    broadcast_primary_object,
    distributed_barrier,
    gather_rank_objects,
    gather_rank_rng_states,
    unwrap_distributed_model,
    wrap_distributed_model,
)
from ember.gate_zero_runtime import batch_provenance_keys, set_global_seed, sha256_file


class GateZeroBaseTrainError(RuntimeError):
    """Raised when source-base training or resume mechanics drift."""


_VERIFIED_TRAINING_FILE_AUTHORITIES: set[tuple[str, str, str, str, str]] = set()


def require_base_fit_authorization(spec: dict[str, Any], *, mode: str) -> None:
    if mode not in {"resume-probe", "topology-probe", "train"}:
        raise GateZeroBaseTrainError(f"unknown source-base mode: {mode}")
    selection = spec["base_fit"]["batch_calibration"]["selection_authority"]
    if selection["status"] != "frozen_matched_resource_authority":
        raise GateZeroBaseTrainError("matched batch-shape authority is not frozen")
    if selection["authorized_as_batch_shape"] is not True:
        raise GateZeroBaseTrainError("batch shape is not authorized")
    if mode == "train" and selection["formal_base_fit_authorized"] is not True:
        raise GateZeroBaseTrainError("formal base fit is not authorized before resume identity")


def should_log_training_step(step: int, *, target_step: int, every: int) -> bool:
    if step <= 0 or target_step < step or every <= 0:
        raise GateZeroBaseTrainError("invalid training log cadence")
    return step == 1 or step == target_step or step % every == 0


def initialize_tracking(
    spec: dict[str, Any],
    args: argparse.Namespace,
    context: DistributedContext,
    topology: TrainingTopology,
) -> Any:
    if not context.is_primary:
        return None
    import trackio

    trackio.init(
        project=spec["tracking"]["project"],
        name=args.output_dir.name,
        group={
            "resume-probe": "base_resume_probe",
            "topology-probe": "base_topology_probe",
            "train": "source_base_fit",
        }[args.mode],
        config={
            "mode": args.mode,
            "effective_batch_size": spec["base_fit"]["effective_batch_size"],
            "world_size": topology.world_size,
            "per_rank_micro_batch_size": topology.per_rank_micro_batch_size,
            "model_revision": spec["authority"]["model_revision"],
        },
        auto_log_gpu=spec["tracking"]["log_system_metrics"],
        gpu_log_interval=1.0,
        auto_log_cpu=spec["tracking"]["log_system_metrics"],
        cpu_log_interval=1.0,
    )
    return trackio


def validate_output(
    args: argparse.Namespace, *, result_name: str, context: DistributedContext
) -> None:
    primary_error: BaseException | None = None
    if context.is_primary:
        try:
            if not args.output_dir.is_absolute() or not args.latest_link.is_absolute():
                raise GateZeroBaseTrainError("output and latest paths must be absolute")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            if (args.output_dir / result_name).exists():
                raise GateZeroBaseTrainError("refusing to overwrite completed source-base result")
        except BaseException as error:
            primary_error = error
    broadcast_primary_error(context, primary_error)
    distributed_barrier(context)


def write_result(
    result: dict[str, Any],
    args: argparse.Namespace,
    *,
    result_name: str,
    context: DistributedContext,
) -> None:
    if not context.is_primary:
        raise GateZeroBaseTrainError("only rank 0 may publish a scientific result")
    result_path = args.output_dir / result_name
    temporary = args.output_dir / f".{result_name}.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, result_path)
    checksum = f"{sha256_file(result_path)}  {result_name}\n"
    (args.output_dir / "checksums.sha256").write_text(checksum, encoding="utf-8")
    update_latest_link(args.output_dir, args.latest_link)


def checkpoint_evidence(checkpoint_dir: Path) -> dict[str, Any]:
    manifest = validate_source_base_checkpoint(checkpoint_dir)
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST
    return {
        "step": manifest["step"],
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(manifest["files"]) + 1,
        "payload_bytes": sum(item["bytes"] for item in manifest["files"].values())
        + manifest_path.stat().st_size,
        "payload_file_manifest_sha256": canonical_state_sha256(manifest["files"]),
        "checkpoint_role": manifest["checkpoint_role"],
        "schema_version": manifest["schema_version"],
        "topology": manifest["topology"],
        "distributed_rng": manifest.get("distributed_rng"),
    }


def coordinate_training_file_authority(
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    manifest_path: Path,
    dataset_root: Path,
    base_path: Path,
    context: DistributedContext,
) -> bool:
    """Hash once per process/run, then let every loader reuse the shared proof."""

    key = (
        str(manifest_path.resolve()),
        spec["authority"]["canonical_manifest_sha256"],
        str(dataset_root.resolve()),
        str(base_path.resolve()),
        spec["authority"]["model_weight_sha256"],
    )
    if key in _VERIFIED_TRAINING_FILE_AUTHORITIES:
        return False
    primary_error: BaseException | None = None
    if context.is_primary:
        try:
            validate_base_training_files_authority(
                spec,
                phase0,
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                base_path=base_path,
            )
        except BaseException as error:
            primary_error = error
    broadcast_primary_error(context, primary_error)
    _VERIFIED_TRAINING_FILE_AUTHORITIES.add(key)
    return False


@dataclass
class TrainingRuntime:
    dataset: Any
    policy: Any
    base_policy: Any
    preprocessor: Any
    postprocessor: Any
    optimizer: torch.optim.Optimizer
    scheduler: Any
    loader: Any
    iterator: Any
    completed_step: int
    topology: TrainingTopology
    distributed_context: DistributedContext


def log_training_progress(
    trackio: Any,
    runtime: TrainingRuntime,
    record: dict[str, Any],
    spec: dict[str, Any],
    *,
    target_step: int,
) -> None:
    if not runtime.distributed_context.is_primary:
        return
    if not should_log_training_step(
        runtime.completed_step,
        target_step=target_step,
        every=spec["tracking"]["log_every_optimizer_steps"],
    ):
        return
    metrics = {
        "base/loss": record["loss"],
        "base/gradient_norm": record["gradient_norm"],
        "base/learning_rate_used": record["learning_rate_used"],
        "base/next_learning_rate": record["next_learning_rate"],
        "base/samples_per_second": spec["base_fit"]["effective_batch_size"]
        / record["wall_seconds"],
    }
    if trackio is None:
        raise GateZeroBaseTrainError("rank 0 tracking was not initialized")
    trackio.log(metrics, step=runtime.completed_step)
    print(
        json.dumps(
            {
                "event": "source_base_progress",
                "completed_step": runtime.completed_step,
                "target_step": target_step,
                **metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _update_state_digest(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode() + b"\0")
    else:
        raise GateZeroBaseTrainError(
            f"unsupported state value for hashing: {type(value).__name__}"
        )


def canonical_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def assert_exact_resume_equivalence(
    uninterrupted: dict[str, Any], resumed: dict[str, Any]
) -> dict[str, Any]:
    surfaces = (
        "completed_step",
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "rng_state_sha256",
        "next_raw_batch_sha256",
        "next_row_keys_sha256",
    )
    comparisons: dict[str, bool] = {}
    for surface in surfaces:
        if surface not in uninterrupted or surface not in resumed:
            raise GateZeroBaseTrainError(f"resume authority missing: {surface}")
        comparisons[surface] = uninterrupted[surface] == resumed[surface]
        if not comparisons[surface]:
            raise GateZeroBaseTrainError(f"resume mismatch: {surface}")
    return {"all_exact": True, "surfaces": comparisons}


def build_source_base_checkpoint_metadata(
    spec: dict[str, Any],
    *,
    config_path: Path,
    phase0_path: Path,
    completed_step: int,
    topology_config_path: Path,
    topology: TrainingTopology,
) -> dict[str, Any]:
    repository_root = config_path.resolve().parent.parent
    implementation_paths = (
        "src/ember/gate_zero_base_train.py",
        "src/ember/gate_zero_base_probe.py",
        "src/ember/gate_zero_base_session.py",
        "src/ember/gate_zero_base_runtime.py",
        "src/ember/gate_zero_checkpoint.py",
        "src/ember/gate_zero_contract.py",
        "src/ember/gate_zero_data.py",
        "src/ember/gate_zero_runtime.py",
        "src/ember/gate_zero_distributed.py",
    )
    selection = spec["base_fit"]["batch_calibration"]["selection_authority"]
    checkpoint = spec["base_fit"]["checkpoint"]
    role = (
        checkpoint["scientific_policy_role"]
        if completed_step == checkpoint["scientific_checkpoint_step"]
        else checkpoint["recovery_policy_role"]
    )
    return {
        "checkpoint_role": role,
        "topology": {
            "world_size": topology.world_size,
            "global_effective_batch_size": topology.global_effective_batch_size,
            "micro_batch_size": topology.per_rank_micro_batch_size,
            "per_rank_micro_batch_size": topology.per_rank_micro_batch_size,
            "gradient_accumulation_steps": topology.gradient_accumulation_steps,
            "num_workers": topology.data_workers_per_rank,
            "data_workers_per_rank": topology.data_workers_per_rank,
            "total_num_workers": topology.total_data_workers,
            "global_slot_algorithm": topology.global_slot_algorithm,
            "flow_input_authority": topology.flow_input_authority,
            "ddp_static_graph": topology.ddp_static_graph,
            "checkpoint_writer_rank": 0,
        },
        "authorities": {
            "base_revision": spec["authority"]["model_revision"],
            "base_weight_sha256": spec["authority"]["model_weight_sha256"],
            "normalization_sha256": spec["authority"]["source_normalization_sha256"],
            "gate_zero_contract_sha256": sha256_file(config_path),
            "phase0_contract_sha256": sha256_file(phase0_path),
            "canonical_manifest_sha256": spec["authority"]["canonical_manifest_sha256"],
            "batch_calibration_result_sha256": selection["result_sha256"],
            "topology_contract_sha256": sha256_file(topology_config_path),
            "implementation_files_sha256": {
                relative: sha256_file(repository_root / relative)
                for relative in implementation_paths
            },
        },
        "sampler": {
            "algorithm": topology.global_slot_algorithm,
            "within_slot_draw_algorithm": spec["base_fit"]["batch_calibration"][
                "effective_batch_draw_algorithm"
            ],
            "seed": spec["base_fit"]["seed"],
            "next_optimizer_step": completed_step,
            "rank_local_state": "reconstructed_from_absolute_step_rank_world_size_and_seed",
        },
        "precision": spec["base_fit"]["precision"],
        "optimizer": spec["base_fit"]["optimizer"],
        "scheduler": spec["base_fit"]["scheduler_implementation"],
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def new_training_runtime(
    args: argparse.Namespace,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    target_step: int,
    topology: TrainingTopology,
    distributed_context: DistributedContext,
    resume_from: Path | None = None,
) -> TrainingRuntime:
    set_global_seed(spec["base_fit"]["seed"])
    verify_dataset_sha256 = coordinate_training_file_authority(
        spec,
        phase0,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        base_path=args.base_path,
        context=distributed_context,
    )
    dataset, base_policy, preprocessor, postprocessor = load_base_training_components(
        spec=spec,
        phase0=phase0,
        manifest_path=args.manifest,
        normalization_path=args.normalization,
        dataset_root=args.dataset_root,
        base_path=args.base_path,
        vlm_path=args.vlm_path,
        verify_dataset_sha256=verify_dataset_sha256,
        verify_base_weight_sha256=verify_dataset_sha256,
    )
    optimizer = build_base_optimizer(
        [value for value in base_policy.parameters() if value.requires_grad], spec
    )
    scheduler = build_base_scheduler(optimizer, spec)
    completed_step = 0
    if resume_from is not None:
        resume_from = resume_from.resolve(strict=True)
        expected = build_source_base_checkpoint_metadata(
            spec,
            config_path=args.config,
            phase0_path=args.phase0_contract,
            completed_step=int(resume_from.name),
            topology_config_path=args.topology_config,
            topology=topology,
        )
        from ember.gate_zero_checkpoint import load_source_base_training_state_without_rng

        completed_step, optimizer, scheduler = load_source_base_training_state_without_rng(
            resume_from,
            policy=base_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            expected=expected,
        )
    if not 0 <= completed_step < target_step:
        raise GateZeroBaseTrainError("resume step must precede target step")
    policy = wrap_distributed_model(
        base_policy,
        distributed_context,
        static_graph=topology.ddp_static_graph,
    )
    loader = make_base_loader(
        dataset,
        micro_batch_size=topology.per_rank_micro_batch_size,
        effective_batch_size=spec["base_fit"]["effective_batch_size"],
        optimizer_steps=target_step - completed_step,
        start_optimizer_step=completed_step,
        sampler_seed=spec["base_fit"]["seed"],
        num_workers=topology.data_workers_per_rank,
        prefetch_factor=spec["base_fit"]["batch_calibration"]["prefetch_factor"],
        persistent_workers=spec["base_fit"]["batch_calibration"]["persistent_workers"],
        pin_memory=spec["base_fit"]["batch_calibration"]["pin_memory"],
        rank=distributed_context.rank,
        world_size=distributed_context.world_size,
    )
    iterator = iter(loader)
    if resume_from is not None:
        restore_source_base_checkpoint_rng(
            resume_from,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
        )
    else:
        # Training RNG begins after one-time model/DDP/loader/authority setup.
        # First-in-process setup may consume ambient Python RNG; reseeding here
        # gives fresh branches and same-topology resumes one explicit boundary.
        set_global_seed(spec["base_fit"]["seed"])
    return TrainingRuntime(
        dataset=dataset,
        policy=policy,
        base_policy=base_policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=loader,
        iterator=iterator,
        completed_step=completed_step,
        topology=topology,
        distributed_context=distributed_context,
    )


def close_training_runtime(runtime: TrainingRuntime) -> None:
    del runtime.iterator, runtime.loader
    runtime.dataset.close()
    del runtime.policy, runtime.base_policy, runtime.optimizer, runtime.scheduler
    gc.collect()
    torch.cuda.empty_cache()


def run_one_step(runtime: TrainingRuntime, spec: dict[str, Any]) -> dict[str, Any]:
    record = optimizer_step(
        runtime.iterator,
        policy=runtime.policy,
        preprocessor=runtime.preprocessor,
        optimizer=runtime.optimizer,
        spec=spec,
        optimizer_step_index=runtime.completed_step,
        accumulation_steps=runtime.topology.gradient_accumulation_steps,
        fixed_flow_seed=None,
        topology=runtime.topology,
        distributed_context=runtime.distributed_context,
    )
    runtime.scheduler.step()
    runtime.completed_step += 1
    record["completed_step"] = runtime.completed_step
    record["next_learning_rate"] = float(runtime.optimizer.param_groups[0]["lr"])
    return record


def _rng_state_sha256() -> str:
    from lerobot.utils.random_utils import serialize_rng_state

    return canonical_state_sha256(serialize_rng_state())


def _next_batch_authority(
    runtime: TrainingRuntime, spec: dict[str, Any]
) -> dict[str, str]:
    rng_before = _rng_state_sha256()
    loader = make_base_loader(
        runtime.dataset,
        micro_batch_size=runtime.topology.per_rank_micro_batch_size,
        effective_batch_size=spec["base_fit"]["effective_batch_size"],
        optimizer_steps=1,
        start_optimizer_step=runtime.completed_step,
        sampler_seed=spec["base_fit"]["seed"],
        num_workers=runtime.topology.data_workers_per_rank,
        prefetch_factor=spec["base_fit"]["batch_calibration"]["prefetch_factor"],
        persistent_workers=spec["base_fit"]["batch_calibration"]["persistent_workers"],
        pin_memory=spec["base_fit"]["batch_calibration"]["pin_memory"],
        rank=runtime.distributed_context.rank,
        world_size=runtime.distributed_context.world_size,
    )
    iterator = iter(loader)
    raw_batch = next(iterator)
    local = {
        "rank": runtime.distributed_context.rank,
        "raw_batch_sha256": canonical_state_sha256(raw_batch),
        "row_keys": batch_provenance_keys(raw_batch),
    }
    gathered = gather_rank_objects(local, runtime.distributed_context)
    result = None
    if runtime.distributed_context.is_primary:
        result = {
            "next_raw_batch_sha256": canonical_state_sha256(
                [item["raw_batch_sha256"] for item in gathered]
            ),
            "next_row_keys_sha256": canonical_state_sha256(
                [item["row_keys"] for item in gathered]
            ),
        }
    result = broadcast_primary_object(runtime.distributed_context, result)
    del raw_batch, iterator, loader
    gc.collect()
    if _rng_state_sha256() != rng_before:
        raise GateZeroBaseTrainError("next-batch audit changed global RNG")
    return result


def capture_runtime_authority(
    runtime: TrainingRuntime, spec: dict[str, Any]
) -> dict[str, Any]:
    rng_digests = gather_rank_objects(_rng_state_sha256(), runtime.distributed_context)
    authority = None
    if runtime.distributed_context.is_primary:
        authority = {
            "completed_step": runtime.completed_step,
            "model_state_sha256": canonical_state_sha256(runtime.base_policy.state_dict()),
            "optimizer_state_sha256": canonical_state_sha256(runtime.optimizer.state_dict()),
            "scheduler_state_sha256": canonical_state_sha256(runtime.scheduler.state_dict()),
            "rng_state_sha256": canonical_state_sha256(rng_digests),
            "rank_rng_state_sha256": rng_digests,
        }
    next_batch = _next_batch_authority(runtime, spec)
    if runtime.distributed_context.is_primary:
        authority.update(next_batch)
    return broadcast_primary_object(runtime.distributed_context, authority)


def save_runtime_checkpoint(
    runtime: TrainingRuntime,
    args: argparse.Namespace,
    spec: dict[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any] | None:
    rng_before = _rng_state_sha256()
    rank_rng_states = gather_rank_rng_states(runtime.distributed_context)
    manifest = None
    primary_error: BaseException | None = None
    if runtime.distributed_context.is_primary:
        try:
            metadata = build_source_base_checkpoint_metadata(
                spec,
                config_path=args.config,
                phase0_path=args.phase0_contract,
                completed_step=runtime.completed_step,
                topology_config_path=args.topology_config,
                topology=runtime.topology,
            )
            manifest = save_source_base_checkpoint(
                checkpoint_dir,
                step=runtime.completed_step,
                policy=runtime.base_policy,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                preprocessor=runtime.preprocessor,
                postprocessor=runtime.postprocessor,
                metadata=metadata,
                rank_rng_states=rank_rng_states,
            )
        except BaseException as error:
            primary_error = error
    broadcast_primary_error(runtime.distributed_context, primary_error)
    if _rng_state_sha256() != rng_before:
        raise GateZeroBaseTrainError("checkpoint save changed global RNG")
    return manifest
