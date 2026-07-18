"""Canonical source-base trainer and exact checkpoint/resume probe for Gate 0."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import tomllib
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
)
from ember.gate_zero_checkpoint import (
    CHECKPOINT_MANIFEST,
    restore_source_base_checkpoint_rng,
    rotate_source_base_recovery_checkpoints,
    save_source_base_checkpoint,
    validate_source_base_checkpoint,
)
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_runtime import batch_provenance_keys, set_global_seed, sha256_file


class GateZeroBaseTrainError(RuntimeError):
    """Raised when source-base training or resume mechanics drift."""


@dataclass
class _TrainingRuntime:
    dataset: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any
    optimizer: torch.optim.Optimizer
    scheduler: Any
    loader: Any
    iterator: Any
    completed_step: int


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
        raise GateZeroBaseTrainError(f"unsupported state value for hashing: {type(value).__name__}")


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


def require_base_fit_authorization(spec: dict[str, Any], *, mode: str) -> None:
    if mode not in {"resume-probe", "train"}:
        raise GateZeroBaseTrainError(f"unknown source-base mode: {mode}")
    selection = spec["base_fit"]["batch_calibration"]["selection_authority"]
    if selection["status"] != "frozen_matched_resource_authority":
        raise GateZeroBaseTrainError("matched batch-shape authority is not frozen")
    if selection["authorized_as_batch_shape"] is not True:
        raise GateZeroBaseTrainError("batch shape is not authorized")
    if mode == "train" and selection["formal_base_fit_authorized"] is not True:
        raise GateZeroBaseTrainError("formal base fit is not authorized before resume identity")


def build_source_base_checkpoint_metadata(
    spec: dict[str, Any],
    *,
    config_path: Path,
    phase0_path: Path,
    completed_step: int,
) -> dict[str, Any]:
    repository_root = config_path.resolve().parent.parent
    implementation_paths = (
        "src/ember/gate_zero_base_train.py",
        "src/ember/gate_zero_base_runtime.py",
        "src/ember/gate_zero_checkpoint.py",
        "src/ember/gate_zero_contract.py",
        "src/ember/gate_zero_data.py",
        "src/ember/gate_zero_runtime.py",
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
            "world_size": checkpoint["world_size"],
            "micro_batch_size": selection["selected_micro_batch_size"],
            "gradient_accumulation_steps": selection[
                "selected_gradient_accumulation_steps"
            ],
            "num_workers": checkpoint["num_workers"],
        },
        "authorities": {
            "base_revision": spec["authority"]["model_revision"],
            "base_weight_sha256": spec["authority"]["model_weight_sha256"],
            "normalization_sha256": spec["authority"]["source_normalization_sha256"],
            "gate_zero_contract_sha256": sha256_file(config_path),
            "phase0_contract_sha256": sha256_file(phase0_path),
            "canonical_manifest_sha256": spec["authority"]["canonical_manifest_sha256"],
            "batch_calibration_result_sha256": selection["result_sha256"],
            "implementation_files_sha256": {
                relative: sha256_file(repository_root / relative)
                for relative in implementation_paths
            },
        },
        "sampler": {
            "algorithm": spec["base_fit"]["batch_calibration"][
                "effective_batch_draw_algorithm"
            ],
            "seed": spec["base_fit"]["seed"],
            "next_optimizer_step": completed_step,
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


def _selected_topology(spec: dict[str, Any]) -> tuple[int, int]:
    selection = spec["base_fit"]["batch_calibration"]["selection_authority"]
    return (
        selection["selected_micro_batch_size"],
        selection["selected_gradient_accumulation_steps"],
    )


def _new_training_runtime(
    args: argparse.Namespace,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    target_step: int,
    resume_from: Path | None = None,
) -> _TrainingRuntime:
    set_global_seed(spec["base_fit"]["seed"])
    dataset, policy, preprocessor, postprocessor = load_base_training_components(
        spec=spec,
        phase0=phase0,
        manifest_path=args.manifest,
        normalization_path=args.normalization,
        dataset_root=args.dataset_root,
        base_path=args.base_path,
        vlm_path=args.vlm_path,
    )
    optimizer = build_base_optimizer(
        [value for value in policy.parameters() if value.requires_grad], spec
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
        )
        from ember.gate_zero_checkpoint import load_source_base_training_state_without_rng

        completed_step, optimizer, scheduler = load_source_base_training_state_without_rng(
            resume_from,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            expected=expected,
        )
    if not 0 <= completed_step < target_step:
        raise GateZeroBaseTrainError("resume step must precede target step")
    micro_batch_size, _ = _selected_topology(spec)
    checkpoint = spec["base_fit"]["checkpoint"]
    loader = make_base_loader(
        dataset,
        micro_batch_size=micro_batch_size,
        effective_batch_size=spec["base_fit"]["effective_batch_size"],
        optimizer_steps=target_step - completed_step,
        start_optimizer_step=completed_step,
        sampler_seed=spec["base_fit"]["seed"],
        num_workers=checkpoint["num_workers"],
        prefetch_factor=spec["base_fit"]["batch_calibration"]["prefetch_factor"],
        persistent_workers=spec["base_fit"]["batch_calibration"]["persistent_workers"],
        pin_memory=spec["base_fit"]["batch_calibration"]["pin_memory"],
    )
    iterator = iter(loader)
    if resume_from is not None:
        restore_source_base_checkpoint_rng(resume_from)
    return _TrainingRuntime(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=loader,
        iterator=iterator,
        completed_step=completed_step,
    )


def _close_runtime(runtime: _TrainingRuntime) -> None:
    del runtime.iterator, runtime.loader
    runtime.dataset.close()
    del runtime.policy, runtime.optimizer, runtime.scheduler
    gc.collect()
    torch.cuda.empty_cache()


def _run_one_step(runtime: _TrainingRuntime, spec: dict[str, Any]) -> dict[str, Any]:
    _, accumulation_steps = _selected_topology(spec)
    record = optimizer_step(
        runtime.iterator,
        policy=runtime.policy,
        preprocessor=runtime.preprocessor,
        optimizer=runtime.optimizer,
        spec=spec,
        optimizer_step_index=runtime.completed_step,
        accumulation_steps=accumulation_steps,
        fixed_flow_seed=None,
    )
    runtime.scheduler.step()
    runtime.completed_step += 1
    record["completed_step"] = runtime.completed_step
    record["next_learning_rate"] = float(runtime.optimizer.param_groups[0]["lr"])
    return record


def _next_batch_authority(runtime: _TrainingRuntime, spec: dict[str, Any]) -> dict[str, str]:
    rng_before = _rng_state_sha256()
    micro_batch_size, _ = _selected_topology(spec)
    checkpoint = spec["base_fit"]["checkpoint"]
    loader = make_base_loader(
        runtime.dataset,
        micro_batch_size=micro_batch_size,
        effective_batch_size=spec["base_fit"]["effective_batch_size"],
        optimizer_steps=1,
        start_optimizer_step=runtime.completed_step,
        sampler_seed=spec["base_fit"]["seed"],
        num_workers=checkpoint["num_workers"],
        prefetch_factor=spec["base_fit"]["batch_calibration"]["prefetch_factor"],
        persistent_workers=spec["base_fit"]["batch_calibration"]["persistent_workers"],
        pin_memory=spec["base_fit"]["batch_calibration"]["pin_memory"],
    )
    iterator = iter(loader)
    raw_batch = next(iterator)
    row_keys = batch_provenance_keys(raw_batch)
    result = {
        "next_raw_batch_sha256": canonical_state_sha256(raw_batch),
        "next_row_keys_sha256": canonical_state_sha256(row_keys),
    }
    del raw_batch, iterator, loader
    gc.collect()
    if _rng_state_sha256() != rng_before:
        raise GateZeroBaseTrainError("next-batch audit changed global RNG")
    return result


def _rng_state_sha256() -> str:
    from lerobot.utils.random_utils import serialize_rng_state

    return canonical_state_sha256(serialize_rng_state())


def _capture_runtime_authority(
    runtime: _TrainingRuntime, spec: dict[str, Any]
) -> dict[str, Any]:
    authority = {
        "completed_step": runtime.completed_step,
        "model_state_sha256": canonical_state_sha256(runtime.policy.state_dict()),
        "optimizer_state_sha256": canonical_state_sha256(runtime.optimizer.state_dict()),
        "scheduler_state_sha256": canonical_state_sha256(runtime.scheduler.state_dict()),
        "rng_state_sha256": _rng_state_sha256(),
    }
    authority.update(_next_batch_authority(runtime, spec))
    return authority


def _save_runtime_checkpoint(
    runtime: _TrainingRuntime,
    args: argparse.Namespace,
    spec: dict[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    rng_before = _rng_state_sha256()
    metadata = build_source_base_checkpoint_metadata(
        spec,
        config_path=args.config,
        phase0_path=args.phase0_contract,
        completed_step=runtime.completed_step,
    )
    manifest = save_source_base_checkpoint(
        checkpoint_dir,
        step=runtime.completed_step,
        policy=runtime.policy,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        preprocessor=runtime.preprocessor,
        postprocessor=runtime.postprocessor,
        metadata=metadata,
    )
    if _rng_state_sha256() != rng_before:
        raise GateZeroBaseTrainError("checkpoint save changed global RNG")
    return manifest


def _initialize_tracking(spec: dict[str, Any], args: argparse.Namespace) -> Any:
    import trackio

    trackio.init(
        project=spec["tracking"]["project"],
        name=args.output_dir.name,
        group="base_resume_probe" if args.mode == "resume-probe" else "source_base_fit",
        config={
            "mode": args.mode,
            "effective_batch_size": spec["base_fit"]["effective_batch_size"],
            "micro_batch_size": _selected_topology(spec)[0],
            "model_revision": spec["authority"]["model_revision"],
        },
        auto_log_gpu=spec["tracking"]["log_system_metrics"],
        gpu_log_interval=1.0,
        auto_log_cpu=spec["tracking"]["log_system_metrics"],
        cpu_log_interval=1.0,
    )
    return trackio


def _validate_output(args: argparse.Namespace, *, result_name: str) -> None:
    if not args.output_dir.is_absolute() or not args.latest_link.is_absolute():
        raise GateZeroBaseTrainError("output and latest paths must be absolute")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / result_name).exists():
        raise GateZeroBaseTrainError("refusing to overwrite completed source-base result")


def _write_result(
    result: dict[str, Any], args: argparse.Namespace, *, result_name: str
) -> None:
    result_path = args.output_dir / result_name
    temporary = args.output_dir / f".{result_name}.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, result_path)
    checksum = f"{sha256_file(result_path)}  {result_name}\n"
    (args.output_dir / "checksums.sha256").write_text(checksum, encoding="utf-8")
    update_latest_link(args.output_dir, args.latest_link)


def _checkpoint_evidence(checkpoint_dir: Path) -> dict[str, Any]:
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
    }


def _cleanup_probe_checkpoint(checkpoint_dir: Path) -> None:
    validate_source_base_checkpoint(checkpoint_dir)
    checkpoint_root = checkpoint_dir.parent
    sidecar = checkpoint_root / f"{checkpoint_dir.name}.manifest.sha256"
    last = checkpoint_root / "last"
    if last.is_symlink():
        last.unlink()
    shutil.rmtree(checkpoint_dir)
    sidecar.unlink()
    if any(checkpoint_root.iterdir()):
        raise GateZeroBaseTrainError("unexpected files prevent transient checkpoint cleanup")
    checkpoint_root.rmdir()


def run_resume_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.resume_from is not None:
        raise GateZeroBaseTrainError("resume probe does not accept an external checkpoint")
    _validate_output(args, result_name="resume_probe_result.json")
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    require_base_fit_authorization(spec, mode="resume-probe")
    probe = spec["base_fit"]["resume_probe"]
    target_step = probe["uninterrupted_target_step"]
    checkpoint_step = probe["interrupted_checkpoint_step"]
    if target_step != 2 or checkpoint_step != 1:
        raise GateZeroBaseTrainError("resume probe step contract changed")
    started = time.perf_counter()
    trackio = _initialize_tracking(spec, args)
    try:
        uninterrupted_runtime = _new_training_runtime(
            args, spec, phase0, target_step=target_step
        )
        while uninterrupted_runtime.completed_step < target_step:
            _run_one_step(uninterrupted_runtime, spec)
        uninterrupted = _capture_runtime_authority(uninterrupted_runtime, spec)
        _close_runtime(uninterrupted_runtime)

        interrupted_runtime = _new_training_runtime(
            args, spec, phase0, target_step=checkpoint_step
        )
        while interrupted_runtime.completed_step < checkpoint_step:
            _run_one_step(interrupted_runtime, spec)
        checkpoint_dir = args.output_dir / "checkpoints" / f"{checkpoint_step:06d}"
        _save_runtime_checkpoint(interrupted_runtime, args, spec, checkpoint_dir)
        checkpoint_evidence = _checkpoint_evidence(checkpoint_dir)
        _close_runtime(interrupted_runtime)

        resumed_runtime = _new_training_runtime(
            args,
            spec,
            phase0,
            target_step=target_step,
            resume_from=checkpoint_dir,
        )
        while resumed_runtime.completed_step < target_step:
            _run_one_step(resumed_runtime, spec)
        resumed = _capture_runtime_authority(resumed_runtime, spec)
        _close_runtime(resumed_runtime)
        comparison = assert_exact_resume_equivalence(uninterrupted, resumed)
        _cleanup_probe_checkpoint(checkpoint_dir)
        result = {
            "schema_version": 1,
            "status": "resume_probe_passed",
            "scientific_outcome_metrics_recorded": False,
            "source_policy_outcome_recorded": False,
            "scientific_gate_decision_authorized": False,
            "writer_authorized": False,
            "gpu_count": 1,
            "authorities": build_source_base_checkpoint_metadata(
                spec,
                config_path=args.config,
                phase0_path=args.phase0_contract,
                completed_step=checkpoint_step,
            )["authorities"],
            "topology": build_source_base_checkpoint_metadata(
                spec,
                config_path=args.config,
                phase0_path=args.phase0_contract,
                completed_step=checkpoint_step,
            )["topology"],
            "uninterrupted": uninterrupted,
            "resumed": resumed,
            "comparison": comparison,
            "checkpoint_evidence": checkpoint_evidence,
            "transient_full_checkpoint_cleaned": True,
            "wall_seconds": time.perf_counter() - started,
            "tracking": {
                "backend": "trackio",
                "project": spec["tracking"]["project"],
                "run": args.output_dir.name,
                "dashboard_command": spec["tracking"]["dashboard_command"],
            },
        }
        trackio.log({f"resume/{key}": int(value) for key, value in comparison["surfaces"].items()})
        trackio.log({"resume/all_exact": 1})
        trackio.finish()
        _write_result(result, args, result_name="resume_probe_result.json")
        return result
    except BaseException:
        trackio.finish()
        raise


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    _validate_output(args, result_name="training_result.json")
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    require_base_fit_authorization(spec, mode="train")
    target_step = spec["base_fit"]["steps"]
    started = time.perf_counter()
    trackio = _initialize_tracking(spec, args)
    runtime: _TrainingRuntime | None = None
    last_record: dict[str, Any] | None = None
    try:
        runtime = _new_training_runtime(
            args,
            spec,
            phase0,
            target_step=target_step,
            resume_from=args.resume_from,
        )
        while runtime.completed_step < target_step:
            last_record = _run_one_step(runtime, spec)
            trackio.log(
                {
                    "base/loss": last_record["loss"],
                    "base/gradient_norm": last_record["gradient_norm"],
                    "base/learning_rate_used": last_record["learning_rate_used"],
                    "base/next_learning_rate": last_record["next_learning_rate"],
                    "base/samples_per_second": spec["base_fit"]["effective_batch_size"]
                    / last_record["wall_seconds"],
                },
                step=runtime.completed_step,
            )
            checkpoint_due = (
                runtime.completed_step % spec["base_fit"]["checkpoint_every_steps"] == 0
                or runtime.completed_step == target_step
            )
            if checkpoint_due:
                checkpoint_dir = (
                    args.output_dir / "checkpoints" / f"{runtime.completed_step:06d}"
                )
                _save_runtime_checkpoint(runtime, args, spec, checkpoint_dir)
                rotate_source_base_recovery_checkpoints(
                    checkpoint_dir.parent,
                    keep=spec["base_fit"]["recoverable_checkpoints_to_keep"],
                )
        if last_record is None:
            raise GateZeroBaseTrainError("formal base fit performed no optimizer step")
        final_checkpoint = args.output_dir / "checkpoints" / f"{target_step:06d}"
        result = {
            "schema_version": 1,
            "status": "source_base_fit_completed_pending_competence",
            "completed_steps": target_step,
            "effective_batch_size": spec["base_fit"]["effective_batch_size"],
            "source_policy_outcome_recorded": False,
            "gate_zero_authorized": False,
            "writer_authorized": False,
            "authorities": build_source_base_checkpoint_metadata(
                spec,
                config_path=args.config,
                phase0_path=args.phase0_contract,
                completed_step=target_step,
            )["authorities"],
            "final_checkpoint": _checkpoint_evidence(final_checkpoint),
            "wall_seconds": time.perf_counter() - started,
            "tracking": {
                "backend": "trackio",
                "project": spec["tracking"]["project"],
                "run": args.output_dir.name,
                "dashboard_command": spec["tracking"]["dashboard_command"],
            },
        }
        trackio.finish()
        _write_result(result, args, result_name="training_result.json")
        return result
    except BaseException:
        trackio.finish()
        raise
    finally:
        if runtime is not None:
            _close_runtime(runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("resume-probe", "train"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.absolute()
    args.latest_link = args.latest_link.absolute()
    result = run_resume_probe(args) if args.mode == "resume-probe" else run_training(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
