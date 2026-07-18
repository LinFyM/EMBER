"""One-load, source-only SmolVLA batch calibration for Gate 0."""

from __future__ import annotations

import argparse
import gc
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from ember.eval_artifacts import update_latest_link
from ember.gate_zero_base_runtime import (
    build_base_optimizer,
    capture_trainable_state,
    gradient_accumulation_steps as _base_accumulation_steps,
    load_base_training_components,
    make_base_loader,
    optimizer_step as run_base_optimizer_step,
    optimizer_state_summary,
    restore_trainable_state,
)
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_data import SourceHdf5Dataset
from ember.gate_zero_runtime import (
    parameter_summary,
    set_global_seed,
    sha256_file,
)


MIB = 1024 * 1024


class GateZeroBatchCalibrationError(RuntimeError):
    """Raised when technical calibration cannot preserve its frozen contract."""


def gradient_accumulation_steps(effective_batch_size: int, micro_batch_size: int) -> int:
    try:
        return _base_accumulation_steps(effective_batch_size, micro_batch_size)
    except ValueError as error:
        raise GateZeroBatchCalibrationError(str(error)) from error


def select_calibration_candidate(
    records: list[dict[str, Any]],
    *,
    minimum_free_memory_mib: int,
    effective_batch_size: int = 64,
) -> dict[str, Any]:
    safe = [
        record
        for record in records
        if record.get("status") == "completed"
        and isinstance(record.get("samples_per_second"), (int, float))
        and record.get("minimum_free_memory_mib", -1) >= minimum_free_memory_mib
    ]
    if not safe:
        raise GateZeroBatchCalibrationError("no completed safe candidate retains memory headroom")
    winner = max(safe, key=lambda item: (item["samples_per_second"], item["micro_batch_size"]))
    micro_batch_size = int(winner["micro_batch_size"])
    return {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps(
            effective_batch_size, micro_batch_size
        ),
        "samples_per_second": winner["samples_per_second"],
        "minimum_free_memory_mib": winner["minimum_free_memory_mib"],
        "selection_rule": "highest_measured_samples_per_second_with_headroom",
    }


def validate_output_destination(output_dir: Path) -> None:
    if not output_dir.is_absolute():
        raise GateZeroBatchCalibrationError("output directory must be absolute")
    if (output_dir / "calibration_result.json").exists():
        raise GateZeroBatchCalibrationError("refusing to overwrite completed calibration")
    output_dir.mkdir(parents=True, exist_ok=True)


def assert_matched_candidate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed unless every completed candidate used the same effective batches."""

    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        raise GateZeroBatchCalibrationError("no completed candidate to match")
    reference = completed[0]
    row_digests = reference.get("optimizer_step_row_keys_sha256")
    fixed_flow_seed = reference.get("fixed_flow_seed")
    if (
        not isinstance(row_digests, list)
        or not row_digests
        or any(not isinstance(value, str) or len(value) != 64 for value in row_digests)
    ):
        raise GateZeroBatchCalibrationError("invalid effective-batch draw authority")
    if not isinstance(fixed_flow_seed, int):
        raise GateZeroBatchCalibrationError("invalid fixed flow-noise authority")
    for record in completed:
        if record.get("matched_initial_trainable_state") is not True:
            raise GateZeroBatchCalibrationError("candidate initial trainable state was not restored")
        if record.get("optimizer_step_row_keys_sha256") != row_digests:
            raise GateZeroBatchCalibrationError("candidate effective-batch draws differ")
        if record.get("fixed_flow_seed") != fixed_flow_seed:
            raise GateZeroBatchCalibrationError("candidate fixed flow noise/time differs")
    return {
        "completed_candidate_count": len(completed),
        "fixed_flow_seed": fixed_flow_seed,
        "optimizer_step_row_keys_sha256": row_digests,
    }


def _run_candidate(
    dataset: SourceHdf5Dataset,
    *,
    policy: Any,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    micro_batch_size: int,
) -> dict[str, Any]:
    base_fit = spec["base_fit"]
    calibration = base_fit["batch_calibration"]
    accumulation_steps = gradient_accumulation_steps(
        base_fit["effective_batch_size"], micro_batch_size
    )
    loader = make_base_loader(
        dataset,
        micro_batch_size=micro_batch_size,
        effective_batch_size=base_fit["effective_batch_size"],
        optimizer_steps=calibration["technical_steps_per_candidate"],
        start_optimizer_step=0,
        sampler_seed=calibration["calibration_seed"],
        num_workers=calibration["num_workers"],
        prefetch_factor=calibration["prefetch_factor"],
        persistent_workers=calibration["persistent_workers"],
        pin_memory=calibration["pin_memory"],
    )
    iterator = iter(loader)
    measured_seconds: list[float] = []
    optimizer_step_row_keys_sha256: list[str] = []
    minimum_free_memory_mib: int | None = None
    try:
        for optimizer_step in range(calibration["technical_steps_per_candidate"]):
            if optimizer_step == calibration["warmup_optimizer_steps_per_candidate"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            step_record = run_base_optimizer_step(
                iterator,
                policy=policy,
                preprocessor=preprocessor,
                optimizer=optimizer,
                spec=spec,
                optimizer_step_index=optimizer_step,
                accumulation_steps=accumulation_steps,
                fixed_flow_seed=calibration["calibration_seed"],
            )
            optimizer_step_row_keys_sha256.append(step_record["row_keys_sha256"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if optimizer_step >= calibration["warmup_optimizer_steps_per_candidate"]:
                measured_seconds.append(elapsed)
                free_mib = int(torch.cuda.mem_get_info()[0] // MIB)
                minimum_free_memory_mib = (
                    free_mib
                    if minimum_free_memory_mib is None
                    else min(minimum_free_memory_mib, free_mib)
                )
    finally:
        del iterator, loader
        gc.collect()
    expected_measured = calibration["measured_optimizer_steps_per_candidate"]
    if len(measured_seconds) != expected_measured or minimum_free_memory_mib is None:
        raise GateZeroBatchCalibrationError("calibration measured-step accounting changed")
    measured_samples = base_fit["effective_batch_size"] * expected_measured
    return {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "status": "completed",
        "warmup_optimizer_steps": calibration["warmup_optimizer_steps_per_candidate"],
        "measured_optimizer_steps": expected_measured,
        "measured_samples": measured_samples,
        "optimizer_step_seconds": measured_seconds,
        "samples_per_second": measured_samples / sum(measured_seconds),
        "torch_peak_allocated_mib": int(torch.cuda.max_memory_allocated() // MIB),
        "torch_peak_reserved_mib": int(torch.cuda.max_memory_reserved() // MIB),
        "minimum_free_memory_mib": minimum_free_memory_mib,
        "headroom_pass": minimum_free_memory_mib >= calibration["minimum_free_memory_mib"],
        "matched_initial_trainable_state": True,
        "fixed_flow_seed": calibration["calibration_seed"],
        "optimizer_step_row_keys_sha256": optimizer_step_row_keys_sha256,
        "optimizer_state": optimizer_state_summary(optimizer),
    }


def _write_result(result: dict[str, Any], output_dir: Path, latest_link: Path) -> None:
    result_path = output_dir / "calibration_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = f"{sha256_file(result_path)}  {result_path.name}\n"
    (output_dir / "checksums.sha256").write_text(checksum, encoding="utf-8")
    update_latest_link(output_dir, latest_link)


def _prepare_runtime(
    args: argparse.Namespace, spec: dict[str, Any], phase0: dict[str, Any]
) -> tuple[SourceHdf5Dataset, Any, Any]:
    dataset, policy, preprocessor, _ = load_base_training_components(
        spec=spec,
        phase0=phase0,
        manifest_path=args.manifest,
        normalization_path=args.normalization,
        dataset_root=args.dataset_root,
        base_path=args.base_path,
        vlm_path=args.vlm_path,
    )
    return dataset, policy, preprocessor


def _initialize_tracking(spec: dict[str, Any], run_name: str) -> Any:
    import trackio

    trackio.init(
        project=spec["tracking"]["project"],
        name=run_name,
        group="batch_calibration",
        config={
            "effective_batch_size": spec["base_fit"]["effective_batch_size"],
            "candidates": spec["base_fit"]["batch_calibration"]["micro_batch_candidates"],
            "git_model_revision": spec["authority"]["model_revision"],
        },
        auto_log_gpu=spec["tracking"]["log_system_metrics"],
        gpu_log_interval=1.0,
        auto_log_cpu=spec["tracking"]["log_system_metrics"],
        cpu_log_interval=1.0,
    )
    return trackio


def _oom_record(
    spec: dict[str, Any], micro_batch_size: int, error: torch.cuda.OutOfMemoryError
) -> dict[str, Any]:
    return {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps(
            spec["base_fit"]["effective_batch_size"], micro_batch_size
        ),
        "status": "oom",
        "failure_class": "resource_throughput",
        "error_type": type(error).__name__,
        "samples_per_second": None,
        "minimum_free_memory_mib": None,
    }


def _log_candidate(trackio: Any, record: dict[str, Any], *, step: int) -> None:
    if record["status"] == "completed":
        trackio.log(
            {
                "calibration/micro_batch_size": record["micro_batch_size"],
                "calibration/samples_per_second": record["samples_per_second"],
                "calibration/peak_allocated_mib": record["torch_peak_allocated_mib"],
                "calibration/peak_reserved_mib": record["torch_peak_reserved_mib"],
                "calibration/minimum_free_memory_mib": record["minimum_free_memory_mib"],
            },
            step=step,
        )
    else:
        trackio.log(
            {"calibration/oom": 1, "calibration/micro_batch_size": record["micro_batch_size"]},
            step=step,
        )


def _calibrate_candidates(
    dataset: SourceHdf5Dataset,
    *,
    policy: Any,
    preprocessor: Any,
    spec: dict[str, Any],
    trackio: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    initial_trainable_state = capture_trainable_state(policy)
    for index, micro_batch_size in enumerate(
        spec["base_fit"]["batch_calibration"]["micro_batch_candidates"]
    ):
        restore_trainable_state(policy, initial_trainable_state)
        set_global_seed(spec["base_fit"]["batch_calibration"]["calibration_seed"])
        optimizer = build_base_optimizer(
            [value for value in policy.parameters() if value.requires_grad], spec
        )
        torch.cuda.empty_cache()
        try:
            record = _run_candidate(
                dataset,
                policy=policy,
                preprocessor=preprocessor,
                optimizer=optimizer,
                spec=spec,
                micro_batch_size=micro_batch_size,
            )
        except torch.cuda.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            record = _oom_record(spec, micro_batch_size, error)
        records.append(record)
        _log_candidate(trackio, record, step=index)
        del optimizer
        if record["status"] != "completed":
            break
    return records


def _build_result(
    spec: dict[str, Any], policy: Any, records: list[dict[str, Any]], *, started: float, run: str
) -> dict[str, Any]:
    match_authority = assert_matched_candidate_records(records)
    selected = select_calibration_candidate(
        records,
        minimum_free_memory_mib=spec["base_fit"]["batch_calibration"][
            "minimum_free_memory_mib"
        ],
        effective_batch_size=spec["base_fit"]["effective_batch_size"],
    )
    return {
        "schema_version": 1,
        "status": "calibration_completed",
        "scientific_outcome_metrics_recorded": False,
        "scientific_gate_decision_authorized": False,
        "writer_authorized": False,
        "gpu_count": 1,
        "manifest_sha256": spec["authority"]["canonical_manifest_sha256"],
        "normalization_sha256": spec["authority"]["source_normalization_sha256"],
        "base_weight_sha256": spec["authority"]["model_weight_sha256"],
        "parameter_summary": parameter_summary(policy),
        "effective_batch_size": spec["base_fit"]["effective_batch_size"],
        "candidate_records": records,
        "matched_initial_trainable_state": True,
        "matched_effective_batch_draws": True,
        "matched_flow_noise_and_time": True,
        "matched_candidate_authority": match_authority,
        "selected": selected,
        "wall_seconds": time.perf_counter() - started,
        "tracking": {
            "backend": "trackio",
            "project": spec["tracking"]["project"],
            "run": run,
            "dashboard_command": spec["tracking"]["dashboard_command"],
        },
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_destination(args.output_dir)
    started = time.perf_counter()
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    dataset, policy, preprocessor = _prepare_runtime(args, spec, phase0)
    trackio = _initialize_tracking(spec, args.output_dir.name)
    try:
        records = _calibrate_candidates(
            dataset,
            policy=policy,
            preprocessor=preprocessor,
            spec=spec,
            trackio=trackio,
        )
        result = _build_result(
            spec, policy, records, started=started, run=args.output_dir.name
        )
        trackio.finish()
        _write_result(result, args.output_dir, args.latest_link)
        return result
    except BaseException:
        trackio.finish()
        raise
    finally:
        dataset.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.absolute()
    args.latest_link = args.latest_link.absolute()
    result = run_calibration(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
