"""One-load, source-only SmolVLA batch calibration for Gate 0."""

from __future__ import annotations

import argparse
import gc
import json
import time
import tomllib
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from ember.eval_artifacts import update_latest_link
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_data import (
    GateZeroSurface,
    SourceHdf5Dataset,
    TaskDemoFrameBatchSampler,
    load_surface_authorities,
)
from ember.gate_zero_runtime import (
    batch_provenance_keys,
    deterministic_flow_inputs,
    load_smolvla_policy,
    load_source_normalization,
    parameter_summary,
    preprocess_smolvla_batch,
    set_global_seed,
    sha256_file,
    smolvla_flow_loss,
)


MIB = 1024 * 1024


class GateZeroBatchCalibrationError(RuntimeError):
    """Raised when technical calibration cannot preserve its frozen contract."""


def gradient_accumulation_steps(effective_batch_size: int, micro_batch_size: int) -> int:
    if effective_batch_size <= 0 or micro_batch_size <= 0:
        raise GateZeroBatchCalibrationError("batch sizes must be positive")
    if effective_batch_size % micro_batch_size:
        raise GateZeroBatchCalibrationError("microbatch must divide the effective batch")
    return effective_batch_size // micro_batch_size


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


def _training_row_keys(
    raw_batch: dict[str, Any],
    *,
    micro_batch_size: int,
    optimizer_step: int,
    accumulation_step: int,
) -> list[str]:
    return [
        f"{key}/mb{micro_batch_size}/step{optimizer_step}/acc{accumulation_step}/slot{slot}"
        for slot, key in enumerate(batch_provenance_keys(raw_batch))
    ]


def _make_loader(
    dataset: SourceHdf5Dataset,
    *,
    micro_batch_size: int,
    effective_batch_size: int,
    optimizer_steps: int,
    seed: int,
    calibration: dict[str, Any],
) -> DataLoader:
    sampler = TaskDemoFrameBatchSampler(
        dataset,
        micro_batch_size=micro_batch_size,
        optimizer_steps=optimizer_steps,
        gradient_accumulation_steps=gradient_accumulation_steps(
            effective_batch_size, micro_batch_size
        ),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=calibration["num_workers"],
        pin_memory=calibration["pin_memory"],
        persistent_workers=calibration["persistent_workers"],
        prefetch_factor=calibration["prefetch_factor"],
    )


def _optimizer_step(
    iterator: Iterator[dict[str, Any]],
    *,
    policy: Any,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    micro_batch_size: int,
    optimizer_step: int,
    accumulation_steps: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for accumulation_step in range(accumulation_steps):
        raw_batch = next(iterator)
        keys = _training_row_keys(
            raw_batch,
            micro_batch_size=micro_batch_size,
            optimizer_step=optimizer_step,
            accumulation_step=accumulation_step,
        )
        batch = preprocess_smolvla_batch(
            raw_batch, preprocessor, list(policy.config.image_features)
        )
        noise, flow_time = deterministic_flow_inputs(
            keys,
            action_shape=(spec["data"]["action_chunk_size"], policy.config.max_action_dim),
            noise_seed=spec["base_fit"]["batch_calibration"]["calibration_seed"],
            time_seed=spec["base_fit"]["batch_calibration"]["calibration_seed"] + 1,
            device=torch.device("cuda"),
        )
        loss = smolvla_flow_loss(policy, batch, noise, flow_time)
        (loss / accumulation_steps).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [value for value in policy.parameters() if value.requires_grad],
        spec["base_fit"]["gradient_clip_norm"],
    )
    if not torch.isfinite(grad_norm):
        raise GateZeroBatchCalibrationError("non-finite calibration gradient")
    optimizer.step()


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
    loader = _make_loader(
        dataset,
        micro_batch_size=micro_batch_size,
        effective_batch_size=base_fit["effective_batch_size"],
        optimizer_steps=calibration["technical_steps_per_candidate"],
        seed=calibration["calibration_seed"] + micro_batch_size,
        calibration=calibration,
    )
    iterator = iter(loader)
    measured_seconds: list[float] = []
    minimum_free_memory_mib: int | None = None
    try:
        for optimizer_step in range(calibration["technical_steps_per_candidate"]):
            if optimizer_step == calibration["warmup_optimizer_steps_per_candidate"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            _optimizer_step(
                iterator,
                policy=policy,
                preprocessor=preprocessor,
                optimizer=optimizer,
                spec=spec,
                micro_batch_size=micro_batch_size,
                optimizer_step=optimizer_step,
                accumulation_steps=accumulation_steps,
            )
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
    }


def _write_result(result: dict[str, Any], output_dir: Path, latest_link: Path) -> None:
    result_path = output_dir / "calibration_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum = f"{sha256_file(result_path)}  {result_path.name}\n"
    (output_dir / "checksums.sha256").write_text(checksum, encoding="utf-8")
    update_latest_link(output_dir, latest_link)


def _prepare_runtime(
    args: argparse.Namespace, spec: dict[str, Any], phase0: dict[str, Any]
) -> tuple[SourceHdf5Dataset, Any, Any, torch.optim.Optimizer]:
    if sha256_file(args.base_path / "model.safetensors") != spec["authority"]["model_weight_sha256"]:
        raise GateZeroBatchCalibrationError("base policy weight authority changed")
    authorities, demo_indices = load_surface_authorities(
        spec,
        phase0,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        surface=GateZeroSurface.BASE_FIT,
    )
    dataset = SourceHdf5Dataset(
        authorities,
        demo_indices=demo_indices,
        action_chunk_size=spec["data"]["action_chunk_size"],
        verify_sha256=True,
    )
    stats = load_source_normalization(
        args.normalization,
        expected_sha256=spec["authority"]["source_normalization_sha256"],
        expected_task_ids=phase0["splits"]["source"],
        expected_count=183555,
    )
    policy = load_smolvla_policy(args.base_path, args.vlm_path, spec)
    policy.train()
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    preprocessor, _ = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
    set_global_seed(spec["base_fit"]["batch_calibration"]["calibration_seed"])
    optimizer = torch.optim.AdamW(
        [value for value in policy.parameters() if value.requires_grad],
        lr=spec["base_fit"]["learning_rate"],
        betas=tuple(spec["base_fit"]["betas"]),
        eps=spec["base_fit"]["epsilon"],
        weight_decay=spec["base_fit"]["weight_decay"],
    )
    return dataset, policy, preprocessor, optimizer


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
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    trackio: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, micro_batch_size in enumerate(
        spec["base_fit"]["batch_calibration"]["micro_batch_candidates"]
    ):
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
        if record["status"] != "completed":
            break
    return records


def _build_result(
    spec: dict[str, Any], policy: Any, records: list[dict[str, Any]], *, started: float, run: str
) -> dict[str, Any]:
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
    dataset, policy, preprocessor, optimizer = _prepare_runtime(args, spec, phase0)
    trackio = _initialize_tracking(spec, args.output_dir.name)
    try:
        records = _calibrate_candidates(
            dataset,
            policy=policy,
            preprocessor=preprocessor,
            optimizer=optimizer,
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
