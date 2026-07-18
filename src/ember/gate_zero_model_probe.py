"""One-GPU SmolVLA/PEFT mechanics probe on a legal source support surface."""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
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
    build_lora_config,
    configure_smolvla,
    deterministic_flow_inputs,
    load_source_normalization,
    physical_lora_deltas,
    sha256_file,
)


class GateZeroModelProbeError(RuntimeError):
    """Raised when the model probe cannot preserve its pinned mechanics."""


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    trainable = [value for value in parameters if value.requires_grad]
    return {
        "total_parameters": sum(value.numel() for value in parameters),
        "trainable_parameters": sum(value.numel() for value in trainable),
        "trainable_tensors": len(trainable),
    }


def validate_output_destination(output_dir: Path) -> None:
    if not output_dir.is_absolute():
        raise GateZeroModelProbeError("output directory must be absolute")
    if (output_dir / "probe_result.json").exists():
        raise GateZeroModelProbeError("refusing to overwrite a completed probe")
    output_dir.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_policy(base_path: Path, vlm_path: Path, spec: dict[str, Any]):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_policy  # noqa: F401 - registers config choices
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(base_path)
    config = configure_smolvla(
        config,
        local_vlm_path=vlm_path,
        device="cuda",
        pretrained_path=base_path,
        pretrained_revision=spec["authority"]["model_revision"],
    )
    return SmolVLAPolicy.from_pretrained(
        base_path,
        config=config,
        local_files_only=True,
        strict=True,
    )


def _preprocess(batch: dict[str, Any], preprocessor: Any, image_keys: list[str]) -> dict[str, Any]:
    for key in image_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(dtype=torch.float32).div_(255.0)
    return preprocessor(batch)


def _batch_row_keys(batch: dict[str, Any]) -> list[str]:
    return [
        f"task{int(task)}/demo{int(demo)}/frame{int(frame)}"
        for task, demo, frame in zip(
            batch["task_id"], batch["demo_index"], batch["frame_index"], strict=True
        )
    ]


def _loss(
    model: Any,
    batch: dict[str, Any],
    noise: torch.Tensor,
    flow_time: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = model.forward(batch, noise=noise, time=flow_time)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise GateZeroModelProbeError("non-finite or non-scalar mechanics loss")
    return loss


def _delta_summary(deltas: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "frobenius_norm": float(torch.linalg.vector_norm(value)),
            "maximum_absolute_value": float(value.abs().max()),
            "finite": bool(torch.isfinite(value).all()),
        }
        for name, value in deltas.items()
    }


def _require_functional_zero(
    deltas: dict[str, torch.Tensor],
    base_loss: torch.Tensor,
    initial_loss: torch.Tensor,
) -> None:
    """Fail before training unless the adapter is an exact functional no-op."""

    if not deltas or any(torch.count_nonzero(value).item() != 0 for value in deltas.values()):
        raise GateZeroModelProbeError("adapter initialization has a nonzero physical delta")
    if not torch.equal(base_loss, initial_loss):
        raise GateZeroModelProbeError("adapter initialization changed the fixed loss")


def _save_adapter_bundle(
    model: Any,
    base_policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    output_dir: Path,
) -> Path:
    adapter_dir = output_dir / "adapter_roundtrip"
    model.save_pretrained(adapter_dir, safe_serialization=True, selected_adapters=["default"])
    base_policy.config._save_pretrained(adapter_dir)
    preprocessor.save_pretrained(adapter_dir)
    postprocessor.save_pretrained(adapter_dir)
    return adapter_dir


def _roundtrip_delta_check(
    adapter_dir: Path,
    base_path: Path,
    vlm_path: Path,
    spec: dict[str, Any],
    expected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    from peft import PeftConfig, PeftModel

    base = _load_policy(base_path, vlm_path, spec)
    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    loaded = PeftModel.from_pretrained(
        base,
        adapter_dir,
        config=adapter_config,
        is_trainable=False,
        autocast_adapter_dtype=True,
    )
    actual = physical_lora_deltas(loaded, spec["oracle"]["target_modules"])
    comparisons = {}
    for name in sorted(expected):
        delta = (actual[name] - expected[name]).abs()
        comparisons[name] = {
            "exact": bool(torch.equal(actual[name], expected[name])),
            "maximum_absolute_delta": float(delta.max()),
        }
    if not all(value["exact"] for value in comparisons.values()):
        raise GateZeroModelProbeError("adapter physical delta changed after round-trip")
    del loaded, base
    torch.cuda.empty_cache()
    return comparisons


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_destination(args.output_dir)
    started = time.perf_counter()
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    import trackio

    trackio.init(
        project=spec["tracking"]["project"],
        name=args.output_dir.name,
        group="model_mechanics",
        config={
            "mode": args.mode,
            "oracle_task_id": args.oracle_task_id,
            "micro_batch_size": args.micro_batch_size,
            "git_model_revision": spec["authority"]["model_revision"],
        },
        auto_log_gpu=spec["tracking"]["log_system_metrics"],
        gpu_log_interval=1.0,
        auto_log_cpu=spec["tracking"]["log_system_metrics"],
        cpu_log_interval=1.0,
    )
    if args.mode not in {"base", "adapter"}:
        raise GateZeroModelProbeError("probe mode must be base or adapter")
    if args.oracle_task_id not in spec["data"]["task_ids"]:
        raise GateZeroModelProbeError("probe task is not in the predeclared source pair")
    if sha256_file(args.base_path / "model.safetensors") != spec["authority"]["model_weight_sha256"]:
        raise GateZeroModelProbeError("base policy weight authority changed")
    authorities, demo_indices = load_surface_authorities(
        spec,
        phase0,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        surface=GateZeroSurface.SUPPORT,
        oracle_task_id=args.oracle_task_id,
    )
    dataset = SourceHdf5Dataset(
        authorities,
        demo_indices=demo_indices,
        action_chunk_size=spec["data"]["action_chunk_size"],
        verify_sha256=True,
    )
    sampler = TaskDemoFrameBatchSampler(
        dataset,
        micro_batch_size=args.micro_batch_size,
        optimizer_steps=1,
        gradient_accumulation_steps=1,
        seed=spec["oracle"]["seed"],
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    raw_batch = next(iter(loader))
    stats = load_source_normalization(
        args.normalization,
        expected_sha256=spec["authority"]["source_normalization_sha256"],
        expected_task_ids=phase0["splits"]["source"],
        expected_count=183555,
    )
    policy = _load_policy(args.base_path, args.vlm_path, spec)
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    preprocessor, postprocessor = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
    row_keys = _batch_row_keys(raw_batch)
    batch = _preprocess(raw_batch, preprocessor, list(policy.config.image_features))
    noise, flow_time = deterministic_flow_inputs(
        row_keys,
        action_shape=(spec["data"]["action_chunk_size"], policy.config.max_action_dim),
        noise_seed=spec["oracle"]["selection"]["fixed_noise_seed"],
        time_seed=spec["oracle"]["selection"]["fixed_time_seed"],
        device=torch.device("cuda"),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "mechanics_started",
        "mode": args.mode,
        "oracle_task_id": args.oracle_task_id,
        "micro_batch_size": args.micro_batch_size,
        "row_keys": row_keys,
        "base_weight_sha256": spec["authority"]["model_weight_sha256"],
        "manifest_sha256": spec["authority"]["canonical_manifest_sha256"],
        "normalization_sha256": spec["authority"]["source_normalization_sha256"],
        "peft_version": phase0["environment"]["peft"],
        "trackio_version": phase0["environment"]["trackio"],
    }
    if args.mode == "adapter":
        lora_config = build_lora_config(
            targets=spec["oracle"]["target_modules"],
            rank=spec["oracle"]["rank"],
            alpha=spec["oracle"]["alpha"],
            dropout=spec["oracle"]["dropout"],
            init_lora_weights=spec["oracle"]["init_lora_weights"],
            base_revision=spec["authority"]["model_revision"],
        )
        _set_seed(spec["oracle"]["seed"])
        model = policy.wrap_with_peft(peft_config=lora_config)
        actual_targets = set(model.base_model.targeted_module_names)
        if actual_targets != set(spec["oracle"]["target_modules"]):
            raise GateZeroModelProbeError("PEFT resolved a different target set")
        parameters = parameter_summary(model)
        if parameters["trainable_parameters"] != spec["oracle"]["expected_trainable_parameters"]:
            raise GateZeroModelProbeError("PEFT trainable parameter count changed")
        initial_delta = physical_lora_deltas(model, spec["oracle"]["target_modules"])
        with model.disable_adapter():
            base_loss = _loss(model, batch, noise, flow_time).detach()
        initial_loss = _loss(model, batch, noise, flow_time)
        _require_functional_zero(initial_delta, base_loss, initial_loss.detach())
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad],
            lr=spec["oracle"]["learning_rate"],
            betas=tuple(spec["oracle"]["betas"]),
            eps=spec["oracle"]["epsilon"],
            weight_decay=spec["oracle"]["weight_decay"],
        )
        optimizer.zero_grad(set_to_none=True)
        initial_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad],
            spec["oracle"]["gradient_clip_norm"],
        )
        if not torch.isfinite(grad_norm):
            raise GateZeroModelProbeError("adapter gradient norm is non-finite")
        optimizer.step()
        trained_delta = physical_lora_deltas(model, spec["oracle"]["target_modules"])
        adapter_dir = _save_adapter_bundle(model, policy, preprocessor, postprocessor, args.output_dir)
        roundtrip = _roundtrip_delta_check(
            adapter_dir, args.base_path, args.vlm_path, spec, trained_delta
        )
        result.update(
            {
                "parameter_summary": parameters,
                "resolved_targets": sorted(actual_targets),
                "base_fixed_loss": float(base_loss),
                "adapter_initial_fixed_loss": float(initial_loss.detach()),
                "initial_loss_absolute_delta": float((initial_loss.detach() - base_loss).abs()),
                "functional_zero_initialization": True,
                "gradient_norm": float(grad_norm),
                "initial_physical_deltas": _delta_summary(initial_delta),
                "trained_physical_deltas": _delta_summary(trained_delta),
                "roundtrip": roundtrip,
                "adapter_model_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
            }
        )
    else:
        parameters = parameter_summary(policy)
        optimizer = torch.optim.AdamW(
            [value for value in policy.parameters() if value.requires_grad],
            lr=spec["base_fit"]["learning_rate"],
            betas=tuple(spec["base_fit"]["betas"]),
            eps=spec["base_fit"]["epsilon"],
            weight_decay=spec["base_fit"]["weight_decay"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(policy, batch, noise, flow_time)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in policy.parameters() if value.requires_grad],
            spec["base_fit"]["gradient_clip_norm"],
        )
        if not torch.isfinite(grad_norm):
            raise GateZeroModelProbeError("base gradient norm is non-finite")
        optimizer.step()
        result.update(
            {
                "parameter_summary": parameters,
                "base_fixed_loss": float(loss.detach()),
                "gradient_norm": float(grad_norm),
            }
        )
    result.update(
        {
            "status": "mechanics_pass",
            "wall_seconds": time.perf_counter() - started,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() // (1024 * 1024),
            "gpu_count": 1,
            "scientific_gate_decision_authorized": False,
            "writer_authorized": False,
        }
    )
    tracked = {
        "mechanics/loss": result["base_fixed_loss"],
        "mechanics/gradient_norm": result["gradient_norm"],
        "resources/peak_allocated_mib": result["torch_peak_allocated_mib"],
        "resources/peak_reserved_mib": result["torch_peak_reserved_mib"],
        "resources/wall_seconds": result["wall_seconds"],
    }
    if "initial_loss_absolute_delta" in result:
        tracked["mechanics/initial_loss_absolute_delta"] = result[
            "initial_loss_absolute_delta"
        ]
    trackio.log(tracked, step=0)
    trackio.finish()
    result["tracking"] = {
        "backend": "trackio",
        "project": spec["tracking"]["project"],
        "run": args.output_dir.name,
        "dashboard_command": spec["tracking"]["dashboard_command"],
    }
    result_path = args.output_dir / "probe_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name not in {"checksums.sha256", "gpu_telemetry.csv"}:
            checksums.append(f"{sha256_file(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    update_latest_link(args.output_dir, args.latest_link)
    dataset.close()
    return result


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
    parser.add_argument("--mode", choices=("base", "adapter"), required=True)
    parser.add_argument("--oracle-task-id", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.absolute()
    args.latest_link = args.latest_link.absolute()
    result = run_probe(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
