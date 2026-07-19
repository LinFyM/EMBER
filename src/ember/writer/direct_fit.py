"""Exact-resume action-supervised direct-LoRA validation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from ember.evaluation_identity import _load_policy
from ember.gate_zero_oracle_artifacts import (
    load_recovery_artifact,
    save_recovery_artifact,
    validate_recovery_artifact,
)
from ember.gate_zero_oracle_session import augment_support_images
from ember.gate_zero_runtime import (
    batch_provenance_keys,
    build_lora_config,
    preprocess_smolvla_batch,
    set_global_seed,
    smolvla_flow_loss,
)
from ember.writer.core import load_writer_contract, sha256_file
from ember.writer.data import WriterQueryDataset, WriterSpecAuthority, WriterTaskBatchSampler
from ember.writer.train import (
    _authorities,
    _base_owner,
    _lora_targets,
    _paths,
    repository_root,
)
from ember.writer.validation_contract import WriterValidationError, require


def load_source_teacher_contract(path: Path, *, repo_root: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WriterValidationError("source-teacher contract is unreadable") from error
    require(spec.get("schema_version"), 1, "source-teacher schema")
    require(
        spec.get("status"),
        "predeclared_source_teacher_fit_before_teacher_outcomes",
        "source-teacher predeclaration",
    )
    authority = spec["authority"]
    writer_path = repo_root / authority["writer_contract_relative_path"]
    split_path = repo_root / authority["split_reseal_relative_path"]
    require(sha256_file(writer_path), authority["writer_contract_sha256"], "Writer contract")
    require(sha256_file(split_path), authority["split_reseal_sha256"], "split reseal")
    split = json.loads(split_path.read_text(encoding="utf-8"))["active_split"]
    tasks = spec["teacher_task_ids"]
    if len(tasks) != len(set(tasks)) or not set(tasks) <= set(split["source"]):
        raise WriterValidationError("source-teacher tasks left the source split")
    if set(tasks) & (set(split["validation"]) | set(split["held_out"])):
        raise WriterValidationError("source-teacher tasks overlap evaluation")
    if {str(task) for task in tasks} != set(spec["teacher_categories"]):
        raise WriterValidationError("source-teacher categories are incomplete")
    require(authority["validation_numeric_access"], False, "validation access")
    require(authority["test_held_numeric_access"], False, "test/held access")
    return spec


def direct_final_path(root: Path) -> Path:
    return root / "final"


def _data_chain(previous: str, row_keys: Sequence[str]) -> str:
    digest = hashlib.sha256(bytes.fromhex(previous) if previous else b"")
    for key in row_keys:
        digest.update(key.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _scheduler(optimizer: torch.optim.Optimizer, direct: Mapping[str, Any]) -> Any:
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=direct["warmup_steps"],
        num_decay_steps=direct["decay_steps"],
        peak_lr=direct["learning_rate"],
        decay_lr=direct["decay_learning_rate"],
    ).build(optimizer, num_training_steps=direct["decay_steps"])


def _publish_final(
    root: Path,
    *,
    task_id: int,
    step: int,
    state: Mapping[str, torch.Tensor],
    contract_sha256: str,
    wall_seconds: float,
    final_loss: float,
    artifact_role: str,
) -> None:
    destination = direct_final_path(root)
    if destination.exists():
        return
    staging = root / f".final.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    state_path = staging / "trainable_state.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, state_path)
    manifest = {
        "schema_version": 1,
        "status": "fixed_step_matched_direct_task_local_lora",
        "artifact_role": artifact_role,
        "task_id": task_id,
        "step": step,
        "validation_contract_sha256": contract_sha256,
        "trainable_parameters": sum(value.numel() for value in state.values()),
        "trainable_tensors": len(state),
        "final_support_flow_loss": final_loss,
        "wall_seconds": wall_seconds,
        "state_sha256": sha256_file(state_path),
        "test_held_accessed": False,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, destination)


def _resume_metadata(task_root: Path) -> tuple[Path | None, int, str]:
    recovery = task_root / "recovery" / "last"
    if not recovery.is_symlink():
        return None, 0, ""
    recovery = recovery.resolve()
    manifest = validate_recovery_artifact(recovery)
    return recovery, int(manifest["step"]), manifest["authorities"]["data_chain"]


def _make_loader(
    dataset: WriterQueryDataset,
    *,
    task_id: int,
    direct: Mapping[str, Any],
    start_step: int,
) -> DataLoader:
    sampler = WriterTaskBatchSampler(
        dataset,
        task_ids=[task_id],
        per_rank_batch_size=direct["micro_batch_size"],
        start_step=start_step,
        stop_step=direct["optimizer_steps"],
        rank=0,
        world_size=1,
        seed=direct["seed"],
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


@dataclass
class DirectFitRuntime:
    runtime: list[Any]
    policy: torch.nn.Module
    trainable: dict[str, torch.Tensor]
    dataset: WriterQueryDataset
    optimizer: torch.optim.Optimizer
    scheduler: Any
    iterator: Any
    start_step: int
    data_chain: str


def _open_fit_runtime(
    *,
    direct: Mapping[str, Any],
    writer_spec: Mapping[str, Any],
    authority: WriterSpecAuthority,
    source_checkpoint: Path,
    mature_path: Path,
    task_root: Path,
) -> DirectFitRuntime:
    set_global_seed(direct["seed"])
    runtime = list(_load_policy(
        source_checkpoint / "pretrained_model",
        {"task_suite": "libero_90", "task_id": authority.task_id},
    ))
    lora = writer_spec["lora"]
    policy = runtime[0].wrap_with_peft(peft_config=build_lora_config(
        targets=_lora_targets(mature_path), rank=lora["rank"], alpha=lora["alpha"],
        dropout=lora["dropout"], init_lora_weights="gaussian",
        base_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    ))
    trainable = {name: value for name, value in policy.named_parameters() if value.requires_grad}
    require(sum(value.numel() for value in trainable.values()), lora["expected_parameter_count"], "direct capacity")
    bounds = direct["support_episode_bounds"]
    dataset = WriterQueryDataset(
        [authority], demo_indices=list(range(bounds[0], bounds[1] + 1)),
        action_chunk_size=writer_spec["data"]["functional_action_chunk_size"],
    )
    optimizer = torch.optim.AdamW(
        trainable.values(), lr=direct["learning_rate"], betas=tuple(direct["betas"]),
        eps=direct["epsilon"], weight_decay=direct["weight_decay"],
    )
    scheduler = _scheduler(optimizer, direct)
    recovery, start_step, data_chain = _resume_metadata(task_root)
    loader = _make_loader(dataset, task_id=authority.task_id, direct=direct, start_step=start_step)
    iterator = iter(loader)
    if recovery is not None:
        restored = load_recovery_artifact(
            recovery, model=policy, optimizer=optimizer, scheduler=scheduler,
            expected={"variant": "writer_validation_direct_lora", "task_id": authority.task_id},
        )
        require(restored, start_step, "direct resume step")
    return DirectFitRuntime(
        runtime, policy, trainable, dataset, optimizer, scheduler, iterator,
        start_step, data_chain,
    )


def _train_fit(runtime: DirectFitRuntime, *, direct: Mapping[str, Any], task_id: int,
               task_root: Path, contract_sha256: str) -> float:
    final_loss = float("nan")
    for step in range(runtime.start_step + 1, direct["optimizer_steps"] + 1):
        raw = next(runtime.iterator)
        keys = batch_provenance_keys(raw)
        runtime.data_chain = _data_chain(runtime.data_chain, keys)
        raw = augment_support_images(
            raw, row_keys=keys, optimizer_step=step, seed=direct["augmentation_seed"],
            scale_min=direct["augmentation_scale_min"], scale_max=direct["augmentation_scale_max"],
        )
        owner = _base_owner(runtime.policy)
        batch = preprocess_smolvla_batch(raw, runtime.runtime[1], list(owner.config.image_features))
        runtime.optimizer.zero_grad(set_to_none=True)
        loss = smolvla_flow_loss(runtime.policy, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(runtime.trainable.values(), direct["gradient_clip_norm"])
        if not torch.isfinite(norm):
            raise WriterValidationError("direct-LoRA gradient is non-finite")
        runtime.optimizer.step()
        runtime.scheduler.step()
        final_loss = float(loss.detach())
        if step == 1 or step % 50 == 0:
            print(json.dumps({
                "event": "writer_validation_direct_fit", "task_id": task_id,
                "step": step, "target_step": direct["optimizer_steps"],
                "support_flow_loss": final_loss,
                "learning_rate": runtime.optimizer.param_groups[0]["lr"],
            }, sort_keys=True), flush=True)
        if step in direct["checkpoint_steps"]:
            save_recovery_artifact(
                task_root, variant="writer_validation_direct_lora", task_id=task_id,
                step=step, trainable_state=runtime.trainable,
                optimizer=runtime.optimizer, scheduler=runtime.scheduler,
                authorities={
                    "validation_contract_sha256": contract_sha256,
                    "completed_step": step,
                    "consumed_query_frames": step * direct["micro_batch_size"],
                    "data_chain": runtime.data_chain,
                    "scaler_enabled": False,
                    "test_held_accessed": False,
                },
            )
    return final_loss


def fit_direct_lora(
    *,
    spec: Mapping[str, Any],
    writer_spec: Mapping[str, Any],
    authority: WriterSpecAuthority,
    source_checkpoint: Path,
    mature_path: Path,
    output_dir: Path,
    validation_contract_sha256: str,
) -> None:
    task_root = output_dir / "direct_lora" / f"task_{authority.task_id:03d}"
    final = direct_final_path(task_root)
    if final.is_dir():
        manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        require(manifest["step"], spec["direct_baseline"]["fixed_final_step"], "direct final step")
        require(manifest["validation_contract_sha256"], validation_contract_sha256, "direct contract")
        require(sha256_file(final / "trainable_state.safetensors"), manifest["state_sha256"], "direct state")
        return
    direct = spec["direct_baseline"]
    runtime = _open_fit_runtime(
        direct=direct, writer_spec=writer_spec, authority=authority,
        source_checkpoint=source_checkpoint, mature_path=mature_path, task_root=task_root,
    )
    started = time.perf_counter()
    try:
        final_loss = _train_fit(
            runtime, direct=direct, task_id=authority.task_id, task_root=task_root,
            contract_sha256=validation_contract_sha256,
        )
        _publish_final(
            task_root, task_id=authority.task_id, step=direct["fixed_final_step"],
            state=runtime.trainable,
            contract_sha256=validation_contract_sha256,
            wall_seconds=time.perf_counter() - started, final_loss=final_loss,
            artifact_role=spec["direct_baseline"]["role"],
        )
    finally:
        runtime.dataset.close()
        torch.cuda.empty_cache()


def run_source_teacher(args: argparse.Namespace) -> None:
    root = repository_root()
    spec = load_source_teacher_contract(args.config, repo_root=root)
    if args.task_id not in spec["teacher_task_ids"]:
        raise WriterValidationError("requested task is not a frozen source teacher")
    paths = _paths(root)
    writer_spec = load_writer_contract(
        root / spec["authority"]["writer_contract_relative_path"],
        phase0_path=paths["phase0"],
        split_path=paths["split"],
        gate_zero_path=paths["gate_zero"],
        mature_lora_path=paths["mature"],
    )
    with paths["phase0"].open("rb") as handle:
        phase0 = tomllib.load(handle)
    authorities = {
        authority.task_id: authority
        for authority in _authorities(
            writer_spec,
            phase0,
            manifest_path=args.output_root
            / spec["authority"]["canonical_manifest_relative_path"],
            dataset_root=args.data_root / spec["authority"]["dataset_relative_path"],
        )
    }
    source_checkpoint = (
        args.output_root
        / spec["authority"]["source_base_output_relative_path"]
        / "checkpoints"
        / f"{spec['authority']['source_base_checkpoint_step']:06d}"
    )
    require(
        sha256_file(source_checkpoint / "ember_checkpoint_manifest.json"),
        spec["authority"]["source_base_checkpoint_manifest_sha256"],
        "source-base checkpoint",
    )
    fit_direct_lora(
        spec=spec,
        writer_spec=writer_spec,
        authority=authorities[args.task_id],
        source_checkpoint=source_checkpoint,
        mature_path=paths["mature"],
        output_dir=args.output_dir,
        validation_contract_sha256=sha256_file(args.config),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    args = parser.parse_args()
    for name in ("config", "output_root", "data_root", "output_dir"):
        if not getattr(args, name).is_absolute():
            raise WriterValidationError(f"--{name.replace('_', '-')} must be absolute")
    run_source_teacher(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
