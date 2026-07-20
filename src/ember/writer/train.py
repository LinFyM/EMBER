"""Eight-rank functional training for the direct complete-LoRA Writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from ember.evaluation_identity import _load_policy
from ember.gate_zero_oracle_session import augment_support_images
from ember.gate_zero_runtime import (
    batch_provenance_keys,
    build_lora_config,
    deterministic_flow_inputs,
    preprocess_smolvla_batch,
    set_global_seed,
)
from ember.writer.core import (
    CompleteLoRAWriter,
    WriterColdStartError,
    build_lora_tensor_specs,
    cosine_warmup_scheduler,
    load_physical_update_teachers,
    load_writer_checkpoint,
    physical_lora_delta_squared_distance,
    physical_lora_delta_l2,
    load_writer_contract,
    save_writer_checkpoint,
    sha256_file,
)
from ember.writer.topology import bind_current_process_to_cuda_numa
from ember.writer.data import (
    WriterQueryDataset,
    WriterSpecAuthority,
    WriterTaskBatchSampler,
    iter_action_hidden_video_chunks,
)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "phase0": root / "configs/phase0.toml",
        "split": root / "configs/libero90_split_reseal.json",
        "gate_zero": root / "configs/gate_zero_oracle_pilot.toml",
        "mature": root / "configs/gate_zero_mature_lora_positive_control.toml",
    }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _init_distributed(spec: dict[str, Any]) -> tuple[int, int, int]:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise WriterColdStartError("invalid torchrun rank environment") from error
    if world_size != spec["train"]["world_size"] or not 0 <= rank < world_size:
        raise WriterColdStartError("Writer launch must use the frozen eight-rank topology")
    torch.cuda.set_device(local_rank)
    bind_current_process_to_cuda_numa(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", init_method="env://")
    return rank, local_rank, world_size


def _barrier() -> None:
    torch.distributed.barrier()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WriterColdStartError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise WriterColdStartError(f"invalid {label}: {path}")
    return value


def _authorities(
    spec: dict[str, Any], phase0: dict[str, Any], *, manifest_path: Path, dataset_root: Path
) -> list[WriterSpecAuthority]:
    if sha256_file(manifest_path) != spec["authority"]["canonical_manifest_sha256"]:
        raise WriterColdStartError("canonical data manifest changed")
    manifest = _load_json(manifest_path, "canonical data manifest")
    by_id = {row.get("task_index"): row for row in manifest.get("tasks", [])}
    result = []
    for task_id in phase0["splits"]["source"]:
        row = by_id.get(task_id)
        if not isinstance(row, dict) or row.get("split") != "source":
            raise WriterColdStartError(f"source task {task_id} left the permanent split")
        hdf5 = row.get("hdf5", {})
        result.append(
            WriterSpecAuthority(
                task_id,
                row["language"],
                dataset_root / hdf5["filename"],
                hdf5["bytes"],
                hdf5.get("sha256"),
            )
        )
    return result


def _lora_targets(mature_path: Path) -> list[str]:
    with mature_path.open("rb") as handle:
        mature = tomllib.load(handle)
    return list(mature["fit"]["mature_official_default_r32"]["target_modules"])


def _open_frozen_policy(
    spec: dict[str, Any], mature_path: Path, source_checkpoint: Path, task_id: int
) -> tuple[Any, Any, dict[str, torch.Tensor]]:
    runtime = _load_policy(
        source_checkpoint / "pretrained_model", {"task_suite": "libero_90", "task_id": task_id}
    )
    policy, preprocessor = runtime[0], runtime[1]
    targets = _lora_targets(mature_path)
    lora = spec["lora"]
    peft_config = build_lora_config(
        targets=targets,
        rank=lora["rank"],
        alpha=lora["alpha"],
        dropout=lora["dropout"],
        init_lora_weights="gaussian",
        base_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    )
    policy = policy.wrap_with_peft(peft_config=peft_config)
    if sorted(policy.base_model.targeted_module_names) != sorted(targets):
        raise WriterColdStartError("PEFT resolved a different Writer target set")
    template = {
        name: value.detach().to(torch.float32).cpu().contiguous()
        for name, value in policy.named_parameters()
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name
    }
    if sum(value.numel() for value in template.values()) != lora["expected_parameter_count"]:
        raise WriterColdStartError("Writer LoRA template parameter count changed")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    return policy, preprocessor, template


def _base_owner(policy: Any) -> Any:
    return policy.get_base_model() if hasattr(policy, "get_base_model") else policy


def prepare_writer_images(owner: Any, images: torch.Tensor) -> torch.Tensor:
    prepared, _ = owner.prepare_images({"observation.images.camera1": images})
    return prepared[0]


@torch.inference_mode()
def _encode_spec_features(
    policy: Any,
    authority: WriterSpecAuthority,
    *,
    demo_indices: list[int],
    encode_batch_size: int,
) -> dict[str, torch.Tensor]:
    """Encode all frames and full language while preserving episode boundaries."""

    device = next(policy.parameters()).device
    owner = _base_owner(policy)
    video_chunks: list[torch.Tensor] = []
    episode_offsets = [0]
    active_demo: int | None = None
    active_frames = 0
    active_length = 0
    for demo_index, start, episode_length, frames in iter_action_hidden_video_chunks(
        authority, demo_indices, chunk_size=encode_batch_size
    ):
        if active_demo != demo_index:
            if active_demo is not None:
                if active_frames != active_length:
                    raise WriterColdStartError("action-hidden episode encoding was incomplete")
                episode_offsets.append(episode_offsets[-1] + active_frames)
            active_demo = demo_index
            active_frames = 0
            active_length = episode_length
        if start != active_frames or episode_length != active_length:
            raise WriterColdStartError("action-hidden episode order changed")
        images = torch.from_numpy(frames).to(device=device, dtype=torch.float32).div_(255.0)
        prepared = prepare_writer_images(owner, images)
        visual_tokens = owner.model.vlm_with_expert.embed_image(prepared)
        video_chunks.append(visual_tokens.mean(dim=1).to(torch.float16).cpu())
        active_frames += frames.shape[0]
    if active_demo is None or active_frames != active_length:
        raise WriterColdStartError("action-hidden video set is empty or incomplete")
    episode_offsets.append(episode_offsets[-1] + active_frames)

    tokenizer = owner.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer(authority.language, return_tensors="pt", truncation=False)
    token_ids = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device=device, dtype=torch.bool)
    language_tokens = owner.model.vlm_with_expert.embed_language_tokens(token_ids)[0, mask[0]]
    return {
        "language_tokens": language_tokens.to(torch.float16).cpu().contiguous(),
        "video_features": torch.cat(video_chunks).contiguous(),
        "episode_offsets": torch.tensor(episode_offsets, dtype=torch.int64),
    }


def _load_task_input(path: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    tensors = load_file(path)
    required = {"language_tokens", "video_features", "episode_offsets"}
    if set(tensors) != required:
        raise WriterColdStartError("cached Writer task input is incomplete")
    return (
        tensors["language_tokens"].to(device=device, dtype=torch.float32, non_blocking=True),
        tensors["video_features"].to(device=device, dtype=torch.float32, non_blocking=True),
        tensors["episode_offsets"],
    )


def _feature_cache(
    spec: dict[str, Any],
    policy: Any,
    authorities: list[WriterSpecAuthority],
    output_dir: Path,
    *,
    rank: int,
    world_size: int,
) -> Path:
    cache = output_dir / "writer_spec_features"
    cache.mkdir(parents=True, exist_ok=True)
    bounds = spec["data"]["writer_spec_episode_bounds"]
    demos = list(range(bounds[0], bounds[1] + 1))
    for authority in authorities[rank::world_size]:
        path = cache / f"task_{authority.task_id:03d}.safetensors"
        if not path.exists():
            features = _encode_spec_features(
                policy,
                authority,
                demo_indices=demos,
                encode_batch_size=spec["writer"]["vision_encode_batch_size"],
            )
            if (
                features["video_features"].shape[1] != spec["writer"]["vision_feature_dim"]
                or features["language_tokens"].shape[1]
                != spec["writer"]["language_feature_dim"]
                or not all(torch.isfinite(value).all() for value in features.values())
            ):
                raise WriterColdStartError("frozen Writer sequence features have wrong values")
            temporary = cache / f".{path.name}.tmp-rank{rank}"
            save_file(features, temporary)
            os.replace(temporary, path)
    _barrier()
    manifest_path = cache / "feature_manifest.json"
    if rank == 0:
        records = {}
        for authority in authorities:
            path = cache / f"task_{authority.task_id:03d}.safetensors"
            features = load_file(path)
            offsets = features["episode_offsets"].tolist()
            if (
                features["video_features"].shape[1] != spec["writer"]["vision_feature_dim"]
                or features["language_tokens"].shape[1]
                != spec["writer"]["language_feature_dim"]
                or offsets[-1] != features["video_features"].shape[0]
            ):
                raise WriterColdStartError("cached Writer sequence feature shape changed")
            records[str(authority.task_id)] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "language": authority.language,
                "episode_count": len(offsets) - 1,
                "episode_lengths": [right - left for left, right in zip(offsets, offsets[1:])],
                "frame_count": offsets[-1],
                "visible_fields": spec["data"]["writer_visible_fields"],
            }
        temporary = cache / ".feature_manifest.json.tmp"
        temporary.write_text(
            json.dumps({"schema_version": 1, "tasks": records}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    _barrier()
    return cache


def _writer_model(
    spec: dict[str, Any], template: dict[str, torch.Tensor], device: torch.device
) -> CompleteLoRAWriter:
    writer = spec["writer"]
    return CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        vision_feature_dim=writer["vision_feature_dim"],
        language_feature_dim=writer["language_feature_dim"],
        hidden_dim=writer["hidden_dim"],
        attention_heads=writer["attention_heads"],
        temporal_chunk_size=writer["temporal_chunk_size"],
        chunk_memory_tokens=writer["chunk_memory_tokens"],
        episode_memory_tokens=writer["episode_memory_tokens"],
        task_memory_tokens=writer["task_memory_tokens"],
        decoder_hidden_dim=writer["decoder_hidden_dim"],
    ).to(device)


def _data_chain(previous: str, row_keys: list[str]) -> str:
    digest = hashlib.sha256(bytes.fromhex(previous) if previous else b"")
    for key in row_keys:
        digest.update(key.encode("utf-8") + b"\0")
    return digest.hexdigest()


def _gather_objects(value: Any, *, rank: int, world_size: int) -> list[Any] | None:
    gathered = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(value, gathered, dst=0)
    return gathered


def _gather_rng(rank: int, world_size: int) -> list[dict[str, torch.Tensor]] | None:
    from lerobot.utils.random_utils import serialize_rng_state

    return _gather_objects(serialize_rng_state(), rank=rank, world_size=world_size)


def _checkpoint(
    args: argparse.Namespace,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    step: int,
    data_chain: str,
    rank: int,
    world_size: int,
    spec_sha256: str,
    train: dict[str, Any],
) -> None:
    chains = _gather_objects(data_chain, rank=rank, world_size=world_size)
    rng_states = _gather_rng(rank, world_size)
    if rank == 0:
        owner = writer.module if hasattr(writer, "module") else writer
        save_writer_checkpoint(
            args.output_dir / "checkpoints" / f"{step:06d}",
            step=step,
            writer=owner,
            optimizer=optimizer,
            scheduler=scheduler,
            rank_rng_states=rng_states,
            metadata={
                "world_size": world_size,
                "per_rank_micro_batch_size": train["per_rank_micro_batch_size"],
                "global_batch_size": train["global_batch_size"],
                "authority": spec_sha256,
                "sampler": {
                    "completed_step": step,
                    "consumed_query_frames": step * train["global_batch_size"],
                    "rank_data_chain": chains,
                },
            },
        )
    _barrier()


def _tracking(spec: dict[str, Any], args: argparse.Namespace, rank: int) -> Any:
    if rank != 0 or args.mode == "smoke":
        return None
    import trackio

    trackio.init(
        project="EMBER_writer",
        name=args.output_dir.name,
        group="writer_cold_start",
        config={
            "world_size": spec["train"]["world_size"],
            "global_batch_size": spec["train"]["global_batch_size"],
            "lora_parameters": spec["lora"]["expected_parameter_count"],
            "source_tasks": len(spec["data"].get("functional_training_task_ids", [])) or 60,
            "teacher_physical_update_auxiliary": "teacher_auxiliary" in spec,
        },
        auto_log_gpu=True,
        gpu_log_interval=1.0,
        auto_log_cpu=True,
        cpu_log_interval=1.0,
    )
    return trackio


def _write_stage_result(
    args: argparse.Namespace,
    *,
    step: int,
    elapsed: float,
    final_metrics: dict[str, float],
    spec_sha256: str,
) -> None:
    result = {
        "schema_version": 1,
        "status": "writer_cold_start_training_segment_completed_pending_validation",
        "completed_step": step,
        "wall_seconds": elapsed,
        "final_global_functional_loss": final_metrics["global_functional_loss"],
        "final_global_objective_loss": final_metrics["global_objective_loss"],
        "final_global_physical_delta_l2": final_metrics["global_physical_delta_l2"],
        "final_global_teacher_relative_physical_delta_squared_error": final_metrics[
            "global_teacher_relative_physical_delta_squared_error"
        ],
        "writer_contract_sha256": spec_sha256,
        "validation_performance_accessed": False,
        "test_held_accessed": False,
    }
    path = args.output_dir / "writer_cold_start_stage_result.json"
    temporary = args.output_dir / ".writer_cold_start_stage_result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    (args.output_dir / "checksums.sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


@dataclass
class TrainRuntime:
    args: argparse.Namespace
    spec: dict[str, Any]
    rank: int
    world_size: int
    device: torch.device
    policy: Any
    preprocessor: Any
    template: dict[str, torch.Tensor]
    teacher_states: dict[int, dict[str, torch.Tensor]]
    teacher_norm_squares: dict[int, float]
    feature_cache: Path
    writer: torch.nn.Module
    writer_owner: CompleteLoRAWriter
    optimizer: torch.optim.Optimizer
    scheduler: Any
    dataset: WriterQueryDataset
    iterator: Any
    tracker: Any
    start_step: int
    target_step: int
    data_chain: str
    task_inputs: dict[int, tuple[torch.Tensor, ...]]


def _functional_query_loader(
    spec: dict[str, Any],
    phase0: dict[str, Any],
    authorities: list[WriterSpecAuthority],
    *,
    start_step: int,
    target_step: int,
    rank: int,
    world_size: int,
) -> tuple[WriterQueryDataset, Any]:
    bounds = spec["data"]["functional_train_episode_bounds"]
    training_task_ids = spec["data"].get(
        "functional_training_task_ids", phase0["splits"]["source"]
    )
    training_task_set = set(training_task_ids)
    training_authorities = [
        authority for authority in authorities if authority.task_id in training_task_set
    ]
    if len(training_authorities) != len(training_task_ids):
        raise WriterColdStartError("Writer functional training task authority changed")
    dataset = WriterQueryDataset(
        training_authorities,
        demo_indices=list(range(bounds[0], bounds[1] + 1)),
        action_chunk_size=spec["data"]["functional_action_chunk_size"],
    )
    train = spec["train"]
    sampler = WriterTaskBatchSampler(
        dataset,
        task_ids=training_task_ids,
        per_rank_batch_size=train["per_rank_micro_batch_size"],
        start_step=start_step,
        stop_step=target_step,
        rank=rank,
        world_size=world_size,
        seed=train["seed"],
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=train["num_workers_per_rank"],
        pin_memory=True,
        persistent_workers=train["num_workers_per_rank"] > 0,
        prefetch_factor=2 if train["num_workers_per_rank"] > 0 else None,
    )
    return dataset, iter(loader)


def _setup_runtime(args: argparse.Namespace) -> TrainRuntime:
    root = repository_root()
    paths = _paths(root)
    spec = load_writer_contract(
        args.config,
        phase0_path=paths["phase0"],
        split_path=paths["split"],
        gate_zero_path=paths["gate_zero"],
        mature_lora_path=paths["mature"],
    )
    rank, local_rank, world_size = _init_distributed(spec)
    train = spec["train"]
    device = torch.device("cuda", local_rank)
    set_global_seed(train["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths["phase0"].open("rb") as handle:
        phase0 = tomllib.load(handle)
    manifest = args.output_root / spec["authority"]["canonical_manifest_relative_path"]
    dataset_root = args.data_root / spec["authority"]["dataset_relative_path"]
    source_checkpoint = (
        args.output_root
        / spec["authority"]["source_base_output_relative_path"]
        / "checkpoints"
        / f"{spec['authority']['source_base_checkpoint_step']:06d}"
    )
    if sha256_file(source_checkpoint / "ember_checkpoint_manifest.json") != spec["authority"]["source_base_checkpoint_manifest_sha256"]:
        raise WriterColdStartError("source-base checkpoint manifest changed")
    authorities = _authorities(
        spec, phase0, manifest_path=manifest, dataset_root=dataset_root
    )
    policy, preprocessor, template = _open_frozen_policy(
        spec, paths["mature"], source_checkpoint, authorities[rank].task_id
    )
    teacher_states, teacher_norm_squares = load_physical_update_teachers(
        spec, output_root=args.output_root, template=template, device=device
    )
    training_task_ids = set(
        spec["data"].get("functional_training_task_ids", phase0["splits"]["source"])
    )
    cache_authorities = [
        authority for authority in authorities if authority.task_id in training_task_ids
    ]
    if len(cache_authorities) != len(training_task_ids):
        raise WriterColdStartError("Writer video cache task authority changed")
    feature_cache = _feature_cache(
        spec,
        policy,
        cache_authorities,
        args.output_root / spec["authority"]["feature_cache_relative_path"],
        rank=rank,
        world_size=world_size,
    )
    writer_owner = _writer_model(spec, template, device)
    writer: torch.nn.Module = torch.nn.parallel.DistributedDataParallel(
        writer_owner,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=train["learning_rate"],
        betas=tuple(train["betas"]),
        eps=train["epsilon"],
        weight_decay=train["weight_decay"],
    )
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_steps=train["warmup_steps"],
        total_steps=train["maximum_steps"],
        minimum_ratio=train["minimum_learning_rate"] / train["learning_rate"],
    )
    start_step = int(args.resume_checkpoint.name) if args.resume_checkpoint else 0
    target_step = args.stop_after_step or train["first_segment_steps"]
    if not start_step < target_step <= train["maximum_steps"]:
        raise WriterColdStartError("Writer target step is outside the resumable ladder")
    dataset, iterator = _functional_query_loader(
        spec,
        phase0,
        authorities,
        start_step=start_step,
        target_step=target_step,
        rank=rank,
        world_size=world_size,
    )
    data_chain = ""
    if args.resume_checkpoint:
        restored_step, data_chain = load_writer_checkpoint(
            args.resume_checkpoint,
            writer=writer_owner,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            world_size=world_size,
            expected_authority=sha256_file(args.config),
        )
        if restored_step != start_step:
            raise WriterColdStartError("resume checkpoint name and payload differ")
    tracker = _tracking(spec, args, rank)
    policy_named = dict(policy.named_parameters())
    if set(template) - set(policy_named):
        raise WriterColdStartError("functional LoRA parameter names changed")
    return TrainRuntime(
        args, spec, rank, world_size, device, policy, preprocessor, template,
        teacher_states, teacher_norm_squares,
        feature_cache, writer, writer_owner, optimizer, scheduler, dataset,
        iterator, tracker, start_step, target_step, data_chain, {},
    )


def _functional_step(runtime: TrainRuntime, step: int) -> dict[str, float]:
        step_started = time.perf_counter()
        raw_batch = next(runtime.iterator)
        task_ids = {int(value) for value in raw_batch["task_id"]}
        if len(task_ids) != 1:
            raise WriterColdStartError("one functional batch mixed task-specific LoRAs")
        task_id = next(iter(task_ids))
        row_keys = batch_provenance_keys(raw_batch)
        runtime.data_chain = _data_chain(runtime.data_chain, row_keys)
        train = runtime.spec["train"]
        raw_batch = augment_support_images(
            raw_batch,
            row_keys=row_keys,
            optimizer_step=step,
            seed=train["seed"] + 11,
            scale_min=0.9,
            scale_max=1.0,
        )
        owner = _base_owner(runtime.policy)
        batch = preprocess_smolvla_batch(
            raw_batch, runtime.preprocessor, list(owner.config.image_features)
        )
        noise, flow_time = deterministic_flow_inputs(
            row_keys,
            action_shape=(runtime.spec["data"]["functional_action_chunk_size"], owner.config.max_action_dim),
            noise_seed=train["flow_noise_seed"],
            time_seed=train["flow_time_seed"],
            device=runtime.device,
        )
        if task_id not in runtime.task_inputs:
            runtime.task_inputs[task_id] = _load_task_input(
                runtime.feature_cache / f"task_{task_id:03d}.safetensors", runtime.device
            )
        task_input = runtime.task_inputs[task_id]
        runtime.optimizer.zero_grad(set_to_none=True)
        generated = {key: value[0] for key, value in runtime.writer(*task_input).items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            functional_loss, _ = torch.func.functional_call(
                runtime.policy,
                generated,
                (batch,),
                {"noise": noise, "time": flow_time},
                strict=False,
            )
        physical_delta_l2 = physical_lora_delta_l2(
            generated,
            alpha=runtime.spec["lora"]["alpha"],
            rank=runtime.spec["lora"]["rank"],
        )
        soft_cap = float(train.get("physical_delta_l2_soft_cap", float("inf")))
        coefficient = float(train.get("physical_delta_excess_coefficient", 0.0))
        excess = torch.relu(physical_delta_l2 - soft_cap)
        teacher_relative_error = torch.zeros_like(physical_delta_l2)
        teacher_coefficient = 0.0
        if runtime.teacher_states:
            if task_id not in runtime.teacher_states:
                raise WriterColdStartError("functional task lacks a physical-update teacher")
            teacher_distance = physical_lora_delta_squared_distance(
                generated,
                runtime.teacher_states[task_id],
                alpha=runtime.spec["lora"]["alpha"],
                rank=runtime.spec["lora"]["rank"],
            )
            teacher_relative_error = teacher_distance / max(
                runtime.teacher_norm_squares[task_id], 1e-12
            )
            teacher_coefficient = float(
                runtime.spec["teacher_auxiliary"][
                    "relative_physical_delta_squared_error_coefficient"
                ]
            )
        loss = (
            functional_loss
            + coefficient * excess.square()
            + teacher_coefficient * teacher_relative_error
        )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise WriterColdStartError("Writer functional loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            runtime.writer.parameters(), train["gradient_clip_norm"]
        )
        if not torch.isfinite(gradient_norm):
            raise WriterColdStartError("Writer gradient is non-finite")
        runtime.optimizer.step()
        runtime.scheduler.step()
        reduced = torch.stack(
            (
                functional_loss.detach(),
                loss.detach(),
                physical_delta_l2.detach(),
                teacher_relative_error.detach(),
            )
        )
        torch.distributed.all_reduce(reduced)
        reduced.div_(runtime.world_size)
        return {
            "global_functional_loss": float(reduced[0]),
            "global_objective_loss": float(reduced[1]),
            "global_physical_delta_l2": float(reduced[2]),
            "global_teacher_relative_physical_delta_squared_error": float(reduced[3]),
            "gradient_norm": float(gradient_norm),
            "learning_rate": runtime.scheduler.get_last_lr()[0],
            "global_samples_per_second": train["global_batch_size"]
            / (time.perf_counter() - step_started),
            "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        }


def _log_step(runtime: TrainRuntime, step: int, metrics: dict[str, float]) -> None:
    if runtime.rank == 0 and (
        step == 1 or step % 10 == 0 or step == runtime.target_step
    ):
            record = {
                "event": "writer_cold_start_progress",
                "step": step,
                "target_step": runtime.target_step,
                **metrics,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            if runtime.tracker is not None:
                runtime.tracker.log(
                    {
                        f"writer/{key}": value
                        for key, value in record.items()
                        if isinstance(value, (int, float))
                    },
                    step=step,
                )


def _train_loop(runtime: TrainRuntime) -> tuple[dict[str, float], float]:
    started = time.perf_counter()
    final_metrics: dict[str, float] | None = None
    train = runtime.spec["train"]
    for step in range(runtime.start_step + 1, runtime.target_step + 1):
        metrics = _functional_step(runtime, step)
        final_metrics = metrics
        _log_step(runtime, step, metrics)
        if runtime.args.mode == "train" and step in train["checkpoint_steps"]:
            _checkpoint(
                runtime.args,
                runtime.writer,
                runtime.optimizer,
                runtime.scheduler,
                step=step,
                data_chain=runtime.data_chain,
                rank=runtime.rank,
                world_size=runtime.world_size,
                spec_sha256=sha256_file(runtime.args.config),
                train=train,
            )
    if final_metrics is None:
        raise WriterColdStartError("Writer training segment completed no steps")
    return final_metrics, time.perf_counter() - started


def _finish_runtime(
    runtime: TrainRuntime, *, final_metrics: dict[str, float], elapsed: float
) -> None:
    _barrier()
    if runtime.rank == 0:
        if runtime.args.mode == "smoke":
            smoke = {
                "status": "mechanical_smoke_passed_no_performance_outcome",
                "world_size": runtime.world_size,
                "completed_steps": runtime.target_step - runtime.start_step,
                "base_trainable_parameters": sum(
                    p.numel() for p in runtime.policy.parameters() if p.requires_grad
                ),
                "writer_gradient_finite": True,
                "test_held_accessed": False,
            }
            (runtime.args.output_dir / "mechanical_smoke.json").write_text(
                json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            _write_stage_result(
                runtime.args,
                step=runtime.target_step,
                elapsed=elapsed,
                final_metrics=final_metrics,
                spec_sha256=sha256_file(runtime.args.config),
            )
    if runtime.tracker is not None:
        runtime.tracker.finish()
    runtime.dataset.close()
    _barrier()
    torch.distributed.destroy_process_group()


def run(args: argparse.Namespace) -> None:
    runtime = _setup_runtime(args)
    final_metrics, elapsed = _train_loop(runtime)
    _finish_runtime(runtime, final_metrics=final_metrics, elapsed=elapsed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    for name in ("config", "output_root", "data_root", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            raise WriterColdStartError(f"--{name.replace('_', '-')} must be absolute")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_absolute():
        raise WriterColdStartError("--resume-checkpoint must be absolute")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
