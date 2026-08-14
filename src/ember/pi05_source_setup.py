"""Model, data, and distributed setup for PI05 source-base training."""

from __future__ import annotations

import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import (
    DistributedContext,
    Pi05SourceTrainingError,
    canonical_hash,
    read_json,
    sha256_file,
)
from ember.writer.data import WriterTaskAuthority
from ember.writer.topology import bind_current_process_to_cuda_numa, cuda_numa_node


REPO_ROOT = Path(__file__).resolve().parents[2]


def initialize_distributed(
    *,
    require_numa: bool = False,
    defer_process_group: bool = False,
) -> DistributedContext:
    if not torch.cuda.is_available():
        raise Pi05SourceTrainingError("PI05 source training requires CUDA")
    import os

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise Pi05SourceTrainingError("LOCAL_RANK is outside visible CUDA devices")
    torch.cuda.set_device(local_rank)
    hostname = os.uname().nodename.split(".", 1)[0].lower()
    if (
        world_size > 1
        and hostname in {"gpu01", "gpu02", "bci-gpu01", "bci-gpu02"}
        and torch.cuda.get_device_name(local_rank) == "NVIDIA A40"
        and os.environ.get("NCCL_P2P_DISABLE") != "1"
    ):
        raise Pi05SourceTrainingError(
            "BCI A40 multi-GPU requires explicit NCCL_P2P_DISABLE=1; "
            "NCCL 2.28 direct P2P/CUMEM is a reproduced transport hang"
        )
    numa_node = cuda_numa_node(local_rank)
    cpu_affinity = bind_current_process_to_cuda_numa(local_rank)
    if require_numa and (numa_node is None or not cpu_affinity):
        raise Pi05SourceTrainingError("formal PI05 training requires GPU-local NUMA affinity")
    if world_size > 1 and not defer_process_group:
        dist.init_process_group(backend="nccl")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        numa_node=numa_node,
        cpu_affinity=cpu_affinity,
    )


def initialize_deferred_process_group(
    context: DistributedContext,
    *,
    rendezvous_root: Path,
    collective_timeout: timedelta | None = None,
) -> None:
    """Rendezvous outside NCCL, then start NCCL at collective-ready state."""

    if context.world_size <= 1:
        return
    if dist.is_initialized():
        raise Pi05SourceTrainingError(
            "deferred NCCL process group was initialized before local CUDA setup"
        )
    import os
    import re
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    master_port = os.environ.get("MASTER_PORT", "")
    if not run_id or not master_port:
        raise Pi05SourceTrainingError(
            "deferred NCCL setup requires one torchrun rendezvous identity"
        )
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{run_id}-{master_port}")
    rendezvous_root.mkdir(parents=True, exist_ok=True)
    ready_path = rendezvous_root / f".rank-local-cuda-ready-{token}"
    store = dist.FileStore(str(ready_path), context.world_size)
    torch.cuda.synchronize(context.device)
    store.set(f"rank-{context.rank}", b"ready")
    store.wait(
        [f"rank-{rank}" for rank in range(context.world_size)],
        timedelta(minutes=30),
    )
    if collective_timeout is None:
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="nccl", timeout=collective_timeout)
    dist.barrier(device_ids=[context.local_rank])
    del store
    if context.is_main:
        ready_path.unlink(missing_ok=True)
    dist.barrier(device_ids=[context.local_rank])


def seed_everything(seed: int, context: DistributedContext) -> None:
    rank_seed = seed + context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ember_pi05_source_base_v1":
        raise Pi05SourceTrainingError("unsupported PI05 source-base config schema")
    if (
        config.get("features", {}).get("right_wrist_padding")
        != "zero_image_with_false_mask_via_missing_feature_key"
    ):
        raise Pi05SourceTrainingError("PI05 source base must mask the missing right wrist")
    for name in (
        "overlap_audit",
        "source_manifest",
        "normalization",
        "recipe",
        "model_manifest",
        "tokenizer_manifest",
    ):
        authority = config["authorities"][name]
        artifact = REPO_ROOT / authority["path"]
        if sha256_file(artifact) != authority["sha256"]:
            raise Pi05SourceTrainingError(f"sealed {name} authority changed")
    return config


def validate_runtime_assets(
    *,
    config: dict[str, Any],
    foundation_path: Path,
    tokenizer_path: Path,
    context: DistributedContext,
    verify_weight_hash: bool,
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            model = config["models"]["foundation"]
            tokenizer = config["models"]["tokenizer"]
            weights = foundation_path / "model.safetensors"
            model_config = foundation_path / "config.json"
            if (
                not weights.is_file()
                or weights.stat().st_size != int(model["weights_bytes"])
                or sha256_file(model_config) != model["config_sha256"]
            ):
                raise Pi05SourceTrainingError("foundation file size or config hash changed")
            if (
                not tokenizer_path.is_file()
                or tokenizer_path.stat().st_size != int(tokenizer["bytes"])
                or sha256_file(tokenizer_path) != tokenizer["sha256"]
            ):
                raise Pi05SourceTrainingError("OpenPI tokenizer identity changed")
            observed_weight_hash = (
                sha256_file(weights) if verify_weight_hash else "not_rehashed_in_smoke"
            )
            if verify_weight_hash and observed_weight_hash != model["weights_sha256"]:
                raise Pi05SourceTrainingError("foundation weight hash changed")
            payload[0] = {
                "foundation_config_sha256": model["config_sha256"],
                "foundation_weights_bytes": weights.stat().st_size,
                "foundation_weights_sha256": observed_weight_hash,
                "tokenizer_sha256": tokenizer["sha256"],
                "full_weight_hash_verified": verify_weight_hash,
            }
        except Exception as error:  # broadcast prevents peers hanging at a later barrier
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise Pi05SourceTrainingError(payload[0]["error"])
    return payload[0]


def load_authorities(
    config: dict[str, Any],
    data_root: Path,
    *,
    task_limit: int | None,
) -> tuple[list[WriterTaskAuthority], dict[str, Any]]:
    manifest_path = REPO_ROOT / config["authorities"]["source_manifest"]["path"]
    manifest = read_json(manifest_path)
    expected_ids = tuple(config["data"]["active_task_ids"])
    observed_ids = tuple(manifest["summary"]["active_source_task_ids"])
    if observed_ids != expected_ids or int(manifest["summary"]["active_tasks"]) != len(expected_ids):
        raise Pi05SourceTrainingError("source manifest task identities changed")
    records = sorted(manifest["tasks"], key=lambda row: int(row["task_index"]))
    if task_limit is not None:
        if not 1 <= task_limit <= len(records):
            raise Pi05SourceTrainingError("smoke task limit is outside the active corpus")
        records = records[:task_limit]
    authorities = []
    for record in records:
        hdf5 = record["hdf5"]
        authorities.append(
            WriterTaskAuthority(
                task_id=int(record["task_index"]),
                language=str(record["language"]),
                path=data_root / hdf5["filename"],
                expected_bytes=int(hdf5["bytes"]),
                expected_sha256=None,
            )
        )
    return authorities, manifest


def validate_source_files(
    *,
    authorities: Sequence[WriterTaskAuthority],
    manifest: dict[str, Any],
    context: DistributedContext,
    verify_hashes: bool,
) -> dict[str, Any]:
    """Verify the 52.7GB corpus once on rank zero rather than eight times."""

    payload: list[Any] = [None]
    if context.is_main:
        try:
            records = {int(row["task_index"]): row for row in manifest["tasks"]}
            for authority in authorities:
                expected = records[authority.task_id]["hdf5"]
                if (
                    not authority.path.is_file()
                    or authority.path.stat().st_size != int(expected["bytes"])
                ):
                    raise Pi05SourceTrainingError(
                        f"source HDF5 size changed: task {authority.task_id}"
                    )
                if verify_hashes and sha256_file(authority.path) != expected["sha256"]:
                    raise Pi05SourceTrainingError(
                        f"source HDF5 hash changed: task {authority.task_id}"
                    )
            payload[0] = {
                "tasks_checked": len(authorities),
                "bytes_checked": sum(item.expected_bytes for item in authorities),
                "full_sha256_verified": verify_hashes,
                "manifest_hdf5_aggregate_sha256": manifest["summary"][
                    "hdf5_aggregate_sha256"
                ],
            }
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise Pi05SourceTrainingError(payload[0]["error"])
    return payload[0]


def load_stats(config: dict[str, Any], task_ids: Sequence[int]) -> dict[str, Any]:
    path = REPO_ROOT / config["authorities"]["normalization"]["path"]
    normalization = read_json(path)
    authority = normalization.get("authority", {})
    expected_ids = config["data"]["active_task_ids"]
    if (
        authority.get("active_task_ids") != expected_ids
        or int(authority.get("episodes_per_task", 0)) != 50
        or int(authority.get("validation_or_test_numeric_reads", -1)) != 0
        or not set(task_ids) <= set(expected_ids)
    ):
        raise Pi05SourceTrainingError("normalization is not source-only authority")
    return normalization["stats"]


def policy_config(model_path: Path, config: dict[str, Any], device: torch.device) -> Any:
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION, OBS_STATE

    value = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(value, PI05Config):
        raise Pi05SourceTrainingError("foundation checkpoint is not a PI05 policy")
    features = config["features"]
    value.device = str(device)
    value.dtype = str(config["optimization"]["precision"])
    value.chunk_size = int(features["chunk_size"])
    value.n_action_steps = int(features["n_action_steps"])
    value.tokenizer_max_length = int(features["tokenizer_max_length"])
    value.freeze_vision_encoder = False
    value.train_expert_only = False
    value.gradient_checkpointing = bool(config["optimization"]["gradient_checkpointing"])
    value.compile_model = False
    value.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE, shape=(int(features["state_dim"]),)
    )
    value.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION, shape=(int(features["action_dim"]),)
    )
    return value


def load_policy(model_path: Path, config: dict[str, Any], device: torch.device) -> Any:
    """Strictly load PI05 without the upstream silent random-initialization fallback."""

    from lerobot.policies.pi05 import PI05Policy
    from safetensors.torch import load_file

    policy = PI05Policy(policy_config(model_path, config, device))
    weights = model_path / "model.safetensors"
    if not weights.is_file():
        raise Pi05SourceTrainingError(f"missing PI05 weights: {weights}")
    state = load_file(str(weights), device=str(device))
    if any(not name.startswith("model.") for name in state):
        state = policy._fix_pytorch_state_dict_keys(state, policy.config)
        state = {
            name if name.startswith("model.") else f"model.{name}": tensor
            for name, tensor in state.items()
        }
    missing, unexpected = policy.load_state_dict(state, strict=False)
    del state
    if missing or unexpected:
        raise Pi05SourceTrainingError(
            f"PI05 strict load failed: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    policy.train()
    return policy


def trainable_contract(policy: torch.nn.Module) -> dict[str, Any]:
    names = sorted(name for name, value in policy.named_parameters() if value.requires_grad)
    frozen = sorted(name for name, value in policy.named_parameters() if not value.requires_grad)
    if not names or frozen:
        raise Pi05SourceTrainingError(
            f"source base must full-finetune PI05; unexpectedly frozen: {frozen[:5]}"
        )
    dtypes: dict[str, int] = {}
    for parameter in policy.parameters():
        key = str(parameter.dtype)
        dtypes[key] = dtypes.get(key, 0) + parameter.numel()
    return {
        "method": "full_action_sft",
        "trainable_parameter_count": sum(value.numel() for value in policy.parameters()),
        "parameter_name_count": len(names),
        "parameter_names_sha256": canonical_hash(names),
        "dtype_parameter_counts": dtypes,
        "frozen_parameter_count": 0,
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, peak_lr: float
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps <= 0:
        raise Pi05SourceTrainingError("official PI05 schedule needs positive warmup")

    def factor(step: int) -> float:
        return min((step + 1) / (warmup_steps + 1), 1.0)

    for group in optimizer.param_groups:
        group["lr"] = peak_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def update_ema(ema_policy: torch.nn.Module, policy: torch.nn.Module, decay: float) -> None:
    ema_parameters = tuple(ema_policy.parameters())
    parameters = tuple(policy.parameters())
    if len(ema_parameters) != len(parameters):
        raise Pi05SourceTrainingError("EMA and train policy topologies differ")
    torch._foreach_mul_(ema_parameters, decay)
    torch._foreach_add_(ema_parameters, parameters, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_policy.buffers(), policy.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def reduce_mean(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / context.world_size)


def reduce_max(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())
