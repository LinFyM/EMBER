"""Direct complete-LoRA Writer contract, model, and resumable state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file


class WriterColdStartError(RuntimeError):
    """Raised when the direct-Writer contract or state is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise WriterColdStartError(f"{label} changed: {actual!r} != {expected!r}")


def _load_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WriterColdStartError(f"invalid {label}: {path}") from error
    return value


def _validate_teacher_auxiliary(spec: dict[str, Any], phase0: dict[str, Any]) -> None:
    teacher = spec.get("teacher_auxiliary")
    if teacher is None:
        return
    tasks = teacher.get("task_ids")
    if (
        not isinstance(tasks, list)
        or len(tasks) != len(set(tasks))
        or not set(tasks) <= set(phase0["splits"]["source"])
        or spec.get("data", {}).get("functional_training_task_ids") != tasks
    ):
        raise WriterColdStartError("physical-update teacher task authority changed")
    categories = teacher.get("task_categories", {})
    if set(categories) != {str(task) for task in tasks} or len(set(categories.values())) < 2:
        raise WriterColdStartError("physical-update teacher categories changed")
    relative = Path(str(teacher.get("bundle_relative_path", "")))
    coefficient = teacher.get("relative_physical_delta_squared_error_coefficient")
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or not isinstance(teacher.get("bundle_sha256"), str)
        or len(teacher["bundle_sha256"]) != 64
        or coefficient is None
        or not 0 < coefficient <= 1
        or teacher.get("raw_factor_mse") is not False
        or teacher.get("validation_numeric_access") is not False
        or teacher.get("test_held_numeric_access") is not False
    ):
        raise WriterColdStartError("invalid physical-update teacher auxiliary contract")


def load_writer_contract(
    path: Path,
    *,
    phase0_path: Path,
    split_path: Path,
    gate_zero_path: Path,
    mature_lora_path: Path,
) -> dict[str, Any]:
    """Load the single active Writer contract and bind all upstream authority."""

    spec = _load_toml(path, "Writer contract")
    _require(spec.get("schema_version"), 1, "Writer schema")
    if spec.get("status") not in {
        "frozen_before_writer_implementation_or_training_outcomes",
        "predeclared_physical_norm_recovery_after_cross_category_failure",
        "predeclared_source_physical_update_auxiliary_recovery_after_norm_recovery_failed_closed_loop",
        "predeclared_source_physical_update_weight_recovery_after_data_scale_underfit",
    }:
        raise WriterColdStartError("Writer status is not an active frozen contract")
    authority = spec.get("authority", {})
    for key, upstream in (
        ("phase0_contract_sha256", phase0_path),
        ("split_reseal_sha256", split_path),
        ("gate_zero_contract_sha256", gate_zero_path),
        ("mature_lora_contract_sha256", mature_lora_path),
    ):
        _require(authority.get(key), sha256_file(upstream), key)
    phase0 = _load_toml(phase0_path, "Phase-0 contract")
    mature = _load_toml(mature_lora_path, "mature LoRA contract")
    try:
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WriterColdStartError("invalid split reseal") from error
    lora = spec.get("lora", {})
    mature_lora = mature["fit"]["mature_official_default_r32"]
    _require(len(mature_lora["target_modules"]), lora.get("target_count"), "LoRA targets")
    for key in ("rank", "alpha", "dropout"):
        _require(lora.get(key), mature_lora.get(key), f"LoRA {key}")
    _require(
        lora.get("expected_parameter_count"),
        mature_lora.get("expected_trainable_parameters"),
        "LoRA parameter count",
    )
    validation = spec.get("validation", {}).get("task_ids")
    if not isinstance(validation, list) or len(validation) != len(set(validation)):
        raise WriterColdStartError("validation task IDs are invalid")
    _require(
        sorted(validation),
        sorted(set(validation) & set(split["active_split"]["validation"])),
        "validation split",
    )
    if set(phase0["splits"]["source"]) != set(split["active_split"]["source"]):
        raise WriterColdStartError("source split differs from permanent reseal")
    if set(validation) & set(split["active_split"]["held_out"]):
        raise WriterColdStartError("Writer validation overlaps held tasks")
    train = spec.get("train", {})
    if (
        train.get("world_size") != 8
        or train.get("global_batch_size")
        != train.get("world_size")
        * train.get("per_rank_micro_batch_size")
        * train.get("gradient_accumulation_steps")
    ):
        raise WriterColdStartError("eight-rank global batch is inconsistent")
    soft_cap = train.get("physical_delta_l2_soft_cap")
    coefficient = train.get("physical_delta_excess_coefficient")
    if (soft_cap is None) != (coefficient is None) or (
        soft_cap is not None and (soft_cap <= 0 or coefficient < 0)
    ):
        raise WriterColdStartError("physical-delta recovery parameters are invalid")
    if spec.get("writer", {}).get("bank_geometry_or_shared_subspace") is not False:
        raise WriterColdStartError("removed shared-structure mechanism reappeared")
    if authority.get("test_held_numeric_access") is not False:
        raise WriterColdStartError("test/held access must remain closed")
    _validate_teacher_auxiliary(spec, phase0)
    return spec


@dataclass(frozen=True)
class LoraTensorSpec:
    name: str
    module: str
    module_index: int
    factor_index: int
    rank: int
    width: int
    transpose_output: bool


def build_lora_tensor_specs(state: Mapping[str, torch.Tensor]) -> tuple[LoraTensorSpec, ...]:
    """Convert a real PEFT state into paired row-oriented Writer outputs."""

    marker_a = ".lora_A.default.weight"
    marker_b = ".lora_B.default.weight"
    modules: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for name, value in state.items():
        marker = marker_a if name.endswith(marker_a) else marker_b if name.endswith(marker_b) else None
        if marker is None or value.ndim != 2:
            raise WriterColdStartError(f"non-LoRA tensor in Writer template: {name}")
        module = name[: -len(marker)]
        factor = "A" if marker == marker_a else "B"
        modules.setdefault(module, {})[factor] = (name, value)
    if not modules or any(set(pair) != {"A", "B"} for pair in modules.values()):
        raise WriterColdStartError("every LoRA module must contain an A/B pair")
    result: list[LoraTensorSpec] = []
    for module_index, module in enumerate(sorted(modules)):
        pair = modules[module]
        name_a, value_a = pair["A"]
        name_b, value_b = pair["B"]
        rank, width_a = value_a.shape
        width_b, rank_b = value_b.shape
        if rank <= 0 or rank_b != rank:
            raise WriterColdStartError(f"LoRA pair rank differs for {module}")
        result.extend(
            (
                LoraTensorSpec(name_a, module, module_index, 0, rank, width_a, False),
                LoraTensorSpec(name_b, module, module_index, 1, rank, width_b, True),
            )
        )
    return tuple(result)


def physical_lora_delta_l2(
    state: Mapping[str, torch.Tensor], *, alpha: float, rank: int
) -> torch.Tensor:
    """Return the physical LoRA update norm without materializing full BA matrices."""

    if alpha <= 0 or rank <= 0:
        raise WriterColdStartError("invalid LoRA scale for physical norm")
    squared: torch.Tensor | None = None
    marker_a = ".lora_A.default.weight"
    for name, factor_a in state.items():
        if not name.endswith(marker_a):
            continue
        factor_b = state.get(name.replace(marker_a, ".lora_B.default.weight"))
        if factor_b is None or factor_a.shape[-2] != rank or factor_b.shape[-1] != rank:
            raise WriterColdStartError("physical norm received an incomplete LoRA pair")
        gram_a = factor_a @ factor_a.transpose(-2, -1)
        gram_b = factor_b.transpose(-2, -1) @ factor_b
        term = (gram_a * gram_b).sum(dim=(-2, -1))
        squared = term if squared is None else squared + term
    if squared is None:
        raise WriterColdStartError("physical norm received no LoRA factors")
    return (float(alpha) / rank) * torch.sqrt(torch.clamp_min(squared, 1e-24))


def physical_lora_delta_squared_distance(
    state: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    alpha: float,
    rank: int,
) -> torch.Tensor:
    """Gauge-invariant squared distance between two physical LoRA updates."""

    if alpha <= 0 or rank <= 0 or set(state) != set(target):
        raise WriterColdStartError("invalid physical-update distance inputs")
    squared: torch.Tensor | None = None
    marker_a = ".lora_A.default.weight"
    marker_b = ".lora_B.default.weight"
    for name, factor_a in state.items():
        if not name.endswith(marker_a):
            continue
        factor_b = state.get(name.replace(marker_a, marker_b))
        target_a = target.get(name)
        target_b = target.get(name.replace(marker_a, marker_b))
        if (
            factor_b is None
            or target_a is None
            or target_b is None
            or factor_a.shape != target_a.shape
            or factor_b.shape != target_b.shape
            or factor_a.shape[-2] != rank
            or factor_b.shape[-1] != rank
        ):
            raise WriterColdStartError("physical-update distance received an incomplete LoRA pair")
        state_a_gram = factor_a @ factor_a.transpose(-2, -1)
        state_b_gram = factor_b.transpose(-2, -1) @ factor_b
        target_a_gram = target_a @ target_a.transpose(-2, -1)
        target_b_gram = target_b.transpose(-2, -1) @ target_b
        cross_a = factor_a @ target_a.transpose(-2, -1)
        cross_b = factor_b.transpose(-2, -1) @ target_b
        term = (
            (state_a_gram * state_b_gram).sum(dim=(-2, -1))
            + (target_a_gram * target_b_gram).sum(dim=(-2, -1))
            - 2 * (cross_a * cross_b).sum(dim=(-2, -1))
        )
        squared = term if squared is None else squared + term
    if squared is None:
        raise WriterColdStartError("physical-update distance received no LoRA factors")
    return (float(alpha) / rank) ** 2 * torch.clamp_min(squared, 0.0)


def load_physical_update_teachers(
    spec: dict[str, Any],
    *,
    output_root: Path,
    template: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, float]]:
    """Load frozen source-only physical updates bound by the Writer contract."""

    auxiliary = spec.get("teacher_auxiliary")
    if auxiliary is None:
        return {}, {}
    root = output_root.resolve()

    def authority_path(relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise WriterColdStartError("physical-update teacher path escaped output root")
        return path

    bundle_path = authority_path(auxiliary["bundle_relative_path"])
    if sha256_file(bundle_path) != auxiliary["bundle_sha256"]:
        raise WriterColdStartError("physical-update teacher bundle changed")
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WriterColdStartError("invalid physical-update teacher bundle") from error
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != 1
        or bundle.get("status")
        != "source_action_supervised_physical_update_teachers_complete"
        or bundle.get("source_teacher_contract_sha256")
        != auxiliary["source_teacher_contract_sha256"]
        or bundle.get("validation_numeric_access") is not False
        or bundle.get("test_held_accessed") is not False
    ):
        raise WriterColdStartError("physical-update teacher bundle authority changed")
    expected_lora = spec["lora"]
    bundle_lora = bundle.get("lora", {})
    for key, expected in (
        ("target_count", expected_lora["target_count"]),
        ("rank", expected_lora["rank"]),
        ("alpha", expected_lora["alpha"]),
        ("dropout", expected_lora["dropout"]),
        ("parameters", expected_lora["expected_parameter_count"]),
    ):
        if bundle_lora.get(key) != expected:
            raise WriterColdStartError(f"physical-update teacher LoRA {key} changed")
    rows = bundle.get("teacher_tasks", {})
    task_ids = auxiliary["task_ids"]
    if set(rows) != {str(task) for task in task_ids}:
        raise WriterColdStartError("physical-update teacher task set changed")
    states: dict[int, dict[str, torch.Tensor]] = {}
    norm_squares: dict[int, float] = {}
    for task_id in task_ids:
        row = rows[str(task_id)]
        state_path = authority_path(row["state_relative_path"])
        if sha256_file(state_path) != row["state_sha256"]:
            raise WriterColdStartError(f"physical-update teacher state changed for task {task_id}")
        cpu_state = load_file(state_path)
        if set(cpu_state) != set(template) or any(
            cpu_state[name].shape != template[name].shape for name in template
        ):
            raise WriterColdStartError(f"physical-update teacher shape changed for task {task_id}")
        state = {
            name: value.to(device=device, dtype=torch.float32, non_blocking=True)
            for name, value in cpu_state.items()
        }
        norm = float(
            physical_lora_delta_l2(
                state, alpha=expected_lora["alpha"], rank=expected_lora["rank"]
            )
        )
        if not math.isclose(norm, float(row["physical_delta_l2"]), rel_tol=1e-5, abs_tol=1e-6):
            raise WriterColdStartError(f"physical-update teacher norm changed for task {task_id}")
        states[task_id] = state
        norm_squares[task_id] = norm * norm
    return states, norm_squares


class CompleteLoRAWriter(torch.nn.Module):
    """Layer/module/rank-aware generator for every tensor in one LoRA."""

    def __init__(
        self,
        tensor_specs: tuple[LoraTensorSpec, ...],
        *,
        template_state: Mapping[str, torch.Tensor],
        feature_dim: int,
        hidden_dim: int,
        module_embedding_dim: int,
        factor_embedding_dim: int,
        rank_embedding_dim: int,
        decoder_hidden_dim: int,
    ) -> None:
        super().__init__()
        if not tensor_specs or set(template_state) != {item.name for item in tensor_specs}:
            raise WriterColdStartError("Writer template and tensor specification differ")
        ranks = {item.rank for item in tensor_specs}
        if len(ranks) != 1:
            raise WriterColdStartError("one Writer contract cannot mix LoRA ranks")
        self.tensor_specs = tensor_specs
        self.feature_dim = feature_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_dim),
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
        )
        module_count = max(item.module_index for item in tensor_specs) + 1
        rank = next(iter(ranks))
        self.module_embedding = torch.nn.Embedding(module_count, module_embedding_dim)
        self.factor_embedding = torch.nn.Embedding(2, factor_embedding_dim)
        self.rank_embedding = torch.nn.Embedding(rank, rank_embedding_dim)
        decoder_input = hidden_dim + module_embedding_dim + factor_embedding_dim + rank_embedding_dim
        self.heads = torch.nn.ModuleDict()
        for width in sorted({item.width for item in tensor_specs}):
            head = torch.nn.Sequential(
                torch.nn.Linear(decoder_input, decoder_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(decoder_hidden_dim, width),
            )
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
            self.heads[str(width)] = head
        self._template_buffers: dict[str, str] = {}
        for index, item in enumerate(tensor_specs):
            value = template_state[item.name].detach().to(torch.float32).contiguous()
            if item.factor_index == 1 and torch.count_nonzero(value):
                raise WriterColdStartError("Writer LoRA-B template must be physical zero")
            buffer_name = f"template_{index:03d}"
            self.register_buffer(buffer_name, value, persistent=True)
            self._template_buffers[item.name] = buffer_name

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise WriterColdStartError("Writer feature tensor has wrong shape")
        task = self.encoder(features.to(torch.float32))
        batch_size = task.shape[0]
        result: dict[str, torch.Tensor] = {}
        for item in self.tensor_specs:
            ranks = torch.arange(item.rank, device=task.device)
            module_ids = torch.full_like(ranks, item.module_index)
            factor_ids = torch.full_like(ranks, item.factor_index)
            condition = torch.cat(
                (
                    task[:, None, :].expand(batch_size, item.rank, -1),
                    self.module_embedding(module_ids)[None].expand(batch_size, -1, -1),
                    self.factor_embedding(factor_ids)[None].expand(batch_size, -1, -1),
                    self.rank_embedding(ranks)[None].expand(batch_size, -1, -1),
                ),
                dim=-1,
            )
            rows = self.heads[str(item.width)](condition)
            generated = rows.transpose(-1, -2) if item.transpose_output else rows
            template = getattr(self, self._template_buffers[item.name])
            result[item.name] = generated + template[None]
        return result


def _hashed_tree(root: Path, *, excluded: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = excluded or set()
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_writer_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    rank_rng_states: list[dict[str, torch.Tensor]] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish complete Writer/optimizer/scheduler/RNG/cursor state."""

    if not checkpoint_dir.is_absolute() or checkpoint_dir.name != f"{step:06d}":
        raise WriterColdStartError("checkpoint path must be absolute and step-named")
    if checkpoint_dir.exists():
        raise WriterColdStartError("refusing to overwrite Writer checkpoint")
    world_size = metadata.get("world_size")
    if world_size != 1 and (rank_rng_states is None or len(rank_rng_states) != world_size):
        raise WriterColdStartError("Writer checkpoint lacks every-rank RNG state")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        state = {
            key: value.detach().to(device="cpu", copy=True).contiguous()
            for key, value in writer.state_dict().items()
        }
        save_file(state, staging / "writer.safetensors")
        from lerobot.common.train_utils import save_training_state
        from lerobot.utils.random_utils import serialize_rng_state

        save_training_state(
            staging,
            step,
            optimizer,
            scheduler,
            num_processes=world_size,
            batch_size=metadata.get("per_rank_micro_batch_size"),
        )
        states = rank_rng_states or [serialize_rng_state()]
        rng_dir = staging / "distributed_rng"
        rng_dir.mkdir()
        for rank, rng_state in enumerate(states):
            save_file(
                {key: value.detach().cpu().contiguous() for key, value in rng_state.items()},
                rng_dir / f"rank_{rank:05d}.safetensors",
            )
        _write_json(staging / "scaler.json", {"enabled": False, "state": {}})
        _write_json(staging / "sampler.json", metadata.get("sampler", {}))
        manifest = {
            "schema_version": 1,
            "step": step,
            **metadata,
            "writer_parameter_count": sum(value.numel() for value in writer.parameters()),
            "files": _hashed_tree(staging),
        }
        _write_json(staging / "writer_checkpoint_manifest.json", manifest)
        os.replace(staging, checkpoint_dir)
        sidecar = checkpoint_dir.parent / f"{checkpoint_dir.name}.manifest.sha256"
        sidecar.write_text(
            f"{sha256_file(checkpoint_dir / 'writer_checkpoint_manifest.json')}  writer_checkpoint_manifest.json\n",
            encoding="utf-8",
        )
        last_tmp = checkpoint_dir.parent / f".last.tmp-{uuid.uuid4().hex}"
        last_tmp.symlink_to(checkpoint_dir.name)
        os.replace(last_tmp, checkpoint_dir.parent / "last")
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_writer_checkpoint(checkpoint_dir: Path, *, world_size: int) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "writer_checkpoint_manifest.json"
    sidecar = checkpoint_dir.parent / f"{checkpoint_dir.name}.manifest.sha256"
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != sha256_file(manifest_path):
        raise WriterColdStartError("Writer checkpoint manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version"), 1, "checkpoint schema")
    _require(manifest.get("world_size"), world_size, "checkpoint topology")
    for relative, record in manifest.get("files", {}).items():
        path = checkpoint_dir / relative
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise WriterColdStartError(f"Writer checkpoint payload changed: {relative}")
    return manifest


def load_writer_checkpoint(
    checkpoint_dir: Path,
    *,
    writer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    rank: int,
    world_size: int,
    expected_authority: str | None = None,
) -> tuple[int, str]:
    """Restore exact Writer state after the caller constructs its data iterator."""

    manifest = _validate_writer_checkpoint(checkpoint_dir, world_size=world_size)
    if expected_authority is not None:
        _require(manifest.get("authority"), expected_authority, "checkpoint authority")
    incompatible = writer.load_state_dict(load_file(checkpoint_dir / "writer.safetensors"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise WriterColdStartError("Writer model state is incomplete")
    from lerobot.common.train_utils import load_training_step
    from lerobot.optim import load_optimizer_state, load_scheduler_state
    from lerobot.utils.random_utils import deserialize_rng_state

    training_state = checkpoint_dir / "training_state"
    step = load_training_step(training_state)
    _require(step, manifest["step"], "checkpoint step")
    load_optimizer_state(optimizer, training_state)
    load_scheduler_state(scheduler, training_state)
    deserialize_rng_state(load_file(checkpoint_dir / "distributed_rng" / f"rank_{rank:05d}.safetensors"))
    sampler = manifest.get("sampler", {})
    _require(sampler.get("completed_step"), step, "sampler cursor")
    chains = sampler.get("rank_data_chain", [""] * world_size)
    return step, chains[rank]


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_steps: int, total_steps: int, minimum_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 <= minimum_ratio <= 1 or not 0 < warmup_steps < total_steps:
        raise WriterColdStartError("invalid Writer scheduler contract")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
