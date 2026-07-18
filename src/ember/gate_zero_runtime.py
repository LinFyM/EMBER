"""Pinned SmolVLA/PEFT runtime primitives for Gate 0 and Gate 1."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open


class GateZeroRuntimeError(RuntimeError):
    """Raised when policy, normalization, or adapter runtime authority drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_normalization(
    path: Path,
    *,
    expected_sha256: str,
    expected_task_ids: Sequence[int],
    expected_count: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load only the already-sealed all-source normalization statistics."""

    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise GateZeroRuntimeError("source normalization SHA256 changed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroRuntimeError("invalid source normalization JSON") from error
    authority = payload.get("authority", {})
    if authority.get("split") != "source" or authority.get("episode_bounds_inclusive") != [8, 27]:
        raise GateZeroRuntimeError("source normalization episode authority changed")
    if authority.get("task_indices") != list(expected_task_ids):
        raise GateZeroRuntimeError("source normalization task authority changed")
    dimensions = {"observation.state": 8, "action": 7}
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for feature, dimension in dimensions.items():
        record = payload.get(feature)
        if not isinstance(record, dict) or record.get("count") != expected_count:
            raise GateZeroRuntimeError(f"source normalization count changed for {feature}")
        converted: dict[str, torch.Tensor] = {}
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            value = torch.as_tensor(record.get(name), dtype=torch.float32)
            if value.shape != (dimension,) or not torch.isfinite(value).all():
                raise GateZeroRuntimeError(f"invalid source normalization {feature}.{name}")
            converted[name] = value
        if torch.any(converted["std"] <= 0):
            raise GateZeroRuntimeError(f"non-positive source normalization std for {feature}")
        stats[feature] = converted
    return stats


def inspect_lora_targets(
    weight_path: Path,
    targets: Sequence[str],
    *,
    rank: int,
) -> dict[str, Any]:
    """Verify target weights without instantiating the 500M-parameter policy."""

    if rank <= 0 or not targets or len(set(targets)) != len(targets):
        raise GateZeroRuntimeError("invalid LoRA target declaration")
    records = []
    total = 0
    try:
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for target in targets:
                key = f"{target}.weight"
                if key not in available:
                    raise GateZeroRuntimeError(f"missing LoRA target weight: {key}")
                shape = list(handle.get_slice(key).get_shape())
                if len(shape) != 2:
                    raise GateZeroRuntimeError(f"LoRA target is not a matrix: {key}")
                out_features, in_features = shape
                parameters = rank * (out_features + in_features)
                records.append(
                    {
                        "module": target,
                        "weight_key": key,
                        "shape": shape,
                        "rank": rank,
                        "trainable_parameters": parameters,
                    }
                )
                total += parameters
    except OSError as error:
        raise GateZeroRuntimeError(f"cannot inspect policy weights: {weight_path}") from error
    return {"targets": records, "trainable_parameters": total}


def build_lora_config(
    *,
    targets: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
    init_lora_weights: bool | str,
    base_revision: str,
):
    """Build the one primary LoRA configuration pinned by the pilot contract."""

    from peft import LoraConfig

    if len(base_revision) != 40 or any(character not in "0123456789abcdef" for character in base_revision):
        raise GateZeroRuntimeError("base revision must be a full lowercase Git SHA")
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(targets),
        modules_to_save=[],
        init_lora_weights=init_lora_weights,
        lora_dropout=dropout,
        bias="none",
        lora_bias=False,
        use_rslora=False,
        fan_in_fan_out=False,
        revision=base_revision,
    )


def _row_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def deterministic_flow_inputs(
    row_keys: Sequence[str],
    *,
    action_shape: tuple[int, int],
    noise_seed: int,
    time_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create per-row common random numbers independent of batch order/size."""

    if not row_keys or len(set(row_keys)) != len(row_keys):
        raise GateZeroRuntimeError("fixed-flow row keys must be unique and non-empty")
    if len(action_shape) != 2 or any(value <= 0 for value in action_shape):
        raise GateZeroRuntimeError("invalid fixed-flow action shape")
    noises = []
    times = []
    for key in row_keys:
        noise_generator = torch.Generator(device="cpu").manual_seed(_row_seed(noise_seed, key))
        time_generator = torch.Generator(device="cpu").manual_seed(_row_seed(time_seed, key))
        noises.append(torch.randn(action_shape, generator=noise_generator, dtype=torch.float32))
        # Beta(1.5, 1.0) has CDF x**1.5, hence inverse CDF u**(2/3).
        uniform = torch.rand((), generator=time_generator, dtype=torch.float32)
        times.append(uniform.pow(2.0 / 3.0) * 0.999 + 0.001)
    return torch.stack(noises).to(device), torch.stack(times).to(device)


def configure_smolvla(
    base_config: Any,
    *,
    local_vlm_path: Path,
    device: str,
    pretrained_path: str | Path,
    pretrained_revision: str,
) -> Any:
    """Derive the explicit 8D-state/7D-action/two-camera LIBERO policy config."""

    from lerobot.configs import FeatureType, PolicyFeature

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    if not (local_vlm_path / "model.safetensors").is_file():
        raise GateZeroRuntimeError("pinned local VLM snapshot is incomplete")
    return dataclasses.replace(
        base_config,
        input_features=input_features,
        output_features=output_features,
        device=device,
        use_amp=False,
        use_peft=False,
        push_to_hub=False,
        repo_id=None,
        private=None,
        pretrained_path=str(pretrained_path),
        pretrained_revision=pretrained_revision,
        empty_cameras=1,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        vlm_model_name=str(local_vlm_path.resolve()),
        load_vlm_weights=True,
        compile_model=False,
    )


def physical_lora_deltas(peft_model: Any, targets: Sequence[str], *, adapter: str = "default") -> dict[str, torch.Tensor]:
    """Extract canonical physical ``ΔW`` tensors through PEFT's scaling-aware API."""

    base = peft_model.get_base_model()
    result: dict[str, torch.Tensor] = {}
    for target in targets:
        layer = base.get_submodule(target)
        if adapter not in layer.lora_A or adapter not in layer.lora_B:
            raise GateZeroRuntimeError(f"adapter {adapter!r} missing from {target}")
        result[f"{target}.weight"] = (
            layer.get_delta_weight(adapter).detach().to(torch.float32).cpu().contiguous()
        )
    return result
