"""Sealed PI05 task-LoRA topology with method-owned capacity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from ember.lora import LoRAContractError, LoRATarget


PI05_LORA_SCHEMA = "ember_pi05_lora_v1"
PI05_TARGET_POLICY = (
    "all 18 action-expert q_proj/v_proj modules plus action_in_proj/action_out_proj"
)


def pi05_target_names() -> tuple[str, ...]:
    names = tuple(
        f"model.paligemma_with_expert.gemma_expert.model.layers.{layer}."
        f"self_attn.{projection}"
        for layer in range(18)
        for projection in ("q_proj", "v_proj")
    )
    return (*names, "model.action_in_proj", "model.action_out_proj")


@dataclass(frozen=True)
class Pi05LoRAContract:
    """PI05-specific authority for one complete task-local adapter."""

    targets: tuple[LoRATarget, ...]
    rank: int
    alpha: int
    dropout: float
    identity_seed: int
    foundation_repository: str
    foundation_revision: str
    foundation_weights_sha256: str
    foundation_config_sha256: str
    source_base_config_sha256: str
    recipe_sha256: str

    @property
    def parameter_count(self) -> int:
        return self.rank * sum(target.parameter_count_per_rank for target in self.targets)

    @property
    def state_tensor_count(self) -> int:
        return 2 * len(self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PI05_LORA_SCHEMA,
            "backbone": {
                "policy_type": "pi05",
                "initialization_repository": self.foundation_repository,
                "initialization_revision": self.foundation_revision,
                "initialization_weights_sha256": self.foundation_weights_sha256,
                "initialization_config_sha256": self.foundation_config_sha256,
            },
            "authorities": {
                "source_base_config_sha256": self.source_base_config_sha256,
                "recipe_sha256": self.recipe_sha256,
            },
            "target_policy": PI05_TARGET_POLICY,
            "adapter": {
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "identity_seed": self.identity_seed,
                "identity_initialization": "A deterministic random, B exactly zero",
            },
            "target_count": len(self.targets),
            "state_tensor_count": self.state_tensor_count,
            "trainable_parameter_count": self.parameter_count,
            "targets": [
                {
                    "name": target.name,
                    "in_features": target.in_features,
                    "out_features": target.out_features,
                }
                for target in self.targets
            ],
        }


def derive_pi05_lora_rank(
    contract: Pi05LoRAContract,
    *,
    rank: int,
) -> Pi05LoRAContract:
    """Change only the method-owned LoRA rank while preserving scale one."""

    if rank <= 0:
        raise LoRAContractError("derived PI05 LoRA rank must be positive")
    return replace(contract, rank=int(rank), alpha=int(rank))


def _require_sha256(value: Any, label: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise LoRAContractError(f"PI05 {label} is not a lowercase SHA-256")
    return result


def _require_git_revision(value: Any) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{40}", result) is None:
        raise LoRAContractError("PI05 foundation revision is not a full Git revision")
    return result


def _validate_pi05_contract(contract: Pi05LoRAContract, raw: dict[str, Any]) -> None:
    if tuple(target.name for target in contract.targets) != pi05_target_names():
        raise LoRAContractError("PI05 LoRA target names or order differ from the sealed topology")
    if len({target.name for target in contract.targets}) != len(contract.targets):
        raise LoRAContractError("PI05 LoRA target names are not unique")
    expected_shapes = tuple(
        [(1024, 2048), (1024, 256)] * 18 + [(32, 1024), (1024, 32)]
    )
    observed_shapes = tuple(
        (target.in_features, target.out_features) for target in contract.targets
    )
    if observed_shapes != expected_shapes:
        raise LoRAContractError("PI05 LoRA target shapes differ from pi05_base")
    if (
        contract.foundation_repository != "lerobot/pi05_base"
        or contract.rank <= 0
        or contract.alpha != contract.rank
        or contract.dropout != 0.0
        or contract.identity_seed < 0
    ):
        raise LoRAContractError("PI05 LoRA backbone or adapter hyperparameters changed")
    declared = {
        "target_count": len(contract.targets),
        "state_tensor_count": contract.state_tensor_count,
        "trainable_parameter_count": contract.parameter_count,
    }
    for key, actual in declared.items():
        if int(raw.get(key, -1)) != actual:
            raise LoRAContractError(f"declared PI05 {key} does not match target shapes")


def load_pi05_lora_contract(path: Path) -> Pi05LoRAContract:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != PI05_LORA_SCHEMA:
        raise LoRAContractError("unsupported PI05 LoRA contract schema")
    backbone = raw.get("backbone", {})
    authorities = raw.get("authorities", {})
    adapter = raw.get("adapter", {})
    if backbone.get("policy_type") != "pi05" or raw.get("target_policy") != PI05_TARGET_POLICY:
        raise LoRAContractError("PI05 LoRA policy identity or target policy changed")
    contract = Pi05LoRAContract(
        targets=tuple(
            LoRATarget(
                name=str(item["name"]),
                in_features=int(item["in_features"]),
                out_features=int(item["out_features"]),
            )
            for item in raw.get("targets", [])
        ),
        rank=int(adapter["rank"]),
        alpha=int(adapter["alpha"]),
        dropout=float(adapter["dropout"]),
        identity_seed=int(adapter["identity_seed"]),
        foundation_repository=str(backbone["initialization_repository"]),
        foundation_revision=_require_git_revision(backbone["initialization_revision"]),
        foundation_weights_sha256=_require_sha256(
            backbone["initialization_weights_sha256"], "foundation weights"
        ),
        foundation_config_sha256=_require_sha256(
            backbone["initialization_config_sha256"], "foundation config"
        ),
        source_base_config_sha256=_require_sha256(
            authorities["source_base_config_sha256"], "source-base config"
        ),
        recipe_sha256=_require_sha256(authorities["recipe_sha256"], "recipe"),
    )
    if adapter.get("identity_initialization") != "A deterministic random, B exactly zero":
        raise LoRAContractError("PI05 identity initialization changed")
    _validate_pi05_contract(contract, raw)
    return contract


def derive_pi05_targets(policy: torch.nn.Module) -> tuple[LoRATarget, ...]:
    """Derive the exact 38-target topology from a PI05 policy instance."""

    module_map = dict(policy.named_modules())
    result: list[LoRATarget] = []
    for name in pi05_target_names():
        module = module_map.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise LoRAContractError(f"required PI05 LoRA target is not a Linear: {name}")
        result.append(LoRATarget(name, module.in_features, module.out_features))
    return tuple(result)


def validate_pi05_weight_metadata(weights_path: Path, contract: Pi05LoRAContract) -> None:
    """Validate target tensor shapes without materializing the multi-billion-param policy."""

    from safetensors import safe_open

    if not weights_path.is_file():
        raise LoRAContractError(f"missing PI05 safetensors: {weights_path}")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for target in contract.targets:
            candidates = (
                f"{target.name}.weight",
                f"{target.name.removeprefix('model.')}.weight",
            )
            matches = [name for name in candidates if name in keys]
            if len(matches) != 1:
                raise LoRAContractError(
                    f"PI05 weight metadata has {len(matches)} matches for {target.name}"
                )
            shape = tuple(handle.get_slice(matches[0]).get_shape())
            expected = (target.out_features, target.in_features)
            if shape != expected:
                raise LoRAContractError(
                    f"PI05 weight shape differs for {target.name}: {shape} != {expected}"
                )
