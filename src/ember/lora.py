"""One sealed task-local LoRA space shared by EMBER and matched baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from peft import LoraConfig, inject_adapter_in_model


LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


class LoRAContractError(RuntimeError):
    """Raised when a policy or adapter state violates the sealed LoRA contract."""


@dataclass(frozen=True)
class LoRATarget:
    name: str
    in_features: int
    out_features: int

    @property
    def parameter_count_per_rank(self) -> int:
        return self.in_features + self.out_features


@dataclass(frozen=True)
class SmolVLALoRAContract:
    targets: tuple[LoRATarget, ...]
    rank: int
    alpha: int
    dropout: float
    identity_seed: int
    foundation_revision: str | None = None

    @property
    def parameter_count(self) -> int:
        return self.rank * sum(target.parameter_count_per_rank for target in self.targets)

    @property
    def state_tensor_count(self) -> int:
        return 2 * len(self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ember_smolvla_lora_v1",
            "foundation_revision": self.foundation_revision,
            "adapter": {
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "identity_seed": self.identity_seed,
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


def canonical_contract_sha256(contract: SmolVLALoRAContract) -> str:
    encoded = json.dumps(
        contract.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_lora_contract(path: Path) -> SmolVLALoRAContract:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "ember_smolvla_lora_v1":
        raise LoRAContractError("unsupported LoRA contract schema")
    adapter = raw.get("adapter", {})
    targets = tuple(
        LoRATarget(
            name=str(item["name"]),
            in_features=int(item["in_features"]),
            out_features=int(item["out_features"]),
        )
        for item in raw.get("targets", [])
    )
    contract = SmolVLALoRAContract(
        targets=targets,
        rank=int(adapter["rank"]),
        alpha=int(adapter["alpha"]),
        dropout=float(adapter["dropout"]),
        identity_seed=int(adapter["identity_seed"]),
        foundation_revision=raw.get("foundation_revision"),
    )
    declared = {
        "target_count": len(contract.targets),
        "state_tensor_count": contract.state_tensor_count,
        "trainable_parameter_count": contract.parameter_count,
    }
    for key, actual in declared.items():
        if int(raw.get(key, -1)) != actual:
            raise LoRAContractError(f"declared {key} does not match target shapes")
    if not targets or contract.rank <= 0 or contract.alpha <= 0 or contract.dropout < 0:
        raise LoRAContractError("invalid empty or non-positive LoRA contract")
    if len({target.name for target in targets}) != len(targets):
        raise LoRAContractError("LoRA target names are not unique")
    return contract


def derive_smolvla_targets(policy: torch.nn.Module) -> tuple[LoRATarget, ...]:
    """Enumerate the canonical 16-layer action-expert q/v plus five projections."""

    module_map = dict(policy.named_modules())
    names = [
        f"model.vlm_with_expert.lm_expert.layers.{layer}.self_attn.{projection}"
        for layer in range(16)
        for projection in ("q_proj", "v_proj")
    ]
    names.extend(
        (
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
        )
    )
    targets: list[LoRATarget] = []
    for name in names:
        module = module_map.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise LoRAContractError(f"required LoRA target is not a Linear: {name}")
        targets.append(
            LoRATarget(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
            )
        )
    return tuple(targets)


def validate_policy_against_contract(
    policy: torch.nn.Module, contract: SmolVLALoRAContract
) -> None:
    module_map = dict(policy.named_modules())
    for expected in contract.targets:
        module = module_map.get(expected.name)
        if not isinstance(module, torch.nn.Linear):
            raise LoRAContractError(
                f"sealed LoRA target is not a Linear in the policy: {expected.name}"
            )
        actual = LoRATarget(expected.name, module.in_features, module.out_features)
        if actual != expected:
            raise LoRAContractError(
                f"policy LoRA topology differs at {expected.name}: "
                f"expected {expected}, found {actual}"
            )


def expected_lora_state_shapes(
    contract: SmolVLALoRAContract,
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for target in contract.targets:
        result[target.name + LORA_A_SUFFIX] = (contract.rank, target.in_features)
        result[target.name + LORA_B_SUFFIX] = (target.out_features, contract.rank)
    return result


def task_lora_state_dict(
    policy: torch.nn.Module, *, clone: bool = False
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().clone() if clone else value
        for name, value in policy.named_parameters()
        if name.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX))
    }
    if not state:
        raise LoRAContractError("policy has no injected task-local LoRA")
    return state


def validate_lora_state(
    state: Mapping[str, torch.Tensor], contract: SmolVLALoRAContract
) -> None:
    expected = expected_lora_state_shapes(contract)
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise LoRAContractError(f"LoRA state keys differ; missing={missing[:3]} extra={extra[:3]}")
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape:
            raise LoRAContractError(
                f"LoRA tensor shape differs for {name}: expected {shape}, "
                f"found {tuple(state[name].shape)}"
            )


def _stable_tensor_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@torch.no_grad()
def initialize_identity_lora_(
    policy: torch.nn.Module, contract: SmolVLALoRAContract
) -> None:
    """Create a deterministic A/random, B/zero physical identity adapter."""

    state = task_lora_state_dict(policy)
    validate_lora_state(state, contract)
    for name, value in state.items():
        if name.endswith(LORA_B_SUFFIX):
            value.zero_()
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_tensor_seed(contract.identity_seed, name))
        initialized = torch.empty(value.shape, dtype=torch.float32, device="cpu")
        torch.nn.init.kaiming_uniform_(initialized, a=math.sqrt(5), generator=generator)
        value.copy_(initialized.to(device=value.device, dtype=value.dtype))


def inject_task_lora(
    policy: torch.nn.Module, contract: SmolVLALoRAContract
) -> torch.nn.Module:
    """Inject only the sealed adapter and freeze every shared/base parameter."""

    if any(".lora_" in name for name, _ in policy.named_parameters()):
        raise LoRAContractError("policy already contains a LoRA adapter")
    validate_policy_against_contract(policy, contract)
    config = LoraConfig(
        r=contract.rank,
        lora_alpha=contract.alpha,
        lora_dropout=contract.dropout,
        bias="none",
        target_modules=[target.name for target in contract.targets],
        init_lora_weights=True,
    )
    injected = inject_adapter_in_model(config, policy, adapter_name="default")
    initialize_identity_lora_(injected, contract)
    state = task_lora_state_dict(injected)
    trainable = {name for name, value in injected.named_parameters() if value.requires_grad}
    if trainable != set(state):
        raise LoRAContractError("injection left trainable parameters outside the task-local LoRA")
    if sum(value.numel() for value in state.values()) != contract.parameter_count:
        raise LoRAContractError("injected LoRA parameter count differs from sealed contract")
    return injected


@torch.no_grad()
def copy_task_lora_state_(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: SmolVLALoRAContract,
) -> None:
    validate_lora_state(state, contract)
    destination = task_lora_state_dict(policy)
    validate_lora_state(destination, contract)
    for name, value in state.items():
        destination[name].copy_(value.to(destination[name]))


def functional_lora_call(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: SmolVLALoRAContract,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call the frozen policy with differentiable task-local LoRA tensors."""

    validate_lora_state(state, contract)
    return torch.func.functional_call(
        policy, dict(state), args=args, kwargs=kwargs, strict=False
    )
