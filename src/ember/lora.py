"""One sealed task-local LoRA space shared by EMBER and matched baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

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


class LoRAContract(Protocol):
    """Structural interface shared by retired SmolVLA and active PI05 contracts."""

    targets: tuple[LoRATarget, ...]
    rank: int
    alpha: int
    dropout: float
    identity_seed: int

    @property
    def parameter_count(self) -> int: ...

    @property
    def state_tensor_count(self) -> int: ...

    def to_dict(self) -> dict[str, Any]: ...


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


def canonical_contract_sha256(contract: LoRAContract) -> str:
    encoded = json.dumps(
        contract.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def lora_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash LoRA names, tensor metadata, and exact bytes in stable order."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().to(device="cpu").contiguous()
        digest.update(
            f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8")
        )
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_policy_against_contract(
    policy: torch.nn.Module, contract: LoRAContract
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
    contract: LoRAContract,
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
    state: Mapping[str, torch.Tensor], contract: LoRAContract
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
    policy: torch.nn.Module, contract: LoRAContract
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
    policy: torch.nn.Module, contract: LoRAContract
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
    contract: LoRAContract,
) -> None:
    validate_lora_state(state, contract)
    destination = task_lora_state_dict(policy)
    validate_lora_state(destination, contract)
    for name, value in state.items():
        destination[name].copy_(value.to(destination[name]))


def functional_lora_call(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call the frozen policy with differentiable task-local LoRA tensors."""

    validate_lora_state(state, contract)
    return torch.func.functional_call(
        policy, dict(state), args=args, kwargs=kwargs, strict=False
    )
