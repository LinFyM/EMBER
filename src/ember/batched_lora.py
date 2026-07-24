"""Per-sample LoRA execution for one ordinary batched policy forward."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch

from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRAContract,
    LoRAContractError,
    validate_lora_state,
)


class BatchedLoRAInference:
    """Add a distinct sealed LoRA delta to each sample through forward hooks.

    The physical PEFT adapter must remain at its identity initialization. Hooks
    add per-sample deltas after the ordinary base-plus-identity PEFT forward, so
    the policy retains its native batch dimension without materializing a
    different adapter for every rollout.
    """

    def __init__(self, policy: torch.nn.Module, contract: LoRAContract) -> None:
        modules = dict(policy.named_modules())
        self._contract = contract
        self._active_state: dict[str, torch.Tensor] | None = None
        self._closed = False
        self._handles: list[Any] = []
        self._targets: list[tuple[str, Any, float]] = []

        expected_scale = float(contract.alpha) / float(contract.rank)
        for target in contract.targets:
            module = modules.get(target.name)
            if module is None:
                raise LoRAContractError(
                    f"missing injected LoRA target for batched inference: {target.name}"
                )
            try:
                lora_a = module.lora_A["default"].weight
                lora_b = module.lora_B["default"].weight
                scale = float(module.scaling["default"])
            except (AttributeError, KeyError, TypeError) as error:
                raise LoRAContractError(
                    f"target is not an injected PEFT LoRA layer: {target.name}"
                ) from error
            if (
                tuple(lora_a.shape) != (contract.rank, target.in_features)
                or tuple(lora_b.shape) != (target.out_features, contract.rank)
                or not math.isclose(scale, expected_scale, rel_tol=0.0, abs_tol=0.0)
            ):
                raise LoRAContractError(
                    f"injected LoRA shape or scale changed at {target.name}"
                )
            if torch.count_nonzero(lora_b).item() != 0:
                raise LoRAContractError(
                    "physical LoRA must be identity before batched inference"
                )
            self._targets.append((target.name, module, scale))
            self._handles.append(
                module.register_forward_hook(self._make_hook(target.name, scale))
            )

    def _make_hook(self, target_name: str, scale: float):
        a_name = target_name + LORA_A_SUFFIX
        b_name = target_name + LORA_B_SUFFIX

        def add_per_sample_delta(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            state = self._active_state
            if state is None:
                return output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise LoRAContractError(
                    f"batched LoRA target has no tensor input: {target_name}"
                )
            value = inputs[0]
            lora_a = state[a_name]
            lora_b = state[b_name]
            if value.shape[0] != lora_a.shape[0]:
                raise LoRAContractError(
                    f"policy batch and LoRA batch differ at {target_name}: "
                    f"{value.shape[0]} != {lora_a.shape[0]}"
                )
            flattened = value.to(lora_a.dtype).reshape(
                value.shape[0], -1, value.shape[-1]
            )
            hidden = torch.bmm(flattened, lora_a.transpose(1, 2))
            delta = torch.bmm(hidden, lora_b.transpose(1, 2))
            delta = delta.reshape(*value.shape[:-1], lora_b.shape[1])
            return output + delta.to(output.dtype) * scale

        return add_per_sample_delta

    @contextmanager
    def activate(
        self, states: Sequence[Mapping[str, torch.Tensor]]
    ) -> Iterator[None]:
        """Activate one complete adapter per policy-batch sample."""

        if self._closed:
            raise LoRAContractError("batched LoRA inference hooks are closed")
        if self._active_state is not None:
            raise LoRAContractError("batched LoRA inference is not reentrant")
        if not states:
            raise LoRAContractError("batched LoRA inference received no states")
        for state in states:
            validate_lora_state(state, self._contract)

        stacked: dict[str, torch.Tensor] = {}
        for target_name, module, _scale in self._targets:
            for suffix, destination in (
                (LORA_A_SUFFIX, module.lora_A["default"].weight),
                (LORA_B_SUFFIX, module.lora_B["default"].weight),
            ):
                name = target_name + suffix
                stacked[name] = torch.stack(
                    [
                        state[name].to(
                            device=destination.device,
                            dtype=destination.dtype,
                        )
                        for state in states
                    ],
                    dim=0,
                )
        self._active_state = stacked
        try:
            yield
        finally:
            self._active_state = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active_state is not None:
            raise LoRAContractError("cannot close active batched LoRA inference")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._closed = True
