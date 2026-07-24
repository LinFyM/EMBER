"""Shared per-sample functional-LoRA execution for Writer evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ember.batched_lora import BatchedLoRAInference
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.writer.model import WriterModelError


class WriterLoRARolloutAdapter:
    """Batch distinct generated LoRAs over one resident PI05 policy."""

    def _initialize_rollout(
        self,
        *,
        policy: torch.nn.Module,
        lora_contract: Any,
        identity_state: Mapping[str, torch.Tensor],
        evaluation_adapter: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.lora_contract = lora_contract
        self.identity_state = {
            name: value.detach().clone() for name, value in identity_state.items()
        }
        self.device = device
        self.evaluation_adapter = dict(evaluation_adapter)
        self.batched_lora = BatchedLoRAInference(policy, lora_contract)
        self._physical_lora_is_identity = True

    @torch.inference_mode()
    def install(self, prepared: Any) -> None:
        validate_lora_state(prepared.state, self.lora_contract)
        copy_task_lora_state_(self.policy, prepared.state, self.lora_contract)
        self._physical_lora_is_identity = False

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        prepared: Sequence[Any],
        batch: Mapping[str, torch.Tensor],
        *,
        noise: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        if len(prepared) != int(noise.shape[0]):
            raise WriterModelError("Writer LoRA batch and policy noise batch differ")
        if not self._physical_lora_is_identity:
            copy_task_lora_state_(
                self.policy, self.identity_state, self.lora_contract
            )
            self._physical_lora_is_identity = True
        with self.batched_lora.activate([item.state for item in prepared]):
            return self.policy.predict_action_chunk(
                dict(batch),
                noise=noise,
                num_steps=num_steps,
            )
