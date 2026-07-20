from __future__ import annotations

import torch

from ember.lora import LoRATarget, SmolVLALoRAContract, lora_state_sha256
from ember.writer.functional import (
    prepare_frozen_writer_policy,
    writer_functional_action_loss,
    writer_success_weighted_flow_loss,
)
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


class _LossPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del noise, time
        value = self.projection(batch["value"])
        per_sample = value.square().mean(dim=1)
        if reduction == "none":
            return per_sample, {"loss": float(per_sample.mean().detach())}
        loss = per_sample.mean()
        return loss, {"loss": float(loss.detach())}


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("projection", 3, 4),),
        rank=2,
        alpha=1,
        dropout=0.0,
        identity_seed=29,
    )


def _writer(template: dict[str, torch.Tensor]) -> CompleteLoRAWriter:
    return CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        vision_feature_dim=7,
        language_feature_dim=5,
        hidden_dim=12,
        attention_heads=3,
        temporal_chunk_size=4,
        chunk_memory_tokens=2,
        episode_memory_tokens=2,
        task_memory_tokens=2,
        decoder_hidden_dim=10,
    )


def test_functional_action_loss_only_backpropagates_into_writer() -> None:
    policy = _LossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)

    loss, details = writer_functional_action_loss(
        writer,
        policy,
        _contract(),
        language_features=torch.randn(3, 5),
        video_features=torch.randn(9, 7),
        episode_offsets=torch.tensor([0, 4, 9]),
        batch={"value": torch.ones(6, 3)},
    )
    loss.backward()

    assert details["loss"] == float(loss.detach())
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in writer.parameters()
    )


def test_tensor_state_hash_covers_names_metadata_and_bytes() -> None:
    state = {
        "b": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "a": torch.tensor([3, 4], dtype=torch.int64),
    }
    digest = lora_state_sha256(state)
    assert digest == lora_state_sha256({"a": state["a"], "b": state["b"]})
    changed = {**state, "b": state["b"].clone()}
    changed["b"][0, 0] = 0
    assert digest != lora_state_sha256(changed)


def test_success_weighted_flow_loss_weights_episodes_equally() -> None:
    policy = _LossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)
    loss, details = writer_success_weighted_flow_loss(
        writer,
        policy,
        _contract(),
        language_features=torch.randn(3, 5),
        video_features=torch.randn(9, 7),
        episode_offsets=torch.tensor([0, 4, 9]),
        batch={
            "value": torch.tensor(
                [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]
            )
        },
        rollout_episode_ids=torch.tensor([0, 0, 1]),
    )
    loss.backward()
    assert details["successful_episodes"] == 2
    assert details["successful_chunks"] == 3
    assert bool(torch.isfinite(loss))
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert any(parameter.grad is not None for parameter in writer.parameters())
