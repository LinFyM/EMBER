from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.lora import LoRATarget, SmolVLALoRAContract, lora_state_sha256
from ember.writer.functional import (
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    prepare_frozen_writer_policy,
    scoped_policy_flow_time_sampling,
    scoped_policy_randomness,
    writer_functional_action_loss,
    writer_success_weighted_flow_loss,
)


class _FlowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            time_sampling_beta_alpha=1.5,
            time_sampling_beta_beta=1.0,
            time_sampling_scale=0.999,
            time_sampling_offset=0.001,
        )

    def sample_time(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.full((batch_size,), -1.0, device=device)


class _FlowPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FlowModel()


def test_latin_beta_time_is_exactly_stratified_replayed_and_scoped() -> None:
    policy = _FlowPolicy()
    torch.manual_seed(101)
    with scoped_policy_randomness(303, torch.device("cpu")):
        with scoped_policy_flow_time_sampling(
            policy, LATIN_BETA_TIME_SAMPLING_SCHEME
        ):
            first = policy.model.sample_time(20, torch.device("cpu"))
    after = torch.rand(4)

    torch.manual_seed(101)
    expected_after = torch.rand(4)
    with scoped_policy_randomness(303, torch.device("cpu")):
        with scoped_policy_flow_time_sampling(
            policy, LATIN_BETA_TIME_SAMPLING_SCHEME
        ):
            second = policy.model.sample_time(20, torch.device("cpu"))
    beta = (first - 0.001) / 0.999
    uniform = beta.pow(1.5)
    strata = torch.floor(20 * uniform).to(torch.long).sort().values
    assert torch.equal(first, second)
    assert torch.equal(strata, torch.arange(20))
    assert bool(((first >= 0.001) & (first <= 1.0)).all())
    assert torch.equal(after, expected_after)
    assert torch.equal(
        policy.model.sample_time(2, torch.device("cpu")),
        torch.full((2,), -1.0),
    )


class _LossPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4, bias=False)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, float]]:
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


class _TinyWriter(torch.nn.Module):
    def __init__(self, template: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))
        self._names = {}
        for index, (name, value) in enumerate(template.items()):
            key = f"template_{index}"
            self.register_buffer(key, value.detach().clone())
            self._names[name] = key

    def forward(self, *_args: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            name: value + self.scale.to(value) * torch.ones_like(value)
            for name, key in self._names.items()
            for value in (getattr(self, key),)
        }


def _writer(template: dict[str, torch.Tensor]) -> _TinyWriter:
    return _TinyWriter(template)


def test_functional_action_loss_only_backpropagates_into_writer() -> None:
    policy = _LossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)

    loss, details = writer_functional_action_loss(
        writer,
        policy,
        _contract(),
        language_features=torch.randn(3, 5),
        video_features=torch.randn(9, 4, 7),
        episode_offsets=torch.tensor([0, 9]),
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


def test_detached_lora_gradient_bridge_backpropagates_exact_writer_gradient() -> None:
    policy = _LossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)
    with torch.no_grad():
        writer.scale.fill_(0.01)
    language = torch.randn(3, 5)
    video = torch.randn(9, 4, 7)
    offsets = torch.tensor([0, 9])
    batch = {"value": torch.ones(6, 3)}
    direct_loss, _ = writer_functional_action_loss(
        writer,
        policy,
        _contract(),
        language_features=language,
        video_features=video,
        episode_offsets=offsets,
        batch=batch,
    )
    direct_gradient = torch.autograd.grad(direct_loss, writer.scale)[0]
    state = writer(language, video, offsets)
    loss, details, gradients = functional_lora_loss_gradient(
        policy,
        state,
        _contract(),
        batch=batch,
    )
    bridged_gradient = torch.autograd.grad(
        tuple(state.values()),
        writer.scale,
        grad_outputs=tuple(gradients[name] for name in state),
    )
    assert details["loss"] == float(loss)
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert torch.allclose(
        bridged_gradient[0], direct_gradient, atol=1e-7, rtol=1e-6
    )


def test_success_weighted_flow_loss_weights_episodes_equally() -> None:
    policy = _LossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)
    loss, details = writer_success_weighted_flow_loss(
        writer,
        policy,
        _contract(),
        language_features=torch.randn(3, 5),
        video_features=torch.randn(9, 4, 7),
        episode_offsets=torch.tensor([0, 9]),
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
