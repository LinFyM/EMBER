from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    functional_lora_call,
    lora_state_sha256,
)
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    prepare_frozen_writer_policy,
    scoped_policy_flow_noise_sampling,
    scoped_policy_flow_time_sampling,
    scoped_policy_randomness,
)
from ember.writer.errors import WriterModelError


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

    def sample_noise(
        self, shape: tuple[int, ...], device: torch.device | str
    ) -> torch.Tensor:
        return torch.full(shape, -1.0, device=device)


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


def test_antithetic_gaussian_noise_is_zero_mean_replayed_and_scoped() -> None:
    policy = _FlowPolicy()
    torch.manual_seed(101)
    with scoped_policy_randomness(303, torch.device("cpu")):
        with scoped_policy_flow_noise_sampling(
            policy, ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME
        ):
            first = policy.model.sample_noise((20, 5, 7), torch.device("cpu"))
    after = torch.rand(4)

    torch.manual_seed(101)
    expected_after = torch.rand(4)
    with scoped_policy_randomness(303, torch.device("cpu")):
        with scoped_policy_flow_noise_sampling(
            policy, ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME
        ):
            second = policy.model.sample_noise((20, 5, 7), torch.device("cpu"))
            with pytest.raises(WriterModelError, match="positive even batch"):
                policy.model.sample_noise((19, 5, 7), torch.device("cpu"))
    assert torch.equal(first, second)
    assert torch.allclose(first.sum(dim=0), torch.zeros(5, 7), atol=1e-6)
    assert bool(torch.isfinite(first).all())
    assert torch.equal(after, expected_after)
    assert torch.equal(
        policy.model.sample_noise((2, 3), torch.device("cpu")),
        torch.full((2, 3), -1.0),
    )


def test_variance_reduced_microbatch_slices_reconstruct_full_draws() -> None:
    policy = _FlowPolicy()
    with scoped_policy_randomness(303, torch.device("cpu")):
        with scoped_policy_flow_time_sampling(
            policy, LATIN_BETA_TIME_SAMPLING_SCHEME
        ):
            full_time = policy.model.sample_time(20, torch.device("cpu"))
        with scoped_policy_flow_noise_sampling(
            policy, ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME
        ):
            full_noise = policy.model.sample_noise((20, 5), torch.device("cpu"))

    time_chunks = []
    noise_chunks = []
    for offset in range(0, 20, 2):
        with scoped_policy_randomness(303, torch.device("cpu")):
            with scoped_policy_flow_time_sampling(
                policy,
                LATIN_BETA_TIME_SAMPLING_SCHEME,
                logical_batch_size=20,
                batch_offset=offset,
            ):
                time_chunks.append(
                    policy.model.sample_time(2, torch.device("cpu"))
                )
            with scoped_policy_flow_noise_sampling(
                policy,
                ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
                logical_batch_size=20,
                batch_offset=offset,
            ):
                noise_chunks.append(
                    policy.model.sample_noise((2, 5), torch.device("cpu"))
                )
    assert torch.equal(torch.cat(time_chunks), full_time)
    assert torch.equal(torch.cat(noise_chunks), full_noise)


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


class _RandomLossPolicy(_LossPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FlowModel()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, object]]:
        del reduction
        value = self.projection(batch["value"])
        noise = self.model.sample_noise(value.shape, value.device)
        time = self.model.sample_time(value.shape[0], value.device)
        losses = (value - noise * time[:, None]).square()
        loss = losses.mean()
        return loss, {
            "loss": float(loss.detach()),
            "loss_per_dim": losses.mean(dim=0).detach().tolist(),
        }


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

    state = writer(
        torch.randn(3, 5), torch.randn(9, 4, 7), torch.tensor([0, 9])
    )
    loss, details = functional_lora_call(
        policy, state, _contract(), {"value": torch.ones(6, 3)}
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
    direct_state = writer(language, video, offsets)
    direct_loss, _ = functional_lora_call(
        policy, direct_state, _contract(), batch
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


@pytest.mark.parametrize(
    ("time_scheme", "noise_scheme"),
    (
        (
            LATIN_BETA_TIME_SAMPLING_SCHEME,
            ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        ),
        (
            INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        ),
    ),
)
def test_microbatched_functional_gradient_preserves_logical_b20_estimator(
    time_scheme: str,
    noise_scheme: str,
) -> None:
    policy = _RandomLossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)
    with torch.no_grad():
        writer.scale.fill_(0.01)
    state = writer(torch.randn(3, 5), torch.randn(9, 4, 7), torch.tensor([0, 9]))
    batch = {"value": torch.randn(20, 3)}
    common = {
        "policy_rng_seed": 303,
        "policy_rng_device": torch.device("cpu"),
        "flow_time_sampling_scheme": time_scheme,
        "flow_noise_sampling_scheme": noise_scheme,
    }
    full_loss, full_details, full_gradients = functional_lora_loss_gradient(
        policy,
        state,
        _contract(),
        batch=batch,
        **common,
    )
    micro_loss, micro_details, micro_gradients = functional_lora_loss_gradient(
        policy,
        state,
        _contract(),
        batch=batch,
        policy_microbatch_size=2,
        **common,
    )
    assert torch.allclose(micro_loss, full_loss, atol=1e-7, rtol=1e-6)
    assert micro_details["loss"] == pytest.approx(full_details["loss"])
    assert micro_details["loss_per_dim"] == pytest.approx(
        full_details["loss_per_dim"]
    )
    assert set(micro_gradients) == set(full_gradients)
    for name in full_gradients:
        assert torch.allclose(
            micro_gradients[name],
            full_gradients[name],
            atol=1e-7,
            rtol=1e-6,
        )


def test_functional_gradient_rejects_zero_microbatch() -> None:
    policy = _RandomLossPolicy()
    template = prepare_frozen_writer_policy(policy, _contract())
    writer = _writer(template)
    state = writer(torch.randn(3, 5), torch.randn(9, 4, 7), torch.tensor([0, 9]))
    with pytest.raises(WriterModelError, match="invalid functional policy microbatch"):
        functional_lora_loss_gradient(
            policy,
            state,
            _contract(),
            batch={"value": torch.randn(20, 3)},
            policy_rng_seed=303,
            policy_rng_device=torch.device("cpu"),
            flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
            flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            policy_microbatch_size=0,
        )
