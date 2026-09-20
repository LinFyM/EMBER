from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from types import SimpleNamespace

import pytest
import torch

from ember.lora import (
    LoRATarget,
    functional_lora_call,
    lora_state_sha256,
)
from ember.pi05_lora import Pi05LoRAContract, load_pi05_lora_contract
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
    INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    pi05_mean_flow_loss,
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
        with scoped_policy_flow_time_sampling(policy, LATIN_BETA_TIME_SAMPLING_SCHEME):
            first = policy.model.sample_time(20, torch.device("cpu"))
    after = torch.rand(4)

    torch.manual_seed(101)
    expected_after = torch.rand(4)
    with scoped_policy_randomness(303, torch.device("cpu")):
        with scoped_policy_flow_time_sampling(policy, LATIN_BETA_TIME_SAMPLING_SCHEME):
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
        with scoped_policy_flow_time_sampling(policy, LATIN_BETA_TIME_SAMPLING_SCHEME):
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
                time_chunks.append(policy.model.sample_time(2, torch.device("cpu")))
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


class _TinyPi05Core(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            time_sampling_beta_alpha=1.5,
            time_sampling_beta_beta=1.0,
            time_sampling_scale=0.999,
            time_sampling_offset=0.001,
        )
        self.projection = torch.nn.Linear(3, 4, bias=False)
        self.action_out_proj = torch.nn.Linear(4, 3, bias=False)

    def sample_noise(
        self,
        shape: tuple[int, ...],
        device: torch.device | str,
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def sample_time(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        return torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)

    def forward(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        token_masks: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del images, image_masks, tokens, token_masks
        hidden = self.projection(actions)
        velocity = self.action_out_proj(hidden)
        velocity = velocity + time[:, None, None] * 0.01
        target = noise - actions
        return (target - velocity).square()


class _TinyPi05Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from lerobot.utils.constants import ACTION

        self.model = _TinyPi05Core()
        self.config = SimpleNamespace(
            output_features={ACTION: SimpleNamespace(shape=(3,))}
        )

    def _preprocess_images(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        image = batch["image"]
        return [image], [torch.ones(image.shape[0], dtype=torch.bool)]

    def prepare_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        from lerobot.utils.constants import ACTION

        return batch[ACTION]


def _contract() -> Pi05LoRAContract:
    return replace(
        load_pi05_lora_contract(Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"),
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

    state = writer(torch.randn(3, 5), torch.randn(9, 4, 7), torch.tensor([0, 9]))
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


def test_pi05_loss_only_masks_action_chunk_tail() -> None:
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    policy = _TinyPi05Policy()
    actions = torch.tensor([[[0.2, -0.1, 0.3], [9.0, 8.0, 7.0]]])
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        ACTION: actions,
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }
    loss = pi05_mean_flow_loss(
        policy, batch, action_is_pad=torch.tensor([[False, True]])
    )
    velocity = policy.model.action_out_proj(policy.model.projection(actions)) + 0.005
    expected = (-actions - velocity)[0, 0].square().mean()
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("offset,random_size,seed", [(40, 64, 303), (-1, 64, 303), (0, 16, 303), (0, 64, None)])
def test_condition_flow_slice_rejects_out_of_range_or_unkeyed_draws(offset, random_size, seed):
    from ember.writer.functional import functional_microbatch_contract
    with pytest.raises(WriterModelError, match="condition exceeds|keyed sliceable"):
        functional_microbatch_contract(
            {"value": torch.zeros(32, 3)}, 32, policy_rng_seed=seed,
            flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            policy_random_batch_size=random_size, policy_batch_offset=offset,
        )
