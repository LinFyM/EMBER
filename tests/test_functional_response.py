from dataclasses import dataclass
from types import SimpleNamespace

import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

from ember.functional_adaptation.functional_response import (
    build_functional_response_target,
    functional_response_distillation_loss,
    pi05_denoised_action_response,
    pi05_flow_response,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget


class FakeLoRAProjection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(3, 2, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 4, bias=False)}
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lora_B["default"](self.lora_A["default"](value))


class FakeCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            time_sampling_beta_alpha=1.5,
            time_sampling_beta_beta=1.0,
            time_sampling_scale=0.999,
            time_sampling_offset=0.001,
        )
        self.q_proj = FakeLoRAProjection()
        self.action_out_proj = torch.nn.Linear(4, 32, bias=False)

    def sample_noise(self, shape, device):
        return torch.randn(*shape, device=device)

    def sample_time(self, batch_size, device):
        return torch.rand(batch_size, device=device)


class FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeCore()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=7)

    def forward(self, batch):
        noise = self.model.sample_noise(batch["query"].shape, batch["query"].device)
        time = self.model.sample_time(batch["query"].shape[0], batch["query"].device)
        query = batch["query"] + noise * time[:, None, None]
        response = self.model.action_out_proj(self.model.q_proj(query))
        return response.square().mean(), {}

    def predict_action_chunk(self, batch, *, noise, num_steps):
        assert num_steps == 10
        query = batch["query"] + noise[..., :3]
        response = self.model.q_proj(query)
        return torch.nn.functional.pad(response, (0, 3))


@dataclass(frozen=True)
class FakeContract:
    targets: tuple[LoRATarget, ...] = (LoRATarget("model.q_proj", 3, 4),)
    rank: int = 2
    alpha: int = 2
    dropout: float = 0.0
    identity_seed: int = 7

    @property
    def parameter_count(self) -> int:
        return 14

    @property
    def state_tensor_count(self) -> int:
        return 2

    def to_dict(self) -> dict:
        return {}


def _state(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "model.q_proj" + LORA_A_SUFFIX: a,
        "model.q_proj" + LORA_B_SUFFIX: b,
    }


def test_flow_response_keeps_token_and_padded_action_axes() -> None:
    policy = FakePolicy().requires_grad_(False)
    state = _state(torch.ones(2, 3), torch.ones(4, 2))
    batch = {"query": torch.ones(2, 50, 3)}

    response = pi05_flow_response(
        policy, state, FakeContract(), batch, policy_seed=19
    )

    assert response.shape == (2, 50, 32)


def test_denoised_action_response_uses_paired_noise_and_complete_chunk() -> None:
    policy = FakePolicy().requires_grad_(False)
    contract = FakeContract()
    a = torch.ones(2, 3)
    identity = _state(a, torch.zeros(4, 2))
    expert = _state(a, torch.ones(4, 2))
    batch = {
        "query": torch.ones(2, 50, 3),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 16, dtype=torch.long),
    }

    identity_response = pi05_denoised_action_response(
        policy, identity, contract, batch, policy_seed=29
    )
    expert_response = pi05_denoised_action_response(
        policy, expert, contract, batch, policy_seed=29
    )
    repeated = pi05_denoised_action_response(
        policy, expert, contract, batch, policy_seed=29
    )

    assert expert_response.shape == (2, 50, 7)
    assert not torch.equal(identity_response, expert_response)
    assert torch.equal(expert_response, repeated)


def test_functional_distillation_matches_expert_delta_and_backpropagates() -> None:
    policy = FakePolicy().requires_grad_(False)
    contract = FakeContract()
    batch = {"query": torch.ones(2, 50, 3)}
    a = torch.ones(2, 3)
    identity = _state(a, torch.zeros(4, 2))
    expert = _state(a, torch.ones(4, 2))
    target = build_functional_response_target(
        policy,
        identity,
        expert,
        contract,
        batch,
        policy_seed=23,
    )
    candidate_b = torch.zeros(4, 2, requires_grad=True)
    candidate = _state(a, candidate_b)

    identity_loss = functional_response_distillation_loss(
        policy,
        candidate,
        contract,
        batch,
        target,
        policy_seed=23,
    )
    identity_loss.backward()
    expert_loss = functional_response_distillation_loss(
        policy,
        expert,
        contract,
        batch,
        target,
        policy_seed=23,
    )

    assert identity_loss.item() > 0.9
    assert candidate_b.grad is not None
    assert torch.count_nonzero(candidate_b.grad)
    assert expert_loss.item() < 1e-12
