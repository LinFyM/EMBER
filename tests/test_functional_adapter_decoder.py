from dataclasses import dataclass
from pathlib import Path

import torch

from ember.functional_adaptation.decoder import (
    FunctionalAdapterDecoder,
    FunctionalCodebook,
    relative_effective_update_loss,
)
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    identity_lora_state,
)
from ember.pi05_lora import load_pi05_lora_contract


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_decoder_initializes_to_one_complete_identity_lora() -> None:
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    template = identity_lora_state(contract)
    decoder = FunctionalAdapterDecoder(contract, template)

    generated = decoder(torch.randn(32))

    assert set(generated) == set(template)
    assert all(torch.equal(generated[name], template[name]) for name in template)
    assert sum(value.numel() for value in generated.values()) == contract.parameter_count


def test_decoder_uses_one_code_for_all_targets_without_task_routes() -> None:
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    decoder = FunctionalAdapterDecoder(contract, identity_lora_state(contract))
    with torch.no_grad():
        for head in decoder.factor_heads.values():
            head.weight.normal_(std=0.01)

    generated = decoder(torch.stack((torch.zeros(32), torch.ones(32))))

    assert all(value.shape[0] == 2 for value in generated.values())
    assert any(
        not torch.equal(value[0], value[1]) for value in generated.values()
    )
    action_in = next(
        target for target in contract.targets if target.name.endswith("action_in_proj")
    )
    assert generated[action_in.name + LORA_A_SUFFIX].shape == (2, 16, 32)
    assert generated[action_in.name + LORA_B_SUFFIX].shape == (2, 1024, 16)


def test_privileged_codebook_starts_in_a_centered_whitened_gauge() -> None:
    codebook = FunctionalCodebook(56, 32, seed=7)

    assert codebook.gauge_loss().item() < 1e-12
    assert codebook(torch.tensor([0, 55])).shape == (2, 32)


@dataclass(frozen=True)
class TinyContract:
    targets: tuple[LoRATarget, ...]
    rank: int = 2
    alpha: int = 2
    dropout: float = 0.0
    identity_seed: int = 7

    @property
    def parameter_count(self) -> int:
        return self.rank * sum(
            target.parameter_count_per_rank for target in self.targets
        )

    @property
    def state_tensor_count(self) -> int:
        return 2 * len(self.targets)

    def to_dict(self) -> dict:
        return {}


def test_effective_warm_start_is_invariant_to_rank_gauge() -> None:
    contract = TinyContract((LoRATarget("module.q_proj", 3, 4),))
    generator = torch.Generator().manual_seed(7)
    target = {
        "module.q_proj" + LORA_A_SUFFIX: torch.randn(2, 3, generator=generator),
        "module.q_proj" + LORA_B_SUFFIX: torch.randn(4, 2, generator=generator),
    }
    permutation = torch.tensor([1, 0])
    equivalent = {
        "module.q_proj" + LORA_A_SUFFIX: target[
            "module.q_proj" + LORA_A_SUFFIX
        ].index_select(0, permutation),
        "module.q_proj" + LORA_B_SUFFIX: target[
            "module.q_proj" + LORA_B_SUFFIX
        ].index_select(1, permutation),
    }

    loss = relative_effective_update_loss(equivalent, target, contract)

    assert loss.item() < 1e-12
