from pathlib import Path

import torch

from ember.ecp.contracts import TargetFamily, build_target_owners
from ember.ecp.observer import ActionLayerStateCapture
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.meta_lora import MetaLoRAStack


def _owners():
    root = Path(__file__).resolve().parents[2]
    return build_target_owners(load_pi05_lora_contract(root / "configs/pi05_lora_v1.json"))


def test_target_owner_contract_matches_all_38_deployed_targets() -> None:
    owners = _owners()

    assert len(owners) == 38
    assert [owner.index for owner in owners] == list(range(38))
    assert [owner.family for owner in owners[:4]] == [
        TargetFamily.Q,
        TargetFamily.V,
        TargetFamily.Q,
        TargetFamily.V,
    ]
    assert owners[0].layer == 0
    assert owners[35].layer == 17
    assert owners[36].family is TargetFamily.ACTION_IN
    assert owners[37].family is TargetFamily.ACTION_OUT


class _ExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(4)
        self.self_attn = torch.nn.ModuleDict({
            name: torch.nn.Linear(4, 4, bias=False)
            for name in MetaLoRAStack.PROJECTIONS
        })

    def forward(self, value):
        normalized = self.input_layernorm(value)
        attention = sum(self.self_attn[name](normalized) for name in ("q_proj", "k_proj", "v_proj"))
        return value + 0.01 * self.self_attn["o_proj"](attention)


def test_native_capture_retains_all_horizons_and_optional_action_meta_gradients() -> None:
    torch.manual_seed(3)
    expert = torch.nn.Module()
    expert.layers = torch.nn.ModuleList([_ExpertLayer() for _ in range(18)])
    expert.norm = torch.nn.LayerNorm(4)
    expert.requires_grad_(False)
    meta = MetaLoRAStack(expert.layers, rank=4)
    value = torch.randn(2, 50, 4)
    with meta.installed(expert), ActionLayerStateCapture(expert, detach=False) as capture:
        for layer in expert.layers:
            value = layer(value)
        expert.norm(value)
        boundaries = capture.stacked()
    assert boundaries.shape == (2, 19, 50, 4)
    torch.testing.assert_close(boundaries[:, -1], value)
    boundaries[:, 1:].square().mean().backward()
    assert all(parameter.grad is None for parameter in expert.parameters())
    assert all(adapter.b.grad is not None and bool(adapter.b.grad.abs().sum() > 0)
               for adapter in meta.adapters.values())
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in expert.modules())
