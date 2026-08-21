from pathlib import Path
from types import SimpleNamespace

import torch

from ember.ecp.contracts import TargetFamily, build_target_owners
from ember.ecp.events import (
    EventConditionedHorizonBinding,
    OrderedEventSegmenter,
    TaskGroundedTransitionMatcher,
)
from ember.ecp.observer import ECPNativeObserver
from ember.pi05_lora import load_pi05_lora_contract


def _owners():
    root = Path(__file__).resolve().parents[2]
    return build_target_owners(load_pi05_lora_contract(root / "configs/pi05_lora_v1.json"))


class _FakeExpertLayer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(width)
        self.self_attn = SimpleNamespace(q_proj=torch.nn.Linear(width, width, bias=False))
        self.update = torch.nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.05 * self.update(self.input_layernorm(value))


class _FakeExpert(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeExpertLayer(width) for _ in range(18)])
        self.norm = torch.nn.LayerNorm(width)


class _FakeBridge(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        expert = _FakeExpert(width)
        self.gemma_expert = SimpleNamespace(model=expert)
        self.expert = expert

    def forward(self, *, inputs_embeds, **_kwargs):
        prefix, suffix = inputs_embeds
        for layer in self.expert.layers:
            suffix = layer(suffix)
        return (prefix, self.expert.norm(suffix)), None


class _FakeCore(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.paligemma_with_expert = _FakeBridge(width)
        self.action_in_proj = torch.nn.Linear(32, width, bias=False)
        self.action_out_proj = torch.nn.Linear(width, 32, bias=False)

    def embed_suffix(self, noise: torch.Tensor, _time: torch.Tensor):
        suffix = self.action_in_proj(noise)
        padding = torch.ones(noise.shape[:2], dtype=torch.bool, device=noise.device)
        attention = torch.zeros_like(padding)
        attention[:, 0] = True
        adarms = torch.zeros(noise.shape[0], suffix.shape[-1], device=noise.device)
        return suffix, padding, attention, adarms

    @staticmethod
    def _prepare_attention_masks_4d(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, None]


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


def test_native_observer_captures_all_layers_then_compacts_to_owner_lattice() -> None:
    torch.manual_seed(4)
    core = _FakeCore(width=16).requires_grad_(False)
    observer = ECPNativeObserver(
        _owners(),
        prefix_width=16,
        expert_width=16,
        program_width=12,
        padded_action_dim=32,
        image_tokens=2,
    )
    prefix = torch.randn(2, 6, 16)
    prefix_padding = torch.ones(2, 6, dtype=torch.bool)
    noise = torch.randn(2, 50, 32)

    output = observer(
        core,
        prefix,
        prefix_padding,
        noise,
        torch.ones(2),
    )
    output.owner_lattice.square().mean().backward()

    assert output.owner_lattice.shape == (2, 38, 50, 12)
    assert output.patch_states.shape == (2, 2, 12)
    assert output.language_states.shape == (2, 4, 12)
    assert output.flow_velocity.shape == (2, 50, 32)
    assert all(parameter.grad is None for parameter in core.parameters())
    assert any(parameter.grad is not None for parameter in observer.parameters())


def test_event_binding_uses_all_horizons_and_segmenter_is_ordered() -> None:
    torch.manual_seed(7)
    batch, frames, width = 2, 7, 16
    frame_mask = torch.tensor(
        [[True] * 7, [True] * 5 + [False] * 2], dtype=torch.bool
    )
    language_mask = torch.tensor([[True, True, True], [True, True, False]])
    matcher = TaskGroundedTransitionMatcher(width=width)
    binding = EventConditionedHorizonBinding(width=width)
    segmenter = OrderedEventSegmenter(width=width)
    patches = torch.randn(batch, frames, 8, width)
    language = torch.randn(batch, 3, width)
    lattice = torch.randn(batch, frames, 38, 50, width)

    candidates, confidence = matcher(
        patches, language, frame_mask, language_mask
    )
    one_frame_candidates, _ = matcher(
        patches[:, :1], language, frame_mask[:, :1], language_mask
    )
    bound = binding(candidates, confidence, lattice, frame_mask)
    changed_lattice = lattice.clone()
    changed_lattice[:, :, :, -1] += 3.0
    changed = binding(candidates, confidence, changed_lattice, frame_mask)
    program = segmenter(bound, confidence, frame_mask)
    program.process.square().mean().backward()

    assert candidates.shape == (batch, frames, 4, width)
    assert one_frame_candidates.shape == (batch, 1, 4, width)
    assert bound.shape == (batch, frames, 4, 38, width)
    assert not torch.allclose(bound, changed)
    assert program.process.shape == (batch, 8, 38, width)
    assert program.presence.shape == (batch, 8)
    assert program.uncertainty.shape == program.process.shape
    assert program.assignment.shape == (batch, 8, frames, 4)
    assert not torch.count_nonzero(program.assignment[1, :, 5:])
    expected_slot = torch.einsum(
        "bte,e->bt",
        program.state_posterior,
        torch.arange(8, dtype=program.state_posterior.dtype),
    )
    assert torch.all(expected_slot[0, 1:] >= expected_slot[0, :-1] - 1e-5)
    assert any(parameter.grad is not None for parameter in matcher.parameters())
    assert any(parameter.grad is not None for parameter in binding.parameters())
    assert any(parameter.grad is not None for parameter in segmenter.parameters())
