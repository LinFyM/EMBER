from pathlib import Path

import torch

from ember.ecp.compiler import select_compiled_state
from ember.ecp.contracts import build_target_owners
from ember.ecp.policy_teacher import PrivilegedPolicyEvidence
from ember.ecp.program import VisibleProgramProjector
from ember.ecp.stage0 import ECPVideoEncoderOutput
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_objective import exact_effective_update_loss
from ember.lora import identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract


REPO_ROOT = Path(__file__).resolve().parents[2]


def _encoded(*, presence: torch.Tensor | None = None) -> ECPVideoEncoderOutput:
    videos, frames, events, owners, width = 2, 3, 8, 38, 128
    return ECPVideoEncoderOutput(
        process=torch.randn(videos, events, owners, width),
        presence=(
            torch.rand(videos, events) if presence is None else presence
        ),
        uncertainty=torch.rand(videos, events, owners, width) + 0.1,
        assignment=torch.rand(videos, events, frames, 4),
        state_posterior=torch.rand(videos, frames, events),
        confidence=torch.rand(videos, frames, 4),
        frame_mask=torch.ones(videos, frames, dtype=torch.bool),
        program_summary=torch.rand(videos, width),
        frame_owner_evidence=torch.rand(videos, frames, owners, width),
        language_summary=torch.rand(videos, width),
        scene_transition=torch.rand(videos, 3 * width),
    )


def _contract_and_states() -> tuple[object, tuple, dict[str, torch.Tensor]]:
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    owners = build_target_owners(contract)
    template = identity_lora_state(contract)
    return contract, owners, template


def _expert_evidence(
    template: dict[str, torch.Tensor], *, members: int = 2
) -> PrivilegedPolicyEvidence:
    states = {}
    for name, value in template.items():
        stacked = value[None].expand(members, *value.shape).clone()
        if ".lora_B." in name:
            stacked.normal_(std=0.01)
        states[name] = stacked
    return PrivilegedPolicyEvidence(
        member_states=states,
        phase_response=torch.randn(members, 8, 32),
        reliability=torch.tensor([0.7, 0.9][:members]),
    )


def test_visible_program_video_set_is_permutation_invariant() -> None:
    contract, owners, _ = _contract_and_states()
    projector = VisibleProgramProjector(owners)
    encoded = _encoded()
    forward = projector(encoded, torch.zeros(2, dtype=torch.long))
    indices = torch.tensor([1, 0])
    fields = {
        name: getattr(encoded, name).index_select(0, indices)
        for name in encoded.__dataclass_fields__
    }
    reverse = projector(
        ECPVideoEncoderOutput(**fields), torch.zeros(2, dtype=torch.long)
    )
    for name in ("language", "scene", "process", "presence", "uncertainty"):
        torch.testing.assert_close(getattr(forward, name), getattr(reverse, name))


def test_q_pi_cannot_create_process_outside_visible_presence() -> None:
    contract, owners, template = _contract_and_states()
    model = ECPStage1Model(owners, contract, template)
    encoded = _encoded(presence=torch.zeros(2, 8))
    output = model(
        encoded,
        _expert_evidence(template),
        torch.zeros(2, dtype=torch.long),
    )
    expected = output.anchors.process.expand(2, -1, -1, -1)
    torch.testing.assert_close(output.teacher.member_programs.process, expected)


def test_compiler_emits_one_complete_rank16_state_per_program() -> None:
    contract, owners, template = _contract_and_states()
    model = ECPStage1Model(owners, contract, template)
    output = model(
        _encoded(),
        _expert_evidence(template),
        torch.zeros(2, dtype=torch.long),
    )
    assert len(output.consensus_compilation.state) == 76
    consensus = select_compiled_state(output.consensus_compilation.state, 0)
    assert sum(value.numel() for value in consensus.values()) == 1_287_168
    assert all(value.shape[0] == 2 for value in output.member_compilation.state.values())
    attention = float(output.consensus_compilation.exact_owner_attention.detach())
    assert 0.0 <= attention <= 1.0


def test_exact_effective_update_loss_is_gauge_invariant_and_zero_on_identity() -> None:
    contract, _, template = _contract_and_states()
    target = _expert_evidence(template).member_states
    assert float(exact_effective_update_loss(target, target, contract)) < 1e-6
    transformed = {}
    scale = 2.0
    for name, value in target.items():
        transformed[name] = (
            value * scale if ".lora_A." in name else value / scale
        )
    assert float(exact_effective_update_loss(transformed, target, contract)) < 1e-5
