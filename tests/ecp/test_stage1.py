from pathlib import Path

import torch

from ember.ecp.compiler import TargetFamilyCompiler, select_compiled_state
from ember.ecp.contracts import build_target_owners
from ember.ecp.policy_teacher import PrivilegedPolicyEvidence
from ember.ecp.program import ECPProgram, VisibleProgramProjector
from ember.ecp.stage0 import ECPVideoEncoderOutput
from ember.ecp.stage1_data import gauge_canonicalize_factors
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_objective import (
    canonical_factor_loss,
    effective_update_cosine_matrix,
    exact_effective_update_loss,
)
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
        policy_response=torch.randn(members, 8, 38, 5, 4, 128),
        policy_response_weights=torch.ones(members, 8, 5),
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
    assert output.teacher.evidence_gate.shape == (2, 8, 38, 1)


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
    cosine, left_energy, right_energy = effective_update_cosine_matrix(
        target, target, contract
    )
    torch.testing.assert_close(cosine.diagonal(), torch.ones(2))
    torch.testing.assert_close(left_energy, right_energy)


def test_compact_svd_gauge_preserves_update_and_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(11)
    a = torch.randn(2, 4, 7, generator=generator)
    b = torch.randn(2, 6, 4, generator=generator)
    canonical_a, canonical_b = gauge_canonicalize_factors(a, b)
    repeat_a, repeat_b = gauge_canonicalize_factors(a, b)
    torch.testing.assert_close(canonical_b @ canonical_a, b @ a)
    torch.testing.assert_close(canonical_a, repeat_a)
    torch.testing.assert_close(canonical_b, repeat_b)


def test_absolute_compiler_uses_prior_only_or_full_surface() -> None:
    contract, owners, template = _contract_and_states()
    for value in template.values():
        value.normal_(std=0.01)
    compiler = TargetFamilyCompiler(owners, contract, template)
    for head in (*compiler.factor_a.values(), *compiler.factor_b.values()):
        head.weight.data.zero_()
    common = {
        "language": torch.randn(1, 38, 128),
        "scene": torch.randn(1, 38, 128),
        "process": torch.randn(1, 8, 38, 128),
        "uncertainty": torch.ones(1, 8, 38, 128),
    }
    prior = compiler(
        ECPProgram(**common, presence=torch.zeros(1, 8))
    ).state
    full = compiler(
        ECPProgram(**common, presence=torch.ones(1, 8))
    ).state
    for name, target in template.items():
        torch.testing.assert_close(prior[name][0], target)
        torch.testing.assert_close(full[name], torch.zeros_like(full[name]))
    assert float(canonical_factor_loss(prior, template, contract).detach()) < 1e-7


def test_compiler_address_queries_cannot_write_without_program_content() -> None:
    contract, owners, template = _contract_and_states()
    compiler = TargetFamilyCompiler(owners, contract, template)
    zero = ECPProgram(
        language=torch.zeros(1, 38, 128),
        scene=torch.zeros(1, 38, 128),
        process=torch.zeros(1, 8, 38, 128),
        presence=torch.ones(1, 8),
        uncertainty=torch.zeros(1, 8, 38, 128),
    )
    state = compiler(zero).state
    for value in state.values():
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_query_content_modulation_reaches_rank_outputs() -> None:
    contract, owners, template = _contract_and_states()
    compiler = TargetFamilyCompiler(owners, contract, template)
    program = ECPProgram(
        language=torch.randn(1, 38, 128),
        scene=torch.randn(1, 38, 128),
        process=torch.randn(1, 8, 38, 128),
        presence=torch.ones(1, 8),
        uncertainty=torch.rand(1, 8, 38, 128),
    )
    first = next(iter(compiler(program).state.values()))
    (first[:, 0].float().square().mean()).backward()
    gradient = compiler.query_content_modulation.weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0


def test_q_pi_ignores_masked_policy_support_channels() -> None:
    contract, owners, template = _contract_and_states()
    model = ECPStage1Model(owners, contract, template).eval()
    encoded = _encoded()
    evidence = _expert_evidence(template)
    weights = evidence.policy_response_weights.clone()
    weights[..., -1] = 0
    baseline = PrivilegedPolicyEvidence(
        **{
            **evidence.__dict__,
            "policy_response_weights": weights,
        }
    )
    changed_response = evidence.policy_response.clone()
    changed_response[..., -1, :, :] = 1_000_000
    changed = PrivilegedPolicyEvidence(
        **{
            **baseline.__dict__,
            "policy_response": changed_response,
        }
    )
    first = model(encoded, baseline, torch.zeros(2, dtype=torch.long))
    second = model(encoded, changed, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(
        first.teacher.program.process, second.teacher.program.process
    )


def test_policy_support_content_reaches_q_pi_correction() -> None:
    contract, owners, template = _contract_and_states()
    model = ECPStage1Model(owners, contract, template)
    output = model(
        _encoded(),
        _expert_evidence(template),
        torch.zeros(2, dtype=torch.long),
    )
    output.teacher.program.process.float().square().mean().backward()
    gradient = model.policy_teacher.support_value.weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0
