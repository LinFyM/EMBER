from collections import Counter
from pathlib import Path

import torch

from ember.ecp.compiler import TargetFamilyCompiler, select_compiled_state
from ember.ecp.contracts import TargetFamily, TargetOwner, build_target_owners
from ember.ecp.low_rank import replace_low_rank_modes
from ember.ecp.policy_teacher import PrivilegedPolicyEvidence
from ember.ecp.policy_response import owner_resolved_response_distillation_loss
from ember.ecp.program import ECPProgram, VisibleProgramProjector
from ember.ecp.stage0 import ECPVideoEncoderOutput
from ember.ecp.stage1_data import (
    ECPStage1Task,
    build_stage1_schedule,
    gauge_canonicalize_factors,
)
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_materialization import (
    OUTCOME_PROJECTION_SCHEMA,
    resolve_stage1_materialization_config,
)
from ember.ecp.stage1_outcome import (
    COMPILER_BINDING,
    PROGRAM_BINDING,
    outcome_coordinate,
    outcome_surrogate_loss,
    structured_outcome_perturbation,
)
from ember.ecp.stage1_objective import (
    canonical_factor_loss,
    effective_update_cosine_matrix,
    exact_effective_update_loss,
)
from ember.ecp.stage1_support import (
    PolicySupportPanel,
    SUPPORT_PRESERVATION_BASELINE_BARRIER,
    policy_support_loss_from_response,
)
from ember.ecp.stage1_support_audit import summarize_policy_support_audit
from ember.lora import LoRATarget, SmolVLALoRAContract, identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.reward.credit import AntitheticCredit


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


def _tiny_compiler() -> tuple[TargetFamilyCompiler, dict[str, torch.Tensor]]:
    target = LoRATarget("tiny", in_features=5, out_features=6)
    contract = SmolVLALoRAContract(
        targets=(target,), rank=2, alpha=2, dropout=0.0, identity_seed=1
    )
    owner = TargetOwner(
        index=0,
        target_name=target.name,
        family=TargetFamily.Q,
        layer=0,
        in_features=target.in_features,
        out_features=target.out_features,
    )
    template = {
        "tiny.lora_A.default.weight": torch.randn(2, 5),
        "tiny.lora_B.default.weight": torch.randn(6, 2),
    }
    return (
        TargetFamilyCompiler(
            (owner,),
            contract,
            template,
            program_width=4,
            compiler_width=8,
            event_slots=2,
        ),
        template,
    )


def _tiny_program() -> ECPProgram:
    return ECPProgram(
        language=torch.randn(1, 1, 4),
        scene=torch.randn(1, 1, 4),
        process=torch.randn(1, 2, 1, 4),
        presence=torch.ones(1, 2),
        uncertainty=torch.rand(1, 2, 1, 4),
    )


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


def test_stage1_decision_prefixes_are_task_equal() -> None:
    tasks = tuple(
        ECPStage1Task(
            ordinal=ordinal,
            global_task_id=ordinal,
            suite="suite",
            task_id=ordinal,
            language=f"task {ordinal}",
            path=Path(f"task_{ordinal}.hdf5"),
            expected_bytes=1,
            episode_lengths=tuple(40 + ordinal + index for index in range(50)),
            fold_role="fit",
        )
        for ordinal in range(19)
    )
    config = {
        "roles": {"fit_task_ordinals": list(range(19))},
        "data": {
            "frame_stride": 5,
            "visible_videos_per_visit": 2,
            "pair_seed": 17,
        },
        "optimization": {
            "visits_per_fit_task": 24,
            "task_balance_block_rounds": 6,
            "stage_stop_task_visits": [228, 456],
            "seed": 23,
        },
    }
    schedule = build_stage1_schedule(
        config=config,
        tasks=tasks,
        world_size=6,
        total_task_visits=456,
        mode="formal",
    )
    for prefix, expected in ((114, 6), (228, 12), (456, 24)):
        counts = Counter(ordinal for ordinal, _ in schedule[:prefix])
        assert counts == Counter({ordinal: expected for ordinal in range(19)})


def test_outcome_materialization_uses_v14_cursor_and_v10_model_contract() -> None:
    resolved = resolve_stage1_materialization_config(
        REPO_ROOT / "configs/pi05_ecp_stage1_outcome_binding_v14.json"
    )
    assert resolved.stage == "stage1_outcome_binding_v14"
    assert resolved.cursor_name == "outcome_macro"
    assert resolved.checkpoint_cursors == (1,)
    assert resolved.projection_schema == OUTCOME_PROJECTION_SCHEMA
    assert resolved.base["schema_version"] == (
        "ember_ecp_stage1_process_value_selector_v10"
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
    assert float(
        output.consensus_compilation.rank_replacement_fraction.detach()
    ) == 0.0
    assert output.consensus_compilation.rank_angles.shape == (1, 38, 16)


def test_outcome_binding_offsets_preserve_structured_coordinates() -> None:
    contract, owners, template = _contract_and_states()
    model = ECPStage1Model(owners, contract, template)
    for selector in model.compiler.rank_selector.values():
        selector.weight.data.zero_()
    encoded = _encoded()
    evidence = _expert_evidence(template)
    baseline = model(encoded, evidence, torch.zeros(2, dtype=torch.long))
    event_owner = torch.zeros(1, 8, 38, 1)
    event_owner[:, 2, 7] = 0.5
    changed_program = model(
        encoded,
        evidence,
        torch.zeros(2, dtype=torch.long),
        evidence_logit_offset=event_owner,
    )
    assert baseline.teacher.evidence_gate_logits.shape == (2, 8, 38, 1)
    torch.testing.assert_close(
        changed_program.teacher.evidence_gate_logits
        - baseline.teacher.evidence_gate_logits,
        event_owner.expand(2, -1, -1, -1),
    )
    owner = torch.zeros(1, 38, 1)
    owner[:, 7] = 0.1
    changed_compiler = model(
        encoded,
        evidence,
        torch.zeros(2, dtype=torch.long),
        rank_angle_offset=owner,
    )
    delta = (
        changed_compiler.consensus_compilation.rank_angles
        - baseline.consensus_compilation.rank_angles
    )
    torch.testing.assert_close(delta[:, 7], torch.full_like(delta[:, 7], 0.1))
    torch.testing.assert_close(delta[:, :7], torch.zeros_like(delta[:, :7]))
    torch.testing.assert_close(delta[:, 8:], torch.zeros_like(delta[:, 8:]))
    coordinate_delta = outcome_coordinate(
        changed_compiler, COMPILER_BINDING
    ) - outcome_coordinate(baseline, COMPILER_BINDING)
    torch.testing.assert_close(coordinate_delta, owner.squeeze(-1))


def test_structured_outcome_coordinates_reach_q_pi_and_compiler() -> None:
    contract, owners, template = _contract_and_states()
    model = ECPStage1Model(owners, contract, template)
    for selector in model.compiler.rank_selector.values():
        selector.weight.data.normal_(std=0.1)
    output = model(
        _encoded(),
        _expert_evidence(template),
        torch.zeros(2, dtype=torch.long),
    )
    for index, coordinate in enumerate((PROGRAM_BINDING, COMPILER_BINDING)):
        perturbation = structured_outcome_perturbation(
            output, coordinate=coordinate, sigma=0.1, seed=17 + index
        )
        value = outcome_coordinate(output, coordinate)
        assert perturbation.epsilon.shape == value.shape
        credit = AntitheticCredit(
            gradient=perturbation.epsilon,
            plus_scores=(1.0, 0.0),
            minus_scores=(0.0, 0.0),
            lane_advantages=(1.0, 0.0),
            plus_successes=1,
            minus_successes=0,
            plus_progress_mean=0.5,
            minus_progress_mean=0.0,
        )
        model.zero_grad(set_to_none=True)
        outcome_surrogate_loss(
            output, credit, coordinate=coordinate, weight=0.1
        ).backward(retain_graph=index == 0)
        if coordinate == PROGRAM_BINDING:
            gradient = model.policy_teacher.evidence_gate.weight.grad
        else:
            gradient = model.compiler.rank_selector["q"].weight.grad
        assert gradient is not None and float(gradient.abs().sum()) > 0


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


def test_rank_selector_starts_from_the_exact_complete_prior() -> None:
    contract, owners, template = _contract_and_states()
    for value in template.values():
        value.normal_(std=0.01)
    compiler = TargetFamilyCompiler(owners, contract, template)
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
    assert float(canonical_factor_loss(prior, template, contract).detach()) < 1e-7
    assert float(exact_effective_update_loss(full, template, contract).detach()) < 1e-5
    assert float(
        compiler(
            ECPProgram(**common, presence=torch.ones(1, 8))
        ).rank_replacement_fraction.detach()
    ) == 0.0


def test_language_scene_and_presence_cannot_write_without_process_values() -> None:
    contract, owners, template = _contract_and_states()
    for value in template.values():
        value.normal_(std=0.01)
    compiler = TargetFamilyCompiler(owners, contract, template)
    for selector in compiler.rank_selector.values():
        selector.weight.data.normal_(std=0.1)
    zero = ECPProgram(
        language=torch.randn(1, 38, 128),
        scene=torch.randn(1, 38, 128),
        process=torch.zeros(1, 8, 38, 128),
        presence=torch.ones(1, 8),
        uncertainty=torch.zeros(1, 8, 38, 128),
    )
    state = compiler(zero).state
    assert float(exact_effective_update_loss(state, template, contract).detach()) < 1e-5


def test_language_and_scene_condition_process_value_queries() -> None:
    compiler, _ = _tiny_compiler()
    for selector in compiler.rank_selector.values():
        selector.weight.data.normal_(std=0.1)
    first_program = _tiny_program()
    second_program = ECPProgram(
        **{
            **first_program.__dict__,
            "language": first_program.language + torch.randn_like(
                first_program.language
            ),
            "scene": first_program.scene + torch.randn_like(first_program.scene),
        }
    )
    first = compiler(first_program).state["tiny.lora_A.default.weight"]
    second = compiler(second_program).state["tiny.lora_A.default.weight"]
    assert float((first - second).detach().abs().sum()) > 0.0


def test_rank_mode_replacement_is_amplitude_invariant_and_differentiable() -> None:
    generator = torch.Generator().manual_seed(17)
    base_a = torch.randn(2, 5, generator=generator)
    base_b = torch.randn(6, 2, generator=generator)
    replacement_a = torch.randn(1, 2, 5, generator=generator, requires_grad=True)
    replacement_b = torch.randn(1, 6, 2, generator=generator, requires_grad=True)
    angles = torch.tensor([[0.2, -0.4]], requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        selected_a, selected_b = replace_low_rank_modes(
            base_a=base_a,
            base_b=base_b,
            replacement_a=replacement_a,
            replacement_b=replacement_b,
            angles=angles,
        )
    scaled_a, scaled_b = replace_low_rank_modes(
        base_a=base_a,
        base_b=base_b,
        replacement_a=7.0 * replacement_a,
        replacement_b=7.0 * replacement_b,
        angles=angles,
    )
    torch.testing.assert_close(selected_a, scaled_a)
    torch.testing.assert_close(selected_b, scaled_b)
    zero_a, zero_b = replace_low_rank_modes(
        base_a=base_a,
        base_b=base_b,
        replacement_a=replacement_a,
        replacement_b=replacement_b,
        angles=torch.zeros_like(angles),
    )
    torch.testing.assert_close(zero_a[0], base_a)
    torch.testing.assert_close(zero_b[0], base_b)
    (selected_b @ selected_a).square().mean().backward()
    assert replacement_a.grad is not None and torch.isfinite(replacement_a.grad).all()
    assert replacement_b.grad is not None and torch.isfinite(replacement_b.grad).all()
    assert angles.grad is not None and torch.isfinite(angles.grad).all()


def test_selector_learns_before_replacement_directions_receive_gradient() -> None:
    compiler, template = _tiny_compiler()
    program = _tiny_program()
    first = compiler(program)
    for name, value in template.items():
        torch.testing.assert_close(first.state[name][0], value)
    dense = (
        first.state["tiny.lora_B.default.weight"]
        @ first.state["tiny.lora_A.default.weight"]
    )
    target = torch.randn_like(dense)
    (dense - target).square().mean().backward()
    selector = compiler.rank_selector["q"].weight
    assert selector.grad is not None and float(selector.grad.abs().sum()) > 0
    factor_grad = sum(
        float(head.weight.grad.abs().sum())
        for head in (compiler.factor_a["q"], compiler.factor_b["q"])
        if head.weight.grad is not None
    )
    assert factor_grad == 0.0
    with torch.no_grad():
        selector.add_(selector.grad, alpha=-0.1)
    compiler.zero_grad(set_to_none=True)
    second = compiler(program)
    second_dense = (
        second.state["tiny.lora_B.default.weight"]
        @ second.state["tiny.lora_A.default.weight"]
    )
    (second_dense - target).square().mean().backward()
    assert float(second.rank_replacement_fraction.detach()) > 0.0
    assert float(compiler.factor_a["q"].weight.grad.abs().sum()) > 0.0
    assert float(compiler.factor_b["q"].weight.grad.abs().sum()) > 0.0


def test_query_content_modulation_reaches_rank_outputs() -> None:
    contract, owners, template = _contract_and_states()
    compiler = TargetFamilyCompiler(owners, contract, template)
    for value in template.values():
        value.normal_(std=0.01)
    for selector in compiler.rank_selector.values():
        selector.weight.data.normal_(std=0.1)
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


def test_policy_support_response_baselines_share_one_normalization() -> None:
    source = torch.zeros(2, 3, 4)
    expert = torch.ones(1, 2, 3, 4)
    panel = PolicySupportPanel(
        panel_id=0,
        kind="learner",
        trajectory_path=Path("unused"),
        trajectory_bytes=0,
        selected_indices=(0, 1),
        policy_seed=1,
        source_response=source,
        shared_response=torch.full_like(source, 0.5),
        expert_responses=expert,
        expert_weights=torch.ones(1),
        outcome_weight=0.25,
        source_support_weight=1.0,
        shared_support_weight=1.0,
        learner_success=False,
    )
    source_loss = policy_support_loss_from_response(candidate=source, panel=panel)
    shared_loss = policy_support_loss_from_response(
        candidate=panel.shared_response, panel=panel
    )
    expert_loss = policy_support_loss_from_response(
        candidate=expert[0], panel=panel
    )
    torch.testing.assert_close(source_loss.response, torch.tensor(0.25))
    torch.testing.assert_close(shared_loss.response, torch.tensor(0.0625))
    torch.testing.assert_close(expert_loss.response, torch.tensor(0.0))


def test_policy_support_barrier_only_penalizes_baseline_regression() -> None:
    source = torch.zeros(2, 3, 4)
    expert = torch.ones(1, 2, 3, 4)
    panel = PolicySupportPanel(
        panel_id=0,
        kind="successful",
        trajectory_path=Path("unused"),
        trajectory_bytes=0,
        selected_indices=(0, 1),
        policy_seed=1,
        source_response=source,
        shared_response=torch.full_like(source, 0.5),
        expert_responses=expert,
        expert_weights=torch.ones(1),
        outcome_weight=0.25,
        source_support_weight=1.0,
        shared_support_weight=1.0,
        learner_success=None,
    )
    improved = policy_support_loss_from_response(
        candidate=torch.full_like(source, 0.75),
        panel=panel,
        preservation=SUPPORT_PRESERVATION_BASELINE_BARRIER,
    )
    torch.testing.assert_close(improved.source_support, torch.tensor(0.0))
    torch.testing.assert_close(improved.shared_support, torch.tensor(0.0))

    regressed_response = torch.full_like(source, -1.0, requires_grad=True)
    regressed = policy_support_loss_from_response(
        candidate=regressed_response,
        panel=panel,
        preservation=SUPPORT_PRESERVATION_BASELINE_BARRIER,
    )
    torch.testing.assert_close(regressed.source_support, torch.tensor(0.75))
    torch.testing.assert_close(regressed.shared_support, torch.tensor(0.9375))
    (regressed.source_support + regressed.shared_support).backward()
    assert regressed_response.grad is not None
    assert float(regressed_response.grad.abs().sum()) > 0


def test_owner_resolved_response_distillation_is_policy_native_and_differentiable() -> None:
    source = torch.zeros(2, 3, 4, 5)
    experts = torch.stack(
        (
            torch.ones_like(source),
            torch.ones_like(source) * 1.2,
        )
    )
    target = 0.25 * experts[0] + 0.75 * experts[1]
    matched = owner_resolved_response_distillation_loss(
        candidate=target,
        source=source,
        experts=experts,
        expert_weights=torch.tensor([0.25, 0.75]),
        outcome_weight=1.0,
    )
    torch.testing.assert_close(matched.loss, torch.tensor(0.0))

    candidate = torch.zeros_like(source, requires_grad=True)
    missed = owner_resolved_response_distillation_loss(
        candidate=candidate,
        source=source,
        experts=experts,
        expert_weights=torch.tensor([0.25, 0.75]),
        outcome_weight=0.25,
    )
    assert float(missed.loss.detach()) > 0
    assert float(missed.active_owner_fraction) == 1.0
    missed.loss.backward()
    assert candidate.grad is not None and float(candidate.grad.abs().sum()) > 0


def test_policy_support_audit_gate_is_task_equal_across_fit_and_held() -> None:
    summary = {
        "panels": 2,
        "candidate_response": 0.5,
        "source_response": 1.0,
        "shared_response": 0.8,
        "consensus_response": 0.1,
    }
    tasks = [
        {
            "fold_role": "fit" if ordinal < 19 else "held_transform_only",
            "summary": {"all": summary, "successful": summary, "learner": None},
        }
        for ordinal in range(24)
    ]
    aggregates, gate = summarize_policy_support_audit(
        tasks=tasks,
        thresholds={
            "minimum_fit_tasks_better_than_source": 13,
            "minimum_fit_tasks_better_than_shared": 13,
            "minimum_held_tasks_better_than_source": 3,
            "minimum_held_tasks_better_than_shared": 3,
        },
    )
    assert gate["passed"] is True
    assert aggregates["fit19"]["all"]["tasks"] == 19
    assert aggregates["held5"]["all"]["tasks"] == 5
