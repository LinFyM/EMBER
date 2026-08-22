import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.compiler import LayerResolvedCompiler, select_compiled_state
from ember.ecp.contracts import TargetFamily, TargetOwner, build_target_owners
from ember.ecp.policy_teacher import PrivilegedPolicyEvidence
from ember.ecp.policy_response import (
    FrozenTargetActivationEffects,
    lora_activation_effects,
    target_activation_effect_distillation_loss,
)
from ember.ecp.program import ECPProgram, VisibleProgramProjector
from ember.ecp.stage0 import ECPVideoEncoderOutput
from ember.ecp.stage1_data import (
    ECPStage1Task,
    build_stage1_schedule,
    gauge_canonicalize_factors,
)
from ember.ecp.stage1 import ECPStage1Model
from ember.ecp.stage1_materialization import resolve_stage1_materialization_config
from ember.ecp.stage1_config import load_stage1_config
from ember.ecp.stage1_prior_calibration import calibrate_prior_heads
from ember.ecp.stage1_objective import (
    effective_update_cosine_matrix,
    exact_effective_update_loss,
)
from ember.ecp.stage1_support import (
    PolicySupportPanel,
    SUPPORT_PRESERVATION_BASELINE_BARRIER,
    policy_support_loss_from_response,
)
from ember.expert_manifold.projection import inspect_projected_task_expert_bank
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    identity_lora_state,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_eval.occupancy_selection import successful_expert_occupancy_tasks
from ember.pi05_eval_contract import TargetTaskContract
from ember.pi05_source_checkpoint import write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]


def _encoded(*, presence: torch.Tensor | None = None) -> ECPVideoEncoderOutput:
    videos, frames, events, owners, width = 2, 3, 8, 38, 128
    return ECPVideoEncoderOutput(
        process=torch.randn(videos, events, owners, width),
        presence=(torch.rand(videos, events) if presence is None else presence),
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


def _tiny_compiler() -> tuple[LayerResolvedCompiler, dict[str, torch.Tensor]]:
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
        LayerResolvedCompiler(
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


def test_failed_v24_stage1_authority_is_sealed() -> None:
    with pytest.raises(
        ValueError, match="unsupported ECP Stage 1 mapping-diverse compiler contract"
    ):
        resolve_stage1_materialization_config(
            REPO_ROOT
            / "configs/pi05_ecp_stage1_layer_resolved_single_surface_compiler_v24.json"
        )


def test_mdco_is_the_only_active_stage1_authority() -> None:
    config = load_stage1_config(
        REPO_ROOT / "configs/pi05_ecp_stage1_mapping_diverse_compiler_oracle.json"
    )
    assert config["data"]["fit_mappings"] == 90
    assert config["roles"]["held_task_ordinals"] == [90, 91, 92, 93, 94]
    assert config["training_ownership"]["policy_teacher_trainable"] is True


def test_mdco_occupancy_selection_covers_all_71_mappings() -> None:
    path = REPO_ROOT / "configs/pi05_ecp_stage1_mdco_nonheld_occupancy_selection.json"
    manifest = json.loads(path.read_text())
    by_task = {}
    for row in manifest["rows"]:
        by_task.setdefault(int(row["task_id"]), str(row["language"]))
    tasks = tuple(
        TargetTaskContract(
            suite="libero_90",
            task_id=task_id,
            split_role="meta_train",
            language=language,
            problem_folder="libero_90",
            bddl_file=f"task_{task_id}.bddl",
            bddl_bytes=1,
            init_states_file=f"task_{task_id}.pruned_init",
            init_states_bytes=1,
            installed_init_state_count=50,
            horizon=400,
            init_state_ids=tuple(range(50)),
        )
        for task_id, language in sorted(by_task.items())
    )
    selected, capture = successful_expert_occupancy_tasks(
        SimpleNamespace(mode="formal", role="nonheld_meta"),
        tasks,
        output_dir=Path("/tmp/mdco-capture"),
        writer_kind="task_expert",
        selection_path=path,
        manifest=manifest,
        rows=manifest["rows"],
    )
    assert len(selected) == 71
    assert sum(len(task.init_state_ids) for task in selected) == 142
    assert capture["training_gradient_use"] is True
    assert capture["category_counts"] == {
        "gained": 58,
        "retained_success": 71,
        "direct_success_fallback": 13,
    }


def test_mdco_projection_binds_only_the_frozen_leave_task_out_panel(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in (
        "stage1_config",
        "stage1_checkpoint",
        "base_projection_manifest",
        "policy_support_bank",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets[name] = {"path": str(path), "bytes": path.stat().st_size}
    base_tasks = []
    projected_tasks = []
    for ordinal in range(24):
        checkpoint = tmp_path / f"checkpoint_{ordinal:02d}"
        base_tasks.append(
            {
                "suite": "libero_train",
                "task_id": ordinal,
                "ordinal": ordinal,
                "global_task_id": ordinal,
                "checkpoint": str(checkpoint),
            }
        )
        if ordinal < 5:
            adapter = tmp_path / f"projected_{ordinal:02d}.safetensors"
            adapter.write_bytes(b"projected")
            projected_tasks.append(
                {
                    "suite": "libero_train",
                    "task_id": ordinal,
                    "ordinal": ordinal,
                    "stage1_ordinal": 90 + ordinal,
                    "global_task_id": ordinal,
                    "expert_checkpoint": str(checkpoint),
                    "projected_adapter": str(adapter),
                    "projected_adapter_bytes": adapter.stat().st_size,
                }
            )
    manifest = tmp_path / "projection.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": (
                "ember_ecp_stage1_mapping_diverse_compiler_oracle_projection_v1"
            ),
            "projection_kind": "ecp_stage1_mapping_diverse_compiler_oracle",
            "repository": {"dirty_paths": []},
            **assets,
            "optimization": {
                "task_visits": 540,
                "fit_task_count": 90,
                "held_task_count": 5,
                "held_shared_gradient_steps": 0,
                "compiler_trainable_during_training": True,
                "visible_program_frozen_during_training": True,
                "policy_teacher_frozen_during_training": False,
                "compiler_frozen_for_materialization": True,
                "single_complete_lora": True,
                "final_lora_averaging": False,
                "rank": 16,
                "all_ranks_writable": True,
                "parameterization": (
                    "one layer-resolved direct-absolute A/B surface with continuous "
                    "static/process fusion"
                ),
                "content_address_separated": True,
                "query_content_modulated": True,
                "policy_support_teacher": True,
                "raw_factor_amplitude_retained": True,
                "fixed_rank_partition": False,
                "second_adapter_deployed": False,
                "objective_phase": (
                    "task_equal_mapping_diverse_q_pi_compiler_identification"
                ),
            },
            "information_wall": {
                "role": "development_train_leave_task_out_oracle_only",
                "deployment_carrier": False,
                "privileged_q_pi": (
                    "fit90 shared training and frozen held5 inference only"
                ),
                "second_adapter_deployed": False,
            },
            "tasks": projected_tasks,
        },
    )
    projected = inspect_projected_task_expert_bank(
        {
            "tasks": base_tasks,
            "information_wall": {"evaluation_role": "development_train"},
        },
        manifest,
    )
    assert len(projected["tasks"]) == 5
    assert projected["arm"] == "ecp_stage1_mdco_tv540"
    assert all("projected_adapter" in row for row in projected["tasks"])


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
    assert all(
        value.shape[0] == 2 for value in output.member_compilation.state.values()
    )
    attention = float(output.consensus_compilation.exact_owner_attention.detach())
    assert 0.0 <= attention <= 1.0
    assert any(
        float(
            (output.prior_compilation.state[name][0] - target)
            .detach()
            .abs()
            .sum()
        )
        > 0.0
        for name, target in template.items()
    )


def test_stage1_decision_prefix_is_task_equal() -> None:
    tasks = tuple(
        ECPStage1Task(
            ordinal=ordinal,
            global_task_id=ordinal,
            suite="suite",
            task_id=ordinal,
            language=f"task {ordinal}",
            path=Path(f"task_{ordinal}.hdf5"),
            expected_bytes=1,
            episode_lengths=tuple(
                40 + ordinal % 10 + index for index in range(50)
            ),
            fold_role="fit",
        )
        for ordinal in range(90)
    )
    config = {
        "roles": {"fit_task_ordinals": list(range(90))},
        "data": {
            "frame_stride": 5,
            "visible_videos_per_visit": 2,
            "pair_seed": 17,
        },
        "optimization": {
            "visits_per_fit_task": 6,
            "stage_stop_task_visits": [540],
            "seed": 23,
        },
    }
    schedule = build_stage1_schedule(
        config=config,
        tasks=tasks,
        world_size=6,
        total_task_visits=540,
        mode="formal",
    )
    counts = Counter(ordinal for ordinal, _ in schedule)
    assert counts == Counter({ordinal: 6 for ordinal in range(90)})


def test_exact_effective_update_loss_is_gauge_invariant_and_zero_on_identity() -> None:
    contract, _, template = _contract_and_states()
    target = _expert_evidence(template).member_states
    assert float(exact_effective_update_loss(target, target, contract)) < 1e-6
    transformed = {}
    scale = 2.0
    for name, value in target.items():
        transformed[name] = value * scale if ".lora_A." in name else value / scale
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


def test_prior_and_full_programs_share_direct_absolute_factor_heads() -> None:
    contract, owners, template = _contract_and_states()
    for value in template.values():
        value.normal_(std=0.01)
    compiler = LayerResolvedCompiler(owners, contract, template)
    common = {
        "language": torch.randn(1, 38, 128),
        "scene": torch.randn(1, 38, 128),
        "process": torch.randn(1, 8, 38, 128),
        "uncertainty": torch.ones(1, 8, 38, 128),
    }
    prior = compiler(ECPProgram(**common, presence=torch.zeros(1, 8))).state
    full = compiler(ECPProgram(**common, presence=torch.ones(1, 8))).state
    assert any(
        float((prior[name][0] - target).detach().abs().sum()) > 0.0
        for name, target in template.items()
    )
    for name in template:
        torch.testing.assert_close(full[name], prior[name])
    with torch.no_grad():
        compiler.static_process_interaction.weight.normal_(std=0.01)
    learned_prior = compiler(
        ECPProgram(**common, presence=torch.zeros(1, 8))
    ).state
    learned_full = compiler(ECPProgram(**common, presence=torch.ones(1, 8))).state
    for name in template:
        torch.testing.assert_close(learned_prior[name], prior[name])
    assert any(
        float((learned_full[name] - learned_prior[name]).detach().abs().sum()) > 0.0
        for name in template
    )


def test_target_local_absolute_factor_heads_retain_content_amplitude() -> None:
    compiler, _ = _tiny_compiler()
    program = _tiny_program()
    first = compiler(program).state
    with torch.no_grad():
        compiler.factor_a["owner_00"].weight.mul_(2.0)
        compiler.factor_b["owner_00"].weight.mul_(3.0)
    second = compiler(program).state
    torch.testing.assert_close(
        second["tiny.lora_A.default.weight"],
        2.0 * first["tiny.lora_A.default.weight"],
    )
    torch.testing.assert_close(
        second["tiny.lora_B.default.weight"],
        3.0 * first["tiny.lora_B.default.weight"],
    )


def test_language_and_scene_condition_process_value_queries() -> None:
    compiler, _ = _tiny_compiler()
    first_program = _tiny_program()
    second_program = ECPProgram(
        **{
            **first_program.__dict__,
            "language": first_program.language
            + torch.randn_like(first_program.language),
            "scene": first_program.scene + torch.randn_like(first_program.scene),
        }
    )
    first = compiler(first_program).state["tiny.lora_A.default.weight"]
    second = compiler(second_program).state["tiny.lora_A.default.weight"]
    assert float((first - second).detach().abs().sum()) > 0.0


def test_target_local_absolute_factor_heads_receive_first_step_gradient() -> None:
    compiler, _ = _tiny_compiler()
    program = _tiny_program()
    first = compiler(program)
    dense = (
        first.state["tiny.lora_B.default.weight"]
        @ first.state["tiny.lora_A.default.weight"]
    )
    target = torch.randn_like(dense)
    (dense - target).square().mean().backward()
    assert float(compiler.factor_a["owner_00"].weight.grad.abs().sum()) > 0.0
    assert float(compiler.factor_b["owner_00"].weight.grad.abs().sum()) > 0.0


def test_absent_process_is_exactly_removed_before_single_surface_fusion() -> None:
    compiler, _ = _tiny_compiler()
    first = _tiny_program()
    first = ECPProgram(**{**first.__dict__, "presence": torch.zeros(1, 2)})
    changed = ECPProgram(
        **{
            **first.__dict__,
            "process": 1_000 * torch.randn_like(first.process),
            "uncertainty": 1_000 * torch.rand_like(first.uncertainty),
        }
    )
    left = compiler(first).state
    right = compiler(changed).state
    for name in left:
        torch.testing.assert_close(left[name], right[name])


def test_static_process_interaction_receives_full_program_gradient() -> None:
    compiler, _ = _tiny_compiler()
    state = compiler(_tiny_program()).state
    loss = sum(value.float().square().mean() for value in state.values())
    loss.backward()
    gradient = compiler.static_process_interaction.weight.grad
    assert gradient is not None and float(gradient.abs().sum()) > 0.0


def test_fit_prior_calibration_is_minimum_change_and_reduces_residual() -> None:
    compiler, _ = _tiny_compiler()
    program = _tiny_program()
    summary = calibrate_prior_heads(
        compiler, {1: program}, relative_ridge=1e-4
    )
    assert summary["fit_programs"] == 1
    assert summary["relative_residual_after"] < summary["relative_residual_before"]


def test_query_content_modulation_reaches_rank_outputs() -> None:
    contract, owners, template = _contract_and_states()
    compiler = LayerResolvedCompiler(owners, contract, template)
    for value in template.values():
        value.normal_(std=0.01)
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
    expert_loss = policy_support_loss_from_response(candidate=expert[0], panel=panel)
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


def test_target_activation_effect_is_gauge_invariant_local_and_differentiable() -> None:
    contract = SmolVLALoRAContract(
        targets=(
            LoRATarget("first", in_features=3, out_features=4),
            LoRATarget("second", in_features=4, out_features=2),
        ),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=1,
    )
    state = {
        "first.lora_A.default.weight": torch.randn(2, 3),
        "first.lora_B.default.weight": torch.randn(4, 2),
        "second.lora_A.default.weight": torch.randn(2, 4),
        "second.lora_B.default.weight": torch.randn(2, 2),
    }
    reference = {
        "first": torch.randn(2, 3, 3),
        "second": torch.randn(2, 3, 4),
    }
    baseline = lora_activation_effects(
        state=state, reference_inputs=reference, contract=contract
    )
    gauge = {name: value.clone() for name, value in state.items()}
    for name in gauge:
        gauge[name] = gauge[name] * (3.0 if ".lora_A." in name else 1.0 / 3.0)
    transformed = lora_activation_effects(
        state=gauge, reference_inputs=reference, contract=contract
    )
    for name in baseline:
        torch.testing.assert_close(transformed[name], baseline[name])

    candidate = {name: value.clone().requires_grad_() for name, value in state.items()}
    targets = FrozenTargetActivationEffects(
        reference_inputs=reference,
        expert_effects={
            "first": (baseline["first"] + 0.25)[None],
            "second": baseline["second"][None],
        },
    )
    missed = target_activation_effect_distillation_loss(
        candidate_state=candidate,
        targets=targets,
        contract=contract,
        expert_weights=torch.ones(1),
        outcome_weight=0.25,
    )
    assert float(missed.loss.detach()) > 0
    missed.loss.backward()
    assert all(
        candidate[name].grad is not None and float(candidate[name].grad.abs().sum()) > 0
        for name in candidate
        if name.startswith("first")
    )
    assert all(
        candidate[name].grad is not None
        and float(candidate[name].grad.abs().sum()) == 0
        for name in candidate
        if name.startswith("second")
    )
