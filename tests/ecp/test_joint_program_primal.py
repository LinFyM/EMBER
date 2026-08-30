from pathlib import Path
from types import SimpleNamespace

import torch

from ember.ecp.bank_conditioning.primal_capacity import (
    BANK_INTERACTION_CONTROL_SCHEMA,
)
from ember.ecp.joint_program_primal.evaluation import (
    _language_program,
    _normalized,
    _task_conditions,
    balanced_task_assignments,
    load_joint_program_primal_gate,
)
from ember.ecp.joint_program_primal.evaluation_gate import _interaction
from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_TASK_IDS,
    SCORER_NATIVE_HEADS_ONLY,
    _scorer_parameter_ownership,
    fixed_routing_program,
    fixed_routing_token,
    load_routing_control_config,
)
from ember.ecp.joint_program_primal.routing_initialization import (
    FUNCTIONAL_CODE_INITIALIZATION,
    R10_FUNCTIONAL_CONTENT,
    R9_STABLE_CONTENT,
    minimum_norm_head_solution,
)
from ember.ecp.joint_program_primal.runtime import (
    BANK_COMPATIBILITY_SCHEMA,
    CHART_RECONNECT_SCHEMA,
    FUNCTIONAL_CHART_ACQUISITION_SCHEMA,
    FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA,
    FUNCTIONAL_REFINEMENT_SCHEMA,
    RAW_STAGE0_SUFFICIENCY_SCHEMA,
    SCORER_ALL_PARAMETERS,
    SCORER_FEATURE_CHART_ONLY,
    SCORER_NATIVE_HEADS_ONLY as JOINT_NATIVE_HEADS_ONLY,
    _joint_parameter_ownership,
    load_joint_program_primal_config,
)
from ember.ecp.joint_program_primal.raw_stage0 import (
    RAW_STAGE0_PROGRAM_INPUT,
    prepare_raw_stage0_primal_condition,
)
from ember.ecp.joint_program_primal.routing_control_evaluation import (
    _training_world_size,
    routing_task_assignments,
)
from ember.ecp.joint_program_primal.train_step import (
    _outer_update_cosine,
    counterfactual_arm,
    counterfactual_hinge,
    counterfactual_task_pairs,
    joint_task_group,
)
from ember.ecp.bank_conditioning.mapping import MappingCondition
from ember.ecp.natural_program import (
    FrozenProgramEvidence,
    NaturalProgram,
    NaturalProgramModel,
)


def test_joint_task_schedule_is_role_balanced_and_task_equal() -> None:
    runtime = SimpleNamespace(
        config={
            "task_split": {
                "gradient_meta": [1, 8, 9, 32, 52],
                "gradient_target": [72, 73, 75, 93, 94],
            }
        }
    )
    groups = tuple(joint_task_group(runtime, step) for step in range(5))
    assert all(len(group) == len(set(group)) == 6 for group in groups)
    assert all(sum(task < 72 for task in group) == 3 for group in groups)
    counts = {
        task: sum(task in group for group in groups)
        for task in (1, 8, 9, 32, 52, 72, 73, 75, 93, 94)
    }
    assert set(counts.values()) == {3}


def test_counterfactual_schedule_is_same_role_cyclic_and_alternating() -> None:
    runtime = SimpleNamespace(
        config={
            "task_split": {
                "gradient_meta": [1, 8, 9, 32, 52],
                "gradient_target": [72, 73, 75, 93, 94],
            }
        }
    )
    group = joint_task_group(runtime, 0)
    pairs = counterfactual_task_pairs(runtime, group)
    assert pairs == {1: 8, 8: 9, 9: 1, 72: 73, 73: 75, 75: 72}
    assert counterfactual_arm(0) == "wrong_program"
    assert counterfactual_arm(1) == "wrong_bank"
    active, gap, margin, hinge = counterfactual_hinge(
        correct_loss=0.20,
        negative_loss=0.205,
        margin_scale=0.10,
        normalized_margin=0.10,
    )
    assert active is True
    assert torch.allclose(
        torch.tensor((gap, margin, hinge)), torch.tensor((0.005, 0.01, 0.005))
    )
    assert counterfactual_hinge(
        correct_loss=0.20,
        negative_loss=0.22,
        margin_scale=0.10,
        normalized_margin=0.10,
    )[0] is False


def test_language_control_removes_every_video_program_field() -> None:
    program = NaturalProgram(
        p_lang=torch.randn(38, 4),
        p_scene=torch.randn(38, 4),
        p_process=torch.randn(8, 38, 4),
        rho=torch.rand(8),
        tau=torch.rand(8, 2),
        sigma=torch.randn(8, 38, 4),
    )
    control = _language_program(program)
    assert torch.equal(control.p_lang, program.p_lang)
    assert torch.count_nonzero(control.p_scene) == 0
    assert torch.count_nonzero(control.p_process) == 0
    assert torch.count_nonzero(control.sigma) == 0
    assert torch.allclose(control.rho, torch.full((8,), 0.125))
    assert torch.allclose(control.tau[0], torch.tensor([0.0, 0.125]))
    assert torch.allclose(control.tau[-1], torch.tensor([0.875, 1.0]))


def test_functional_recovery_and_interaction_use_one_free_primal_measure() -> None:
    def arm(generated: float) -> dict[str, float]:
        return _normalized(
            {
                "carrier_loss": 1.0,
                "generated_loss": generated,
                "benefit_over_carrier": 1.0 - generated,
                "visits": [],
            },
            free_loss=0.5,
        )

    controls = {
        "primary_correct": arm(0.6),
        "wrong_program_correct_bank": arm(0.8),
        "correct_program_wrong_bank": arm(0.85),
        "wrong_program_wrong_bank": arm(0.95),
    }
    assert controls["primary_correct"]["functional_recovery"] == 0.8
    assert abs(_interaction(controls) - 0.2) < 1e-12


def test_true_task_holdout_uses_lowest_three_sealed_videos() -> None:
    sealed = (4, 25, 27, 31, 32, 38, 39, 40, 46)
    rows = tuple(MappingCondition(2, "meta_fit", demo, 10 + demo) for demo in sealed)
    runtime = SimpleNamespace(
        mapping_split=SimpleNamespace(
            fit_by_task={}, video_held_by_task={}, task_held=rows
        ),
        # Functional-panel action episodes are a separate authority from the
        # sealed native-teacher videos used to compile the Program and bank.
        panels={
            2: SimpleNamespace(
                role="meta_fit", program_video_demos=tuple(range(9))
            )
        },
    )
    selected = _task_conditions(runtime, 2)
    assert tuple(row.video_demo for row in selected) == (4, 25, 27)


def test_six_worker_gate_assignment_keeps_exactly_two_tasks_per_worker() -> None:
    task_ids = (1, 2, 8, 9, 32, 52, 72, 73, 74, 75, 93, 94)
    fit_ids = set(task_ids) - {2, 74}
    fit = {
        task: tuple(
            MappingCondition(task, "meta_fit" if task < 72 else "target_fit", demo, 10)
            for demo in (0, 1)
        )
        for task in fit_ids
    }
    held = {
        task: (
            MappingCondition(
                task, "meta_fit" if task < 72 else "target_fit", 2, 10
            ),
        )
        for task in fit_ids
    }
    task_held = tuple(
        MappingCondition(task, "meta_fit" if task < 72 else "target_fit", demo, 10)
        for task in (2, 74)
        for demo in (0, 1, 2)
    )
    runtime = SimpleNamespace(
        mapping_split=SimpleNamespace(
            fit_by_task=fit,
            video_held_by_task=held,
            task_held=task_held,
        ),
        panels={
            task: SimpleNamespace(role="meta_fit" if task < 72 else "target_fit")
            for task in task_ids
        },
    )
    assignments = balanced_task_assignments(runtime, worker_count=6)
    assert all(len(tasks) == 2 for tasks in assignments)
    assert {task for tasks in assignments for task in tasks} == set(task_ids)


def test_routing_control_tokens_are_fixed_mean_zero_and_orthogonal() -> None:
    tokens = torch.stack([fixed_routing_token(task) for task in ROUTING_TASK_IDS])
    assert torch.equal(tokens.mean(1), torch.zeros(len(ROUTING_TASK_IDS)))
    assert torch.equal(tokens @ tokens.T, 128 * torch.eye(len(ROUTING_TASK_IDS)))

    runtime = SimpleNamespace(
        owners=tuple(range(38)),
        compiler=SimpleNamespace(event_slots=8, program_width=128),
        context=SimpleNamespace(device=torch.device("cpu")),
    )
    program = fixed_routing_program(runtime, ROUTING_TASK_IDS[0])
    assert program.p_lang.shape == (38, 128)
    assert program.p_scene.shape == (38, 128)
    assert program.p_process.shape == (8, 38, 128)
    assert program.sigma.shape == (8, 38, 128)
    assert torch.allclose(program.rho, torch.full((8,), 0.125))
    assert torch.allclose(program.tau[0], torch.tensor([0.0, 0.125]))
    assert torch.allclose(program.tau[-1], torch.tensor([0.875, 1.0]))


def test_routing_control_gate_balances_only_ten_gradient_tasks() -> None:
    costs = {
        "1": 197,
        "8": 195,
        "9": 97,
        "32": 199,
        "52": 195,
        "72": 195,
        "73": 99,
        "75": 198,
        "93": 108,
        "94": 105,
    }
    assignments = routing_task_assignments(
        worker_count=6, task_cost_seconds=costs
    )
    assert max(map(len, assignments)) == 2
    assert {task for row in assignments for task in row} == set(ROUTING_TASK_IDS)
    loads = [sum(costs[str(task)] for task in row) for row in assignments]
    assert max(loads) == 303


def test_routing_control_gate_accepts_the_recorded_flexible_training_world() -> None:
    runtime = SimpleNamespace(
        config={"profile": {"allowed_world_sizes": [1, 2, 3, 4, 5, 6]}}
    )
    run_contract = {
        "world_topology": [{"rank": rank} for rank in range(3)]
    }
    manifest = {"world_size": 3}
    assert _training_world_size(runtime, run_contract, manifest) == 3


def test_routing_control_critic_is_fit_only_and_never_a_deployment_input() -> None:
    root = Path(__file__).resolve().parents[2]
    critic_control = load_routing_control_config(
        root / "configs/pi05_ecp_routing_token_grouped_decoder_r3_v1.json"
    )
    assert (
        critic_control["model"]["output_primal_decoder"]
        == "owner_group_specific_linear_heads"
    )
    critic = critic_control["optimization"]["privileged_critic"]
    assert critic["kind"] == "fit_only_set_valued_paired_update_direction"
    assert critic["weight"] == 0.2
    assert critic["deployment_input"] is False
    assert critic["held_or_validation_reads"] is False


def test_routing_functional_code_head_fit_is_exact_and_minimum_norm() -> None:
    generator = torch.Generator().manual_seed(20260830)
    features = torch.randn(8, 16, generator=generator)
    labels = torch.randn(8, 5, generator=generator)
    weight, report = minimum_norm_head_solution(features, labels)
    assert weight.shape == (5, 16)
    assert report["rank"] == 8
    assert report["fp64_relative_fit_error"] < 1e-10
    assert torch.allclose(
        features.double() @ weight.double().T,
        labels.double(),
        atol=1e-10,
        rtol=1e-10,
    )
    null_projection = torch.eye(16, dtype=torch.float64) - torch.linalg.pinv(
        features.double()
    ) @ features.double()
    assert torch.linalg.norm(weight.double() @ null_projection) < 1e-10


def test_routing_r4_uses_functional_code_initialization_without_critic() -> None:
    root = Path(__file__).resolve().parents[2]
    control = load_routing_control_config(
        root / "configs/pi05_ecp_routing_token_functional_code_init_r4_v1.json"
    )
    assert (
        control["model"]["primal_scorer_initialization"]
        == FUNCTIONAL_CODE_INITIALIZATION
    )
    assert "privileged_critic" not in control["optimization"]
    assert control["optimization"]["loss"] == "two_correct_fit_video_functional_only"
    assert control["information_wall"]["task_local_fixed_scales_used"] is False


def test_routing_r5_freezes_feature_chart_and_trains_only_native_heads() -> None:
    root = Path(__file__).resolve().parents[2]
    control = load_routing_control_config(
        root
        / "configs/pi05_ecp_routing_token_functional_code_chart_frozen_r5_v1.json"
    )
    assert (
        control["model"]["primal_scorer_trainable_partition"]
        == SCORER_NATIVE_HEADS_ONLY
    )

    class TinyCompiler(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scorer = torch.nn.Module()
            scorer.feature_chart = torch.nn.Linear(4, 4)
            scorer.input_primal_heads = torch.nn.ModuleList(
                [torch.nn.Linear(4, 3, bias=False)]
            )
            scorer.output_primal_heads = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [torch.nn.Linear(4, 2, bias=False)]
                    )
                ]
            )
            self.primal_scorer = scorer

    program = torch.nn.Linear(2, 2)
    compiler = TinyCompiler()
    writer, trainable, frozen = _scorer_parameter_ownership(
        program, compiler, partition=SCORER_NATIVE_HEADS_ONLY
    )
    expected = {
        id(parameter)
        for module in (
            compiler.primal_scorer.input_primal_heads,
            compiler.primal_scorer.output_primal_heads,
        )
        for parameter in module.parameters()
    }
    assert {id(parameter) for parameter in trainable} == expected
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in frozen)
    assert all(
        not parameter.requires_grad
        for parameter in compiler.primal_scorer.feature_chart.parameters()
    )
    assert set(writer.state_dict()) == {
        "primal_scorer.feature_chart.weight",
        "primal_scorer.feature_chart.bias",
        "primal_scorer.input_primal_heads.0.weight",
        "primal_scorer.output_primal_heads.0.0.weight",
    }


def test_r6_reconnects_natural_program_without_unfreezing_feature_chart() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root / "configs/pi05_ecp_natural_program_chart_reconnect_r6_v1.json"
    )
    assert config["schema_version"] == CHART_RECONNECT_SCHEMA
    assert config["optimization"]["loss"] == (
        "generated_rank16_cross_episode_pi05_flow_only"
    )
    assert "counterfactual" not in config["optimization"]["joint"]
    assert config["information_wall"]["fixed_routing_token_deployment_input"] is False

    class TinyProgram(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.language_reader = torch.nn.Linear(2, 2)
            self.scene_reader = torch.nn.Linear(2, 2)
            self.process_fusion = torch.nn.Sequential(torch.nn.Linear(2, 2))
            self.aligner = torch.nn.Linear(2, 2)

    class TinyCompiler(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scorer = torch.nn.Module()
            scorer.feature_chart = torch.nn.Linear(4, 4)
            scorer.input_primal_heads = torch.nn.ModuleList(
                [torch.nn.Linear(4, 3, bias=False)]
            )
            scorer.output_primal_heads = torch.nn.ModuleList(
                [torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False)])]
            )
            self.primal_scorer = scorer
            self.scale_head = torch.nn.Linear(2, 1)

    program = TinyProgram()
    compiler = TinyCompiler()
    _, trainable, frozen = _joint_parameter_ownership(
        program,
        compiler,
        scorer_partition=JOINT_NATIVE_HEADS_ONLY,
    )
    expected_modules = (
        program.language_reader,
        program.scene_reader,
        program.process_fusion,
        program.aligner,
        compiler.primal_scorer.input_primal_heads,
        compiler.primal_scorer.output_primal_heads,
    )
    expected = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert {id(parameter) for parameter in trainable} == expected
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in frozen)
    assert all(not parameter.requires_grad for parameter in program.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in program.decoder.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in compiler.primal_scorer.feature_chart.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in compiler.scale_head.parameters()
    )


def test_r7_acquires_content_chart_with_fixed_validated_heads() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root / "configs/pi05_ecp_functional_code_chart_acquisition_r7_v1.json"
    )
    assert config["schema_version"] == FUNCTIONAL_CHART_ACQUISITION_SCHEMA
    assert config["optimization"]["loss"] == (
        "fit_only_functional_code_outer_direction_only"
    )
    assert config["information_wall"]["native_heads_frozen"] is True

    class TinyProgram(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.language_reader = torch.nn.Linear(2, 2)
            self.scene_reader = torch.nn.Linear(2, 2)
            self.process_fusion = torch.nn.Sequential(torch.nn.Linear(2, 2))
            self.aligner = torch.nn.Linear(2, 2)

    class TinyCompiler(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            scorer = torch.nn.Module()
            scorer.feature_chart = torch.nn.Linear(4, 4)
            scorer.input_primal_heads = torch.nn.ModuleList(
                [torch.nn.Linear(4, 3, bias=False)]
            )
            scorer.output_primal_heads = torch.nn.ModuleList(
                [torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False)])]
            )
            self.primal_scorer = scorer
            self.scale_head = torch.nn.Linear(2, 1)

    program = TinyProgram()
    compiler = TinyCompiler()
    _, trainable, frozen = _joint_parameter_ownership(
        program,
        compiler,
        scorer_partition=SCORER_FEATURE_CHART_ONLY,
    )
    expected_modules = (
        program.language_reader,
        program.scene_reader,
        program.process_fusion,
        program.aligner,
        compiler.primal_scorer.feature_chart,
    )
    expected = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
    }
    assert {id(parameter) for parameter in trainable} == expected
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in frozen)
    assert all(
        not parameter.requires_grad
        for parameter in compiler.primal_scorer.input_primal_heads.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in compiler.primal_scorer.output_primal_heads.parameters()
    )


def test_r9_jointly_acquires_functional_code_from_stable_chart() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root
        / "configs/pi05_ecp_functional_code_stable_chart_joint_r9_v1.json"
    )
    assert config["schema_version"] == FUNCTIONAL_CODE_STABLE_JOINT_SCHEMA
    assert config["model"]["primal_scorer_trainable_partition"] == (
        SCORER_ALL_PARAMETERS
    )
    assert config["model"]["primal_scorer_initialization"] == (
        "r5_shared_functional_chart_step110"
    )
    assert config["optimization"]["loss"] == (
        "fit_only_functional_code_outer_direction_only"
    )
    assert config["information_wall"]["native_heads_trainable"] is True
    assert config["information_wall"][
        "absolute_outer_code_target_anchors_moving_heads"
    ] is True
    assert config["information_wall"][
        "stable_r5_shared_chart_initialization"
    ] is True
    gate = load_joint_program_primal_gate(
        root
        / "configs/pi05_ecp_functional_code_stable_chart_joint_r9_gate_v1.json"
    )
    assert gate["checkpoint_optimizer_steps"] == [70, 110]
    assert gate["gate"]["true_task_held_mean_minimum"] == 0.40


def test_r10_refines_r9_content_with_functional_flow_only() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root
        / "configs/pi05_ecp_r9_initialized_functional_refinement_r10_v1.json"
    )
    assert config["schema_version"] == FUNCTIONAL_REFINEMENT_SCHEMA
    assert config["model"]["program_initialization"] == R9_STABLE_CONTENT
    assert config["model"]["primal_scorer_initialization"] == R9_STABLE_CONTENT
    assert config["model"]["primal_scorer_trainable_partition"] == (
        JOINT_NATIVE_HEADS_ONLY
    )
    assert config["optimization"]["loss"] == (
        "generated_rank16_cross_episode_pi05_flow_only"
    )
    assert "counterfactual" not in config["optimization"]["joint"]
    assert config["information_wall"]["primal_scorer_feature_chart_frozen"] is True
    assert config["information_wall"]["outer_code_loss_active"] is False
    gate = load_joint_program_primal_gate(
        root
        / "configs/pi05_ecp_r9_initialized_functional_refinement_r10_gate_v1.json"
    )
    assert gate["checkpoint_optimizer_steps"] == [70, 110]
    assert gate["gate"]["true_task_held_mean_minimum"] == 0.40


def test_r11_swaps_only_to_raw_frozen_stage0_functional_input() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root / "configs/pi05_ecp_raw_stage0_sufficiency_r11_v1.json"
    )
    assert config["schema_version"] == RAW_STAGE0_SUFFICIENCY_SCHEMA
    assert config["model"]["program_input"] == RAW_STAGE0_PROGRAM_INPUT
    assert config["model"]["program_initialization"] == R9_STABLE_CONTENT
    assert config["model"]["primal_scorer_initialization"] == R9_STABLE_CONTENT
    assert config["model"]["primal_scorer_trainable_partition"] == (
        JOINT_NATIVE_HEADS_ONLY
    )
    assert config["optimization"]["loss"] == (
        "generated_rank16_cross_episode_pi05_flow_only"
    )
    assert config["information_wall"]["diagnostic_only"] is True
    assert config["information_wall"]["deployment_writer"] is False
    gate = load_joint_program_primal_gate(
        root / "configs/pi05_ecp_raw_stage0_sufficiency_r11_gate_v1.json"
    )
    assert gate["checkpoint_optimizer_steps"] == [70, 110]


def test_r12_learns_cross_video_bank_compatibility_from_r10() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root / "configs/pi05_ecp_bank_compatibility_r12_v1.json"
    )
    assert config["schema_version"] == BANK_COMPATIBILITY_SCHEMA
    assert config["model"]["program_initialization"] == R10_FUNCTIONAL_CONTENT
    assert config["model"]["primal_scorer_initialization"] == (
        R10_FUNCTIONAL_CONTENT
    )
    assert config["model"]["primal_scorer_trainable_partition"] == (
        JOINT_NATIVE_HEADS_ONLY
    )
    compatibility = config["optimization"]["joint"]["bank_compatibility"]
    assert compatibility["positive_pairing"] == (
        "each_fit_program_to_other_same_task_fit_bank"
    )
    assert compatibility["correct_functional_operator"] == (
        "full_inverse_teacher_forced"
    )
    assert compatibility["deployment_operator"] == (
        "hard_full_if_supported_else_half"
    )
    assert config["information_wall"]["action_meta_installed"] is False
    gate = load_joint_program_primal_gate(
        root / "configs/pi05_ecp_bank_compatibility_r12_gate_v1.json"
    )
    assert gate["gate"]["matched_full_route_fraction_minimum"] == 0.80
    assert gate["gate"]["mismatched_full_route_fraction_maximum"] == 0.20


def test_bank_interaction_positive_control_uses_fixed_half_operator() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_joint_program_primal_config(
        root / "configs/pi05_ecp_bank_interaction_positive_control_v1.json"
    )
    assert config["schema_version"] == BANK_INTERACTION_CONTROL_SCHEMA
    assert config["model"]["inverse_covariance_power"] == 0.5
    assert config["optimization"]["task_local_positive_control"][
        "initialization"
    ] == "fit_symmetric_transport"
    assert config["information_wall"]["held_video_backward_calls"] == 0
    assert config["information_wall"]["wrong_bank_backward_calls"] == 0
    assert config["information_wall"]["action_meta_installed"] is False


def test_raw_stage0_view_preserves_direct_event_fields_and_k1_time() -> None:
    model = NaturalProgramModel(
        torch.nn.Identity(),
        prefix_width=6,
        width=4,
        owners=3,
        event_slots=2,
    )
    generator = torch.Generator().manual_seed(20260830)
    process = torch.randn(2, 1, 2, 3, 4, generator=generator)
    uncertainty = torch.rand(2, 1, 2, 3, 4, generator=generator) + 0.1
    presence = torch.rand(2, 1, 2, generator=generator) + 0.1
    posterior = torch.rand(2, 1, 3, 2, generator=generator)
    posterior = posterior / posterior.sum(-1, keepdim=True)
    evidence = FrozenProgramEvidence(
        language_embeddings=torch.randn(1, 5, 6, generator=generator),
        language_mask=torch.ones(1, 5, dtype=torch.bool),
        patch_states=torch.randn(1, 3, 2, 4, generator=generator),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        process=process,
        uncertainty=uncertainty,
        presence=presence,
        state_posterior=posterior,
        frame_indices=torch.tensor([0, 5, 10]),
        raw_frame_counts=torch.tensor([11]),
        video_offsets=torch.tensor([0, 3]),
        video_set_offsets=torch.tensor([0, 1]),
        frame_condition_ids=torch.zeros(3, dtype=torch.long),
    )
    program, output = prepare_raw_stage0_primal_condition(
        program_model=model,
        condition=SimpleNamespace(evidence=evidence, videos=(object(),)),
        query_times=torch.linspace(0, 1, 5)[None],
    )
    expected_sigma = (
        uncertainty.square().mean(0)
        + (process - process.mean(0, keepdim=True)).square().mean(0)
    )[0].clamp_min(1e-6).sqrt()
    assert torch.equal(program.p_process, process.mean(0)[0])
    assert torch.equal(program.rho, presence.mean(0)[0])
    assert torch.allclose(program.sigma, expected_sigma)
    assert program.tau.shape == (2, 2)
    assert program.p_lang.shape == (3, 4)
    assert program.p_scene.shape == (3, 4)
    assert output.program.p_process.shape == (1, 2, 3, 4)


def test_functional_code_outer_cosine_is_rank_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(7)
    input_code = torch.randn(4, 7, generator=generator)
    output_code = torch.randn(2, 4, 5, generator=generator)
    permutation = torch.tensor([2, 0, 3, 1])
    signs = torch.tensor([1.0, -1.0, -1.0, 1.0])
    predicted_input = input_code[permutation] * signs[:, None]
    predicted_output = output_code[:, permutation] * signs[None, :, None]
    scale = torch.ones(4)
    assert torch.allclose(
        _outer_update_cosine(
            predicted_input,
            predicted_output,
            input_code,
            output_code,
            scale,
        ),
        torch.tensor(1.0),
        atol=1e-6,
    )
