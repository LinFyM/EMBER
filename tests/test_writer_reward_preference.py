from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from fixtures.writer_model import _inputs, _model as _writer_model
import ember.writer.reward_cycle as reward_cycle
from ember.reward.expert_teacher import pad_expert_lora_to_public_rank
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    expected_lora_state_shapes,
    inject_task_lora,
)
from ember.reward.rollout import (
    RewardTrajectory,
    query_successful_expert_occupancy_actions,
)
from ember.writer.as_step import parameter_layout
from ember.writer.reward_gradient_update import (
    apply_reward_step,
    lora_response,
    median_capped_task_mean,
    preconditioned_candidate_commitment,
)
from ember.writer.reward_preference import (
    SuccessfulExpertOccupancyCreditSummary,
    cross_video_gradient_geometry,
    functional_successful_expert_occupancy_gradient,
    functional_successful_expert_occupancy_objective,
    mean_cross_video_task_gradient,
    stratified_occupancy_weights,
    unit_residual_endpoint_distillation,
)


def test_formal_credit_retains_all_four_views_for_global_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo_sets = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
    runtime = SimpleNamespace(
        args=SimpleNamespace(mode="formal"),
        config={"data": {"credit_views_per_active_task": 4}},
        video_schedule=SimpleNamespace(
            cross_video_credit_demos_for_task_visit=lambda *args, **kwargs: demo_sets
        ),
    )
    task = SimpleNamespace(global_task_id=9)
    states = tuple(object() for _ in range(4))
    offsets = tuple(torch.tensor([index, index + 1]) for index in range(4))
    packed = (None, None, None, offsets[0])

    encoded_index = iter(range(1, 4))

    def encode_condition(*args: object, **kwargs: object) -> tuple[object, ...]:
        index = next(encoded_index)
        return (None, None, None, offsets[index]), {}, states[index], {}

    differentiated_index = iter(range(4))

    def differentiate_view(
        *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, object]:
        index = next(differentiated_index)
        return torch.tensor(
            [float(index + 1)]
        ), SuccessfulExpertOccupancyCreditSummary(
            objective=float(index),
            expert_action_distance=float(index + 1),
            successful_trajectories=1,
            selected_credit_states=8,
            replay_rows=8,
            successful_action_steps=8,
            matched_expert_student_action_rms=1.0,
            functional_policy_forwards=1,
            functional_policy_backwards=1,
            lora_gradient_rms=1.0,
        )

    monkeypatch.setattr(reward_cycle, "_encode_candidate_condition", encode_condition)
    monkeypatch.setattr(reward_cycle, "_differentiate_credit_view", differentiate_view)
    gradients, rows, views, observed_demos = reward_cycle._differentiate_credit_views(
        runtime,
        task,
        0,
        demo_sets[0],
        packed,
        states[0],
        {},
        {},
        torch.tensor([0]),
        torch.zeros(1),
    )

    assert len(gradients) == len(rows) == len(views) == 4
    assert observed_demos == demo_sets
    assert tuple(view.conditioning_state for view in views) == states
    assert tuple(view.before_credit_objective for view in views) == (0.0, 1.0, 2.0, 3.0)
    for view, expected_offsets in zip(views, offsets, strict=True):
        assert torch.equal(view.condition_video_offsets, expected_offsets)


def test_cross_video_gradient_mean_is_permutation_invariant_and_unit_weight() -> None:
    gradients = tuple(
        torch.tensor([float(index), float(index + 2)], dtype=torch.float32)
        for index in range(4)
    )
    expected = torch.tensor([1.5, 3.5], dtype=torch.float32)
    torch.testing.assert_close(mean_cross_video_task_gradient(gradients), expected)
    torch.testing.assert_close(
        mean_cross_video_task_gradient(tuple(reversed(gradients))), expected
    )
    duplicate = torch.tensor([0.125, -3.0], dtype=torch.float32)
    assert torch.equal(mean_cross_video_task_gradient((duplicate,) * 4), duplicate)


def test_cross_video_gradient_geometry_reports_shared_descent() -> None:
    geometry = cross_video_gradient_geometry(
        (
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]),
        )
    )
    assert geometry["pairwise_cosine_mean"] == pytest.approx(1.0 / 3.0)
    assert geometry["pairwise_cosine_minimum"] == 0.0
    assert geometry["pairwise_cosine_maximum"] == 1.0
    assert geometry["shared_mean_descent_coverage"] == 1.0
    assert geometry["view_to_shared_mean_cosine_minimum"] == pytest.approx(2.0**-0.5)
    assert geometry["shared_mean_energy_over_view_energy"] == pytest.approx(0.5)


def test_preconditioned_commitment_preserves_actual_adam_candidate() -> None:
    gradient = torch.tensor([1.0, 0.1], dtype=torch.float32)
    adam_delta = torch.tensor([-0.2, -0.2], dtype=torch.float32)
    final, geometry = preconditioned_candidate_commitment(gradient, adam_delta)
    torch.testing.assert_close(final, adam_delta)
    assert geometry["full_candidate_to_adam_candidate_cosine"] == 1.0
    assert geometry["radius_relative_error"] <= 1e-6
    assert geometry["adam_candidate_to_negative_optimizer_gradient_cosine"] < 0.8


def test_lora_response_reports_native_q_v_and_action_writeout() -> None:
    before, after = {}, {}
    modules = (
        "model.gemma_expert.model.layers.0.self_attn.q_proj",
        "model.gemma_expert.model.layers.0.self_attn.v_proj",
        "model.action_in_proj",
    )
    for index, module in enumerate(modules, start=1):
        a_name = f"{module}.lora_A.default.weight"
        b_name = f"{module}.lora_B.default.weight"
        before[a_name] = torch.zeros(2, 3)
        before[b_name] = torch.zeros(4, 2)
        after[a_name] = torch.full((2, 3), float(index))
        after[b_name] = torch.full((4, 2), float(index))
    response = lora_response(before, after)
    assert response["effective_ba_response_rms"] > 0
    assert all(
        response["effective_ba_response_rms_by_kind"][kind] > 0
        for kind in ("q", "v", "action")
    )


def test_expert_rank_padding_preserves_effective_ba() -> None:
    target = LoRATarget("model.projection", 3, 4)
    expert = SmolVLALoRAContract(
        targets=(target,), rank=2, alpha=2, dropout=0.0, identity_seed=7
    )
    public = SmolVLALoRAContract(
        targets=(target,), rank=4, alpha=4, dropout=0.0, identity_seed=7
    )
    state = {
        "model.projection.lora_A.default.weight": torch.randn(2, 3),
        "model.projection.lora_B.default.weight": torch.randn(4, 2),
    }
    padded = pad_expert_lora_to_public_rank(
        state, expert_contract=expert, public_contract=public
    )
    torch.testing.assert_close(
        padded["model.projection.lora_B.default.weight"]
        @ padded["model.projection.lora_A.default.weight"],
        state["model.projection.lora_B.default.weight"]
        @ state["model.projection.lora_A.default.weight"],
    )
    assert padded["model.projection.lora_A.default.weight"].shape == (4, 3)
    assert padded["model.projection.lora_B.default.weight"].shape == (4, 4)


class _PrefixOwner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = SimpleNamespace(_attn_implementation=None)
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(config=config))
        )

    def forward(self, **_kwargs):
        return None, ()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(7, 7, bias=False)
        self.projection.weight.data.zero_()
        self.paligemma_with_expert = _PrefixOwner()

    @staticmethod
    def _rtc_enabled() -> bool:
        return False

    @staticmethod
    def _prepare_attention_masks_4d(value):
        return value

    @staticmethod
    def embed_prefix(_images, _image_masks, tokens, _masks):
        size = tokens.shape[0]
        return (
            torch.zeros((size, 1, 7)),
            torch.ones((size, 1), dtype=torch.bool),
            torch.zeros((size, 1), dtype=torch.long),
        )

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        del prefix_pad_masks, past_key_values, timestep
        return self.projection(x_t)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Model()
        self.config = SimpleNamespace(
            output_features={ACTION: SimpleNamespace(shape=(7,))},
            max_action_dim=7,
            chunk_size=50,
        )

    def _preprocess_images(self, batch):
        size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        return [torch.zeros((size, 3, 2, 2))], [torch.ones(size, dtype=torch.bool)]


def test_successful_expert_endpoint_uses_one_prediction_and_descends() -> None:
    policy = _Policy()
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("model.projection", 7, 7),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=29,
    )
    inject_task_lora(policy, contract)
    policy.requires_grad_(False)
    generator = torch.Generator().manual_seed(17)
    state = {
        name: (
            torch.zeros(shape)
            if name.endswith(".lora_B.default.weight")
            else torch.randn(shape, generator=generator).mul_(0.1)
        )
        for name, shape in expected_lora_state_shapes(contract).items()
    }
    actions = torch.stack(
        tuple(
            torch.full((50, 7), value)
            for value in (0.01, 0.03, 0.05, 0.07)
        )
    )
    batch = {
        ACTION: actions,
        "executed_action_steps": torch.tensor([5, 3, 4, 2]),
        "action_is_pad": torch.zeros((4, 50), dtype=torch.bool),
        "policy_noise_seed": torch.tensor([101, 103, 107, 109]),
        OBS_LANGUAGE_TOKENS: torch.ones((4, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((4, 2), dtype=torch.bool),
    }
    trajectory_ids = torch.tensor([0, 1, 1, 1])
    kwargs = dict(
        policy=policy,
        state=state,
        contract=contract,
        batch=batch,
        trajectory_ids=trajectory_ids,
        endpoint_action_batch_size=8,
        num_inference_steps=10,
        device=torch.device("cpu"),
    )
    gradient, summary = functional_successful_expert_occupancy_gradient(**kwargs)
    assert summary.functional_policy_forwards == 1
    assert summary.functional_policy_backwards == 1
    assert summary.successful_trajectories == 2
    assert summary.selected_credit_states == 4
    assert summary.matched_expert_student_action_rms > 0
    assert any(bool(torch.count_nonzero(value)) for value in gradient.values())
    assert all(
        not bool(torch.count_nonzero(value))
        for name, value in gradient.items()
        if name.endswith(".lora_A.default.weight")
    )
    assert all(parameter.grad is None for parameter in policy.parameters())
    before = functional_successful_expert_occupancy_objective(**kwargs)
    gradient_norm = torch.cat([value.flatten() for value in gradient.values()]).norm()
    updated = {
        name: state[name] - (1e-2 / gradient_norm) * gradient[name] for name in state
    }
    after = functional_successful_expert_occupancy_objective(
        **{**kwargs, "state": updated}
    )
    assert after["expert_distillation_objective"] < before["expert_distillation_objective"]


def test_unit_residual_endpoint_geometry_and_mask() -> None:
    targets = torch.tensor(
        [[[0.0, 0.0], [2.0, 0.0], [99.0, 99.0]], [[1.0, -1.0], [77.0, 77.0], [77.0, 77.0]]],
        requires_grad=True,
    )
    valid = torch.tensor([2, 1])
    predicted = torch.zeros_like(targets).detach().requires_grad_()
    rms, normalized = unit_residual_endpoint_distillation(predicted, targets, valid)
    torch.testing.assert_close(rms, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(normalized, torch.tensor([1.0, 1.0]), rtol=1e-5, atol=1e-6)
    assert targets.grad is None


def test_unit_residual_gradient_is_scale_invariant() -> None:
    base_targets = torch.tensor([[[2.0, 0.0], [-1.0, 1.0]]])
    gradients = []
    for scale in (0.25, 8.0):
        targets = (base_targets * scale).requires_grad_()
        predicted = torch.zeros_like(targets).requires_grad_()
        _, normalized = unit_residual_endpoint_distillation(
            predicted, targets, torch.tensor([2])
        )
        normalized.sum().backward()
        gradients.append(predicted.grad.detach().clone())
        assert targets.grad is None
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-6, atol=1e-7)


def test_stratified_occupancy_weights_equalize_unequal_trajectory_lengths() -> None:
    ids = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    weights = stratified_occupancy_weights(ids)
    torch.testing.assert_close(weights, torch.tensor([0.5, 1 / 6, 1 / 6, 1 / 6]))
    torch.testing.assert_close(weights[ids == 0].sum(), weights[ids == 1].sum())


def _trajectory(*, cursor: int, success: bool, marker: float) -> RewardTrajectory:
    observation = {
        OBS_LANGUAGE_TOKENS: torch.ones((1, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((1, 2), dtype=torch.bool),
    }
    return RewardTrajectory(
        suite="libero_goal",
        task_id=4,
        global_task_id=24,
        adaptation_seed=3,
        rollout_cursor=cursor,
        env_seed=7 + cursor,
        policy_seed_root=11,
        success=success,
        steps=5,
        reward_sum=float(success),
        dummy_settling_steps=10,
        policy_noise_seeds=(13 + cursor,),
        observations=(observation,),
        action_chunks=(torch.full((1, 50, 7), marker),),
        valid_action_steps=(5,),
    )


def test_successful_occupancy_queries_expert_and_student_with_identical_batches(
    monkeypatch,
) -> None:
    class CounterfactualPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)
            self.arm_value = -1.0
            self.batch_sizes: list[int] = []

        def reset(self) -> None:
            pass

        def predict_action_chunk(self, batch, *, noise, num_steps):
            assert num_steps == 10
            size = batch[OBS_LANGUAGE_TOKENS].shape[0]
            self.batch_sizes.append(size)
            value = self.arm_value + noise[:, 0, 0].float().mul(1e-3)
            return value[:, None, None].expand(size, 50, 7).contiguous()

    def with_chunks(value: RewardTrajectory, count: int) -> RewardTrajectory:
        return replace(
            value,
            policy_noise_seeds=tuple(100 + index for index in range(count)),
            observations=value.observations * count,
            action_chunks=value.action_chunks * count,
            valid_action_steps=tuple(5 for _ in range(count)),
        )

    first = with_chunks(_trajectory(cursor=0, success=True, marker=2.0), 2)
    second = with_chunks(_trajectory(cursor=1, success=True, marker=1.0), 3)
    policy = CounterfactualPolicy()

    def install(target, state, _contract):
        target.arm_value = float(state["arm"])

    monkeypatch.setattr("ember.reward.rollout.copy_task_lora_state_", install)
    runtime = SimpleNamespace(
        policy=policy,
        lora_contract=object(),
        identity_state={"arm": torch.tensor(-1.0)},
        context=SimpleNamespace(device=torch.device("cpu")),
        config={
            "optimization": {"counterfactual_action_batch_size": 2},
            "environment": {"num_inference_steps": 10},
        },
    )
    actions, metrics = query_successful_expert_occupancy_actions(
        policy=runtime.policy,
        lora_contract=runtime.lora_contract,
        identity_state=runtime.identity_state,
        trajectories=(first, second),
        expert_lora={"arm": torch.tensor(10.0)},
        student_lora={"arm": torch.tensor(20.0)},
        device=runtime.context.device,
        microbatch_size=2,
        num_inference_steps=10,
    )
    assert [len(value) for value in actions["expert"]] == [2, 3]
    assert [len(value) for value in actions["student"]] == [2, 3]
    assert all(
        float(value.mean()) < 11 for trajectory in actions["expert"] for value in trajectory
    )
    assert all(
        float(value.mean()) > 19 for trajectory in actions["student"] for value in trajectory
    )
    assert policy.batch_sizes == [2, 2, 1, 2, 2, 1]
    assert policy.arm_value == -1.0
    assert metrics["complete_occupancy_chunks"] == 5
    assert metrics["matched_policy_forwards"] == 6
    assert metrics["matched_query_batch_sizes"] == [2, 2, 1]
    assert metrics["stored_expert_to_matched_requery_rms"] > 0


def _single_condition_inputs(k: int) -> tuple[torch.Tensor, ...]:
    frames = torch.arange(k * 2 * 3 * 4 * 4, dtype=torch.uint8).reshape(k * 2, 3, 4, 4)
    indices = torch.tensor([0, 5] * k, dtype=torch.long)
    video_offsets = torch.arange(0, 2 * k + 1, 2, dtype=torch.long)
    condition_offsets = torch.tensor([0, k], dtype=torch.long)
    tokens = torch.tensor([[1, 10, 11, 12, 0]], dtype=torch.long)
    masks = tokens.ne(0)
    spans = torch.tensor([[False, False, True, True, False]])
    return frames, indices, video_offsets, condition_offsets, tokens, masks, spans


@pytest.mark.parametrize("k", (1, 2, 3, 4))
def test_cached_reference_is_exact_as139_for_dynamic_k(k: int) -> None:
    writer, _ = _writer_model()
    inputs = _single_condition_inputs(k)
    with torch.no_grad():
        writer.query_delta.weight.normal_(std=1.0)
        state = writer.encode_conditioning_state(*inputs, policy=torch.nn.Identity())
        reference = writer.compile_conditioning_state(
            state, inputs[3], use_query_delta=False
        ).program
        saved = writer.query_delta.weight.detach().clone()
        writer.query_delta.weight.zero_()
        exact = writer.encode_program(*inputs, policy=torch.nn.Identity()).program
        writer.query_delta.weight.copy_(saved)
    torch.testing.assert_close(reference, exact, rtol=0, atol=0)


def test_cached_candidate_recompiles_query_only_without_another_backbone() -> None:
    writer, _ = _writer_model()
    inputs = _inputs()
    calls = 0
    original = writer.backbone_memory_encoder.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    writer.backbone_memory_encoder.forward = counted
    with torch.no_grad():
        writer.query_delta.weight.normal_(std=1.0)
        state = writer.encode_conditioning_state(*inputs, policy=torch.nn.Identity())
        reference = writer.compile_conditioning_state(
            state, inputs[3], use_query_delta=False
        ).program
        candidate = writer.compile_conditioning_state(
            state, inputs[3], use_query_delta=True
        ).program
    assert not torch.equal(candidate, reference)
    assert calls == 1

    writer.requires_grad_(False)
    writer.query_delta.weight.requires_grad_(True)
    recompiled = writer.compile_conditioning_state(
        state, inputs[3], use_query_delta=True
    )
    sum(
        value.float().sum()
        for value in writer.decode_program(recompiled.program).values()
    ).backward()
    assert writer.query_delta.weight.grad is not None
    assert writer.query_delta.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in writer.named_parameters()
        if name != "query_delta.weight"
    )


def _global_test_credit_rows(
    scale: float, *, all_fail: bool = False
) -> tuple[dict[str, object], ...]:
    rows = []
    for task_id in range(2):
        if all_fail:
            before, after = 0.0, 0.0 if scale == 0 else 0.25
        else:
            before = 1.0
            after = 0.75 if task_id == 0 else 1.25
        for view_index in range(4):
            rows.append(
                {
                    "task_id": task_id,
                    "suite": f"suite_{task_id}",
                    "view_index": view_index,
                    "before_credit_objective": before,
                    "after_credit_objective": after,
                    "credit_objective_delta": after - before,
                    "after_expert_action_distance": 2.0 + after,
                    "credit_descent": after < before,
                }
            )
    return tuple(rows)


def test_median_task_cap_never_amplifies_small_tangents() -> None:
    mean, detail = median_capped_task_mean(
        torch.tensor([[3.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 0.5]])
    )
    torch.testing.assert_close(mean, torch.tensor([0.75, 0.375]))
    assert detail["median_task_gradient_norm"] == pytest.approx(1.5)
    assert detail["task_gradient_scales"] == pytest.approx([0.5, 1.0, 0.75, 1.0])
    assert detail["capped_task_count"] == 2
    assert detail["small_task_amplification"] is False


def test_optimizer_uses_median_capped_active_task_mean() -> None:
    class _Writer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parameter_grid = torch.nn.Module()
            self.parameter_grid.branch = torch.nn.ModuleDict(
                {"payload": torch.nn.Linear(1, 2, bias=False)}
            )

    writer = _Writer()
    head = writer.parameter_grid.branch["payload"].weight
    head.data.copy_(torch.tensor([[0.5], [-0.25]]))
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        config={
            "optimization": {"optimizer": {"gradient_clip_norm": 10.0}},
            "commitment": {"max_backtracks": 0},
        },
        writer=writer,
        optimizer=optimizer,
        trainable_parameters=tuple(writer.parameters()),
        gradient_layout=parameter_layout(writer),
        tasks=(SimpleNamespace(global_task_id=0), SimpleNamespace(global_task_id=1)),
    )
    evaluated_scales = []

    def evaluate(scale: float) -> tuple[dict[str, object], ...]:
        evaluated_scales.append(scale)
        return _global_test_credit_rows(scale)

    task_gradients = {
        0: torch.tensor([-1.0, 0.0]),
        1: torch.tensor([-0.25, 0.0]),
    }
    step = apply_reward_step(
        runtime,
        2,
        task_gradients,
        evaluate,
    )
    assert step.active_tasks == 2
    torch.testing.assert_close(
        optimizer.state[head]["exp_avg"].reshape(-1),
        torch.tensor([-0.04375, 0.0]),
        rtol=0,
        atol=1e-7,
    )
    assert step.parameter_delta_rms[
        "parameter_grid.branch.payload.weight"
    ] > 0
    assert step.gradient_coexistence["shared_mean_descent_coverage"] == 1.0
    assert step.gradient_coexistence["final_delta_descent_coverage"] == 1.0
    assert step.gradient_coexistence["task_tangent_balance"][
        "capped_task_count"
    ] == 1
    assert step.gradient_coexistence["task_tangent_balance"][
        "median_task_gradient_norm"
    ] == pytest.approx(0.625)
    assert step.gradient_coexistence["task_tangent_balance"][
        "task_gradient_scales"
    ] == pytest.approx([0.625, 1.0])
    assert step.commitment_geometry["final_to_adam_candidate_cosine"] == pytest.approx(
        1.0
    )
    assert step.commitment_geometry["radius_relative_error"] <= 1e-6
    assert evaluated_scales == [1.0]
    assert step.commitment_geometry["search_accepted"] is True
    assert step.commitment_geometry["accepted_backtrack_index"] == 0
    assert step.commitment_geometry["accepted_radius_scale"] == 1.0
    assert step.commitment_geometry["search_trial_count"] == 1
    assert step.commitment_geometry["descending_task_view_count"] == 4
    assert (
        step.commitment_geometry[
            "all_active_task_view_credit_descent_diagnostic"
        ]
        is False
    )
    assert step.commitment_geometry["repeated_step0_baseline_forward"] is False
    assert step.commitment_geometry["global_active_task_ids"] == [0, 1]
    assert step.commitment_geometry["global_task_view_count"] == 8
    assert all(
        row["before_credit_objective"] == 1.0
        for row in step.commitment_credit_rows
    )
    assert sum(
        bool(row["credit_descent"])
        for row in step.commitment_credit_rows
    ) == 4

    accepted_parameters = head.detach().clone()
    failed = apply_reward_step(
        runtime,
        2,
        task_gradients,
        lambda scale: _global_test_credit_rows(scale, all_fail=True),
    )
    assert not torch.equal(head, accepted_parameters)
    assert failed.commitment_geometry["search_accepted"] is True
    assert failed.commitment_geometry["search_trial_count"] == 1
    assert failed.commitment_geometry["accepted_radius_scale"] == 1.0
    assert failed.commitment_geometry["final_delta_l2"] > 0
    assert (
        failed.commitment_geometry[
            "all_active_task_view_credit_descent_diagnostic"
        ]
        is False
    )
    assert all(
        row["credit_objective_delta"] == 0.25
        for row in failed.commitment_credit_rows
    )
