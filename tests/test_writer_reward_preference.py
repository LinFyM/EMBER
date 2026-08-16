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
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    expected_lora_state_shapes,
    inject_task_lora,
)
from ember.reward.rollout import (
    RewardTrajectory,
    query_matched_occupancy_actions,
)
from ember.writer.as_step import parameter_layout
from ember.writer.reward_cycle import select_discordant_trajectory_pairs
from ember.writer.reward_gradient_update import (
    apply_reward_step,
    lora_response,
    preconditioned_candidate_commitment,
)
from ember.writer.reward_preference import (
    cross_video_gradient_geometry,
    functional_matched_stratified_occupancy_lora_gradient,
    functional_matched_stratified_occupancy_margin,
    mean_cross_video_task_gradient,
    stratified_occupancy_pair_weights,
)


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
    assert torch.equal(
        mean_cross_video_task_gradient((duplicate,) * 4), duplicate
    )


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
    assert geometry["view_to_shared_mean_cosine_minimum"] == pytest.approx(
        2.0**-0.5
    )
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


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(7, 7, bias=False)
        self.config = SimpleNamespace(
            time_sampling_beta_alpha=1.5,
            time_sampling_beta_beta=1.0,
            time_sampling_scale=0.999,
            time_sampling_offset=0.001,
        )

    def forward(self, images, image_masks, tokens, masks, actions, noise, time):
        del images, image_masks, tokens, masks
        signal = actions + noise.mul(0.125) + time[:, None, None].mul(0.25)
        return self.projection(signal).square()


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
        size = batch[ACTION].shape[0]
        return [torch.zeros((size, 3, 2, 2))], [torch.ones(size, dtype=torch.bool)]

    @staticmethod
    def prepare_action(batch):
        return batch[ACTION]


def test_matched_stratified_occupancy_preference_is_microbatch_semantic() -> None:
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
    state = {
        name: torch.randn(shape, generator=torch.Generator().manual_seed(17))
        for name, shape in expected_lora_state_shapes(contract).items()
    }
    actions = torch.stack(
        tuple(
            torch.full((50, 7), value)
            for value in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
        )
    )
    batch = {
        ACTION: actions,
        "executed_action_steps": torch.tensor([5, 5, 3, 3, 4, 4, 2, 2]),
        "action_is_pad": torch.zeros((8, 50), dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones((8, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((8, 2), dtype=torch.bool),
    }
    trajectory_ids = torch.tensor([0, 1, 1, 1])
    kwargs = dict(
        policy=policy,
        state=state,
        contract=contract,
        batch=batch,
        trajectory_ids=trajectory_ids,
        mc_samples=4,
        flow_seed_root=31,
        cycle=1,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    first, first_summary = functional_matched_stratified_occupancy_lora_gradient(
        **kwargs, physical_microbatch_size=2
    )
    second, second_summary = functional_matched_stratified_occupancy_lora_gradient(
        **kwargs, physical_microbatch_size=8
    )
    assert first_summary.functional_policy_forwards == 16
    assert second_summary.functional_policy_forwards == 4
    assert first_summary.discordant_trajectories == 2
    assert first_summary.selected_credit_pairs == 4
    assert first_summary.matched_winner_loser_action_rms > 0
    for name in state:
        torch.testing.assert_close(first[name], second[name], rtol=2e-6, atol=2e-6)
    assert any(bool(torch.count_nonzero(value)) for value in first.values())
    assert all(parameter.grad is None for parameter in policy.parameters())
    before = functional_matched_stratified_occupancy_margin(
        policy,
        state,
        contract,
        batch,
        trajectory_ids,
        mc_samples=4,
        physical_microbatch_size=8,
        flow_seed_root=31,
        cycle=1,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    updated = {name: state[name] - 1e-3 * first[name] for name in state}
    after = functional_matched_stratified_occupancy_margin(
        policy,
        updated,
        contract,
        batch,
        trajectory_ids,
        mc_samples=4,
        physical_microbatch_size=8,
        flow_seed_root=31,
        cycle=1,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert after["preference_margin"] < before["preference_margin"]


def test_stratified_occupancy_weights_equalize_unequal_trajectory_lengths() -> None:
    ids = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    weights = stratified_occupancy_pair_weights(ids)
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


def test_pair_selection_orders_the_unique_winner_before_the_loser() -> None:
    reference = (
        _trajectory(cursor=0, success=False, marker=0.0),
        _trajectory(cursor=1, success=True, marker=1.0),
    )
    candidate = (
        _trajectory(cursor=0, success=True, marker=2.0),
        _trajectory(cursor=1, success=False, marker=3.0),
    )
    pairs, labels = select_discordant_trajectory_pairs(reference, candidate)
    assert labels == ("candidate", "reference")
    assert pairs[0] == (candidate[0], reference[0])
    assert pairs[1] == (reference[1], candidate[1])


def test_matched_occupancy_queries_both_arms_with_identical_batches(
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

    candidate_winner = with_chunks(
        _trajectory(cursor=0, success=True, marker=2.0), 2
    )
    reference_loser = _trajectory(cursor=0, success=False, marker=0.0)
    reference_winner = with_chunks(
        _trajectory(cursor=1, success=True, marker=1.0), 3
    )
    candidate_loser = _trajectory(cursor=1, success=False, marker=3.0)
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
    actions, metrics = query_matched_occupancy_actions(
        policy=runtime.policy,
        lora_contract=runtime.lora_contract,
        identity_state=runtime.identity_state,
        pairs=(
            (candidate_winner, reference_loser),
            (reference_winner, candidate_loser),
        ),
        active_labels=("candidate", "reference"),
        reference_lora={"arm": torch.tensor(10.0)},
        candidate_lora={"arm": torch.tensor(20.0)},
        device=runtime.context.device,
        microbatch_size=2,
        num_inference_steps=10,
    )
    assert [len(value) for value in actions["reference"]] == [2, 3]
    assert [len(value) for value in actions["candidate"]] == [2, 3]
    assert all(float(value.mean()) < 11 for pair in actions["reference"] for value in pair)
    assert all(float(value.mean()) > 19 for pair in actions["candidate"] for value in pair)
    assert policy.batch_sizes == [2, 2, 1, 2, 2, 1]
    assert policy.arm_value == -1.0
    assert metrics["complete_occupancy_chunks"] == 5
    assert metrics["matched_policy_forwards"] == 6
    assert metrics["matched_query_batch_sizes"] == [2, 2, 1]
    assert metrics["stored_winner_to_matched_requery_rms"] > 0
    assert metrics["stored_loser_to_matched_first_requery_rms"] > 0


def _single_condition_inputs(k: int) -> tuple[torch.Tensor, ...]:
    frames = torch.arange(k * 2 * 3 * 4 * 4, dtype=torch.uint8).reshape(
        k * 2, 3, 4, 4
    )
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
        state = writer.encode_conditioning_state(
            *inputs, policy=torch.nn.Identity()
        )
        reference = writer.compile_conditioning_state(
            state, inputs[3], use_query_delta=False
        ).program
        saved = writer.query_delta.weight.detach().clone()
        writer.query_delta.weight.zero_()
        exact = writer.encode_program(
            *inputs, policy=torch.nn.Identity()
        ).program
        writer.query_delta.weight.copy_(saved)
    torch.testing.assert_close(reference, exact, rtol=0, atol=0)


def test_cached_candidate_recompiles_query_only_without_another_backbone() -> None:
    writer, _ = _writer_model()
    inputs = _inputs()
    calls = 0
    original = writer.base_writer.encode_video_evidence

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    writer.base_writer.encode_video_evidence = counted
    with torch.no_grad():
        writer.query_delta.weight.normal_(std=1.0)
        state = writer.encode_conditioning_state(
            *inputs, policy=torch.nn.Identity()
        )
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
    sum(value.float().sum() for value in writer.decode_program(recompiled.program).values()).backward()
    assert writer.query_delta.weight.grad is not None
    assert writer.query_delta.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in writer.named_parameters()
        if name != "query_delta.weight"
    )


def test_optimizer_uses_equal_mean_over_active_tasks() -> None:
    class _Writer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.factor_commitment = torch.nn.Module()
            self.factor_commitment.heads = torch.nn.ModuleDict(
                {"q_b": torch.nn.Linear(1, 2, bias=False)}
            )

    writer = _Writer()
    head = writer.factor_commitment.heads["q_b"].weight
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
            "commitment": {"max_backtracks": 3},
        },
        writer=writer,
        optimizer=optimizer,
        trainable_parameters=tuple(writer.parameters()),
        gradient_layout=parameter_layout(writer),
        tasks=(SimpleNamespace(global_task_id=0), SimpleNamespace(global_task_id=1)),
    )
    evaluated_scales = []

    def evaluate(scale: float) -> tuple[dict[str, float | int | bool], ...]:
        evaluated_scales.append(scale)
        after = 1.5 if scale == 0 else 1.25 if scale <= 0.5 else 1.75
        return tuple(
            {
                "view_index": index,
                "before_preference_margin": 1.0,
                "after_preference_margin": after,
                "preference_margin_delta": after - 1.0,
                "after_preference_objective": 2.0 + after,
                "preference_descent": after < 1.0,
            }
            for index in range(4)
        )

    step = apply_reward_step(
        runtime,
        torch.tensor([-1.0, 0.0]),
        2,
        {0: torch.tensor([-0.5, 0.0]), 1: torch.tensor([-0.5, 0.0])},
        evaluate,
    )
    assert step.active_tasks == 2
    torch.testing.assert_close(
        optimizer.state[head]["exp_avg"].reshape(-1),
        torch.tensor([-0.05, 0.0]),
        rtol=0,
        atol=1e-7,
    )
    assert step.parameter_delta_rms["factor_commitment.heads.q_b.weight"] > 0
    assert step.gradient_coexistence["shared_mean_descent_coverage"] == 1.0
    assert step.gradient_coexistence["final_delta_descent_coverage"] == 1.0
    assert step.commitment_geometry[
        "final_to_adam_candidate_cosine"
    ] == pytest.approx(1.0)
    assert step.commitment_geometry["radius_relative_error"] <= 1e-6
    assert evaluated_scales == [0.0, 1.0, 0.5]
    assert step.commitment_geometry["search_accepted"] is True
    assert step.commitment_geometry["accepted_backtrack_index"] == 1
    assert step.commitment_geometry["accepted_radius_scale"] == 0.5
    assert step.commitment_geometry[
        "gradient_path_to_inference_baseline_margin_deltas"
    ] == [0.5] * 4
    assert all(
        row["before_preference_margin"] == 1.5
        and row["after_preference_margin"] == 1.25
        and row["preference_margin_delta"] == -0.25
        for row in step.commitment_preference_rows
    )
    assert all(row["preference_descent"] for row in step.commitment_preference_rows)

    accepted_parameters = head.detach().clone()
    failed = apply_reward_step(
        runtime,
        torch.tensor([-1.0, 0.0]),
        2,
        {0: torch.tensor([-0.5, 0.0]), 1: torch.tensor([-0.5, 0.0])},
        lambda scale: tuple(
            {
                "view_index": index,
                "before_preference_margin": 0.0,
                "after_preference_margin": 0.0 if scale == 0 else 0.25,
                "preference_margin_delta": 0.0 if scale == 0 else 0.25,
                "after_preference_objective": 1.0 if scale == 0 else 1.25,
                "preference_descent": False,
            }
            for index in range(4)
        ),
    )
    torch.testing.assert_close(head, accepted_parameters)
    assert failed.commitment_geometry["search_accepted"] is False
    assert failed.commitment_geometry["search_trial_count"] == 4
    assert failed.commitment_geometry["accepted_radius_scale"] == 0.0
    assert all(
        row["preference_margin_delta"] == 0.0
        for row in failed.commitment_preference_rows
    )
