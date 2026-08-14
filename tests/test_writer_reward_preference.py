from __future__ import annotations

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
from ember.reward.rollout import RewardTrajectory
from ember.writer.as_step import parameter_layout
from ember.writer.reward_cycle import _apply_step, select_unique_success_trajectories
from ember.writer.reward_preference import (
    functional_selected_success_lora_gradient,
    selected_trajectory_chunk_weights,
)


def test_selected_success_weights_make_each_target_trajectory_equal() -> None:
    trajectory_ids = torch.tensor([0, 1, 1, 1])
    weights = selected_trajectory_chunk_weights(trajectory_ids)
    per_trajectory = torch.stack(
        [weights[trajectory_ids == target].sum() for target in range(2)]
    )
    torch.testing.assert_close(
        per_trajectory, torch.tensor([0.5, 0.5]), rtol=0, atol=0
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


def test_selected_success_lora_credit_is_microbatch_semantic() -> None:
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
    trajectory_ids = torch.tensor([0, 1, 1, 1])
    batch = {
        ACTION: torch.ones((4, 50, 7)),
        "executed_action_steps": torch.tensor([5, 5, 3, 4]),
        "action_is_pad": torch.zeros((4, 50), dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones((4, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((4, 2), dtype=torch.bool),
    }
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
    first, first_summary = functional_selected_success_lora_gradient(
        **kwargs, physical_microbatch_size=2
    )
    second, second_summary = functional_selected_success_lora_gradient(
        **kwargs, physical_microbatch_size=8
    )
    assert first_summary.functional_policy_forwards == 8
    assert second_summary.functional_policy_forwards == 4
    assert first_summary.target_trajectories == 2
    for name in state:
        torch.testing.assert_close(first[name], second[name], rtol=2e-6, atol=2e-6)
    assert any(bool(torch.count_nonzero(value)) for value in first.values())
    assert all(parameter.grad is None for parameter in policy.parameters())


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


def test_pair_selection_uses_only_the_uniquely_successful_arm() -> None:
    reference = (
        _trajectory(cursor=0, success=False, marker=0.0),
        _trajectory(cursor=1, success=True, marker=1.0),
    )
    candidate = (
        _trajectory(cursor=0, success=True, marker=2.0),
        _trajectory(cursor=1, success=False, marker=3.0),
    )
    selected, labels = select_unique_success_trajectories(reference, candidate)
    assert labels == ("candidate", "reference")
    assert selected[0] is candidate[0]
    assert selected[1] is reference[1]


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
            self.query_delta = torch.nn.Linear(2, 1, bias=False)

    writer = _Writer()
    writer.query_delta.weight.data.copy_(torch.tensor([[0.5, -0.25]]))
    optimizer = torch.optim.AdamW(
        writer.parameters(),
        lr=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    runtime = SimpleNamespace(
        context=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        config={"optimization": {"optimizer": {"gradient_clip_norm": 10.0}}},
        writer=writer,
        optimizer=optimizer,
        trainable_parameters=tuple(writer.parameters()),
        gradient_layout=parameter_layout(writer),
    )
    step = _apply_step(runtime, torch.tensor([-1.0, 0.0]), 2)
    assert step.active_tasks == 2
    torch.testing.assert_close(
        optimizer.state[writer.query_delta.weight]["exp_avg"],
        torch.tensor([[-0.05, 0.0]]),
        rtol=0,
        atol=1e-7,
    )
    assert step.parameter_delta_rms["query_delta.weight"] > 0
