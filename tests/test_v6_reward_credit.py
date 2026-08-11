from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_step import (
    GeneratedConditionGraph,
    program_cotangent,
)
from ember.expert_manifold.v6_reward_credit import (
    functional_success_lora_gradients,
    project_blind_program_direction,
)
from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    expected_lora_state_shapes,
    inject_task_lora,
)
from ember.reward.protocol import RewardProtocolError


def _program(*entries: tuple[int, int, float]) -> torch.Tensor:
    value = torch.zeros((320, 256), dtype=torch.float32)
    for row, column, scalar in entries:
        value[row, column] = scalar
    return value


def test_no_success_and_raw_feasible_proposals_are_bit_exact() -> None:
    blind = torch.randn((320, 256), generator=torch.Generator().manual_seed(7))
    no_guard, no_guard_summary = project_blind_program_direction(blind, ())
    assert torch.equal(no_guard, blind)
    assert no_guard_summary.raw_feasible
    assert not no_guard_summary.changed
    feasible, feasible_summary = project_blind_program_direction(
        blind, (-blind.clone(),)
    )
    assert torch.equal(feasible, blind)
    assert feasible_summary.raw_feasible
    assert not feasible_summary.changed


def test_single_and_multiple_conflicts_have_expected_analytic_projection() -> None:
    blind = _program((0, 0, 2.0), (0, 1, 3.0), (0, 2, -4.0))
    first = _program((0, 0, 1.0))
    second = _program((0, 1, 1.0))
    single, single_summary = project_blind_program_direction(blind, (first,))
    torch.testing.assert_close(
        single, _program((0, 1, 3.0), (0, 2, -4.0)), rtol=0, atol=0
    )
    assert single_summary.active_constraint_ordinals == (0,)
    multiple, multiple_summary = project_blind_program_direction(
        blind, (first, second)
    )
    torch.testing.assert_close(multiple, _program((0, 2, -4.0)), rtol=0, atol=0)
    assert multiple_summary.active_constraint_count == 2
    assert multiple_summary.maximum_constraint_value <= 0


def test_duplicate_rank_deficient_and_permuted_constraints_preserve_solution() -> None:
    blind = _program((0, 0, 2.0), (0, 1, 3.0), (0, 2, 5.0))
    first = _program((0, 0, 1.0))
    duplicate = _program((0, 0, 7.0))
    second = _program((0, 1, 1.0))
    expected, _ = project_blind_program_direction(blind, (first, second))
    redundant, summary = project_blind_program_direction(
        blind, (duplicate, second, first)
    )
    permuted, _ = project_blind_program_direction(
        blind, (second, first, duplicate)
    )
    torch.testing.assert_close(redundant, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(permuted, expected, rtol=0, atol=1e-6)
    assert summary.constraint_count == 3
    assert summary.maximum_constraint_value <= 1e-6


def test_zero_projection_and_every_nonzero_projection_keep_source_descent() -> None:
    first = _program((0, 0, 1.0))
    zero, zero_summary = project_blind_program_direction(first, (first,))
    assert torch.count_nonzero(zero) == 0
    assert zero_summary.safe_direction_rms == 0
    blind = _program((0, 0, 2.0), (0, 1, 3.0))
    safe, summary = project_blind_program_direction(blind, (first,))
    inner = torch.dot(blind.flatten().double(), safe.flatten().double())
    safe_energy = safe.flatten().double().square().sum()
    assert inner >= safe_energy > 0
    assert summary.source_descent_ratio > 0
    assert summary.blind_safe_cosine > 0


def test_projection_is_homogeneous_for_small_program_tangents() -> None:
    blind = _program((0, 0, 2.0), (0, 1, 3.0), (0, 2, -4.0))
    guards = (_program((0, 0, 1.0)), _program((0, 1, 1.0)))
    reference, _ = project_blind_program_direction(blind, guards)
    scale = 1e-14
    small, summary = project_blind_program_direction(blind * scale, guards)
    permuted, _ = project_blind_program_direction(
        blind * scale, tuple(reversed(guards))
    )
    assert summary.changed
    assert not summary.raw_feasible
    torch.testing.assert_close(small, reference * scale, rtol=2e-6, atol=1e-21)
    torch.testing.assert_close(permuted, small, rtol=2e-6, atol=1e-21)
    direction_norm = float(torch.linalg.vector_norm((blind * scale).double()))
    assert summary.maximum_constraint_value <= (
        64 * torch.finfo(torch.float32).eps * direction_norm
    )


def test_projection_rejects_invalid_success_guards() -> None:
    blind = torch.zeros((320, 256), dtype=torch.float32)
    with pytest.raises(ExpertManifoldError, match="zero energy"):
        project_blind_program_direction(blind, (blind,))
    with pytest.raises(ExpertManifoldError, match="blind Program"):
        project_blind_program_direction(blind.double(), ())
    with pytest.raises(ExpertManifoldError, match="more than four"):
        project_blind_program_direction(
            blind, tuple(_program((0, index, 1.0)) for index in range(5))
        )


def test_program_graph_supports_multiple_explicit_vjps() -> None:
    program = torch.randn(1, 320, 256, requires_grad=True)
    decoded = {"projection.lora_A": program[0, :, :4].to(torch.bfloat16)}
    graph = GeneratedConditionGraph(
        correct_lora=decoded,
        program_leaf=program,
        program_input_before=program.detach(),
        correct_feature=torch.ones(256),
        negative_feature=torch.zeros(256),
        correct_raw_frames=10,
        correct_sampled_frames=3,
        negative_raw_frames=10,
        negative_sampled_frames=3,
    )
    first = program_cotangent(
        graph,
        {"projection.lora_A": torch.ones((320, 4), dtype=torch.float32)},
        retain_graph=True,
    )
    second = program_cotangent(
        graph,
        {"projection.lora_A": torch.full((320, 4), 2.0)},
    )
    assert first.dtype == second.dtype == torch.float32
    torch.testing.assert_close(second, first * 2)


class _ProjectedModel(torch.nn.Module):
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


class _ProjectedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ProjectedModel()
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


def _policy_contract_state():
    policy = _ProjectedPolicy()
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("model.projection", 7, 7),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=29,
    )
    inject_task_lora(policy, contract)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    shapes = expected_lora_state_shapes(contract)
    generator = torch.Generator().manual_seed(17)
    state = {
        name: torch.randn(shape, generator=generator, requires_grad=True)
        for name, shape in shapes.items()
    }
    return policy, contract, state


def _success_replay() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    episode_ids = torch.tensor([0, 0, 2, 2, 2])
    actions = torch.zeros((episode_ids.numel(), 50, 7))
    actions[:, :5] = torch.arange(1, episode_ids.numel() + 1)[:, None, None]
    batch = {
        ACTION: actions,
        "executed_action_steps": torch.tensor([5, 3, 4, 5, 2]),
        "action_is_pad": torch.zeros((episode_ids.numel(), 50), dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones((episode_ids.numel(), 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(
            (episode_ids.numel(), 2), dtype=torch.bool
        ),
    }
    return batch, episode_ids, torch.tensor([1.0, 0.0, 1.0, 0.0])


def _panel_rows() -> tuple[torch.Tensor, int]:
    return torch.tensor([0, 1, 4, 5, 6], dtype=torch.long), 7


def test_success_gradients_are_episode_equal_and_microbatch_semantic() -> None:
    policy, contract, state = _policy_contract_state()
    batch, episode_ids, successes = _success_replay()
    panel_rows, panel_chunks = _panel_rows()
    first, first_summary = functional_success_lora_gradients(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        panel_row_indices=panel_rows,
        panel_total_chunks=panel_chunks,
        mc_samples=4,
        physical_microbatch_size=1,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    second, second_summary = functional_success_lora_gradients(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        panel_row_indices=panel_rows,
        panel_total_chunks=panel_chunks,
        mc_samples=4,
        physical_microbatch_size=3,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert first_summary.success_episode_ids == second_summary.success_episode_ids == (
        0,
        2,
    )
    assert first_summary.functional_policy_forwards == 20
    assert second_summary.functional_policy_forwards == 8
    for first_episode, second_episode in zip(first, second, strict=True):
        for name in state:
            torch.testing.assert_close(
                first_episode[name], second_episode[name], rtol=2e-6, atol=2e-6
            )
    assert all(value > 0 for value in first_summary.lora_gradient_rms)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_success_gradient_keeps_full_k4_flow_row_identity() -> None:
    policy, contract, state = _policy_contract_state()
    batch, episode_ids, successes = _success_replay()
    panel_rows, panel_chunks = _panel_rows()
    together, _ = functional_success_lora_gradients(
        policy,
        state,
        contract,
        batch,
        episode_ids,
        successes,
        panel_row_indices=panel_rows,
        panel_total_chunks=panel_chunks,
        mc_samples=4,
        physical_microbatch_size=3,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    keep = episode_ids == 2
    only_batch = {name: value[keep] for name, value in batch.items()}
    alone, _ = functional_success_lora_gradients(
        policy,
        state,
        contract,
        only_batch,
        episode_ids[keep],
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        panel_row_indices=panel_rows[keep],
        panel_total_chunks=panel_chunks,
        mc_samples=4,
        physical_microbatch_size=2,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    for name in state:
        torch.testing.assert_close(
            together[1][name], alone[0][name], rtol=2e-6, atol=2e-6
        )


def test_all_failure_replay_has_no_gradient_and_failure_chunks_are_rejected() -> None:
    policy, contract, state = _policy_contract_state()
    gradients, summary = functional_success_lora_gradients(
        policy,
        state,
        contract,
        {},
        torch.empty(0, dtype=torch.long),
        torch.zeros(4),
        panel_row_indices=torch.empty(0, dtype=torch.long),
        panel_total_chunks=4,
        mc_samples=4,
        physical_microbatch_size=3,
        flow_seed_root=31,
        cycle=0,
        global_task_id=4,
        device=torch.device("cpu"),
    )
    assert gradients == ()
    assert summary.failures == 4
    assert summary.functional_policy_forwards == 0
    with pytest.raises(RewardProtocolError, match="retention batch"):
        functional_success_lora_gradients(
            policy,
            state,
            contract,
            {},
            torch.empty(0, dtype=torch.long),
            torch.zeros(4),
            panel_row_indices=torch.empty(0, dtype=torch.long),
            panel_total_chunks=4,
            mc_samples=3,
            physical_microbatch_size=3,
            flow_seed_root=31,
            cycle=0,
            global_task_id=4,
            device=torch.device("cpu"),
        )
    with pytest.raises(RewardProtocolError, match="all-failure"):
        functional_success_lora_gradients(
            policy,
            state,
            contract,
            {"observation.leaked": torch.zeros(1)},
            torch.empty(0, dtype=torch.long),
            torch.zeros(4),
            panel_row_indices=torch.empty(0, dtype=torch.long),
            panel_total_chunks=4,
            mc_samples=4,
            physical_microbatch_size=3,
            flow_seed_root=31,
            cycle=0,
            global_task_id=4,
            device=torch.device("cpu"),
        )
    batch, episode_ids, successes = _success_replay()
    panel_rows, panel_chunks = _panel_rows()
    with pytest.raises(RewardProtocolError, match="success-only"):
        functional_success_lora_gradients(
            policy,
            state,
            contract,
            batch,
            torch.tensor([0, 0, 1, 2, 2]),
            successes,
            panel_row_indices=panel_rows,
            panel_total_chunks=panel_chunks,
            mc_samples=4,
            physical_microbatch_size=3,
            flow_seed_root=31,
            cycle=0,
            global_task_id=4,
            device=torch.device("cpu"),
        )
