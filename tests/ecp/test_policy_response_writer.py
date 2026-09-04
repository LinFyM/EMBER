from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import G1_RESIDUAL_RANK, native_output_group_count
from ember.ecp.policy_response_writer import (
    FrozenPolicyResponseVideo,
    PolicyResponseEventToFactorWriter,
)
from ember.ecp.policy_response_writer.composer import (
    FrameBankFactorBlock,
    _effective_update_cap_factor,
    _effective_update_rms,
)
from ember.ecp.policy_response_writer.shared import _optimizer, functional_objective
from ember.ecp.policy_response_writer.shared_execution import (
    assignment_makespan,
    cost_balanced_task_assignment,
)
from ember.ecp.policy_response_writer.shared_schedule import (
    counted_task_group,
    task_group_counts,
    task_occurrence_schedule,
    training_video_demos,
)
from ember.ecp.policy_response_writer.training import load_policy_response_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "q", TargetFamily.Q, 0, 6, 16),
        TargetOwner(1, "v", TargetFamily.V, 1, 7, 8),
        TargetOwner(2, "action_in", TargetFamily.ACTION_IN, None, 2, 8),
        TargetOwner(3, "action_out", TargetFamily.ACTION_OUT, None, 5, 3),
    )


def _video(seed: int, *, frames: int = 6) -> FrozenPolicyResponseVideo:
    generator = torch.Generator().manual_seed(seed)
    owners = _owners()
    outputs = tuple(
        torch.randn(
            frames, 2, ACTION_HORIZON, owner.out_features, generator=generator
        )
        for owner in owners
    )
    return FrozenPolicyResponseVideo(
        patch_states=torch.randn(frames, 5, 10, generator=generator),
        language_states=torch.randn(frames, 3, 10, generator=generator),
        language_mask=torch.ones(frames, 3, dtype=torch.bool),
        layer_states=torch.randn(
            frames, 2, 19, ACTION_HORIZON, 12, generator=generator
        ),
        flow_velocity=torch.randn(
            frames, 2, ACTION_HORIZON, 32, generator=generator
        ),
        suffix_noise=torch.randn(2, ACTION_HORIZON, 32, generator=generator),
        native_inputs=tuple(
            torch.randn(
                frames, 2, ACTION_HORIZON, owner.in_features, generator=generator
            )
            for owner in owners
        ),
        native_outputs=outputs,
        final_outputs=tuple(value[-1] for value in outputs),
        frame_positions=torch.linspace(0.0, 1.0, frames),
    )


def _repeat_frames(value: torch.Tensor) -> torch.Tensor:
    return value[:1].expand_as(value).clone()


def _static_repeated_video(seed: int, *, frames: int = 6) -> FrozenPolicyResponseVideo:
    video = _video(seed, frames=frames)
    outputs = tuple(_repeat_frames(value) for value in video.native_outputs)
    return replace(
        video,
        patch_states=_repeat_frames(video.patch_states),
        language_states=_repeat_frames(video.language_states),
        language_mask=_repeat_frames(video.language_mask),
        layer_states=_repeat_frames(video.layer_states),
        flow_velocity=_repeat_frames(video.flow_velocity),
        native_inputs=tuple(_repeat_frames(value) for value in video.native_inputs),
        native_outputs=outputs,
        final_outputs=tuple(value[-1] for value in outputs),
    )


def _reverse_video(video: FrozenPolicyResponseVideo) -> FrozenPolicyResponseVideo:
    outputs = tuple(value.flip(0) for value in video.native_outputs)
    return replace(
        video,
        patch_states=video.patch_states.flip(0),
        language_states=video.language_states.flip(0),
        language_mask=video.language_mask.flip(0),
        layer_states=video.layer_states.flip(0),
        flow_velocity=video.flow_velocity.flip(0),
        native_inputs=tuple(value.flip(0) for value in video.native_inputs),
        native_outputs=outputs,
        final_outputs=tuple(value[-1] for value in outputs),
    )


def _model(*, task_local: bool = False) -> PolicyResponseEventToFactorWriter:
    return PolicyResponseEventToFactorWriter(
        _owners(),
        prefix_width=10,
        expert_width=12,
        width=16,
        event_slots=4,
        heads=4,
        frame_blocks=1,
        temporal_blocks=1,
        event_blocks=1,
        composer_blocks=1,
        pooling_frame_chunk=2,
        task_local=task_local,
    )


def _residual_loss(model_output: object) -> torch.Tensor:
    residual = model_output.residual
    return sum(value.square().mean() for value in (*residual.a, *residual.b))


def _group_gradient(model: torch.nn.Module, prefix: str) -> float:
    rows = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return float(torch.stack(rows).sum().sqrt()) if rows else 0.0


def test_axial_writer_preserves_shapes_and_one_functional_gradient_path() -> None:
    model = _model()
    output = model((_video(7),), s_ref=torch.full((4,), 0.2))

    assert tuple(value.shape for value in output.residual.a) == (
        (4, 6),
        (4, 7),
        (4, 2),
        (4, 5),
    )
    assert tuple(value.shape for value in output.residual.b) == (
        (4, 16),
        (4, 8),
        (4, 8),
        (4, 3),
    )
    assert output.residual.scales.shape == (4, G1_RESIDUAL_RANK)

    _residual_loss(output).backward()
    for prefix in (
        "process.response",
        "process.frame_blocks",
        "process.temporal_blocks",
        "process.events",
        "composer.blocks",
        "composer.blocks.0.bank_attention",
        "composer.input_contrast_query",
        "composer.output_contrast_query",
    ):
        assert _group_gradient(model, prefix) > 0.0


def test_full_horizon_is_explicit_and_coarse_is_rejected() -> None:
    model = _model()
    video = _video(11)
    response = model.process.response(video)
    assert response.shape == (
        video.frame_count,
        len(_owners()),
        ACTION_HORIZON * 8,
        16,
    )

    changed = video.layer_states.clone()
    changed[:, :, :, 17] += 3.0
    mutated = model.process.response(replace(video, layer_states=changed))
    assert not torch.equal(
        response[:, :, 17 * 8 : 18 * 8],
        mutated[:, :, 17 * 8 : 18 * 8],
    )
    assert torch.equal(response[:, :, : 17 * 8], mutated[:, :, : 17 * 8])
    with pytest.raises(ValueError, match="full policy-response"):
        model((video,), s_ref=torch.full((4,), 0.2), representation="coarse")


def test_static_repeated_video_cannot_open_either_dynamic_factor() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model(
            (_static_repeated_video(19),),
            s_ref=torch.full((4,), 0.2),
        )
    assert output.processes[0].frame_innovations.abs().max() < 1e-5
    assert output.processes[0].events.abs().max() < 1e-5
    assert max(value.abs().max() for value in output.residual.a) < 1e-5
    assert max(value.abs().max() for value in output.residual.b) < 1e-5


def test_order_changes_events_and_factors() -> None:
    model = _model().eval()
    video = _video(23, frames=7)
    with torch.no_grad():
        forward = model((video,), s_ref=torch.full((4,), 0.2))
        reverse = model((_reverse_video(video),), s_ref=torch.full((4,), 0.2))
    assert not torch.allclose(forward.processes[0].events, reverse.processes[0].events)
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(
            (*forward.residual.a, *forward.residual.b),
            (*reverse.residual.a, *reverse.residual.b),
            strict=True,
        )
    )


def test_video_set_is_permutation_invariant_and_chunking_is_exact() -> None:
    model = _model().eval()
    left, right = _video(29), _video(31, frames=7)
    with torch.no_grad():
        original = model((left, right), s_ref=torch.full((4,), 0.2)).residual
        permuted = model((right, left), s_ref=torch.full((4,), 0.2)).residual
        model.composer.pooling_frame_chunk = 3
        rechunked = model((left, right), s_ref=torch.full((4,), 0.2)).residual
    for expected, observed, chunked in zip(
        (*original.a, *original.b),
        (*permuted.a, *permuted.b),
        (*rechunked.a, *rechunked.b),
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(chunked, expected, rtol=2e-5, atol=2e-6)


def test_native_bank_keeps_every_candidate_axis_for_frame_local_read() -> None:
    model = _model().eval()
    video = _video(37, frames=5)
    with torch.no_grad():
        for target, owner in enumerate(_owners()):
            candidates = model.composer._bank_candidates(target, (video,))
            groups = native_output_group_count(owner)
            expected = video.frame_count * 2 * ACTION_HORIZON * (1 + groups * 4)
            observed = sum(
                chunk.context_tokens.numel() // model.composer.width
                for chunk in candidates[0].chunks
            )
            assert observed == expected
            assert sum(row.frame_count for row in candidates) == video.frame_count


def test_frame_bank_block_preserves_zero_dynamic_path_and_reads_local_bank() -> None:
    torch.manual_seed(41)
    block = FrameBankFactorBlock(16, 4).double().eval()
    query = torch.randn(4, 16, dtype=torch.double)
    positions = (torch.linspace(0.0, 1.0, 5, dtype=torch.double),)
    zero_event = (torch.zeros(4, 3, 16, dtype=torch.double),)
    zero_frame = (torch.zeros(5, 4, 16, dtype=torch.double),)
    repeated_bank = torch.randn(1, 12, 16, dtype=torch.double).expand(
        5, -1, -1
    ).clone()
    with torch.no_grad():
        zero_aligned = block(
            query, zero_event, zero_frame, positions, ((repeated_bank,),)
        )
    torch.testing.assert_close(zero_aligned[0], zero_frame[0], atol=1e-12, rtol=0.0)

    event = (torch.randn(4, 3, 16, dtype=torch.double),)
    frame = torch.randn(5, 4, 16, dtype=torch.double)
    bank = torch.randn(5, 12, 16, dtype=torch.double)
    changed = bank.clone()
    changed[2] = 3.0 * changed[2].flip(0)
    with torch.no_grad():
        original = block(query, event, (frame,), positions, ((bank,),))
        mutated = block(query, event, (frame,), positions, ((changed,),))
    assert not torch.allclose(original[0], mutated[0])
    assert not torch.allclose(
        original[0][2], mutated[0][2]
    )


def test_complete_target_update_is_capped_once() -> None:
    generator = torch.Generator().manual_seed(43)
    a = torch.randn(4, 13, generator=generator)
    b = torch.randn(4, 17, generator=generator)
    cap = torch.tensor(0.07)
    factor = _effective_update_cap_factor(a, b, cap)
    assert _effective_update_rms(a, b * factor) <= cap + 1e-6


def test_shared_optimizer_owns_the_whole_writer_in_one_group() -> None:
    writer = _model()
    policy = torch.nn.Linear(2, 2).requires_grad_(False)
    stage0 = torch.nn.Linear(2, 2).requires_grad_(False)
    runtime = SimpleNamespace(
        writer=writer,
        policy=policy,
        stage0=stage0,
        config={
            "optimization": {
                "shared": {
                    "training_stage": "joint_functional_positive_only",
                    "learning_rate": 1e-4,
                    "decay_learning_rate": 1e-6,
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                    "warmup_updates": 2,
                    "effective_updates": 4,
                }
            }
        },
    )
    parameters, optimizer, _ = _optimizer(runtime)
    assert {id(value) for value in parameters} == {
        id(value) for value in writer.parameters()
    }
    assert len(optimizer.param_groups) == 1


def test_positive_functional_objective_has_no_negative_video_term() -> None:
    row = functional_objective(
        generated_loss=1.2,
        normalizer=2.0,
        task_weight=0.25,
    )
    assert row["gradient_mass"] == pytest.approx(0.125)
    assert set(row) == {
        "functional_normalized",
        "gradient_mass",
    }


def test_task_batch_size_and_role_ratio_are_experiment_settings() -> None:
    meta = tuple(range(10))
    target = tuple(range(20, 26))
    assert task_group_counts(
        {
            "global_tasks_per_update": 5,
            "tasks_per_update_by_role": {"meta": 4, "target": 1},
        },
        meta=meta,
        target=target,
    ) == (4, 1)
    group = counted_task_group((meta, target), (4, 1), 0, seed=17)
    assert len(group) == 5
    assert len(set(group).intersection(meta)) == 4
    assert len(set(group).intersection(target)) == 1


def test_task_local_occurrence_drives_k_without_global_step_aliasing() -> None:
    groups = ((1, 2), (3, 4), (1, 3), (2, 4), (1, 4))
    occurrences = task_occurrence_schedule(groups)
    assert [row.get(1) for row in occurrences if 1 in row] == [0, 1, 2]
    selected = [
        training_video_demos(
            (0, 1, 2, 3),
            task_occurrence=row[1],
            task=1,
            cardinalities=(1, 2, 4),
            seed=19,
        )
        for row in occurrences
        if 1 in row
    ]
    assert {len(value) for value in selected} == {1, 2, 4}


def test_dynamic_cost_assignment_reduces_tail_without_changing_tasks() -> None:
    costs = {0: 19, 1: 17, 2: 13, 3: 11, 4: 7, 5: 5}
    eligibility = {task: (0, 1, 2) for task in costs}
    assignment = cost_balanced_task_assignment(
        tuple(costs), costs, eligibility, world_size=3
    )
    assert {task for row in assignment for task in row} == set(costs)
    assert assignment_makespan(assignment, costs) <= 25


def test_frame_bank_config_is_canonical_and_old_serial_config_is_rejected() -> None:
    current = load_policy_response_config(
        REPO_ROOT / "configs/pi05_ecp_policy_response_writer_frame_bank_v1.json"
    )
    assert current["model"]["temporal_blocks"] == 2
    assert current["model"]["representation_arms"] == ["full"]
    assert current["optimization"]["objective"].endswith("positive_only")
    assert "composer_gain_blocks" not in current["model"]
    with pytest.raises(ValueError, match="invalid Policy-Response Writer config"):
        load_policy_response_config(
            REPO_ROOT
            / "configs/pi05_ecp_policy_response_writer_factor_set_relative_gain_role_equal_v1.json"
        )
