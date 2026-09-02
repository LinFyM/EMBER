from __future__ import annotations

from dataclasses import replace

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.policy_response_writer import (
    FrozenPolicyResponseVideo,
    PolicyResponseEventToFactorWriter,
)
from ember.ecp.policy_response_writer.shared import (
    balanced_task_owners,
    causal_cutoff,
    functional_objective,
    shared_task_group,
)


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
        torch.randn(frames, 2, 50, owner.out_features, generator=generator)
        for owner in owners
    )
    return FrozenPolicyResponseVideo(
        patch_states=torch.randn(frames, 5, 10, generator=generator),
        language_states=torch.randn(frames, 3, 10, generator=generator),
        language_mask=torch.ones(frames, 3, dtype=torch.bool),
        layer_states=torch.randn(frames, 2, 19, 50, 12, generator=generator),
        flow_velocity=torch.randn(frames, 2, 50, 32, generator=generator),
        suffix_noise=torch.stack(
            (
                torch.randn(50, 32, generator=generator),
                torch.randn(50, 32, generator=generator),
            )
        ),
        native_inputs=tuple(
            torch.randn(frames, 2, 50, owner.in_features, generator=generator)
            for owner in owners
        ),
        native_outputs=outputs,
        final_outputs=tuple(value[-1] for value in outputs),
        frame_positions=torch.linspace(0.0, 1.0, frames),
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
        event_blocks=1,
        composer_blocks=1,
        pooling_frame_chunk=2,
        task_local=task_local,
    )


def test_full_writer_has_functional_gradients_and_frozen_causal_target() -> None:
    model = _model()
    video = _video(7)
    output = model((video,), s_ref=torch.full((4,), 0.2))

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
    process_loss = model.causal_prediction_loss(
        (video,), cutoffs=((5,),), future_offset=1
    )
    factor_loss = sum(
        value.square().mean() for value in output.residual.a + output.residual.b
    )
    (factor_loss + process_loss).backward()

    parameters = (
        model.process.patch_projection.weight,
        model.process.frame_blocks[0].response_attention.in_proj_weight,
        model.process.events.blocks[0].event_attention.in_proj_weight,
        model.process.prediction_head[-1].weight,
        model.composer.common_query.weight,
        model.composer.input_positive_query.weight,
        model.composer.input_projection["6"].weight,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert not any(
        name.startswith("teacher_") for name, _ in model.named_parameters()
    )


def test_causal_prefix_cannot_read_mutated_future_frames() -> None:
    model = _model().eval()
    video = _video(11)
    stop = 5
    changed = replace(
        video,
        patch_states=torch.cat((video.patch_states[:stop], video.patch_states[stop:] + 50)),
        language_states=torch.cat(
            (video.language_states[:stop], video.language_states[stop:] - 40)
        ),
        layer_states=torch.cat((video.layer_states[:stop], video.layer_states[stop:] * 3)),
        flow_velocity=torch.cat(
            (video.flow_velocity[:stop], video.flow_velocity[stop:] - 25)
        ),
    )
    with torch.no_grad():
        left = model.process(video.frame_slice(stop), causal=True)
        right = model.process(changed.frame_slice(stop), causal=True)

    torch.testing.assert_close(left.events, right.events, rtol=0, atol=0)
    torch.testing.assert_close(left.frame_innovation, right.frame_innovation, rtol=0, atol=0)


def test_composer_zero_innovation_chunking_and_video_order_contracts() -> None:
    model = _model(task_local=True).eval()
    videos = (_video(17), _video(19, frames=7))
    with torch.no_grad():
        processes = tuple(model.process(video) for video in videos)
        initialized = model.composer(
            videos, processes, s_ref=torch.full((4,), 0.2)
        )
        assert any(torch.count_nonzero(value) > 0 for value in initialized.a)
        assert all(torch.count_nonzero(value) == 0 for value in initialized.b)
        zero = tuple(
            replace(
                process,
                innovations=torch.zeros_like(process.innovations),
                frame_innovation=torch.zeros_like(process.frame_innovation),
            )
            for process in processes
        )
        absent = model.composer(videos, zero, s_ref=torch.ones(4))
        assert all(torch.count_nonzero(value) == 0 for value in absent.a + absent.b)

        model.composer.pooling_frame_chunk = 1
        chunked = model.composer(videos, processes, s_ref=torch.full((4,), 0.2))
        model.composer.pooling_frame_chunk = 20
        whole = model.composer(videos, processes, s_ref=torch.full((4,), 0.2))
        reversed_order = model.composer(
            tuple(reversed(videos)),
            tuple(reversed(processes)),
            s_ref=torch.full((4,), 0.2),
        )

    for left, right, permuted in zip(
        chunked.a + chunked.b,
        whole.a + whole.b,
        reversed_order.a + reversed_order.b,
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(left, permuted, atol=2e-4, rtol=2e-4)


def test_shared_schedule_ownership_and_positive_only_objective() -> None:
    meta = (1, 8, 9, 32, 52)
    target = (72, 73, 75, 93, 94)
    groups = [shared_task_group(meta, target, step) for step in range(5)]
    assert all(len(group) == 6 for group in groups)
    assert all(len(set(group[:3])) == len(set(group[3:])) == 3 for group in groups)
    assert {
        task: sum(task in group for group in groups) for task in (*meta, *target)
    } == {task: 3 for task in (*meta, *target)}

    owners = balanced_task_owners(
        {task: 100 + index for index, task in enumerate((*meta, *target, 2, 74))},
        6,
    )
    assert sorted(task for row in owners for task in row) == sorted(
        (*meta, *target, 2, 74)
    )
    assert max(map(len, owners)) == 2

    protected = functional_objective(
        generated_loss=0.12,
        carrier_loss=0.10,
        normalizer=0.10,
        task_weight=1 / 6,
        preservation_weight=0.05,
        preservation_epsilon=0.0,
    )
    improving = functional_objective(
        generated_loss=0.08,
        carrier_loss=0.10,
        normalizer=0.10,
        task_weight=1 / 6,
        preservation_weight=0.05,
        preservation_epsilon=0.0,
    )
    assert protected["preservation_active"] is True
    assert improving["preservation_active"] is False
    assert protected["gradient_mass"] > improving["gradient_mass"]
    assert causal_cutoff(20, 8, optimizer_step=100, task=93, demo=2) in range(8, 20)
