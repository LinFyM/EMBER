from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.policy_response_writer import (
    FrozenPolicyResponseVideo,
    PolicyResponseEventToFactorWriter,
)
from ember.ecp.policy_response_writer.composer import (
    _effective_update_cap_factor,
    _effective_update_rms,
)
from ember.ecp.policy_response_writer.shared import (
    _video_splits,
    balanced_task_owners,
    causal_cutoff,
    functional_objective,
    owner_balanced_task_group,
    role_balanced_task_owners,
    shared_task_group,
    training_video_demos,
)
from ember.ecp.policy_response_writer.shared_training import (
    _clip_scale_and_direction_gradients,
)
from ember.ecp.policy_response_writer.training import (
    _functional_panel_config,
    _functional_runtime_inputs,
    _runtime_tasks_and_panels,
    _selected_task_ids,
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
        model.composer.scale_head.bias.fill_(10.0)
        bounded = model.composer(
            videos, processes, s_ref=torch.full((4,), 0.2)
        )
        assert all(
            _effective_update_rms(a, b) <= 0.2 + 2e-6
            for a, b in zip(bounded.a, bounded.b, strict=True)
        )
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


def test_complete_target_effective_update_is_capped_by_s_ref() -> None:
    generator = torch.Generator().manual_seed(23)
    a = torch.randn(4, 7, generator=generator, requires_grad=True)
    b = torch.randn(4, 11, generator=generator, requires_grad=True)
    cap = torch.tensor(0.2)
    uncapped = _effective_update_rms(a, b)
    torch.testing.assert_close(
        uncapped,
        (b.transpose(0, 1) @ a).square().mean().sqrt(),
    )
    factor = _effective_update_cap_factor(a, b, cap)
    capped = _effective_update_rms(a, b * factor)

    assert uncapped > cap
    torch.testing.assert_close(capped, cap, atol=2e-7, rtol=2e-6)
    assert 0 < factor < 1
    capped.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()

    small_b = b.detach() * 1e-4
    torch.testing.assert_close(
        _effective_update_cap_factor(a.detach(), small_b, cap),
        torch.tensor(1.0),
        atol=0,
        rtol=0,
    )

    zero_b = torch.zeros_like(b.detach(), requires_grad=True)
    zero_factor = _effective_update_cap_factor(a.detach(), zero_b, cap)
    torch.testing.assert_close(zero_factor, torch.tensor(1.0), atol=0, rtol=0)
    (zero_b * zero_factor).sum().backward()
    assert zero_b.grad is not None and torch.isfinite(zero_b.grad).all()


def test_shared_scale_and_direction_gradients_have_independent_clip_budgets() -> None:
    direction = torch.nn.Parameter(torch.zeros(2))
    scale = torch.nn.Parameter(torch.zeros(2))
    direction.grad = torch.tensor([3.0, 4.0])
    scale.grad = torch.tensor([6.0, 8.0])

    combined = _clip_scale_and_direction_gradients(
        parameters=(direction, scale),
        scale_parameters=(scale,),
        max_norm=1.0,
    )

    torch.testing.assert_close(combined, torch.sqrt(torch.tensor(125.0)))
    torch.testing.assert_close(direction.grad.norm(), torch.tensor(1.0))
    torch.testing.assert_close(scale.grad.norm(), torch.tensor(1.0))


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


def test_scaled_schedule_and_mixed_k_cover_registered_choices() -> None:
    meta = tuple(range(1, 56))
    target = tuple(range(72, 90))
    groups = [
        shared_task_group(meta, target, step, tasks_per_role=3, seed=19)
        for step in range(330)
    ]
    assert all(len(group) == 6 for group in groups)
    assert len({sum(task in group for group in groups) for task in meta}) == 1
    assert len({sum(task in group for group in groups) for task in target}) == 1

    tasks = (*meta, *target, 100, 101)
    owners = role_balanced_task_owners(
        {task: 100 + task % 17 for task in tasks},
        meta=meta,
        target=target,
        held=(100, 101),
        world_size=6,
    )
    owner_by_task = {
        task: rank for rank, row in enumerate(owners) for task in row
    }
    owner_groups = [
        owner_balanced_task_group(
            meta,
            target,
            step,
            task_owners=owners,
            tasks_per_role=3,
            seed=19,
        )
        for step in range(330)
    ]
    active_owner_counts = [
        len({owner_by_task[task] for task in group}) for group in owner_groups
    ]
    assert min(active_owner_counts) == 5
    assert sum(value == 6 for value in active_owner_counts) >= 0.96 * len(
        active_owner_counts
    )
    assert len(
        {sum(task in group for group in owner_groups) for task in meta}
    ) <= 2
    assert len(
        {sum(task in group for group in owner_groups) for task in target}
    ) <= 2

    three_owners = role_balanced_task_owners(
        {task: 100 + task % 17 for task in tasks},
        meta=meta,
        target=target,
        held=(100, 101),
        world_size=3,
    )
    three_owner_by_task = {
        task: rank for rank, row in enumerate(three_owners) for task in row
    }
    three_rank_loads = []
    for step in range(330):
        group = owner_balanced_task_group(
            meta,
            target,
            step,
            task_owners=three_owners,
            tasks_per_role=3,
            seed=19,
        )
        loads = [
            sum(three_owner_by_task[task] == rank for task in group)
            for rank in range(3)
        ]
        three_rank_loads.append(loads)
    assert all(min(loads) >= 1 for loads in three_rank_loads)
    assert sum(sorted(loads) == [2, 2, 2] for loads in three_rank_loads) >= (
        0.96 * len(three_rank_loads)
    )

    four_owners = role_balanced_task_owners(
        {task: 100 + task % 17 for task in tasks},
        meta=meta,
        target=target,
        held=(100, 101),
        world_size=4,
    )
    four_owner_by_task = {
        task: rank for rank, row in enumerate(four_owners) for task in row
    }
    four_rank_loads = []
    for step in range(330):
        group = owner_balanced_task_group(
            meta,
            target,
            step,
            task_owners=four_owners,
            tasks_per_role=3,
            seed=19,
        )
        four_rank_loads.append(
            [
                sum(four_owner_by_task[task] == rank for task in group)
                for rank in range(4)
            ]
        )
    assert all(max(loads) <= 2 for loads in four_rank_loads)
    assert sum(min(loads) >= 1 for loads in four_rank_loads) >= (
        0.70 * len(four_rank_loads)
    )

    four = [
        training_video_demos(
            (3, 7, 11, 13),
            optimizer_step=step,
            task=8,
            cardinalities=(1, 2, 4),
            seed=23,
        )
        for step in range(12)
    ]
    two = [
        training_video_demos(
            (3, 7),
            optimizer_step=step,
            task=8,
            cardinalities=(1, 2, 4),
            seed=23,
        )
        for step in range(8)
    ]
    assert {len(value) for value in four} == {1, 2, 4}
    assert {len(value) for value in two} == {1, 2}
    assert all(len(value) == len(set(value)) for value in (*four, *two))


def test_scalable_panel_roots_and_video_split_are_outcome_independent(
    tmp_path: Path,
) -> None:
    selected = (1, 72, 2, 74)
    roots = []
    for index, tasks in enumerate(((1, 72), (2, 74))):
        root = tmp_path / f"source_{index}"
        shard = root / "shard_0"
        shard.mkdir(parents=True)
        (root / "completion.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        for task in tasks:
            (shard / f"task_{task:03d}.json").write_text("{}", encoding="utf-8")
        roots.append(
            {
                "root": str(root),
                "completion": "completion.json",
                "task_count": len(tasks),
            }
        )
    config = {
        "authorities": {"functional_panel_sources": roots},
        "task_split": {
            "gradient_meta": [1],
            "gradient_target": [72],
            "true_task_held_meta": [2],
            "true_task_held_target": [74],
        },
    }
    resolved = _functional_panel_config(config, asset_root=tmp_path)
    assert _selected_task_ids(config) == selected
    assert set(resolved["authorities"]["functional_panel_records"]) == {
        "1",
        "2",
        "72",
        "74",
    }

    panels = {
        task: SimpleNamespace(program_video_demos=(1, 3, 5, 7, 9))
        for task in selected
    }
    runtime = SimpleNamespace(
        config={
            "data": {
                "video_split": {
                    "source": "functional_panel_program_video_demos",
                    "fit_pool_max": 4,
                    "held_selection": "last_sorted",
                    "selection_uses_outcomes": False,
                }
            }
        },
        panels=panels,
        video_store=SimpleNamespace(frame_counts=lambda task, demo: (20, demo + 10)),
    )
    splits, costs = _video_splits(runtime, selected)
    assert splits == {task: ((1, 3, 5, 7), 9) for task in selected}
    assert costs == {task: 75 for task in selected}


def test_deployment_runtime_has_no_functional_data_path() -> None:
    tasks = (
        SimpleNamespace(authority_id=72, role="target_fit", domain_task_id=2),
        SimpleNamespace(authority_id=76, role="target_held", domain_task_id=9),
        SimpleNamespace(authority_id=71, role="target_held", domain_task_id=0),
    )
    selected, panels = _runtime_tasks_and_panels(
        SimpleNamespace(), {}, tasks, deployment_global_ids=(0, 9)
    )
    dataset, processor = _functional_runtime_inputs(
        authorities=(),
        source_config={},
        base={},
        args=SimpleNamespace(),
        context=SimpleNamespace(),
        enabled=False,
    )

    assert tuple(task.authority_id for task in selected) == (71, 76)
    assert panels == {}
    assert dataset is None
    assert processor is None
