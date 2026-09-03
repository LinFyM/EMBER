from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ember.ecp.policy_response_writer.composer as composer_module
from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    OnlineSoftmaxAccumulator,
    native_output_group_count,
)
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
from ember.ecp.policy_response_writer.shared_schedule import (
    counted_task_group,
    task_group_counts,
)
from ember.ecp.policy_response_writer.shared_training import (
    _clip_scale_and_direction_gradients,
)
from ember.ecp.policy_response_writer.shared_execution import (
    assignment_makespan,
    cost_balanced_task_assignment,
    selective_replication_plan,
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


def _static_repeated_video(seed: int, *, frames: int = 8) -> FrozenPolicyResponseVideo:
    video = _video(seed, frames=frames)
    repeat = lambda value: value[:1].expand_as(value).clone()
    outputs = tuple(repeat(value) for value in video.native_outputs)
    return replace(
        video,
        patch_states=repeat(video.patch_states),
        language_states=repeat(video.language_states),
        language_mask=repeat(video.language_mask),
        layer_states=repeat(video.layer_states),
        flow_velocity=repeat(video.flow_velocity),
        native_inputs=tuple(repeat(value) for value in video.native_inputs),
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
        model.process.events.frame_position_projection.weight,
        model.process.prediction_head[-1].weight,
        model.composer.common_query.weight,
        model.composer.input_positive_query.weight,
        model.composer.relation_embedding.weight,
        model.composer.input_projection["6"].weight,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert torch.count_nonzero(
        model.process.events.frame_position_projection.weight.grad
    )
    assert torch.count_nonzero(model.composer.relation_embedding.weight.grad)
    assert not any(name.startswith("teacher_") for name, _ in model.named_parameters())


def test_composer_consumes_explicit_event_relation_assignment() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = _model().eval()
    video = _video(13)
    with torch.no_grad():
        process = model.process(video)
        permuted = replace(process, assignment=process.assignment.roll(1, dims=-1))
        model.composer.scale_head.bias.fill_(10.0)
        original = model.composer((video,), (process,), s_ref=torch.full((4,), 0.2))
        changed = model.composer((video,), (permuted,), s_ref=torch.full((4,), 0.2))

    differences = [
        torch.max(
            torch.abs(
                left_b.transpose(0, 1) @ left_a - right_b.transpose(0, 1) @ right_a
            )
        )
        for left_a, left_b, right_a, right_b in zip(
            original.a, original.b, changed.a, changed.b, strict=True
        )
    ]
    assert max(differences) > 1e-6


def test_factored_relation_logits_match_explicit_native_width_product() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = _model().eval()
    composer = model.composer
    query = torch.randn(4, 16, requires_grad=True)
    keys = torch.randn(3, 2, 5, 16, requires_grad=True)
    relation_innovation = torch.randn(3, 4, 16, requires_grad=True)

    actual = composer._branch_logits(
        query,
        keys,
        relation_innovation,
        log_base_mass=-math.log(30),
        output=False,
    )
    common_query = composer.common_query(query)
    positive_query = composer.input_positive_query(query)
    negative_query = composer.input_negative_query(query)
    common = torch.einsum("rd,...d->r...", common_query, keys) / math.sqrt(16)
    relation_scale = 1.0 + torch.tanh(composer.relation_embedding.weight)
    key_feature = torch.tanh(keys)
    explicit = []
    for relation in range(4):
        innovation = (
            composer.innovation_key(relation_innovation[:, relation])
            * relation_scale[relation]
        )[:, None, None]
        feature = innovation * key_feature
        positive = torch.einsum(
            "rd,...d->r...", positive_query, feature
        ) / math.sqrt(16)
        negative = torch.einsum(
            "rd,...d->r...", negative_query, feature
        ) / math.sqrt(16)
        explicit.append(
            torch.stack((common + positive, common + negative), dim=1)
            - math.log(120)
        )
    expected = torch.logsumexp(torch.stack(explicit), dim=0)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    differentiated = (
        query,
        keys,
        relation_innovation,
        composer.innovation_key.weight,
        composer.relation_embedding.weight,
    )
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), differentiated, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), differentiated)
    for left, right in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)


def test_retired_coarse_representation_is_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="only active representation"):
        model.process(_video(9), representation="coarse")


def test_causal_prefix_cannot_read_mutated_future_frames() -> None:
    model = _model().eval()
    video = _video(11)
    stop = 5
    changed = replace(
        video,
        patch_states=torch.cat(
            (video.patch_states[:stop], video.patch_states[stop:] + 50)
        ),
        language_states=torch.cat(
            (video.language_states[:stop], video.language_states[stop:] - 40)
        ),
        layer_states=torch.cat(
            (video.layer_states[:stop], video.layer_states[stop:] * 3)
        ),
        flow_velocity=torch.cat(
            (video.flow_velocity[:stop], video.flow_velocity[stop:] - 25)
        ),
    )
    with torch.no_grad():
        left = model.process(video.frame_slice(stop), causal=True)
        right = model.process(changed.frame_slice(stop), causal=True)

    torch.testing.assert_close(left.events, right.events, rtol=0, atol=0)
    torch.testing.assert_close(
        left.frame_innovation, right.frame_innovation, rtol=0, atol=0
    )


def test_composer_zero_innovation_chunking_and_video_order_contracts() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _model(task_local=True).eval()
    videos = (_video(17), _video(19, frames=7))
    with torch.no_grad():
        processes = tuple(model.process(video) for video in videos)
        initialized = model.composer(videos, processes, s_ref=torch.full((4,), 0.2))
        assert any(torch.count_nonzero(value) > 0 for value in initialized.a)
        assert all(torch.count_nonzero(value) == 0 for value in initialized.b)
        model.composer.scale_head.bias.fill_(10.0)
        bounded = model.composer(videos, processes, s_ref=torch.full((4,), 0.2))
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

    for left_a, left_b, right_a, right_b, permuted_a, permuted_b in zip(
        chunked.a,
        chunked.b,
        whole.a,
        whole.b,
        reversed_order.a,
        reversed_order.b,
        strict=True,
    ):
        left = left_b.transpose(0, 1) @ left_a
        right = right_b.transpose(0, 1) @ right_a
        permuted = permuted_b.transpose(0, 1) @ permuted_a
        # Exact online reductions preserve the mathematical set result; the
        # extra relation marginal changes only FP32 reduction order here.
        torch.testing.assert_close(left, right, atol=4e-4, rtol=4e-4)
        torch.testing.assert_close(left, permuted, atol=4e-4, rtol=4e-4)


def test_composer_bank_memory_retains_every_action_horizon() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = _model().eval()
        videos = (_video(23), _video(29, frames=7))
        with torch.no_grad():
            processes = tuple(model.process(video) for video in videos)
            memory, _ = model.composer._bank_candidates(0, videos, processes)

        groups = native_output_group_count(_owners()[0])
        tokens_per_frame = 2 * ACTION_HORIZON * (1 + 4 * groups)
        assert sum(chunk.shape[0] for chunk in memory) == (
            sum(video.frame_count for video in videos) * tokens_per_frame
        )
        assert all(chunk.shape[1:] == (model.composer.width,) for chunk in memory)


def test_streaming_bank_attention_matches_dense_attention(monkeypatch) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        generator = torch.Generator().manual_seed(41)
        block = _model().composer.blocks[0].eval()
        query = torch.randn(1, 4, 16, generator=generator)
        memory = (
            torch.randn(7, 16, generator=generator),
            torch.randn(11, 16, generator=generator),
            torch.randn(5, 16, generator=generator),
        )
        monkeypatch.setattr(composer_module, "STREAMING_BANK_BLOCK_TOKEN_LIMIT", 12)
        with torch.no_grad():
            streamed = block._streaming_bank_attention(query, memory)
            dense = torch.cat(tuple(block.bank_norm(chunk) for chunk in memory))[None]
            expected, _ = block.bank_attention(query, dense, dense, need_weights=False)

        torch.testing.assert_close(streamed, expected, atol=2e-6, rtol=2e-5)


def test_fused_bank_attention_matches_streaming_outputs_and_gradients() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        generator = torch.Generator().manual_seed(43)
        dense_block = _model().composer.blocks[0].eval()
        streaming_block = _model().composer.blocks[0].eval()
        streaming_block.load_state_dict(dense_block.state_dict())
        dense_query = torch.randn(1, 4, 16, generator=generator, requires_grad=True)
        dense_memory = tuple(
            torch.randn(size, 16, generator=generator, requires_grad=True)
            for size in (7, 11, 5)
        )
        streaming_query = dense_query.detach().clone().requires_grad_(True)
        streaming_memory = tuple(
            value.detach().clone().requires_grad_(True) for value in dense_memory
        )

        dense = dense_block._dense_bank_attention(dense_query, dense_memory)
        streamed = streaming_block._streaming_bank_attention(
            streaming_query, streaming_memory
        )
        dense.square().sum().backward()
        streamed.square().sum().backward()

        torch.testing.assert_close(dense, streamed, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            dense_query.grad, streaming_query.grad, atol=3e-6, rtol=3e-5
        )
        for left, right in zip(dense_memory, streaming_memory, strict=True):
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-5)
        for left, right in zip(
            dense_block.parameters(), streaming_block.parameters(), strict=True
        ):
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-5)


def test_fused_video_pooling_matches_chunked_outputs_and_gradients() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        model = _model().eval()
        videos = (_video(47, frames=7), _video(53, frames=5))
        processes = tuple(model.process(video) for video in videos)
        bank, candidates = model.composer._bank_candidates(0, videos, processes)
        query = model.composer._query_target(0, processes, bank)
        fused = model.composer._pool_target(0, query, candidates)

        owner = model.composer.owners[0]
        groups = native_output_group_count(owner)
        group_width = owner.out_features // groups
        input_accumulator = OnlineSoftmaxAccumulator(
            ranks=G1_RESIDUAL_RANK,
            width=owner.in_features,
            device=query.device,
        )
        output_accumulators = tuple(
            OnlineSoftmaxAccumulator(
                ranks=G1_RESIDUAL_RANK,
                width=group_width,
                device=query.device,
            )
            for _ in range(groups)
        )
        for video in candidates:
            input_mass = -math.log(video.frame_count * 2 * ACTION_HORIZON)
            output_mass = -math.log(video.frame_count * 2 * ACTION_HORIZON * 4)
            for chunk in video.chunks:
                input_accumulator.add(
                    model.composer._branch_logits(
                        query,
                        chunk.input_keys,
                        chunk.innovation,
                        log_base_mass=input_mass,
                        output=False,
                    ),
                    chunk.input_values,
                )
                for group, accumulator in enumerate(output_accumulators):
                    accumulator.add(
                        model.composer._branch_logits(
                            query,
                            chunk.output_keys[group],
                            chunk.innovation,
                            log_base_mass=output_mass,
                            output=True,
                        ),
                        chunk.output_values[group],
                    )
        reference = (
            input_accumulator.signed_mean(),
            torch.cat(
                tuple(value.signed_mean() for value in output_accumulators), dim=-1
            ),
        )

        for left, right in zip(fused, reference, strict=True):
            torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
        parameters = (
            query,
            model.composer.common_query.weight,
            model.composer.innovation_key.weight,
            model.composer.input_positive_query.weight,
            model.composer.input_negative_query.weight,
            model.composer.output_positive_query.weight,
            model.composer.output_negative_query.weight,
        )
        fused_gradients = torch.autograd.grad(
            sum(value.square().sum() for value in fused),
            parameters,
            retain_graph=True,
        )
        reference_gradients = torch.autograd.grad(
            sum(value.square().sum() for value in reference), parameters
        )
        for left, right in zip(fused_gradients, reference_gradients, strict=True):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)


def test_static_repeated_video_cannot_open_mobile_lora() -> None:
    # The invariant is about a repeated video's dynamic content, not about
    # whichever global RNG state earlier tests happened to leave behind.
    # Keep the randomly initialized projections fixed so the small accepted
    # floating-point residue has a stable, meaningful bound.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _model().eval()
    video = _static_repeated_video(29)
    with torch.no_grad():
        process = model.process(video)
        model.composer.scale_head.bias.fill_(10.0)
        output = model.composer((video,), (process,), s_ref=torch.full((4,), 0.2))

    assert process.innovations.float().square().mean().sqrt() < 1e-6
    assert process.frame_innovation.float().square().mean().sqrt() < 1e-6
    assert all(
        _effective_update_rms(a, b) < 1e-4
        for a, b in zip(output.a, output.b, strict=True)
    )


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
    owner_by_task = {task: rank for rank, row in enumerate(owners) for task in row}
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
    assert len({sum(task in group for group in owner_groups) for task in meta}) <= 2
    assert len({sum(task in group for group in owner_groups) for task in target}) <= 2

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


def test_execution_assignment_is_exact_and_does_not_change_the_task_group() -> None:
    group = (1, 8, 9, 72, 73, 93)
    costs = {1: 20, 8: 9, 9: 11, 72: 12, 73: 14, 93: 30}
    eligibility = {
        1: (0,),
        8: (0, 2),
        9: (1,),
        72: (1, 3),
        73: (2,),
        93: (3,),
    }
    assignment = cost_balanced_task_assignment(group, costs, eligibility, world_size=4)

    assert sorted(task for row in assignment for task in row) == sorted(group)
    assert sum(len(row) for row in assignment) == len(group)
    assert assignment_makespan(assignment, costs) == 30
    assert assignment == cost_balanced_task_assignment(
        group, costs, eligibility, world_size=4
    )

    large = tuple(range(24))
    large_assignment = cost_balanced_task_assignment(
        large,
        {task: 1 + task % 7 for task in large},
        {task: tuple(range(6)) for task in large},
        world_size=6,
    )
    assert sorted(task for row in large_assignment for task in row) == list(large)
    assert max(len(row) for row in large_assignment) <= 5


def test_selective_replication_reaches_unconstrained_tail_without_full_copy() -> None:
    # Task 777 is registered and cached but absent from this short finite profile.
    owners = ((1, 93, 777), (8, 9), (72,), (73,))
    steps = (
        {1: 20, 8: 9, 9: 11, 72: 12, 73: 14, 93: 30},
        {1: 20, 8: 9, 9: 11, 72: 12, 73: 14, 93: 30},
    )
    plan = selective_replication_plan(
        steps,
        base_task_owners=owners,
        cache_bytes={
            **{task: 100 + task for task in steps[0]},
            777: 877,
        },
        extra_budget_bytes=10_000,
    )

    assert plan["predicted_total_cost"] == plan["ideal_total_cost"]
    assert plan["predicted_tail_cost"] == plan["ideal_tail_cost"] == 30
    assert 0 < plan["extra_cache_bytes"] < sum(3 * (100 + task) for task in steps[0])
    execution = plan["execution_ownership"]
    assert sorted({task for row in execution for task in row}) == sorted(steps[0])


def test_selective_replication_scales_to_full_meta_task_inventory() -> None:
    tasks = tuple(range(71))
    owners = tuple(
        tuple(task for task in tasks if task % 6 == rank) for rank in range(6)
    )
    steps = []
    for step in range(100):
        owner = step % 6
        pool = owners[owner]
        start = (step // 6) % len(pool)
        group = tuple(pool[(start + offset) % len(pool)] for offset in range(6))
        steps.append({task: 5 + (task * 7 + step) % 23 for task in group})

    plan = selective_replication_plan(
        steps,
        base_task_owners=owners,
        cache_bytes={task: 1024 for task in tasks},
        extra_budget_bytes=8 * 1024,
    )

    assert 0 < len(plan["replicas"]) <= 8
    assert plan["extra_cache_bytes"] <= 8 * 1024
    assert plan["predicted_total_cost"] < plan["base_total_cost"]
    assert plan["predicted_total_cost"] >= plan["ideal_total_cost"]
    assert plan["replica_search"] == ("direct_move_gain_per_byte_then_exact_objective")
    assert sorted(
        {task for row in plan["execution_ownership"] for task in row}
    ) == list(tasks)


def test_task_batch_size_and_role_ratio_are_experiment_config_not_runtime_policy() -> (
    None
):
    cell = {
        "global_tasks_per_update": 4,
        "tasks_per_update_by_role": {"meta": 1, "target": 3},
    }
    counts = task_group_counts(cell, meta=(1, 8), target=(72, 73, 75, 93))
    groups = tuple(
        counted_task_group(((1, 8), (72, 73, 75, 93)), counts, step, seed=19)
        for step in range(8)
    )

    assert counts == (1, 3)
    assert all(len(group) == 4 for group in groups)
    assert all(len(set(group[:1])) == 1 for group in groups)
    assert all(len(set(group[1:])) == 3 for group in groups)
    assert {sum(task in group for group in groups) for task in (1, 8)} == {4}
    assert {sum(task in group for group in groups) for task in (72, 73, 75, 93)} == {6}

    target_only = task_group_counts(
        {
            "global_tasks_per_update": 2,
            "tasks_per_update_by_role": {"meta": 0, "target": 2},
        },
        meta=(),
        target=(72, 73, 75),
    )
    assert target_only == (0, 2)
    assert counted_task_group(((), (72, 73, 75)), target_only, 0, seed=3) == (
        73,
        75,
    )
    assert _selected_task_ids(
        {
            "task_split": {
                "gradient_meta": [],
                "gradient_target": [72, 73, 75],
                "true_task_held_meta": [],
                "true_task_held_target": [74],
            }
        }
    ) == (72, 73, 75, 74)


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
        task: SimpleNamespace(program_video_demos=(1, 3, 5, 7, 9)) for task in selected
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
    _, training_costs = _video_splits(runtime, selected, gradient_tasks=(1, 72))
    assert training_costs == {1: 56, 72: 56, 2: 75, 74: 75}


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
