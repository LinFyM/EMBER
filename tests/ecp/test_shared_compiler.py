from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import NativeTargetChunk, NativeVideoReadout
from ember.ecp.natural_program import FrozenProgramEvidence, NaturalProgram
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler import SharedCompilerVideo, SharedNativeFactorCompiler
from ember.ecp.shared_compiler_data import SharedCompilerCondition
from ember.ecp.bank_conditioning.frozen_condition_cache import (
    FrozenMappingConditionCache,
    frozen_condition_cache_authority,
)
from ember.ecp.policy_effects import ExecutionPolicyPrefix, PolicyEffectResponse
from ember.ecp.shared_compiler_effects import (
    SharedCompilerEffectBank,
    member_effect_losses,
)
from ember.ecp.shared_compiler_train_step import (
    _clip_parameter_groups,
    _native_teacher_loss,
)
from ember.ecp.shared_compiler_training import _trainable_groups


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "q", TargetFamily.Q, 0, 4, 16),
        TargetOwner(1, "v", TargetFamily.V, 0, 4, 6),
        TargetOwner(2, "action_in", TargetFamily.ACTION_IN, None, 2, 4),
        TargetOwner(3, "action_out", TargetFamily.ACTION_OUT, None, 4, 2),
    )


def _program(owners: int, width: int, events: int) -> NaturalProgram:
    generator = torch.Generator().manual_seed(13)
    return NaturalProgram(
        p_lang=torch.randn(owners, width, generator=generator),
        p_scene=torch.randn(owners, width, generator=generator),
        p_process=torch.randn(events, owners, width, generator=generator),
        rho=torch.linspace(0.2, 0.9, events),
        tau=torch.stack(
            (torch.linspace(0.0, 1.0, events), torch.full((events,), 0.1)), -1
        ),
        sigma=torch.rand(events, owners, width, generator=generator) + 0.1,
    )


def _video(
    owners: tuple[TargetOwner, ...],
    *,
    seed: int,
    chunks: tuple[int, ...],
    width: int,
    events: int,
) -> SharedCompilerVideo:
    generator = torch.Generator().manual_seed(seed)
    frames = sum(chunks)
    inputs = tuple(
        torch.randn(frames, 2, 50, owner.in_features, generator=generator)
        for owner in owners
    )
    outputs = tuple(
        torch.randn(frames, 2, 50, owner.out_features, generator=generator)
        for owner in owners
    )

    def stream():
        start = 0
        for count in chunks:
            stop = start + count
            yield NativeTargetChunk(
                start_frame=start,
                inputs=tuple(value[start:stop] for value in inputs),
                outputs=tuple(value[start:stop] for value in outputs),
            )
            start = stop

    assignment = torch.full((frames, events), 0.02)
    assignment[torch.arange(frames), torch.arange(frames) % events] = 0.88
    assignment = assignment / assignment.sum(-1, keepdim=True)
    native = NativeVideoReadout(
        frame_count=frames,
        process=torch.empty(0),
        state_posterior=torch.empty(0),
        final_outputs=tuple(value[-1] for value in outputs),
        chunks=stream,
    )
    return SharedCompilerVideo(
        native=native,
        canonical_assignment=assignment,
        frame_positions=torch.linspace(0.0, 1.0, frames),
        local_scene=torch.randn(len(owners), width, generator=generator),
        local_process=torch.randn(
            events, len(owners), width, generator=generator
        ),
        local_presence=torch.linspace(0.2, 0.9, events),
        local_tau=torch.stack(
            (torch.linspace(0.0, 1.0, events), torch.full((events,), 0.1)), -1
        ),
        local_sigma=torch.rand(
            events, len(owners), width, generator=generator
        )
        + 0.1,
    )


def test_shared_compiler_is_chunk_equivalent_and_has_gradients() -> None:
    torch.manual_seed(5)
    owners = _owners()
    width = 8
    events = 4
    compiler = SharedNativeFactorCompiler(
        owners, program_width=width, event_slots=events
    )
    program = _program(len(owners), width, events)
    one_chunk = _video(
        owners, seed=17, chunks=(5,), width=width, events=events
    )
    two_chunks = _video(
        owners, seed=17, chunks=(2, 3), width=width, events=events
    )
    scale = torch.ones(len(owners))
    expected = compiler(program, (one_chunk,), s_ref=scale)
    observed = compiler(program, (two_chunks,), s_ref=scale)

    state = compiler.primal_scorer.program_state(program)
    input_primals = compiler.primal_scorer.input_primals(state)
    output_primals = compiler.primal_scorer.output_primals(state)
    prepared = compiler.bank_operator.prepare(one_chunk)
    replayed = compiler.bank_operator.apply(
        prepared, input_primals, output_primals
    )
    direct = compiler.bank_operator(
        one_chunk, input_primals, output_primals
    )
    materialized = compiler.bank_operator.apply_materialized(
        compiler.bank_operator.materialize(prepared),
        input_primals,
        output_primals,
    )
    materialized_output = compiler.forward_materialized(
        program,
        (compiler.bank_operator.materialize(prepared),),
        s_ref=scale,
    )
    compact_output = compiler.forward_compact(
        program,
        (compiler.bank_operator.compact(prepared),),
        s_ref=scale,
    )
    for left, right in zip(
        (*replayed.input_values, *replayed.output_values),
        (*direct.input_values, *direct.output_values),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    for left, right in zip(
        (*materialized.input_values, *materialized.output_values),
        (*direct.input_values, *direct.output_values),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
    for left, right in zip(
        (*compact_output.residual.a, *compact_output.residual.b),
        (*expected.residual.a, *expected.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(replayed.solve_metrics, direct.solve_metrics)
    for left, right in zip(
        (*materialized_output.residual.a, *materialized_output.residual.b),
        (*expected.residual.a, *expected.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)

    assert torch.equal(expected.video_weights, torch.ones(1))
    for target, (a_direction, b_direction) in enumerate(
        zip(observed.input_directions, observed.output_directions, strict=True)
    ):
        torch.testing.assert_close(a_direction, observed.residual.a[target])
        torch.testing.assert_close(
            b_direction * observed.residual.scales[target, :, None],
            observed.residual.b[target],
        )
    for left, right in zip(
        (*expected.residual.a, *expected.residual.b),
        (*observed.residual.a, *observed.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
    for measure in observed.frame_measures:
        torch.testing.assert_close(
            measure.sum(-1), torch.ones_like(measure[..., 0])
        )

    generator = torch.Generator().manual_seed(101)
    loss = sum(
        (value * torch.randn(value.shape, generator=generator)).mean()
        for value in observed.residual.a
    )
    loss = loss + sum(
        (value * torch.randn(value.shape, generator=generator)).mean()
        for value in observed.residual.b
    )
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in compiler.primal_scorer.parameters()
        if parameter.requires_grad
    )
    families = {family.value for family in TargetFamily}
    assert set(compiler.primal_scorer.program_context) == families
    assert set(compiler.primal_scorer.input_trunk) == families
    assert set(compiler.primal_scorer.output_trunk) == families
    assert len(compiler.primal_scorer.input_primal_heads) == len(owners)
    assert len(compiler.primal_scorer.output_primal_heads) == len(owners)
    assert compiler.primal_scorer.input_primal_heads[0].weight.grad is not None
    assert compiler.primal_scorer.output_primal_heads[0].weight.grad is not None
    assert bool(torch.count_nonzero(compiler.primal_scorer.owner_embedding.grad))
    assert bool(torch.count_nonzero(compiler.primal_scorer.group_embedding.grad))
    assert compiler.scale_head[-1].weight.grad is not None
    assert bool(torch.isfinite(observed.solve_metrics).all())
    assert observed.conditioning_metrics.shape == (1, 6)
    assert bool(torch.isfinite(observed.conditioning_metrics).all())
    assert float(observed.conditioning_metrics[..., 0].min()) > 0
    assert float(observed.conditioning_metrics[..., 1].min()) > 0
    assert float(observed.conditioning_metrics[..., 3].min()) > 0
    assert float(observed.conditioning_metrics[..., 4].min()) > 0
    assert float(observed.conditioning_metrics[..., 5].min()) > 0
    assert all(torch.equal(gain, torch.ones_like(gain)) for gain in observed.output_group_gains)
    names = set(dict(compiler.named_parameters()))
    assert not any(
        token in name
        for name in names
        for token in ("candidate", "compatibility", "functional_polar", "task_lookup")
    )


def test_frozen_mapping_condition_cache_round_trips_exact_bank(tmp_path) -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4
    )
    program = _program(len(owners), 8, 4)
    condition = SharedCompilerCondition(
        program=program,
        videos=(
            _video(owners, seed=59, chunks=(2, 3), width=8, events=4),
        ),
        metrics={"K": 1, "video_demos": [3], "sampled_frames": [5]},
    )
    authority = frozen_condition_cache_authority(
        config_schema="test",
        config_bytes=17,
        source_checkpoint=tmp_path / "source",
        g2_program_checkpoint=tmp_path / "g2",
        native_observer_checkpoint=tmp_path / "observer",
        frame_stride=5,
        owners=owners,
    )
    cache = FrozenMappingConditionCache(
        tmp_path / "cache",
        owners=owners,
        operator=compiler.bank_operator,
        authority=authority,
    )
    calls = 0

    def build() -> SharedCompilerCondition:
        nonlocal calls
        calls += 1
        return condition

    first = cache.get_or_build(
        authority_id=7,
        video_demo=3,
        device=torch.device("cpu"),
        builder=build,
    )
    second = cache.get_or_build(
        authority_id=7,
        video_demo=3,
        device=torch.device("cpu"),
        builder=build,
    )
    assert calls == 1
    assert not first.hit and second.hit
    assert first.file_bytes == second.file_bytes > 0
    assert first.condition.metrics == second.condition.metrics
    assert all(
        first.condition.metrics[key] == value
        for key, value in condition.metrics.items()
    )
    assert first.condition.metrics["native_replay"] == "ephemeral_frozen_XY_cache"
    ephemeral_first = cache.get_or_build(
        authority_id=8,
        video_demo=4,
        device=torch.device("cpu"),
        builder=build,
        retain=False,
    )
    ephemeral_second = cache.get_or_build(
        authority_id=8,
        video_demo=4,
        device=torch.device("cpu"),
        builder=build,
        retain=False,
    )
    assert calls == 3
    assert not ephemeral_first.hit and not ephemeral_second.hit
    assert ephemeral_first.file_bytes == ephemeral_second.file_bytes == 0
    assert not (cache.root / "task_008_video_004.safetensors").exists()
    scale = torch.ones(len(owners))
    left = compiler.forward_compact(
        first.condition.program, first.condition.videos, s_ref=scale
    )
    right = compiler.forward_compact(
        second.condition.program, second.condition.videos, s_ref=scale
    )
    for first_value, second_value in zip(
        (*left.residual.a, *left.residual.b),
        (*right.residual.a, *right.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(first_value, second_value)


def test_joint_cache_stores_frozen_evidence_but_not_program(tmp_path) -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4
    )
    tensors = {
        "language_embeddings": torch.randn(1, 3, 8),
        "language_mask": torch.ones(1, 3, dtype=torch.bool),
        "patch_states": torch.randn(1, 5, 2, 8),
        "frame_mask": torch.ones(1, 5, dtype=torch.bool),
        "process": torch.randn(2, 1, 4, 4, 8),
        "uncertainty": torch.rand(2, 1, 4, 4, 8),
        "presence": torch.rand(2, 1, 4),
        "state_posterior": torch.rand(2, 1, 5, 4),
        "frame_indices": torch.arange(5),
        "raw_frame_counts": torch.tensor([21]),
        "video_offsets": torch.tensor([0, 5]),
        "video_set_offsets": torch.tensor([0, 1]),
        "frame_condition_ids": torch.zeros(5, dtype=torch.long),
    }
    evidence = FrozenProgramEvidence(**tensors)
    condition = SharedCompilerCondition(
        program=_program(len(owners), 8, 4),
        videos=(_video(owners, seed=61, chunks=(2, 3), width=8, events=4),),
        metrics={"K": 1},
        evidence=evidence,
    )
    cache = FrozenMappingConditionCache(
        tmp_path / "joint_cache",
        owners=owners,
        operator=compiler.bank_operator,
        authority=frozen_condition_cache_authority(
            config_schema="joint-test",
            config_bytes=19,
            source_checkpoint=tmp_path / "source",
            g2_program_checkpoint=tmp_path / "g2",
            native_observer_checkpoint=tmp_path / "observer",
            frame_stride=5,
            owners=owners,
        ),
        cache_program=False,
    )
    first = cache.get_or_build(
        authority_id=7,
        video_demo=3,
        device=torch.device("cpu"),
        builder=lambda: condition,
    )
    second = cache.get_or_build(
        authority_id=7,
        video_demo=3,
        device=torch.device("cpu"),
        builder=lambda: condition,
    )
    assert first.condition.program is None
    assert second.condition.program is None
    assert first.condition.evidence is not None
    assert second.condition.evidence is not None
    for name, expected in tensors.items():
        torch.testing.assert_close(
            getattr(second.condition.evidence, name), expected
        )


def test_primal_heads_have_fixed_target_and_output_group_ownership() -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4
    )
    compiler.to(torch.device("cpu"))
    scorer = compiler.primal_scorer
    state = scorer.program_state(_program(len(owners), 8, 4))
    input_before = tuple(value.detach().clone() for value in scorer.input_primals(state))
    output_before = tuple(value.detach().clone() for value in scorer.output_primals(state))

    with torch.no_grad():
        scorer.input_primal_heads[0].weight.add_(0.5)
        scorer.group_embedding[0].add_(0.5)

    input_after = scorer.input_primals(state)
    output_after = scorer.output_primals(state)
    assert not torch.equal(input_before[0], input_after[0])
    for left, right in zip(input_before[1:], input_after[1:], strict=True):
        torch.testing.assert_close(left, right)
    assert not torch.equal(output_before[0], output_after[0])
    assert not torch.equal(output_before[1][0], output_after[1][0])


def test_frozen_scale_prior_is_task_agnostic_and_exact_before_f4() -> None:
    owners = _owners()
    prior = torch.tensor(
        [
            [0.05, 0.10, 0.15, 0.20],
            [0.12, 0.18, 0.24, 0.30],
            [0.08, 0.16, 0.32, 0.40],
            [0.07, 0.14, 0.21, 0.28],
        ]
    )
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=8,
        event_slots=4,
        scale_prior_ratio=prior,
    )
    scale = torch.tensor([2.0, 3.0, 4.0, 5.0])
    video = _video(owners, seed=71, chunks=(2, 2), width=8, events=4)
    first = compiler(_program(len(owners), 8, 4), (video,), s_ref=scale)
    second = compiler(_program(len(owners), 8, 4), (video,), s_ref=scale)
    expected = scale[:, None] * prior
    torch.testing.assert_close(first.residual.scales, expected)
    torch.testing.assert_close(second.residual.scales, expected)
    assert not compiler.scale_prior_ratio.requires_grad


def test_anchor_code_uses_video_program_fields() -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4
    )
    first = _program(len(owners), 8, 4)
    changed = _program(len(owners), 8, 4)
    generator = torch.Generator().manual_seed(97)
    changed = NaturalProgram(
        p_lang=first.p_lang.clone(),
        p_scene=torch.randn(first.p_scene.shape, generator=generator),
        p_process=torch.randn(first.p_process.shape, generator=generator),
        rho=torch.rand(first.rho.shape, generator=generator) + 0.1,
        tau=torch.rand(first.tau.shape, generator=generator),
        sigma=torch.rand(first.sigma.shape, generator=generator) + 0.1,
    )
    first_state = compiler.primal_scorer.program_state(first)
    changed_state = compiler.primal_scorer.program_state(changed)
    torch.testing.assert_close(
        first_state.stable_rank, changed_state.stable_rank
    )
    assert not torch.allclose(first_state.rank, changed_state.rank)
    assert not torch.allclose(first_state.event_weights, changed_state.event_weights)


def test_shared_compiler_video_set_is_permutation_invariant() -> None:
    owners = _owners()
    width = 8
    events = 4
    compiler = SharedNativeFactorCompiler(
        owners, program_width=width, event_slots=events
    )
    program = _program(len(owners), width, events)
    videos = tuple(
        _video(owners, seed=seed, chunks=(2, 2), width=width, events=events)
        for seed in (31, 37, 41, 43)
    )
    scale = torch.ones(len(owners))
    forward = compiler(program, videos, s_ref=scale)
    reverse = compiler(program, tuple(reversed(videos)), s_ref=scale)

    torch.testing.assert_close(
        forward.video_weights, reverse.video_weights.flip(0), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(forward.video_weights, torch.full((4,), 0.25))
    for left, right in zip(
        (*forward.residual.a, *forward.residual.b),
        (*reverse.residual.a, *reverse.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)


def test_scale_gradient_cannot_consume_selection_clip_budget() -> None:
    compiler = SharedNativeFactorCompiler(
        _owners(), program_width=8, event_slots=4
    )
    selection, scale_video = _trainable_groups(compiler)
    assert {id(value) for value in selection}.isdisjoint(
        {id(value) for value in scale_video}
    )
    assert len(selection) + len(scale_video) == len(tuple(compiler.parameters()))
    for parameter in selection:
        parameter.grad = torch.full_like(parameter, 1e-4)
    for parameter in scale_video:
        parameter.grad = torch.full_like(parameter, 1e4)
    selection_before = tuple(value.grad.clone() for value in selection)

    norms = _clip_parameter_groups(
        selection_parameters=selection,
        scale_video_parameters=scale_video,
        optimizer={
            "selection_gradient_clip_norm": 1.0,
            "scale_video_gradient_clip_norm": 1.0,
        },
    )

    assert norms["selection"] < 1.0
    assert norms["scale_video"] > 1.0
    for before, parameter in zip(selection_before, selection, strict=True):
        torch.testing.assert_close(parameter.grad, before)
    assert torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in scale_video])
    ) <= 1.00001


def test_scale_head_does_not_backpropagate_into_primal_context() -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4
    )
    selection, scale_video = _trainable_groups(compiler)
    program = _program(len(owners), 8, 4)
    videos = tuple(
        _video(owners, seed=seed, chunks=(2, 2), width=8, events=4)
        for seed in (71, 73)
    )

    output = compiler(program, videos, s_ref=torch.ones(len(owners)))
    scale_gradients = torch.autograd.grad(
        output.residual.scales.sum(),
        (*selection, *scale_video),
        allow_unused=True,
    )
    selection_scale = scale_gradients[: len(selection)]
    scale_head = [
        gradient
        for parameter, gradient in zip(
            scale_video, scale_gradients[len(selection) :], strict=True
        )
        if "scale_head" in next(
            name
            for name, candidate in compiler.named_parameters()
            if candidate is parameter
        )
    ]
    assert all(
        gradient is None or not bool(torch.count_nonzero(gradient))
        for gradient in selection_scale
    )
    assert scale_head and any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in scale_head
    )

    assert all(torch.equal(value, torch.ones_like(value)) for value in output.output_group_gains)
    torch.testing.assert_close(output.video_weights, torch.full((2,), 0.5))


def test_multivideo_training_condition_cannot_read_native_teacher_tensors() -> None:
    class Store:
        tensor_reads = 0

        def lookup_members(self, **kwargs):
            assert kwargs["k"] in (2, 4)
            assert kwargs["video_demo"] is None
            return None

    runtime = SimpleNamespace(
        native_teachers=Store(),
        config={"optimization": {"loss_weights": {}}},
        owners=_owners(),
    )
    output = SimpleNamespace(
        residual=SimpleNamespace(scales=torch.ones(len(_owners()), 4)),
        input_directions=(),
        output_directions=(),
    )
    bank = SimpleNamespace(
        member_names=("member",),
        family_weights=torch.full((1, len(_owners())), 0.25),
    )
    for k in (2, 4):
        sample = NaturalProgramSample(
            video_demos=tuple(range(k)),
            action_demos=(),
            k=k,
            robustness_view="test",
        )
        loss, metrics = _native_teacher_loss(
            runtime,
            task_id=1,
            sample=sample,
            output=output,
            bank=bank,
            responsibilities=torch.ones(1),
        )
        assert float(loss.total) == 0.0
        assert metrics["native_teacher_member_lookups"] == 0
        assert metrics["native_teacher_tensor_reads"] == 0
    assert runtime.native_teachers.tensor_reads == 0


def test_shared_member_effect_uses_one_global_member_responsibility() -> None:
    states = 4
    carrier = PolicyEffectResponse(
        owner=torch.zeros(states, 38, 4, 128),
        flow=torch.zeros(states, 10, 50, 32),
        action=torch.zeros(states, 10, 50, 7),
    )
    members = PolicyEffectResponse(
        owner=torch.ones(1, states, 38, 4, 128),
        flow=torch.ones(1, states, 10, 50, 32),
        action=torch.ones(1, states, 10, 50, 7),
    )
    bank = SharedCompilerEffectBank(
        prefix=ExecutionPolicyPrefix(torch.empty(states, 0, 0), torch.empty(states, 0)),
        suffix_noise=torch.empty(states, 50, 32),
        validity=torch.ones(1, states, dtype=torch.bool),
        trajectory_ids=torch.zeros(states, dtype=torch.long),
        carrier=carrier,
        members=members,
        reliability=torch.ones(1),
        family_weights=torch.full((1, 38), 1.0 / 38.0),
        member_names=("member",),
        projections=({},),
        anchors=tuple({} for _ in range(states)),
    )
    value = torch.tensor(0.25, requires_grad=True)
    candidate = PolicyEffectResponse(
        owner=value * torch.ones_like(carrier.owner),
        flow=value * torch.ones_like(carrier.flow),
        action=value * torch.ones_like(carrier.action),
    )

    loss = member_effect_losses(candidate, bank)
    combined = (
        loss.global_effect
        + loss.family_functional
        + loss.member_flow_response
        + loss.action_response
    )
    combined.backward()

    torch.testing.assert_close(loss.responsibilities, torch.ones(1))
    torch.testing.assert_close(loss.global_effect, loss.member_totals[0])
    assert value.grad is not None and float(value.grad) < 0.0
