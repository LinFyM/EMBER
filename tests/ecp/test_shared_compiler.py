from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import NativeTargetChunk, NativeVideoReadout
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.natural_program_data import NaturalProgramSample
from ember.ecp.shared_compiler import SharedCompilerVideo, SharedNativeFactorCompiler
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
        owners, program_width=width, event_slots=events, anchor_width=6
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

    loss = sum(value.square().mean() for value in observed.residual.a)
    loss = loss + sum(value.square().mean() for value in observed.residual.b)
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in compiler.anchor_scorer.parameters()
        if parameter.requires_grad
    )
    families = {family.value for family in TargetFamily}
    assert set(compiler.anchor_scorer.program_context) == families
    assert len(compiler.anchor_scorer.input_candidates) == len(owners)
    assert len(compiler.anchor_scorer.output_candidates) == len(owners)
    assert set(compiler.anchor_scorer.input_candidate_trunks) == families
    assert set(compiler.anchor_scorer.output_candidate_trunks) == families
    assert set(compiler.anchor_scorer.input_compatibility_heads) == families
    assert set(compiler.anchor_scorer.output_compatibility_heads) == families
    assert compiler.anchor_scorer.input_anchor_query["q"][-1].weight.grad is not None
    assert compiler.anchor_scorer.output_anchor_query["q"][-1].weight.grad is not None
    assert (
        compiler.anchor_scorer.input_candidates[0].direction.weight.grad
        is not None
    )
    assert (
        compiler.anchor_scorer.output_candidates[0].direction.weight.grad
        is not None
    )
    for scorer in (
        compiler.anchor_scorer.input_compatibility_heads["q"],
        compiler.anchor_scorer.output_compatibility_heads["q"],
    ):
        for parameter in (
            scorer.query_projection.weight,
            scorer.key_projection.weight,
        ):
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            assert bool(torch.count_nonzero(parameter.grad))
    assert compiler.anchor_scorer.query_owner_film.input_shift.grad is not None
    assert compiler.anchor_scorer.query_owner_film.output_shift[0].grad is not None
    assert bool(
        torch.count_nonzero(compiler.anchor_scorer.query_owner_film.input_shift.grad)
    )
    assert bool(
        torch.count_nonzero(compiler.anchor_scorer.query_owner_film.output_shift[0].grad)
    )
    assert compiler.anchor_scorer.group_gain["q"][-1].weight.grad is not None
    assert compiler.scale_head[-1].weight.grad is not None
    assert bool(torch.isfinite(observed.solve_metrics).all())
    assert observed.conditioning_metrics.shape == (1, 6)
    assert bool(torch.isfinite(observed.conditioning_metrics).all())
    assert float(observed.conditioning_metrics[..., 0].min()) > 0
    assert float(observed.conditioning_metrics[..., 1].min()) > 0
    assert float(observed.conditioning_metrics[..., 3].min()) > 0
    assert float(observed.conditioning_metrics[..., 4].min()) > 0
    assert float(observed.conditioning_metrics[..., 5].min()) > 0
    assert all(
        bool(((gain >= 0.0) & (gain <= 1.0)).all())
        for gain in observed.output_group_gains
    )
    names = set(dict(compiler.named_parameters()))
    assert not {"input_logits", "output_logits", "event_logits"} & names


def test_query_film_has_fixed_owner_and_output_group_ownership() -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4, anchor_width=6
    )
    compiler.to(torch.device("cpu"))
    state = compiler.anchor_scorer.program_state(_program(len(owners), 8, 4))
    input_before = compiler.anchor_scorer.input_queries(state).detach().clone()
    output_groups = compiler.anchor_scorer.query_owner_film.output_shift[0].shape[0]
    output_before = compiler.anchor_scorer.output_queries(
        state, target=0, groups=output_groups
    ).detach().clone()

    with torch.no_grad():
        compiler.anchor_scorer.query_owner_film.input_shift[0, 0, 0] = 0.5
        compiler.anchor_scorer.query_owner_film.output_shift[0][0, 0, 0] = 0.5

    input_after = compiler.anchor_scorer.input_queries(state).detach()
    output_after = compiler.anchor_scorer.output_queries(
        state, target=0, groups=output_groups
    ).detach()
    assert not torch.equal(input_before[0], input_after[0])
    torch.testing.assert_close(input_before[1:], input_after[1:])
    assert not torch.equal(output_before[0], output_after[0])
    torch.testing.assert_close(output_before[1:], output_after[1:])


def test_signed_queries_and_projected_bilinear_have_stable_initialization() -> None:
    compiler = SharedNativeFactorCompiler(
        _owners(), program_width=8, event_slots=4, anchor_width=6
    )
    scorer = compiler.anchor_scorer
    for side in ("input", "output"):
        heads = getattr(scorer, f"{side}_anchor_query")
        for head in heads.values():
            final = head[-1]
            width = scorer.feature_width
            torch.testing.assert_close(
                final.weight[width:], -final.weight[:width]
            )
            torch.testing.assert_close(final.bias[width:], -final.bias[:width])
        compatibilities = getattr(scorer, f"{side}_compatibility_heads")
        for compatibility in compatibilities.values():
            assert compatibility.query_projection.bias is None
            assert compatibility.key_projection.bias is None


def test_anchor_code_uses_video_program_fields() -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4, anchor_width=6
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
    first_state = compiler.anchor_scorer.program_state(first)
    changed_state = compiler.anchor_scorer.program_state(changed)
    torch.testing.assert_close(
        first_state.stable_rank_event, changed_state.stable_rank_event
    )
    torch.testing.assert_close(first_state.stable_rank, changed_state.stable_rank)
    assert not torch.allclose(
        compiler.anchor_scorer.input_queries(first_state),
        compiler.anchor_scorer.input_queries(changed_state),
    )
    assert not torch.allclose(first_state.event_weights, changed_state.event_weights)


def test_shared_compiler_video_set_is_permutation_invariant() -> None:
    owners = _owners()
    width = 8
    events = 4
    compiler = SharedNativeFactorCompiler(
        owners, program_width=width, event_slots=events, anchor_width=6
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
        _owners(), program_width=8, event_slots=4, anchor_width=6
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


def test_scale_and_group_gain_heads_do_not_backpropagate_into_selection_context() -> None:
    owners = _owners()
    compiler = SharedNativeFactorCompiler(
        owners, program_width=8, event_slots=4, anchor_width=6
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

    output = compiler(program, videos, s_ref=torch.ones(len(owners)))
    gain_gradients = torch.autograd.grad(
        sum(value.sum() for value in output.output_group_gains),
        (*selection, *scale_video),
        allow_unused=True,
    )
    assert all(
        gradient is None or not bool(torch.count_nonzero(gradient))
        for gradient in gain_gradients[: len(selection)]
    )
    assert any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in gain_gradients[len(selection) :]
    )
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
