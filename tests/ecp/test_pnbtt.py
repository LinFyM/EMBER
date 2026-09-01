from __future__ import annotations

import torch

from ember.ecp.bank_conditioning.key_value_replay import (
    differentiable_key_moments,
    safe_rms_normalize,
    signed_key_value_pool,
    whiten_queries,
)
from ember.ecp.bank_conditioning.tangent_parameterization import (
    PNBTT_SIDES,
    TaskLocalFreeTangentQuery,
)
from ember.ecp.bank_conditioning.tangent_transport import TangentTransportVideo
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.joint_program_primal.pnbtt_evaluation import (
    _same_frozen_training_git,
)
from ember.ecp.joint_program_primal.pnbtt_tangent_spectrum import (
    projection_spectrum,
    safe_rms_vjp,
    weighted_cross_covariance,
    whitened_cross_covariance,
)
from ember.ecp.native_factors import NativeTargetChunk, NativeVideoReadout
from ember.ecp.native_materialization import small_core_balanced_svd
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.shared_compiler import SharedCompilerVideo, SharedNativeFactorCompiler


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "q", TargetFamily.Q, 0, 4, 16),
        TargetOwner(1, "v", TargetFamily.V, 0, 4, 6),
        TargetOwner(2, "action_in", TargetFamily.ACTION_IN, None, 2, 4),
        TargetOwner(3, "action_out", TargetFamily.ACTION_OUT, None, 4, 2),
    )


def test_pnbtt_tangent_spectrum_matches_direct_covariance_and_autograd() -> None:
    generator = torch.Generator().manual_seed(101)
    scopes, events, candidates, value_width, key_width = 2, 3, 17, 7, 5
    values = torch.randn(scopes, candidates, value_width, generator=generator)
    keys = torch.randn(scopes, candidates, key_width, generator=generator)
    mass = torch.rand(scopes, events, candidates, generator=generator) + 0.1
    mass = mass / mass.sum(-1, keepdim=True)
    expected = []
    for scope in range(scopes):
        by_event = []
        for event in range(events):
            weights = mass[scope, event]
            value_mean = (weights[:, None] * values[scope]).sum(0)
            key_mean = (weights[:, None] * keys[scope]).sum(0)
            by_event.append(
                torch.einsum(
                    "n,nd,nm->dm",
                    weights,
                    values[scope] - value_mean,
                    keys[scope] - key_mean,
                )
            )
        expected.append(torch.stack(by_event))
    cross = weighted_cross_covariance(values, keys, mass)
    torch.testing.assert_close(cross, torch.stack(expected), rtol=2e-5, atol=2e-5)

    covariance = torch.einsum(
        "sen,senm,senp->semp",
        mass,
        keys[:, None] - torch.einsum("sen,snm->sem", mass, keys)[:, :, None],
        keys[:, None] - torch.einsum("sen,snm->sem", mass, keys)[:, :, None],
    )
    identity = torch.eye(key_width)
    cholesky = torch.linalg.cholesky(covariance + 1e-2 * identity)
    whitened = whitened_cross_covariance(cross, cholesky)
    torch.testing.assert_close(
        whitened,
        cross @ torch.linalg.inv(cholesky).transpose(-1, -2),
        rtol=2e-5,
        atol=2e-5,
    )

    value = torch.randn(4, 11, generator=generator, requires_grad=True)
    upstream = torch.randn(4, 11, generator=generator)
    (safe_rms_normalize(value, epsilon=1e-2) * upstream).sum().backward()
    torch.testing.assert_close(
        safe_rms_vjp(value.detach(), upstream, epsilon=1e-2),
        value.grad,
        rtol=2e-5,
        atol=2e-5,
    )


def test_pnbtt_projection_spectrum_reports_reachable_and_missing_gradients() -> None:
    matrix = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float32
    )
    gradients = torch.tensor(
        [[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float32
    )
    result = projection_spectrum(matrix, {"probe": gradients})
    retention = result["functional_gradient"]["probe"]["retention"]
    assert retention["maximum"] == 1.0
    assert retention["minimum"] == 0.0
    assert result["effective_rank_by_relative_singular_value"]["0.001"] == 2
    zero = projection_spectrum(torch.zeros(3, 2), {"probe": gradients})
    assert zero["spectral_energy_rank"]["99pct"] == 0
    assert zero["functional_gradient"]["probe"]["retention"]["maximum"] == 0.0


def test_frozen_training_git_allows_only_authority_tip_to_advance() -> None:
    training = {
        "branch": "",
        "commit": "1" * 40,
        "origin_main": "1" * 40,
        "upstream": None,
        "upstream_commit": None,
        "authority_ref": "origin/main",
        "authority_commit": "1" * 40,
        "authority_contains_commit": True,
        "dirty_paths": [],
    }
    current = {
        **training,
        "origin_main": "2" * 40,
        "authority_commit": "2" * 40,
    }
    assert _same_frozen_training_git(training, current)
    assert not _same_frozen_training_git(
        training, {**current, "commit": "3" * 40}
    )
    assert not _same_frozen_training_git(
        training, {**current, "authority_contains_commit": False}
    )


def _program(*, targets: int, width: int, events: int) -> NaturalProgram:
    generator = torch.Generator().manual_seed(71)
    return NaturalProgram(
        p_lang=torch.randn(targets, width, generator=generator),
        p_scene=torch.randn(targets, width, generator=generator),
        p_process=torch.randn(events, targets, width, generator=generator),
        rho=torch.linspace(0.2, 0.8, events),
        tau=torch.stack(
            (torch.linspace(0.0, 1.0, events), torch.full((events,), 0.1)), -1
        ),
        sigma=torch.rand(events, targets, width, generator=generator) + 0.1,
    )


def _video(
    owners: tuple[TargetOwner, ...],
    *,
    seed: int,
    width: int,
    events: int,
    zero_values: bool = False,
) -> SharedCompilerVideo:
    generator = torch.Generator().manual_seed(seed)
    frames = 3
    inputs = tuple(
        torch.randn(frames, 2, 50, owner.in_features, generator=generator)
        for owner in owners
    )
    outputs = tuple(
        torch.randn(frames, 2, 50, owner.out_features, generator=generator)
        for owner in owners
    )
    if zero_values:
        inputs = tuple(torch.zeros_like(value) for value in inputs)
        outputs = tuple(torch.zeros_like(value) for value in outputs)

    def chunks():
        yield NativeTargetChunk(start_frame=0, inputs=inputs, outputs=outputs)

    assignment = torch.full((frames, events), 0.1)
    assignment[torch.arange(frames), torch.arange(frames) % events] = 0.9
    assignment /= assignment.sum(-1, keepdim=True)
    return SharedCompilerVideo(
        native=NativeVideoReadout(
            frame_count=frames,
            process=torch.empty(0),
            state_posterior=torch.empty(0),
            final_outputs=tuple(value[-1] for value in outputs),
            chunks=chunks,
        ),
        canonical_assignment=assignment,
        frame_positions=torch.linspace(0.0, 1.0, frames),
        local_scene=torch.randn(len(owners), width, generator=generator),
        local_process=torch.randn(events, len(owners), width, generator=generator),
        local_presence=torch.linspace(0.2, 0.8, events),
        local_tau=torch.stack(
            (torch.linspace(0.0, 1.0, events), torch.full((events,), 0.1)), -1
        ),
        local_sigma=torch.rand(events, len(owners), width, generator=generator)
        + 0.1,
    )


def _compiler(owners: tuple[TargetOwner, ...]) -> SharedNativeFactorCompiler:
    return SharedNativeFactorCompiler(
        owners,
        program_width=8,
        event_slots=2,
        key_width=4,
        query_hidden_width=16,
        covariance_ridge=1e-3,
        native_rms_epsilon=1e-6,
        direction_epsilon=1e-3,
        query_epsilon=1e-3,
        score_epsilon=1e-3,
        replay_chunk_size=17,
    )


def test_pnbtt_signed_replay_is_exactly_zero_chunk_equivalent_and_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(19)
    scopes, candidates, width, events, ranks = 2, 31, 4, 3, 4
    keys = torch.randn(
        scopes, candidates, width, generator=generator, requires_grad=True
    )
    values = torch.randn(scopes, candidates, 7, generator=generator)
    mass = torch.rand(scopes, events, candidates, generator=generator) + 0.1
    queries = torch.randn(scopes, ranks, events, width, generator=generator)

    def pool(
        selected_keys: torch.Tensor,
        selected_values: torch.Tensor,
        selected_mass: torch.Tensor,
        *,
        chunk: int,
    ):
        moments = differentiable_key_moments(
            selected_keys, selected_mass, ridge=1e-3
        )
        return signed_key_value_pool(
            keys=selected_keys,
            values=selected_values,
            moments=moments,
            whitened_queries=whiten_queries(queries, moments),
            temperature=torch.ones(scopes),
            score_epsilon=1e-3,
            chunk_size=chunk,
        )

    expected = pool(keys, values, mass, chunk=candidates)
    chunked = pool(keys, values, mass, chunk=5)
    torch.testing.assert_close(chunked.direction, expected.direction)
    permutation = torch.randperm(candidates, generator=generator)
    permuted = pool(
        keys[:, permutation], values[:, permutation], mass[:, :, permutation], chunk=7
    )
    torch.testing.assert_close(permuted.direction, expected.direction, rtol=2e-6, atol=2e-6)
    zero = pool(keys, torch.zeros_like(values), mass, chunk=5)
    assert torch.equal(zero.direction, torch.zeros_like(zero.direction))

    expected.direction.square().mean().backward()
    assert keys.grad is not None
    assert bool(torch.isfinite(keys.grad).all())
    assert bool(torch.count_nonzero(keys.grad))


def test_pnbtt_full_transport_has_zero_step0_fixed_k_mass_and_bank_sensitivity() -> None:
    owners = _owners()
    compiler = _compiler(owners)
    program = _program(targets=len(owners), width=8, events=2)
    first = _video(owners, seed=11, width=8, events=2)
    second = _video(owners, seed=13, width=8, events=2)
    swapped = _video(owners, seed=17, width=8, events=2)
    zero = _video(owners, seed=23, width=8, events=2, zero_values=True)
    scale = torch.ones(len(owners))

    tangent_videos = tuple(
        TangentTransportVideo(
            native=compiler._compact_native(video),
            context=compiler._bank_context(video),
        )
        for video in (first, second)
    )
    candidates_per_video = first.native.frame_count * 2 * 50
    for scope in (
        compiler.tangent_transport._input_scope(tangent_videos, target=0),
        compiler.tangent_transport._output_scope(tangent_videos, target=0),
    ):
        torch.testing.assert_close(
            scope.event_mass[..., :candidates_per_video].sum(-1),
            torch.full_like(scope.event_mass[..., 0], 0.5),
        )
        torch.testing.assert_close(
            scope.event_mass[..., candidates_per_video:].sum(-1),
            torch.full_like(scope.event_mass[..., 0], 0.5),
        )

    with torch.no_grad():
        step0 = compiler(program, (first,), s_ref=scale)
        assert all(torch.equal(value, torch.zeros_like(value)) for value in step0.residual.b)
        zero_output = compiler(
            program,
            (zero,),
            s_ref=scale,
            query_override=torch.randn(len(owners), 4, 2, len(PNBTT_SIDES), 4),
        )
        assert all(
            torch.equal(value, torch.zeros_like(value))
            for value in (*zero_output.residual.a, *zero_output.residual.b)
        )

    query = torch.randn(
        len(owners), 4, 2, len(PNBTT_SIDES), 4, requires_grad=True
    )
    ordered = compiler(
        program, (first, second), s_ref=scale, query_override=query
    )
    reversed_videos = compiler(
        program, (second, first), s_ref=scale, query_override=query
    )
    torch.testing.assert_close(ordered.video_weights, torch.tensor([0.5, 0.5]))
    for left, right in zip(
        (*ordered.residual.a, *ordered.residual.b),
        (*reversed_videos.residual.a, *reversed_videos.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)

    changed = compiler(
        program, (first, swapped), s_ref=scale, query_override=query
    )
    assert not torch.allclose(ordered.solve_metrics, changed.solve_metrics)
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(
            (*ordered.residual.a, *ordered.residual.b),
            (*changed.residual.a, *changed.residual.b),
            strict=True,
        )
    )
    sum(
        value.square().mean()
        for value in (*ordered.residual.a, *ordered.residual.b)
    ).backward()
    assert query.grad is not None and bool(torch.isfinite(query.grad).all())
    key_parameters = tuple(compiler.tangent_transport.key_encoder.parameters())
    assert all(parameter.grad is not None for parameter in key_parameters)
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in key_parameters)
    names = tuple(name for name, _ in compiler.named_parameters())
    assert any("family_content_trunks" in name for name in names)
    assert any("family_side_heads" in name for name in names)
    assert any("target_side_residual" in name for name in names)
    assert not any(
        token in name
        for name in names
        for token in ("key_encoder.input_projection", "key_encoder.output_projection")
    )
    assert not any("task_lookup" in name or "video_reliability" in name for name in names)


def test_pnbtt_free_query_identity_and_final_canonical_update_are_exact() -> None:
    owners = _owners()
    query = TaskLocalFreeTangentQuery(
        (1, 93), owners, event_slots=2, key_width=4, query_epsilon=1e-3
    )
    correct = query(1)
    wrong = query(1)
    assert torch.equal(correct, wrong)
    assert query.raw_query.shape[0] == 2

    generator = torch.Generator().manual_seed(29)
    a = torch.randn(4, 13, generator=generator)
    b = torch.randn(17, 4, generator=generator)
    canonical_a, canonical_b = small_core_balanced_svd(a, b)
    torch.testing.assert_close(
        canonical_b @ canonical_a, b @ a, rtol=2e-5, atol=2e-5
    )
