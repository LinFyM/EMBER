from __future__ import annotations

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import NativeTargetChunk, NativeVideoReadout
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.shared_compiler import SharedCompilerVideo, SharedNativeFactorCompiler


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
    owners = _owners()
    width = 8
    events = 4
    compiler = SharedNativeFactorCompiler(
        owners, program_width=width, event_slots=events, key_width=6
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
    assert compiler.input_query.weight.grad is not None
    assert compiler.output_query.weight.grad is not None
    assert compiler.input_keys["4"].content.weight.grad is not None
    assert compiler.output_keys["2"].content.weight.grad is not None
    assert compiler.scale_head.weight.grad is not None
    names = set(dict(compiler.named_parameters()))
    assert not {"input_logits", "output_logits", "event_logits"} & names


def test_shared_compiler_video_set_is_permutation_invariant() -> None:
    owners = _owners()
    width = 8
    events = 4
    compiler = SharedNativeFactorCompiler(
        owners, program_width=width, event_slots=events, key_width=6
    )
    torch.nn.init.normal_(compiler.video_reliability[-1].weight, std=0.03)
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
    assert float(forward.video_weights.detach().max()) < 0.625
    for left, right in zip(
        (*forward.residual.a, *forward.residual.b),
        (*reverse.residual.a, *reverse.residual.b),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
