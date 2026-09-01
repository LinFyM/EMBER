from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    NativeTargetChunk,
    NativeVideoReadout,
)
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
    _native_teacher_loss,
)


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
    with patch(
        "ember.ecp.bank_conditioning.frozen_condition_cache._candidate_mean",
        side_effect=AssertionError("new cache mean should load without fallback"),
    ):
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
        first.condition.program,
        first.condition.videos,
        s_ref=scale,
        bank_contexts=(compiler._bank_context(condition.videos[0]),),
    )
    right = compiler.forward_compact(
        second.condition.program,
        second.condition.videos,
        s_ref=scale,
        bank_contexts=(compiler._bank_context(condition.videos[0]),),
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
    assert not compiler.tangent_transport.scale_prior_ratio.requires_grad


def test_shared_compiler_video_set_is_permutation_invariant() -> None:
    torch.manual_seed(229)
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
    reference = videos[0]
    videos = tuple(
        replace(
            video,
            canonical_assignment=reference.canonical_assignment,
            frame_positions=reference.frame_positions,
            local_scene=reference.local_scene,
            local_process=reference.local_process,
            local_presence=reference.local_presence,
            local_tau=reference.local_tau,
            local_sigma=reference.local_sigma,
        )
        for video in videos
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
        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-4)


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
