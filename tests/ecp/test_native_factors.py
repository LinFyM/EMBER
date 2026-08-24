from __future__ import annotations

from dataclasses import replace

import torch

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.native_factors import (
    G1_PROBE_COUNT,
    G1_RESIDUAL_RANK,
    NativeFactorResidual,
    NativeOutputBankState,
    NativeTargetCapture,
    NativeTargetChunk,
    NativeVideoReadout,
    OnlineSoftmaxAccumulator,
    TaskLocalNativeFactorOracle,
)
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    extract_rank12_carrier,
    extract_rank4_residual,
    residual_lora_state,
    small_core_balanced_svd,
)
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
    validate_lora_state,
)


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "first", TargetFamily.ACTION_IN, None, 6, 7),
        TargetOwner(1, "second", TargetFamily.ACTION_OUT, None, 7, 5),
    )


class _TwoTargetPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(6, 7, bias=False)
        self.second = torch.nn.Linear(7, 5, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value))


def test_native_target_capture_is_scoped_and_preserves_probe_horizon_axes() -> None:
    policy = _TwoTargetPolicy().requires_grad_(False)
    value = torch.randn(6, 50, 6)

    with NativeTargetCapture(policy, _owners()) as capture:
        policy(value)
        chunk = capture.chunk(start_frame=4, frame_count=3, probe_count=2)

    assert chunk.start_frame == 4
    assert chunk.inputs[0].shape == (3, 2, 50, 6)
    assert chunk.outputs[0].shape == (3, 2, 50, 7)
    assert chunk.inputs[1].shape == (3, 2, 50, 7)
    assert chunk.outputs[1].shape == (3, 2, 50, 5)
    assert not policy.first._forward_hooks
    assert not policy.second._forward_hooks


def test_output_banks_keep_previous_first_and_final_across_chunks() -> None:
    raw = torch.randn(5, G1_PROBE_COUNT, 50, 3)
    chunked = NativeOutputBankState(final=raw[-1])
    observed = torch.cat(
        (
            chunked.build(raw[:2], start_frame=0),
            chunked.build(raw[2:], start_frame=2),
        )
    )
    reference = NativeOutputBankState(final=raw[-1]).build(raw, start_frame=0)
    other_video = NativeOutputBankState(final=raw[1]).build(raw[:2], start_frame=0)

    assert torch.equal(observed, reference)
    assert torch.count_nonzero(observed[0, :, :, 1]) == 0
    assert torch.count_nonzero(observed[0, :, :, 2]) == 0
    assert torch.count_nonzero(observed[-1, :, :, 3]) == 0
    assert torch.count_nonzero(other_video[0, :, :, 1]) == 0


def test_online_softmax_matches_nonchunked_signed_reference() -> None:
    generator = torch.Generator().manual_seed(12)
    logits = torch.randn(4, 2, 7, 2, 5, generator=generator)
    values = torch.randn(7, 2, 5, 6, generator=generator)
    accumulator = OnlineSoftmaxAccumulator(ranks=4, width=6, device=torch.device("cpu"))
    accumulator.add(logits[:, :, :3], values[:3])
    accumulator.add(logits[:, :, 3:], values[3:])
    flattened = logits.flatten(2)
    flat_values = values.reshape(-1, values.shape[-1])
    reference = torch.einsum(
        "rn,nd->rd", flattened[:, 0].softmax(-1), flat_values
    ) - torch.einsum("rn,nd->rd", flattened[:, 1].softmax(-1), flat_values)

    assert torch.allclose(accumulator.signed_mean(), reference, atol=2e-6, rtol=2e-6)


def _video(
    *,
    frames: int,
    owners: tuple[TargetOwner, ...],
    split: int,
    seed: int,
) -> NativeVideoReadout:
    generator = torch.Generator().manual_seed(seed)
    inputs = tuple(
        torch.randn(frames, 2, 50, owner.in_features, generator=generator)
        for owner in owners
    )
    outputs = tuple(
        torch.randn(frames, 2, 50, owner.out_features, generator=generator)
        for owner in owners
    )
    process = torch.randn(3, len(owners), 7, generator=generator)
    posterior = torch.rand(frames, 3, generator=generator).softmax(-1)

    def chunks():
        boundaries = (0, split, frames) if 0 < split < frames else (0, frames)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            yield NativeTargetChunk(
                start_frame=start,
                inputs=tuple(value[start:stop] for value in inputs),
                outputs=tuple(value[start:stop] for value in outputs),
            )

    return NativeVideoReadout(
        frame_count=frames,
        process=process,
        state_posterior=posterior,
        final_outputs=tuple(value[-1] for value in outputs),
        chunks=chunks,
    )


def test_task_local_free_code_is_chunk_equivalent_and_all_variables_receive_gradient() -> (
    None
):
    owners = _owners()
    oracle = TaskLocalNativeFactorOracle(
        owners,
        frame_counts=(4,),
        event_slots=3,
        program_width=7,
        initialization_seed=9,
    )
    split = _video(frames=4, owners=owners, split=2, seed=19)
    whole = replace(
        split, chunks=_video(frames=4, owners=owners, split=0, seed=19).chunks
    )
    s_ref = torch.tensor([0.2, 0.4])
    chunked = oracle((split,), s_ref=s_ref)
    reference = oracle((whole,), s_ref=s_ref)

    for left, right in zip(
        chunked.a + chunked.b, reference.a + reference.b, strict=True
    ):
        assert torch.allclose(left, right, atol=3e-4, rtol=3e-4)
    loss = sum(
        (value * torch.linspace(0.5, 1.5, value.numel()).reshape_as(value)).mean()
        for value in chunked.a + chunked.b
    )
    loss.backward()
    for parameter in oracle.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)


def test_q_output_pooling_uses_all_eight_native_attention_head_groups() -> None:
    owners = (TargetOwner(0, "q", TargetFamily.Q, 0, 6, 16),)
    oracle = TaskLocalNativeFactorOracle(
        owners,
        frame_counts=(4,),
        event_slots=3,
        program_width=7,
        initialization_seed=11,
    )
    video = _video(frames=4, owners=owners, split=2, seed=23)
    whole = replace(
        video, chunks=_video(frames=4, owners=owners, split=0, seed=23).chunks
    )

    residual = oracle((video,), s_ref=torch.tensor([0.3]))
    reference = oracle((whole,), s_ref=torch.tensor([0.3]))
    residual.b[0].square().sum().backward()

    assert oracle.output_group_counts.tolist() == [8]
    assert oracle.output_group_offsets.tolist() == [0, 8]
    assert oracle.output_logits.shape[0] == 8
    assert residual.b[0].shape == (G1_RESIDUAL_RANK, 16)
    assert torch.allclose(residual.b[0], reference.b[0], atol=3e-4, rtol=3e-4)
    gradient = oracle.output_logits.grad
    assert gradient is not None
    per_group_nonzero = torch.count_nonzero(
        gradient, dim=tuple(range(1, gradient.ndim))
    )
    assert torch.all(per_group_nonzero > 0)


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("first", 6, 7), LoRATarget("second", 7, 5)),
        rank=16,
        alpha=16,
        dropout=0.0,
        identity_seed=3,
    )


def test_small_core_svd_and_rank12_plus4_preserve_one_complete_update() -> None:
    generator = torch.Generator().manual_seed(21)
    residual = NativeFactorResidual(
        a=(
            torch.randn(4, 6, generator=generator),
            torch.randn(4, 7, generator=generator),
        ),
        b=(
            torch.randn(4, 7, generator=generator),
            torch.randn(4, 5, generator=generator),
        ),
        scales=torch.ones(2, 4),
    )
    contract = _contract()
    rank4_contract = replace(contract, rank=4, alpha=4)
    raw = residual_lora_state(residual, rank4_contract, canonicalize=False)
    canonical = residual_lora_state(residual, rank4_contract, canonicalize=True)
    carrier: dict[str, torch.Tensor] = {}
    for target in contract.targets:
        carrier[target.name + LORA_A_SUFFIX] = torch.randn(
            16, target.in_features, generator=generator
        )
        carrier_b = torch.randn(target.out_features, 16, generator=generator)
        carrier_b[:, 12:] = 0
        carrier[target.name + LORA_B_SUFFIX] = carrier_b
    rank12 = extract_rank12_carrier(carrier, contract)
    complete = compose_rank12_plus_rank4(
        carrier_state=rank12,
        residual_state=canonical,
        rank16_contract=contract,
    )

    validate_lora_state(complete, contract)
    extracted = extract_rank4_residual(complete, contract, carrier_state=carrier)
    for target in contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        raw_update = raw[b_name] @ raw[a_name]
        canonical_update = canonical[b_name] @ canonical[a_name]
        complete_update = complete[b_name] @ complete[a_name]
        carrier_update = rank12[b_name] @ rank12[a_name]
        assert torch.allclose(raw_update, canonical_update, atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            complete_update, carrier_update + raw_update, atol=3e-5, rtol=3e-5
        )
        assert torch.equal(extracted[a_name], canonical[a_name])
        assert torch.equal(extracted[b_name], canonical[b_name])


def test_small_core_balanced_svd_has_rank_four_shapes() -> None:
    a = torch.randn(G1_RESIDUAL_RANK, 11)
    b = torch.randn(13, G1_RESIDUAL_RANK)

    canonical_a, canonical_b = small_core_balanced_svd(a, b)

    assert canonical_a.shape == a.shape
    assert canonical_b.shape == b.shape
