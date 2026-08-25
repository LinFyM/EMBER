from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.shared_compiler_native_teacher import (
    NativeTeacherAuthorityError,
    NativeTeacherStore,
    factor_subspace_loss,
    low_rank_update_direction_loss,
    native_teacher_from_lora_state,
    native_teacher_supervision_loss,
    publish_native_teacher_root,
    small_core_singular_values,
    write_native_teacher_task_shard,
)
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
)
from ember.pi05_source_checkpoint import read_json


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(
            LoRATarget("q", 7, 8),
            LoRATarget("v", 6, 7),
            LoRATarget("action_in", 5, 6),
            LoRATarget("action_out", 8, 5),
        ),
        rank=4,
        alpha=4,
        dropout=0.0,
        identity_seed=3,
    )


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "q", TargetFamily.Q, 0, 7, 8),
        TargetOwner(1, "v", TargetFamily.V, 0, 6, 7),
        TargetOwner(2, "action_in", TargetFamily.ACTION_IN, None, 5, 6),
        TargetOwner(3, "action_out", TargetFamily.ACTION_OUT, None, 8, 5),
    )


def _directions(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    owners = _owners()
    a = tuple(
        torch.randn(4, owner.in_features, generator=generator) for owner in owners
    )
    b = tuple(
        torch.randn(4, owner.out_features, generator=generator) for owner in owners
    )
    scales = torch.rand(len(owners), 4, generator=generator) + 0.25
    return a, b, scales


def _teacher(*, video_demo: int = 3, member: str = "member_a"):
    contract = _contract()
    a, b, scales = _directions()
    state = {}
    for index, target in enumerate(contract.targets):
        state[target.name + LORA_A_SUFFIX] = a[index]
        state[target.name + LORA_B_SUFFIX] = (
            b[index] * scales[index, :, None]
        ).transpose(0, 1)
    teacher = native_teacher_from_lora_state(
        authority_id=7,
        video_demo=video_demo,
        member_name=member,
        state=state,
        scales=scales,
        contract=contract,
        provenance={"source": "unit"},
    )
    return teacher, a, b, scales


def _paired_transform(
    a: torch.Tensor, b: torch.Tensor, transform: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return transform @ a, b @ torch.linalg.inv(transform)


def test_low_rank_losses_ignore_rank_gauge() -> None:
    generator = torch.Generator().manual_seed(23)
    a = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    b = torch.randn(10, 4, generator=generator, dtype=torch.float64)
    permutation = torch.eye(4, dtype=torch.float64)[torch.tensor([2, 0, 3, 1])]
    common_signs = torch.diag(
        torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    )
    basis = torch.tensor(
        [
            [1.2, 0.2, -0.1, 0.0],
            [0.1, 0.9, 0.3, 0.1],
            [0.0, -0.2, 1.1, 0.2],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=torch.float64,
    )

    for transform in (permutation, common_signs, basis):
        transformed_a, transformed_b = _paired_transform(a, b, transform)
        torch.testing.assert_close(
            factor_subspace_loss(transformed_a, a), torch.zeros(())
        )
        torch.testing.assert_close(
            factor_subspace_loss(transformed_b.transpose(0, 1), b.transpose(0, 1)),
            torch.zeros(()),
        )
        torch.testing.assert_close(
            low_rank_update_direction_loss(transformed_a, transformed_b, a, b),
            torch.zeros(()),
            atol=2e-7,
            rtol=0,
        )
        torch.testing.assert_close(
            small_core_singular_values(transformed_a, transformed_b),
            small_core_singular_values(a, b),
            atol=2e-6,
            rtol=2e-6,
        )


def test_native_teacher_loss_has_disjoint_gradient_owners() -> None:
    teacher, teacher_a, teacher_b, teacher_scales = _teacher()
    student_a = tuple(
        (value + 0.1 * torch.randn_like(value)).requires_grad_()
        for value in teacher_a
    )
    student_b = tuple(
        (value + 0.1 * torch.randn_like(value)).requires_grad_()
        for value in teacher_b
    )
    student_scales = (teacher_scales * 0.8).requires_grad_()
    losses = native_teacher_supervision_loss(
        student_a_directions=student_a,
        student_b_directions=student_b,
        student_scales=student_scales,
        teachers=(teacher,),
        owners=_owners(),
    )

    selection_grads = torch.autograd.grad(
        losses.selection,
        (*student_a, *student_b, student_scales),
        retain_graph=True,
        allow_unused=True,
    )
    assert all(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in selection_grads[:-1]
    )
    assert selection_grads[-1] is None

    spectrum_grads = torch.autograd.grad(
        losses.spectrum_scale,
        (*student_a, *student_b, student_scales),
        allow_unused=True,
    )
    assert all(gradient is None for gradient in spectrum_grads[:-1])
    assert spectrum_grads[-1] is not None
    assert torch.count_nonzero(spectrum_grads[-1]).item() > 0
    assert losses.metrics()["native_teacher_selection_owns_scales"] == 0.0
    assert losses.metrics()["native_teacher_spectrum_owns_scales"] == 1.0


def _sealed_store(tmp_path: Path) -> tuple[NativeTeacherStore, object]:
    contract = _contract()
    teacher_a, *_ = _teacher(video_demo=3, member="member_a")
    teacher_b, *_ = _teacher(video_demo=4, member="member_a")
    worker = tmp_path / "workers" / "worker_000"
    record = write_native_teacher_task_shard(
        worker_dir=worker,
        task={"authority_id": 7, "role": "meta_fit", "language": "fit task"},
        teachers=(teacher_a, teacher_b),
        contract=contract,
        provenance={"schedule": "macro1-40"},
    )
    root = publish_native_teacher_root(
        output_dir=tmp_path,
        records=(record,),
        contract=contract,
        fit_authority_roles={7: "meta_fit", 8: "target_fit"},
        provenance={"claim": "fit K1 only"},
    )
    return NativeTeacherStore(
        root,
        contract=contract,
        expected_fit_task_ids={7},
        expected_full_fit_task_ids={7, 8},
        device="cpu",
    ), teacher_a


def test_native_teacher_store_is_exact_fit_k1_and_k2_k4_are_zero_read(
    tmp_path: Path,
) -> None:
    store, expected = _sealed_store(tmp_path)
    root = read_json(store.root_manifest)
    assert root["fit_authority_task_ids"] == [7, 8]
    assert root["fit_authority_task_count"] == 2
    assert root["fit_authority_roles"] == {"meta_fit": 1, "target_fit": 1}
    assert root["K1_covered_task_count"] == 1
    assert root["K1_missing_task_ids"] == [8]
    assert root["coverage"]["roles"] == {"meta_fit": 1, "target_fit": 0}
    assert store.tensor_reads == 0
    assert (
        store.lookup(
            authority_id=999, k=2, video_demo=None, member_name=None
        )
        is None
    )
    assert (
        store.lookup_members(
            authority_id=999, k=4, video_demo=None, member_names=("missing",)
        )
        is None
    )
    assert store.tensor_reads == 0

    observed = store.lookup(
        authority_id=7, k=1, video_demo=3, member_name="member_a"
    )
    assert observed is not None
    assert store.tensor_reads == 1
    for left, right in zip(observed.a, expected.a, strict=True):
        torch.testing.assert_close(left, right)
    for left, right in zip(observed.b, expected.b, strict=True):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(observed.scales, expected.scales)
    store.lookup(authority_id=7, k=1, video_demo=4, member_name="member_a")
    assert store.tensor_reads == 1

    with pytest.raises(NativeTeacherAuthorityError, match="uncovered"):
        store.lookup(authority_id=8, k=1, video_demo=3, member_name="member_a")
    with pytest.raises(NativeTeacherAuthorityError, match="held"):
        store.lookup(authority_id=999, k=1, video_demo=3, member_name="member_a")
    with pytest.raises(NativeTeacherAuthorityError, match="cache miss"):
        store.lookup(authority_id=7, k=1, video_demo=9, member_name="member_a")
    with pytest.raises(NativeTeacherAuthorityError, match="cache miss"):
        store.lookup(authority_id=7, k=1, video_demo=3, member_name="member_b")


def test_native_teacher_safetensors_contains_only_directions_and_scales(
    tmp_path: Path,
) -> None:
    _sealed_store(tmp_path)
    path = (
        tmp_path
        / "workers"
        / "worker_000"
        / "task_007"
        / "native_teachers.safetensors"
    )
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
    assert keys
    assert all(name.endswith("scales") or ".a." in name or ".b." in name for name in keys)
    assert not any(token in name for name in keys for token in ("bank", "logit", "weight"))
    assert metadata is not None
    assert metadata["schema_version"] == "ember_ecp_g3_k1_native_teacher_task_v1"
