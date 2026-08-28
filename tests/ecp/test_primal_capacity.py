from __future__ import annotations

from pathlib import Path

import torch

from ember.ecp.bank_conditioning.primal_capacity import TaskLocalPrimalCode
from ember.ecp.bank_conditioning.primal_capacity_run import (
    P1_SCHEMA,
    load_primal_capacity_config,
)
from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.shared_compiler_native_teacher import NativeTeacherFactors


REPO_ROOT = Path(__file__).resolve().parents[2]


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "q", TargetFamily.Q, 0, 4, 16),
        TargetOwner(1, "v", TargetFamily.V, 0, 4, 8),
        TargetOwner(2, "action_in", TargetFamily.ACTION_IN, None, 8, 16),
        TargetOwner(3, "action_out", TargetFamily.ACTION_OUT, None, 12, 8),
    )


def _teacher(seed: int, owners: tuple[TargetOwner, ...]) -> NativeTeacherFactors:
    generator = torch.Generator().manual_seed(seed)
    return NativeTeacherFactors(
        authority_id=1,
        video_demo=5,
        member_name=f"member_{seed}",
        a=tuple(
            torch.randn(4, owner.in_features, generator=generator)
            for owner in owners
        ),
        b=tuple(
            torch.randn(4, owner.out_features, generator=generator)
            for owner in owners
        ),
        scales=torch.rand(len(owners), 4, generator=generator) + 0.1,
        provenance={"kind": "test"},
    )


def test_preregistered_primal_capacity_config_loads() -> None:
    config = load_primal_capacity_config(
        REPO_ROOT / "configs/pi05_ecp_primal_capacity_p1_v1.json"
    )
    assert config["schema_version"] == P1_SCHEMA


def test_task_local_primal_code_has_shared_primals_and_fixed_fit_scale() -> None:
    owners = _owners()
    code = TaskLocalPrimalCode(
        owners,
        (_teacher(3, owners), _teacher(7, owners)),
        s_ref=torch.ones(len(owners)),
    )
    inputs = code.input_primals()
    outputs = code.output_primals()
    assert [tuple(value.shape) for value in inputs] == [
        (4, owner.in_features) for owner in owners
    ]
    assert [value.shape[1] for value in outputs] == [4] * len(owners)
    scales = code.scales(torch.ones(len(owners)))
    assert scales.shape == (len(owners), 4)
    assert bool((scales > 0).all())
    assert "fixed_scales" in dict(code.named_buffers())
    assert not any("scale" in name for name, _ in code.named_parameters())
    loss = sum(value.square().mean() for value in (*inputs, *outputs))
    loss.backward()
    assert all(
        value.grad is not None and bool(torch.isfinite(value.grad).all())
        for value in code.parameters()
    )
