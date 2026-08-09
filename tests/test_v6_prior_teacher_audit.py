from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import load_v6_prior_config
from ember.expert_manifold.v6_prior_runtime import RuntimeSegment, _resolve_segment
from ember.expert_manifold.v6_prior_run_contract import (
    V6_PRIOR_TEACHER_AUDIT_COMPLETION_SCHEMA,
    V6_PRIOR_TEACHER_AUDIT_SCHEMA,
    comparison_checkpoint,
    teacher_audit_runtime,
)
from ember.expert_manifold.v6_prior_teacher_audit import (
    TeacherAuditBindings,
    gradient_span_relationships,
    run_teacher_audit,
)
from ember.pi05_source_checkpoint import DistributedContext


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/pi05_v6_condition_local_tangent_tube_writer_v3.json"
)


def test_teacher_audit_readiness_is_exact_and_fail_closed() -> None:
    config = load_v6_prior_config(CONFIG)
    assert config["teacher_audit"]["result"]["authorize_cefd"] is False
    with pytest.raises(ExpertManifoldError, match="teacher audit is not ready"):
        teacher_audit_runtime(config)

    wrong_gate = deepcopy(config)
    wrong_gate["teacher_audit"]["status"] = "ready_after_tangent_strict_nonpass"
    wrong_gate["teacher_audit"]["result"] = None
    wrong_gate["teacher_audit"]["gradient_residual_ratio_min"] = 0.2
    with pytest.raises(ExpertManifoldError, match="teacher audit is not ready"):
        teacher_audit_runtime(wrong_gate)


def test_teacher_audit_segment_is_fresh_one_macro_and_strict_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v6_prior_config(CONFIG)
    config["teacher_audit"]["status"] = "ready_after_tangent_strict_nonpass"
    config["teacher_audit"]["result"] = None
    comparison_commit = config["teacher_audit"]["comparison_commit"]
    audit_commit = "teacher-audit-descendant"
    state = {
        "branch": "codex/teacher-audit",
        "commit": audit_commit,
        "origin_main": "main",
        "upstream": "origin/codex/teacher-audit",
        "upstream_commit": audit_commit,
        "dirty_paths": [],
    }
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime.git_state", lambda _root: state
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime.git_commit_is_strict_ancestor",
        lambda ancestor, descendant: (ancestor, descendant)
        == (comparison_commit, audit_commit),
    )
    args = SimpleNamespace(
        mode="teacher-audit",
        resume=None,
        stop_after_macro=1,
        num_workers=2,
    )
    context = DistributedContext(0, 0, 6, torch.device("cuda:0"))
    assert _resolve_segment(args, config, context) == RuntimeSegment(1, (), 0, 1, 49, 50)

    args.resume = Path("checkpoints/macro_00000001")
    with pytest.raises(ExpertManifoldError):
        _resolve_segment(args, config, context)


def test_teacher_audit_comparison_asset_is_the_sealed_tangent_macro10() -> None:
    config = load_v6_prior_config(CONFIG)
    checkpoint = comparison_checkpoint(config)
    assert checkpoint.name == "macro_00000010"
    assert checkpoint.parent.parent == Path(
        config["formal_run"]["formal_result"]["training_root"]
    ).resolve()


def _span_components() -> tuple[tuple[tuple[str, int, int], ...], dict[str, torch.Tensor]]:
    layout = (
        ("compiler.weight", 0, 4),
        ("factor_heads.weight", 4, 8),
    )
    components = {
        "positive": torch.tensor([1.0, 0.0, 0.0, 0.0] * 2),
        "completion": torch.tensor([0.0, 1.0, 0.0, 0.0] * 2),
        "ranking": torch.tensor([0.0, 0.0, 1.0, 0.0] * 2),
        "distillation": torch.tensor([1.0, 1.0, 0.0, 1.0] * 2),
    }
    return layout, components


def test_flow_teacher_gradient_span_uses_explicit_block_residual() -> None:
    layout, components = _span_components()
    relationships = gradient_span_relationships(
        components, layout, pinv_rtol=1e-5
    )
    expected = 1.0 / (3.0**0.5)
    for group in ("compiler", "factor_heads", "global"):
        assert relationships[group]["existing_span_residual_ratio"] == pytest.approx(
            expected
        )
        assert relationships[group]["existing_span_effective_rank"] == 3


def test_flow_teacher_gradient_span_drops_fp32_near_collinear_noise() -> None:
    layout = (("compiler.weight", 0, 4), ("factor_heads.weight", 4, 8))
    rows = {
        "positive": [1.0, 0.0, 0.0, 0.0],
        "completion": [1.0, 1e-4, 0.0, 0.0],
        "ranking": [0.0, 0.0, 1.0, 0.0],
        "distillation": [0.0, 1.0, 0.0, 0.0],
    }
    components = {
        name: torch.tensor(value * 2, dtype=torch.float32)
        for name, value in rows.items()
    }
    relationships = gradient_span_relationships(
        components, layout, pinv_rtol=1e-5
    )
    for group in ("compiler", "factor_heads", "global"):
        evidence = relationships[group]
        assert evidence["existing_span_effective_rank"] == 2
        assert evidence["existing_span_residual_ratio"] == pytest.approx(1.0, abs=2e-4)
        assert all(
            torch.isfinite(torch.tensor(value))
            for value in evidence["normalized_projection_coefficients"].values()
        )


def test_flow_teacher_audit_writes_gate_decision_without_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler = torch.nn.Parameter(torch.zeros(4))
    factor = torch.nn.Parameter(torch.zeros(4))
    batches = iter({"ordinal": ordinal} for ordinal in range(24))

    def task_objective(_runtime, *, macro, microtask, batch):
        assert macro == 49
        assert microtask == batch["ordinal"]
        return SimpleNamespace(
            ordinal=batch["ordinal"],
            pair=object(),
            auxiliary=object(),
            flow_audit=SimpleNamespace(
                positive_gradients={"state": torch.ones(1)},
                distillation_gradients={"state": torch.ones(1)},
            ),
        )

    def components(**kwargs):
        assert kwargs["completion_only"] is True
        return SimpleNamespace(
            positive=(torch.tensor([1.0, 0.0, 0.0, 0.0]),) * 2,
            projection=(torch.tensor([0.0, 1.0, 0.0, 0.0]),) * 2,
            ranking=(torch.tensor([0.0, 0.0, 1.0, 0.0]),) * 2,
            distillation=(torch.tensor([0.0, 0.0, 0.0, 1.0]),) * 2,
        )

    def record(value):
        ordinal = value.ordinal
        passing = ordinal < 18
        expert = 0.5 if passing else 1.5
        return {
            "task_ordinal": ordinal,
            "suite": ("spatial", "object", "goal", "long")[ordinal // 6],
            "counterfactual_kind": ("reversed", "shuffled", "wrong")[ordinal % 3],
            "expert_target_loss": expert,
            "macro0_target_loss": 1.0,
            "tangent10_target_loss": 1.1,
            "distillation_loss": 0.2,
            "expert_better_than_both": passing,
        }

    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_teacher_audit.parameter_gradient_components",
        components,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_teacher_audit.teacher_audit_task_record",
        record,
    )
    runtime = SimpleNamespace(
        comparison_decoder=object(),
        context=DistributedContext(0, 0, 1, torch.device("cpu")),
        trainable_names=("compiler.weight", "factor_heads.weight"),
        trainable_parameters=(compiler, factor),
        iterator=batches,
        segment=RuntimeSegment(1, (), 0, 1, 49, 50),
        config={
            "objective": {"positive_policy_randomness": {"scheme": "matched"}},
            "teacher_audit": {
                "teacher_quality_min_tasks": 18,
                "teacher_quality_min_suites": 3,
                "gradient_span_pinv_rtol": 1e-5,
                "gradient_residual_ratio_min": 0.25,
                "physical_policy_forwards_per_task": 6,
                "real_action_dimensions": 7,
                "parameter_updates": 0,
                "rollouts": 0,
            },
        },
        policy=torch.nn.Identity(),
        args=SimpleNamespace(output_dir=tmp_path),
        run_contract={
            "data": {
                "consumed_schedule": {
                    "query": {
                        "global_examples": 480,
                        "unique_query_rows": 480,
                    }
                }
            }
        },
    )
    run_teacher_audit(
        runtime,
        TeacherAuditBindings(
            task_objective=task_objective,
            gather_task_records=lambda records, _context: records,
            runtime_maximums=lambda _context, _started, input_wait: (
                40.0,
                3_000,
                4_000,
                input_wait,
            ),
            component_layout=lambda _runtime: (
                ("compiler.weight", 0, 4),
                ("factor_heads.weight", 4, 8),
            ),
            component_norms=lambda value, _layout: {
                "global": float(torch.linalg.vector_norm(value))
            },
        ),
    )
    result = json.loads((tmp_path / "teacher_audit.json").read_text())
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert result["schema_version"] == V6_PRIOR_TEACHER_AUDIT_SCHEMA
    assert sorted(result["suite_task_counts"].values()) == [6, 6, 6, 6]
    assert result["teacher_quality_gate"]["passing_tasks"] == 18
    assert result["teacher_quality_gate"]["passing_suites"] == 3
    assert result["gradient_nonredundancy_gate"]["passed"] is True
    assert result["decision"]["authorize_cefd"] is True
    assert result["parameter_updates"] == result["rollouts"] == 0
    assert completion["schema_version"] == V6_PRIOR_TEACHER_AUDIT_COMPLETION_SCHEMA
    assert completion["parameter_updates"] == completion["rollouts"] == 0
    assert compiler.grad is factor.grad is None
