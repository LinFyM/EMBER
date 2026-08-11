from __future__ import annotations

import argparse
import copy
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import ember.pi05_eval.rank_reserved_cache_transform as transform_module
import ember.pi05_eval.rank_reserved_compiler_diagnostic as diagnostic_module
import ember.pi05_eval.rank_reserved_compiler_evidence as evidence_module
import ember.writer.rank_reserved_compiler as compiler_module
from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_CANONICAL_CONFIG,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.paired_metrics import summarize_panel
from ember.pi05_eval.rank_reserved_cache_transform import (
    _action_factor_names,
    _canonical_batch_population_plan,
    _compile_qv_batch,
    _direct_action_state,
    _validate_action_file_copy,
)
from ember.pi05_eval.rank_reserved_compiler_diagnostic import (
    COMPILER_DIAGNOSTIC_ONLINE_ROOT,
    COMPILER_DIAGNOSTIC_SOURCE_ROOT,
    COMPILER_DIAGNOSTIC_TARGET_ROOT,
    compiler_diagnostic_authority_payload,
    compiler_diagnostic_lineage_matches,
    validate_compiler_diagnostic_prepare_args,
)
from ember.pi05_eval.rank_reserved_compiler_evidence import (
    compiler_diagnostic_evidence,
)
from ember.pi05_eval.run_contract import _parallel_contract
from ember.writer.condition_update import (
    compact_rank2_effective_tangent,
    pivot_preserving_base_factors,
)
from ember.writer.errors import WriterModelError
from ember.writer.rank_reserved_compiler import (
    RANK_RESERVED_QV_BASE_RANK,
    compile_rank_reserved_qv_factors,
)


def test_shared_qv_helper_is_exact_old_inline_macro0_owner() -> None:
    generator = torch.Generator().manual_seed(19)
    base_a = torch.randn(2, 18, 16, 20, generator=generator).to(torch.bfloat16)
    base_b = torch.randn(2, 18, 24, 16, generator=generator).to(torch.bfloat16)

    old_a, old_b, old_pivots = pivot_preserving_base_factors(
        base_a, base_b, keep=RANK_RESERVED_QV_BASE_RANK
    )
    old_public_a = torch.cat(
        (old_a, old_a.new_zeros(*old_a.shape[:-2], 2, old_a.shape[-1])),
        dim=-2,
    )
    old_public_b = torch.cat(
        (old_b, old_b.new_zeros(*old_b.shape[:-2], old_b.shape[-2], 2)),
        dim=-1,
    )
    public_a, public_b, pivots = compile_rank_reserved_qv_factors(base_a, base_b)

    assert torch.equal(public_a, old_public_a)
    assert torch.equal(public_b, old_public_b)
    assert torch.equal(pivots, old_pivots)
    assert torch.equal(
        public_b[..., :14],
        torch.gather(
            base_b,
            -1,
            pivots.unsqueeze(-2).expand(*base_b.shape[:-1], 14),
        ),
    )
    assert torch.count_nonzero(public_a[..., 14:, :]) == 0
    assert torch.count_nonzero(public_b[..., 14:]) == 0


def test_compiler_diagnostic_cli_dispatches_all_four_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.pi05_eval.rank_reserved_cache_launch as launch_module

    calls: list[str] = []

    def prepare_run(args):
        calls.append(args.command)
        return "prepared"

    monkeypatch.setattr(
        diagnostic_module,
        "validate_compiler_diagnostic_prepare_args",
        lambda args: calls.append(f"validated:{args.command}"),
    )
    monkeypatch.setattr(
        launch_module,
        "compiler_cache_run",
        lambda args: calls.append(args.command) or "cached",
    )
    monkeypatch.setattr(
        launch_module,
        "compiler_cache_worker_run",
        lambda args: calls.append(args.command) or "worker",
    )
    monkeypatch.setattr(
        evidence_module,
        "compiler_evidence_run",
        lambda args: calls.append(args.command) or "evidence",
    )

    prepare_args = SimpleNamespace(command="rank-reserved-compiler-prepare")
    assert (
        diagnostic_module.compiler_diagnostic_cli_run(prepare_args, prepare_run)
        == "prepared"
    )
    assert prepare_args.writer_cache_population_recipe["mode"] == "prefilled"
    expected = (
        ("rank-reserved-compiler-cache", "cached"),
        ("rank-reserved-compiler-cache-worker", "worker"),
        ("rank-reserved-compiler-evidence", "evidence"),
    )
    for command, result in expected:
        assert diagnostic_module.compiler_diagnostic_cli_run(
            SimpleNamespace(command=command), prepare_run
        ) == result
    assert calls == [
        "validated:rank-reserved-compiler-prepare",
        "rank-reserved-compiler-prepare",
        "rank-reserved-compiler-cache",
        "rank-reserved-compiler-cache-worker",
        "rank-reserved-compiler-evidence",
    ]


def test_shared_qv_helper_keeps_cycle1_pivot_before_compact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_a = torch.zeros(1, 18, 16, 40, dtype=torch.bfloat16)
    base_b = torch.zeros(1, 18, 40, 16, dtype=torch.bfloat16)
    tangent_a = torch.zeros_like(base_a, dtype=torch.float32)
    tangent_b = torch.zeros_like(base_b, dtype=torch.float32)
    calls: list[str] = []

    def pivot(value_a, value_b, *, keep):
        assert value_a is base_a
        assert value_b is base_b
        assert keep == 14
        calls.append("pivot")
        return (
            torch.ones(1, 18, 14, 40, dtype=torch.bfloat16),
            torch.ones(1, 18, 40, 14, dtype=torch.bfloat16),
            torch.arange(14).expand(1, 18, 14),
        )

    def compact(value_a, value_b, delta_a, delta_b):
        assert calls == ["pivot"]
        assert value_a is base_a
        assert value_b is base_b
        assert delta_a is tangent_a
        assert delta_b is tangent_b
        calls.append("compact")
        return (
            torch.full((1, 18, 2, 40), 2, dtype=torch.bfloat16),
            torch.full((1, 18, 40, 2), 3, dtype=torch.bfloat16),
        )

    monkeypatch.setattr(compiler_module, "pivot_preserving_base_factors", pivot)
    monkeypatch.setattr(
        compiler_module, "compact_rank2_effective_tangent", compact
    )

    public_a, public_b, pivots = compile_rank_reserved_qv_factors(
        base_a,
        base_b,
        tangent_a=tangent_a,
        tangent_b=tangent_b,
    )

    assert calls == ["pivot", "compact"]
    assert torch.count_nonzero(public_a[..., :14, :] != 1) == 0
    assert torch.count_nonzero(public_a[..., 14:, :] != 2) == 0
    assert torch.count_nonzero(public_b[..., :14] != 1) == 0
    assert torch.count_nonzero(public_b[..., 14:] != 3) == 0
    assert torch.equal(pivots, torch.arange(14).expand(1, 18, 14))


def test_shared_qv_helper_is_exact_old_inline_cycle1_owner() -> None:
    generator = torch.Generator().manual_seed(23)
    base_a = torch.randn(1, 18, 16, 40, generator=generator).to(torch.bfloat16)
    base_b = torch.randn(1, 18, 40, 16, generator=generator).to(torch.bfloat16)
    tangent_a = torch.randn(base_a.shape, generator=generator) * 0.01
    tangent_b = torch.randn(base_b.shape, generator=generator) * 0.01

    old_a, old_b, old_pivots = pivot_preserving_base_factors(
        base_a, base_b, keep=RANK_RESERVED_QV_BASE_RANK
    )
    old_residual_a, old_residual_b = compact_rank2_effective_tangent(
        base_a, base_b, tangent_a, tangent_b
    )
    old_public_a = torch.cat((old_a, old_residual_a), dim=-2)
    old_public_b = torch.cat((old_b, old_residual_b), dim=-1)
    public_a, public_b, pivots = compile_rank_reserved_qv_factors(
        base_a,
        base_b,
        tangent_a=tangent_a,
        tangent_b=tangent_b,
    )

    assert tangent_a.dtype == tangent_b.dtype == torch.float32
    assert public_a.dtype == base_a.dtype == torch.bfloat16
    assert public_b.dtype == base_b.dtype == torch.bfloat16
    torch.testing.assert_close(public_a, old_public_a)
    assert torch.isfinite(public_a).all()
    assert torch.equal(public_b, old_public_b)
    assert torch.equal(pivots, old_pivots)


def test_transform_invokes_shared_helper_inside_cuda_bf16_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_a = torch.zeros(8, 18, 16, 4, dtype=torch.bfloat16)
    source_b = torch.zeros(8, 18, 6, 16, dtype=torch.bfloat16)
    active = False
    contexts: list[tuple[str, torch.dtype]] = []
    helper_calls = 0

    def stack(_states, targets, *, device):
        assert len(targets) == 18
        assert device == torch.device("cuda")
        return source_a, source_b

    @contextmanager
    def autocast(*, device_type, dtype):
        nonlocal active
        assert active is False
        contexts.append((device_type, dtype))
        active = True
        try:
            yield
        finally:
            active = False

    def helper(base_a, base_b):
        nonlocal helper_calls
        assert active is True
        helper_calls += 1
        return base_a, base_b, torch.zeros(8, 18, 14, dtype=torch.long)

    monkeypatch.setattr(transform_module, "_stack_qv", stack)
    monkeypatch.setattr(transform_module.torch, "autocast", autocast)
    monkeypatch.setattr(
        transform_module, "compile_rank_reserved_qv_factors", helper
    )

    result = _compile_qv_batch(
        (),
        tuple(f"q{index}" for index in range(18)),
        tuple(f"v{index}" for index in range(18)),
        device=torch.device("cuda"),
    )

    assert set(result) == {"q", "v"}
    assert helper_calls == 2
    assert contexts == [("cuda", torch.bfloat16), ("cuda", torch.bfloat16)]


def test_partial_resume_preserves_full_b8_compile_and_writes_only_missing() -> None:
    requests = tuple(f"source-{index}" for index in range(8))
    complete = (True, False, True, True, False, True, True, True)

    forward, missing = _canonical_batch_population_plan(requests, complete)

    assert forward == requests
    assert missing == (1, 4)
    assert _canonical_batch_population_plan(requests, (True,) * 8) == ((), ())
    with pytest.raises(WriterModelError, match="canonical B8"):
        _canonical_batch_population_plan(requests[:7], complete[:7])


def test_action_copy_is_verified_from_independent_target_safetensors(
    tmp_path: Path,
) -> None:
    action_targets = ("model.action_in_proj", "model.action_out_proj")
    action_names = _action_factor_names(action_targets)
    source_state = {
        name: torch.arange(12, dtype=torch.float32).reshape(3, 4) + index
        for index, name in enumerate(action_names)
    }
    selected = _direct_action_state(source_state, action_names)
    assert all(selected[name] is source_state[name] for name in action_names)

    request = SimpleNamespace(entry_id="episode")
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_file = source_root / "entries/episode/lora.safetensors"
    target_file = target_root / "entries/episode/lora.safetensors"
    source_file.parent.mkdir(parents=True)
    target_file.parent.mkdir(parents=True)
    save_file(source_state, str(source_file))
    save_file(
        {name: value.clone() for name, value in source_state.items()},
        str(target_file),
    )
    source_contract = {"writer_lora_cache": {"root": str(source_root)}}
    target_contract = {"writer_lora_cache": {"root": str(target_root)}}

    assert _validate_action_file_copy(
        source_contract,
        request,
        target_contract,
        request,
        action_names=action_names,
    ) == 4
    assert _validate_action_file_copy(
        source_contract,
        request,
        target_contract,
        request,
        action_names=action_names,
        source_state=source_state,
    ) == 4

    changed = {name: value.clone() for name, value in source_state.items()}
    changed[action_names[0]][0, 0] += 1
    target_file.unlink()
    save_file(changed, str(target_file))
    with pytest.raises(WriterModelError, match="not bit-exact"):
        _validate_action_file_copy(
            source_contract,
            request,
            target_contract,
            request,
            action_names=action_names,
        )


def test_prefilled_parallel_contract_has_no_fake_generators_or_handoff() -> None:
    authorities = SimpleNamespace(
        config={
            "parallel": {
                "physical_gpu_count": 8,
                "allowed_replicas_per_gpu": [1, 2, 3, 6],
            }
        }
    )
    generated = _parallel_contract(
        authorities,
        physical_gpu_ids=(0, 2, 5),
        replicas_per_gpu=3,
        writer_adapter=True,
        prefilled_population=False,
        writer_generators_per_gpu=1,
        writer_generation_batch_size=8,
    )
    prefilled = _parallel_contract(
        authorities,
        physical_gpu_ids=(0, 2, 5),
        replicas_per_gpu=3,
        writer_adapter=True,
        prefilled_population=True,
        writer_generators_per_gpu=1,
        writer_generation_batch_size=8,
    )

    assert generated["writer_generators_per_gpu"] == 1
    assert generated["writer_generation_worker_count"] == 3
    assert generated["generator_source_policy_processes_reused_for_rollout"] is True
    assert prefilled["writer_generators_per_gpu"] == 0
    assert prefilled["writer_generation_worker_count"] == 0
    assert prefilled["generator_source_policy_processes_reused_for_rollout"] is False
    assert prefilled["writer_generation_batch_size"] == 8
    assert prefilled["physical_gpu_ids"] == [0, 2, 5]
    assert prefilled["worker_count"] == 9


def test_compiler_prepare_is_bound_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    macro0 = tmp_path / "macro0"
    config = {"assets": {"macro0": {"checkpoint": str(macro0)}}}
    monkeypatch.setattr(
        diagnostic_module, "load_rank_reserved_config", lambda _path: config
    )
    monkeypatch.setattr(
        diagnostic_module,
        "rank_reserved_asset",
        lambda _config, checkpoint: {
            "kind": "v6_qv_rank14_zero_program_load_only",
            "method_macro": 0,
            "enable_program_residual": False,
        }
        if checkpoint == macro0.resolve()
        else None,
    )
    args = argparse.Namespace(
        output_dir=diagnostic_module.REPO_ROOT / COMPILER_DIAGNOSTIC_TARGET_ROOT,
        mode="formal",
        role="validation",
        state_count=50,
        writer_generation_batch_size=8,
        writer_lora_cache_root=None,
        expert_manifold_config=RANK_RESERVED_CANONICAL_CONFIG,
        expert_manifold_checkpoint=macro0,
        expert_manifold_video_condition="correct",
        expert_manifold_video_sampling="without_replacement",
        expert_manifold_video_data_root=tmp_path,
        gpu_indices="0,2,5",
        source_sft_config=None,
        source_sft_checkpoint=None,
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
    )

    validate_compiler_diagnostic_prepare_args(args)
    args.mode = "screen"
    with pytest.raises(Pi05EvaluationError, match="arguments changed"):
        validate_compiler_diagnostic_prepare_args(args)


@pytest.mark.parametrize(
    "changed_field", ("model", "policy", "environment", "tasks")
)
def test_target_rollout_identity_is_exactly_bound_to_old134(
    changed_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "mode": "formal",
        "role": "validation",
        "git": {"commit": "a" * 40},
        "model": {"checkpoint": "/canonical/source-step1000", "step": 1000},
        "tokenizer": {
            "path": "/canonical/tokenizer.model",
            "bytes": 123,
            "manifest_path": "/old-worktree/configs/split/tokenizer.json",
        },
        "normalization": {
            "path": "/old-worktree/configs/source/normalization.json",
            "bytes": 456,
            "source_only_numeric_reads": True,
            "validation_or_test_numeric_reads": 0,
        },
        "policy": {"arm": "frozen_pi05_source_base", "replan_steps": 5},
        "environment": {"render_resolution": 256, "terminate_on_success": True},
        "rng": {"inference_seed": 7},
        "libero_paths": {"assets": "/canonical/libero-assets"},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 1,
                "init_state_ids": [0, 1],
            }
        ],
        "parallel": {"physical_gpu_ids": [0]},
        "adapter": {"source": {"checkpoint": "/canonical/source-step1000"}},
    }
    target = copy.deepcopy(source)
    target["git"] = {"commit": "b" * 40}
    target["tokenizer"]["manifest_path"] = (
        "/new-worktree/configs/split/tokenizer.json"
    )
    target["normalization"]["path"] = (
        "/new-worktree/configs/source/normalization.json"
    )
    # Fresh contracts carry tuples in memory; sealed JSON contracts reload lists.
    # That representation-only difference must not change rollout identity.
    target["tasks"][0]["init_state_ids"] = (0, 1)
    paired_fields = (
        "mode",
        "role",
        "git",
        "model",
        "tokenizer",
        "normalization",
        "tasks",
        "environment",
        "policy",
        "rng",
        "parallel",
    )
    target["paired_control"] = {
        name: copy.deepcopy(target[name]) for name in paired_fields
    }
    monkeypatch.setattr(
        diagnostic_module,
        "load_compiler_diagnostic_source_contract",
        lambda _authority: source,
    )

    assert diagnostic_module.compiler_diagnostic_rollout_contract_matches(
        {}, target
    )
    if changed_field == "tasks":
        target["tasks"][0]["task_id"] = 2
    else:
        target[changed_field]["drift"] = True
    target["paired_control"][changed_field] = copy.deepcopy(target[changed_field])
    assert not diagnostic_module.compiler_diagnostic_rollout_contract_matches(
        {}, target
    )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_lineage_allows_authority_then_docs_but_rejects_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "ember@example.invalid")
    _git(repo, "config", "user.name", "EMBER Test")
    (repo / "src").mkdir()
    (repo / "src/implementation.py").write_text("owner = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "implementation")
    implementation = _git(repo, "rev-parse", "HEAD")

    authority_path = repo / "configs/diagnostic.json"
    authority_path.parent.mkdir()
    authority_path.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "authority")
    authority_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(diagnostic_module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        diagnostic_module, "COMPILER_DIAGNOSTIC_AUTHORITY", authority_path
    )
    authority = {"implementation_commit": implementation}

    assert compiler_diagnostic_lineage_matches(authority, authority_commit)
    (repo / "README.md").write_text("decision\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-qm", "docs")
    docs_commit = _git(repo, "rev-parse", "HEAD")
    assert compiler_diagnostic_lineage_matches(authority, docs_commit)

    (repo / "src/implementation.py").write_text("owner = 2\n", encoding="utf-8")
    _git(repo, "add", "src/implementation.py")
    _git(repo, "commit", "-qm", "source drift")
    source_commit = _git(repo, "rev-parse", "HEAD")
    assert not compiler_diagnostic_lineage_matches(authority, source_commit)


def _paired_rows() -> tuple[dict, dict, dict]:
    keys = [
        ("libero_spatial", task_id, init_state_id)
        for task_id in range(8)
        for init_state_id in range(50)
    ]
    old_counts = (0, 5, 48, 34, 0, 35, 11, 1)
    old_success = {
        ("libero_spatial", task_id, state)
        for task_id, count in enumerate(old_counts)
        for state in range(count)
    }
    old_task2 = sorted(key for key in old_success if key[1] == 2)
    failures = [key for key in keys if key not in old_success]
    online_success = (old_success - set(old_task2[:21])) | set(failures[:15])
    task1_failures = [key for key in failures if key[1] == 1]
    compiler_success = (old_success - set(old_task2[:4])) | set(task1_failures[:14])

    def rows(successes):
        return {
            key: {
                "suite": key[0],
                "task_id": key[1],
                "init_state_id": key[2],
                "success": key in successes,
            }
            for key in keys
        }

    return rows(old_success), rows(compiler_success), rows(online_success)


def test_evidence_triangle_never_retroactively_passes_gate_b_or_cycle1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "target"
    old_root = tmp_path / "old"
    online_root = tmp_path / "online"
    output.mkdir()
    old_root.mkdir()
    online_root.mkdir()
    contract = {"git": {"commit": "e" * 40}}
    authority = compiler_diagnostic_authority_payload("d" * 40)
    old_rows, compiler_rows, online_rows = _paired_rows()
    panels = {
        old_root.resolve(): old_rows,
        output.resolve(): compiler_rows,
        online_root.resolve(): online_rows,
    }
    original = {
        "passed": False,
        "immutable_reference_valid": True,
        "old_full_rank_macro0": {"correct": 134, "breadth": 6},
        "new_rank14_macro0": {"correct": 128, "breadth": 7},
        "paired_transition": {"overall": {"gained": 15, "lost": 21}},
    }
    config = {
        "evaluation": {
            "gates": {
                "macro0_correct_min": 130,
                "macro0_breadth_min": 6,
                "macro0_lost_to_paired_old134_max": 10,
            }
        }
    }

    monkeypatch.setattr(evidence_module, "load_run_contract", lambda _path: contract)
    monkeypatch.setattr(
        evidence_module,
        "validate_compiler_diagnostic_contract",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        evidence_module, "load_rank_reserved_config", lambda _path: config
    )
    monkeypatch.setattr(
        evidence_module,
        "compiler_diagnostic_output_path",
        lambda relative, **_kwargs: (
            old_root if relative == COMPILER_DIAGNOSTIC_SOURCE_ROOT else online_root
        ),
    )
    monkeypatch.setattr(
        evidence_module, "rank_reserved_macro0_evidence", lambda _config: original
    )

    def panel(root, **_kwargs):
        rows = panels[root.resolve()]
        return {}, rows, summarize_panel(list(rows.values()))

    monkeypatch.setattr(evidence_module, "_panel", panel)
    monkeypatch.setattr(evidence_module, "_assert_row_pairing", lambda *_a, **_k: None)

    evidence = compiler_diagnostic_evidence(output)

    assert evidence["old134"] == {"correct": 134, "breadth": 6}
    assert evidence["compiler_only"] == {"correct": 144, "breadth": 6}
    assert evidence["online_regenerated"] == {"correct": 128, "breadth": 7}
    assert evidence["triangle"]["old134_to_compiler_only"]["overall"]["lost"] == 4
    assert evidence["counterfactual_gate_passed"] is True
    assert evidence["original_gate_b_passed"] is False
    assert evidence["retroactively_changes_original_gate_b"] is False
    assert evidence["authorizes_cycle1"] is False
    assert evidence["success_sets"]["three_way_union"]["count"] > 0
    assert authority["transform"]["action_tensor_equal_checks"] == 1_600
