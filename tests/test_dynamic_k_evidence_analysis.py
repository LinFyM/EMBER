from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval import analysis
from ember.pi05_eval.benchmark_comparison import (
    BENCHMARK_COMPARISON_SCHEMA,
    analyze_benchmark_references,
    benchmark_reference_comparison,
    v6_fast_143_reference,
)
from pi05_eval_analysis_fixture import (
    TASKS,
    result as _result,
    success_keys as _success_keys,
)


_WRITER_IDENTITY_KEYS = (
    "execution_backend",
    "config",
    "writer_asset",
    "evaluation_authority",
    "video_data",
    "lora_contract",
    "video_schedule",
    "pairing_reference",
)


def _dynamic_result(macro: int) -> dict:
    result = _result(macro, "correct", set())
    adapter = result["adapter"]
    adapter.update(
        {
            "schema_version": (
                "ember_pi05_dynamic_k_backbone_memory_rank8_eval_adapter_v1"
            ),
            "kind": "dynamic_k_backbone_memory_writer",
            "arm": "dynamic_k_backbone_memory_rank8_correct",
            "config": {
                "schema": "ember_pi05_dynamic_k_backbone_memory_rank8_as_writer_v1"
            },
            "evaluation_authority": {"formal_status": "sealed"},
            "lora_contract": {
                "reference": "rank8-lora-v1",
                "rank": 8,
                "target_count": 38,
            },
            "pairing_reference": "ember_pi05_dynamic_k_one_shot_pairing_v1",
        }
    )
    adapter["writer_asset"].update(
        {
            "kind": "dynamic_k_writer_macro_checkpoint",
            "method_macro": macro,
            "training_mode": "formal",
        }
    )
    adapter["information_wall"]["writer_input"] = (
        "exact task language plus one action-hidden teacher video through "
        "the dynamic-K graph"
    )
    for row in result["rows"]:
        writer = row["writer"]
        writer.update(
            {
                "schema_version": (
                    "ember_pi05_dynamic_k_backbone_memory_rank8_episode_v1"
                ),
                "method_arm": adapter["arm"],
                "writer_method_macro": macro,
                "writer_checkpoint_kind": "dynamic_k_writer_macro_checkpoint",
                "lora_contract_reference": "rank8-lora-v1",
                "pairing_reference": adapter["pairing_reference"],
                "evaluation_k": 1,
                "condition_video_offsets": [0, 1],
            }
        )
    result["arm"] = adapter["arm"]
    result["paired_control"]["writer"] = {
        key: adapter[key] for key in _WRITER_IDENTITY_KEYS
    }
    return result


def _direct_family_b_result(macro: int) -> dict:
    result = _dynamic_result(macro)
    adapter = result["adapter"]
    adapter.update(
        {
            "schema_version": (
                "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_"
                "eval_adapter_v1"
            ),
            "arm": "dynamic_k_semantic_address_direct_family_b_rank8_correct",
            "config": {
                "schema": (
                    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_"
                    "as_writer_v1"
                )
            },
        }
    )
    adapter["writer_asset"]["kind"] = (
        "dynamic_k_semantic_address_direct_family_b_rank8_macro_checkpoint"
    )
    for row in result["rows"]:
        row["writer"].update(
            {
                "schema_version": (
                    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_"
                    "episode_v1"
                ),
                "method_arm": adapter["arm"],
                "writer_checkpoint_kind": adapter["writer_asset"]["kind"],
            }
        )
    result["arm"] = adapter["arm"]
    result["paired_control"]["writer"] = {
        key: adapter[key] for key in _WRITER_IDENTITY_KEYS
    }
    return result


def test_dynamic_k_rank8_family_accepts_incremental_macro50_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analysis,
        "paired_writer_identity",
        lambda adapter: {key: adapter[key] for key in _WRITER_IDENTITY_KEYS},
    )
    calls = []

    def validate(_adapter: dict, evidence: dict, **identity: object) -> bool:
        calls.append(
            (
                evidence["evaluation_k"],
                identity["init_state_id"],
                identity["registered_episode_schema"],
            )
        )
        return True

    monkeypatch.setattr(analysis, "_validate_dynamic_k_episode_evidence", validate)
    result = analysis.checkpoint_curve_analysis({"macro50": _dynamic_result(50)})
    assert result["method_family"] == "dynamic_k_backbone_memory_rank8_v1"
    assert result["panels"]["correct400"]["50"]["overall"]["episodes"] == 400
    assert len(calls) == 400
    assert {call[2] for call in calls} == {
        "ember_pi05_dynamic_k_backbone_memory_rank8_episode_v1"
    }


def test_direct_family_b_formal_panel_accepts_runtime_writer_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analysis,
        "paired_writer_identity",
        lambda adapter: {key: adapter[key] for key in _WRITER_IDENTITY_KEYS},
    )
    monkeypatch.setattr(
        analysis, "_validate_dynamic_k_episode_evidence", lambda *args, **kwargs: True
    )
    indexed = analysis._formal_panel_index(_direct_family_b_result(50))
    assert len(indexed) == 400


def test_dynamic_k_family_rejects_rank16_or_nonprefix_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analysis,
        "paired_writer_identity",
        lambda adapter: {key: adapter[key] for key in _WRITER_IDENTITY_KEYS},
    )
    monkeypatch.setattr(
        analysis, "_validate_dynamic_k_episode_evidence", lambda *args, **kwargs: True
    )
    rank16 = _dynamic_result(50)
    rank16["adapter"]["lora_contract"]["rank"] = 16
    rank16["paired_control"]["writer"]["lora_contract"]["rank"] = 16
    with pytest.raises(Pi05EvaluationError, match="sealed formal"):
        analysis.checkpoint_curve_analysis({"macro50": rank16})
    with pytest.raises(Pi05EvaluationError, match="non-empty prefix"):
        analysis.checkpoint_curve_analysis({"macro100": _dynamic_result(100)})


def test_benchmark_comparison_separates_strict_churn_from_v6_fast_counts() -> None:
    reference_successes = _success_keys(
        lambda _suite, _task, state: state == 0
    )
    candidate_successes = _success_keys(
        lambda suite, task, state: state == 0
        or (suite == "libero_spatial" and task == 1 and state == 1)
    )
    reference = _result(0, "correct", reference_successes, family="legacy")
    candidate = _result(10, "correct", candidate_successes, family="ecp")
    comparison = benchmark_reference_comparison(
        candidate,
        strict_references={"latest_strict_baseline": reference},
        per_task_references={"v6_fast_143": v6_fast_143_reference(TASKS)},
    )
    assert comparison["schema_version"] == BENCHMARK_COMPARISON_SCHEMA
    strict = comparison["strict_paired_references"]["latest_strict_baseline"]
    assert strict["reference_to_candidate"]["overall"]["gained"] == 1
    assert strict["reference_to_candidate"]["overall"]["lost"] == 0
    assert strict["reference_to_candidate"]["overall"]["churn"] == 1
    counts = comparison["per_task_only_references"]["v6_fast_143"]
    assert counts["reference_to_candidate"]["gained_lost_churn"] is None
    assert counts["reference_to_candidate"]["success_delta"] == 9 - 143

    drifted = copy.deepcopy(reference)
    drifted["paired_control"]["model"]["checkpoint"] = "/different-source"
    with pytest.raises(Pi05EvaluationError, match="shared policy contract"):
        benchmark_reference_comparison(
            candidate, strict_references={"drifted": drifted}
        )


def test_benchmark_comparison_rejects_false_count_only_pairing_claim() -> None:
    candidate = _result(10, "correct", set(), family="ecp")
    invalid = {
        "successes": 0,
        "comparison_scope": "strict_episode_paired",
        "reason_not_episode_paired": "none",
        "per_task": [
            {"suite": suite, "task_id": task_id, "successes": 0}
            for suite, task_id in TASKS
        ],
    }
    with pytest.raises(Pi05EvaluationError, match="false pairing"):
        benchmark_reference_comparison(
            candidate, per_task_references={"invalid": invalid}
        )


def test_benchmark_comparison_canonicalizes_only_worktree_config_prefixes() -> None:
    reference = _result(0, "correct", set(), family="legacy")
    candidate = _result(10, "correct", set(), family="ecp")
    for value, worktree in ((reference, "old-wt"), (candidate, "new-wt")):
        paired = value["paired_control"]
        paired["normalization"] = {
            "path": (
                f"/data1/x/worktrees/{worktree}/configs/pi05_source_corpus_v1/"
                "source_normalization.json"
            ),
            "bytes": 5759,
        }
        paired["tokenizer"] = {
            "path": "/data1/x/tokenizer.model",
            "bytes": 4264023,
            "manifest_path": (
                f"/data1/x/worktrees/{worktree}/configs/libero_24_8_8_v1/"
                "pi05_tokenizer_manifest.json"
            ),
        }

    benchmark_reference_comparison(
        candidate,
        strict_references={"old": reference},
    )

    config_drift = copy.deepcopy(candidate)
    config_drift["paired_control"]["normalization"]["path"] = (
        "/data1/x/worktrees/new-wt/configs/pi05_source_corpus_v2/"
        "source_normalization.json"
    )
    with pytest.raises(Pi05EvaluationError, match="shared policy contract"):
        benchmark_reference_comparison(
            config_drift,
            strict_references={"old": reference},
        )

    content_drift = copy.deepcopy(candidate)
    content_drift["paired_control"]["normalization"]["bytes"] += 1
    with pytest.raises(Pi05EvaluationError, match="shared policy contract"):
        benchmark_reference_comparison(
            content_drift,
            strict_references={"old": reference},
        )


def test_benchmark_comparison_reaggregates_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_root = tmp_path / "candidate"
    reference_root = tmp_path / "reference"
    candidate_root.mkdir()
    reference_root.mkdir()
    (candidate_root / "results.json").write_text("{}\n", encoding="utf-8")
    (reference_root / "results.json").write_text("{}\n", encoding="utf-8")
    payloads = {
        candidate_root.resolve(): _result(10, "correct", set(), family="ecp"),
        reference_root.resolve(): _result(0, "correct", set(), family="legacy"),
    }
    monkeypatch.setattr(
        "ember.pi05_eval_results.aggregate_run", lambda root: payloads[root]
    )
    output = tmp_path / "comparison.json"
    result = analyze_benchmark_references(
        candidate_root,
        strict_reference_roots={"old": reference_root},
        output_path=output,
    )
    assert result["schema_version"] == BENCHMARK_COMPARISON_SCHEMA
    assert result["per_task_only_references"]["v6_fast_143"]["reference"][
        "successes"
    ] == 143
    assert output.is_file()
