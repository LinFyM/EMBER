from __future__ import annotations

from ember.ecp.g3_gate import _g3_adapter_authority, _success_metrics
from ember.static_task_lora import STATIC_TASK_LORA_KIND


KEYS = {
    ("libero_spatial", 0),
    ("libero_spatial", 9),
    ("libero_object", 8),
    ("libero_goal", 5),
    ("libero_10", 6),
}


def _adapter(name: str, *, language: bool = False) -> dict[str, object]:
    condition = "learned_language_only" if language else name
    adapter: dict[str, object] = {
        "kind": STATIC_TASK_LORA_KIND,
        "single_complete_rank16": True,
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "tasks": [
            {
                "adapter_path": f"/{condition}/{index}",
                "single_complete_rank16": True,
            }
            for index in range(5)
        ],
        "condition": {
            "name": condition,
            "K": 0 if language else 4,
            "video_demos": (
                []
                if language
                else ([0, 1, 2, 3] if name == "same_task_other" else [5, 6, 7, 8])
            ),
        },
        "information_wall": {
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
            "shuffled_or_reversed_use": False,
        },
        "training_commit": "a" * 40,
        "materialization_commit": "b" * 40,
    }
    if language:
        adapter["shared_run_contract"] = {
            "schema_version": "ember_ecp_g3_language_only_baseline_v1",
            "stage": "g3_learned_language_only",
            "mode": "formal",
            "method": {"fixed": True},
            "held_video_reads": 0,
            "held_action_or_reward_reads": 0,
        }
    else:
        adapter["compiler_checkpoint"] = {"path": "/checkpoint", "macro": 5}
    return adapter


def _arm(name: str, successes: list[bool], *, language: bool = False) -> dict:
    ordered = sorted(KEYS)
    rows = {
        (*key, 0): {"success": value}
        for key, value in zip(ordered, successes, strict=True)
    }
    per_task = {
        key: int(value)
        for key, value in zip(ordered, successes, strict=True)
    }
    arm_name = (
        "ecp_shared_compiler_g3_learned_language_only"
        if language
        else f"ecp_shared_compiler_g3_{name}"
    )
    return {
        "arm": arm_name,
        "rows": rows,
        "per_task": per_task,
        "successes": sum(successes),
        "breadth": sum(successes),
        "adapter": _adapter(name, language=language),
    }


def test_g3_gate_metrics_and_single_checkpoint_authority() -> None:
    arms = {
        "carrier": {
            "rows": _arm("carrier", [True, True, True, False, False])["rows"],
            "per_task": _arm(
                "carrier", [True, True, True, False, False]
            )["per_task"],
            "successes": 3,
        },
        "learned_language_only": _arm(
            "learned_language_only", [True, True, False, False, False], language=True
        ),
        "correct_full": _arm(
            "correct_full", [True, True, True, True, True]
        ),
        "first_final": _arm(
            "first_final", [True, True, False, False, False]
        ),
        "same_task_other": _arm(
            "same_task_other", [True, True, True, True, False]
        ),
    }
    gate = {
        "language_baseline": {"fixed": True},
        "video_panel": {
            "correct_full": [5, 6, 7, 8],
            "first_final": [5, 6, 7, 8],
            "same_task_other": [0, 1, 2, 3],
        },
    }
    metrics = _success_metrics(arms, KEYS)
    checks, checkpoint = _g3_adapter_authority(arms, gate)
    assert metrics["carrier_successes_retained"] == 3
    assert metrics["full_over_language"] == 3
    assert metrics["full_over_first_final"] == 3
    assert metrics["same_task_retention"] == 0.8
    assert metrics["suite_successes"]["libero_goal"] == 1
    assert all(checks.values())
    assert checkpoint["compiler_macro"] == 5
