from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from ember.expert_manifold.inference import (
    EXPERT_MANIFOLD_ADAPTER_SCHEMA,
    EXPERT_MANIFOLD_WRITER_KIND,
    expected_expert_manifold_episode_evidence,
)
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    reference_demo_index,
    shuffled_frame_permutation,
    task_video_mapping,
    video_schedule_contract,
    video_selection_seed,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_eval_results import _per_task_rows
from ember.pi05_evaluation import SHARD_RESULT_SCHEMA, validate_shard_result


def _rows() -> list[dict]:
    return [
        {
            "suite": "libero_spatial",
            "task_id": 0,
            "split_role": "train",
            "language": "task zero",
            "init_state_id": state_id,
            "env_seed": 7,
            "policy_seed_root": 7,
            "policy_noise_seeds": [
                policy_noise_seed(7, "libero_spatial", 0, state_id, 0)
            ],
            "success": state_id == 0,
            "steps": 1,
            "wall_seconds": 0.1,
            "finished_at": 0.1,
        }
        for state_id in (0, 1)
    ]


def _writer_adapter(condition: str = "correct") -> dict:
    keys = tuple(
        (suite, 0)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    )
    roles = {key: "train" for key in keys}
    schedule, pairing = video_schedule_contract(
        seed=7,
        demo_count=50,
        sampling_mode="without_replacement",
    )
    return {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "arm": f"expert_manifold_hard_routed_{condition}",
        "video_condition": condition,
        "writer_asset": {
            "reference": "test:policy-effective:step2000:subspace96:hard1",
            "learned_parameter_count": 0,
            "expert_step": 2000,
            "expert_count": 24,
            "deployed_coefficient_support": 1,
        },
        "lora_contract": {"reference": "rank16:76tensors"},
        "video_schedule": schedule,
        "task_video_mapping_reference": "next_suite_v1",
        "task_video_mapping": list(task_video_mapping(keys, roles, condition)),
        "pairing_reference": pairing,
    }


def test_per_task_rows_summarizes_one_shot_teacher_videos() -> None:
    rows = _rows()
    for row, demo in zip(rows, (0, 1), strict=True):
        row["writer"] = {
            "condition": "correct",
            "teacher_demo_indices": [demo],
            "writer_generation_seconds": 0.25,
        }
    tasks = {
        ("libero_spatial", 0): {
            "split_role": "train",
            "language": "task zero",
        }
    }

    writer = _per_task_rows(rows, tasks)[0]["writer"]

    assert writer["videos_per_condition"] == 1
    assert writer["unique_teacher_videos"] == 2
    assert writer["teacher_demo_counts"] == {"0": 1, "1": 1}
    assert writer["unique_teacher_video_sets"] == 2
    assert writer["teacher_demo_set_counts"] == {"0": 1, "1": 1}
    assert writer["generation_wall_seconds"] == 0.5


def test_writer_video_schedule_and_wrong_map_are_order_independent() -> None:
    request = (7, "libero_spatial", 6, 0)
    assert video_selection_seed(
        *request, sampling_mode="without_replacement"
    ) == video_selection_seed(*request, sampling_mode="without_replacement")
    assert (
        0
        <= reference_demo_index(
            *request,
            demo_count=50,
            sampling_mode="without_replacement",
        )
        < 50
    )
    keys = (
        ("libero_spatial", 1),
        ("libero_spatial", 3),
        ("libero_object", 1),
        ("libero_object", 3),
        ("libero_goal", 3),
        ("libero_goal", 6),
        ("libero_10", 1),
        ("libero_10", 2),
    )
    roles = {key: "validation" for key in keys}
    forward = task_video_mapping(keys, roles, "cross_suite_wrong")
    reverse = task_video_mapping(tuple(reversed(keys)), roles, "cross_suite_wrong")
    shuffled = task_video_mapping(keys, roles, "shuffled")
    shuffled_keep_first = task_video_mapping(keys, roles, "shuffled_keep_first")
    reversed_video = task_video_mapping(keys, roles, "reversed")
    same_task_other = task_video_mapping(keys, roles, "same_task_other")
    assert forward == reverse
    assert shuffled == shuffled_keep_first == reversed_video == same_task_other
    assert all(row["suite"] == row["video_suite"] for row in shuffled)
    assert len({row["video_global_task_id"] for row in forward}) == len(keys)
    assert all(row["suite"] != row["video_suite"] for row in forward)
    assert all(row["language_split_role"] == row["video_split_role"] for row in forward)
    by_key = {(row["suite"], row["task_id"]): row for row in forward}
    assert (
        by_key[("libero_spatial", 1)]["video_suite"],
        by_key[("libero_spatial", 1)]["video_task_id"],
    ) == ("libero_object", 1)
    assert (
        by_key[("libero_goal", 6)]["video_suite"],
        by_key[("libero_goal", 6)]["video_task_id"],
    ) == ("libero_10", 2)


def test_shuffled_keep_first_changes_only_the_anchor_position() -> None:
    shuffled = shuffled_frame_permutation(20, 7, keep_first=False)
    keep_first = shuffled_frame_permutation(20, 7, keep_first=True)
    assert keep_first[0].item() == 0
    assert keep_first[1:].tolist() == [
        index for index in shuffled.tolist() if index != 0
    ]
    assert sorted(keep_first.tolist()) == list(range(20))


def test_same_task_other_changes_only_the_teacher_demo() -> None:
    correct = expected_expert_manifold_episode_evidence(
        _writer_adapter("correct"),
        suite="libero_spatial",
        task_id=0,
        init_state_id=4,
        lora_reference="correct",
    )
    other = expected_expert_manifold_episode_evidence(
        _writer_adapter("same_task_other"),
        suite="libero_spatial",
        task_id=0,
        init_state_id=4,
        lora_reference="other",
    )
    assert other["teacher_demo_indices"] == [
        (value + SAME_TASK_OTHER_OFFSET) % 50
        for value in correct["teacher_demo_indices"]
    ]
    assert other["teacher_reference_demo_indices"] == correct["teacher_demo_indices"]
    assert other["video_global_task_id"] == correct["video_global_task_id"]
    assert other["teacher_video_order_seeds"] == correct["teacher_video_order_seeds"]


def test_writer_row_contract_recomputes_video_schedule_and_mapping(
    tmp_path: Path,
) -> None:
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "mode": "smoke",
        "arm": "expert_manifold_hard_routed_correct",
        "role": "test",
        "output_dir": str(tmp_path),
        "content_hash_policy": "disabled_by_owner",
        "model": {"optimizer_step": 1},
        "normalization": {"bytes": 1},
        "tokenizer": {"bytes": 1},
        "rng": {"inference_seed": 7},
        "policy": {"replan_steps": 5},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 0,
                "split_role": "train",
                "language": "task zero",
                "init_state_ids": [0, 1],
            }
        ],
        "adapter": _writer_adapter(),
        "contract_reference": f"{RUN_CONTRACT_SCHEMA}:test-contract",
    }
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    payload = {
        "schema_version": SHARD_RESULT_SCHEMA,
        "contract_reference": contract["contract_reference"],
        "job_id": shard.job_id,
        "shard": asdict(shard),
        "producer": {"worker_id": "0-r0", "claim_token": "a" * 32, "attempt": 1},
        "started_unix": 10.0,
        "finished_unix": 12.0,
        "rows": _rows(),
    }
    for row in payload["rows"]:
        row["writer"] = {
            **expected_expert_manifold_episode_evidence(
                contract["adapter"],
                suite=row["suite"],
                task_id=row["task_id"],
                init_state_id=row["init_state_id"],
                lora_reference=f"row:{row['init_state_id']}",
            ),
            "writer_generation_seconds": 0.25,
        }
    assert len(validate_shard_result(payload, contract=contract, shard=shard)) == 2
    payload["rows"][0]["writer"]["teacher_demo_indices"][0] += 1
    with pytest.raises(Pi05EvaluationError, match="row contract changed"):
        validate_shard_result(payload, contract=contract, shard=shard)
