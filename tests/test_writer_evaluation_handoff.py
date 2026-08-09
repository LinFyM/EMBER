from __future__ import annotations

import json
from pathlib import Path

from ember.pi05_eval_results import _worker_lifecycle, _writer_generation_summary


def test_writer_generator_lifecycle_proves_resident_policy_handoff(
    tmp_path: Path,
) -> None:
    contract = {
        "parallel": {"replicas_per_gpu": 1, "physical_gpu_ids": [0]},
        "contract_reference": "ember_pi05_target_eval_launch_v2:test",
    }
    worker_id = "0-r0"
    invocation_id = "b" * 32
    common = {
        "worker_id": worker_id,
        "pid": 123,
        "invocation_id": invocation_id,
        "contract_reference": contract["contract_reference"],
    }
    events = (
        {"event": "process_started", "unix": 1.0, **common},
        {
            "event": "ready",
            "unix": 3.0,
            "physical_gpu": 0,
            "gpu_uuid": "GPU-0",
            "gpu_name": "NVIDIA A40",
            "replica": 0,
            "numa_node": 0,
            "cpu_affinity": [0],
            "model_load_seconds": 2.0,
            "writer_generator": True,
            **common,
        },
        {
            "event": "writer_generation_finished",
            "unix": 7.0,
            "assigned_entries": 2,
            "generated_entries": 2,
            "reused_entries": 0,
            "generated_batches": 1,
            "generation_batch_size": 2,
            "generation_wall_seconds": 4.0,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 120,
            "post_release_allocated_bytes": 40,
            "post_release_reserved_bytes": 50,
            "source_policy_reused_for_rollout": True,
            "writer_modules_released": True,
            "redundant_writer_forwards": 0,
            "batch_shape_bf16_roundoff_accepted": True,
            "batches": [
                {
                    "batch_ordinal": 0,
                    "entry_ids": ["entry-0", "entry-1"],
                    "batch_size": 2,
                    "raw_frame_counts": [101, 120],
                    "sampled_frame_counts": [21, 25],
                    "wall_seconds": 4.0,
                }
            ],
            **common,
        },
        {
            "event": "rollout_ready_with_retained_policy",
            "unix": 8.0,
            "source_policy_reloaded": False,
            **common,
        },
        {
            "event": "finished",
            "unix": 10.0,
            "completed_shards": 1,
            "adopted_shards": 0,
            **common,
        },
    )
    path = tmp_path / "workers" / f"{worker_id}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    observed = _worker_lifecycle(
        tmp_path,
        contract,
        invocation_id=invocation_id,
        worker_id=worker_id,
    )
    assert observed["writer_generator"] is True
    assert observed["gpu_name"] == "NVIDIA A40"
    assert observed["source_policy_reloaded"] is False
    assert observed["rollout_ready_unix"] == 8.0
    assert observed["writer_generation"]["generated_entries"] == 2
    assert observed["writer_generation"]["redundant_writer_forwards"] == 0
    summary = _writer_generation_summary((observed,))
    assert summary is not None
    assert summary["gpu_names"] == ["NVIDIA A40"]
    assert summary["all_source_policies_not_reloaded"] is True
    assert summary["batch_shape_bf16_roundoff_accepted"] is True
    assert summary["max_observed_forward_batch_size"] == 2
    assert summary["max_sampled_video_frames"] == 25


def test_writer_generation_summary_allows_full_cache_reuse() -> None:
    summary = _writer_generation_summary(
        (
            {
                "gpu_name": "NVIDIA A40",
                "source_policy_reloaded": False,
                "writer_generation": {
                    "assigned_entries": 8,
                    "generated_entries": 0,
                    "reused_entries": 8,
                    "generated_batches": 0,
                    "generation_batch_size": 32,
                    "generation_wall_seconds": 0.1,
                    "peak_allocated_bytes": 100,
                    "peak_reserved_bytes": 120,
                    "post_release_allocated_bytes": 40,
                    "post_release_reserved_bytes": 50,
                    "source_policy_reused_for_rollout": True,
                    "writer_modules_released": True,
                    "redundant_writer_forwards": 0,
                    "batch_shape_bf16_roundoff_accepted": True,
                    "batches": [],
                },
            },
        )
    )
    assert summary is not None
    assert summary["reused_entries"] == 8
    assert summary["max_observed_forward_batch_size"] == 0
    assert summary["max_sampled_video_frames"] == 0
