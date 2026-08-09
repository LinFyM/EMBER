from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ember.pi05_eval_results import _worker_lifecycle
from ember.writer.errors import WriterModelError
from ember.writer.evaluation_runtime import (
    _state_max_abs_difference,
    _warmstart_reproduction_required,
)


def test_v6_prior_reproduction_comparison_is_exact_and_fail_closed() -> None:
    staged = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([[3.0]], dtype=torch.bfloat16),
    }
    direct = {
        "a": torch.tensor([1.0, 2.000004]),
        "b": torch.tensor([[3.0]], dtype=torch.bfloat16),
    }
    assert _state_max_abs_difference(staged, direct) == pytest.approx(
        4.0531158447265625e-6
    )
    with pytest.raises(WriterModelError, match="state names changed"):
        _state_max_abs_difference(staged, {"a": direct["a"]})
    with pytest.raises(WriterModelError, match="nonfinite"):
        _state_max_abs_difference(
            staged,
            {**direct, "a": torch.tensor([1.0, float("nan")])},
        )


def test_only_historical_correct_smoke_requires_direct_v6_comparison() -> None:
    contract = {
        "mode": "smoke",
        "adapter": {
            "kind": "expert_manifold_writer",
            "video_condition": "correct",
            "writer_asset": {"kind": "historical_v6_macro400_load_only"},
        },
    }
    assert _warmstart_reproduction_required(contract)
    contract["mode"] = "screen"
    assert not _warmstart_reproduction_required(contract)
    contract["mode"] = "smoke"
    contract["adapter"]["video_condition"] = "reversed"
    assert not _warmstart_reproduction_required(contract)
    assert not _warmstart_reproduction_required({"mode": "smoke", "adapter": []})


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
    assert observed["rollout_ready_unix"] == 8.0
    assert observed["writer_generation"]["generated_entries"] == 2
