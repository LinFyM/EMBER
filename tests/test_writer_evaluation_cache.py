from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ember.lora import LoRATarget, SmolVLALoRAContract
from ember.writer.evaluation_cache import (
    assigned_writer_cache_requests,
    build_writer_lora_cache_descriptor,
    finalize_writer_cache,
    load_writer_cache_entry,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_episode_request_map,
    writer_cache_manifest_is_ready,
    writer_cache_requests,
)
from ember.writer.inference import (
    WRITER_ADAPTER_SCHEMA,
    expected_writer_episode_evidence,
)
from ember.writer.model import WriterModelError


def _lora_contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("layer", 3, 4),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=7,
    )


def _contract(root: Path, *, replicas: int = 2, state_count: int = 3) -> dict:
    lora = _lora_contract()
    mapping = [
        {
            "suite": "libero_spatial",
            "task_id": 1,
            "language_global_task_id": 1,
            "language_split_role": "validation",
            "video_suite": "libero_spatial",
            "video_task_id": 1,
            "video_global_task_id": 1,
            "video_split_role": "validation",
        }
    ]
    contract = {
        "adapter": {
            "schema_version": WRITER_ADAPTER_SCHEMA,
            "kind": "as_writer",
            "writer_method": "as_writer",
            "arm": "as_writer_correct_video",
            "video_condition": "correct",
            "checkpoint": {"cursor": 300, "reference": "run:300"},
            "lora_contract": {"reference": "test:2tensors:14parameters"},
            "video_schedule": {
                "seed": 7,
                "demo_count": 50,
                "videos_per_condition": 4,
                "sampling_mode": "without_replacement",
            },
            "task_video_mapping_reference": "identity_v1",
            "task_video_mapping": mapping,
            "pairing_reference": "paired_k4_v1",
        },
        "model": {"optimizer_step": 1000},
        "tokenizer": {"path": "/tokenizer.model"},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 1,
                "init_state_ids": list(range(state_count)),
            }
        ],
        "policy": {"num_inference_steps": 10},
        "rng": {"inference_seed": 7},
        "parallel": {"physical_gpu_count": 1, "replicas_per_gpu": replicas},
    }
    contract["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        contract,
        root=root,
        generators_per_gpu=1,
        generation_batch_size=2,
        lora_parameter_count=lora.parameter_count,
        lora_tensor_count=lora.state_tensor_count,
    )
    return contract


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "layer.lora_A.default.weight": torch.full((2, 3), value, dtype=torch.bfloat16),
        "layer.lora_B.default.weight": torch.full((4, 2), value + 1, dtype=torch.bfloat16),
    }


def _populate(contract: dict, lora: SmolVLALoRAContract) -> None:
    for request in writer_cache_requests(contract):
        evidence = expected_writer_episode_evidence(
            contract["adapter"],
            suite=request.suite,
            task_id=request.task_id,
            init_state_id=request.init_state_id,
            lora_reference=f"run:300:{request.entry_id}",
        )
        evidence["writer_generation_seconds"] = 0.1
        write_writer_cache_entry(
            contract,
            request,
            state=_state(float(request.ordinal)),
            evidence=evidence,
            generation={"generator_worker_id": "0-r0"},
            lora_contract=lora,
        )
    invocation = "b" * 32
    write_generator_marker(
        contract,
        invocation_id=invocation,
        worker_id="0-r0",
        generator_index=0,
        summary={"source_policy_reused_for_rollout": True, "writer_modules_released": True},
    )
    finalize_writer_cache(contract, invocation_id=invocation, worker_ids=("0-r0",))


def test_cache_identity_ignores_rollout_replica_count(tmp_path: Path) -> None:
    first = _contract(tmp_path / "first", replicas=2)
    second = _contract(tmp_path / "second", replicas=6)
    assert first["writer_lora_cache"]["identity"] == second["writer_lora_cache"]["identity"]


def test_k4_cache_retains_one_entry_per_episode(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache", state_count=50)
    assert len(writer_cache_requests(contract)) == 50
    assert len(writer_cache_episode_request_map(contract)) == 50
    assert len({request.entry_id for request in writer_cache_requests(contract)}) == 50


def test_writer_cache_is_atomic_complete_and_loadable_without_hashes(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    lora = _lora_contract()
    _populate(contract, lora)
    requests = writer_cache_requests(contract)
    assert assigned_writer_cache_requests(contract, generator_index=0) == requests
    assert writer_cache_manifest_is_ready(contract)
    manifest = validate_writer_cache_manifest(contract, verify_entry_files=True)
    assert manifest["entry_ids"] == [request.entry_id for request in requests]
    state, evidence = load_writer_cache_entry(
        contract, requests[1], lora_contract=lora, device=torch.device("cpu")
    )
    assert state["layer.lora_A.default.weight"].shape == (2, 3)
    assert evidence["lora_reference"].endswith(requests[1].entry_id)
    assert "sha256" not in str(manifest).lower()


def test_writer_cache_rejects_descriptor_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    changed = copy.deepcopy(contract)
    changed["tasks"][0]["init_state_ids"] = [0, 2]
    with pytest.raises(WriterModelError, match="descriptor changed"):
        assigned_writer_cache_requests(changed, generator_index=0)
