from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ember.lora import (
    LoRATarget,
    SmolVLALoRAContract,
    canonical_contract_sha256,
    lora_state_sha256,
)
from ember.writer.evaluation_cache import (
    assigned_writer_cache_requests,
    build_writer_lora_cache_descriptor,
    finalize_writer_cache,
    load_writer_cache_entry,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_manifest_is_ready,
    writer_cache_requests,
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


def _contract(root: Path, *, replicas: int = 2) -> dict:
    lora = _lora_contract()
    contract = {
        "adapter": {
            "kind": "as_writer",
            "arm": "as_writer_correct_video",
            "lora_contract_sha256": canonical_contract_sha256(lora),
        },
        "model": {"checkpoint_manifest_sha256": "1" * 64},
        "tokenizer": {"sha256": "2" * 64},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 1,
                "init_state_ids": [0, 1, 2],
            }
        ],
        "policy": {"num_inference_steps": 10},
        "rng": {"inference_seed": 7},
        "parallel": {
            "physical_gpu_count": 1,
            "replicas_per_gpu": replicas,
        },
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
        "layer.lora_A.default.weight": torch.full(
            (2, 3), value, dtype=torch.bfloat16
        ),
        "layer.lora_B.default.weight": torch.full(
            (4, 2), value + 1, dtype=torch.bfloat16
        ),
    }


def test_cache_identity_decouples_rollout_replicas(tmp_path: Path) -> None:
    first = _contract(tmp_path / "first", replicas=2)
    second = _contract(tmp_path / "second", replicas=6)
    assert (
        first["writer_lora_cache"]["identity_sha256"]
        == second["writer_lora_cache"]["identity_sha256"]
    )
    assert first["parallel"]["replicas_per_gpu"] != second["parallel"]["replicas_per_gpu"]


def test_writer_cache_is_atomic_complete_and_loadable(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    lora = _lora_contract()
    requests = writer_cache_requests(contract)
    for request in requests:
        state = _state(float(request.ordinal))
        write_writer_cache_entry(
            contract,
            request,
            state=state,
            evidence={
                "lora_sha256": lora_state_sha256(state),
                "writer_generation_seconds": 0.1,
            },
            generation={
                "generator_worker_id": "0-r0",
                "batch_ordinal": request.ordinal // 2,
            },
            lora_contract=lora,
        )
    assigned = assigned_writer_cache_requests(contract, generator_index=0)
    assert assigned == requests
    invocation_id = "a" * 32
    write_generator_marker(
        contract,
        invocation_id=invocation_id,
        worker_id="0-r0",
        generator_index=0,
        summary={
            "source_policy_reused_for_rollout": True,
            "writer_modules_released": True,
        },
    )
    assert not writer_cache_manifest_is_ready(contract)
    finalize_writer_cache(
        contract,
        invocation_id=invocation_id,
        worker_ids=("0-r0",),
    )
    assert writer_cache_manifest_is_ready(contract)
    assert (
        validate_writer_cache_manifest(contract, verify_entry_files=True)[
            "entry_ids"
        ]
        == [request.entry_id for request in requests]
    )
    state, evidence = load_writer_cache_entry(
        contract,
        requests[1],
        lora_contract=lora,
        device=torch.device("cpu"),
    )
    assert lora_state_sha256(state) == evidence["lora_sha256"]


def test_writer_cache_rejects_descriptor_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    changed = copy.deepcopy(contract)
    changed["tasks"][0]["init_state_ids"] = [0, 2]
    with pytest.raises(WriterModelError, match="descriptor changed"):
        writer_cache_requests(changed)
        assigned_writer_cache_requests(changed, generator_index=0)
