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
from ember.pi05_source_checkpoint import canonical_hash
from ember.writer.evaluation_cache import (
    WRITER_LORA_REQUEST_ORDER,
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
    WRITER_EPISODE_EVIDENCE_V5_1,
    expected_writer_episode_evidence,
    validate_writer_episode_evidence,
)
from ember.writer.evaluation_runtime import FrozenCachedWriterTaskAdapter
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
            "schema_version": WRITER_ADAPTER_SCHEMA,
            "kind": "as_writer",
            "writer_method": "as_writer",
            "arm": "as_writer_correct_video",
            "video_condition": "correct",
            "checkpoint": {
                "cursor": 300,
                "manifest_file_sha256": "3" * 64,
                "writer_state_sha256": "4" * 64,
            },
            "lora_contract_sha256": canonical_contract_sha256(lora),
            "video_schedule": {"seed": 7, "demo_count": 50},
            "task_video_mapping_sha256": "5" * 64,
            "task_video_mapping": [
                {
                    "suite": "libero_spatial",
                    "task_id": 1,
                    "language_global_task_id": 1,
                    "video_suite": "libero_spatial",
                    "video_task_id": 1,
                    "video_global_task_id": 1,
                    "video_split_role": "validation",
                }
            ],
            "pairing_sha256": "6" * 64,
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


def _populate_writer_cache(
    contract: dict,
    lora: SmolVLALoRAContract,
) -> None:
    for request in writer_cache_requests(contract):
        state = _state(float(request.ordinal))
        evidence = expected_writer_episode_evidence(
            contract["adapter"],
            suite=request.suite,
            task_id=request.task_id,
            init_state_id=request.init_state_id,
            lora_sha256=lora_state_sha256(state),
        )
        evidence["writer_generation_seconds"] = 0.1
        write_writer_cache_entry(
            contract,
            request,
            state=state,
            evidence=evidence,
            generation={"generator_worker_id": "0-r0"},
            lora_contract=lora,
        )
    invocation_id = "b" * 32
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
    finalize_writer_cache(
        contract,
        invocation_id=invocation_id,
        worker_ids=("0-r0",),
    )


def test_cache_identity_decouples_rollout_replicas(tmp_path: Path) -> None:
    first = _contract(tmp_path / "first", replicas=2)
    second = _contract(tmp_path / "second", replicas=6)
    assert (
        first["writer_lora_cache"]["identity_sha256"]
        == second["writer_lora_cache"]["identity_sha256"]
    )
    assert first["parallel"]["replicas_per_gpu"] != second["parallel"]["replicas_per_gpu"]


def test_writer_cache_deduplicates_repeated_visible_videos(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    contract["tasks"][0]["init_state_ids"] = list(range(50))
    lora = _lora_contract()
    contract["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        contract,
        root=tmp_path / "cache",
        generators_per_gpu=1,
        generation_batch_size=100,
        lora_parameter_count=lora.parameter_count,
        lora_tensor_count=lora.state_tensor_count,
    )
    requests = writer_cache_requests(contract)
    episode_requests = writer_cache_episode_request_map(contract)
    assert len(requests) == 32
    assert len(episode_requests) == 50
    assert episode_requests[("libero_spatial", 1, 1)] is episode_requests[
        ("libero_spatial", 1, 18)
    ]
    assert all(request.is_video_keyed for request in requests)


def test_legacy_per_state_cache_descriptor_remains_loadable(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "legacy")
    descriptor = contract["writer_lora_cache"]
    recipe = descriptor["generation_recipe"]
    recipe.pop("cache_key_algorithm")
    recipe.pop("episode_evidence_schema")
    recipe["request_order"] = WRITER_LORA_REQUEST_ORDER
    descriptor["identity"]["generation_recipe"] = copy.deepcopy(recipe)
    descriptor["identity_sha256"] = canonical_hash(descriptor["identity"])
    requests = writer_cache_requests(contract)
    assert [request.entry_id for request in requests] == [
        "libero_spatial_task_01_state_000",
        "libero_spatial_task_01_state_001",
        "libero_spatial_task_01_state_002",
    ]


def test_writer_generation_order_randomness_is_video_keyed(
    tmp_path: Path,
) -> None:
    adapter = _contract(tmp_path / "cache")["adapter"]
    first = expected_writer_episode_evidence(
        adapter,
        suite="libero_spatial",
        task_id=1,
        init_state_id=1,
        lora_sha256="7" * 64,
    )
    repeated = expected_writer_episode_evidence(
        adapter,
        suite="libero_spatial",
        task_id=1,
        init_state_id=18,
        lora_sha256="7" * 64,
    )
    assert first["schema_version"] == WRITER_EPISODE_EVIDENCE_V5_1
    assert first["teacher_demo_index"] == repeated["teacher_demo_index"]
    assert first["teacher_video_selection_seed"] != repeated[
        "teacher_video_selection_seed"
    ]
    assert first["teacher_video_order_seed"] == repeated[
        "teacher_video_order_seed"
    ]
    first["writer_generation_seconds"] = 0.1
    assert validate_writer_episode_evidence(
        adapter,
        first,
        suite="libero_spatial",
        task_id=1,
        init_state_id=1,
    )


def test_cached_runtime_reuses_one_lora_for_duplicate_video_aliases(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path / "cache")
    contract["writer_lora_execution"] = {"b_scale": 1.5}
    contract["tasks"][0]["init_state_ids"] = list(range(50))
    lora = _lora_contract()
    contract["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        contract,
        root=tmp_path / "cache",
        generators_per_gpu=1,
        generation_batch_size=100,
        lora_parameter_count=lora.parameter_count,
        lora_tensor_count=lora.state_tensor_count,
    )
    _populate_writer_cache(contract, lora)
    runtime = FrozenCachedWriterTaskAdapter.__new__(FrozenCachedWriterTaskAdapter)
    runtime.lora_contract = lora
    runtime.device = torch.device("cpu")
    runtime.evaluation_adapter = contract["adapter"]
    runtime._initialize_cache(contract)
    runtime.activate_cache()
    first = runtime.prepare_episode(
        suite="libero_spatial", task_id=1, init_state_id=1
    )
    repeated = runtime.prepare_episode(
        suite="libero_spatial", task_id=1, init_state_id=18
    )
    request = writer_cache_episode_request_map(contract)[
        ("libero_spatial", 1, 1)
    ]
    original, _ = load_writer_cache_entry(
        contract,
        request,
        lora_contract=lora,
        device=torch.device("cpu"),
    )
    assert first.state is repeated.state
    assert torch.equal(
        first.state["layer.lora_A.default.weight"],
        original["layer.lora_A.default.weight"],
    )
    assert torch.equal(
        first.state["layer.lora_B.default.weight"],
        original["layer.lora_B.default.weight"] * 1.5,
    )
    assert first.evidence["lora_sha256"] == repeated.evidence["lora_sha256"]
    assert first.evidence["teacher_video_selection_seed"] != repeated.evidence[
        "teacher_video_selection_seed"
    ]
    assert len(runtime._state_cache) == 1


def test_writer_cache_is_atomic_complete_and_loadable(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    lora = _lora_contract()
    requests = writer_cache_requests(contract)
    for request in requests:
        state = _state(float(request.ordinal))
        evidence = expected_writer_episode_evidence(
            contract["adapter"],
            suite=request.suite,
            task_id=request.task_id,
            init_state_id=request.init_state_id,
            lora_sha256=lora_state_sha256(state),
        )
        evidence["writer_generation_seconds"] = 0.1
        write_writer_cache_entry(
            contract,
            request,
            state=state,
            evidence=evidence,
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
