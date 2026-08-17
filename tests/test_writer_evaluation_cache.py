from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ember.eval_adapters import (
    DYNAMIC_K_WRITER_KIND,
    expected_writer_episode,
    validate_writer_episode,
    writer_episode_schema,
)
from ember.lora import LoRATarget, SmolVLALoRAContract
from ember.writer.evaluation_cache import (
    DYNAMIC_K_WRITER_LORA_CACHE_SCHEMA,
    DYNAMIC_K_WRITER_LORA_VIDEO_KEY_ALGORITHM,
    DYNAMIC_K_WRITER_LORA_VIDEO_SET_KEY_ALGORITHM,
    WRITER_LORA_ASSIGNMENT,
    WRITER_LORA_LEGACY_ASSIGNMENT,
    assigned_writer_cache_batches,
    assigned_writer_cache_requests,
    build_writer_lora_cache_descriptor,
    finalize_writer_cache,
    load_writer_cache_entry,
    stage_writer_lora_states_to_cpu,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_episode_request_map,
    writer_cache_entry_is_complete,
    writer_cache_manifest_is_ready,
    writer_cache_requests,
)
from ember.writer.errors import WriterModelError
from ember.writer.evaluation import (
    DYNAMIC_K_ADAPTER_SCHEMA,
    DYNAMIC_K_EPISODE_SCHEMA,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def _lora_contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("layer", 3, 4),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=7,
    )


def _lora_storage() -> dict:
    return {
        "tensor_count": 2,
        "parameter_count": 14,
        "tensor_bytes": 28,
        "dtype_tensor_counts": {"BF16": 2},
        "dtype_parameter_counts": {"BF16": 14},
        "dtype_by_name": {
            "layer.lora_A.default.weight": "BF16",
            "layer.lora_B.default.weight": "BF16",
        },
    }


def _contract(
    root: Path,
    *,
    replicas: int = 2,
    state_count: int = 3,
    physical_gpu_count: int = 1,
    generators_per_gpu: int = 1,
    generation_batch_size: int = 2,
) -> dict:
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
            "schema_version": DYNAMIC_K_ADAPTER_SCHEMA,
            "kind": DYNAMIC_K_WRITER_KIND,
            "config": {
                "schema": (
                    "ember_pi05_layer_matched_memory_program_compiler_writer_v1"
                )
            },
            "arm": "layer_matched_memory_program_compiler_correct",
            "video_condition": "correct",
            "writer_asset": {
                "reference": "dynamic-k:m25:rank16",
                "kind": "layer_matched_memory_program_compiler_macro_checkpoint",
                "method_macro": 25,
                "writer_parameter_count": 123,
                "generated_lora_tensor_count": 2,
            },
            "lora_contract": {"reference": "test:2tensors:14parameters"},
            "video_schedule": {
                "seed": 7,
                "demo_count": 50,
                "videos_per_condition": 1,
                "sampling_mode": "without_replacement",
                "backbone_total_frames_per_condition": 64,
            },
            "information_wall": {"evaluation_k": 1},
            "task_video_mapping_reference": "identity_k1_v1",
            "task_video_mapping": mapping,
            "pairing_reference": "ember_pi05_dynamic_k_one_shot_pairing_v1",
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
        "git": {"commit": "a" * 40},
        "parallel": {
            "physical_gpu_count": physical_gpu_count,
            "replicas_per_gpu": replicas,
        },
    }
    contract["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        contract,
        root=root,
        generators_per_gpu=generators_per_gpu,
        generation_batch_size=generation_batch_size,
        lora_parameter_count=lora.parameter_count,
        lora_tensor_count=lora.state_tensor_count,
        lora_storage_per_entry=_lora_storage(),
    )
    return contract


def _state(value: float) -> dict[str, torch.Tensor]:
    return {
        "layer.lora_A.default.weight": torch.full((2, 3), value, dtype=torch.bfloat16),
        "layer.lora_B.default.weight": torch.full(
            (4, 2), value + 1, dtype=torch.bfloat16
        ),
    }


def _populate(contract: dict, lora: SmolVLALoRAContract) -> None:
    for request in writer_cache_requests(contract):
        evidence = expected_writer_episode(
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
        summary={
            "source_policy_reused_for_rollout": True,
            "writer_modules_released": True,
        },
    )
    finalize_writer_cache(contract, invocation_id=invocation, worker_ids=("0-r0",))


def test_cache_identity_ignores_rollout_replica_count(tmp_path: Path) -> None:
    first = _contract(tmp_path / "first", replicas=2)
    second = _contract(tmp_path / "second", replicas=6)
    assert (
        first["writer_lora_cache"]["identity"]
        == second["writer_lora_cache"]["identity"]
    )


def test_one_shot_cache_retains_one_entry_per_episode(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache", state_count=50)
    assert len(writer_cache_requests(contract)) == 50
    assert len(writer_cache_episode_request_map(contract)) == 50
    assert len({request.entry_id for request in writer_cache_requests(contract)}) == 50


def test_active_cache_descriptor_uses_explicit_global_batch_first_assignment(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path / "cache")
    recipe = contract["writer_lora_cache"]["generation_recipe"]
    assert recipe["assignment"] == WRITER_LORA_ASSIGNMENT
    assert recipe["assignment"] != WRITER_LORA_LEGACY_ASSIGNMENT


def test_global_batch_membership_and_position_are_worker_count_invariant(
    tmp_path: Path,
) -> None:
    contracts = (
        _contract(
            tmp_path / "two-workers",
            replicas=2,
            state_count=23,
            physical_gpu_count=2,
            generation_batch_size=4,
        ),
        _contract(
            tmp_path / "six-workers",
            replicas=3,
            state_count=23,
            physical_gpu_count=2,
            generators_per_gpu=3,
            generation_batch_size=4,
        ),
    )
    observed = []
    for contract in contracts:
        worker_count = contract["writer_lora_cache"]["generation_recipe"][
            "generator_worker_count"
        ]
        worker_batches = tuple(
            assigned_writer_cache_batches(contract, generator_index=worker)
            for worker in range(worker_count)
        )
        assert all(
            batch.ordinal % worker_count == worker
            for worker, batches in enumerate(worker_batches)
            for batch in batches
        )
        batches = sorted(
            (batch for batches in worker_batches for batch in batches),
            key=lambda batch: batch.ordinal,
        )
        observed.append(
            tuple(
                (
                    batch.ordinal,
                    tuple(request.ordinal for request in batch.requests),
                    batch.canonical_global,
                )
                for batch in batches
            )
        )
    assert observed[0] == observed[1]
    assert observed[0] == (
        (0, (0, 1, 2, 3), True),
        (1, (4, 5, 6, 7), True),
        (2, (8, 9, 10, 11), True),
        (3, (12, 13, 14, 15), True),
        (4, (16, 17, 18, 19), True),
        (5, (20, 21, 22), True),
    )


def test_fresh_correct400_batch8_has_exactly_fifty_global_batches(
    tmp_path: Path,
) -> None:
    contract = _contract(
        tmp_path / "cache",
        replicas=3,
        state_count=400,
        physical_gpu_count=4,
        generators_per_gpu=3,
        generation_batch_size=8,
    )
    batches = tuple(
        batch
        for worker in range(12)
        for batch in assigned_writer_cache_batches(contract, generator_index=worker)
    )
    assert len(batches) == 50
    assert sorted(batch.ordinal for batch in batches) == list(range(50))
    assert all(len(batch.requests) == 8 for batch in batches)
    assert sorted(
        request.ordinal for batch in batches for request in batch.requests
    ) == list(range(400))


def test_legacy_assignment_keeps_request_modulo_worker_semantics(
    tmp_path: Path,
) -> None:
    contract = _contract(
        tmp_path / "cache",
        replicas=2,
        state_count=10,
        physical_gpu_count=2,
        generators_per_gpu=2,
        generation_batch_size=2,
    )
    descriptor = contract["writer_lora_cache"]
    descriptor["generation_recipe"]["assignment"] = WRITER_LORA_LEGACY_ASSIGNMENT
    descriptor["identity"]["generation_recipe"][
        "assignment"
    ] = WRITER_LORA_LEGACY_ASSIGNMENT

    batches = assigned_writer_cache_batches(contract, generator_index=1)

    assert tuple(
        tuple(request.ordinal for request in batch.requests) for batch in batches
    ) == ((1, 5), (9,))
    assert all(batch.canonical_global is False for batch in batches)

    changed = copy.deepcopy(contract)
    changed["writer_lora_cache"]["generation_recipe"]["assignment"] = "unknown"
    changed["writer_lora_cache"]["identity"]["generation_recipe"][
        "assignment"
    ] = "unknown"
    with pytest.raises(WriterModelError, match="assignment algorithm is unsupported"):
        assigned_writer_cache_batches(changed, generator_index=1)


def test_cache_staging_preserves_native_tensor_dtypes() -> None:
    staged = stage_writer_lora_states_to_cpu((_state(1.0), _state(2.0)))
    assert len(staged) == 2
    assert all(
        value.device.type == "cpu" for state in staged for value in state.values()
    )
    assert all(
        value.dtype == torch.bfloat16 for state in staged for value in state.values()
    )


def test_cache_staging_rejects_nonfinite_batch_without_per_tensor_cuda_sync() -> None:
    state = _state(1.0)
    state["layer.lora_A.default.weight"][0, 0] = torch.nan
    with pytest.raises(WriterModelError, match="non-finite LoRA"):
        stage_writer_lora_states_to_cpu((state,))


def test_dynamic_k_cache_dispatches_k1_episode_evidence(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "legacy")
    adapter = contract["adapter"]
    adapter.update(
        {
            "schema_version": DYNAMIC_K_ADAPTER_SCHEMA,
            "kind": DYNAMIC_K_WRITER_KIND,
            "arm": "layer_matched_memory_program_compiler_correct",
            "config": {
                "schema": "ember_pi05_layer_matched_memory_program_compiler_writer_v1"
            },
            "writer_asset": {
                "reference": "dynamic-k:m25:rank16",
                "kind": "layer_matched_memory_program_compiler_macro_checkpoint",
                "method_macro": 50,
                "writer_parameter_count": 123,
                "generated_lora_tensor_count": 2,
            },
            "lora_contract": {"reference": "rank8:test:2tensors:14parameters"},
            "task_video_mapping_reference": "identity_k1_v1",
            "pairing_reference": "ember_pi05_dynamic_k_one_shot_pairing_v1",
        }
    )
    adapter["video_schedule"]["backbone_total_frames_per_condition"] = 64
    contract["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        contract,
        root=tmp_path / "dynamic",
        generators_per_gpu=1,
        generation_batch_size=2,
        lora_parameter_count=14,
        lora_tensor_count=2,
        lora_storage_per_entry=_lora_storage(),
    )
    row = expected_writer_episode(
        adapter,
        suite="libero_spatial",
        task_id=1,
        init_state_id=2,
        lora_reference="dynamic-k:episode:2",
    )
    row["writer_generation_seconds"] = 0.1
    assert row["condition_video_offsets"] == [0, 1]
    assert row["backbone_total_frames_per_condition"] == 64
    assert writer_episode_schema(adapter) == DYNAMIC_K_EPISODE_SCHEMA
    assert validate_writer_episode(
        adapter,
        row,
        suite="libero_spatial",
        task_id=1,
        init_state_id=2,
    )
    assert contract["writer_lora_cache"]["generation_recipe"][
        "episode_evidence_schema"
    ] == DYNAMIC_K_EPISODE_SCHEMA
    assert contract["writer_lora_cache"]["schema_version"] == (
        DYNAMIC_K_WRITER_LORA_CACHE_SCHEMA
    )
    assert contract["writer_lora_cache"]["generation_recipe"][
        "cache_key_algorithm"
    ] == DYNAMIC_K_WRITER_LORA_VIDEO_KEY_ALGORITHM


def test_dynamic_k_cache_dispatches_nested_k4_video_sets(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "dynamic-k4")
    adapter = contract["adapter"]
    adapter.update(
        {
            "schema_version": DYNAMIC_K_ADAPTER_SCHEMA,
            "kind": DYNAMIC_K_WRITER_KIND,
            "config": {
                "schema": "ember_pi05_layer_matched_memory_program_compiler_writer_v1"
            },
            "information_wall": {"evaluation_k": 4},
        }
    )
    adapter["video_schedule"].update(
        {
            "videos_per_condition": 4,
            "backbone_total_frames_per_condition": 64,
        }
    )
    descriptor = build_writer_lora_cache_descriptor(
        contract,
        root=tmp_path / "dynamic-k4-cache",
        generators_per_gpu=1,
        generation_batch_size=2,
        lora_parameter_count=14,
        lora_tensor_count=2,
        lora_storage_per_entry=_lora_storage(),
    )
    assert descriptor["generation_recipe"]["cache_key_algorithm"] == (
        DYNAMIC_K_WRITER_LORA_VIDEO_SET_KEY_ALGORITHM
    )


def test_writer_cache_is_atomic_complete_and_loadable_without_hashes(
    tmp_path: Path,
) -> None:
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
    assert state["layer.lora_A.default.weight"].dtype == torch.bfloat16
    assert evidence["lora_reference"].endswith(requests[1].entry_id)
    assert "sha256" not in str(manifest).lower()


def test_writer_cache_rejects_precision_widening(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    lora = _lora_contract()
    request = writer_cache_requests(contract)[0]
    evidence = expected_writer_episode(
        contract["adapter"],
        suite=request.suite,
        task_id=request.task_id,
        init_state_id=request.init_state_id,
        lora_reference="run:native-dtype",
    )
    evidence["writer_generation_seconds"] = 0.1
    with pytest.raises(WriterModelError, match="native storage changed"):
        write_writer_cache_entry(
            contract,
            request,
            state={name: value.float() for name, value in _state(1.0).items()},
            evidence=evidence,
            generation={"generator_worker_id": "0-r0"},
            lora_contract=lora,
        )


def test_writer_cache_load_rejects_storage_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache", state_count=1)
    lora = _lora_contract()
    _populate(contract, lora)
    request = writer_cache_requests(contract)[0]
    root = tmp_path / "cache" / "entries" / request.entry_id
    lora_path = root / "lora.safetensors"
    save_file(
        {name: value.float() for name, value in _state(1.0).items()},
        str(lora_path),
    )
    record = read_json(root / "entry.json")
    record["lora_file"]["bytes"] = lora_path.stat().st_size
    write_json_atomic(root / "entry.json", record)
    with pytest.raises(WriterModelError, match="loaded LoRA storage changed"):
        load_writer_cache_entry(
            contract, request, lora_contract=lora, device=torch.device("cpu")
        )


def test_writer_cache_rejects_descriptor_drift(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "cache")
    changed = copy.deepcopy(contract)
    changed["tasks"][0]["init_state_ids"] = [0, 2]
    with pytest.raises(WriterModelError, match="descriptor changed"):
        assigned_writer_cache_requests(changed, generator_index=0)


def test_partial_cache_entry_cannot_cross_video_conditions(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    correct = _contract(root, state_count=1)
    lora = _lora_contract()
    request = writer_cache_requests(correct)[0]
    evidence = expected_writer_episode(
        correct["adapter"],
        suite=request.suite,
        task_id=request.task_id,
        init_state_id=request.init_state_id,
        lora_reference="run:correct-partial",
    )
    evidence["writer_generation_seconds"] = 0.1
    write_writer_cache_entry(
        correct,
        request,
        state=_state(1.0),
        evidence=evidence,
        generation={"generator_worker_id": "0-r0"},
        lora_contract=lora,
    )

    wrong = copy.deepcopy(correct)
    wrong["adapter"]["video_condition"] = "wrong"
    wrong["writer_lora_cache"] = build_writer_lora_cache_descriptor(
        wrong,
        root=root,
        generators_per_gpu=1,
        generation_batch_size=2,
        lora_parameter_count=lora.parameter_count,
        lora_tensor_count=lora.state_tensor_count,
        lora_storage_per_entry=_lora_storage(),
    )
    with pytest.raises(WriterModelError, match="cache entry changed"):
        writer_cache_entry_is_complete(wrong, writer_cache_requests(wrong)[0])
