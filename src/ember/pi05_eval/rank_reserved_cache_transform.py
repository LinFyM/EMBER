"""GPU cache population for the one-time rank14 compiler-only diagnostic."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from ember.eval_adapters import expected_writer_episode, validate_writer_episode
from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_CANONICAL_CONFIG,
    RANK_RESERVED_EPISODE_SCHEMA,
    load_rank_reserved_config,
)
from ember.expert_manifold.v6_prior_contract import authority_path
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.rank_reserved_compiler_diagnostic import (
    COMPILER_DIAGNOSTIC_ENTRY_SCHEMA,
    COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
    COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE,
    COMPILER_DIAGNOSTIC_SOURCE_ROOT,
    COMPILER_DIAGNOSTIC_TRANSFORM,
    COMPILER_DIAGNOSTIC_TRANSFORM_SCHEMA,
    compiler_diagnostic_output_path,
    load_compiler_diagnostic_source_contract,
    validate_completed_compiler_transform,
    validate_compiler_diagnostic_contract,
)
from ember.pi05_eval_contract import load_run_contract
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.evaluation_cache import (
    finalize_prefilled_writer_cache,
    load_writer_cache_entry,
    validate_writer_cache_entry_record,
    validate_writer_cache_manifest,
    writer_cache_entry_is_complete,
    writer_cache_episode_request_map,
    writer_cache_requests,
    write_writer_cache_entry,
)
from ember.writer.errors import WriterModelError
from ember.writer.rank_reserved_compiler import compile_rank_reserved_qv_factors
_VIDEO_IDENTITY_FIELDS = (
    "condition",
    "language_global_task_id",
    "teacher_demo_indices",
    "teacher_reference_demo_indices",
    "teacher_video_count",
    "teacher_video_frames_used",
    "teacher_video_kind",
    "teacher_video_order_seeds",
    "teacher_video_sampling_mode",
    "teacher_video_seed_root",
    "teacher_video_selection_seed",
    "video_global_task_id",
    "video_split_role",
    "video_suite",
    "video_task_id",
    "pairing_reference",
    "task_video_mapping_reference",
)


def _source_contract(authority: Mapping[str, Any]) -> dict[str, Any]:
    source = authority["source_old134"]
    contract = load_compiler_diagnostic_source_contract(authority)
    cache = source["cache"]
    manifest_record = cache["manifest"]
    manifest_path = compiler_diagnostic_output_path(
        str(manifest_record["path"]),
        label="compiler-only source cache manifest",
        require_file=True,
    )
    results_record = source["results"]
    results_path = compiler_diagnostic_output_path(
        str(results_record["path"]),
        label="compiler-only source results",
        require_file=True,
    )
    results = read_json(results_path)
    descriptor = contract.get("writer_lora_cache", {})
    if (
        contract.get("adapter", {}).get("video_condition") != "correct"
        or results_path.stat().st_size != int(results_record["bytes"])
        or results.get("schema_version") != results_record["schema"]
        or int(results.get("overall", {}).get("successes", -1))
        != int(source["correct"])
        or sum(
            int(row.get("successes", 0)) > 0
            for row in results.get("per_task", ())
        )
        != int(source["breadth"])
        or manifest_path.stat().st_size != int(manifest_record["bytes"])
        or descriptor.get("reference") != cache["reference"]
        or int(descriptor.get("entry_count", -1)) != int(cache["entry_count"])
        or Path(str(descriptor.get("root", ""))).resolve()
        != compiler_diagnostic_output_path(
            str(cache["root"]), label="compiler-only source cache"
        )
        or Path(str(descriptor.get("root", ""))).is_symlink()
    ):
        raise Pi05EvaluationError("compiler-only source cache authority changed")
    manifest = validate_writer_cache_manifest(contract, verify_entry_files=False)
    if (
        manifest.get("schema_version") != manifest_record["schema"]
        or manifest.get("cache_reference") != cache["reference"]
        or len(manifest.get("entry_ids", ())) != int(cache["entry_count"])
    ):
        raise Pi05EvaluationError("compiler-only source cache manifest changed")
    return contract


def _video_identity(evidence: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(evidence[name] for name in _VIDEO_IDENTITY_FIELDS)
    except KeyError as error:
        raise WriterModelError(
            "compiler-only source video identity is incomplete"
        ) from error


def _cache_lora_path(contract: Mapping[str, Any], request: Any) -> Path:
    return (
        Path(str(contract["writer_lora_cache"]["root"]))
        / "entries"
        / request.entry_id
        / "lora.safetensors"
    )


def _action_factor_names(action_targets: Sequence[str]) -> tuple[str, ...]:
    names = tuple(
        target + suffix
        for target in action_targets
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)
    )
    if len(names) != 4 or len(set(names)) != 4:
        raise WriterModelError("compiler-only action factor layout changed")
    return names


def _direct_action_state(
    source_state: Mapping[str, torch.Tensor],
    action_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Select native CPU action tensors without arithmetic, cast, or cloning."""

    result = {}
    for name in action_names:
        value = source_state[name]
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise WriterModelError("compiler-only action native source changed")
        result[name] = value
    return result


def _validate_action_file_copy(
    source_contract: Mapping[str, Any],
    source_request: Any,
    target_contract: Mapping[str, Any],
    target_request: Any,
    *,
    action_names: Sequence[str],
    source_state: Mapping[str, torch.Tensor] | None = None,
) -> int:
    """Compare four independently loaded target tensors with their v8 source."""

    source_path = _cache_lora_path(source_contract, source_request)
    target_path = _cache_lora_path(target_contract, target_request)
    try:
        with safe_open(target_path, framework="pt", device="cpu") as target_file:
            if source_state is None:
                with safe_open(
                    source_path, framework="pt", device="cpu"
                ) as source_file:
                    equal = True
                    for name in action_names:
                        target_value = target_file.get_tensor(name)
                        source_value = source_file.get_tensor(name)
                        equal = equal and (
                            target_value.dtype == torch.float32
                            and source_value.dtype == torch.float32
                            and torch.equal(target_value, source_value)
                        )
            else:
                source_values = _direct_action_state(source_state, action_names)
                equal = True
                for name in action_names:
                    target_value = target_file.get_tensor(name)
                    equal = equal and (
                        target_value.dtype == torch.float32
                        and torch.equal(target_value, source_values[name])
                    )
    except (KeyError, OSError, RuntimeError) as error:
        raise WriterModelError(
            "compiler-only action cache tensor is unreadable"
        ) from error
    if not equal:
        raise WriterModelError("compiler-only action cache copy is not bit-exact")
    return len(action_names)


def _qv_target_names(lora: Any, projection: str) -> tuple[str, ...]:
    names = tuple(
        target.name
        for target in lora.targets
        if target.name.endswith(f"self_attn.{projection}_proj")
    )
    if len(names) != 18:
        raise WriterModelError("compiler-only q/v target layout changed")
    return names


def _stack_qv(
    states: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values_a = torch.stack(
        [
            torch.stack([state[name + LORA_A_SUFFIX] for name in targets])
            for state in states
        ]
    )
    values_b = torch.stack(
        [
            torch.stack([state[name + LORA_B_SUFFIX] for name in targets])
            for state in states
        ]
    )
    if (
        values_a.shape[:2] != (8, 18)
        or values_b.shape[:2] != (8, 18)
        or values_a.dtype != torch.bfloat16
        or values_b.dtype != torch.bfloat16
        or values_a.device.type != "cpu"
        or values_b.device.type != "cpu"
    ):
        raise WriterModelError("compiler-only q/v B8x18 source topology changed")
    return values_a.to(device=device), values_b.to(device=device)


def _compile_qv_batch(
    states: Sequence[Mapping[str, torch.Tensor]],
    q_targets: Sequence[str],
    v_targets: Sequence[str],
    *,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if device.type != "cuda":
        raise WriterModelError("compiler-only q/v compile requires CUDA")
    result = {}
    for projection, targets in (("q", q_targets), ("v", v_targets)):
        source_a, source_b = _stack_qv(states, targets, device=device)
        # Online cache generation calls the Writer inside this exact ambient
        # autocast context.  The pivot helper intentionally inherits it, so the
        # old-cache counterfactual must do the same to isolate regeneration.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            public_a, public_b, _ = compile_rank_reserved_qv_factors(
                source_a, source_b
            )
        if (
            public_a.dtype != torch.bfloat16
            or public_b.dtype != torch.bfloat16
            or public_a.shape[:2] != (8, 18)
            or public_b.shape[:2] != (8, 18)
        ):
            raise WriterModelError("compiler-only q/v native topology changed")
        result[projection] = (public_a.cpu(), public_b.cpu())
    return result


def _canonical_batch_population_plan(
    source_requests: Sequence[Any],
    complete: Sequence[bool],
) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    """Retain the online B8 numeric shape while writing only missing entries."""

    if len(source_requests) != 8 or len(complete) != 8:
        raise WriterModelError("compiler-only transform lost canonical B8")
    missing = tuple(index for index, value in enumerate(complete) if not value)
    return (tuple(source_requests) if missing else ()), missing


def _entry_generation_matches(
    record: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    contract: Mapping[str, Any],
    source_entry_id: str,
    batch_ordinal: int,
    position: int,
    batch_entry_ids: Sequence[str],
) -> bool:
    generation = record.get("generation", {})
    return all(
        (
            generation.get("schema_version") == COMPILER_DIAGNOSTIC_ENTRY_SCHEMA,
            generation.get("population_mode") == "compiler_only_prefill",
            generation.get("implementation_commit")
            == authority["implementation_commit"],
            generation.get("evaluation_commit")
            == contract.get("git", {}).get("commit"),
            generation.get("source_cache_reference")
            == COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
            generation.get("source_contract_reference")
            == COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE,
            generation.get("source_entry_id") == source_entry_id,
            int(generation.get("batch_ordinal", -1)) == batch_ordinal,
            int(generation.get("position_in_batch", -1)) == position,
            generation.get("batch_entry_ids") == list(batch_entry_ids),
            int(generation.get("batch_size", -1)) == 8,
            generation.get("qv_leading_shape") == [8, 18],
            generation.get("ambient_bf16_autocast") is True,
            generation.get("action_copy")
            == "cpu_native_direct_tensor_copy_without_arithmetic",
            generation.get("action_bit_exact_validation")
            == "target_safetensors_vs_source_tensor_torch_equal",
        )
    )


def _validated_completed_entry(
    target_contract: Mapping[str, Any],
    target_request: Any,
    *,
    source_contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    source_request: Any,
    batch_ordinal: int,
    position: int,
    batch_entry_ids: Sequence[str],
) -> dict[str, Any]:
    record = validate_writer_cache_entry_record(target_contract, target_request)
    evidence = record.get("evidence", {})
    source_record = validate_writer_cache_entry_record(source_contract, source_request)
    if (
        not validate_writer_episode(
            target_contract["adapter"],
            evidence,
            suite=target_request.suite,
            task_id=target_request.task_id,
            init_state_id=target_request.init_state_id,
        )
        or _video_identity(source_record.get("evidence", {}))
        != _video_identity(evidence)
        or not _entry_generation_matches(
            record,
            authority=authority,
            contract=target_contract,
            source_entry_id=source_request.entry_id,
            batch_ordinal=batch_ordinal,
            position=position,
            batch_entry_ids=batch_entry_ids,
        )
    ):
        raise WriterModelError("compiler-only completed cache entry changed")
    return record


def _write_missing_batch_entries(
    *,
    target_contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    lora: Any,
    requests: Sequence[Any],
    source_requests: Sequence[Any],
    source_states: Sequence[Mapping[str, torch.Tensor]],
    source_evidence: Sequence[Mapping[str, Any]],
    missing_positions: Sequence[int],
    compiled: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    q_targets: Sequence[str],
    v_targets: Sequence[str],
    action_names: Sequence[str],
    batch_entry_ids: Sequence[str],
    batch_ordinal: int,
    batch_seconds: float,
) -> tuple[int, int, int]:
    generated = action_assignments = action_checks = 0
    for position, (request, source_request, old_state, old_evidence) in enumerate(
        zip(
            requests,
            source_requests,
            source_states,
            source_evidence,
            strict=True,
        )
    ):
        if position not in missing_positions:
            continue
        state: dict[str, torch.Tensor] = {}
        for projection, targets in (("q", q_targets), ("v", v_targets)):
            public_a, public_b = compiled[projection]
            for layer, target in enumerate(targets):
                state[target + LORA_A_SUFFIX] = public_a[position, layer]
                state[target + LORA_B_SUFFIX] = public_b[position, layer]
        action_state = _direct_action_state(old_state, action_names)
        state.update(action_state)
        action_assignments += len(action_state)
        reference = target_contract["adapter"]["writer_asset"]["reference"]
        evidence = expected_writer_episode(
            target_contract["adapter"],
            suite=request.suite,
            task_id=request.task_id,
            init_state_id=request.init_state_id,
            lora_reference=(
                f"{reference}:{request.suite}:{request.task_id}:"
                f"{request.init_state_id}"
            ),
            evidence_schema=RANK_RESERVED_EPISODE_SCHEMA,
        )
        if _video_identity(old_evidence) != _video_identity(evidence):
            raise WriterModelError("compiler-only v8-to-v9 video identity changed")
        evidence["writer_generation_seconds"] = batch_seconds / 8
        generation = {
            "schema_version": COMPILER_DIAGNOSTIC_ENTRY_SCHEMA,
            "population_mode": "compiler_only_prefill",
            "implementation_commit": authority["implementation_commit"],
            "evaluation_commit": target_contract["git"]["commit"],
            "source_cache_reference": COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
            "source_contract_reference": (
                COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE
            ),
            "source_entry_id": source_request.entry_id,
            "source_episode_schema": old_evidence["schema_version"],
            "batch_ordinal": batch_ordinal,
            "position_in_batch": position,
            "batch_entry_ids": list(batch_entry_ids),
            "batch_size": 8,
            "qv_leading_shape": [8, 18],
            "qv_shared_helper": "compile_rank_reserved_qv_factors",
            "qv_zero_residual_slots": 2,
            "ambient_bf16_autocast": True,
            "action_copy": "cpu_native_direct_tensor_copy_without_arithmetic",
            "action_bit_exact_validation": (
                "target_safetensors_vs_source_tensor_torch_equal"
            ),
            "teacher_video_or_action_reads": 0,
            "policy_forwards": 0,
        }
        write_writer_cache_entry(
            target_contract,
            request,
            state=state,
            evidence=evidence,
            generation=generation,
            lora_contract=lora,
        )
        action_checks += _validate_action_file_copy(
            source_contract,
            source_request,
            target_contract,
            request,
            action_names=action_names,
            source_state=old_state,
        )
        generated += 1
    return generated, action_assignments, action_checks


def _populate_compiler_entries(
    *,
    target_contract: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    target_requests: Sequence[Any],
    source_by_episode: Mapping[tuple[str, int, int], Any],
    lora: Any,
    q_targets: Sequence[str],
    v_targets: Sequence[str],
    action_names: Sequence[str],
    device: torch.device,
) -> dict[str, int]:
    counts = {
        "generated": 0,
        "reused": 0,
        "action_assignments": 0,
        "action_checks": 0,
        "source_reads": 0,
        "transformed_batches": 0,
    }
    for batch_ordinal, offset in enumerate(range(0, len(target_requests), 8)):
        requests = target_requests[offset : offset + 8]
        if len(requests) != 8:
            raise WriterModelError("compiler-only transform lost full B8 batches")
        batch_entry_ids = [request.entry_id for request in requests]
        source_requests, complete = [], []
        for request in requests:
            source_request = source_by_episode.get(
                (request.suite, request.task_id, request.init_state_id)
            )
            if source_request is None:
                raise WriterModelError("compiler-only source episode mapping changed")
            source_requests.append(source_request)
            complete.append(writer_cache_entry_is_complete(target_contract, request))
        for position, (request, source_request, is_complete) in enumerate(
            zip(requests, source_requests, complete, strict=True)
        ):
            if not is_complete:
                continue
            _validated_completed_entry(
                target_contract,
                request,
                source_contract=source_contract,
                authority=authority,
                source_request=source_request,
                batch_ordinal=batch_ordinal,
                position=position,
                batch_entry_ids=batch_entry_ids,
            )
            counts["action_checks"] += _validate_action_file_copy(
                source_contract,
                source_request,
                target_contract,
                request,
                action_names=action_names,
            )
            counts["reused"] += 1
        forward_requests, missing = _canonical_batch_population_plan(
            source_requests, complete
        )
        if not forward_requests:
            continue
        batch_started = time.monotonic()
        source_states, source_evidence = [], []
        for source_request in forward_requests:
            state, evidence = load_writer_cache_entry(
                source_contract,
                source_request,
                lora_contract=lora,
                device=torch.device("cpu"),
            )
            source_states.append(state)
            source_evidence.append(evidence)
            counts["source_reads"] += 1
        compiled = _compile_qv_batch(
            source_states, q_targets, v_targets, device=device
        )
        added = _write_missing_batch_entries(
            target_contract=target_contract,
            source_contract=source_contract,
            authority=authority,
            lora=lora,
            requests=requests,
            source_requests=source_requests,
            source_states=source_states,
            source_evidence=source_evidence,
            missing_positions=missing,
            compiled=compiled,
            q_targets=q_targets,
            v_targets=v_targets,
            action_names=action_names,
            batch_entry_ids=batch_entry_ids,
            batch_ordinal=batch_ordinal,
            batch_seconds=time.monotonic() - batch_started,
        )
        counts["generated"] += added[0]
        counts["action_assignments"] += added[1]
        counts["action_checks"] += added[2]
        counts["transformed_batches"] += 1
    return counts


def _seal_transform(
    output_dir: Path,
    target_contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    transform: Mapping[str, Any],
) -> dict[str, Any]:
    transform_path = output_dir / COMPILER_DIAGNOSTIC_TRANSFORM
    write_json_atomic(transform_path, transform)
    population_evidence = {
        "schema_version": COMPILER_DIAGNOSTIC_TRANSFORM_SCHEMA,
        "path": COMPILER_DIAGNOSTIC_TRANSFORM,
        "bytes": transform_path.stat().st_size,
        "implementation_commit": authority["implementation_commit"],
        "evaluation_commit": target_contract["git"]["commit"],
        "source_cache_reference": COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
        "authorizes_cycle1": False,
    }
    finalize_prefilled_writer_cache(target_contract, evidence=population_evidence)
    validate_completed_compiler_transform(authority, output_dir, target_contract)
    return dict(transform)


@torch.inference_mode()
def transform_compiler_only_cache(
    output_dir: Path,
    *,
    physical_gpu_index: int,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if torch.backends.cuda.matmul.allow_tf32 is not True:
        raise WriterModelError("compiler-only TF32 context differs from online Writer")
    target_contract = load_run_contract(output_dir / "run_contract.json")
    authority = validate_compiler_diagnostic_contract(
        output_dir,
        target_contract,
        require_cache_ready=False,
    )
    source_contract = _source_contract(authority)
    target_requests = writer_cache_requests(target_contract)
    source_by_episode = writer_cache_episode_request_map(source_contract)
    if len(target_requests) != 400 or len(source_by_episode) != 400:
        raise WriterModelError("compiler-only cache panel changed")
    lora = load_pi05_lora_contract(
        authority_path(
            load_rank_reserved_config(RANK_RESERVED_CANONICAL_CONFIG),
            "lora_contract",
        )
    )
    q_targets = _qv_target_names(lora, "q")
    v_targets = _qv_target_names(lora, "v")
    action_targets = tuple(
        target.name for target in lora.targets if ".action_" in target.name
    )
    if len(action_targets) != 2:
        raise WriterModelError("compiler-only action target layout changed")
    action_names = _action_factor_names(action_targets)

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    counts = _populate_compiler_entries(
        target_contract=target_contract,
        source_contract=source_contract,
        authority=authority,
        target_requests=target_requests,
        source_by_episode=source_by_episode,
        lora=lora,
        q_targets=q_targets,
        v_targets=v_targets,
        action_names=action_names,
        device=device,
    )
    generated = counts["generated"]
    reused = counts["reused"]
    action_assignments = counts["action_assignments"]
    action_file_equal_checks = counts["action_checks"]
    source_reads = counts["source_reads"]
    transformed_batches = counts["transformed_batches"]

    if (
        generated + reused != 400
        or action_assignments != generated * 4
        or action_file_equal_checks != 1_600
    ):
        raise WriterModelError("compiler-only transform completion changed")
    transform = {
        "schema_version": COMPILER_DIAGNOSTIC_TRANSFORM_SCHEMA,
        "status": "complete",
        "root": str(output_dir),
        "implementation_commit": authority["implementation_commit"],
        "evaluation_commit": target_contract["git"]["commit"],
        "source": {
            "root": str(
                compiler_diagnostic_output_path(
                    COMPILER_DIAGNOSTIC_SOURCE_ROOT,
                    label="compiler-only source root",
                )
            ),
            "contract_reference": COMPILER_DIAGNOSTIC_SOURCE_CONTRACT_REFERENCE,
            "cache_reference": COMPILER_DIAGNOSTIC_SOURCE_CACHE_REFERENCE,
            "entry_count": 400,
            "read_only": True,
        },
        "target": {
            "contract_reference": target_contract["contract_reference"],
            "cache_reference": target_contract["writer_lora_cache"]["reference"],
            "entry_count": 400,
        },
        "preflight": dict(preflight),
        "physical_gpu_index": physical_gpu_index,
        "batch_size": 8,
        "batch_count": 50,
        "transformed_batches_this_invocation": transformed_batches,
        "generated_entries_this_invocation": generated,
        "reused_entries": reused,
        "source_entries_read_this_invocation": source_reads,
        "qv_leading_shape": [8, 18],
        "qv_target_count": 36,
        "action_target_count": 2,
        "action_tensor_direct_assignments_this_invocation": action_assignments,
        "source_target_action_tensor_equal_checks": action_file_equal_checks,
        "source_target_action_tensors_bit_exact": True,
        "source_target_video_identities_equal": 400,
        "native_storage": "72_BF16_plus_4_F32",
        "ambient_bf16_autocast": True,
        "fp32_matmul_allow_tf32": True,
        "teacher_video_or_action_reads": 0,
        "policy_forwards": 0,
        "rollouts": 0,
        "training_updates": 0,
        "content_hashes": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "authorizes_cycle1": False,
    }
    return _seal_transform(output_dir, target_contract, authority, transform)
