"""Writer cache generation and rollout-only adapter handoff."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ember.eval_adapters import (
    ARCHIVAL_WRITER_CACHE_KIND,
    DYNAMIC_K_WRITER_KIND,
    FUNCTIONAL_CODE_WRITER_KIND,
    WRITER_ADAPTER_KINDS,
    expected_writer_episode,
    reinspect_writer_adapter,
    validate_writer_episode,
)
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.writer.evaluation_cache import (
    WriterCacheGenerationBatch,
    WriterCacheRequest,
    assigned_writer_cache_batches,
    load_writer_cache_entry,
    stage_writer_lora_states_to_cpu,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_entry_is_complete,
    writer_cache_episode_request_map,
    writer_cache_manifest_path,
)
from ember.writer.errors import WriterModelError
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.lora_rollout import PreparedWriterLoRA, WriterLoRARolloutAdapter


class FrozenCachedWriterTaskAdapter(WriterLoRARolloutAdapter):
    """Load sealed episode LoRAs without loading any Writer or teacher video."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        source: Mapping[str, Any],
        evaluation_adapter: Mapping[str, Any],
        task_keys: Sequence[tuple[str, int]],
        device: torch.device,
        tokenizer_path: Any,
        require_formal: bool,
        cache_contract: Mapping[str, Any],
    ) -> None:
        del tokenizer_path
        if evaluation_adapter.get("kind") not in WRITER_ADAPTER_KINDS:
            raise WriterModelError("cached rollout requires a canonical Writer")
        observed = reinspect_writer_adapter(
            evaluation_adapter,
            source=source,
            task_keys=task_keys,
            require_formal=require_formal,
        )
        if observed != dict(evaluation_adapter):
            raise WriterModelError(
                "PI05 Writer evaluation artifacts changed after prepare"
            )
        config_path = Path(observed["config"]["path"])
        if observed["kind"] == DYNAMIC_K_WRITER_KIND:
            from ember.writer.as_config import authority_path, load_writer_config

            config = load_writer_config(config_path)
            lora_path = authority_path(config, "lora_contract")
        elif observed["kind"] == FUNCTIONAL_CODE_WRITER_KIND:
            from ember.functional_adaptation.decoder_training import (
                authority_path,
                load_functional_adapter_config,
            )

            repo_root = Path(__file__).resolve().parents[3]
            config = load_functional_adapter_config(config_path, repo_root)
            lora_path = authority_path(config, "lora_contract", repo_root)
        elif observed["kind"] == ARCHIVAL_WRITER_CACHE_KIND:
            from ember.writer.archival_projection import (
                load_archival_lora_contract,
            )

            lora = load_archival_lora_contract(observed)
            lora_path = None
        else:
            raise WriterModelError("cached Writer kind changed")
        if lora_path is not None:
            lora = load_pi05_lora_contract(lora_path)
        observed_rank = int(observed["lora_contract"]["rank"])
        if observed["kind"] == DYNAMIC_K_WRITER_KIND and observed_rank != lora.rank:
            lora = derive_pi05_lora_rank(lora, rank=observed_rank)
        elif observed_rank != lora.rank:
            raise WriterModelError("cached functional Writer rank changed")
        template = prepare_frozen_writer_policy(policy, lora)
        self._initialize_rollout(
            policy=policy,
            lora_contract=lora,
            identity_state=template,
            evaluation_adapter=observed,
            device=device,
        )
        self._initialize_cache(cache_contract)
        self.activate_cache()

    @classmethod
    def from_live(
        cls,
        generator: WriterLoRARolloutAdapter,
        *,
        cache_contract: Mapping[str, Any],
    ) -> FrozenCachedWriterTaskAdapter:
        result = cls.__new__(cls)
        result.policy = generator.policy
        result.lora_contract = generator.lora_contract
        result.identity_state = generator.identity_state
        result.device = generator.device
        result.evaluation_adapter = generator.evaluation_adapter
        result.batched_lora = generator.batched_lora
        result._physical_lora_is_identity = generator._physical_lora_is_identity
        result._initialize_cache(cache_contract)
        return result

    def _initialize_cache(self, cache_contract: Mapping[str, Any]) -> None:
        self.cache_contract = dict(cache_contract)
        self._request_by_key = writer_cache_episode_request_map(self.cache_contract)
        self._state_cache: dict[
            str, tuple[Mapping[str, torch.Tensor], dict[str, Any]]
        ] = {}
        self._prepared_cache: dict[tuple[str, int, int], PreparedWriterLoRA] = {}
        self._cache_validated = False

    def activate_cache(self) -> None:
        validate_writer_cache_manifest(
            self.cache_contract,
            verify_entry_files=False,
        )
        self._cache_validated = True

    @torch.inference_mode()
    def prepare_episode(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> PreparedWriterLoRA:
        if not self._cache_validated:
            raise WriterModelError("Writer LoRA cache is not sealed")
        key = suite, task_id, init_state_id
        request = self._request_by_key.get(key)
        if request is None:
            raise WriterModelError("rollout episode is outside the Writer LoRA cache")
        prepared = self._prepared_cache.get(key)
        if prepared is None:
            cached = self._state_cache.get(request.entry_id)
            if cached is None:
                cached = load_writer_cache_entry(
                    self.cache_contract,
                    request,
                    lora_contract=self.lora_contract,
                    device=self.device,
                )
                if not validate_writer_episode(
                    self.evaluation_adapter,
                    cached[1],
                    suite=request.suite,
                    task_id=request.task_id,
                    init_state_id=request.init_state_id,
                ):
                    raise WriterModelError("cached Writer source evidence changed")
                self._state_cache[request.entry_id] = cached
            state, cached_evidence = cached
            evidence = expected_writer_episode(
                self.evaluation_adapter,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                lora_reference=str(cached_evidence["lora_reference"]),
                evidence_schema=str(cached_evidence["schema_version"]),
            )
            evidence["writer_generation_seconds"] = float(
                cached_evidence["writer_generation_seconds"]
            )
            if not validate_writer_episode(
                self.evaluation_adapter,
                evidence,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
            ):
                raise WriterModelError("cached Writer episode evidence changed")
            prepared = PreparedWriterLoRA(state=state, evidence=evidence)
            self._prepared_cache[key] = prepared
        return prepared


def _finish_generation_handoff(
    runtime: Any,
    *,
    invocation_id: str,
    generation: Mapping[str, Any],
    append_event: Callable[[Any, Mapping[str, Any]], None],
) -> dict[str, Any]:
    runtime.task_adapter = runtime.task_adapter.release_to_cache(runtime.contract)
    torch.cuda.empty_cache()
    summary = {
        "source_policy_reused_for_rollout": True,
        "writer_modules_released": True,
        **dict(generation),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "post_release_allocated_bytes": int(torch.cuda.memory_allocated()),
        "post_release_reserved_bytes": int(torch.cuda.memory_reserved()),
    }
    write_generator_marker(
        runtime.contract,
        invocation_id=invocation_id,
        worker_id=runtime.worker_id,
        generator_index=int(generation["generator_index"]),
        summary=summary,
    )
    append_event(
        runtime.event_path,
        {
            "event": "writer_generation_finished",
            "unix": time.time(),
            "worker_id": runtime.worker_id,
            "pid": os.getpid(),
            "invocation_id": invocation_id,
            "contract_reference": runtime.contract["contract_reference"],
            **summary,
        },
    )
    while not writer_cache_manifest_path(runtime.contract).is_file():
        time.sleep(0.2)
    runtime.task_adapter.activate_cache()
    append_event(
        runtime.event_path,
        {
            "event": "rollout_ready_with_retained_policy",
            "unix": time.time(),
            "worker_id": runtime.worker_id,
            "pid": os.getpid(),
            "invocation_id": invocation_id,
            "source_policy_reloaded": False,
            "contract_reference": runtime.contract["contract_reference"],
        },
    )
    return summary


def _prepare_writer_generation_batch(
    runtime: Any,
    requests: Sequence[WriterCacheRequest],
) -> tuple[Sequence[Any], tuple[dict[str, Any], ...], Sequence[Any], float]:
    batch_started = time.monotonic()
    prepared = runtime.task_adapter.prepare_episodes(
        [
            {
                "suite": request.suite,
                "task_id": request.task_id,
                "init_state_id": request.init_state_id,
            }
            for request in requests
        ]
    )
    if len(prepared) != len(requests):
        raise WriterModelError("Writer generation batch coverage changed")
    profile_reader = getattr(
        runtime.task_adapter,
        "last_generation_batch_profile",
        None,
    )
    if not callable(profile_reader):
        raise WriterModelError("Writer generation batch lacks video-length evidence")
    video_profile = tuple(profile_reader())
    observed_profile = tuple(
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))
        for row in video_profile
    )
    expected_profile = tuple(
        (request.suite, request.task_id, request.init_state_id) for request in requests
    )
    if observed_profile != expected_profile:
        raise WriterModelError("Writer generation video-length ownership changed")
    staged_states = stage_writer_lora_states_to_cpu(
        tuple(item.state for item in prepared)
    )
    return (
        prepared,
        video_profile,
        staged_states,
        time.monotonic() - batch_started,
    )


def _generate_writer_cache_batch(
    runtime: Any,
    *,
    assigned_batch: WriterCacheGenerationBatch,
    generator_index: int,
) -> dict[str, Any]:
    requests = assigned_batch.requests
    complete = tuple(
        writer_cache_entry_is_complete(runtime.contract, request)
        for request in requests
    )
    if all(complete):
        return {
            "generated_entries": 0,
            "reused_entries": len(requests),
            "redundant_writer_forwards": 0,
            "batch": None,
        }
    pending = tuple(
        request
        for request, was_complete in zip(requests, complete, strict=True)
        if not was_complete
    )
    forward_requests = requests if assigned_batch.canonical_global else pending
    prepared, profile, states, batch_seconds = _prepare_writer_generation_batch(
        runtime,
        forward_requests,
    )
    complete_in_forward = (
        complete if assigned_batch.canonical_global else (False,) * len(pending)
    )
    generated_entries = 0
    batch_entry_ids = [request.entry_id for request in forward_requests]
    for position, (request, was_complete, item, state, video) in enumerate(
        zip(
            forward_requests,
            complete_in_forward,
            prepared,
            states,
            profile,
            strict=True,
        )
    ):
        if was_complete:
            continue
        evidence = dict(item.evidence)
        evidence["writer_generation_seconds"] = batch_seconds / len(forward_requests)
        write_writer_cache_entry(
            runtime.contract,
            request,
            state=state,
            evidence=evidence,
            generation={
                "generator_worker_id": runtime.worker_id,
                "generator_index": generator_index,
                "batch_ordinal": assigned_batch.ordinal,
                "position_in_batch": position,
                "batch_entry_ids": batch_entry_ids,
                "batch_size": len(forward_requests),
                "batch_wall_seconds": batch_seconds,
                "raw_frames": int(video["raw_frames"]),
                "available_stride5_frames": int(
                    video.get("available_stride5_frames", video["sampled_frames"])
                ),
                "sampled_frames": int(video["sampled_frames"]),
            },
            lora_contract=runtime.task_adapter.lora_contract,
        )
        generated_entries += 1
    return {
        "generated_entries": generated_entries,
        "reused_entries": sum(complete),
        "redundant_writer_forwards": (
            sum(complete) if assigned_batch.canonical_global else 0
        ),
        "batch": {
            "batch_ordinal": assigned_batch.ordinal,
            "entry_ids": batch_entry_ids,
            "batch_size": len(forward_requests),
            "raw_frame_counts": [int(row["raw_frames"]) for row in profile],
            "available_stride5_frame_counts": [
                int(row.get("available_stride5_frames", row["sampled_frames"]))
                for row in profile
            ],
            "sampled_frame_counts": [int(row["sampled_frames"]) for row in profile],
            "wall_seconds": batch_seconds,
        },
    }


def run_writer_generation_phase(
    runtime: Any,
    *,
    invocation_id: str,
    append_event: Callable[[Any, Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Populate the sealed cache, release Writer, and retain the source policy."""

    generators_per_gpu = int(runtime.contract["parallel"]["writer_generators_per_gpu"])
    generator_index = runtime.gpu_slot * generators_per_gpu + runtime.replica
    assigned_batches = assigned_writer_cache_batches(
        runtime.contract,
        generator_index=generator_index,
    )
    batch_size = int(runtime.contract["parallel"]["writer_generation_batch_size"])
    if not callable(getattr(runtime.task_adapter, "prepare_episodes", None)):
        raise WriterModelError("Writer generator lacks batched LoRA generation")
    phase_started = time.monotonic()
    generated_entries = reused_entries = generated_batches = 0
    redundant_writer_forwards = 0
    batch_rows = []
    torch.cuda.reset_peak_memory_stats()
    for assigned_batch in assigned_batches:
        result = _generate_writer_cache_batch(
            runtime,
            assigned_batch=assigned_batch,
            generator_index=generator_index,
        )
        generated_entries += int(result["generated_entries"])
        reused_entries += int(result["reused_entries"])
        redundant_writer_forwards += int(result["redundant_writer_forwards"])
        if result["batch"] is not None:
            generated_batches += 1
            batch_rows.append(result["batch"])
    return _finish_generation_handoff(
        runtime,
        invocation_id=invocation_id,
        append_event=append_event,
        generation={
            "generator_index": generator_index,
            "assigned_entries": sum(len(batch.requests) for batch in assigned_batches),
            "generated_entries": generated_entries,
            "reused_entries": reused_entries,
            "generated_batches": generated_batches,
            "generation_batch_size": batch_size,
            "generation_wall_seconds": time.monotonic() - phase_started,
            "redundant_writer_forwards": redundant_writer_forwards,
            "batch_shape_bf16_roundoff_accepted": True,
            "batches": batch_rows,
        },
    )
