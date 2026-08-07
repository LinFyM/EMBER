"""Writer cache generation and rollout-only adapter handoff."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.as_contract import REPO_ROOT, load_writer_config
from ember.writer.evaluation_cache import (
    assigned_writer_cache_requests,
    load_writer_cache_entry,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_entry_is_complete,
    writer_cache_episode_request_map,
    writer_cache_manifest_path,
)
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.inference import (
    expected_writer_episode_evidence,
    inspect_as_writer_evaluation,
    validate_writer_episode_evidence,
)
from ember.writer.live_adapter import FrozenWriterTaskAdapter, PreparedWriterLoRA
from ember.writer.lora_rollout import WriterLoRARolloutAdapter
from ember.writer.model import WriterModelError


def _scaled_public_lora_b(
    state: Mapping[str, torch.Tensor],
    scale: float,
) -> Mapping[str, torch.Tensor]:
    if scale == 1.0:
        return state
    result = {}
    b_factors = 0
    for name, value in state.items():
        if ".lora_B." in name:
            result[name] = value * scale
            b_factors += 1
        else:
            result[name] = value
    if b_factors <= 0 or b_factors * 2 != len(result):
        raise WriterModelError("generated public LoRA A/B topology changed")
    return result


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
        kind = str(evaluation_adapter.get("kind", "as_writer"))
        common = {
            "config_path": Path(evaluation_adapter["config"]["path"]),
            "checkpoint": Path(evaluation_adapter["checkpoint"]["path"]),
            "video_data_root": Path(evaluation_adapter["video_data"]["root"]),
            "source": source,
            "task_keys": task_keys,
            "video_condition": str(evaluation_adapter["video_condition"]),
            "video_seed": int(evaluation_adapter["video_schedule"]["seed"]),
            "video_sampling_mode": (
                str(evaluation_adapter["video_schedule"]["sampling_mode"])
                if "sampling_mode" in evaluation_adapter["video_schedule"]
                else None
            ),
            "require_formal": require_formal,
        }
        if kind == "as_writer":
            observed = inspect_as_writer_evaluation(**common)
            config = load_writer_config(Path(observed["config"]["path"]))
        elif kind == "rl_writer":
            from ember.rl_writer.contract import authority_path, load_rl_writer_config
            from ember.rl_writer.inference import inspect_rl_writer_evaluation

            observed = inspect_rl_writer_evaluation(**common)
            rl_config = load_rl_writer_config(Path(observed["config"]["path"]))
            config = load_writer_config(authority_path(rl_config, "as_writer_config"))
        else:
            raise WriterModelError("cached rollout requires a canonical Writer adapter")
        if observed != dict(evaluation_adapter):
            raise WriterModelError("PI05 Writer evaluation artifacts changed after prepare")
        lora = load_pi05_lora_contract(
            REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
        )
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
        generator: FrozenWriterTaskAdapter,
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
        execution = cache_contract.get("writer_lora_execution")
        self.lora_b_scale = float(
            execution.get("b_scale", 1.0)
            if isinstance(execution, Mapping)
            else 1.0
        )
        if (
            not math.isfinite(self.lora_b_scale)
            or self.lora_b_scale <= 0
            or self.lora_b_scale > 4
        ):
            raise WriterModelError("cached Writer LoRA B scale is invalid")
        self._request_by_key = writer_cache_episode_request_map(
            self.cache_contract
        )
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
                cached = (
                    _scaled_public_lora_b(cached[0], self.lora_b_scale),
                    cached[1],
                )
                if not validate_writer_episode_evidence(
                    self.evaluation_adapter,
                    cached[1],
                    suite=request.suite,
                    task_id=request.task_id,
                    init_state_id=request.init_state_id,
                ):
                    raise WriterModelError("cached Writer source evidence changed")
                self._state_cache[request.entry_id] = cached
            state, cached_evidence = cached
            evidence = expected_writer_episode_evidence(
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
            if not validate_writer_episode_evidence(
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


def run_writer_generation_phase(
    runtime: Any,
    *,
    invocation_id: str,
    append_event: Callable[[Any, Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Populate the sealed cache, release Writer, and retain the source policy."""

    generators_per_gpu = int(
        runtime.contract["parallel"]["writer_generators_per_gpu"]
    )
    generator_index = runtime.gpu_slot * generators_per_gpu + runtime.replica
    requests = assigned_writer_cache_requests(
        runtime.contract,
        generator_index=generator_index,
    )
    batch_size = int(runtime.contract["parallel"]["writer_generation_batch_size"])
    if not callable(getattr(runtime.task_adapter, "prepare_episodes", None)):
        raise WriterModelError("Writer generator lacks batched LoRA generation")
    phase_started = time.monotonic()
    generated_entries = reused_entries = generated_batches = 0
    batch_rows = []
    torch.cuda.reset_peak_memory_stats()
    for batch_ordinal, offset in enumerate(range(0, len(requests), batch_size)):
        request_batch = requests[offset : offset + batch_size]
        complete = [
            writer_cache_entry_is_complete(runtime.contract, request)
            for request in request_batch
        ]
        if all(complete):
            reused_entries += len(request_batch)
            continue
        batch_started = time.monotonic()
        prepared = runtime.task_adapter.prepare_episodes(
            [
                {
                    "suite": request.suite,
                    "task_id": request.task_id,
                    "init_state_id": request.init_state_id,
                }
                for request in request_batch
            ]
        )
        batch_seconds = time.monotonic() - batch_started
        if len(prepared) != len(request_batch):
            raise WriterModelError("Writer generation batch coverage changed")
        generated_batches += 1
        for position, (request, item, was_complete) in enumerate(
            zip(request_batch, prepared, complete, strict=True)
        ):
            write_writer_cache_entry(
                runtime.contract,
                request,
                state=item.state,
                evidence=item.evidence,
                generation={
                    "generator_worker_id": runtime.worker_id,
                    "generator_index": generator_index,
                    "batch_ordinal": batch_ordinal,
                    "position_in_batch": position,
                    "batch_entry_ids": [value.entry_id for value in request_batch],
                    "batch_size": len(request_batch),
                    "batch_wall_seconds": batch_seconds,
                },
                lora_contract=runtime.task_adapter.lora_contract,
            )
            generated_entries += int(not was_complete)
            reused_entries += int(was_complete)
        batch_rows.append(
            {
                "batch_ordinal": batch_ordinal,
                "entry_ids": [request.entry_id for request in request_batch],
                "wall_seconds": batch_seconds,
            }
        )
    return _finish_generation_handoff(
        runtime,
        invocation_id=invocation_id,
        append_event=append_event,
        generation={
            "generator_index": generator_index,
            "assigned_entries": len(requests),
            "generated_entries": generated_entries,
            "reused_entries": reused_entries,
            "generated_batches": generated_batches,
            "generation_batch_size": batch_size,
            "generation_wall_seconds": time.monotonic() - phase_started,
            "batches": batch_rows,
        },
    )
