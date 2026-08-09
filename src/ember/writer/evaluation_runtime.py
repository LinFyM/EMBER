"""Writer cache generation and rollout-only adapter handoff."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ember.eval_adapters import (
    EXPERT_MANIFOLD_WRITER_KIND,
    expected_writer_episode,
    validate_writer_episode,
)
from ember.expert_manifold.v6_prior_contract import (
    authority_path,
    load_v6_prior_config,
)
from ember.expert_manifold.inference import (
    inspect_expert_manifold_writer_evaluation,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.evaluation_cache import (
    assigned_writer_cache_requests,
    load_writer_cache_entry,
    stage_writer_lora_states_to_cpu,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_entry_is_complete,
    writer_cache_episode_request_map,
    writer_cache_manifest_path,
    writer_cache_requests,
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
        if evaluation_adapter.get("kind") != EXPERT_MANIFOLD_WRITER_KIND:
            raise WriterModelError(
                "cached rollout requires the canonical Expert-Manifold Writer"
            )
        common = {
            "config_path": Path(evaluation_adapter["config"]["path"]),
            "checkpoint": Path(evaluation_adapter["writer_asset"]["checkpoint"]),
            "video_data_root": Path(evaluation_adapter["video_data"]["root"]),
            "source": source,
            "task_keys": task_keys,
            "video_condition": str(evaluation_adapter["video_condition"]),
            "video_seed": int(evaluation_adapter["video_schedule"]["seed"]),
            "video_sampling_mode": str(
                evaluation_adapter["video_schedule"]["sampling_mode"]
            ),
            "require_formal": require_formal,
        }
        observed = inspect_expert_manifold_writer_evaluation(**common)
        config = load_v6_prior_config(Path(observed["config"]["path"]))
        if observed != dict(evaluation_adapter):
            raise WriterModelError("PI05 Writer evaluation artifacts changed after prepare")
        lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
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


WRITER_GENERATION_PROFILE_SCHEMA = "ember_pi05_writer_generation_profile_v1"


def profile_writer_generation(
    runtime: Any,
    *,
    batch_sizes: Sequence[int],
    warmup_runs: int,
    measured_runs: int,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure actual end-to-end video-to-native-LoRA throughput on one A40."""

    sizes = tuple(int(value) for value in batch_sizes)
    contract = runtime.contract
    adapter = runtime.task_adapter
    if (
        sizes != tuple(sorted(set(sizes)))
        or len(sizes) < 3
        or not {8, 16, 32}.issubset(sizes)
        or warmup_runs <= 0
        or measured_runs < 2
        or contract.get("mode") != "smoke"
        or contract.get("adapter", {}).get("video_condition") != "correct"
        or int(contract["parallel"]["physical_gpu_count"]) != 1
        or int(contract["parallel"]["replicas_per_gpu"]) != 1
        or int(contract["parallel"]["writer_generators_per_gpu"]) != 1
        or int(contract["parallel"]["writer_generation_batch_size"]) != sizes[-1]
        or not callable(getattr(adapter, "prepare_episodes", None))
        or not callable(getattr(adapter, "generation_request_profiles", None))
        or not callable(getattr(adapter, "last_generation_batch_profile", None))
        or not callable(getattr(adapter, "release_generation_assets", None))
    ):
        raise WriterModelError("Writer generation profile contract changed")
    git = contract.get("git", {})
    if git.get("dirty_paths") or git.get("commit") != git.get("upstream_commit"):
        raise WriterModelError("Writer generation profile requires a clean pushed commit")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA A40":
        raise WriterModelError("Writer generation profile requires an NVIDIA A40")
    requests = writer_cache_requests(contract)
    if len(requests) < sizes[-1]:
        raise WriterModelError("Writer profile panel is smaller than its largest batch")
    identities = tuple(
        {
            "suite": request.suite,
            "task_id": request.task_id,
            "init_state_id": request.init_state_id,
        }
        for request in requests
    )
    metadata = adapter.generation_request_profiles(identities)
    if len(metadata) != len(requests) or any(
        int(row["sampled_frames"]) <= 0 for row in metadata
    ):
        raise WriterModelError("Writer profile video-length evidence changed")
    ordered = sorted(
        zip(requests, identities, metadata, strict=True),
        key=lambda item: (-int(item[2]["sampled_frames"]), item[0].ordinal),
    )
    longest = max(int(row["sampled_frames"]) for row in metadata)
    panel = tuple(ordered[: sizes[-1]])
    panel_entry_ids = tuple(item[0].entry_id for item in panel)
    panel_counts = tuple(int(item[2]["sampled_frames"]) for item in panel)
    panel_total_frames = sum(panel_counts)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    required_headroom = max(512 * 1024**2, total_memory // 100)
    rows = []
    profile_started = time.monotonic()
    for size in sizes:
        chunks = tuple(panel[offset : offset + size] for offset in range(0, len(panel), size))
        forward_batch_sizes = tuple(len(chunk) for chunk in chunks)

        def execute_once() -> tuple[float, tuple[dict[str, Any], ...]]:
            torch.cuda.synchronize()
            started = time.monotonic()
            observed_panel = []
            for chunk in chunks:
                chunk_identities = tuple(item[1] for item in chunk)
                chunk_counts = tuple(int(item[2]["sampled_frames"]) for item in chunk)
                prepared = adapter.prepare_episodes(chunk_identities)
                if len(prepared) != len(chunk):
                    raise WriterModelError("Writer profile forward batch changed")
                staged = stage_writer_lora_states_to_cpu(
                    tuple(item.state for item in prepared)
                )
                observed = adapter.last_generation_batch_profile()
                if (
                    len(observed) != len(chunk)
                    or tuple(int(row["sampled_frames"]) for row in observed)
                    != chunk_counts
                ):
                    raise WriterModelError("Writer profile video batch changed")
                observed_panel.extend(observed)
                del prepared, staged
            wall = time.monotonic() - started
            return wall, tuple(observed_panel)

        for _ in range(warmup_runs):
            execute_once()
        torch.cuda.reset_peak_memory_stats()
        walls = []
        for _ in range(measured_runs):
            wall, observed = execute_once()
            if tuple(int(row["sampled_frames"]) for row in observed) != panel_counts:
                raise WriterModelError("Writer profile request ordering changed")
            walls.append(wall)
        generated = len(panel) * measured_runs
        wall_seconds = sum(walls)
        throughput = generated / wall_seconds
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        headroom = total_memory - peak_reserved
        stable = (
            all(value > 0 and math.isfinite(value) for value in walls)
            and max(walls) / min(walls) <= 1.25
            and headroom >= required_headroom
        )
        rows.append(
            {
                "batch_size": size,
                "generated_entries": generated,
                "max_observed_forward_batch_size": max(forward_batch_sizes),
                "forward_batch_sizes_per_repeat": list(forward_batch_sizes),
                "wall_seconds": wall_seconds,
                "loras_per_second": throughput,
                "repeat_wall_seconds": walls,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "device_total_bytes": total_memory,
                "memory_headroom_bytes": headroom,
                "required_memory_headroom_bytes": required_headroom,
                "comparison_panel_shared_across_candidates": True,
                "panel_entry_count": len(panel),
                "panel_total_sampled_frames": panel_total_frames,
                "longest_video_included": max(panel_counts) == longest,
                "max_sampled_video_frames": max(panel_counts),
                "sampled_frame_counts": list(panel_counts),
                "entry_ids": list(panel_entry_ids),
                "stable": stable,
            }
        )
    eligible = [row for row in rows if row["stable"]]
    if not eligible:
        raise WriterModelError("Writer profile found no stable batch with memory headroom")
    selected = max(
        eligible,
        key=lambda row: (float(row["loras_per_second"]), int(row["batch_size"])),
    )
    adapter.release_generation_assets()
    torch.cuda.empty_cache()
    result = {
        "schema_version": WRITER_GENERATION_PROFILE_SCHEMA,
        "contract_reference": contract["contract_reference"],
        "git": dict(git),
        "root": str(runtime.output_dir),
        "device": device_name,
        "gpu_uuid": runtime.gpu_uuid,
        "physical_gpu": runtime.gpu_index,
        "preflight": dict(preflight),
        "profiled_writer_model_batch_sizes": list(sizes),
        "selected_writer_model_batch_size": int(selected["batch_size"]),
        "selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "throughput_comparison_panel": (
            "same_fixed_longest_first_request_panel_all_candidates"
        ),
        "warmup_runs_per_batch": warmup_runs,
        "measured_runs_per_batch": measured_runs,
        "longest_sampled_video_frames": longest,
        "writer_generation_measurements": rows,
        "profile_wall_seconds": time.monotonic() - profile_started,
        "writer_modules_released": True,
        "source_policy_reused": True,
        "post_release_allocated_bytes": int(torch.cuda.memory_allocated()),
        "post_release_reserved_bytes": int(torch.cuda.memory_reserved()),
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(runtime.output_dir / "writer_generation_profile.json", result)
    return result


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
        pending_requests = tuple(
            request
            for request, was_complete in zip(request_batch, complete, strict=True)
            if not was_complete
        )
        batch_started = time.monotonic()
        prepared = runtime.task_adapter.prepare_episodes(
            [
                {
                    "suite": request.suite,
                    "task_id": request.task_id,
                    "init_state_id": request.init_state_id,
                }
                for request in pending_requests
            ]
        )
        if len(prepared) != len(pending_requests):
            raise WriterModelError("Writer generation batch coverage changed")
        if not callable(
            getattr(runtime.task_adapter, "last_generation_batch_profile", None)
        ):
            raise WriterModelError("Writer generation batch lacks video-length evidence")
        video_profile = runtime.task_adapter.last_generation_batch_profile()
        expected_profile = tuple(
            (request.suite, request.task_id, request.init_state_id)
            for request in pending_requests
        )
        observed_profile = tuple(
            (
                str(row["suite"]),
                int(row["task_id"]),
                int(row["init_state_id"]),
            )
            for row in video_profile
        )
        if observed_profile != expected_profile:
            raise WriterModelError("Writer generation video-length ownership changed")
        staged_states = stage_writer_lora_states_to_cpu(
            tuple(item.state for item in prepared)
        )
        batch_seconds = time.monotonic() - batch_started
        generated_batches += 1
        reused_entries += sum(complete)
        for position, (request, item, state) in enumerate(
            zip(pending_requests, prepared, staged_states, strict=True)
        ):
            evidence = dict(item.evidence)
            evidence["writer_generation_seconds"] = (
                batch_seconds / len(pending_requests)
            )
            write_writer_cache_entry(
                runtime.contract,
                request,
                state=state,
                evidence=evidence,
                generation={
                    "generator_worker_id": runtime.worker_id,
                    "generator_index": generator_index,
                    "batch_ordinal": batch_ordinal,
                    "position_in_batch": position,
                    "batch_entry_ids": [value.entry_id for value in pending_requests],
                    "batch_size": len(pending_requests),
                    "batch_wall_seconds": batch_seconds,
                    "raw_frames": int(video_profile[position]["raw_frames"]),
                    "sampled_frames": int(
                        video_profile[position]["sampled_frames"]
                    ),
                },
                lora_contract=runtime.task_adapter.lora_contract,
            )
            generated_entries += 1
        batch_rows.append(
            {
                "batch_ordinal": batch_ordinal,
                "entry_ids": [request.entry_id for request in pending_requests],
                "batch_size": len(pending_requests),
                "raw_frame_counts": [
                    int(row["raw_frames"]) for row in video_profile
                ],
                "sampled_frame_counts": [
                    int(row["sampled_frames"]) for row in video_profile
                ],
                "wall_seconds": batch_seconds,
            }
        )
        del item, state, prepared, staged_states, video_profile
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
            "redundant_writer_forwards": 0,
            "batch_shape_bf16_roundoff_accepted": True,
            "batches": batch_rows,
        },
    )
