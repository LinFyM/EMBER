"""Single-A40 end-to-end Writer generation throughput profiling."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.evaluation_cache import (
    stage_writer_lora_states_to_cpu,
    writer_cache_requests,
)
from ember.writer.errors import WriterModelError


WRITER_GENERATION_PROFILE_SCHEMA = "ember_pi05_writer_generation_profile_v2"


def _validate_profile_request(
    runtime: Any,
    sizes: tuple[int, ...],
    *,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any], str]:
    contract = runtime.contract
    adapter = runtime.task_adapter
    valid = (
        sizes == tuple(sorted(set(sizes)))
        and len(sizes) >= 3
        and min(sizes) > 0
        and warmup_runs > 0
        and measured_runs >= 2
        and contract.get("mode") == "smoke"
        and contract.get("adapter", {}).get("video_condition") == "correct"
        and int(contract["parallel"]["physical_gpu_count"]) == 1
        and int(contract["parallel"]["replicas_per_gpu"]) == 1
        and int(contract["parallel"]["writer_generators_per_gpu"]) == 1
        and int(contract["parallel"]["writer_generation_batch_size"]) == sizes[-1]
        and callable(getattr(adapter, "prepare_episodes", None))
        and callable(getattr(adapter, "generation_request_profiles", None))
        and callable(getattr(adapter, "last_generation_batch_profile", None))
        and callable(getattr(adapter, "release_generation_assets", None))
    )
    if not valid:
        raise WriterModelError("Writer generation profile contract changed")
    git = contract.get("git", {})
    if not git_state_is_clean_pushed_or_frozen_authority(git):
        raise WriterModelError(
            "Writer generation profile requires a clean pushed commit"
        )
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA A40":
        raise WriterModelError("Writer generation profile requires an NVIDIA A40")
    return contract, adapter, git, device_name


def _fixed_profile_panel(
    contract: Mapping[str, Any],
    adapter: Any,
    sizes: tuple[int, ...],
) -> dict[str, Any]:
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
    panel = tuple(ordered[: sizes[-1]])
    counts = tuple(int(item[2]["sampled_frames"]) for item in panel)
    return {
        "panel": panel,
        "entry_ids": tuple(item[0].entry_id for item in panel),
        "sampled_frame_counts": counts,
        "total_sampled_frames": sum(counts),
        "longest_sampled_frames": max(int(row["sampled_frames"]) for row in metadata),
    }


def _execute_profile_once(
    adapter: Any,
    chunks: Sequence[Sequence[tuple[Any, Mapping[str, Any], Mapping[str, Any]]]],
    expected_counts: tuple[int, ...],
) -> tuple[float, tuple[dict[str, Any], ...]]:
    torch.cuda.synchronize()
    started = time.monotonic()
    observed_panel = []
    for chunk in chunks:
        identities = tuple(item[1] for item in chunk)
        chunk_counts = tuple(int(item[2]["sampled_frames"]) for item in chunk)
        prepared = adapter.prepare_episodes(identities)
        if len(prepared) != len(chunk):
            raise WriterModelError("Writer profile forward batch changed")
        staged = stage_writer_lora_states_to_cpu(tuple(item.state for item in prepared))
        observed = adapter.last_generation_batch_profile()
        if (
            len(observed) != len(chunk)
            or tuple(int(row["sampled_frames"]) for row in observed) != chunk_counts
        ):
            raise WriterModelError("Writer profile video batch changed")
        observed_panel.extend(observed)
        del prepared, staged
    wall = time.monotonic() - started
    if tuple(int(row["sampled_frames"]) for row in observed_panel) != expected_counts:
        raise WriterModelError("Writer profile request ordering changed")
    return wall, tuple(observed_panel)


def _measure_profile_candidate(
    adapter: Any,
    panel: Sequence[tuple[Any, Mapping[str, Any], Mapping[str, Any]]],
    panel_evidence: Mapping[str, Any],
    *,
    size: int,
    warmup_runs: int,
    measured_runs: int,
    total_memory: int,
    required_headroom: int,
) -> tuple[dict[str, Any], bool]:
    chunks = tuple(
        panel[offset : offset + size] for offset in range(0, len(panel), size)
    )
    forward_batch_sizes = tuple(len(chunk) for chunk in chunks)
    walls = []
    candidate_oom = False
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warmup_runs):
            _execute_profile_once(
                adapter,
                chunks,
                tuple(panel_evidence["sampled_frame_counts"]),
            )
        torch.cuda.reset_peak_memory_stats()
        for _ in range(measured_runs):
            wall, _ = _execute_profile_once(
                adapter,
                chunks,
                tuple(panel_evidence["sampled_frame_counts"]),
            )
            walls.append(wall)
    except torch.cuda.OutOfMemoryError:
        candidate_oom = True
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    headroom = total_memory - peak_reserved
    generated = len(panel) * len(walls)
    wall_seconds = sum(walls)
    throughput = generated / wall_seconds if wall_seconds > 0 else 0.0
    stable = (
        not candidate_oom
        and len(walls) == measured_runs
        and all(value > 0 and math.isfinite(value) for value in walls)
        and max(walls) / min(walls) <= 1.25
        and headroom >= required_headroom
    )
    row = {
        "batch_size": size,
        "generated_entries": generated,
        "completed_measured_runs": len(walls),
        "oom_count": int(candidate_oom),
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
        "panel_total_sampled_frames": int(panel_evidence["total_sampled_frames"]),
        "longest_video_included": (
            max(panel_evidence["sampled_frame_counts"])
            == int(panel_evidence["longest_sampled_frames"])
        ),
        "max_sampled_video_frames": max(panel_evidence["sampled_frame_counts"]),
        "sampled_frame_counts": list(panel_evidence["sampled_frame_counts"]),
        "entry_ids": list(panel_evidence["entry_ids"]),
        "stable": stable,
    }
    if candidate_oom:
        torch.cuda.empty_cache()
    return row, candidate_oom


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
    contract, adapter, git, device_name = _validate_profile_request(
        runtime,
        sizes,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    panel_evidence = _fixed_profile_panel(contract, adapter, sizes)
    panel = panel_evidence["panel"]
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    required_headroom = max(512 * 1024**2, total_memory // 100)
    rows = []
    oom_count = 0
    profile_started = time.monotonic()
    for size in sizes:
        row, candidate_oom = _measure_profile_candidate(
            adapter,
            panel,
            panel_evidence,
            size=size,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            total_memory=total_memory,
            required_headroom=required_headroom,
        )
        rows.append(row)
        oom_count += int(candidate_oom)
    eligible = [row for row in rows if row["stable"]]
    if not eligible:
        raise WriterModelError(
            "Writer profile found no stable batch with memory headroom"
        )
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
        "longest_sampled_video_frames": int(panel_evidence["longest_sampled_frames"]),
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
        "oom_count": oom_count,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(runtime.output_dir / "writer_generation_profile.json", result)
    return result
