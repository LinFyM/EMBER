"""Inference-consistent PI05 diagnostics on the sealed validation panel.

This module is owned by :mod:`ember.writer.validation`.  It deliberately reads
only public LoRA cache tensors; retired Writer constructors are never imported.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.lora import (
    canonical_contract_sha256,
    lora_state_sha256,
    validate_lora_state,
)
from ember.pi05_eval_contract import load_run_contract
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
)
from ember.writer.model import WriterModelError


ENDPOINT_RUN_SCHEMA = "ember_pi05_endpoint_validation_run_v1"
ENDPOINT_SUMMARY_SCHEMA = "ember_pi05_endpoint_validation_summary_v1"
PORTABLE_CACHE_SCHEMA = "ember_pi05_endpoint_public_lora_cache_v1"
ENDPOINT_NOISE_SCHEMA = "ember_pi05_inference_consistent_surrogate_v1"
SEALED_PANEL_PAYLOAD_SHA256 = (
    "97ba7b95c48124858f01b50a1400172ad69eae62e7796f54357caed140174b4d"
)
INFERENCE_TIMES = tuple(1.0 - step / 10.0 for step in range(10))
METRICS = (
    "rollout10_executed5_valid_normalized_mse",
    "rollout10_full50_padded_normalized_mse",
    "rollout10_full50_valid_normalized_mse",
    "rollout10_prefix10_valid_normalized_mse",
    "grid10_teacher_bridge_flow_mse",
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class EndpointLoRAEntry:
    path: Path
    bytes: int
    file_sha256: str
    state_sha256: str


@dataclass(frozen=True)
class EndpointCandidate:
    family: str
    candidate_id: str
    checkpoint_cursor: int
    correct400: int
    task_breadth: int
    evaluation_root: Path
    run_contract_file_sha256: str
    run_contract_sha256: str
    results_file_sha256: str
    cache_manifest_file_sha256: str
    entries: Mapping[tuple[int, int], EndpointLoRAEntry]

    def record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "checkpoint_cursor": self.checkpoint_cursor,
            "correct400": self.correct400,
            "task_breadth": self.task_breadth,
            "evaluation_root": str(self.evaluation_root),
            "run_contract_file_sha256": self.run_contract_file_sha256,
            "run_contract_sha256": self.run_contract_sha256,
            "results_file_sha256": self.results_file_sha256,
            "cache_manifest_file_sha256": self.cache_manifest_file_sha256,
        }


def endpoint_schedule(num_steps: int = 10) -> tuple[tuple[float, ...], float]:
    if num_steps != 10:
        raise WriterModelError("endpoint diagnostic requires exactly ten Euler steps")
    return INFERENCE_TIMES, -0.1


def endpoint_noise_seed(
    panel_payload_sha256: str, row: Mapping[str, Any]
) -> int:
    fields = (
        ENDPOINT_NOISE_SCHEMA,
        panel_payload_sha256,
        int(row["global_task_id"]),
        int(row["video_group"]),
        int(row["query_ordinal"]),
        int(row["action_demo_index"]),
        int(row["action_frame_index"]),
    )
    encoded = json.dumps(fields, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def endpoint_noise(
    panel_payload_sha256: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[torch.Tensor, tuple[int, ...]]:
    tensors, seeds = [], []
    for row in rows:
        seed = endpoint_noise_seed(panel_payload_sha256, row)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        tensors.append(torch.randn((50, 32), generator=generator, dtype=torch.float32))
        seeds.append(seed)
    if not tensors:
        raise WriterModelError("endpoint diagnostic received an empty query group")
    return torch.stack(tensors), tuple(seeds)


def exact_endpoint_actions(
    policy: Any,
    batch: Mapping[str, torch.Tensor],
    noise: torch.Tensor,
) -> torch.Tensor:
    """Call the pinned PI05 recursive sampler under a strict no-grad contract."""

    with torch.inference_mode():
        value = policy.predict_action_chunk(dict(batch), noise=noise, num_steps=10)
    expected = (int(noise.shape[0]), 50, 7)
    if tuple(value.shape) != expected or value.requires_grad:
        raise WriterModelError("PI05 endpoint sampler output contract changed")
    return value


def _masked_row_mse(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or mask.shape != values.shape[:2] or values.shape[-1] != 7:
        raise WriterModelError("endpoint metric tensor shape changed")
    counts = mask.sum(dim=1)
    if bool((counts <= 0).any()):
        raise WriterModelError("endpoint metric has a row without valid actions")
    per_dimension = (values * mask[..., None]).sum(dim=1) / counts[:, None]
    return per_dimension.mean(dim=1), per_dimension


def endpoint_metric_rows(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    action_is_pad: torch.Tensor,
    grid_flow_losses: torch.Tensor,
) -> tuple[dict[str, Any], ...]:
    if (
        predicted.shape != teacher.shape
        or predicted.ndim != 3
        or predicted.shape[1:] != (50, 7)
        or action_is_pad.shape != predicted.shape[:2]
        or grid_flow_losses.shape != predicted.shape
    ):
        raise WriterModelError("endpoint output, teacher, padding, or grid shape changed")
    squared = (predicted.float() - teacher.float()).square()
    valid = ~action_is_pad.bool()
    positions = torch.arange(50, device=valid.device)[None]
    masks = {
        METRICS[0]: valid & (positions < 5),
        METRICS[1]: torch.ones_like(valid),
        METRICS[2]: valid,
        METRICS[3]: valid & (positions < 10),
        METRICS[4]: torch.ones_like(valid),
    }
    sources = {name: squared for name in METRICS[:4]}
    sources[METRICS[4]] = grid_flow_losses.float()
    values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        name: _masked_row_mse(sources[name], masks[name]) for name in METRICS
    }
    result = []
    for index in range(predicted.shape[0]):
        result.append(
            {
                "valid_action_steps": int(valid[index].sum()),
                "executed5_valid_steps": int(masks[METRICS[0]][index].sum()),
                "prefix10_valid_steps": int(masks[METRICS[3]][index].sum()),
                "metrics": {
                    name: {
                        "mse": float(values[name][0][index]),
                        "per_action_dimension_mse": [
                            float(value) for value in values[name][1][index]
                        ],
                    }
                    for name in METRICS
                },
            }
        )
    return tuple(result)


def _validated_payload(path: Path, schema: str) -> dict[str, Any]:
    payload = read_json(path)
    digest = payload.get("canonical_payload_sha256")
    unhashed = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    if payload.get("schema_version") != schema or canonical_hash(unhashed) != digest:
        raise WriterModelError(f"endpoint artifact changed: {path}")
    return payload


def _validate_results(root: Path, contract: Mapping[str, Any]) -> tuple[int, int, str]:
    path = root / "results.json"
    results = read_json(path)
    rows = results.get("rows", [])
    keys = [
        (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("init_state_id", -1)),
        )
        for row in rows
    ]
    expected = [
        (str(task["suite"]), int(task["task_id"]), int(state))
        for task in contract["tasks"]
        for state in task["init_state_ids"]
    ]
    successes = sum(bool(row.get("success")) for row in rows)
    per_task = results.get("per_task", [])
    valid = (
        results.get("schema_version") == "ember_pi05_target_eval_results_v1"
        and results.get("contract_sha256") == contract["contract_sha256"]
        and results.get("mode") == "formal"
        and results.get("role") == "validation"
        and results.get("adapter") == contract.get("adapter")
        and sorted(keys) == sorted(expected)
        and len(keys) == len(set(keys)) == 400
        and int(results.get("overall", {}).get("successes", -1)) == successes
        and int(results.get("overall", {}).get("episodes", -1)) == 400
        and len(per_task) == 8
    )
    if not valid:
        raise WriterModelError(f"endpoint candidate results changed: {root}")
    breadth = sum(int(row.get("successes", 0)) > 0 for row in per_task)
    return successes, breadth, sha256_file(path)


def _candidate_task_map(
    contract: Mapping[str, Any],
    panel_tasks: Sequence[int],
) -> dict[tuple[str, int], int]:
    adapter = contract.get("adapter", {})
    mapping = adapter.get("task_video_mapping", [])
    result = {
        (str(row["suite"]), int(row["task_id"])): int(row["language_global_task_id"])
        for row in mapping
    }
    valid = (
        len(result) == len(mapping) == 8
        and set(result.values()) == set(panel_tasks)
        and set(result)
        == {
            (str(row["suite"]), int(row["task_id"]))
            for row in contract.get("tasks", [])
        }
        and all(suite in SUITES for suite, _task_id in result)
        and all(
            global_task_id == SUITES.index(suite) * 10 + task_id
            for (suite, task_id), global_task_id in result.items()
        )
        and all(
            int(row["language_global_task_id"]) == int(row["video_global_task_id"])
            and row["suite"] == row["video_suite"]
            and int(row["task_id"]) == int(row["video_task_id"])
            and row["language_split_role"] == row["video_split_role"] == "validation"
            for row in mapping
        )
    )
    if not valid:
        raise WriterModelError("endpoint candidate task/video mapping changed")
    return result


def _validate_candidate_contract(
    contract: Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    lora_sha256: str,
    data_root: Path,
    panel_tasks: Sequence[int],
) -> dict[tuple[str, int], int]:
    adapter = contract.get("adapter", {})
    policy = contract.get("policy", {})
    tasks = contract.get("tasks", [])
    valid = (
        contract.get("mode") == "formal"
        and contract.get("role") == "validation"
        and contract.get("model") == source
        and contract.get("tokenizer") == tokenizer
        and contract.get("normalization", {}).get(
            "validation_or_test_numeric_reads"
        )
        == 0
        and adapter.get("kind") == "as_writer"
        and adapter.get("video_condition") == "correct"
        and adapter.get("teacher_action_values_read_by_evaluator") == 0
        and adapter.get("lora_contract_sha256") == lora_sha256
        and Path(
            str(adapter.get("video_data", {}).get("root", ""))
        ).resolve()
        == data_root
        and adapter.get("video_schedule", {}).get("sampling_mode")
        == "without_replacement"
        and int(adapter.get("video_schedule", {}).get("demo_count", -1)) == 50
        and contract.get("writer_lora_execution", {}).get("b_scale") == 1.0
        and policy.get("chunk_size") == 50
        and policy.get("num_inference_steps") == 10
        and policy.get("replan_steps") == 5
        and policy.get("action_dim") == 7
        and len(tasks) == 8
        and all(
            task.get("split_role") == "validation"
            and list(map(int, task.get("init_state_ids", []))) == list(range(50))
            for task in tasks
        )
    )
    if not valid:
        raise WriterModelError("endpoint candidate contract changed")
    return _candidate_task_map(contract, panel_tasks)


def _full_cache_header(
    root: Path,
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, Path, Sequence[Mapping[str, Any]]]:
    descriptor = contract.get("writer_lora_cache", {})
    generation = descriptor.get("generation_recipe", {})
    identity = {
        "schema_version": "ember_pi05_writer_lora_cache_v2",
        "adapter": dict(contract["adapter"]),
        "model": dict(contract["model"]),
        "tokenizer": dict(contract["tokenizer"]),
        "tasks": [
            {
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "init_state_ids": list(map(int, row["init_state_ids"])),
            }
            for row in contract["tasks"]
        ],
        "policy": dict(contract["policy"]),
        "rng": {"inference_seed": int(contract["rng"]["inference_seed"])},
        "generation_recipe": dict(generation),
    }
    cache_root = Path(str(descriptor.get("root", ""))).resolve()
    manifest_path = cache_root / "cache_manifest.json"
    manifest = _validated_payload(
        manifest_path, "ember_pi05_writer_lora_cache_manifest_v2"
    )
    entries = manifest.get("entries", [])
    expected = {
        "descriptor_schema": "ember_pi05_writer_lora_cache_v2",
        "descriptor_identity": identity,
        "descriptor_identity_sha256": canonical_hash(identity),
        "cache_root": root / "writer_lora_cache",
        "manifest_cache_identity": descriptor.get("identity_sha256"),
        "manifest_descriptor": descriptor,
        "manifest_entry_ids": [row.get("entry_id") for row in entries],
        "entry_count": int(descriptor.get("entry_count", -1)),
        "nonempty": True,
    }
    observed = {
        "descriptor_schema": descriptor.get("schema_version"),
        "descriptor_identity": descriptor.get("identity"),
        "descriptor_identity_sha256": descriptor.get("identity_sha256"),
        "cache_root": cache_root,
        "manifest_cache_identity": manifest.get("cache_identity_sha256"),
        "manifest_descriptor": manifest.get("descriptor"),
        "manifest_entry_ids": manifest.get("entry_ids"),
        "entry_count": len(entries),
        "nonempty": bool(entries),
    }
    if observed != expected:
        raise WriterModelError("endpoint candidate cache manifest changed")
    return descriptor, cache_root, manifest_path, entries


def _full_cache_entry(
    summary: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    cache_root: Path,
    task_map: Mapping[tuple[str, int], int],
    checkpoint: Mapping[str, Any],
    lora_sha256: str,
) -> tuple[tuple[int, int], EndpointLoRAEntry]:
    entry_id = str(summary["entry_id"])
    if re.fullmatch(r"[a-z0-9_]+", entry_id) is None:
        raise WriterModelError("endpoint candidate cache entry ID is unsafe")
    entry_root = cache_root / "entries" / entry_id
    record_path = entry_root / "entry.json"
    record = _validated_payload(
        record_path, "ember_pi05_writer_lora_cache_entry_v2"
    )
    request = record.get("request", {})
    evidence = record.get("evidence", {})
    language_task = (
        str(request.get("language_suite")),
        int(request.get("language_task_id", -1)),
    )
    global_task_id = task_map.get(language_task, -1)
    demo = int(request.get("teacher_demo_index", -1))
    expected = {
        "cache_identity": descriptor["identity_sha256"],
        "entry_id": entry_id,
        "lora_contract": lora_sha256,
        "record_bytes": record_path.stat().st_size,
        "record_sha256": sha256_file(record_path),
        "lora_state": record.get("lora_state_sha256"),
        "video_condition": "correct",
        "order_transform": "forward",
        "video_matches_language": True,
        "evidence_condition": "correct",
        "language_global_task_id": global_task_id,
        "video_global_task_id": global_task_id,
        "teacher_demo_index": demo,
        "checkpoint_cursor": int(checkpoint["cursor"]),
        "checkpoint_manifest": checkpoint["manifest_file_sha256"],
        "writer_state": checkpoint["writer_state_sha256"],
        "evidence_lora": record.get("lora_state_sha256"),
    }
    observed = {
        "cache_identity": record.get("cache_identity_sha256"),
        "entry_id": record.get("entry_id"),
        "lora_contract": record.get("lora_contract_sha256"),
        "record_bytes": summary.get("record_bytes"),
        "record_sha256": summary.get("record_sha256"),
        "lora_state": summary.get("lora_state_sha256"),
        "video_condition": request.get("video_condition"),
        "order_transform": request.get("order_transform"),
        "video_matches_language": (
            request.get("video_suite") == request.get("language_suite")
            and int(request.get("video_task_id", -2))
            == int(request.get("language_task_id", -1))
        ),
        "evidence_condition": evidence.get("condition"),
        "language_global_task_id": int(
            evidence.get("language_global_task_id", -1)
        ),
        "video_global_task_id": int(evidence.get("video_global_task_id", -2)),
        "teacher_demo_index": int(evidence.get("teacher_demo_index", -1)),
        "checkpoint_cursor": int(
            evidence.get("writer_checkpoint_cursor", -1)
        ),
        "checkpoint_manifest": evidence.get(
            "writer_checkpoint_manifest_sha256"
        ),
        "writer_state": evidence.get("writer_state_sha256"),
        "evidence_lora": evidence.get("lora_sha256"),
    }
    if observed != expected or global_task_id < 0 or not 0 <= demo < 50:
        raise WriterModelError("endpoint candidate cache entry changed")
    lora_file = record["lora_file"]
    return (
        (global_task_id, demo),
        EndpointLoRAEntry(
            path=entry_root / "lora.safetensors",
            bytes=int(lora_file["bytes"]),
            file_sha256=str(lora_file["sha256"]),
            state_sha256=str(record["lora_state_sha256"]),
        ),
    )


def _full_cache_entries(
    root: Path,
    contract: Mapping[str, Any],
    task_map: Mapping[tuple[str, int], int],
    panel_keys: set[tuple[int, int]],
    lora: Any,
) -> tuple[dict[tuple[int, int], EndpointLoRAEntry], str]:
    descriptor, cache_root, manifest_path, entries = _full_cache_header(
        root, contract
    )
    checkpoint = contract["adapter"]["checkpoint"]
    lora_sha256 = canonical_contract_sha256(lora)
    selected: dict[tuple[int, int], EndpointLoRAEntry] = {}
    seen: dict[int, set[int]] = {
        task_id: set() for task_id in task_map.values()
    }
    for summary in entries:
        key, entry = _full_cache_entry(
            summary,
            descriptor,
            cache_root,
            task_map,
            checkpoint,
            lora_sha256,
        )
        global_task_id, demo = key
        if demo in seen[global_task_id]:
            raise WriterModelError(
                "endpoint candidate cache duplicated a teacher video"
            )
        seen[global_task_id].add(demo)
        if key in panel_keys:
            selected[key] = entry
    if (
        any(values != set(range(50)) for values in seen.values())
        or set(selected) != panel_keys
    ):
        raise WriterModelError(
            "endpoint candidate cache does not cover the sealed panel"
        )
    return selected, sha256_file(manifest_path)

def _portable_cache_entries(
    manifest_path: Path,
    evaluation_root: Path,
    contract: Mapping[str, Any],
    panel_keys: set[tuple[int, int]],
    lora: Any,
) -> tuple[dict[tuple[int, int], EndpointLoRAEntry], str]:
    """Load 64 public LoRAs emitted by the checkpoint's historical code.

    The manifest pins the observed correct400 run, its exact adapter, the
    originating Writer commit, the sealed panel, and per-entry generation
    evidence.  It therefore carries tensors across Git history without ever
    instantiating a retired Writer in the current process.
    """

    manifest = _validated_payload(manifest_path, PORTABLE_CACHE_SCHEMA)
    candidate = manifest.get("candidate", {})
    wall = manifest.get("information_wall", {})
    if (
        Path(str(candidate.get("evaluation_root", ""))).resolve() != evaluation_root
        or candidate.get("run_contract_file_sha256")
        != sha256_file(evaluation_root / "run_contract.json")
        or candidate.get("run_contract_sha256") != contract["contract_sha256"]
        or candidate.get("results_file_sha256")
        != sha256_file(evaluation_root / "results.json")
        or candidate.get("adapter_sha256")
        != canonical_hash(dict(contract["adapter"]))
        or candidate.get("writer_constructor_git_commit")
        != contract["adapter"].get("training_run", {}).get("git_commit")
        or manifest.get("panel_manifest_payload_sha256") != SEALED_PANEL_PAYLOAD_SHA256
        or manifest.get("lora_contract_sha256") != canonical_contract_sha256(lora)
        or wall
        != {
            "validation_action_values_read_during_generation": 0,
            "test_action_reads": 0,
            "test_video_value_reads": 0,
        }
    ):
        raise WriterModelError("portable endpoint cache authority changed")
    selected = {}
    checkpoint = contract["adapter"]["checkpoint"]
    for row in manifest.get("entries", []):
        key = (
            int(row.get("global_task_id", -1)),
            int(row.get("teacher_demo_index", -1)),
        )
        evidence = row.get("generation_evidence", {})
        file_record = row.get("lora_file", {})
        path = (manifest_path.parent / str(file_record.get("path", ""))).resolve()
        valid = (
            key in panel_keys
            and key not in selected
            and path.is_relative_to(manifest_path.parent.resolve())
            and int(evidence.get("language_global_task_id", -1)) == key[0]
            and int(evidence.get("video_global_task_id", -2)) == key[0]
            and int(evidence.get("teacher_demo_index", -1)) == key[1]
            and evidence.get("condition") == "correct"
            and int(evidence.get("writer_checkpoint_cursor", -1))
            == int(checkpoint["cursor"])
            and evidence.get("writer_checkpoint_manifest_sha256")
            == checkpoint["manifest_file_sha256"]
            and evidence.get("writer_state_sha256") == checkpoint["writer_state_sha256"]
            and evidence.get("lora_sha256") == row.get("lora_state_sha256")
        )
        if not valid:
            raise WriterModelError("portable endpoint cache entry changed")
        selected[key] = EndpointLoRAEntry(
            path=path,
            bytes=int(file_record.get("bytes", -1)),
            file_sha256=str(file_record.get("sha256", "")),
            state_sha256=str(row.get("lora_state_sha256", "")),
        )
    if set(selected) != panel_keys:
        raise WriterModelError("portable endpoint cache does not cover the sealed panel")
    return selected, sha256_file(manifest_path)


def _verify_lora_entry(
    entry: EndpointLoRAEntry,
    lora: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if (
        not entry.path.is_file()
        or entry.path.stat().st_size != entry.bytes
        or sha256_file(entry.path) != entry.file_sha256
    ):
        raise WriterModelError(f"endpoint public LoRA file changed: {entry.path}")
    state = load_file(str(entry.path), device=str(device))
    validate_lora_state(state, lora)
    if lora_state_sha256(state) != entry.state_sha256:
        raise WriterModelError(f"endpoint public LoRA state changed: {entry.path}")
    return state


def parse_endpoint_candidate_specs(
    values: Sequence[str],
) -> tuple[tuple[str, Path, Path | None], ...]:
    """Parse full retained caches or portable historical 64-LoRA manifests."""

    result = []
    for value in values:
        if "=" not in value:
            raise WriterModelError(
                "endpoint candidate must be "
                "FAMILY=EVALUATION_ROOT[::PORTABLE_MANIFEST]"
            )
        family, location = value.split("=", 1)
        if re.fullmatch(r"[a-z0-9_]+", family) is None:
            raise WriterModelError("endpoint candidate family is unsafe")
        parts = location.split("::")
        if len(parts) not in {1, 2}:
            raise WriterModelError("endpoint candidate portable-cache syntax is invalid")
        result.append(
            (
                family,
                Path(parts[0]).resolve(),
                Path(parts[1]).resolve() if len(parts) == 2 else None,
            )
        )
    if not result:
        raise WriterModelError("endpoint diagnostic requires at least one candidate")
    return tuple(result)


def _load_candidates(
    specs: Sequence[tuple[str, Path, Path | None]],
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    lora: Any,
    data_root: Path,
    verify_files: bool,
) -> tuple[EndpointCandidate, ...]:
    panel_tasks = tuple(map(int, manifest["task_ids"]))
    panel_keys = {
        (int(row["global_task_id"]), int(row["teacher_demo_index"]))
        for row in manifest["rows"]
    }
    candidates = []
    for family, root, portable in specs:
        contract_path = root / "run_contract.json"
        contract = load_run_contract(contract_path)
        task_map = _validate_candidate_contract(
            contract,
            source,
            tokenizer,
            canonical_contract_sha256(lora),
            data_root,
            panel_tasks,
        )
        correct400, breadth, results_sha = _validate_results(root, contract)
        if portable is None:
            entries, cache_sha = _full_cache_entries(
                root, contract, task_map, panel_keys, lora
            )
        else:
            entries, cache_sha = _portable_cache_entries(
                portable, root, contract, panel_keys, lora
            )
        if verify_files:
            for entry in entries.values():
                _verify_lora_entry(entry, lora, torch.device("cpu"))
        cursor = int(contract["adapter"]["checkpoint"]["cursor"])
        candidates.append(
            EndpointCandidate(
                family=family,
                candidate_id=f"{family}_step{cursor:08d}",
                checkpoint_cursor=cursor,
                correct400=correct400,
                task_breadth=breadth,
                evaluation_root=root,
                run_contract_file_sha256=sha256_file(contract_path),
                run_contract_sha256=str(contract["contract_sha256"]),
                results_file_sha256=results_sha,
                cache_manifest_file_sha256=cache_sha,
                entries=entries,
            )
        )
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise WriterModelError("endpoint candidates are duplicated")
    return tuple(candidates)
