#!/usr/bin/env python3
"""Fold a sealed historical rank-32 episode-LoRA cache into canonical rank 16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

from ember.eval_adapters import ARCHIVAL_WRITER_CACHE_KIND
from ember.pi05_eval_contract import load_run_contract
from ember.writer.archival_projection import (
    ARCHIVAL_PROJECTION_SCHEMA,
    archival_lora_reference,
    expected_archival_episode_evidence,
    fold_repeated_a_rank32_state,
    load_archival_lora_contract,
    source_entry_lora_path,
    source_entry_record,
)
from ember.writer.evaluation_cache import (
    assigned_writer_cache_requests,
    finalize_writer_cache,
    validate_writer_cache_manifest,
    write_generator_marker,
    write_writer_cache_entry,
    writer_cache_entry_is_complete,
    writer_cache_manifest_path,
    writer_cache_requests,
)
from ember.writer.errors import WriterModelError


CONVERSION_INVOCATION_ID = "a11ca1c0a11ca1c0a11ca1c0a11ca1c0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Prepared PI05 evaluation root whose archival cache will be populated.",
    )
    return parser.parse_args()


def _generator_worker_ids(contract: dict) -> tuple[str, ...]:
    count = int(contract["parallel"]["writer_generators_per_gpu"])
    return tuple(
        f"{gpu}-r{replica}"
        for gpu in contract["parallel"]["physical_gpu_ids"]
        for replica in range(count)
    )


def import_cache(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    adapter = contract.get("adapter", {})
    if adapter.get("kind") != ARCHIVAL_WRITER_CACHE_KIND:
        raise WriterModelError("evaluation run is not an archival Writer projection")
    manifest_path = writer_cache_manifest_path(contract)
    if manifest_path.is_file():
        manifest = validate_writer_cache_manifest(
            contract,
            verify_entry_files=True,
        )
        return {
            "schema_version": ARCHIVAL_PROJECTION_SCHEMA,
            "output_dir": str(output_dir),
            "cache_root": str(manifest_path.parent),
            "converted_entries": 0,
            "reused_entries": len(manifest["entry_ids"]),
            "tensor_file_bytes": int(manifest["tensor_file_bytes"]),
            "sealed": True,
        }

    target_lora = load_archival_lora_contract(adapter)
    converted = reused = 0
    for request in writer_cache_requests(contract):
        if writer_cache_entry_is_complete(contract, request):
            reused += 1
            continue
        source_record = source_entry_record(
            adapter,
            request.suite,
            request.task_id,
            request.init_state_id,
        )
        source_state = load_file(
            str(
                source_entry_lora_path(
                    adapter,
                    request.suite,
                    request.task_id,
                    request.init_state_id,
                )
            ),
            device="cpu",
        )
        target_state = fold_repeated_a_rank32_state(
            source_state,
            target_contract=target_lora,
        )
        lora_reference = archival_lora_reference(
            adapter,
            suite=request.suite,
            task_id=request.task_id,
            init_state_id=request.init_state_id,
        )
        evidence = expected_archival_episode_evidence(
            adapter,
            suite=request.suite,
            task_id=request.task_id,
            init_state_id=request.init_state_id,
            lora_reference=lora_reference,
        )
        write_writer_cache_entry(
            contract,
            request,
            state=target_state,
            evidence=evidence,
            generation={
                "kind": "archival_rank_projection",
                "projection": adapter["archival_projection"]["projection"],
                "source_entry_id": source_record["entry_id"],
                "source_generation": source_record.get("generation"),
                "training_performed": False,
            },
            lora_contract=target_lora,
        )
        converted += 1

    worker_ids = _generator_worker_ids(contract)
    for generator_index, worker_id in enumerate(worker_ids):
        assigned = assigned_writer_cache_requests(
            contract,
            generator_index=generator_index,
        )
        write_generator_marker(
            contract,
            invocation_id=CONVERSION_INVOCATION_ID,
            worker_id=worker_id,
            generator_index=generator_index,
            summary={
                "source_policy_reused_for_rollout": True,
                "writer_modules_released": True,
                "archival_projection": True,
                "assigned_entries": len(assigned),
                "training_performed": False,
            },
        )
    manifest = finalize_writer_cache(
        contract,
        invocation_id=CONVERSION_INVOCATION_ID,
        worker_ids=worker_ids,
    )
    return {
        "schema_version": ARCHIVAL_PROJECTION_SCHEMA,
        "output_dir": str(output_dir),
        "cache_root": str(manifest_path.parent),
        "converted_entries": converted,
        "reused_entries": reused,
        "tensor_file_bytes": int(manifest["tensor_file_bytes"]),
        "sealed": True,
    }


def main() -> None:
    print(json.dumps(import_cache(parse_args().output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
