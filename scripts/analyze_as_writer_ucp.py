#!/usr/bin/env python3
"""Fail-closed, no-rollout internal analysis for one formal UCP checkpoint.

This entrypoint reuses the canonical Writer, source-policy, data, checkpoint, and
video-schedule owners.  It never opens an ``actions`` dataset and never sends a
teacher state to the Writer.  The only state read is a separately recorded,
fixed policy query used after LoRA generation to measure functional action
changes without entering LIBERO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
    resolve_role_task_keys,
)
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import (
    barrier,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_setup import initialize_distributed, load_stats
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.as_config import load_writer_config
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.inference import (
    expected_writer_episode_evidence,
    inspect_as_writer_evaluation,
    writer_shuffled_frame_permutation,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.ucp_analysis import (
    CONDITIONS,
    COUNTERFACTUAL_CONTRACT,
    build_initial_program,
    compile_with_target_identity_permutation,
    coordinate_summary,
    decode_coordinates,
    effective_ba_error,
    effective_metrics,
    effective_variance,
    fixed_stream_counterfactual,
    lora_geometry,
    mapping_metrics,
    pack_flat,
    program_signature,
    rank_gauge_permute,
    reader_attention,
    reader_attention_summary,
    relative_metrics,
    split_state,
    type_ablation,
    validate_analysis_provenance,
    variance_metrics,
)
from ember.writer.ucp_analysis_summary import (
    matched_diagnostics,
    summarize_records,
    validate_finite_tree,
    validate_rank_payloads,
)
from ember.writer.ucp_analysis_runtime import fixed_policy_query, policy_action
from ember.writer.validation import _build_models


RUN_SCHEMA = "ember_pi05_ucp_internal_analysis_run_v1"
RESULT_SCHEMA = "ember_pi05_ucp_internal_analysis_v1"
EXPECTED_ARCHITECTURE = "pi05_unified_causal_program_target_rank_reader_v1"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--references-per-task", type=int, default=2)
    parser.add_argument("--video-seed", type=int, default=7)
    result = parser.parse_args()
    for name in (
        "repo", "config", "training_run", "checkpoint", "source_run",
        "source_checkpoint", "tokenizer_path", "data_root", "output_dir",
    ):
        setattr(result, name, getattr(result, name).resolve())
    if not 1 <= result.references_per_task <= 50 or result.video_seed < 0:
        raise WriterModelError("invalid UCP analysis reference schedule")
    return result


def _run_program(
    writer: CompleteLoRAWriter,
    initial: torch.Tensor,
    endpoints: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
    target_permutation: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks, value = [], initial
    for block in writer.semantic_program.blocks:
        value = block(value, endpoints, valid_intervals, valid_semantics)
        blocks.append(value)
    if target_permutation is None:
        coordinates, _ = writer.compiler.compile_with_diagnostics(
            value, endpoints, valid_intervals, valid_semantics,
        )
    else:
        coordinates = compile_with_target_identity_permutation(
            writer, value, endpoints, valid_intervals, valid_semantics,
            target_permutation,
        )
    attention = reader_attention(
        writer, value, endpoints, valid_intervals, valid_semantics,
        target_permutation,
    )
    return blocks, value, coordinates, attention


def _condition_inputs(
    *, adapters: Mapping[str, Mapping[str, Any]], task: Mapping[str, Any],
    init_state_id: int, store: RawTeacherVideoStore, device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
    frame_batches, index_batches, rows = [], [], []
    for condition in CONDITIONS:
        evidence = expected_writer_episode_evidence(
            adapters[condition], suite=str(task["suite"]), task_id=int(task["task_id"]),
            init_state_id=init_state_id, lora_sha256="0" * 64,
        )
        teacher = store.load(int(evidence["video_global_task_id"]), int(evidence["teacher_demo_index"]))
        frames = torch.from_numpy(teacher.frames).to(device, non_blocking=True)
        indices = torch.from_numpy(teacher.frame_indices).to(device, non_blocking=True)
        order = torch.arange(frames.shape[0])
        if condition == "reversed":
            order = order.flip(0)
            frames = frames.flip(0)
        elif condition == "shuffled":
            order = writer_shuffled_frame_permutation(
                frames.shape[0], int(evidence["teacher_video_order_seed"]), keep_first=False,
            )
            frames = frames.index_select(0, order.to(device))
        frame_batches.append(frames)
        index_batches.append(indices)
        rows.append({
            "condition": condition,
            "init_state_id": init_state_id,
            "video_global_task_id": int(evidence["video_global_task_id"]),
            "teacher_demo_index": int(evidence["teacher_demo_index"]),
            "sampled_frames": int(frames.shape[0]),
            "raw_frames": int(teacher.raw_frame_count),
            "order": "shuffled" if condition == "shuffled" else "reversed" if condition == "reversed" else "forward",
            "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        })
    return frame_batches, index_batches, rows


def _variant_result(
    *, writer: CompleteLoRAWriter, policy: torch.nn.Module,
    processor: Pi05LiberoProcessor, prepared: Mapping[str, torch.Tensor],
    identity: Mapping[str, torch.Tensor], lora: Any, seed: int,
    initial: torch.Tensor, endpoints: torch.Tensor, valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor, device: torch.device,
    target_permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type in {"cpu", "cuda"},
    ):
        blocks, final, coordinates, attention = _run_program(
            writer, initial, endpoints, valid_intervals, valid_semantics,
            target_permutation,
        )
        factors, public = decode_coordinates(writer, coordinates)
    state = split_state(public, 0)
    validate_lora_state(state, lora)
    action = policy_action(
        policy=policy, processor=processor, prepared=prepared, state=state,
        identity=identity, lora=lora, seed=seed, device=device,
    )
    return {
        "program": program_signature(final[0], valid_intervals[0], valid_semantics[0]),
        "coordinates": coordinates[0].float(),
        "factor": split_state(factors, 0),
        "public": state,
        "action": action,
        "reader": reader_attention_summary(attention, valid_intervals, valid_semantics),
        "coordinate_summary": coordinate_summary(coordinates[0]),
        "geometry": lora_geometry(writer, state),
        "block_rms": [float(torch.sqrt(block.float().square().mean())) for block in blocks],
    }


def _variant_comparison(
    writer: CompleteLoRAWriter, reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "program": relative_metrics(reference["program"], candidate["program"]),
        "coordinates": relative_metrics(reference["coordinates"], candidate["coordinates"]),
        "factor": mapping_metrics(reference["factor"], candidate["factor"]),
        "public_a": mapping_metrics(reference["public"], candidate["public"], select="a"),
        "public_b": mapping_metrics(reference["public"], candidate["public"], select="b"),
        "effective_ba": effective_metrics(writer, reference["public"], candidate["public"]),
        "policy_action": relative_metrics(reference["action"], candidate["action"]),
    }


def _routing_diagnostics(
    *, writer: CompleteLoRAWriter, policy: torch.nn.Module,
    processor: Pi05LiberoProcessor, identity: Mapping[str, torch.Tensor],
    lora: Any, device: torch.device, encoded: Mapping[str, Any],
    shared: Mapping[str, Any], full: Mapping[str, Any],
) -> dict[str, Any]:
    selected = slice(0, 1)
    target_permutation = torch.roll(torch.arange(
        writer.compiler.target_count, device=device,
    ), -1)
    target_variant = _variant_result(
        **shared, initial=encoded["initial"][selected],
        endpoints=encoded["endpoints"][selected],
        valid_intervals=encoded["valid_intervals"][selected],
        valid_semantics=encoded["valid_semantics"][selected],
        target_permutation=target_permutation,
    )
    rank_permutation = torch.roll(torch.arange(
        writer.PUBLIC_LORA_RANK, device=device,
    ), -1)
    gauge_state, per_target = rank_gauge_permute(
        writer, full["public"], rank_permutation,
    )
    validate_lora_state(gauge_state, lora)
    gauge_action = policy_action(
        policy=policy, processor=processor, prepared=encoded["prepared"],
        state=gauge_state, identity=identity, lora=lora,
        seed=encoded["action_seed"], device=device,
    )
    ba_error = effective_ba_error(writer, full["public"], gauge_state)
    action_error = relative_metrics(full["action"], gauge_action)
    action_error["max_absolute_error"] = float(
        (full["action"].float() - gauge_action.float()).abs().max()
    )
    raw_a = mapping_metrics(full["public"], gauge_state, select="a")
    raw_b = mapping_metrics(full["public"], gauge_state, select="b")
    if (
        ba_error["relative_l2"] > 2e-5
        or action_error["relative_l2"] > 2e-5
        or raw_a["relative_l2"] == 0
        or raw_b["relative_l2"] == 0
    ):
        raise WriterModelError("rank gauge permutation violated its sanity contract")
    target_comparison = _variant_comparison(writer, full, target_variant)
    target_mapping = {}
    for spec in writer.tensor_specs:
        target = str(writer._decoding[spec.name][1])
        if target in target_mapping and target_mapping[target] != spec.module:
            raise WriterModelError("real target decode mapping is not unique")
        target_mapping[target] = spec.module
    if len(target_mapping) != writer.compiler.target_count:
        raise WriterModelError("real target decode mapping is incomplete")
    return {
        "target_identity_permutation": {
            "definition": "permute target identities; retain real 38-target decode slots",
            "permutation": target_permutation.cpu().tolist(),
            "real_target_decode_mapping": target_mapping,
            "relative_to_canonical": {key: target_comparison[key] for key in (
                "coordinates", "effective_ba", "policy_action",
            )},
        },
        "rank_gauge_permutation": {
            "definition": "same permutation of each public A row and B column",
            "permutation": rank_permutation.cpu().tolist(),
            "relative_l2_tolerance": 2e-5,
            "public_a": raw_a, "public_b": raw_b,
            "effective_ba_numerical_error": ba_error,
            "fixed_policy_action_numerical_error": action_error,
            "per_real_target_raw_change": per_target,
        },
    }


def _encode_reference(
    *, task: Mapping[str, Any], init_state_id: int,
    adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore,
    authority: WriterTaskAuthority, policy: torch.nn.Module,
    writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any,
    tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor,
    device: torch.device,
) -> dict[str, Any]:
    frame_batches, index_batches, input_rows = _condition_inputs(
        adapters=adapters, task=task, init_state_id=init_state_id,
        store=store, device=device,
    )
    offsets = [0]
    for frames in frame_batches:
        offsets.append(offsets[-1] + int(frames.shape[0]))
    frame_values = torch.cat(frame_batches)
    index_values = torch.cat(index_batches)
    condition_ids = torch.repeat_interleave(
        torch.arange(len(CONDITIONS), device=device),
        torch.tensor([len(value) for value in frame_batches], device=device),
    )
    tokens, masks, spans = tokenizer([str(task["language"])] * len(CONDITIONS))
    raw_action: list[torch.Tensor] = []
    handle = writer.semantic_encoder.interaction_projection.register_forward_pre_hook(
        lambda _module, values: raw_action.append(values[0].detach())
    )
    copy_task_lora_state_(policy, identity, lora)
    try:
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            q_text, x, g, action, valid_tokens = writer.semantic_encoder(
                policy, frame_values, condition_ids, tokens, masks, spans,
            )
            packed_x, packed_g, packed_action, positions, valid_frames = (
                writer._pack_video_program(
                    x, g, action, index_values, tuple(offsets),
                )
            )
            initial, endpoints, valid_intervals, valid_semantics = (
                build_initial_program(
                    packed_x, packed_g, packed_action, positions,
                    valid_frames, valid_tokens,
                )
            )
            blocks, final, coordinates, attention = _run_program(
                writer, initial, endpoints, valid_intervals, valid_semantics,
            )
            factors, public = decode_coordinates(writer, coordinates)
    finally:
        handle.remove()
    packed_raw, raw_valid = pack_flat(torch.cat(raw_action), offsets)
    if not torch.equal(raw_valid, valid_frames):
        raise WriterModelError("raw Action capture lost frame alignment")
    prepared, query_identity = fixed_policy_query(authority, processor, device)
    action_seed = int.from_bytes(
        hashlib.sha256(json.dumps(
            ["ucp_fixed_action", int(task["global_task_id"])],
            separators=(",", ":"),
        ).encode()).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)
    states = [split_state(public, row) for row in range(len(CONDITIONS))]
    factor_states = [split_state(factors, row) for row in range(len(CONDITIONS))]
    actions = []
    for state in states:
        validate_lora_state(state, lora)
        actions.append(policy_action(
            policy=policy, processor=processor, prepared=prepared, state=state,
            identity=identity, lora=lora, seed=action_seed, device=device,
        ))
    return {
        "input_rows": input_rows, "q_text": q_text,
        "packed_m": packed_x - packed_g, "packed_g": packed_g,
        "packed_x": packed_x, "packed_raw": packed_raw,
        "packed_action": packed_action, "valid_tokens": valid_tokens,
        "valid_frames": valid_frames, "initial": initial, "endpoints": endpoints,
        "valid_intervals": valid_intervals, "valid_semantics": valid_semantics,
        "blocks": blocks, "final": final, "coordinates": coordinates,
        "attention": attention, "factor_states": factor_states, "states": states,
        "actions": actions, "prepared": prepared, "query_identity": query_identity,
        "action_seed": action_seed,
    }


def _counterfactual_diagnostics(
    *, writer: CompleteLoRAWriter, policy: torch.nn.Module,
    processor: Pi05LiberoProcessor, identity: Mapping[str, torch.Tensor],
    lora: Any, device: torch.device, encoded: Mapping[str, Any],
) -> dict[str, Any]:
    selected = slice(0, 1)
    shared = {
        "writer": writer, "policy": policy, "processor": processor,
        "prepared": encoded["prepared"], "identity": identity, "lora": lora,
        "seed": encoded["action_seed"], "device": device,
    }
    full = _variant_result(
        **shared, initial=encoded["initial"][selected],
        endpoints=encoded["endpoints"][selected],
        valid_intervals=encoded["valid_intervals"][selected],
        valid_semantics=encoded["valid_semantics"][selected],
    )
    canonical = {**full, "public": encoded["states"][0],
                 "factor": encoded["factor_states"][0],
                 "coordinates": encoded["coordinates"][0],
                 "action": encoded["actions"][0]}
    canonical_recompute = _variant_comparison(writer, canonical, full)
    if any(
        canonical_recompute[key]["relative_l2"] > 2e-5
        for key in (
            "coordinates", "factor", "public_a", "public_b", "effective_ba",
            "policy_action",
        )
    ):
        raise WriterModelError("full counterfactual recompute changed canonical LoRA")
    routing = _routing_diagnostics(
        writer=writer, policy=policy, processor=processor, identity=identity,
        lora=lora, device=device, encoded=encoded, shared=shared, full=full,
    )
    ablations = {"full": {
        "reader": full["reader"], "coordinate_summary": full["coordinate_summary"],
        "geometry": full["geometry"],
    }}
    for name in ("x_only", "dynamic_only", "a_only", "d_only"):
        value = _variant_result(
            **shared, initial=type_ablation(encoded["initial"][selected], name),
            endpoints=encoded["endpoints"][selected],
            valid_intervals=encoded["valid_intervals"][selected],
            valid_semantics=encoded["valid_semantics"][selected],
        )
        ablations[name] = {
            "relative_to_full": _variant_comparison(writer, full, value),
            "reader": value["reader"],
            "coordinate_summary": value["coordinate_summary"],
            "geometry": value["geometry"],
        }
    initial = encoded["initial"]
    task_tokens = (initial.shape[2] - 1) // 2
    fixed_streams = {}
    for output_key, fixed in (
        ("fixed_x_vary_a_d", "x"), ("fixed_a_d_vary_x", "a_d"),
    ):
        comparisons = {}
        for row, condition in enumerate(CONDITIONS):
            candidate = fixed_stream_counterfactual(
                initial, encoded["valid_intervals"], row, fixed=fixed,
            )
            value = _variant_result(
                **shared, initial=candidate,
                endpoints=encoded["endpoints"][selected],
                valid_intervals=encoded["valid_intervals"][selected],
                valid_semantics=encoded["valid_semantics"][selected],
            )
            comparisons[condition] = _variant_comparison(writer, full, value)
        fixed_streams[output_key] = comparisons
    scales = {}
    for scale in (.5, 1.0, 2.0):
        candidate = initial[selected].clone()
        candidate[:, :, task_tokens:] *= scale
        value = full if scale == 1 else _variant_result(
            **shared, initial=candidate,
            endpoints=encoded["endpoints"][selected],
            valid_intervals=encoded["valid_intervals"][selected],
            valid_semantics=encoded["valid_semantics"][selected],
        )
        scales[str(scale)] = {
            "relative_to_scale1": _variant_comparison(writer, full, value),
            "reader": value["reader"], "geometry": value["geometry"],
        }
    return {
        "type_ablations": ablations,
        **fixed_streams,
        "dynamic_scale": scales,
        "canonical_recompute": canonical_recompute,
        **routing,
    }


def probe_reference(
    *, task: Mapping[str, Any], init_state_id: int,
    adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore,
    authority: WriterTaskAuthority, policy: torch.nn.Module,
    writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any,
    tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    encoded = _encode_reference(
        task=task, init_state_id=init_state_id, adapters=adapters, store=store,
        authority=authority, policy=policy, writer=writer, identity=identity,
        lora=lora, tokenizer=tokenizer, processor=processor, device=device,
    )
    matched = matched_diagnostics(writer, encoded)
    counterfactuals = _counterfactual_diagnostics(
        writer=writer, policy=policy, processor=processor, identity=identity,
        lora=lora, device=device, encoded=encoded,
    )
    row = {
        "global_task_id": int(task["global_task_id"]),
        "suite": str(task["suite"]), "task_id": int(task["task_id"]),
        "reference_ordinal": init_state_id,
        "conditions": encoded["input_rows"],
        "comparisons_to_correct": matched["comparisons"],
        "reader_attention": matched["readers"],
        "coordinate_routing": matched["coordinates"],
        "lora_geometry": matched["geometry"],
        **counterfactuals,
        "fixed_policy_query": encoded["query_identity"],
        "fixed_policy_action_seed": encoded["action_seed"],
        "information_wall": {
            "teacher_action_values_read": 0,
            "teacher_state_values_sent_to_writer": 0,
            "teacher_reward_or_terminal_values_read": 0,
            "policy_query_observation_state_sent_to_writer": 0,
        },
    }
    state = {
        name: value.detach().cpu() for name, value in encoded["states"][0].items()
    }
    return row, state, encoded["actions"][0].detach().cpu()


def _task_records(config: Mapping[str, Any], data_root: Path) -> tuple[dict[str, Any], ...]:
    target = read_json(Path(__file__).resolve().parents[1] / config["authorities"]["target_data_manifest"]["path"])
    wanted = set(int(value) for value in target["summary"]["roles"]["validation"])
    rows = []
    for row in target["tasks"]:
        if int(row["global_task_id"]) not in wanted:
            continue
        value = dict(row)
        value["path"] = str(data_root / str(row["hdf5"]["relative_path"]))
        rows.append(value)
    rows.sort(key=lambda row: (SUITE_ORDER.index(str(row["suite"])), int(row["task_id"])))
    if len(rows) != 8:
        raise WriterModelError("UCP analysis requires the eight validation tasks")
    return tuple(rows)


def _broadcast(context: Any, value: Any) -> Any:
    payload = [value if context.is_main else None]
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if isinstance(payload[0], Mapping) and "error" in payload[0]:
        raise WriterModelError(str(payload[0]["error"]))
    return payload[0]


def _inspect_authority_payload(
    args: argparse.Namespace,
    context: Any,
    authorities: Any,
    task_keys: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    payload: Any = None
    if context.is_main:
        try:
            if args.output_dir.exists():
                raise WriterModelError("analysis output root already exists")
            state = git_state(args.repo)
            source = inspect_source_checkpoint(
                authorities, args.source_run, args.source_checkpoint, evaluation_mode="formal",
            )
            tokenizer_record = inspect_tokenizer(authorities, args.tokenizer_path)
            adapters = {
                condition: inspect_as_writer_evaluation(
                    config_path=args.config, checkpoint=args.checkpoint,
                    video_data_root=args.data_root, source=source,
                    task_keys=task_keys, video_condition=condition,
                    video_seed=args.video_seed, require_formal=True,
                    video_sampling_mode="without_replacement",
                )
                for condition in CONDITIONS
            }
            training = read_json(args.training_run / "run_contract.json")
            provenance = validate_analysis_provenance(
                repo=args.repo, state=state, training=training,
            )
            payload = {
                "state": state, "source": source, "tokenizer": tokenizer_record,
                "adapters": adapters, "training": training, "provenance": provenance,
            }
        except Exception as error:
            payload = {"error": repr(error)}
    return _broadcast(context, payload)


def _publish_run_contract(
    args: argparse.Namespace,
    context: Any,
    payload: Mapping[str, Any],
) -> None:
    if context.is_main:
        args.output_dir.mkdir(parents=True)
        training = payload["training"]
        write_json_atomic(args.output_dir / "run_contract.json", {
            "schema_version": RUN_SCHEMA,
            "host": socket.gethostname(), "command": list(os.sys.argv),
            "git": payload["state"], "provenance": payload["provenance"],
            "analysis_code": {
                "files": {
                    relative: sha256_file(args.repo / relative)
                    for relative in (
                        "scripts/analyze_as_writer_ucp.py",
                        "src/ember/writer/ucp_analysis.py",
                        "src/ember/writer/ucp_analysis_runtime.py",
                        "src/ember/writer/ucp_analysis_summary.py",
                        "src/ember/writer/ucp_geometry.py",
                    )
                },
            },
            "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
            "training_run": {"path": str(args.training_run), "contract_sha256": canonical_hash(training)},
            "checkpoint": {"path": str(args.checkpoint), "manifest_sha256": sha256_file(args.checkpoint / "checkpoint_manifest.json")},
            "source": payload["source"], "tokenizer": payload["tokenizer"],
            "conditions": list(CONDITIONS), "references_per_task": args.references_per_task,
            "video_seed": args.video_seed, "video_sampling_mode": "without_replacement",
            "world_size": context.world_size, "physical_gpu_ids": [4, 5, 6, 7],
            "rollouts": 0, "teacher_action_values_read": 0,
            "teacher_state_values_sent_to_writer": 0,
            "fixed_policy_query": "validation HDF5 observation-only demo0/frame0 after Writer LoRA generation",
            "counterfactual_contract": COUNTERFACTUAL_CONTRACT,
        })
    barrier(context)


def _analyze_local_tasks(
    *, args: argparse.Namespace, context: Any, tasks: Sequence[Mapping[str, Any]],
    adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore,
    task_authorities: Sequence[WriterTaskAuthority], policy: torch.nn.Module,
    writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any,
    tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor,
) -> list[dict[str, Any]]:
    authority_by_id = {value.task_id: value for value in task_authorities}
    local_results = []
    for ordinal, task in enumerate(tasks):
        if ordinal % context.world_size != context.rank:
            continue
        task_rows, correct_states, correct_actions = [], [], []
        for reference in range(args.references_per_task):
            row, state, action = probe_reference(
                task=task, init_state_id=reference, adapters=adapters, store=store,
                authority=authority_by_id[int(task["global_task_id"])],
                policy=policy, writer=writer, identity=identity, lora=lora,
                tokenizer=tokenizer, processor=processor, device=context.device,
            )
            task_rows.append(row)
            correct_states.append(state)
            correct_actions.append(action)
        variance = {
            "effective_ba": effective_variance(writer, correct_states),
            "fixed_policy_action": variance_metrics(correct_actions),
        }
        for row in task_rows:
            row["same_task_video_variance"] = variance
        local_results.extend(task_rows)
    return local_results


def _seal_local_rows(
    args: argparse.Namespace,
    context: Any,
    local_results: Sequence[Mapping[str, Any]],
    failure: str | None,
) -> None:
    if failure is None:
        try:
            validate_finite_tree(local_results, f"rank_{context.rank}")
        except Exception as error:
            failure = repr(error)
    statuses: list[Any] = [None] * context.world_size
    dist.all_gather_object(statuses, {"rank": context.rank, "error": failure})
    errors = [status for status in statuses if status["error"]]
    if errors:
        raise WriterModelError(f"UCP analysis rank failure: {errors}")
    write_failure = None
    try:
        write_json_atomic(args.output_dir / f"rows_rank_{context.rank:02d}.json", {
            "rank": context.rank, "rows": list(local_results),
        })
    except Exception as error:
        write_failure = repr(error)
    dist.all_gather_object(statuses, {"rank": context.rank, "error": write_failure})
    errors = [status for status in statuses if status["error"]]
    if errors:
        raise WriterModelError(f"UCP analysis rank write failure: {errors}")


def _finalize_results(
    args: argparse.Namespace,
    context: Any,
    *,
    started: float,
) -> None:
    status: Any = None
    if context.is_main:
        try:
            payloads = []
            for rank in range(context.world_size):
                payloads.append(read_json(
                    args.output_dir / f"rows_rank_{rank:02d}.json"
                ))
            rows = validate_rank_payloads(payloads, args.references_per_task)
            result = {
                "schema_version": RESULT_SCHEMA, "rows": rows,
                "summary": summarize_records(rows), "task_count": 8,
                "references_per_task": args.references_per_task,
                "conditions": list(CONDITIONS), "rollouts": 0,
                "wall_seconds": time.monotonic() - started,
                "run_contract_sha256": canonical_hash(
                    read_json(args.output_dir / "run_contract.json")
                ),
            }
            validate_finite_tree(result)
            write_json_atomic(args.output_dir / "analysis.json", result)
            write_json_atomic(args.output_dir / "summary.json", result["summary"])
            status = {"ok": True}
        except Exception as error:
            status = {"error": repr(error)}
    _broadcast(context, status)


def main() -> None:
    args = parse_args()
    if args.repo != Path(__file__).resolve().parents[1]:
        raise WriterModelError("analysis --repo differs from the executing checkout")
    context = initialize_distributed()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if context.world_size != 4 or visible != ["4", "5", "6", "7"]:
        raise WriterModelError("canonical UCP analysis requires four ranks on physical GPUs4-7")
    if args.training_run != args.checkpoint.parent.parent:
        raise WriterModelError("analysis checkpoint crossed its declared training run")
    config = load_writer_config(args.config)
    if config.get("writer", {}).get("architecture") != EXPECTED_ARCHITECTURE:
        raise WriterModelError("analysis config is not the canonical UCP Writer")
    authorities = load_evaluation_authorities(
        args.repo / "configs/pi05_target_evaluation_v1.json", args.repo,
    )
    task_keys = resolve_role_task_keys(authorities.protocol, "validation")
    payload = _inspect_authority_payload(args, context, authorities, task_keys)
    _publish_run_contract(args, context, payload)
    source, training = payload["source"], payload["training"]
    tasks = _task_records(config, args.data_root)
    task_authorities = tuple(
        WriterTaskAuthority(
            task_id=int(row["global_task_id"]), language=str(row["language"]),
            path=Path(row["path"]), expected_bytes=int(row["hdf5"]["bytes"]),
        )
        for row in tasks
    )
    policy, writer, lora, identity = _build_models(
        training=training, source=source, context=context,
    )
    writer.load_state_dict(load_file(str(args.checkpoint / "writer.safetensors"), device=str(context.device)), strict=True)
    writer.eval()
    source_config = authorities.source_base_config
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        args.tokenizer_path, int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path, int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    store = RawTeacherVideoStore(
        task_authorities, frame_stride=int(config["writer"]["frame_stride"]), max_open_files=2,
    )
    started = time.monotonic()
    local_results: list[dict[str, Any]] = []
    failure = None
    try:
        local_results = _analyze_local_tasks(
            args=args, context=context, tasks=tasks, adapters=payload["adapters"],
            store=store, task_authorities=task_authorities, policy=policy,
            writer=writer, identity=identity, lora=lora, tokenizer=tokenizer,
            processor=processor,
        )
    except Exception as error:
        failure = repr(error)
    finally:
        store.close()
    try:
        _seal_local_rows(args, context, local_results, failure)
        _finalize_results(args, context, started=started)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
