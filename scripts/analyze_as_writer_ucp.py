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
from ember.writer.ucp_analysis_summary import matched_diagnostics
from ember.writer.ucp_analysis_runtime import (
    canonical_program_parity,
    fixed_policy_query,
    policy_action,
)
from ember.writer.ucp_analysis_run import (
    broadcast_value,
    control_barrier,
    control_group_contract,
    create_control_group,
    destroy_process_groups,
    finalize_results,
    record_local_failure,
    seal_local_rows,
)
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
    selected_row: int = 0,
) -> dict[str, Any]:
    if not 0 <= selected_row < initial.shape[0]:
        raise WriterModelError("invalid UCP variant result row")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type in {"cpu", "cuda"},
    ):
        blocks, final, coordinates, attention = _run_program(
            writer, initial, endpoints, valid_intervals, valid_semantics,
            target_permutation,
        )
        factors, public = decode_coordinates(writer, coordinates)
    state = split_state(public, selected_row)
    validate_lora_state(state, lora)
    action = policy_action(
        policy=policy, processor=processor, prepared=prepared, state=state,
        identity=identity, lora=lora, seed=seed, device=device,
    )
    row = slice(selected_row, selected_row + 1)
    valid_grid = (
        valid_intervals[selected_row, :, None]
        & valid_semantics[selected_row, None, :]
    )
    return {
        "program": program_signature(
            final[selected_row], valid_intervals[selected_row],
            valid_semantics[selected_row],
        ),
        "coordinates": coordinates[selected_row].float(),
        "factor": split_state(factors, selected_row),
        "public": state,
        "action": action,
        "reader": reader_attention_summary(
            attention[row], valid_intervals[row], valid_semantics[row],
        ),
        "coordinate_summary": coordinate_summary(coordinates[selected_row]),
        "geometry": lora_geometry(writer, state),
        "block_rms": [
            float(torch.sqrt(block[selected_row][valid_grid].float().square().mean()))
            for block in blocks
        ],
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
    target_permutation = torch.roll(torch.arange(
        writer.compiler.target_count, device=device), -1)
    target_variant = _variant_result(
        **shared, initial=encoded["initial"],
        endpoints=encoded["endpoints"],
        valid_intervals=encoded["valid_intervals"],
        valid_semantics=encoded["valid_semantics"],
        target_permutation=target_permutation,
    )
    rank_permutation = torch.roll(torch.arange(
        writer.PUBLIC_LORA_RANK, device=device), -1)
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
    action_error["max_absolute_error"] = float((
        full["action"].float() - gauge_action.float()
    ).abs().max())
    raw_a = mapping_metrics(full["public"], gauge_state, select="a")
    raw_b = mapping_metrics(full["public"], gauge_state, select="b")
    if (
        ba_error["relative_l2"] > 2e-5 or action_error["relative_l2"] > 2e-5
        or raw_a["relative_l2"] == 0 or raw_b["relative_l2"] == 0
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
            "relative_to_canonical": {
                key: target_comparison[key]
                for key in ("coordinates", "effective_ba", "policy_action")
            },
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
            canonical_parity = canonical_program_parity(
                writer, packed_x, packed_g, packed_action, positions,
                valid_frames, valid_tokens,
                (final, endpoints, valid_intervals, valid_semantics, coordinates),
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
        "canonical_program_parity": canonical_parity, "actions": actions,
        "prepared": prepared, "query_identity": query_identity,
        "action_seed": action_seed,
    }


def _counterfactual_diagnostics(
    *, writer: CompleteLoRAWriter, policy: torch.nn.Module,
    processor: Pi05LiberoProcessor, identity: Mapping[str, torch.Tensor],
    lora: Any, device: torch.device, encoded: Mapping[str, Any],
) -> dict[str, Any]:
    shared = {
        "writer": writer, "policy": policy, "processor": processor,
        "prepared": encoded["prepared"], "identity": identity, "lora": lora,
        "seed": encoded["action_seed"], "device": device,
    }
    full = _variant_result(
        **shared, initial=encoded["initial"],
        endpoints=encoded["endpoints"],
        valid_intervals=encoded["valid_intervals"],
        valid_semantics=encoded["valid_semantics"],
    )
    canonical = {**full, "public": encoded["states"][0],
                 "factor": encoded["factor_states"][0],
                 "coordinates": encoded["coordinates"][0],
                 "action": encoded["actions"][0]}
    variant_recompute = _variant_comparison(writer, canonical, full)
    recompute_keys = (
        "coordinates", "factor", "public_a", "public_b", "effective_ba",
        "policy_action",
    )
    offenders = {
        key: variant_recompute[key]
        for key in recompute_keys
        if variant_recompute[key]["relative_l2"] > 2e-5
    }
    if offenders:
        raise WriterModelError(
            "full counterfactual recompute changed canonical LoRA: "
            f"{offenders}"
        )
    routing = _routing_diagnostics(
        writer=writer, policy=policy, processor=processor, identity=identity,
        lora=lora, device=device, encoded=encoded, shared=shared, full=full,
    )
    ablations = {"full": {
        "reader": full["reader"], "coordinate_summary": full["coordinate_summary"],
        "geometry": full["geometry"],
    }}
    initial = encoded["initial"]

    def carrier_with_reference(candidate: torch.Tensor) -> torch.Tensor:
        if candidate.shape != initial[0:1].shape:
            raise WriterModelError("UCP counterfactual reference shape changed")
        carrier = initial.clone()
        carrier[0:1] = candidate
        return carrier

    for name in ("x_only", "dynamic_only", "a_only", "d_only"):
        candidate = carrier_with_reference(type_ablation(initial[0:1], name))
        value = _variant_result(
            **shared, initial=candidate,
            endpoints=encoded["endpoints"],
            valid_intervals=encoded["valid_intervals"],
            valid_semantics=encoded["valid_semantics"],
        )
        ablations[name] = {
            "relative_to_full": _variant_comparison(writer, full, value),
            "reader": value["reader"],
            "coordinate_summary": value["coordinate_summary"],
            "geometry": value["geometry"],
        }
    task_tokens = (initial.shape[2] - 1) // 2
    fixed_streams = {}
    for output_key, fixed in (
        ("fixed_x_vary_a_d", "x"), ("fixed_a_d_vary_x", "a_d"),
    ):
        comparisons = {}
        for row, condition in enumerate(CONDITIONS):
            candidate = carrier_with_reference(fixed_stream_counterfactual(
                initial, encoded["valid_intervals"], row, fixed=fixed,
            ))
            value = _variant_result(
                **shared, initial=candidate,
                endpoints=encoded["endpoints"],
                valid_intervals=encoded["valid_intervals"],
                valid_semantics=encoded["valid_semantics"],
            )
            comparisons[condition] = _variant_comparison(writer, full, value)
        fixed_streams[output_key] = comparisons
    scales = {}
    for scale in (.5, 1.0, 2.0):
        candidate = initial[0:1].clone()
        candidate[:, :, task_tokens:] *= scale
        candidate = carrier_with_reference(candidate)
        value = full if scale == 1 else _variant_result(
            **shared, initial=candidate,
            endpoints=encoded["endpoints"],
            valid_intervals=encoded["valid_intervals"],
            valid_semantics=encoded["valid_semantics"],
        )
        scales[str(scale)] = {
            "relative_to_scale1": _variant_comparison(writer, full, value),
            "reader": value["reader"], "geometry": value["geometry"],
        }
    return {
        "type_ablations": ablations,
        **fixed_streams,
        "dynamic_scale": scales,
        "variant_recompute": variant_recompute,
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
        "reader_attention": matched["readers"], "coordinate_routing": matched["coordinates"],
        "lora_geometry": matched["geometry"],
        "canonical_program_parity": encoded["canonical_program_parity"], **counterfactuals,
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


def _inspect_authority_payload(
    args: argparse.Namespace,
    context: Any,
    authorities: Any,
    task_keys: Sequence[tuple[str, int]],
    control_group: Any,
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
    return broadcast_value(context, payload, control_group=control_group)


def _publish_run_contract(
    args: argparse.Namespace,
    context: Any,
    payload: Mapping[str, Any],
    control_group: Any,
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
                        "src/ember/writer/ucp_analysis_run.py",
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
            "distributed_control": control_group_contract(),
            "rollouts": 0, "teacher_action_values_read": 0,
            "teacher_state_values_sent_to_writer": 0,
            "fixed_policy_query": "validation HDF5 observation-only demo0/frame0 after Writer LoRA generation",
            "counterfactual_contract": COUNTERFACTUAL_CONTRACT,
        })
    control_barrier(context, control_group)


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
            try:
                row, state, action = probe_reference(
                    task=task, init_state_id=reference, adapters=adapters, store=store,
                    authority=authority_by_id[int(task["global_task_id"])],
                    policy=policy, writer=writer, identity=identity, lora=lora,
                    tokenizer=tokenizer, processor=processor, device=context.device,
                )
            except Exception as error:
                raise WriterModelError(
                    "UCP analysis reference failed: "
                    f"rank={context.rank}, suite={task['suite']}, "
                    f"task_id={task['task_id']}, "
                    f"global_task_id={task['global_task_id']}, "
                    f"reference_ordinal={reference}"
                ) from error
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


def main() -> None:
    args = parse_args()
    if args.repo != Path(__file__).resolve().parents[1]:
        raise WriterModelError("analysis --repo differs from the executing checkout")
    context = initialize_distributed()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if context.world_size != 4 or visible != ["4", "5", "6", "7"]:
        raise WriterModelError("canonical UCP analysis requires four ranks on physical GPUs4-7")
    control_group = create_control_group(context)
    if args.training_run != args.checkpoint.parent.parent:
        raise WriterModelError("analysis checkpoint crossed its declared training run")
    config = load_writer_config(args.config)
    if config.get("writer", {}).get("architecture") != EXPECTED_ARCHITECTURE:
        raise WriterModelError("analysis config is not the canonical UCP Writer")
    authorities = load_evaluation_authorities(
        args.repo / "configs/pi05_target_evaluation_v1.json", args.repo,
    )
    task_keys = resolve_role_task_keys(authorities.protocol, "validation")
    payload = _inspect_authority_payload(
        args, context, authorities, task_keys, control_group,
    )
    _publish_run_contract(args, context, payload, control_group)
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
    try:
        local_results = _analyze_local_tasks(
            args=args, context=context, tasks=tasks, adapters=payload["adapters"],
            store=store, task_authorities=task_authorities, policy=policy,
            writer=writer, identity=identity, lora=lora, tokenizer=tokenizer,
            processor=processor,
        )
        seal_local_rows(
            args.output_dir, context, local_results, control_group,
        )
        finalize_results(
            output_dir=args.output_dir,
            context=context,
            references_per_task=args.references_per_task,
            conditions=CONDITIONS,
            result_schema=RESULT_SCHEMA,
            started=started,
            control_group=control_group,
        )
    except Exception as error:
        record_local_failure(args.output_dir, context.rank, error)
        raise
    finally:
        store.close()
    destroy_process_groups(control_group)


if __name__ == "__main__":
    main()
