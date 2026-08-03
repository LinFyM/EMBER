"""Canonical no-rollout analysis for the Semantic Direction-Store AS-Writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from safetensors.torch import load_file

from ember.lora import (
    copy_task_lora_state_, lora_state_sha256, task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_eval_contract import (
    git_state, inspect_source_checkpoint, inspect_tokenizer,
    load_evaluation_authorities, resolve_role_task_keys,
)
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, load_stats
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.as_config import load_writer_config
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority, _camera
from ember.writer.inference import (
    expected_writer_episode_evidence, inspect_as_writer_evaluation,
    writer_shuffled_frame_permutation,
)
from ember.writer.internal_metrics import (
    CONDITIONS, change_retention, effective_ba_error,
    effective_metrics, effective_variance, fixed_sequence, lora_geometry,
    mapping_metrics, probability_summary, rank_gauge_permute, relative_metrics,
    validate_finite_tree, variance_metrics,
)
from ember.writer.internal_path import (
    PARITY_TOLERANCE,
    REPLAY_TOLERANCE,
    _compile,
    capture_writer,
    counterfactual_states,
)
from ember.writer.internal_decode import state_row as _state
from ember.writer.internal_results import (
    CONTROL_TIMEOUT, barrier, broadcast, create_control_group, finalize,
    lpt_assignment, record_failure, seal_rank_rows,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.validation import _build_models


RUN_SCHEMA = "ember_semantic_direction_store_internal_analysis_run_v1"
RESULT_SCHEMA = "ember_semantic_direction_store_internal_analysis_v1"
ARCHITECTURE = "pi05_semantic_direction_store_program_v1"
PROTECTED = (
    "src/ember/writer/model.py", "src/ember/writer/video_program.py",
    "src/ember/writer/semantic_core.py", "src/ember/writer/semantic_program.py",
    "src/ember/writer/program_compiler.py", "src/ember/writer/architecture.py",
    "src/ember/writer/internal_compiler.py",
    "src/ember/writer/functional.py", "src/ember/lora.py", "src/ember/pi05_lora.py",
    "configs/pi05_as_writer_semantic_direction_store_full24_decay400_bci_v1.json",
)


def _comparison(writer: CompleteLoRAWriter, reference: Mapping[str, Any], candidate: Mapping[str, Any], reference_action: torch.Tensor, candidate_action: torch.Tensor) -> dict[str, Any]:
    return {
        "coordinates": relative_metrics(reference["coordinates"], candidate["coordinates"]),
        "direction_store_weights": relative_metrics(
            reference["store_weights"], candidate["store_weights"]
        ),
        "direction_store_ids_equal": bool(
            torch.equal(reference["store_indices"], candidate["store_indices"])
        ),
        "factor_heads": mapping_metrics(reference["heads"], candidate["heads"]),
        "factor": mapping_metrics(reference["factor"], candidate["factor"]),
        "public_a": mapping_metrics(reference["public"], candidate["public"], select="a"),
        "public_b": mapping_metrics(reference["public"], candidate["public"], select="b"),
        "effective_ba": effective_metrics(writer, reference["public"], candidate["public"]),
        "fixed_policy_action": relative_metrics(reference_action, candidate_action),
    }


def _signature(captured: Mapping[str, Any], row: int) -> dict[str, torch.Tensor]:
    vf = captured["valid_frames"][row]; vt = captured["valid_tokens"][row]
    vi = captured["program"]["valid_intervals"][row]
    grid = vf[:, None] & vt[None]
    program_value = captured["program"]["memory"][row]
    program_grid = vi[None, :, None].expand(program_value.shape[:-1])
    role_read = captured["compiled"]["diagnostic"]["role_read"][row]
    role_grid = torch.ones(
        role_read.shape[:-1], dtype=torch.bool, device=role_read.device
    )
    result = {
        "q_text": fixed_sequence(captured["q"][row], vt),
        "multimodal_m": fixed_sequence(captured["m"][row], grid),
        "grounded_g": fixed_sequence(captured["g"][row], grid),
        "absolute_x": fixed_sequence(captured["x"][row], grid),
        "raw_action": fixed_sequence(captured["a_raw"][row], vf),
        "action_probe": fixed_sequence(captured["a"][row], vf),
        "core_frame_mean": fixed_sequence(captured["core"]["mean"][row], vt),
        "core_mean_carrier": fixed_sequence(captured["core"]["mean_carrier"][row], vt),
        "core_centered_residual": fixed_sequence(captured["core"]["centered_residual"][row], vt),
        "core_pre": fixed_sequence(captured["core"]["pre"][row], vt),
        "core_final": fixed_sequence(captured["core"]["final"][row], vt),
        "program_raw": fixed_sequence(captured["program"]["raw"][row], program_grid),
        "program_memory": fixed_sequence(
            captured["program"]["memory"][row], program_grid
        ),
        "target_query": captured["compiled"]["diagnostic"][
            "target_query"
        ][row].float(),
        "core_read": captured["compiled"]["diagnostic"]["core_read"][row].float(),
        "role_read": fixed_sequence(role_read, role_grid),
        "coordinates": captured["compiled"]["coordinates"][row].float(),
        "task_anchor": captured["task_anchor"][row].float(),
        "direction_store_weights": captured["decoded"]["store_weights"][row].float(),
    }
    for index, value in enumerate(captured["core"]["blocks"], 1): result[f"core_block_{index}"] = fixed_sequence(value[row], vt)
    for index, value in enumerate(captured["program"]["blocks"], 1): result[f"program_block_{index}"] = fixed_sequence(value[row], program_grid)
    return result


def _direction_store_summary(
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, Any]:
    """Report the fixed language route without treating it as generated value."""

    return {
        "store_indices": indices.to(dtype=torch.long, device="cpu").tolist(),
        "store_weights": weights.float().cpu().tolist(),
        "weight_sum": float(weights.float().sum()),
    }


@torch.inference_mode()
def _paired_diagnostics(writer: CompleteLoRAWriter, captured: Mapping[str, Any], actions: Sequence[torch.Tensor]) -> dict[str, Any]:
    reference = _signature(captured, 0); reference_factor = _state(captured["decoded"]["factors"], 0)
    reference_heads = _state(captured["decoded"]["heads"], 0); reference_public = _state(captured["decoded"]["public"], 0); comparisons, attention = {}, {}
    for row, condition in enumerate(CONDITIONS):
        signature = _signature(captured, row)
        stages = {name: relative_metrics(reference[name], signature[name]) for name in reference}
        heads = _state(captured["decoded"]["heads"], row); factor = _state(captured["decoded"]["factors"], row); public = _state(captured["decoded"]["public"], row)
        effective = effective_metrics(writer, reference_public, public)
        action = relative_metrics(actions[0], actions[row])
        chain = [stages["program_raw"]["relative_l2"], stages["program_memory"]["relative_l2"], stages["role_read"]["relative_l2"], effective["relative_l2"], action["relative_l2"]]
        comparisons[condition] = {
            "stages": stages, "factor_heads": mapping_metrics(reference_heads, heads), "factor": mapping_metrics(reference_factor, factor),
            "public_a": mapping_metrics(reference_public, public, select="a"),
            "public_b": mapping_metrics(reference_public, public, select="b"),
            "effective_ba": effective, "fixed_policy_action": action,
            "change_retention": {
                "raw_to_memory": change_retention(chain[0], chain[1]),
                "memory_to_role_read": change_retention(chain[1], chain[2]),
                "role_read_to_ba": change_retention(chain[2], chain[3]),
                "ba_to_action": change_retention(chain[3], chain[4]),
            },
        }
        vf = captured["valid_frames"][row : row + 1]; vt = captured["valid_tokens"][row : row + 1]
        frame = captured["core"]["frame_attention"][row : row + 1].permute(0, 1, 3, 2)
        program = {name: value[row : row + 1] for name, value in captured["program"].items() if isinstance(value, torch.Tensor)}
        device = captured["core"]["final"].device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            reader = _compile(
                writer,
                captured["core"]["final"][row : row + 1],
                vt,
                program,
            )
        attention[condition] = {"core_frame": probability_summary(frame, vt), "compiler": reader["attention"]}
    return {"comparisons": comparisons, "attention": attention, "program_blocks_five_condition_batch": captured["program"]["attention"]}


def fixed_policy_query(authority: WriterTaskAuthority, processor: Pi05LiberoProcessor, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read only demo0/frame0 observations; never open an actions dataset."""
    with h5py.File(authority.path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        base_array = np.asarray(obs["agentview_rgb"][0]); wrist_array = np.asarray(obs["eye_in_hand_rgb"][0])
        state_array = np.concatenate((np.asarray(obs["ee_states"][0], dtype=np.float32), np.asarray(obs["gripper_states"][0], dtype=np.float32)))
    base = torch.from_numpy(_camera(base_array))[None].to(device, dtype=torch.float32).div_(255)
    wrist = torch.from_numpy(_camera(wrist_array))[None].to(device, dtype=torch.float32).div_(255)
    states = torch.from_numpy(state_array)[None].to(device)
    tokens, masks = processor._tokenize_prompts(states, [authority.language])
    identity = {"demo_index": 0, "frame_index": 0, "observation_only": True, "actions_dataset_opened": False}
    for name, value in (("base", base_array), ("wrist", wrist_array), ("state", state_array)):
        identity[f"{name}_sha256"] = hashlib.sha256(np.ascontiguousarray(value)).hexdigest()
    return {"observation.images.base_0_rgb": base, "observation.images.left_wrist_0_rgb": wrist, OBS_LANGUAGE_TOKENS: tokens, OBS_LANGUAGE_ATTENTION_MASK: masks}, identity


def _policy_attention_backends(policy: torch.nn.Module) -> dict[str, str]:
    try:
        bridge = policy.model.paligemma_with_expert
        language = bridge.paligemma.model.language_model.config
        expert = bridge.gemma_expert.model.config
        result = {
            "language": str(language._attn_implementation),
            "expert": str(expert._attn_implementation),
        }
    except (AttributeError, TypeError) as error:
        raise WriterModelError("PI05 attention backend authority changed") from error
    if any(value not in {"eager", "sdpa"} for value in result.values()):
        raise WriterModelError(f"unsupported PI05 attention backend: {result}")
    return result


@contextmanager
def _preserve_policy_attention_backends(
    policy: torch.nn.Module,
) -> Any:
    """Contain PI05 sampler's eager-attention config mutation to one probe."""

    bridge = policy.model.paligemma_with_expert
    language = bridge.paligemma.model.language_model.config
    expert = bridge.gemma_expert.model.config
    before = _policy_attention_backends(policy)
    try:
        yield before
    finally:
        language._attn_implementation = before["language"]
        expert._attn_implementation = before["expert"]
        if _policy_attention_backends(policy) != before:
            raise WriterModelError("PI05 attention backend restore failed")


def policy_action(policy: torch.nn.Module, processor: Pi05LiberoProcessor, prepared: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], identity: Mapping[str, torch.Tensor], lora: Any, seed: int, device: torch.device) -> torch.Tensor:
    identity_sha = lora_state_sha256(identity)
    if lora_state_sha256(task_lora_state_dict(policy)) != identity_sha: raise WriterModelError("source policy LoRA was not identity before fixed query")
    copy_task_lora_state_(policy, state, lora)
    noise = torch.randn(1, int(policy.model.config.chunk_size), int(policy.model.config.max_action_dim), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32).to(device)
    try:
        with _preserve_policy_attention_backends(policy), torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = policy.predict_action_chunk(dict(prepared), noise=noise, num_steps=10)
    finally:
        copy_task_lora_state_(policy, identity, lora)
    if lora_state_sha256(task_lora_state_dict(policy)) != identity_sha: raise WriterModelError("source policy LoRA identity restore failed")
    return processor.unnormalize_action(value).detach()


def _condition_capture(task: Mapping[str, Any], reference: int, adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore, tokenizer: Pi05TeacherPrefixTokenizer, policy: torch.nn.Module, writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames, indices, metadata = [], [], []
    for condition in CONDITIONS:
        evidence = expected_writer_episode_evidence(adapters[condition], suite=str(task["suite"]), task_id=int(task["task_id"]), init_state_id=reference, lora_sha256="0" * 64)
        video = store.load(int(evidence["video_global_task_id"]), int(evidence["teacher_demo_index"]))
        value = torch.from_numpy(video.frames).to(device, non_blocking=True); order = torch.arange(value.shape[0])
        if condition == "reversed": order = order.flip(0); value = value.flip(0)
        elif condition == "shuffled":
            order = writer_shuffled_frame_permutation(value.shape[0], int(evidence["teacher_video_order_seed"]), keep_first=False); value = value.index_select(0, order.to(device))
        frames.append(value); indices.append(torch.from_numpy(video.frame_indices).to(device, non_blocking=True))
        metadata.append({"condition": condition, "video_global_task_id": int(evidence["video_global_task_id"]), "teacher_demo_index": int(evidence["teacher_demo_index"]), "sampled_frames": int(value.shape[0]), "raw_frames": int(video.raw_frame_count), "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
    offsets = [0]
    for value in frames: offsets.append(offsets[-1] + value.shape[0])
    language = [str(task["language"])] * len(CONDITIONS); tokens, masks, spans = tokenizer(language)
    copy_task_lora_state_(policy, identity, lora)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        captured = capture_writer(
            writer,
            policy,
            torch.cat(frames),
            torch.cat(indices),
            torch.tensor(offsets, dtype=torch.long, device=device),
            tokens,
            masks,
            spans,
        )
    return captured, metadata


def probe_reference(task: Mapping[str, Any], reference: int, adapters: Mapping[str, Mapping[str, Any]], store: RawTeacherVideoStore, authority: WriterTaskAuthority, tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor, policy: torch.nn.Module, writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any, device: torch.device, *, replay: bool) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    capture_backends = _policy_attention_backends(policy)
    captured, metadata = _condition_capture(task, reference, adapters, store, tokenizer, policy, writer, identity, lora, device)
    prepared, query_identity = fixed_policy_query(authority, processor, device)
    seed = int.from_bytes(hashlib.sha256(json.dumps(["as_writer_fixed_action", int(task["global_task_id"])], separators=(",", ":")).encode()).digest()[:8], "big") & ((1 << 63) - 1)
    states = [_state(captured["decoded"]["public"], row) for row in range(len(CONDITIONS))]
    for state in states: validate_lora_state(state, lora)
    actions = [policy_action(policy, processor, prepared, state, identity, lora, seed, device) for state in states]
    matched = _paired_diagnostics(writer, captured, actions)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        variants = counterfactual_states(writer, captured)
    variant_actions = {name: policy_action(policy, processor, prepared, value["public"], identity, lora, seed, device) for name, value in variants.items()}
    reference_variant = variants["full"]; reference_action = variant_actions["full"]
    counterfactuals = {name: {"relative_to_full": _comparison(writer, reference_variant, value, reference_action, variant_actions[name]), "geometry": lora_geometry(writer, value["public"]), "compiler_attention": value["attention"]} for name, value in variants.items()}
    memory_permuted = counterfactuals["program_memory/order_reversed"]
    program_memory_counterfactual = {
        **reference_variant["program_memory_authority"],
        "trained": {
            "program_reader_routing": reference_variant["attention"][
                "program_target_rank_routing"
            ],
            "effective_ba": lora_geometry(
                writer, reference_variant["public"]
            ),
            "fixed_action": relative_metrics(
                reference_action, reference_action
            ),
        },
        "order_reversed": {
            "program_reader_routing": variants[
                "program_memory/order_reversed"
            ]["attention"]["program_target_rank_routing"],
            "effective_ba_relative_to_trained": memory_permuted[
                "relative_to_full"
            ]["effective_ba"],
            "fixed_action_relative_to_trained": memory_permuted[
                "relative_to_full"
            ]["fixed_policy_action"],
        },
    }
    permutation = torch.roll(torch.arange(writer.PUBLIC_LORA_RANK, device=device), -1)
    gauge, raw_changes = rank_gauge_permute(writer, states[0], permutation); gauge_action = policy_action(policy, processor, prepared, gauge, identity, lora, seed, device)
    gauge_error = effective_ba_error(writer, states[0], gauge)
    if gauge_error["relative_l2"] > PARITY_TOLERANCE: raise WriterModelError("public rank gauge changed effective BA")
    replay_result = {"executed": False}
    if replay:
        repeated, repeated_metadata = _condition_capture(task, reference, adapters, store, tokenizer, policy, writer, identity, lora, device)
        repeated_state = _state(repeated["decoded"]["public"], 0); repeated_action = policy_action(policy, processor, prepared, repeated_state, identity, lora, seed, device)
        state_error = effective_ba_error(writer, states[0], repeated_state); action_error = relative_metrics(actions[0], repeated_action)
        original_signature = _signature(captured, 0)
        repeated_signature = _signature(repeated, 0)
        stage_errors = {
            name: relative_metrics(original_signature[name], repeated_signature[name])
            for name in original_signature
        }
        worst_stage = max(
            stage_errors,
            key=lambda name: stage_errors[name]["relative_l2"],
        )
        metadata_equal = repeated_metadata == metadata
        if (
            not metadata_equal
            or stage_errors[worst_stage]["relative_l2"] > REPLAY_TOLERANCE
            or state_error["relative_l2"] > REPLAY_TOLERANCE
            or action_error["relative_l2"] > REPLAY_TOLERANCE
        ):
            raise WriterModelError(
                "internal-analysis strict production replay failed: "
                f"metadata_equal={metadata_equal}, "
                f"worst_stage={worst_stage}:{stage_errors[worst_stage]}, "
                f"effective_ba={state_error}, fixed_policy_action={action_error}"
            )
        replay_result = {
            "executed": True,
            "classification": "strict_replay_with_policy_backend_restored",
            "tolerance": REPLAY_TOLERANCE,
            "metadata_exact": metadata_equal,
            "stage_errors": stage_errors,
            "worst_stage": worst_stage,
            "effective_ba": state_error,
            "fixed_policy_action": action_error,
        }
    row = {
        "global_task_id": int(task["global_task_id"]), "suite": str(task["suite"]), "task_id": int(task["task_id"]), "reference_ordinal": reference,
        "conditions": metadata, **matched, "canonical_parity": captured["parity"],
        "direction_store_routing": {
            condition: _direction_store_summary(
                captured["decoded"]["store_indices"][index],
                captured["decoded"]["store_weights"][index],
            )
            for index, condition in enumerate(CONDITIONS)
        },
        "lora_geometry": {condition: lora_geometry(writer, state) for condition, state in zip(CONDITIONS, states, strict=True)},
        "counterfactuals": counterfactuals, "program_memory_counterfactual": program_memory_counterfactual, "rank_gauge": {"permutation": permutation.cpu().tolist(), "raw_changes": raw_changes, "effective_ba_error": gauge_error, "fixed_policy_action_error": relative_metrics(actions[0], gauge_action)},
        "fixed_policy_query": query_identity, "fixed_policy_action_seed": seed, "deterministic_replay": replay_result,
        "attention_backend_lifecycle": {
            "writer_capture": capture_backends,
            "pi05_recursive_sampler_temporarily_forces_eager": True,
            "restored_after_each_fixed_policy_action": (
                _policy_attention_backends(policy) == capture_backends
            ),
        },
        "information_wall": {"teacher_action_values_read": 0, "teacher_state_values_sent_to_writer": 0, "teacher_reward_or_terminal_values_read": 0, "policy_query_observation_state_sent_to_writer": 0},
    }
    validate_finite_tree(row)
    return row, {name: value.detach().cpu() for name, value in states[0].items()}, actions[0].detach().cpu()


def _provenance(repo: Path, state: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    training_commit = str(training.get("git", {}).get("commit", "")); head = str(state.get("commit", ""))
    if len(training_commit) != 40 or len(head) != 40 or state.get("dirty_paths") or state.get("origin_main") != head:
        raise WriterModelError("analysis Git authority is not clean pushed origin/main")
    if subprocess.run(["git", "merge-base", "--is-ancestor", training_commit, head], cwd=repo).returncode:
        raise WriterModelError("analysis code is not descended from training code")
    changed = subprocess.run(["git", "diff", "--name-only", f"{training_commit}..{head}", "--", *PROTECTED], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    if changed: raise WriterModelError(f"trained target-bound-role topology changed after checkpoint: {changed}")
    return {"analysis_commit": head, "training_commit": training_commit, "training_is_ancestor": True, "protected_paths_unchanged": list(PROTECTED)}


def _task_records(repo: Path, config: Mapping[str, Any], data_root: Path, adapters: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = read_json(repo / str(config["authorities"]["target_data_manifest"]["path"])); validation = set(int(value) for value in target["summary"]["roles"]["validation"])
    needed = {int(item[key]) for adapter in adapters.values() for item in adapter["task_video_mapping"] for key in ("language_global_task_id", "video_global_task_id")}
    tasks, authorities = [], []
    for item in target["tasks"]:
        global_id = int(item["global_task_id"]); path = (data_root / str(item["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(data_root): raise WriterModelError("internal-analysis HDF5 escaped data root")
        if global_id in needed: authorities.append({"task_id": global_id, "language": str(item["language"]), "path": str(path), "expected_bytes": int(item["hdf5"]["bytes"])})
        if global_id in validation: tasks.append({**item, "path": str(path)})
    tasks.sort(key=lambda item: (SUITE_ORDER.index(str(item["suite"])), int(item["task_id"])))
    if len(tasks) != 8 or {item["task_id"] for item in authorities} != needed: raise WriterModelError("internal-analysis task authority changed")
    return tasks, authorities


def _task_costs(tasks: Sequence[Mapping[str, Any]], adapters: Mapping[str, Mapping[str, Any]], references: int) -> dict[int, int]:
    result = {}
    for task in tasks:
        cost = 0
        for reference in range(references):
            for condition in CONDITIONS:
                evidence = expected_writer_episode_evidence(adapters[condition], suite=str(task["suite"]), task_id=int(task["task_id"]), init_state_id=reference, lora_sha256="0" * 64)
                table = adapters[condition]["video_data"]["sampled_frame_counts_by_task"]
                task_table = table.get(str(evidence["video_global_task_id"]), table.get(int(evidence["video_global_task_id"])))
                cost += int(task_table.get(str(evidence["teacher_demo_index"]), task_table.get(int(evidence["teacher_demo_index"]))))
        result[int(task["global_task_id"])] = cost
    return result


def _inspect(args: argparse.Namespace, authorities: Any, task_keys: Sequence[tuple[str, int]]) -> dict[str, Any]:
    if args.output_dir.exists(): raise WriterModelError("internal-analysis output root already exists")
    config = load_writer_config(args.config); training = read_json(args.training_run / "run_contract.json")
    if args.training_run != args.checkpoint.parent.parent or config["writer"]["architecture"] != ARCHITECTURE or training["config_sha256"] != sha256_file(args.config):
        raise WriterModelError("internal-analysis checkpoint/config authority changed")
    state = git_state(args.repo); source = inspect_source_checkpoint(authorities, args.source_run, args.source_checkpoint, evaluation_mode="formal")
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    adapters = {condition: inspect_as_writer_evaluation(config_path=args.config, checkpoint=args.checkpoint, video_data_root=args.data_root, source=source, task_keys=task_keys, video_condition=condition, video_seed=args.video_seed, require_formal=True, video_sampling_mode="without_replacement") for condition in CONDITIONS}
    tasks, task_authorities = _task_records(args.repo, config, args.data_root, adapters)
    costs = _task_costs(tasks, adapters, args.references_per_task); assignment = lpt_assignment(costs)
    checkpoint_hashes = {name: sha256_file(args.checkpoint / name) for name in ("checkpoint_manifest.json", "writer.safetensors")}
    return {"git": state, "provenance": _provenance(args.repo, state, training), "config": config, "training": training, "source": source, "tokenizer": tokenizer, "adapters": adapters, "tasks": tasks, "task_authorities": task_authorities, "task_costs": costs, "assignment": assignment, "checkpoint_hashes": checkpoint_hashes}


def _publish(args: argparse.Namespace, payload: Mapping[str, Any], *, rank: int, world_size: int, group: Any | None) -> None:
    if rank == 0:
        args.output_dir.mkdir(parents=True)
        files = (
            "scripts/analyze_as_writer.py",
            "src/ember/writer/internal_analysis.py",
            "src/ember/writer/internal_path.py",
            "src/ember/writer/internal_compiler.py",
            "src/ember/writer/internal_metrics.py",
            "src/ember/writer/internal_results.py",
        )
        write_json_atomic(args.output_dir / "run_contract.json", {
            "schema_version": RUN_SCHEMA, "host": socket.gethostname(), "command": list(os.sys.argv), "git": payload["git"], "provenance": payload["provenance"],
            "analysis_code": {name: sha256_file(args.repo / name) for name in files}, "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
            "training_run": {"path": str(args.training_run), "contract_sha256": canonical_hash(payload["training"])}, "checkpoint": {"path": str(args.checkpoint), **payload["checkpoint_hashes"]},
            "source": payload["source"], "tokenizer": payload["tokenizer"], "conditions": list(CONDITIONS), "adapter_sha256": {name: canonical_hash(value) for name, value in payload["adapters"].items()}, "references_per_task": args.references_per_task,
            "video_seed": args.video_seed, "video_sampling": "without_replacement", "world_size": world_size, "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","), "task_costs": payload["task_costs"], "task_assignment": payload["assignment"], "task_assignment_sha256": canonical_hash({"costs": payload["task_costs"], "assignment": payload["assignment"]}),
            "distributed_control": {"backend": "gloo", "timeout_seconds": int(CONTROL_TIMEOUT.total_seconds())}, "rollouts": 0, "teacher_action_values_read": 0, "teacher_state_values_sent_to_writer": 0, "fixed_policy_query": "validation HDF5 observation-only demo0/frame0 after Writer LoRA generation",
        })
    barrier(world_size, group)


def _local_rows(args: argparse.Namespace, context: Any, payload: Mapping[str, Any], policy: torch.nn.Module, writer: CompleteLoRAWriter, identity: Mapping[str, torch.Tensor], lora: Any, store: RawTeacherVideoStore, tokenizer: Pi05TeacherPrefixTokenizer, processor: Pi05LiberoProcessor) -> list[dict[str, Any]]:
    tasks = {int(value["global_task_id"]): value for value in payload["tasks"]}; authorities = {value.task_id: value for value in store.authorities.values()}
    rows = []
    for task_id in payload["assignment"][context.rank]:
        task_rows, states, actions = [], [], []
        for reference in range(args.references_per_task):
            row, state, action = probe_reference(tasks[task_id], reference, payload["adapters"], store, authorities[task_id], tokenizer, processor, policy, writer, identity, lora, context.device, replay=reference == 0)
            task_rows.append(row); states.append(state); actions.append(action)
        variance = {"effective_ba": effective_variance(writer, states), "fixed_policy_action": variance_metrics(actions)}
        for row in task_rows: row["same_task_video_variance"] = variance
        rows.extend(task_rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "training-run", "checkpoint", "source-run", "source-checkpoint", "tokenizer-path", "data-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--references-per-task", type=int, default=1); parser.add_argument("--video-seed", type=int, default=7)
    result = parser.parse_args(argv)
    for name in ("repo", "config", "training_run", "checkpoint", "source_run", "source_checkpoint", "tokenizer_path", "data_root", "output_dir"): setattr(result, name, getattr(result, name).resolve())
    if not 1 <= result.references_per_task <= 50 or result.video_seed != 7: raise WriterModelError("invalid internal-analysis reference schedule")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv); repo = Path(__file__).resolve().parents[3]
    allowed_configs = {
        repo / "configs/pi05_as_writer_semantic_direction_store_full24_decay400_bci_v1.json",
    }
    if args.repo != repo or args.config not in allowed_configs:
        raise WriterModelError(
            "internal-analysis checkout/config is not canonical semantic direction-store"
        )
    context = initialize_distributed(); visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if context.world_size != 6 or len(visible) != 6 or len(set(visible)) != 6:
        raise WriterModelError("formal internal analysis requires six explicit BCI GPUs")
    group = create_control_group(context.world_size); authorities = load_evaluation_authorities(repo / "configs/pi05_target_evaluation_v1.json", repo); task_keys = resolve_role_task_keys(authorities.protocol, "validation")
    inspected: Any = None
    if context.is_main:
        try: inspected = _inspect(args, authorities, task_keys)
        except Exception as error: inspected = {"error": repr(error)}
    payload = broadcast(inspected, rank=context.rank, world_size=context.world_size, group=group); _publish(args, payload, rank=context.rank, world_size=context.world_size, group=group)
    policy, writer, lora, identity = _build_models(training=payload["training"], source=payload["source"], context=context)
    writer.load_state_dict(load_file(str(args.checkpoint / "writer.safetensors"), device=str(context.device)), strict=True); writer.eval()
    source_config = authorities.source_base_config; processor = Pi05LiberoProcessor(load_stats(source_config, source_config["data"]["active_task_ids"]), args.tokenizer_path, int(source_config["features"]["tokenizer_max_length"]), str(context.device)); tokenizer = Pi05TeacherPrefixTokenizer(args.tokenizer_path, int(source_config["features"]["tokenizer_max_length"]), str(context.device))
    task_authorities = tuple(WriterTaskAuthority(task_id=int(value["task_id"]), language=str(value["language"]), path=Path(value["path"]), expected_bytes=int(value["expected_bytes"])) for value in payload["task_authorities"]); store = RawTeacherVideoStore(task_authorities, frame_stride=int(payload["config"]["writer"]["frame_stride"]), max_open_files=2); started = time.monotonic()
    try:
        rows = _local_rows(args, context, payload, policy, writer, identity, lora, store, tokenizer, processor)
        if any(sha256_file(args.checkpoint / name) != digest for name, digest in payload["checkpoint_hashes"].items()): raise WriterModelError("analysis mutated checkpoint files")
        for row in rows: row["checkpoint_files_unchanged"] = True
        seal_rank_rows(args.output_dir, rank=context.rank, world_size=context.world_size, assigned_task_ids=payload["assignment"][context.rank], rows=rows, group=group)
        finalize(args.output_dir, rank=context.rank, world_size=context.world_size, references_per_task=args.references_per_task, result_schema=RESULT_SCHEMA, started=started, group=group)
    except Exception as error:
        record_failure(args.output_dir, context.rank, error); raise
    finally: store.close()
    if group is not None: dist.destroy_process_group(group)
    dist.destroy_process_group()
