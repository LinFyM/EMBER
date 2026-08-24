"""Collect and read one-round on-policy labels for composite process experts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

from ember.ecp.composite_distillation_data import (
    DISTILLATION_MANIFEST_SCHEMA,
    DISTILLATION_SHARD_SCHEMA,
)
from ember.ecp.process_meta import (
    ProcessMetaAuthority,
    ProcessMetaError,
    ProcessVariant,
    TemporalPredicateOrderEnv,
    load_process_meta_authority,
)
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_assets import configure_libero_runtime_assets, write_json_atomic
from ember.pi05_eval.worker_setup import load_policy
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import libero_policy_input, quat2axisangle
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]


def _state_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "state IDs must be comma-separated integers"
        ) from error
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("state IDs must be unique and nonnegative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--composite-manifest", type=Path, required=True)
    collect.add_argument("--phase-manifest", type=Path, required=True)
    collect.add_argument("--variant", required=True)
    collect.add_argument("--state-ids", type=_state_ids, required=True)
    collect.add_argument("--physical-gpu-id", type=int, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--collection-root", type=Path, required=True)
    finalize.add_argument("--variant", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def _source_contract(authority: ProcessMetaAuthority) -> dict[str, Any]:
    path = authority.source_evaluation_root / "run_contract.json"
    if not path.is_file():
        raise ProcessMetaError("process source run contract is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value["policy"]["replan_steps"]) != 5:
        raise ProcessMetaError("process source policy contract changed")
    return value


def _matching_authorities(
    composite_path: Path,
    phase_path: Path,
    *,
    init_root: Path,
) -> tuple[ProcessMetaAuthority, ProcessMetaAuthority]:
    composite = load_process_meta_authority(
        composite_path, repo_root=REPO_ROOT, libero_init_root=init_root
    )
    phase = load_process_meta_authority(
        phase_path, repo_root=REPO_ROOT, libero_init_root=init_root
    )
    if (
        composite.privileged_teacher_kind != "order_specific_composite_rank16_lora"
        or phase.privileged_teacher_kind != "phase_task_local_rank16_lora"
        or composite.family != phase.family
        or composite.rollout != phase.rollout
        or composite.source_evaluation_root != phase.source_evaluation_root
        or composite.expert_source_checkpoint != phase.expert_source_checkpoint
        or composite.lora_contract_path != phase.lora_contract_path
    ):
        raise ProcessMetaError("composite and phase distillation authorities differ")
    return composite, phase


def _policy_state(observation: Mapping[str, Any]) -> np.ndarray:
    value = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(observation["robot0_eef_quat"])),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    if value.shape != (8,):
        raise ProcessMetaError("distillation policy state is not eight-dimensional")
    return value


def _camera(observation: Mapping[str, Any], name: str) -> np.ndarray:
    value = np.asarray(observation[name], dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ProcessMetaError("distillation camera observation changed")
    return value.copy()


def _noise(
    policy: torch.nn.Module,
    authority: ProcessMetaAuthority,
    *,
    state_id: int,
    replan_index: int,
) -> tuple[torch.Tensor, int]:
    seed = policy_noise_seed(
        int(authority.rollout["policy_seed_root"]),
        authority.family.base_task_suite,
        authority.family.base_task_id,
        state_id,
        replan_index,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=generator,
        dtype=torch.float32,
    ).to(device="cuda:0")
    return noise, seed


def _predict(
    policy: torch.nn.Module,
    preprocess: Any,
    postprocess: Any,
    observation: Mapping[str, Any],
    language: str,
    noise: torch.Tensor,
    num_steps: int,
) -> np.ndarray:
    batch = preprocess(libero_policy_input(observation, language))
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(
            batch, noise=noise, num_steps=num_steps
        )
    value = postprocess(normalized).detach().cpu().numpy()[0]
    if value.shape != (50, 7):
        raise ProcessMetaError("distillation action chunk shape changed")
    return value.astype(np.float32, copy=False)


def _collect_episode(
    *,
    env: TemporalPredicateOrderEnv,
    policy: torch.nn.Module,
    preprocess: Any,
    postprocess: Any,
    authority: ProcessMetaAuthority,
    phase_authority: ProcessMetaAuthority,
    variant: ProcessVariant,
    init_state: np.ndarray,
    state_id: int,
    composite_state: Mapping[str, torch.Tensor],
    phase_states: Mapping[str, Mapping[str, torch.Tensor]],
    lora: Any,
) -> dict[str, Any]:
    rollout = authority.rollout
    dummy = np.asarray(rollout["dummy_action"], dtype=np.float32)
    env.seed(int(rollout["environment_seed"]))
    env.reset()
    observation = env.set_init_state(init_state)
    for _ in range(int(rollout["dummy_settling_steps"])):
        observation, _, _, _ = env.step(dummy)
    env.begin_episode()
    policy.reset()
    rows: dict[str, list[Any]] = {
        "camera1": [],
        "camera2": [],
        "state": [],
        "teacher_action_chunks": [],
        "behavior_action_chunks": [],
        "phase_ids": [],
        "noise_seeds": [],
    }
    phase_keys = tuple(authority.family.predicates)
    replan_index = 0
    while env.steps < int(rollout["horizon"]) and not env.success and not env.invalid:
        phase_key = env.phase_key
        if phase_key is None:
            break
        noise, seed = _noise(
            policy, authority, state_id=state_id, replan_index=replan_index
        )
        copy_task_lora_state_(policy, composite_state, lora)
        behavior = _predict(
            policy,
            preprocess,
            postprocess,
            observation,
            authority.family.exact_language,
            noise,
            int(rollout["num_inference_steps"]),
        )
        copy_task_lora_state_(policy, phase_states[phase_key], lora)
        teacher = _predict(
            policy,
            preprocess,
            postprocess,
            observation,
            phase_authority.family.phase_languages[phase_key],
            noise,
            int(rollout["num_inference_steps"]),
        )
        rows["camera1"].append(_camera(observation, "agentview_image"))
        rows["camera2"].append(_camera(observation, "robot0_eye_in_hand_image"))
        rows["state"].append(_policy_state(observation))
        rows["teacher_action_chunks"].append(teacher)
        rows["behavior_action_chunks"].append(behavior)
        rows["phase_ids"].append(phase_keys.index(phase_key))
        rows["noise_seeds"].append(seed)
        replan_index += 1
        for action in behavior[: int(rollout["replan_steps"])]:
            before_phase = env.phase_key
            observation, _, success, _ = env.step(action)
            if (
                success
                or env.invalid
                or env.steps >= int(rollout["horizon"])
                or env.phase_key != before_phase
            ):
                break
    snapshot = env.snapshot()
    return {
        **rows,
        "state_id": state_id,
        "variant_name": variant.name,
        "required_order": variant.required_order,
        "success": bool(snapshot["success"]),
        "invalid": bool(snapshot["invalid"]),
        "invalid_reason": snapshot["invalid_reason"],
        "steps": int(snapshot["steps"]),
        "completion_steps": dict(snapshot["completion_steps"]),
    }


def _write_episode(handle: h5py.File, result: Mapping[str, Any]) -> int:
    state_id = int(result["state_id"])
    group = handle["episodes"].create_group(f"state_{state_id:03d}")
    count = len(result["state"])
    if count <= 0:
        raise ProcessMetaError("distillation episode produced no policy queries")
    group.attrs["state_id"] = state_id
    group.attrs["success"] = bool(result["success"])
    group.attrs["invalid"] = bool(result["invalid"])
    group.attrs["invalid_reason"] = result["invalid_reason"] or ""
    group.attrs["steps"] = int(result["steps"])
    group.attrs["completion_steps"] = json.dumps(
        result["completion_steps"], sort_keys=True
    )
    for name in ("camera1", "camera2"):
        value = np.stack(result[name]).astype(np.uint8, copy=False)
        group.create_dataset(
            name,
            data=value,
            chunks=(1, *value.shape[1:]),
            compression="lzf",
            shuffle=True,
        )
    group.create_dataset("state", data=np.stack(result["state"]).astype(np.float32))
    for name in ("teacher_action_chunks", "behavior_action_chunks"):
        group.create_dataset(name, data=np.stack(result[name]).astype(np.float32))
    group.create_dataset(
        "phase_ids", data=np.asarray(result["phase_ids"], dtype=np.int8)
    )
    group.create_dataset(
        "policy_noise_seeds", data=np.asarray(result["noise_seeds"], dtype=np.int64)
    )
    return count


def _prepare_policy(
    composite: ProcessMetaAuthority,
    phase: ProcessMetaAuthority,
    variant: ProcessVariant,
    source: Mapping[str, Any],
) -> tuple[Any, ...]:
    normalization = json.loads(composite.normalization_path.read_text(encoding="utf-8"))
    policy, preprocess, postprocess = load_policy(
        Path(source["model"]["model_path"]),
        normalization["stats"],
        composite.tokenizer_path,
        source["policy"],
    )
    lora = load_pi05_lora_contract(composite.lora_contract_path)
    prepare_frozen_writer_policy(policy, lora)
    composite_state = load_file(
        str(composite.variant_experts[variant.name].adapter_path), device="cpu"
    )
    phase_states = {
        key: load_file(str(expert.adapter_path), device="cpu")
        for key, expert in phase.phase_experts.items()
    }
    for state in (composite_state, *phase_states.values()):
        validate_lora_state(state, lora)
    return policy, preprocess, postprocess, lora, composite_state, phase_states


def _prepare_environment(
    composite: ProcessMetaAuthority, variant: ProcessVariant, environment_class: Any
) -> TemporalPredicateOrderEnv:
    base = environment_class(
        bddl_file_name=composite.family.bddl_path,
        camera_heights=int(composite.rollout["render_resolution"]),
        camera_widths=int(composite.rollout["render_resolution"]),
    )
    return TemporalPredicateOrderEnv(
        base,
        predicates=composite.family.predicates,
        required_order=variant.required_order,
    )


def collect(args: argparse.Namespace) -> None:
    composite_path = args.composite_manifest.resolve()
    phase_path = args.phase_manifest.resolve()
    raw = json.loads(composite_path.read_text(encoding="utf-8"))
    source_root = REPO_ROOT / str(raw["source_policy_authority"])
    provisional = json.loads(
        (source_root / "run_contract.json").read_text(encoding="utf-8")
    )
    init_root = Path(provisional["libero_paths"]["init_states"])
    composite, phase = _matching_authorities(
        composite_path, phase_path, init_root=init_root
    )
    source = _source_contract(composite)
    variant = composite.family.variant(args.variant)
    if any(
        state_id not in composite.family.init_state_ids for state_id in args.state_ids
    ):
        raise ProcessMetaError("distillation state is outside the fixed panel")
    args.output_dir = args.output_dir.resolve()
    shard_name = f"{variant.name}-states-{'-'.join(map(str, args.state_ids))}.hdf5"
    shard = args.output_dir / "shards" / shard_name
    summary = args.output_dir / "workers" / shard_name.replace(".hdf5", ".json")
    if shard.exists() or summary.exists():
        raise ProcessMetaError("distillation shard output already exists")
    os.environ.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu_id),
        LIBERO_CONFIG_PATH=str((source_root / "libero_config").resolve()),
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise ProcessMetaError("distillation collector must see its declared GPU")
    configure_libero_runtime_assets(Path(source["libero_paths"]["assets"]))
    from libero.libero.envs import OffScreenRenderEnv

    torch.cuda.set_device(0)
    torch.manual_seed(int(composite.rollout["policy_seed_root"]))
    torch.cuda.manual_seed(int(composite.rollout["policy_seed_root"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    (
        policy,
        preprocess,
        postprocess,
        lora,
        composite_state,
        phase_states,
    ) = _prepare_policy(composite, phase, variant, source)
    init_states = torch.load(
        composite.family.init_states_path, map_location="cpu", weights_only=False
    )
    env = _prepare_environment(composite, variant, OffScreenRenderEnv)
    shard.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        with h5py.File(shard, "w") as handle:
            handle.attrs["schema_version"] = DISTILLATION_SHARD_SCHEMA
            handle.attrs["family_id"] = composite.family.family_id
            handle.attrs["variant_name"] = variant.name
            handle.attrs["language"] = composite.family.exact_language
            handle.attrs["phase_keys"] = json.dumps(list(composite.family.predicates))
            handle.create_group("episodes")
            for state_id in args.state_ids:
                result = _collect_episode(
                    env=env,
                    policy=policy,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    authority=composite,
                    phase_authority=phase,
                    variant=variant,
                    init_state=np.asarray(init_states[state_id]),
                    state_id=state_id,
                    composite_state=composite_state,
                    phase_states=phase_states,
                    lora=lora,
                )
                count = _write_episode(handle, result)
                rows.append(
                    {
                        "state_id": state_id,
                        "queries": count,
                        "success": result["success"],
                        "invalid": result["invalid"],
                        "invalid_reason": result["invalid_reason"],
                        "steps": result["steps"],
                        "completion_steps": result["completion_steps"],
                    }
                )
    finally:
        env.close()
    payload = {
        "schema_version": DISTILLATION_SHARD_SCHEMA,
        "family_id": composite.family.family_id,
        "variant_name": variant.name,
        "required_order": list(variant.required_order),
        "language": composite.family.exact_language,
        "state_ids": list(args.state_ids),
        "queries": sum(int(row["queries"]) for row in rows),
        "shard": str(shard.relative_to(args.output_dir)),
        "shard_bytes": shard.stat().st_size,
        "rows": rows,
        "information_wall": {
            "behavior": "fixed_step1000_composite_expert",
            "labels": "matching_phase_expert_action_chunks",
            "policy_condition": "exact_unified_composite_language_only",
            "target40_action_reads": 0,
        },
    }
    write_json_atomic(summary, payload)
    print(json.dumps(payload, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    root = args.collection_root.resolve()
    summaries = []
    for path in sorted((root / "workers").glob(f"{args.variant}-states-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != DISTILLATION_SHARD_SCHEMA
            or value.get("variant_name") != args.variant
        ):
            raise ProcessMetaError("distillation worker summary changed")
        summaries.append((path, value))
    if not summaries:
        raise ProcessMetaError("no distillation worker summaries resolved")
    state_rows = [row for _, value in summaries for row in value["rows"]]
    state_ids = sorted(int(row["state_id"]) for row in state_rows)
    if state_ids != list(range(50)) or len(set(state_ids)) != 50:
        raise ProcessMetaError("distillation manifest requires the fixed50 panel")
    first = summaries[0][1]
    shards = []
    for _, value in summaries:
        if (
            value["family_id"] != first["family_id"]
            or value["language"] != first["language"]
            or value["required_order"] != first["required_order"]
        ):
            raise ProcessMetaError("distillation shards disagree on task identity")
        shard = root / value["shard"]
        if not shard.is_file() or shard.stat().st_size != int(value["shard_bytes"]):
            raise ProcessMetaError("distillation shard path or size changed")
        shards.append(
            {
                "path": str(shard.relative_to(root)),
                "bytes": shard.stat().st_size,
                "state_ids": list(value["state_ids"]),
                "queries": int(value["queries"]),
            }
        )
    payload = {
        "schema_version": DISTILLATION_MANIFEST_SCHEMA,
        "status": "completed_one_round_on_policy_phase_distillation",
        "family_id": first["family_id"],
        "variant_name": args.variant,
        "required_order": first["required_order"],
        "exact_language": first["language"],
        "state_ids": state_ids,
        "queries": sum(int(row["queries"]) for row in state_rows),
        "shards": shards,
        "behavior_outcomes": {
            "successes": sum(bool(row["success"]) for row in state_rows),
            "invalid": sum(bool(row["invalid"]) for row in state_rows),
            "episodes": 50,
        },
        "training_target": "phase_expert_full50_action_chunk_on_composite_occupancy",
        "information_wall": {
            "deployment_uses_dataset": False,
            "policy_input": ["current_observation", "exact_unified_language"],
            "privileged_label_only": ["phase_key", "phase_language", "phase_expert"],
            "target40_action_reads": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    args.output = args.output.resolve()
    if args.output.exists():
        raise ProcessMetaError("distillation manifest output already exists")
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        collect(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
