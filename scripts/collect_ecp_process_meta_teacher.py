#!/usr/bin/env python3
"""Collect privileged phase-switched teachers for one process-meta variant."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

from ember.ecp.process_meta import (
    ProcessMetaError,
    TemporalPredicateOrderEnv,
    load_process_meta_authority,
)
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_assets import configure_libero_runtime_assets, write_json_atomic
from ember.pi05_eval.worker_setup import load_policy
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import libero_policy_input
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[1]


def _state_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("state IDs must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("state IDs must be unique and nonnegative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_process_meta_v1/manifest.json",
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--state-ids", type=_state_ids, required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-mode",
        choices=("source_phase", "phase_expert"),
        default="source_phase",
    )
    parser.add_argument("--keep-failure-videos", action="store_true")
    return parser


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_video_save(
    path: Path,
    *,
    language: str,
    camera1: list[np.ndarray],
    camera2: list[np.ndarray],
    frame_stride: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray("ember_ecp_action_hidden_video_v1"),
            language=np.asarray(language),
            camera1=np.stack(camera1).astype(np.uint8, copy=False),
            camera2=np.stack(camera2).astype(np.uint8, copy=False),
            source_steps=np.arange(len(camera1), dtype=np.int32),
            model_frame_stride=np.asarray(frame_stride, dtype=np.int32),
        )
    os.replace(temporary, path)


def _capture(observation: Any) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(observation["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8)
    if base.ndim != 3 or wrist.shape != base.shape or base.shape[-1] != 3:
        raise ProcessMetaError("process-meta RGB observation shape changed")
    return base.copy(), wrist.copy()


def _noise(
    policy: torch.nn.Module,
    *,
    root_seed: int,
    task_suite: str,
    task_id: int,
    state_id: int,
    replan_index: int,
) -> tuple[torch.Tensor, int]:
    seed = policy_noise_seed(
        root_seed,
        task_suite,
        task_id,
        state_id,
        replan_index,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device="cuda:0")
    return value, seed


def _collect_episode(
    *,
    env: TemporalPredicateOrderEnv,
    policy: torch.nn.Module,
    preprocess: Any,
    postprocess: Any,
    init_state: np.ndarray,
    state_id: int,
    noise_task_suite: str,
    noise_task_id: int,
    phase_languages: Any,
    install_phase_expert: Callable[[str], None] | None,
    phase_expert_task_ids: Mapping[str, int],
    exact_language: str,
    rollout: Any,
) -> dict[str, Any]:
    dummy = np.asarray(rollout["dummy_action"], dtype=np.float32)
    env.seed(int(rollout["environment_seed"]))
    env.reset()
    observation = env.set_init_state(init_state)
    for _ in range(int(rollout["dummy_settling_steps"])):
        observation, _, _, _ = env.step(dummy)
    env.begin_episode()
    policy.reset()
    base, wrist = _capture(observation)
    camera1, camera2 = [base], [wrist]
    actions: list[np.ndarray] = []
    action_phase_keys: list[str] = []
    replan_phase_keys: list[str] = []
    replan_teacher_task_ids: list[int | None] = []
    noise_seeds: list[int] = []
    replan_index = 0
    started = time.monotonic()
    horizon = int(rollout["horizon"])
    while env.steps < horizon and not env.success and not env.invalid:
        phase_key = env.phase_key
        if phase_key is None:
            break
        if install_phase_expert is not None:
            install_phase_expert(phase_key)
        language = str(phase_languages[phase_key])
        batch = preprocess(libero_policy_input(observation, language))
        noise, seed = _noise(
            policy,
            root_seed=int(rollout["policy_seed_root"]),
            task_suite=noise_task_suite,
            task_id=noise_task_id,
            state_id=state_id,
            replan_index=replan_index,
        )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(
                batch,
                noise=noise,
                num_steps=int(rollout["num_inference_steps"]),
            )
        environment_actions = postprocess(normalized).detach().cpu().numpy()[0]
        replan_phase_keys.append(phase_key)
        replan_teacher_task_ids.append(phase_expert_task_ids.get(phase_key))
        noise_seeds.append(seed)
        replan_index += 1
        for action in environment_actions[: int(rollout["replan_steps"])]:
            before_phase = env.phase_key
            observation, _, success, _ = env.step(action)
            actions.append(np.asarray(action, dtype=np.float32).copy())
            action_phase_keys.append(str(before_phase))
            base, wrist = _capture(observation)
            camera1.append(base)
            camera2.append(wrist)
            if success or env.invalid or env.steps >= horizon or env.phase_key != before_phase:
                break
    snapshot = env.snapshot()
    return {
        "success": bool(snapshot["success"]),
        "invalid": bool(snapshot["invalid"]),
        "invalid_reason": snapshot["invalid_reason"],
        "steps": int(snapshot["steps"]),
        "completion_steps": dict(snapshot["completion_steps"]),
        "predicate_values": dict(snapshot["predicate_values"]),
        "elapsed_seconds": time.monotonic() - started,
        "camera1": camera1,
        "camera2": camera2,
        "actions": np.stack(actions) if actions else np.empty((0, 7), dtype=np.float32),
        "action_phase_keys": tuple(action_phase_keys),
        "replan_phase_keys": tuple(replan_phase_keys),
        "replan_teacher_task_ids": tuple(replan_teacher_task_ids),
        "policy_noise_seeds": tuple(noise_seeds),
        "exact_language": exact_language,
    }


def _runtime_contract(source_root: Path) -> dict[str, Any]:
    path = source_root / "run_contract.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not path.is_file() or int(value["policy"]["replan_steps"]) != 5:
        raise ProcessMetaError("source evaluation authority changed")
    return value


def main() -> None:
    args = build_parser().parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_root = REPO_ROOT / str(raw["source_policy_authority"])
    source_contract = _runtime_contract(source_root)
    os.environ.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu_id),
        LIBERO_CONFIG_PATH=str((source_root / "libero_config").resolve()),
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise ProcessMetaError("collector must see exactly its declared physical GPU")
    authority = load_process_meta_authority(
        args.manifest,
        repo_root=REPO_ROOT,
        libero_init_root=Path(source_contract["libero_paths"]["init_states"]),
    )
    variant = authority.family.variant(args.variant)
    if any(state_id not in authority.family.init_state_ids for state_id in args.state_ids):
        raise ProcessMetaError("requested state is outside the fixed process-meta panel")
    configure_libero_runtime_assets(Path(source_contract["libero_paths"]["assets"]))
    from libero.libero.envs import OffScreenRenderEnv

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProcessMetaError("process-meta collector requires one visible CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(int(authority.rollout["policy_seed_root"]))
    torch.cuda.manual_seed(int(authority.rollout["policy_seed_root"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    normalization = json.loads(authority.normalization_path.read_text(encoding="utf-8"))
    policy, preprocess, postprocess = load_policy(
        Path(source_contract["model"]["model_path"]),
        normalization["stats"],
        authority.tokenizer_path,
        source_contract["policy"],
    )
    install_phase_expert: Callable[[str], None] | None = None
    phase_expert_task_ids: dict[str, int] = {}
    if args.teacher_mode == "phase_expert":
        if (
            authority.lora_contract_path is None
            or authority.expert_source_checkpoint is None
            or set(authority.phase_experts) != set(authority.family.predicates)
            or authority.expert_source_checkpoint.resolve()
            != Path(source_contract["model"]["checkpoint"]).resolve()
        ):
            raise ProcessMetaError("phase-expert teacher authority is incomplete")
        lora = load_pi05_lora_contract(authority.lora_contract_path)
        prepare_frozen_writer_policy(policy, lora)
        expert_states = {
            phase_key: load_file(str(expert.adapter_path), device="cpu")
            for phase_key, expert in authority.phase_experts.items()
        }
        for state in expert_states.values():
            validate_lora_state(state, lora)
        installed_phase: str | None = None

        def install(phase_key: str) -> None:
            nonlocal installed_phase
            if phase_key != installed_phase:
                copy_task_lora_state_(policy, expert_states[phase_key], lora)
                installed_phase = phase_key

        install_phase_expert = install
        phase_expert_task_ids = {
            phase_key: expert.task_id
            for phase_key, expert in authority.phase_experts.items()
        }
    elif authority.phase_experts:
        raise ProcessMetaError("phase-expert manifest requires --teacher-mode phase_expert")
    init_states = torch.load(
        authority.family.init_states_path,
        map_location="cpu",
        weights_only=False,
    )
    base_env = OffScreenRenderEnv(
        bddl_file_name=authority.family.bddl_path,
        camera_heights=int(authority.rollout["render_resolution"]),
        camera_widths=int(authority.rollout["render_resolution"]),
    )
    env = TemporalPredicateOrderEnv(
        base_env,
        predicates=authority.family.predicates,
        required_order=variant.required_order,
    )
    rows = []
    try:
        for state_id in args.state_ids:
            result = _collect_episode(
                env=env,
                policy=policy,
                preprocess=preprocess,
                postprocess=postprocess,
                init_state=np.asarray(init_states[state_id]),
                state_id=state_id,
                noise_task_suite=authority.family.base_task_suite,
                noise_task_id=authority.family.base_task_id,
                phase_languages=authority.family.phase_languages,
                install_phase_expert=install_phase_expert,
                phase_expert_task_ids=phase_expert_task_ids,
                exact_language=authority.family.exact_language,
                rollout=authority.rollout,
            )
            episode_id = f"{authority.family.family_id}-{variant.name}-state{state_id:03d}"
            public_path = args.output_dir / "public_videos" / f"{episode_id}.npz"
            keep_video = bool(result["success"] or args.keep_failure_videos)
            if keep_video:
                _atomic_video_save(
                    public_path,
                    language=authority.family.exact_language,
                    camera1=result["camera1"],
                    camera2=result["camera2"],
                    frame_stride=int(authority.rollout["frame_stride"]),
                )
            privileged_path = (
                args.output_dir / "privileged_ledgers" / f"{episode_id}.pt"
            )
            _atomic_torch_save(
                privileged_path,
                {
                    "schema_version": "ember_ecp_process_meta_privileged_episode_v1",
                    "family_id": authority.family.family_id,
                    "variant_name": variant.name,
                    "required_order": variant.required_order,
                    "state_id": state_id,
                    "success": result["success"],
                    "invalid": result["invalid"],
                    "invalid_reason": result["invalid_reason"],
                    "steps": result["steps"],
                    "completion_steps": result["completion_steps"],
                    "predicate_values": result["predicate_values"],
                    "teacher_actions": torch.from_numpy(result["actions"]),
                    "teacher_mode": args.teacher_mode,
                    "action_phase_keys": result["action_phase_keys"],
                    "replan_phase_keys": result["replan_phase_keys"],
                    "replan_teacher_task_ids": result["replan_teacher_task_ids"],
                    "policy_noise_seeds": result["policy_noise_seeds"],
                    "public_video": (
                        str(public_path.relative_to(args.output_dir))
                        if keep_video
                        else None
                    ),
                },
            )
            rows.append(
                {
                    "family_id": authority.family.family_id,
                    "variant_name": variant.name,
                    "state_id": state_id,
                    "success": result["success"],
                    "invalid": result["invalid"],
                    "invalid_reason": result["invalid_reason"],
                    "steps": result["steps"],
                    "completion_steps": result["completion_steps"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "public_video": (
                        str(public_path.relative_to(args.output_dir))
                        if keep_video
                        else None
                    ),
                    "public_video_bytes": public_path.stat().st_size if keep_video else 0,
                    "privileged_ledger": str(privileged_path.relative_to(args.output_dir)),
                    "privileged_ledger_bytes": privileged_path.stat().st_size,
                }
            )
    finally:
        env.close()
    summary = {
        "schema_version": "ember_ecp_process_meta_teacher_collection_v1",
        "manifest": str(args.manifest),
        "family_id": authority.family.family_id,
        "teacher_mode": args.teacher_mode,
        "phase_expert_task_ids": phase_expert_task_ids,
        "variant_name": variant.name,
        "required_order": list(variant.required_order),
        "state_ids": list(args.state_ids),
        "successes": sum(bool(row["success"]) for row in rows),
        "episodes": len(rows),
        "rows": rows,
    }
    summary_path = (
        args.output_dir
        / "workers"
        / f"{variant.name}-states-{'-'.join(map(str, args.state_ids))}.json"
    )
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
