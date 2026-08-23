"""Replay successful process teachers into policy-SFT HDF5 data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from ember.ecp.process_meta import (
    ProcessMetaError,
    TemporalPredicateOrderEnv,
    load_process_meta_authority,
)
from ember.pi05_assets import configure_libero_runtime_assets, write_json_atomic
from ember.pi05_processing import quat2axisangle


REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_SCHEMA = "ember_ecp_composite_teacher_dataset_v1"


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/pi05_ecp_process_meta_separate_plates_v1/manifest.json"
        ),
    )
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--state-ids", type=_state_ids)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-render-resolution", type=int, default=32)
    return parser


def _source_contract(source_root: Path) -> dict[str, Any]:
    path = source_root / "run_contract.json"
    if not path.is_file():
        raise ProcessMetaError("process source run contract is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value["policy"]["replan_steps"]) != 5:
        raise ProcessMetaError("process source policy contract changed")
    return value


def _policy_state(observation: Any) -> np.ndarray:
    value = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(observation["robot0_eef_quat"])),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    if value.shape != (8,):
        raise ProcessMetaError("replayed policy state is not eight-dimensional")
    return value


def _successful_ledgers(
    collection_root: Path,
    *,
    family_id: str,
    variant_name: str,
    requested_state_ids: tuple[int, ...] | None,
) -> tuple[tuple[Path, dict[str, Any]], ...]:
    pattern = f"{family_id}-{variant_name}-state*.pt"
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((collection_root / "privileged_ledgers").glob(pattern)):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("schema_version") != "ember_ecp_process_meta_privileged_episode_v1":
            raise ProcessMetaError("process privileged ledger schema changed")
        if bool(value["success"]):
            rows.append((path, value))
    rows.sort(key=lambda item: int(item[1]["state_id"]))
    if requested_state_ids is not None:
        requested = set(requested_state_ids)
        rows = [item for item in rows if int(item[1]["state_id"]) in requested]
        if {int(item[1]["state_id"]) for item in rows} != requested:
            raise ProcessMetaError("requested replay state is not a successful teacher row")
    if not rows:
        raise ProcessMetaError("no successful process teachers resolved")
    return tuple(rows)


def _public_video(
    collection_root: Path,
    ledger: dict[str, Any],
    *,
    exact_language: str,
    action_count: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    relative = ledger.get("public_video")
    if not relative:
        raise ProcessMetaError("successful teacher is missing its public video")
    path = collection_root / str(relative)
    with np.load(path, allow_pickle=False) as value:
        required = {
            "schema_version",
            "language",
            "camera1",
            "camera2",
            "source_steps",
            "model_frame_stride",
        }
        if set(value.files) != required:
            raise ProcessMetaError("public teacher video information wall changed")
        camera1 = np.asarray(value["camera1"], dtype=np.uint8)
        camera2 = np.asarray(value["camera2"], dtype=np.uint8)
        language = str(value["language"].item())
        steps = np.asarray(value["source_steps"])
    if (
        language != exact_language
        or camera1.shape != camera2.shape
        or camera1.ndim != 4
        or camera1.shape[0] != action_count + 1
        or camera1.shape[-1] != 3
        or not np.array_equal(steps, np.arange(action_count + 1, dtype=np.int32))
    ):
        raise ProcessMetaError("public teacher video is not aligned to its actions")
    return camera1, camera2, path


def _replay_states(
    env: TemporalPredicateOrderEnv,
    *,
    init_state: np.ndarray,
    actions: np.ndarray,
    expected: dict[str, Any],
    rollout: dict[str, Any],
) -> np.ndarray:
    env.seed(int(rollout["environment_seed"]))
    env.reset()
    observation = env.set_init_state(init_state)
    dummy = np.asarray(rollout["dummy_action"], dtype=np.float32)
    for _ in range(int(rollout["dummy_settling_steps"])):
        observation, _, _, _ = env.step(dummy)
    env.begin_episode()
    states = []
    for action in actions:
        states.append(_policy_state(observation))
        observation, _, _, _ = env.step(action)
    snapshot = env.snapshot()
    if (
        not snapshot["success"]
        or snapshot["invalid"]
        or int(snapshot["steps"]) != len(actions)
        or snapshot["completion_steps"] != expected["completion_steps"]
    ):
        raise ProcessMetaError("deterministic replay diverged from teacher evidence")
    return np.stack(states).astype(np.float32, copy=False)


def _write_demo(
    data: h5py.Group,
    *,
    demo_index: int,
    state_id: int,
    init_state: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    camera1: np.ndarray,
    camera2: np.ndarray,
    ledger_path: Path,
    video_path: Path,
    collection_root: Path,
    completion_steps: dict[str, int],
) -> None:
    demo = data.create_group(f"demo_{demo_index}")
    demo.attrs["num_samples"] = len(actions)
    demo.attrs["source_state_id"] = state_id
    demo.attrs["source_privileged_ledger"] = str(
        ledger_path.relative_to(collection_root)
    )
    demo.attrs["source_public_video"] = str(video_path.relative_to(collection_root))
    demo.attrs["completion_steps"] = json.dumps(completion_steps, sort_keys=True)
    demo.attrs["init_state"] = init_state
    demo.create_dataset("actions", data=actions)
    obs = demo.create_group("obs")
    obs.create_dataset("agentview_rgb", data=camera1[:-1])
    obs.create_dataset("eye_in_hand_rgb", data=camera2[:-1])
    obs.create_dataset("ee_states", data=states[:, :6])
    obs.create_dataset("gripper_states", data=states[:, 6:])


def _build_hdf5(
    args: argparse.Namespace,
    *,
    authority: Any,
    variant: Any,
    ledgers: tuple[tuple[Path, dict[str, Any]], ...],
    init_states: Any,
    env: TemporalPredicateOrderEnv,
) -> tuple[Path, Path, list[dict[str, Any]], int]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = args.output_dir / f"{variant.name}.hdf5"
    manifest_path = args.output_dir / f"{variant.name}.json"
    temporary = hdf5_path.with_suffix(f".hdf5.tmp.{os.getpid()}")
    if hdf5_path.exists() or manifest_path.exists() or temporary.exists():
        raise ProcessMetaError("composite teacher output already exists")
    episode_rows = []
    total_steps = 0
    try:
        with h5py.File(temporary, "w") as handle:
            data = handle.create_group("data")
            for demo_index, (ledger_path, ledger) in enumerate(ledgers):
                state_id = int(ledger["state_id"])
                actions = np.asarray(ledger["teacher_actions"], dtype=np.float32)
                camera1, camera2, video_path = _public_video(
                    args.collection_root,
                    ledger,
                    exact_language=authority.family.exact_language,
                    action_count=len(actions),
                )
                state = np.asarray(init_states[state_id])
                policy_states = _replay_states(
                    env,
                    init_state=state,
                    actions=actions,
                    expected=ledger,
                    rollout=dict(authority.rollout),
                )
                _write_demo(
                    data,
                    demo_index=demo_index,
                    state_id=state_id,
                    init_state=state,
                    actions=actions,
                    states=policy_states,
                    camera1=camera1,
                    camera2=camera2,
                    ledger_path=ledger_path,
                    video_path=video_path,
                    collection_root=args.collection_root,
                    completion_steps=dict(ledger["completion_steps"]),
                )
                total_steps += len(actions)
                episode_rows.append(
                    {
                        "demo_index": demo_index,
                        "source_state_id": state_id,
                        "steps": len(actions),
                        "completion_steps": dict(ledger["completion_steps"]),
                    }
                )
            data.attrs["schema_version"] = DATASET_SCHEMA
            data.attrs["family_id"] = authority.family.family_id
            data.attrs["variant_name"] = variant.name
            data.attrs["required_order"] = json.dumps(list(variant.required_order))
            data.attrs["language"] = authority.family.exact_language
            data.attrs["num_demos"] = len(episode_rows)
            data.attrs["total"] = total_steps
        os.replace(temporary, hdf5_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return hdf5_path, manifest_path, episode_rows, total_steps


def _dataset_manifest(
    args: argparse.Namespace,
    *,
    authority: Any,
    variant: Any,
    hdf5_path: Path,
    episode_rows: list[dict[str, Any]],
    total_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA,
        "status": "completed_privileged_composite_bootstrap",
        "family_id": authority.family.family_id,
        "variant_name": variant.name,
        "required_order": list(variant.required_order),
        "exact_language": authority.family.exact_language,
        "source_collection_root": str(args.collection_root),
        "hdf5": {
            "filename": hdf5_path.name,
            "bytes": hdf5_path.stat().st_size,
        },
        "episodes": len(episode_rows),
        "demo_indices": [0, len(episode_rows) - 1],
        "source_state_ids": [row["source_state_id"] for row in episode_rows],
        "total_action_steps": total_steps,
        "rows": episode_rows,
        "replay": {
            "render_resolution": args.replay_render_resolution,
            "matched_successes": len(episode_rows),
            "divergences": 0,
        },
        "information_wall": {
            "role": "privileged_order_specific_composite_policy_sft_only",
            "target40_action_reads": 0,
            "deployment_uses_this_dataset": False,
            "deployment_inputs_unchanged": [
                "exact_language",
                "action_hidden_ordered_video_frames",
            ],
        },
        "content_hash_policy": "disabled_by_owner",
    }


def main() -> None:
    args = build_parser().parse_args()
    args.manifest = args.manifest.resolve()
    args.collection_root = args.collection_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.replay_render_resolution <= 0:
        raise ProcessMetaError("replay render resolution must be positive")
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_root = REPO_ROOT / str(raw["source_policy_authority"])
    source_contract = _source_contract(source_root)
    os.environ.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu_id),
        LIBERO_CONFIG_PATH=str((source_root / "libero_config").resolve()),
    )
    authority = load_process_meta_authority(
        args.manifest,
        repo_root=REPO_ROOT,
        libero_init_root=Path(source_contract["libero_paths"]["init_states"]),
    )
    variant = authority.family.variant(args.variant)
    ledgers = _successful_ledgers(
        args.collection_root,
        family_id=authority.family.family_id,
        variant_name=variant.name,
        requested_state_ids=args.state_ids,
    )
    configure_libero_runtime_assets(Path(source_contract["libero_paths"]["assets"]))
    from libero.libero.envs import OffScreenRenderEnv

    init_states = torch.load(
        authority.family.init_states_path, map_location="cpu", weights_only=False
    )
    base_env = OffScreenRenderEnv(
        bddl_file_name=authority.family.bddl_path,
        camera_heights=args.replay_render_resolution,
        camera_widths=args.replay_render_resolution,
    )
    env = TemporalPredicateOrderEnv(
        base_env,
        predicates=authority.family.predicates,
        required_order=variant.required_order,
    )
    try:
        hdf5_path, manifest_path, episode_rows, total_steps = _build_hdf5(
            args,
            authority=authority,
            variant=variant,
            ledgers=ledgers,
            init_states=init_states,
            env=env,
        )
    finally:
        env.close()
    manifest = _dataset_manifest(
        args,
        authority=authority,
        variant=variant,
        hdf5_path=hdf5_path,
        episode_rows=episode_rows,
        total_steps=total_steps,
    )
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
