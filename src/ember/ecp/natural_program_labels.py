"""Compact training-only dynamic labels for G2 Natural Program."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

import numpy as np

from ember.pi05_assets import prepare_libero_config, write_json_atomic
from ember.pi05_source_checkpoint import read_json


LABEL_SCHEMA = "ember_ecp_natural_program_labels_v2"
_FILE_ATTRIBUTE = re.compile(r'file=(["\'])(.+?)\1')


@dataclass(frozen=True)
class VideoDynamicLabels:
    progress: np.ndarray
    rising: np.ndarray
    contact: np.ndarray
    contact_mask: np.ndarray
    predicates: np.ndarray
    predicate_mask: np.ndarray


class NaturalProgramLabelStore:
    """Read labels without exposing them to the deployment forward."""

    def __init__(
        self,
        root: Path,
        *,
        tasks: Sequence[Any],
        predicate_slots: int,
        max_open_tasks: int = 8,
    ) -> None:
        self.root = root.resolve()
        self.manifest = read_json(self.root / "manifest.json")
        self.predicate_slots = int(predicate_slots)
        self.max_open_tasks = int(max_open_tasks)
        if (
            self.manifest.get("schema_version") != LABEL_SCHEMA
            or self.manifest.get("status") != "complete"
            or int(self.manifest.get("predicate_slots", -1)) != self.predicate_slots
            or self.max_open_tasks <= 0
        ):
            raise ValueError("Natural Program derived-label authority changed")
        records = {
            int(row["authority_id"]): row for row in self.manifest.get("tasks", [])
        }
        if len(records) != len(tasks):
            raise ValueError("Natural Program label task count changed")
        self.paths: dict[int, Path] = {}
        for task in tasks:
            record = records.get(int(task.authority_id))
            if (
                record is None
                or record.get("domain") != task.domain
                or int(record.get("domain_task_id", -1)) != task.domain_task_id
                or int(record.get("source_hdf5_bytes", -1)) != task.expected_bytes
                or int(record.get("demonstrations", -1)) != 50
            ):
                raise ValueError(
                    f"Natural Program label provenance changed: {task.authority_id}"
                )
            path = self.root / str(record["labels_file"])
            if not path.is_file() or path.stat().st_size != int(record["labels_bytes"]):
                raise ValueError(
                    f"Natural Program label file changed: {task.authority_id}"
                )
            self.paths[task.authority_id] = path
        self._arrays: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def _task(self, authority_id: int) -> dict[str, np.ndarray]:
        if authority_id not in self.paths:
            raise ValueError("Natural Program labels escaped task authority")
        if authority_id in self._arrays:
            self._arrays.move_to_end(authority_id)
            return self._arrays[authority_id]
        with np.load(self.paths[authority_id], allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        offsets = arrays.get("offsets")
        if (
            offsets is None
            or offsets.shape != (51,)
            or int(offsets[0]) != 0
            or np.any(offsets[1:] <= offsets[:-1])
            or arrays.get("progress", np.empty(0)).shape != (int(offsets[-1]),)
            or arrays.get("rising", np.empty(0)).shape != (int(offsets[-1]),)
            or arrays.get("contact", np.empty(0)).shape != (int(offsets[-1]),)
            or arrays.get("contact_mask", np.empty(0)).shape
            != (int(offsets[-1]),)
            or arrays.get("predicates", np.empty(0)).shape
            != (int(offsets[-1]), self.predicate_slots)
            or arrays.get("predicate_mask", np.empty(0)).shape
            != (self.predicate_slots,)
        ):
            raise ValueError("Natural Program label arrays changed schema")
        self._arrays[authority_id] = arrays
        while len(self._arrays) > self.max_open_tasks:
            self._arrays.popitem(last=False)
        return arrays

    def load(self, authority_id: int, demo_index: int) -> VideoDynamicLabels:
        if not 0 <= demo_index < 50:
            raise ValueError("Natural Program label demo is outside 0..49")
        arrays = self._task(authority_id)
        start, stop = map(int, arrays["offsets"][demo_index : demo_index + 2])
        return VideoDynamicLabels(
            progress=arrays["progress"][start:stop],
            rising=arrays["rising"][start:stop],
            contact=arrays["contact"][start:stop],
            contact_mask=arrays["contact_mask"][start:stop],
            predicates=arrays["predicates"][start:stop],
            predicate_mask=arrays["predicate_mask"],
        )

    def close(self) -> None:
        self._arrays.clear()


def _rewrite_model_paths(xml: str, *, assets_root: Path) -> str:
    import robosuite

    robosuite_root = Path(robosuite.__file__).resolve().parent

    def replace(match: re.Match[str]) -> str:
        old = match.group(2)
        if "/chiliocosm/assets/" in old:
            path = assets_root / old.split("/chiliocosm/assets/", 1)[1]
        elif "/robosuite/" in old:
            path = robosuite_root / old.rsplit("/robosuite/", 1)[1]
        else:
            return match.group(0)
        if not path.is_file():
            raise FileNotFoundError(f"Natural Program XML asset is missing: {path}")
        quote = match.group(1)
        return f"file={quote}{path}{quote}"

    return _FILE_ATTRIBUTE.sub(replace, xml)


def _rewrite_legacy_model_names(xml: str, owner: Any) -> str:
    """Match stored pre-rename object XML to installed BDDL model names."""

    for name in {**owner.objects_dict, **owner.fixtures_dict}:
        if not name.startswith("new_"):
            continue
        legacy = name.removeprefix("new_")
        if f"{name}_" not in xml and f"{legacy}_" in xml:
            xml = xml.replace(f"{legacy}_", f"{name}_")
    return xml


def _gripper_object_contact(owner: Any) -> bool:
    entities = {**owner.objects_dict, **owner.fixtures_dict}
    entity_geoms = {
        str(geom)
        for model in entities.values()
        for geom in model.contact_geoms
    }
    gripper_geoms = set(map(str, owner.robots[0].gripper.contact_geoms))
    for contact in owner.sim.data.contact[: owner.sim.data.ncon]:
        left = owner.sim.model.geom_id2name(contact.geom1)
        right = owner.sim.model.geom_id2name(contact.geom2)
        if (left in gripper_geoms and right in entity_geoms) or (
            right in gripper_geoms and left in entity_geoms
        ):
            return True
    return False


def _predicate_rising(
    predicates: np.ndarray,
    initial_predicates: np.ndarray,
    *,
    goal_count: int,
) -> np.ndarray:
    """Mark every false-to-true transition, including state0 to obs0."""

    rising = np.zeros(predicates.shape[0], dtype=np.uint8)
    rising[0] = np.any(
        (predicates[0, :goal_count] == 1)
        & (initial_predicates[:goal_count] == 0)
    )
    rising[1:] = np.any(
        (predicates[1:, :goal_count] == 1)
        & (predicates[:-1, :goal_count] == 0),
        axis=1,
    )
    return rising


def _derive_demo_labels(
    owner: Any,
    environment: Any,
    states: np.ndarray,
    *,
    goal_count: int,
    predicate_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    environment.set_state(states[0])
    owner.sim.forward()
    owner._post_process()
    initial_predicates = np.zeros(predicate_slots, dtype=np.uint8)
    initial_predicates[:goal_count] = tuple(
        bool(owner._eval_predicate(goal))
        for goal in owner.parsed_problem["goal_state"]
    )

    length = states.shape[0]
    predicates = np.zeros((length, predicate_slots), dtype=np.uint8)
    contact = np.zeros(length, dtype=np.uint8)
    contact_mask = np.ones(length, dtype=np.uint8)
    # HDF5 obs[i] is the post-action observation represented by states[i+1].
    # The final post-action state was not stored.
    for frame_index in range(length - 1):
        environment.set_state(states[frame_index + 1])
        owner.sim.forward()
        owner._post_process()
        predicates[frame_index, :goal_count] = tuple(
            bool(owner._eval_predicate(goal))
            for goal in owner.parsed_problem["goal_state"]
        )
        contact[frame_index] = _gripper_object_contact(owner)
    # Every sealed demo is successful, so the absent terminal post-action
    # observation satisfies all goal predicates; its contact stays masked.
    predicates[-1, :goal_count] = 1
    contact_mask[-1] = 0
    progress = predicates[:, :goal_count].mean(1).astype(np.float32)
    rising = _predicate_rising(
        predicates, initial_predicates, goal_count=goal_count
    )
    return progress, rising, contact, contact_mask, predicates


def _seal_task_labels(
    task: Any,
    *,
    output_root: Path,
    assets_root: Path,
    predicate_slots: int,
) -> dict[str, Any]:
    """Restore every demo XML/state and derive compact training-only labels."""

    os.environ.setdefault("MUJOCO_GL", "disable")
    os.environ["EMBER_LIBERO_ASSETS_ROOT"] = str(assets_root)
    with tempfile.TemporaryDirectory(prefix="ember-g2-libero-config-") as config_dir:
        paths = prepare_libero_config(Path(config_dir))
        from ember.pi05_assets import configure_libero_runtime_assets

        configure_libero_runtime_assets(assets_root)
        from libero.libero.envs.env_wrapper import ControlEnv
        import h5py

        bddl_path = Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file
        if not bddl_path.is_file():
            raise FileNotFoundError(f"Natural Program BDDL is missing: {bddl_path}")
        environment = ControlEnv(
            bddl_file_name=str(bddl_path),
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            camera_names=[],
        )
        try:
            owner = environment.env
            goal_count = len(owner.parsed_problem["goal_state"])
            if not 1 <= goal_count <= predicate_slots:
                raise ValueError(
                    f"Natural Program predicate capacity changed: {goal_count}"
                )
            offsets = [0]
            progress_rows: list[np.ndarray] = []
            rising_rows: list[np.ndarray] = []
            contact_rows: list[np.ndarray] = []
            contact_mask_rows: list[np.ndarray] = []
            predicate_rows: list[np.ndarray] = []
            with h5py.File(task.path, "r") as source:
                for demo_index, expected_length in enumerate(task.episode_lengths):
                    demo = source[f"data/demo_{demo_index}"]
                    states = np.asarray(demo["states"])
                    if states.shape[0] != expected_length:
                        raise ValueError(
                            f"Natural Program episode length changed: "
                            f"{task.authority_id}/{demo_index}"
                        )
                    model_file = demo.attrs["model_file"]
                    if isinstance(model_file, bytes):
                        model_file = model_file.decode("utf-8")
                    xml = _rewrite_model_paths(
                        str(model_file), assets_root=assets_root
                    )
                    environment.reset_from_xml_string(
                        _rewrite_legacy_model_names(xml, owner)
                    )

                    (
                        progress,
                        rising,
                        contact,
                        contact_mask,
                        predicates,
                    ) = _derive_demo_labels(
                        owner,
                        environment,
                        states,
                        goal_count=goal_count,
                        predicate_slots=predicate_slots,
                    )
                    progress_rows.append(progress)
                    rising_rows.append(rising)
                    contact_rows.append(contact)
                    contact_mask_rows.append(contact_mask)
                    predicate_rows.append(predicates)
                    offsets.append(offsets[-1] + expected_length)
        finally:
            environment.close()

    labels_path = output_root / f"task_{task.authority_id:03d}.npz"
    temporary = labels_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        offsets=np.asarray(offsets, dtype=np.int64),
        progress=np.concatenate(progress_rows),
        rising=np.concatenate(rising_rows),
        contact=np.concatenate(contact_rows),
        contact_mask=np.concatenate(contact_mask_rows),
        predicates=np.concatenate(predicate_rows),
        predicate_mask=np.arange(predicate_slots) < goal_count,
    )
    os.replace(temporary, labels_path)
    return {
        "authority_id": int(task.authority_id),
        "domain": str(task.domain),
        "domain_task_id": int(task.domain_task_id),
        "role": str(task.role),
        "source_hdf5_bytes": int(task.expected_bytes),
        "demonstrations": 50,
        "raw_frames": int(offsets[-1]),
        "goal_predicates": int(goal_count),
        "labels_file": labels_path.name,
        "labels_bytes": labels_path.stat().st_size,
    }


def _existing_task_record(
    task: Any, *, output_root: Path, predicate_slots: int
) -> dict[str, Any]:
    path = output_root / f"task_{task.authority_id:03d}.npz"
    with np.load(path, allow_pickle=False) as loaded:
        offsets = np.asarray(loaded["offsets"])
        predicate_mask = np.asarray(loaded["predicate_mask"])
        expected_offsets = np.asarray(
            [0, *np.cumsum(task.episode_lengths).tolist()], dtype=np.int64
        )
        total = int(expected_offsets[-1])
        valid = (
            np.array_equal(offsets, expected_offsets)
            and predicate_mask.shape == (predicate_slots,)
            and 1 <= int(predicate_mask.sum()) <= predicate_slots
            and loaded["progress"].shape == (total,)
            and loaded["rising"].shape == (total,)
            and loaded["contact"].shape == (total,)
            and loaded["contact_mask"].shape == (total,)
            and loaded["predicates"].shape == (total, predicate_slots)
        )
    if not valid:
        raise ValueError(
            f"partial Natural Program label file changed: {task.authority_id}"
        )
    return {
        "authority_id": int(task.authority_id),
        "domain": str(task.domain),
        "domain_task_id": int(task.domain_task_id),
        "role": str(task.role),
        "source_hdf5_bytes": int(task.expected_bytes),
        "demonstrations": 50,
        "raw_frames": total,
        "goal_predicates": int(predicate_mask.sum()),
        "labels_file": path.name,
        "labels_bytes": path.stat().st_size,
    }


def seal_natural_program_labels(
    *,
    tasks: Sequence[Any],
    output_root: Path,
    assets_root: Path,
    predicate_slots: int,
    workers: int,
    resume_partial: bool = False,
) -> dict[str, Any]:
    """Seal the 95-task G2 label bank; held labels remain evaluation-only."""

    output_root = output_root.resolve()
    assets_root = assets_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not resume_partial:
        raise FileExistsError(f"Natural Program label output is not empty: {output_root}")
    if not assets_root.is_dir() or not 1 <= workers <= 32:
        raise ValueError("Natural Program label assets/workers are invalid")
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    pending = []
    for task in tasks:
        path = output_root / f"task_{task.authority_id:03d}.npz"
        if resume_partial and path.is_file():
            records.append(
                _existing_task_record(
                    task, output_root=output_root, predicate_slots=predicate_slots
                )
            )
        else:
            pending.append(task)
    arguments = [
        {
            "task": task,
            "output_root": output_root,
            "assets_root": assets_root,
            "predicate_slots": predicate_slots,
        }
        for task in pending
    ]
    if workers == 1:
        records.extend(_seal_task_labels(**row) for row in arguments)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_seal_task_labels, **row) for row in arguments]
            for index, future in enumerate(futures, start=1):
                record = future.result()
                records.append(record)
                print(
                    f"sealed pending Natural Program labels {index}/{len(futures)}: "
                    f"authority_id={record['authority_id']}",
                    flush=True,
                )
    records.sort(key=lambda row: row["authority_id"])
    role_counts = {
        role: sum(record["role"] == role for record in records)
        for role in ("meta_fit", "meta_held", "target_fit", "target_held")
    }
    complete = len(records) == 95 and role_counts == {
        "meta_fit": 56,
        "meta_held": 15,
        "target_fit": 19,
        "target_held": 5,
    }
    manifest = {
        "schema_version": LABEL_SCHEMA,
        "status": "complete" if complete else "profile",
        "purpose": "G2 training-only BDDL progress/rising and simulator contact labels",
        "predicate_slots": int(predicate_slots),
        "information_wall": {
            "deployment_forward_reads_labels": False,
            "fit_roles_with_gradients": ["meta_fit", "target_fit"],
            "held_roles": ["meta_held", "target_held"],
            "video_and_label_episodes_are_disjoint": True,
        },
        "temporal_contract": {
            "video_obs_index_i": "hdf5_state_i_plus_1_except_final",
            "rising_index_zero": "hdf5_state_0_to_state_1",
            "final_predicates": "all_true_from_successful_demo_contract",
            "final_contact": "masked_absent_post_action_state",
        },
        "tasks": records,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest
