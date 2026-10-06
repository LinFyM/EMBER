"""Train-only physical labels; privileged fields never enter the deployment G.

One hand and 32 anonymous entities share a task registry across episodes. Body
origins are MuJoCo body origins, and finger values are raw joint positions.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from ember.operator_writer.data import TASKS


SLOTS, SEMANTIC_DIM, SEMANTIC_SEED = 33, 64, 20261006
SCHEMA = "ember_relation_physical_labels_v1"
FIELDS = ("p", "R", "semantic", "presence", "joint_type", "joint_q", "hand_q",
          "valid", "p_mask", "R_mask", "semantic_mask", "joint_type_mask",
          "joint_q_mask", "hand_q_mask", "frame_indices")
# Registered public-name/archived-XML mismatch, never an asset/pose equivalence.
LEGACY_NAMES = {101: {"new_salad_dressing_1": "salad_dressing_1"}}


def project_semantics(embedding_means: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Project frozen-source name-token means; do not duplicate the embedding table."""
    means = {name: np.asarray(value, dtype=np.float64) for name, value in embedding_means.items()}
    widths = {value.shape for value in means.values()}
    if not means or len(widths) != 1:
        raise ValueError("semantic means must have one frozen embedding width")
    shape = widths.pop()
    if len(shape) != 1 or shape[0] < SEMANTIC_DIM:
        raise ValueError("source embedding must have at least 64 dimensions")
    gaussian = np.random.default_rng(SEMANTIC_SEED).standard_normal((shape[0], SEMANTIC_DIM))
    projection, upper = np.linalg.qr(gaussian, mode="reduced")
    projection *= np.where(np.diag(upper) < 0, -1., 1.)
    result = {}
    for name, value in means.items():
        vector = value @ projection
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 1e-12:
            raise ValueError(f"invalid frozen-source semantic mean: {name}")
        result[name] = (vector / norm).astype(np.float32)
    return result


def _public_objects(bddl_path: Path) -> list[tuple[str, str]]:
    text, result = bddl_path.read_text(), []
    for section in (":objects", ":fixtures"):
        block = re.search(r"\(" + re.escape(section) + r"\s+([^()]*)\)", text)
        if block is None:
            raise ValueError(f"missing public {section}: {bddl_path}")
        tokens, pending, index = block.group(1).split(), [], 0
        while index < len(tokens):
            token = tokens[index]
            if token == "-":
                if not pending or index + 1 >= len(tokens):
                    raise ValueError("invalid public object/category declaration")
                category = tokens[index + 1]
                arena = section == ":fixtures" and {"table", "floor"}.intersection(category.split("_"))
                if not arena:
                    result.extend((name, category) for name in pending)
                pending, index = [], index + 2
            else:
                pending.append(token)
                index += 1
        if pending:
            raise ValueError("public entity names require explicit categories")
    return result


def _semantic_name(value: str) -> str:
    if value.strip("_") == "microdoorroot":
        value = "door"
    words = re.sub(r"\d+", "", value).strip("_").replace("_", " ").split()
    if not words or any(word in {"scene", "path"} for word in words):
        raise ValueError(f"non-generic semantic label: {value}")
    return " ".join(words)


@dataclass(frozen=True)
class Entity:
    body: str
    semantic: str
    source_object: str
    origin: str
    joint: str | None


@dataclass(frozen=True)
class Binding:
    bodies: tuple[int, ...]
    joint_types: tuple[int, ...]  # none=0, slide=1, hinge=2
    qpos_addresses: tuple[int | None, ...]
    grip_site: int
    eef_body: int
    finger_addresses: tuple[int, int]


class Registry:
    """Labels/F-only identity map; anonymous slots keep whole-video correspondence."""

    def __init__(self, task: int, xml: str, bddl_path: Path) -> None:
        if task not in TASKS:
            raise ValueError("physical registry is restricted to the fixed nonheld36")
        self.task, self.bddl_path = int(task), Path(bddl_path)
        tree = ET.fromstring(xml)
        world = tree.find("worldbody")
        if world is None:
            raise ValueError("episode XML has no worldbody")
        parents = {child: parent for parent in world.iter() for child in parent}
        entries, self.legacy_name_bindings = [], []
        for name, category in _public_objects(self.bddl_path):
            actual = LEGACY_NAMES.get(task, {}).get(name, name)
            matches = [b for b in world.iter("body") if b.get("name") == actual
                       or b.get("name", "").startswith(actual + "_")]
            roots = [b for b in matches if parents.get(b) not in matches]
            if len(roots) != 1:
                raise ValueError(f"task{task}: public entity {name} has {len(roots)} body roots")
            root = roots[0]
            if actual != name:
                self.legacy_name_bindings.append({"public_instance": name, "public_category": category,
                    "actual_xml_body": root.get("name"), "actual_xml_instance": actual,
                    "actual_xml_assets": [element.get("file") for element in tree.findall("./asset/*")
                                          if actual in element.get("name", "") and element.get("file")],
                    "geometry": "actual compiled episode body; no asset or pose equivalence asserted"})
            for body in root.iter("body"):
                joints = body.findall("joint") + body.findall("freejoint")
                if len(joints) > 1 or any(j.get("type") == "ball" for j in joints):
                    raise ValueError(f"task{task}: unsupported multiple/ball joints on {body.get('name')}")
                internal = [j for j in joints if j.tag != "freejoint" and j.get("type", "hinge") != "free"]
                if body is not root and not internal:
                    continue
                if internal and not any(True for _ in body.iter("geom")):
                    raise ValueError(f"task{task}: articulated body has no visible geometry: {body.get('name')}")
                semantic = category if body is root else body.get("name", "")[len(actual):]
                entries.append(Entity(body.get("name", ""), _semantic_name(semantic), name,
                                      "root" if body is root else "moving", internal[0].get("name") if internal else None))
        if len({entry.body for entry in entries}) != len(entries) or len(entries) > SLOTS - 1:
            raise ValueError(f"task{task}: entity registry exceeds/duplicates 32-slot schema: {len(entries)}")
        self.entities, self._semantics = tuple(entries), None

    @property
    def semantic_names(self) -> tuple[str, ...]:
        return tuple(sorted({"hand", *(entry.semantic for entry in self.entities)}))

    def set_semantics(self, vectors: Mapping[str, np.ndarray]) -> None:
        values = np.stack([np.asarray(vectors[name], dtype=np.float32)
                           for name in ("hand", *(e.semantic for e in self.entities))])
        if (values.shape != (len(self.entities) + 1, SEMANTIC_DIM) or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=-1), 1., atol=1e-4)):
            raise ValueError("semantic labels require fixed 64-dimensional unit vectors")
        self._semantics = values

    def bind(self, model) -> Binding:
        """Resolve every episode's actual compiled IDs/addresses, never task offsets."""
        import mujoco
        model = getattr(model, "_model", model)
        names = {model.body(i).name: i for i in range(model.nbody)}
        bodies, kinds, addresses = [], [], []
        for entity in self.entities:
            candidates = {entity.body}
            old = LEGACY_NAMES.get(self.task, {}).get(entity.source_object)
            if old is not None:
                candidates.add(entity.source_object + entity.body[len(old):])
            matches = [names[name] for name in candidates if name in names]
            if len(matches) != 1:
                raise ValueError(f"task{self.task}: actual model body binding is ambiguous/missing: {entity.body}")
            body = matches[0]
            joints = np.flatnonzero(model.jnt_bodyid == body)
            internal = [int(j) for j in joints if int(model.jnt_type[j]) in (2, 3)]
            if len(joints) > 1 or any(int(model.jnt_type[j]) == 1 for j in joints):
                raise ValueError(f"task{self.task}: compiled multi/ball joint on {entity.body}")
            actual_name = model.joint(internal[0]).name if internal else None
            if actual_name != entity.joint:
                raise ValueError(f"task{self.task}: episode internal joint mapping changed: {entity.body}")
            bodies.append(int(body))
            kinds.append({int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
                          int(mujoco.mjtJoint.mjJNT_HINGE): 2}[int(model.jnt_type[internal[0]])] if internal else 0)
            addresses.append(int(model.jnt_qposadr[internal[0]]) if internal else None)
        grip = [i for i in range(model.nsite) if model.site(i).name.endswith("_grip_site")]
        hand = [i for i in range(model.nbody) if model.body(i).name.endswith("_right_hand")]
        fingers = [[i for i in range(model.njnt) if model.joint(i).name.endswith(f"finger_joint{side}")]
                   for side in (1, 2)]
        if len(grip) != 1 or len(hand) != 1 or any(len(ids) != 1 for ids in fingers):
            raise ValueError("one Panda hand/grip site/two raw finger joints are required")
        return Binding(tuple(bodies), tuple(kinds), tuple(addresses), grip[0], hand[0],
                       tuple(int(model.jnt_qposadr[ids[0]]) for ids in fingers))

    def description(self, model=None) -> dict:
        result = {"task": self.task, "bddl": str(self.bddl_path), "hand_slot": 0,
                  "entity_capacity": SLOTS - 1, "entity_count": len(self.entities),
                  "legacy_name_bindings": self.legacy_name_bindings,
                  "entities": [{"slot": i + 1, **asdict(e)} for i, e in enumerate(self.entities)]}
        if model is not None:
            raw = getattr(model, "_model", model)
            binding = self.bind(raw)
            result["compiled_binding"] = {**asdict(binding), "actual_body_names": [raw.body(i).name for i in binding.bodies]}
        return result

    def empty(self, frames: Sequence[int]) -> dict[str, np.ndarray]:
        if self._semantics is None:
            raise ValueError("frozen-source semantics must be supplied before label extraction")
        n, present = len(frames), len(self.entities) + 1
        out = {"p": np.zeros((n, SLOTS, 3), np.float32),
               "R": np.broadcast_to(np.eye(3, dtype=np.float32), (n, SLOTS, 3, 3)).copy(),
               "semantic": np.zeros((n, SLOTS, SEMANTIC_DIM), np.float32),
               "presence": np.zeros((n, SLOTS), bool),
               "joint_type": np.zeros((n, SLOTS, 3), np.float32),
               "joint_q": np.zeros((n, SLOTS, 2), np.float32),
               "hand_q": np.zeros((n, 2), np.float32), "valid": np.zeros(n, bool),
               "joint_q_mask": np.zeros((n, SLOTS, 2), bool),
               "hand_q_mask": np.zeros((n, 2), bool), "frame_indices": np.asarray(frames, np.int64)}
        for field in ("p", "R", "semantic", "joint_type"):
            out[field + "_mask"] = np.zeros((n, SLOTS), bool)
        out["presence"][:, :present] = True
        out["semantic"][:, :present] = self._semantics
        out["semantic_mask"][:, :present] = True
        return out

    def _fill(self, out: dict, index: int, data, binding: Binding, position, rotation, fingers) -> None:
        present = len(self.entities) + 1
        out["p"][index, 0], out["R"][index, 0] = position, rotation
        out["p"][index, 1:present] = data.xpos[list(binding.bodies)]
        out["R"][index, 1:present] = data.xmat[list(binding.bodies)].reshape(-1, 3, 3)
        out["p_mask"][index, :present] = out["R_mask"][index, :present] = True
        out["joint_type"][index, 0, 0] = 1
        for slot, (kind, address) in enumerate(zip(binding.joint_types, binding.qpos_addresses, strict=True), 1):
            out["joint_type"][index, slot, kind] = 1
            if kind:
                out["joint_q"][index, slot, kind - 1] = data.qpos[address]
                out["joint_q_mask"][index, slot, kind - 1] = True
        out["joint_type_mask"][index, :present] = True
        out["hand_q"][index], out["hand_q_mask"][index] = fingers, True
        out["valid"][index] = True

    def runtime(self, sim, observation: Mapping) -> dict[str, np.ndarray]:
        """Privileged nonheld F diagnostic only; reads current, never future state."""
        from scipy.spatial.transform import Rotation
        model, data = getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)
        binding = self.bind(model)
        position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        rotation = Rotation.from_quat(observation["robot0_eef_quat"]).as_matrix()
        fingers = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
        if position.shape != (3,) or fingers.shape != (2,):
            raise ValueError("official runtime EEF/finger observation contract changed")
        _check_sync(data, binding, position, rotation, fingers)
        out = self.empty([-1])
        self._fill(out, 0, data, binding, position, rotation, fingers)
        return {key: value[0] for key, value in out.items()}


def compile_episode(xml: str, assets_root: Path, robosuite_root: Path):
    """Compile actual episode XML with local read-only asset mappings, no environment."""
    import mujoco
    tree = ET.fromstring(xml)
    for element in tree.findall("./asset/*"):
        name = element.get("file")
        if name is None:
            continue
        if "/robosuite/" in name and "/models/assets/" in name:
            path = robosuite_root / "models/assets" / name.rsplit("/models/assets/", 1)[-1]
        elif "/assets/" in name:
            path = assets_root / name.rsplit("/assets/", 1)[-1]
        else:
            raise ValueError(f"unmapped episode asset: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        element.set("file", str(path))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))


def _check_sync(data, binding: Binding, position, rotation, fingers) -> tuple[float, float, float]:
    errors = (float(np.linalg.norm(data.site_xpos[binding.grip_site] - position)),
              float(np.linalg.norm(data.xmat[binding.eef_body].reshape(3, 3) - rotation)),
              float(np.max(np.abs(data.qpos[list(binding.finger_addresses)] - fingers))))
    # The validated recovery contract synchronizes grip p and eef-body R.
    # Raw obs gripper_states is itself the hand-q label; reconstructed fingers
    # are a diagnostic, not a replacement or an extra timing acceptance gate.
    if not np.isfinite(errors).all() or errors[0] > 1e-4 or errors[1] > 1e-3:
        raise ValueError(f"observation/full-state timing mismatch (p_m,R_fro,finger_qpos): {errors}")
    return errors


def extract_episode(demo: h5py.Group, registry: Registry, frame_indices: Sequence[int],
                    assets_root: Path, robosuite_root: Path) -> tuple[dict, dict]:
    """states[i+1] minus one integration substep aligns labels with obs/RGB[i]."""
    import mujoco
    from scipy.spatial.transform import Rotation
    frames = np.asarray(frame_indices, dtype=np.int64)
    count = int(demo["obs/ee_pos"].shape[0])
    states = demo["states"]
    model = compile_episode(demo.attrs["model_file"], assets_root, robosuite_root)
    if (frames.ndim != 1 or len(frames) == 0 or len(np.unique(frames)) != len(frames)
            or np.any(frames < 0) or np.any(frames >= count) or states.shape != (count, 1 + model.nq + model.nv)
            or model.na or any(demo[f"obs/{name}"].shape[0] != count for name in
                               ("ee_ori", "gripper_states", "agentview_rgb", "eye_in_hand_rgb"))):
        raise ValueError("requested frames or stored full-state contract changed")
    out, binding, data, errors = registry.empty(frames), registry.bind(model), mujoco.MjData(model), []
    for slot, kind in enumerate((0, *binding.joint_types)):
        out["joint_type"][:, slot, kind] = 1
        out["joint_type_mask"][:, slot] = True
    for index, frame in enumerate(frames):
        if frame == count - 1:
            continue  # Real RGB remains; unavailable terminal geometry stays masked.
        state = np.asarray(states[int(frame) + 1], dtype=np.float64)
        data.time = state[0] - model.opt.timestep
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
        mujoco.mj_forward(model, data)
        position = np.asarray(demo["obs/ee_pos"][frame], dtype=np.float64)
        rotation = Rotation.from_rotvec(np.asarray(demo["obs/ee_ori"][frame], dtype=np.float64)).as_matrix()
        fingers = np.asarray(demo["obs/gripper_states"][frame], dtype=np.float64)
        errors.append(_check_sync(data, binding, position, rotation, fingers))
        registry._fill(out, index, data, binding, position, rotation, fingers)
    if any(not np.isfinite(out[key]).all() for key in ("p", "R", "joint_q", "hand_q")):
        raise ValueError("nonfinite physical labels")
    summary = {"frames": frames.tolist(), "valid_frames": int(out["valid"].sum()),
               "max_sync_errors": np.max(errors, axis=0).tolist() if errors else [0., 0., 0.],
               "hand_q_source": "same-frame obs/gripper_states raw two qpos; rewind difference diagnostic only",
               "timestep_seconds": float(model.opt.timestep), "registry": registry.description(model)}
    return out, summary


def teaching_frames(count: int) -> np.ndarray:
    indices = list(range(0, count, 5))
    if not indices or count <= 0:
        raise ValueError("teaching video must be nonempty")
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return np.asarray(indices, np.int64)


class LabelStore:
    """Lazy train-only teacher/query store; caches only requested physical points."""

    def __init__(self, data, asset_root: Path, *, semantic_vectors=None, cache_root: Path | None = None) -> None:
        from ember.task_protocol import load_task_authorities
        if set(data.tasks) != set(TASKS) or data.role != "train":
            raise ValueError("physical-label store requires the registered nonheld36")
        self.data, self.asset_root = data, Path(asset_root)
        authority = json.loads((self.asset_root / "configs/pi05_writer_data_v1.json").read_text())["authorities"]
        self.assets_root = (self.asset_root / authority["libero_assets"]).resolve()
        self.robosuite_root = Path(importlib.util.find_spec("robosuite").origin).parent
        bddl_root = Path(importlib.util.find_spec("libero").origin).parent / "libero/bddl_files"
        _, manifest = load_task_authorities(self.asset_root, data.spec["source"]["data_protocol"])
        metadata = {row["global_task_id"]: row for row in manifest["tasks"]}
        self.cache_root = None if cache_root is None else Path(cache_root)
        if self.cache_root is not None and not self.cache_root.resolve().is_relative_to(Path("/data1/user/ymdai")):
            raise ValueError("new physical-label caches must remain under data1")
        self.registries, self._episodes, self._plan = {}, OrderedDict(), None
        for task in TASKS:
            row = metadata[task]
            if row["split_role"] != "train":
                raise ValueError("held physical-label metadata crossed allowlist")
            with h5py.File(data.tasks[task].authority.path, "r") as handle:
                bddl = bddl_root / row["suite"] / Path(handle["data"].attrs["bddl_file_name"]).name
                self.registries[task] = Registry(task, handle["data/demo_0"].attrs["model_file"], bddl)
        if semantic_vectors is not None:
            self.set_semantics(semantic_vectors)

    @property
    def semantic_names(self) -> tuple[str, ...]:
        return tuple(sorted({name for registry in self.registries.values() for name in registry.semantic_names}))

    def set_semantics(self, vectors: Mapping[str, np.ndarray]) -> None:
        if self._episodes:
            raise ValueError("semantic labels cannot change after episode extraction")
        for registry in self.registries.values():
            registry.set_semantics(vectors)

    def register_events(self, events: Iterable[dict]) -> dict:
        """Freeze union of stride5 teaching and the preregistered query frames."""
        plan = {(task, demo): set(teaching_frames(length)) for task in TASKS
                for demo, length in enumerate(self.data.tasks[task].episode_lengths)}
        conditions, query_count = set(), 0
        for event in events:
            task, teacher = int(event["task"]), int(event["teacher_demo"])
            if task not in self.registries or event["query_offset"] != 1:
                raise ValueError("event crossed physical-label task/query-offset authority")
            conditions.add((task, teacher))
            for query in event["queries"]:
                demo, frame = int(query["demo"]), int(query["frame"])
                if demo == teacher or (task, demo) not in plan or not 0 <= frame < self.data.tasks[task].episode_lengths[demo] - 1:
                    raise ValueError("query must be an original nonterminal cross-episode point")
                plan[task, demo].add(frame)
                query_count += 1
        if len(conditions) != 1800 or query_count != 50400:
            raise ValueError("physical-label plan requires the fixed first450 events")
        self._plan = {key: np.asarray(sorted(frames), np.int64) for key, frames in plan.items()}
        return {"schema": SCHEMA, "conditions": len(conditions), "queries": query_count,
                "episodes": len(plan), "unique_physical_points": sum(map(len, self._plan.values()))}

    def _episode(self, task: int, demo: int, requested: np.ndarray) -> dict:
        if task not in self.registries or demo not in range(50):
            raise ValueError("physical-label request crossed fixed task/episode authority")
        key = (task, demo)
        frames = self._plan[key] if self._plan is not None else np.unique(requested)
        if np.any(~np.isin(requested, frames)):
            raise ValueError("physical-label request is outside preregistered teaching/query points")
        if key in self._episodes:
            self._episodes.move_to_end(key)
            old = self._episodes[key]
            if np.all(np.isin(requested, old["frame_indices"])):
                return old
            frames = np.union1d(frames, old["frame_indices"])
        path = self.cache_root / f"task{task}_demo{demo}.npz" if self.cache_root else None
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                out = {name: cached[name] for name in FIELDS}
            registered_match = self._plan is None or np.array_equal(out["frame_indices"], frames)
            if not registered_match or np.any(~np.isin(requested, out["frame_indices"])):
                raise ValueError("precomputed label cache frame plan changed")
        else:
            with h5py.File(self.data.tasks[task].authority.path, "r") as handle:
                out, summary = extract_episode(handle[f"data/demo_{demo}"], self.registries[task], frames,
                                               self.assets_root, self.robosuite_root)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **out)
                path.with_suffix(".json").write_text(json.dumps({**self.provenance(task, demo), **summary}, indent=2) + "\n")
        self._episodes[key] = out
        while len(self._episodes) > 2:
            self._episodes.popitem(last=False)
        return out

    def teacher(self, task: int, demo: int, frame_indices=None) -> dict[str, np.ndarray]:
        if task not in self.registries or demo not in range(50):
            raise ValueError("teacher physical labels are nonheld36 only")
        frames = teaching_frames(self.data.tasks[task].episode_lengths[demo]) if frame_indices is None else np.asarray(frame_indices, np.int64)
        out = self._episode(task, demo, frames)
        positions = {int(frame): i for i, frame in enumerate(out["frame_indices"])}
        order = [positions[int(frame)] for frame in frames]
        return {name: value[order] for name, value in out.items()}

    def queries(self, task: int, queries: Sequence[dict]) -> dict[str, np.ndarray]:
        if not queries:
            raise ValueError("physical query batch must be nonempty")
        rows = [self.teacher(task, int(query["demo"]), [int(query["frame"])]) for query in queries]
        if any(not row["valid"][0] for row in rows):
            raise ValueError("query physical labels require the original nonterminal frames")
        return {name: np.concatenate([row[name] for row in rows]) for name in FIELDS}

    def provenance(self, task: int, demo: int) -> dict:
        if task not in self.registries or demo not in range(50):
            raise ValueError("label provenance crossed fixed task/episode authority")
        return {"schema": SCHEMA, "task": task, "demo": demo,
                "source_hdf5": str(self.data.tasks[task].authority.path),
                "state_alignment": "states[i+1]; mj_integratePos(-timestep); mj_forward",
                "hand_position": "obs/ee_pos (grip site)", "hand_rotation": "obs/ee_ori (eef body axis-angle)",
                "semantic_projection_seed": SEMANTIC_SEED, "registry": self.registries[task].description()}

    def query_provenance(self, task: int, queries: Sequence[dict]) -> dict:
        return {"schema": SCHEMA, "task": task, "queries": [dict(row) for row in queries],
                "source_hdf5": str(self.data.tasks[task].authority.path), "query_action_offset": 1,
                "registry": self.registries[task].description()}

    def close(self) -> None:
        self._episodes.clear()


def build_cache(store: LabelStore, events: Iterable[dict], *, episodes=None) -> dict:
    """CPU entry called by the thin launch owner after clean pushed freezing."""
    if store.cache_root is None:
        raise ValueError("physical-label precompute requires a retained data1 cache root")
    plan = store.register_events(events)
    keys = list(store._plan) if episodes is None else list(episodes)
    for task, demo in keys:
        if (task, demo) not in store._plan:
            raise ValueError("cache worker episode crossed the registered label plan")
        store._episode(task, demo, store._plan[task, demo])
    return {**plan, "worker_episodes": len(keys),
            "worker_points": sum(len(store._plan[key]) for key in keys)}


def validate_sync(store: LabelStore) -> dict:
    """Bounded actual first/middle/last-valid synchronization on all36 demo0."""
    rows = []
    for task in TASKS:
        count = store.data.tasks[task].episode_lengths[0]
        frames = np.unique([0, (count - 2) // 2, count - 2])
        with h5py.File(store.data.tasks[task].authority.path, "r") as handle:
            _, summary = extract_episode(handle["data/demo_0"], store.registries[task], frames,
                                         store.assets_root, store.robosuite_root)
        rows.append({"task": task, "demo": 0, **summary})
    return {"tasks": len(rows), "points": sum(len(row["frames"]) for row in rows), "rows": rows}
