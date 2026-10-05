"""Temporary fixed-panel consumers for the registered frozen-T reading study."""
from __future__ import annotations

import atexit
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from safetensors.torch import load_file, save_file

from ember.lora import LORA_B_SUFFIX, expected_lora_state_shapes
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record

STUDY = "query_conditioned_transition_read_20261005"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
REPO = Path(__file__).resolve().parents[3]
PARENT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340")
BASE_BANK = PARENT.parents[4] / "banks/2340/manifest.json"
BASE_RESULTS = PARENT.parents[4] / "evaluation/2340/correct400/results.json"
SEEN_ROOT = Path("/data1/user/ymdai/ember_runs/operator_seen_task_diagnosis_20260929/attempts/scene_canonical144")
SEEN_BANK = SEEN_ROOT / "T/banks/1800/manifest.json"
SEEN_RESULTS = SEEN_ROOT / "T/evaluation/correct144/results.json"
TRAIN_IDS, VALIDATION_IDS = (29, 34, 73), (3, 6, 11, 16, 23, 26, 31, 39)
PANEL_SCHEMA = "ember_transition_read_panel_v1"


def paths(arm, panel):
    if arm not in ("L", "R", "parent") or panel not in ("validation", "seen12") or (arm == "parent" and panel != "seen12"):
        raise ValueError("transition study has only L/R400 and parent/L/R seen12")
    root = ROOT / arm
    return root / "banks" / panel / "manifest.json", root / "evaluation" / panel


def source_geometry(panel):
    if panel not in ("validation", "seen12"):
        raise ValueError("unregistered transition panel")
    original = read_json(BASE_BANK if panel == "validation" else SEEN_BANK)
    ids = VALIDATION_IDS if panel == "validation" else TRAIN_IDS
    tasks = [deepcopy(row) for row in original["tasks"] if row["global_task_id"] in ids]
    states = tuple(range(50)) if panel == "validation" else (32, 33, 34, 35)
    if tuple(row["global_task_id"] for row in tasks) != ids:
        raise ValueError("original transition task allowlist changed")
    for task in tasks:
        episodes = task["episodes"]
        if tuple(row["init_state_id"] for row in episodes) != states or len({row["teacher_demo_indices"][0] for row in episodes}) != len(states):
            raise ValueError("original no-replacement teacher/state pairing changed")
    by_id = {row["condition_id"]: row for row in original["conditions"]}
    conditions = [{key: by_id[episode["condition_id"]][key] for key in
                   ("condition_id", "global_task_id", "teacher_demo", "raw_frames", "sampled_frames")}
                  for task in tasks for episode in task["episodes"]]
    for condition in conditions:
        raw = condition["raw_frames"]
        grid = list(range(0, raw, 5))
        if grid[-1] != raw - 1:
            grid.append(raw - 1)
        if len(grid) != condition["sampled_frames"] or len(grid) < 2:
            raise ValueError("original stride5-plus-last teacher frame grid changed")
        condition["frame_indices"] = grid
    return original, tasks, conditions, states


def checkpoint_identity(checkpoint, arm):
    checkpoint = Path(checkpoint).resolve()
    if arm == "parent":
        if checkpoint != PARENT:
            raise ValueError("parent reference must use T2340")
        return {"study": STUDY, "arm": arm, "endpoint": 2340, "parent": str(PARENT)}
    if arm not in ("L", "R") or not checkpoint.is_relative_to(ROOT / arm) or checkpoint.name != "macro_00000270" or checkpoint.parent.name != "checkpoints":
        raise ValueError("only this arm's complete endpoint270 is eligible")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    world = manifest.get("world_size", 0)
    wanted = {"ecp.safetensors", "trainer_state.pt"} | {f"rank_{rank:02d}_state.pt" for rank in range(world)}
    if (manifest.get("schema_version") != "ember_ecp_checkpoint_v1" or manifest.get("stage") != STUDY
            or manifest.get("run_contract_schema") != "ember_query_conditioned_transition_read_run_v1"
            or manifest.get("next_macro") != 270 or not 1 <= world <= 4 or set(manifest.get("files", {})) != wanted):
        raise ValueError("transition endpoint is not a full ECP270")
    for name, record in manifest["files"].items():
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size != record["bytes"]:
            raise ValueError("transition ECP file is incomplete")
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    from .bank import source_matches
    if ((run.get("study"), run.get("arm"), run.get("parent")) != (STUDY, arm, str(PARENT))
            or run.get("phase") != "train" or not source_matches(run.get("source", {}), read_json(BASE_BANK)["source"])):
        raise ValueError("transition endpoint study/arm/parent provenance changed")
    return {"study": STUDY, "arm": arm, "endpoint": 270, "parent": str(PARENT),
            "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
            "training_contract": file_record(checkpoint.parent.parent / "run_contract.json")}


def panel_identity(arm, panel, checkpoint):
    original, _tasks, _conditions, _states = source_geometry(panel)
    return {"schema_version": PANEL_SCHEMA, **checkpoint_identity(checkpoint, arm),
            "panel": panel, "teacher_mapping": file_record(BASE_BANK if panel == "validation" else SEEN_BANK),
            "pairing_source_results": file_record(BASE_RESULTS if panel == "validation" else SEEN_RESULTS),
            "scene": file_record(Path(original["scene_root"]) / "manifest.json")}


def lora_contract(base):
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
    spec = read_json(Path(base["spec"]["path"]))
    return derive_pi05_lora_rank(load_pi05_lora_contract(Path(base["asset_root"]) / spec["source"]["lora_contract"]), rank=128)


def factor_path(arm, panel, key):
    return paths(arm, panel)[0].parent / f"{key}.safetensors"


def parent_factor(panel, key):
    if panel == "seen12":
        return factor_path("parent", panel, key)
    return Path(next(row for row in read_json(BASE_BANK)["conditions"] if row["condition_id"] == key)["factors"]["path"])


def capture_manifest(arm, panel, tasks):
    full_state = 0 if panel == "validation" else 32
    return {"schema_version": "ember_pi05_registered_trajectory_capture_v1", "study_id": STUDY,
            "task_subset_selection": None, "mode": "compact",
            "full_conditions": [{"suite": row["suite"], "task_id": row["task_id"], "init_state_id": full_state} for row in tasks],
            "passive_control_trace": "ember_operator_read_write_passive_capture_v1", "stage_predicates": True,
            "training_gradient_use": False, "checkpoint_selection_use": False, "validation_use": False, "test_use": False}


def factor_metadata(arm, key, checkpoint):
    return {"study": STUDY, "arm": arm, "condition_id": key, "checkpoint": str(Path(checkpoint).resolve())}


def information_wall(arm):
    return {"teacher_condition": "exact_language_action_hidden_dual_RGB_stride5_last_frame",
            "teacher_runtime_frames": 0, "runtime_reader": arm == "R", "deployment_candidate": arm != "R",
            "validation_test_gradients": False}


def register_bank(arm, panel, checkpoint, *, execution=None, frame_chunk=8):
    from .bank import BANK_SCHEMA, KIND, _factor_header
    from .run import frozen_git
    base = read_json(BASE_BANK)
    original, tasks, conditions, _states = source_geometry(panel)
    identity = panel_identity(arm, panel, checkpoint)
    shape = {name: size for name, size in expected_lora_state_shapes(lora_contract(base)).items() if name.endswith(LORA_B_SUFFIX)}
    for condition in conditions:
        if arm != "R":
            path = factor_path(arm, panel, condition["condition_id"])
            _factor_header(path, shape, metadata=factor_metadata(arm, condition["condition_id"], checkpoint))
            condition["factors"] = file_record(path)
        if arm == "L":
            condition["parent_factors"] = file_record(parent_factor(panel, condition["condition_id"]))
    manifest, _output = paths(arm, panel)
    bank = {"schema_version": BANK_SCHEMA, "kind": KIND, "mode": arm, "transition_read_panel": identity,
            "asset_root": base["asset_root"], "spec": base["spec"], "source": base["source"], "lora": base["lora"],
            "shared": base["shared"], "checkpoint": str(Path(checkpoint).resolve()),
            "native_frame_chunk": frame_chunk,
            "scene_root": original["scene_root"], "tasks": tasks, "conditions": conditions,
            "materialization_git": frozen_git(continuation=True),
            "information_wall": information_wall(arm)}
    if not isinstance(frame_chunk, int) or frame_chunk < 1:
        raise ValueError("native physical frame chunk must be positive")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and read_json(manifest) != bank:
        raise ValueError("published transition panel is immutable")
    write_json_atomic(manifest, bank)
    write_json_atomic(manifest.parent / "capture.json", capture_manifest(arm, panel, tasks))
    if execution is not None:
        write_json_atomic(manifest.parent / "materialization_execution.json", execution)
    return manifest


def inspect(bank, path, source, task_keys, role, require_formal, task_states):
    from .bank import BANK_SCHEMA, EVAL_SCHEMA, KIND, _factor_header, source_matches
    from ember.pi05_eval.scene import inspect_registered_scenes
    arm, panel = bank["mode"], bank["transition_read_panel"]["panel"]
    expected_path, _output = paths(arm, panel)
    base = read_json(BASE_BANK)
    original, tasks, conditions, states = source_geometry(panel)
    expected_role = "validation" if panel == "validation" else "operator_seen_training36"
    facts = ((Path(path), expected_path.resolve()), (bank["schema_version"], BANK_SCHEMA), (bank["kind"], KIND),
        (bank["transition_read_panel"], panel_identity(arm, panel, bank["checkpoint"])),
        (require_formal, True), (role, expected_role), (bank["tasks"], tasks),
        (set(task_keys), {(row["suite"], row["task_id"]) for row in tasks}),
        ({key: tuple(value) for key, value in (task_states or {}).items()}, {key: states for key in task_keys}),
        (source_matches(bank["source"], source), True), (bank["source"], source),
        ({key: bank[key] for key in ("asset_root", "spec", "shared", "lora")},
         {key: base[key] for key in ("asset_root", "spec", "shared", "lora")}),
        (bank["shared"], file_record(Path(bank["shared"]["path"]))),
        (bank["information_wall"], information_wall(arm)), (bank["scene_root"], original["scene_root"]),
        (bank["materialization_git"].get("branch"), ""), (bank["materialization_git"].get("dirty_paths"), []),
        (bank["materialization_git"].get("pushed_ref"), "origin/main"))
    if any(actual != wanted for actual, wanted in facts) or not isinstance(bank["native_frame_chunk"], int) or bank["native_frame_chunk"] < 1:
        raise ValueError("transition source, endpoint or exact panel changed")
    shape = {name: size for name, size in expected_lora_state_shapes(lora_contract(base)).items() if name.endswith(LORA_B_SUFFIX)}
    for row, wanted in zip(bank["conditions"], conditions, strict=True):
        if arm != "R":
            path_value = factor_path(arm, panel, wanted["condition_id"])
            wanted["factors"] = file_record(path_value)
            _factor_header(path_value, shape, metadata=factor_metadata(arm, wanted["condition_id"], bank["checkpoint"]))
        if arm == "L":
            wanted["parent_factors"] = file_record(parent_factor(panel, wanted["condition_id"]))
        if row != wanted:
            raise ValueError("transition teacher/factor condition changed")
    inspect_registered_scenes(Path(original["scene_root"]), original["tasks"], states=states,
                              schema="ember_demonstration_formal_scenes_v1" if panel == "validation" else "ember_operator_seen_task_scenes_v1")
    return {**bank, "schema_version": EVAL_SCHEMA, "arm": "correct", "manifest": file_record(expected_path),
            "scene_manifest": bank["transition_read_panel"]["scene"], "output_dir": str(_output)}


def select_tasks(args, tasks):
    bank = read_json(Path(args.static_task_lora_manifest))
    panel = bank["transition_read_panel"]["panel"]
    _original, wanted, _conditions, states = source_geometry(panel)
    role = "validation" if panel == "validation" else "operator_seen_training36"
    if args.role != role or args.mode != "formal" or args.state_count != len(states) or any(getattr(args, key, None) for key in ("task_subset_selection", "occupancy_capture_selection", "exploration_sigma")):
        raise ValueError("transition selection is only its fixed registered panel")
    keys = {(row["suite"], row["task_id"]) for row in wanted}
    selected = tuple(task for task in tasks if (task.suite, task.task_id) in keys)
    if len(selected) != len(wanted) or any(tuple(task.init_state_ids) != states for task in selected):
        raise ValueError("installed transition panel changed")
    return selected


def registered_capture(args, tasks, output, path, manifest, subset, bank):
    from .capture import PASSIVE_TAG
    arm, panel = bank["mode"], bank["transition_read_panel"]["panel"]
    bank_path, expected_output = paths(arm, panel)
    _original, wanted, _conditions, states = source_geometry(panel)
    expected = capture_manifest(arm, panel, wanted)
    if (manifest != expected or Path(path) != bank_path.parent / "capture.json" or Path(output) != expected_output
            or Path(args.static_task_lora_manifest).resolve() != bank_path or subset is not None
            or args.mode != "formal" or args.role != ("validation" if panel == "validation" else "operator_seen_training36")
            or {(t.suite, t.task_id) for t in tasks} != {(r["suite"], r["task_id"]) for r in wanted}
            or any(tuple(t.init_state_ids) != states for t in tasks)):
        raise ValueError("transition25-full/811-compact capture changed")
    capture = {"schema_version": expected["schema_version"], "selection_path": str(path), "selection_bytes": Path(path).stat().st_size,
               "mode": "compact", "full_conditions": expected["full_conditions"], "trajectory_root": str(output / "trajectories"),
               "passive_trace": {"schema_version": PASSIVE_TAG, "trace_root": str(output / "continuous_traces")},
               "training_gradient_use": False, "checkpoint_selection_use": False, "validation_use": False, "test_use": False}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1", "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False,
             "training_gradient_use": False, "checkpoint_selection_use": False, "validation_action_reads": 0,
             "validation_reward_reads": 0, "held_data_use": False, "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def load_readout(checkpoint, arm, lora, device):
    from .transition_read import TransitionReadout
    checkpoint_identity(checkpoint, arm)
    result = TransitionReadout(lora).to(device)
    result.load_state_dict(load_file(str(Path(checkpoint) / "ecp.safetensors"), device=str(device)), strict=True)
    return result.requires_grad_(False).eval()


class TransitionCompiler:
    def __init__(self, asset_root, config, device, cpu_threads):
        from .data import FormalData
        from .run import build_runtime
        from ember.writer.materialization_workers import _configure_device
        _configure_device(device, cpu_threads)
        base = read_json(BASE_BANK)
        if Path(asset_root).resolve() != Path(base["asset_root"]).resolve():
            raise ValueError("transition compilation must reuse the canonical assets")
        spec = read_json(Path(base["spec"]["path"]))
        self.runtime = build_runtime(asset_root, spec, device, "T")
        self.runtime.writer.load_state_dict(load_file(str(PARENT / "ecp.safetensors"), device=str(device)), strict=True)
        self.runtime.writer.requires_grad_(False).eval()
        self.config = config
        ids = VALIDATION_IDS if config["panel"] == "validation" else TRAIN_IDS
        self.data = FormalData(asset_root, spec, query_labels=False, task_ids=ids, role="validation" if config["panel"] == "validation" else "train")
        self.readout = load_readout(config["checkpoint"], "L", self.runtime.lora, device) if config["arm"] == "L" else None

    def prepare(self, request):
        if request != self.config:
            raise ValueError("transition compiler source changed")

    def compile(self, condition):
        from .transition_read import read_frozen_memory
        started = time.monotonic()
        memory = read_frozen_memory(self.runtime, self.data, condition["global_task_id"], condition["teacher_demo"], frame_chunk=self.config["frame_chunk"])
        frames = memory.frame_indices.cpu().tolist()
        if (memory.raw_frames, memory.sampled_frames, frames) != (condition["raw_frames"], condition["sampled_frames"], condition["frame_indices"]):
            raise ValueError("actual teacher frames differ before B publication")
        with torch.no_grad():
            state = self.readout.linear(memory) if self.readout is not None else memory.state
        factors = {name: value.detach().float().cpu().contiguous() for name, value in state.items() if name.endswith(LORA_B_SUFFIX)}
        path = factor_path(self.config["arm"], self.config["panel"], condition["condition_id"])
        temporary = path.with_suffix(".partial.safetensors")
        save_file(factors, str(temporary), metadata=factor_metadata(self.config["arm"], condition["condition_id"], self.config["checkpoint"]))
        temporary.replace(path)
        return {"condition_id": condition["condition_id"], "raw_frames": memory.raw_frames,
                "sampled_frames": memory.sampled_frames, "frame_indices": frames,
                "compile_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.runtime.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.runtime.device)}

    def close(self):
        self.data.close()


def materialize(arm, panel, checkpoint, *, asset_root, devices, frame_chunk=8, cpu_threads=4):
    from .bank import _factor_header
    from .run import frozen_git
    from ember.writer.materialization_workers import MaterializationWorkers, execution_devices
    if arm not in ("L", "parent") or frame_chunk < 1:
        raise ValueError("only L/parent have complete materialized B")
    panel_identity(arm, panel, checkpoint)
    git = frozen_git(continuation=True)
    bank_path, _output = paths(arm, panel)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    _original, _tasks, conditions, _states = source_geometry(panel)
    shape = {name: size for name, size in expected_lora_state_shapes(lora_contract(read_json(BASE_BANK))).items() if name.endswith(LORA_B_SUFFIX)}
    pending = []
    for condition in conditions:
        path = factor_path(arm, panel, condition["condition_id"])
        if path.exists():
            _factor_header(path, shape, metadata=factor_metadata(arm, condition["condition_id"], checkpoint))
        else:
            pending.append(condition)
        if arm == "L":
            file_record(parent_factor(panel, condition["condition_id"]))
    pending.sort(key=lambda row: row["sampled_frames"], reverse=True)
    config = {"arm": arm, "panel": panel, "checkpoint": str(Path(checkpoint).resolve()), "frame_chunk": frame_chunk}
    records = []
    if pending:
        with MaterializationWorkers(asset_root=asset_root, config=config, devices=execution_devices(devices=devices), cpu_threads=cpu_threads, compiler_factory=TransitionCompiler) as workers:
            for job, result in workers.compile(config, pending):
                if any(result[key] != job[key] for key in ("raw_frames", "sampled_frames", "frame_indices")):
                    raise ValueError("actual original teacher frame grid changed")
                records.append(result)
    return register_bank(arm, panel, checkpoint, frame_chunk=frame_chunk, execution={"git": git, "devices": list(devices), "frame_chunk": frame_chunk, "compiled": records, "reused": len(conditions) - len(pending)})


@dataclass
class PreparedTransition:
    key: str
    evidence: dict
    memory: object = None
    stats: dict = field(default_factory=dict)
    teacher: dict = field(default_factory=dict)
    replans: int = 0


class _EffectWriter:
    def _start_effects(self):
        self.effect_rows = {}
        self.effects_saved = False
        self.effect_closed = False
        atexit.register(self.close)

    def _effect(self, prepared):
        entry = self.effect_rows.setdefault(prepared.key, {"stats": prepared.stats, "evidence": prepared.evidence,
            "teacher": prepared.teacher, "first_replan": 0, "last_replan": -1, "replans": 0, "completed_predictions": 0})
        entry["last_replan"] = prepared.replans
        prepared.replans += 1
        entry["replans"] = prepared.replans

    def _effect_completed(self, prepared):
        for item in prepared:
            self.effect_rows[item.key]["completed_predictions"] += 1

    def _save_effects(self):
        from .transition_read import summarize_stats
        if self.effects_saved:
            return
        root = Path(self.bank["output_dir"]) / "transition_effects"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"worker_{os.getpid()}_{os.environ.get('EMBER_PI05_EVAL_INVOCATION_ID', 'unleased')}.jsonl"
        with path.open("x") as handle:
            for key, entry in sorted(self.effect_rows.items()):
                effect = summarize_stats(entry["stats"])
                invalid = 0
                for row in effect["rows"]:
                    for name, value in row.items():
                        if isinstance(value, float) and not math.isfinite(value):
                            row[name] = None
                            invalid += 1
                effect["nonfinite_statistics"] = invalid
                calls = {}
                for row in effect["rows"]:
                    flow = str(row["flow"])
                    calls[flow] = calls.get(flow, 0) + row["calls"]
                value = {"study": STUDY, "arm": self.bank["mode"], "condition_id": key,
                         "evidence": entry["evidence"], "first_replan": entry["first_replan"], "last_replan": entry["last_replan"],
                         "replans": entry["replans"], "teacher": entry["teacher"],
                         "completed_predictions": entry["completed_predictions"],
                         "worker_pid": os.getpid(), "invocation_id": os.environ.get("EMBER_PI05_EVAL_INVOCATION_ID", "unleased"),
                         "flow_target_calls": calls, "effect": effect}
                handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self.effects_saved = True

    def _teacher(self, condition, frame_indices):
        if frame_indices != condition["frame_indices"]:
            raise ValueError("effect frame positions differ from original stride5-plus-last")
        return {"global_task_id": condition["global_task_id"], "demo_index": condition["teacher_demo"],
                "mapping": self.bank["transition_read_panel"]["teacher_mapping"], "frame_indices": frame_indices,
                "raw_frames": condition["raw_frames"], "sampled_frames": condition["sampled_frames"],
                "real_transitions": len(frame_indices) - 1, "slots_per_transition": 50,
                "frame_axis": "sampled_transition_start_index", "runtime_reader": self.bank["mode"] == "R",
                "reader_padding": "zero_weight" if self.bank["mode"] == "R" else None}


def shared_runtime(policy, bank, device, lora, identity):
    from .model import OperatorReadWrite
    from .run import Runtime
    from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
    spec = read_json(Path(bank["spec"]["path"]))
    assets = Path(bank["asset_root"])
    tokenizer = assets / spec["source"]["tokenizer"]
    parent = OperatorReadWrite(lora, identity, "T").to(device)
    parent.load_state_dict(load_file(str(PARENT / "ecp.safetensors"), device=str(device)), strict=True)
    parent.requires_grad_(False).eval()
    processor = Pi05LiberoProcessor(read_json(assets / spec["source"]["normalization"])["stats"], tokenizer, 200, str(device))
    return Runtime(policy, parent, Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)), processor, lora, bank["source"], device, identity)


class FrozenTransitionReader(_EffectWriter):
    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        from .bank import EVAL_SCHEMA, KIND
        from .data import FormalData
        from .transition_read import ReaderHooks
        from ember.lora import inject_task_lora, identity_lora_state, task_lora_state_dict
        self.bank, self.policy = evaluation_adapter, policy
        if self.bank["mode"] != "R" or self.bank["kind"] != KIND or self.bank["schema_version"] != EVAL_SCHEMA or self.bank["source"] != source or not require_formal:
            raise ValueError("online reader requires its registered diagnostic consumer")
        self.tasks = {(row["suite"], row["task_id"]): row for row in self.bank["tasks"]}
        if set(self.tasks) != set(task_keys):
            raise ValueError("online reader task panel changed")
        self.lora = lora_contract(self.bank)
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        self.identity = identity_lora_state(self.lora)
        self.runtime = shared_runtime(policy, self.bank, device, self.lora, self.identity)
        self.readout = load_readout(self.bank["checkpoint"], "R", self.lora, device)
        panel = self.bank["transition_read_panel"]["panel"]
        spec = read_json(Path(self.bank["spec"]["path"]))
        self.data = FormalData(Path(self.bank["asset_root"]), spec, query_labels=False, task_ids=VALIDATION_IDS if panel == "validation" else TRAIN_IDS, role="validation" if panel == "validation" else "train")
        self.hooks = ReaderHooks(policy, self.lora, self.readout)
        self.max_inference_batch = 0
        self._start_effects()

    @torch.no_grad()
    def prepare_episode(self, *, suite, task_id, init_state_id):
        from .bank import episode_evidence
        from .transition_read import read_frozen_memory
        task = self.tasks[(suite, task_id)]
        episode = next(row for row in task["episodes"] if row["init_state_id"] == init_state_id)
        memory = read_frozen_memory(self.runtime, self.data, task["global_task_id"], episode["teacher_demo_indices"][0], frame_chunk=self.bank["native_frame_chunk"])
        condition = next(row for row in self.bank["conditions"] if row["condition_id"] == episode["condition_id"])
        if (memory.raw_frames, memory.sampled_frames) != (condition["raw_frames"], condition["sampled_frames"]):
            raise ValueError("runtime reader lost original teacher frame grid")
        memory.reader = self.readout.reader_memory(memory)
        memory.keys.clear()
        memory.phi.clear()
        teacher = self._teacher(condition, memory.frame_indices.cpu().tolist())
        return PreparedTransition(episode["condition_id"], episode_evidence(self.bank, task, episode), memory, teacher=teacher)

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if len(prepared) != len(noise) or num_steps != 10 or any(item.memory.reader is None for item in prepared):
            raise ValueError("online reader lost fixed paired conditions/ten-flow consumer")
        self.max_inference_batch = max(self.max_inference_batch, len(prepared))
        for item in prepared:
            self._effect(item)
        try:
            with self.hooks.activate([item.memory for item in prepared], stats=[item.stats for item in prepared]):
                result = self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
            self._effect_completed(prepared)
            return result
        except BaseException:
            self._save_effects()
            raise

    def close(self):
        if self.effect_closed:
            return
        try:
            self._save_effects()
        finally:
            self.hooks.close()
            self.data.close()
            self.effect_closed = True
            atexit.unregister(self.close)


def linear_adapter(**arguments):
    from .bank import FrozenOperatorAdapter
    from .transition_read import LinearEffectHooks

    class FrozenTransitionLinear(FrozenOperatorAdapter, _EffectWriter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.effects = LinearEffectHooks(self.policy, self.lora)
            self.parents = OrderedDict()
            self._start_effects()

        def prepare_episode(self, **kwargs):
            value = super().prepare_episode(**kwargs)
            condition = self.conditions[value.key]
            return PreparedTransition(value.key, value.evidence, teacher=self._teacher(condition, condition["frame_indices"]))

        def _parent(self, key):
            if key not in self.parents:
                if self.conditions[key]["parent_factors"] != file_record(Path(self.conditions[key]["parent_factors"]["path"])):
                    raise ValueError("passive parent factor changed during rollout")
                self.parents[key] = {**self.common, **load_file(self.conditions[key]["parent_factors"]["path"], device="cpu")}
            self.parents.move_to_end(key)
            while len(self.parents) > self.state_cache_capacity:
                self.parents.popitem(last=False)
            device = next(self.policy.parameters()).device
            return {name: value.to(device) for name, value in self.parents[key].items()}

        @torch.no_grad()
        def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
            self.state_cache_capacity = max(self.state_cache_capacity, len(prepared))
            device = next(self.policy.parameters()).device
            states = [{name: value.to(device) for name, value in self._state(item.key).items()} for item in prepared]
            parents = [self._parent(item.key) for item in prepared]
            for item in prepared:
                self._effect(item)
            try:
                with self.effects.activate(states, parents, stats=[item.stats for item in prepared]):
                    result = super().predict_action_chunk(prepared, batch, noise=noise, num_steps=num_steps)
                self._effect_completed(prepared)
                return result
            except BaseException:
                self._save_effects()
                raise

        def close(self):
            if self.effect_closed:
                return
            try:
                self._save_effects()
            finally:
                self.effects.close()
                super().close()
                self.parents.clear()
                self.effect_closed = True
                atexit.unregister(self.close)

    return FrozenTransitionLinear(**arguments)


def prepare_arguments(arm, panel, *, gpu_indices, replicas, envs_per_replica=None):
    bank_path, output = paths(arm, panel)
    base = read_json(BASE_BANK)
    spec = read_json(Path(base["spec"]["path"]))
    config = REPO / "configs/libero_24_8_8_coverage_v1/evaluation.json"
    if envs_per_replica is not None:
        if not isinstance(envs_per_replica, int) or envs_per_replica < 1:
            raise ValueError("physical environment batch must be positive")
        recipe = read_json(config)
        recipe["parallel"]["envs_per_replica"] = envs_per_replica
        config = ROOT / "launch/eval_configs" / f"envs_{envs_per_replica}.json"
        if config.exists() and read_json(config) != recipe:
            raise ValueError("published physical evaluation recipe changed")
        write_json_atomic(config, recipe)
    return SimpleNamespace(config=config,
        source_run=Path(base["source"]["source_run"]), checkpoint=Path(base["source"]["checkpoint"]),
        tokenizer_path=Path(base["asset_root"]) / spec["source"]["tokenizer"], output_dir=output,
        role="validation" if panel == "validation" else "operator_seen_training36", mode="formal",
        state_count=50 if panel == "validation" else 4, init_state_ids=None if panel == "validation" else (32, 33, 34, 35),
        replicas_per_gpu=replicas, gpu_indices=gpu_indices, static_task_lora_manifest=bank_path,
        trajectory_capture_selection=bank_path.parent / "capture.json", task_subset_selection=None,
        occupancy_capture_selection=None, capture_stage_predicates=False, exploration_sigma=False,
        source_sft_config=None, source_sft_checkpoint=None, task_expert_config=None,
        task_expert_bank_root=None, task_expert_step=None)
