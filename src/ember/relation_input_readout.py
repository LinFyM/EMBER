"""Temporary P0/Q0/P180/Q180 banks on the canonical operator evaluator.

Owner: relation_input_compilation_20261007. Retire this module and its three
study dispatches when the fixed 576-row diagnostic exits. No live GT adapter.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file

from ember.lora import LORA_A_SUFFIX, expected_lora_state_shapes
from ember.operator_writer import scope
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.materialization import register_partial
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.scene import inspect_registered_scenes
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.materialization_workers import MaterializationWorkers, _configure_device, execution_devices

REPO = Path(__file__).resolve().parents[2]
TASK = "relation_input_compilation_20261007"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
OLD = ROOT.parent / "relation_grounded_writer_20261006"
SPEC_PATH = REPO / "configs/relation_input_compilation_v1/spec.json"
PARENT = OLD / "train/attempts/fresh/checkpoints/macro_00000450"
SCENES = ROOT.parent / "operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/scenes"
MODE = "relation_input_compilation"
PASSIVE_TAG = "ember_operator_read_write_passive_capture_v1"
RECORD_SCHEMA = "ember_relation_input_condition_v1"


def checkpoint_path(arm, u):
    if arm not in ("P", "Q") or u not in (0, 180):
        raise ValueError("only P0/Q0/P180/Q180 are registered")
    return PARENT if u == 0 else ROOT / "train" / arm / "checkpoints/macro_00000630"


def bank_path(arm, u):
    checkpoint_path(arm, u)
    return ROOT / "readouts" / arm / str(u) / "seen144/banks/manifest.json"


def evaluation_path(arm, u):
    return bank_path(arm, u).parent.parent / "evaluation"


def capture_path(arm, u):
    return bank_path(arm, u).parent.parent / "capture.json"


def _clean_git(git):
    if (not git.get("commit") or git.get("branch") != "" or git.get("dirty_paths") != []
            or git.get("pushed_ref") != "origin/main"):
        raise ValueError("formal P/Q consumers require clean pushed detached main")


def _specification(spec):
    diagnostic = spec.get("diagnostic", {})
    if (spec.get("task") != TASK or spec.get("run_root") != str(ROOT)
            or tuple(spec["events"]["task_ids"]) != TASKS or spec["events"]["seed"] != 20260928
            or spec["execution"]["updates_per_mode"] != 630 or spec["execution"]["checkpoints"] != [540, 630]
            or diagnostic.get("parent_checkpoint") != str(PARENT)
            or diagnostic.get("legacy_frozen") != str(OLD / "frozen_PEFTfix")
            or diagnostic.get("labels_root") != str(OLD / "labels")
            or diagnostic.get("arms") != ["P", "Q"] or diagnostic.get("shared_drop_last") is not True):
        raise ValueError("matched input diagnostic scope or common truncation changed")
    original = read_json(OLD / "frozen_PEFTfix/configs/relation_grounded_writer_v1/spec.json")
    model = {k: v for k, v in original["model"].items() if k not in ("F", "loss")}
    model["loss"] = {"G_FM": 1, "other_losses": 0}
    if (any(spec.get(k) != original[k] for k in ("source", "events", "optimization", "operator"))
            or spec.get("model") != model):
        raise ValueError("P/Q must retain the original source, event, model and optimizer contract")


def source_record(spec, arm, u):
    _specification(spec)
    checkpoint = checkpoint_path(arm, u)
    run_path = checkpoint.parent.parent / "run_contract.json"
    run, manifest = read_json(run_path), read_json(checkpoint / "checkpoint_manifest.json")
    world, files = manifest.get("world_size"), manifest.get("files", {})
    expected = ({"ecp.safetensors", "trainer_state.pt"}
                | {f"rank_{rank:02d}_state.pt" for rank in range(world)}) if world in range(1, 7) else set()
    if (manifest.get("next_macro") != (450 if u == 0 else 630) or not expected or set(files) != expected
            or manifest.get("stage") != run.get("stage")
            or manifest.get("run_contract_schema") != run.get("schema_version")
            or run.get("events") != spec["events"]
            or any(not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size != record.get("bytes")
                   for name, record in files.items())):
        raise ValueError("P/Q readout needs its complete registered checkpoint")
    if u == 0:
        if run["git"]["commit"] != "85614d9c1a146d27dbb36fb833b700037fe4b63e":
            raise ValueError("P0/Q0 must retain the original G450 training source")
    elif (run.get("arm") != arm
          or {k: v for k, v in read_json(Path(run["spec"])).items() if k not in ("evaluation", "budget")}
          != {k: v for k, v in spec.items() if k not in ("evaluation", "budget")}
          or run.get("diagnostic") != spec["diagnostic"]):
        raise ValueError("P/Q trained checkpoint belongs to a different input arm")
    _clean_git(run.get("git", {}))
    return checkpoint, run, run_path


def _training_provenance(spec, run, u):
    path = Path(run["spec"])
    original = read_json(path)
    changed = [k for k in ("evaluation", "budget") if original.get(k) != spec.get(k)] if u else []
    correction = ({"fields": changed, "scope": "evaluation_and_budget_metadata_only",
                   "consumed_contract_unchanged": True, "training_contract_preserved": True} if changed else None)
    return {"training_spec": file_record(path), "metadata_correction": correction}


def _lora(spec, assets):
    return derive_pi05_lora_rank(load_pi05_lora_contract(assets / spec["source"]["lora_contract"]), rank=128)


def _wall(arm):
    return {"privileged_training_diagnostic": arm == "Q", "deployment_candidate": False,
            "teacher_field_source": "frozen_Phi" if arm == "P" else "existing_train_teacher_GT",
            "teacher_video_conditions": 144, "teacher_GT_conditions": 144 if arm == "Q" else 0,
            "shared_drop_last": True, "GT_query_reads": 0, "GT_live_reads": 0,
            "held_geometry_reads": 0, "deployment_adapters": 1, "teacher_runtime_reads": 0,
            "validation_test_gradients": False}


def registration(arm, u, tasks):
    checkpoint_path(arm, u)
    return {"schema_version": "ember_pi05_registered_trajectory_capture_v1", "study_id": TASK,
            "input_arm": arm, "diagnostic_update": u, "task_subset_selection": None,
            "mode": "compact", "full_conditions": [
                {"suite": row["suite"], "task_id": row["task_id"], "init_state_id": 32} for row in tasks],
            "passive_control_trace": PASSIVE_TAG, "stage_predicates": True,
            **{key: False for key in ("training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use")}}


def _validate_record(record, condition, arm, raw, original_sampled, shapes):
    indices = list(range(0, raw, 5))
    if not indices or indices[-1] != raw - 1:
        indices.append(raw - 1)
    expected = {"task": condition["global_task_id"], "teacher_demo": condition["teacher_demo"], "arm": arm,
                "field_source": _wall(arm)["teacher_field_source"], "raw_frames": raw,
                "original_sampled_frames": original_sampled, "sampled_frames": len(indices) - 1,
                "original_indices": indices, "used_indices": indices[:-1], "native_reads": 1,
                "GT_query_reads": 0, "GT_live_reads": 0}
    if len(indices) < 2 or original_sampled != len(indices) or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("condition record lost its field source or common last-frame removal")
    mechanism = record.get("mechanism", {})
    names = {name.removesuffix(LORA_A_SUFFIX) for name in shapes if name.endswith(LORA_A_SUFFIX)}
    targets = mechanism.get("targets", {})
    if set(mechanism) != {"c", "d", "targets"} or set(targets) != names or any(set(t) != {"S", "M"} for t in targets.values()):
        raise ValueError("condition mechanism record is not the original complete38 compiler")
    summaries = [mechanism["c"], mechanism["d"], *(v for t in targets.values() for v in t.values())]
    if any(set(s) != {"rms", "norm", "max_abs", "finite"} or s["finite"] is not True
           or any(not isinstance(s[k], (int, float)) or not 0 <= s[k] < float("inf")
                  for k in ("rms", "norm", "max_abs")) for s in summaries):
        raise ValueError("condition c/d/S/M summary is incomplete or nonfinite")


def write_condition(runtime, data, labels, output, condition, shapes, *, arm, frame_chunk):
    from ember.operator_writer.bank import BANK_SCHEMA, _factor_header

    condition = dict(condition)
    path, record_path = output / f"{condition['condition_id']}.safetensors", output / f"{condition['condition_id']}.json"
    metadata = {"schema_version": BANK_SCHEMA, "condition_id": condition["condition_id"], "mode": f"{MODE}_{arm}"}
    raw, sampled = data.videos.frame_counts(condition["global_task_id"], condition["teacher_demo"])
    started, reused = time.monotonic(), path.exists() and record_path.exists()
    if reused:
        record = read_json(record_path)
        if record.get("schema_version") != RECORD_SCHEMA:
            raise ValueError("partial compilation record schema changed")
        _factor_header(path, shapes, metadata=metadata)
    else:
        from ember.relation_input_compilation import compile_condition

        while True:
            try:
                state, record = compile_condition(runtime, data, labels, condition["global_task_id"],
                                                  condition["teacher_demo"], frame_chunk, capture=True)
                break
            except torch.cuda.OutOfMemoryError:
                if frame_chunk <= 8:
                    raise
                frame_chunk = max(8, frame_chunk // 2)
                torch.cuda.empty_cache()
        factors = {name: value.detach().float().cpu().contiguous() for name, value in state.items()}
        if set(factors) != set(shapes) or any(tuple(factors[n].shape) != tuple(s) or not torch.isfinite(factors[n]).all()
                                           for n, s in shapes.items()):
            raise ValueError("P/Q must emit one complete finite FP32 A/B adapter")
        record = {"schema_version": RECORD_SCHEMA, **record}
        _validate_record(record, condition, arm, raw, sampled, shapes)
        temporary = path.with_suffix(".safetensors.tmp")
        save_file(factors, str(temporary), metadata=metadata)
        temporary.replace(path)
        write_json_atomic(record_path, record)
    _validate_record(record, condition, arm, raw, sampled, shapes)
    condition.update(factors=file_record(path), compilation_record=file_record(record_path),
                     raw_frames=raw, sampled_frames=sampled - 1)
    return condition, {"condition_id": condition["condition_id"], "reused": reused,
                       "device": str(runtime.device), "pid": os.getpid(), "frame_chunk": frame_chunk,
                       "seconds": time.monotonic() - started,
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device) if runtime.device.type == "cuda" else 0}


class InputCompiler:
    """One source and one fixed P/Q compiler per persistent materializer worker."""
    def __init__(self, assets, config, device, cpu_threads):
        _configure_device(device, cpu_threads)
        self.assets, self.config, self.device = assets, config, device
        self.runtime = self.data = self.labels = self.request = None

    def prepare(self, request):
        from ember.relation_input_compilation import build_runtime, make_labels
        if request == self.request:
            return
        self.close()
        checkpoint, source, _output, _shapes, _frame_chunk = request
        self.runtime = build_runtime(self.assets, self.config["spec"], self.device, self.config["arm"], Path(checkpoint))
        if self.runtime.source != source:
            raise ValueError("P/Q materializer source differs from its training lineage")
        self.runtime.writer.requires_grad_(False).eval()
        self.runtime.policy.eval()
        self.data = FormalData(self.assets, self.config["spec"], query_labels=False, task_ids=TASKS, role="train")
        self.labels = make_labels(self.data, self.assets, self.config["spec"]) if self.config["arm"] == "Q" else None
        self.request = request

    def compile(self, condition):
        _checkpoint, _source, output, shapes, frame_chunk = self.request
        return write_condition(self.runtime, self.data, self.labels, Path(output), condition, shapes,
                               arm=self.config["arm"], frame_chunk=frame_chunk)

    def close(self):
        if self.data is not None:
            self.data.close()
        self.runtime = self.data = self.labels = self.request = None


def compile_conditions(assets, spec, arm, checkpoint, source, output, conditions, shapes, *, devices, frame_chunk, cpu_threads):
    data = FormalData(assets, spec, query_labels=False, task_ids=TASKS, role="train")
    completed, statistics, pending = {}, [], []
    try:
        for condition in conditions:
            if ((output / f"{condition['condition_id']}.safetensors").exists()
                    and (output / f"{condition['condition_id']}.json").exists()):
                reader = type("HeaderReader", (), {"device": torch.device("cpu")})()
                value, stats = write_condition(reader, data, None, output, condition, shapes, arm=arm, frame_chunk=frame_chunk)
                completed[value["condition_id"]] = value
                statistics.append(stats)
            else:
                pending.append(condition)
        pending.sort(key=lambda c: data.videos.frame_counts(c["global_task_id"], c["teacher_demo"])[1], reverse=True)
    finally:
        data.close()
    request = (str(checkpoint), source, str(output), shapes, frame_chunk)
    if pending:
        with MaterializationWorkers(asset_root=assets, config={"spec": spec, "arm": arm}, devices=devices,
                                    cpu_threads=cpu_threads, compiler_factory=InputCompiler) as workers:
            for _job, (value, stats) in workers.compile(request, pending):
                if value["condition_id"] in completed:
                    raise ValueError("P/Q dynamic queue returned a duplicate condition")
                completed[value["condition_id"]] = value
                statistics.append(stats)
    conditions[:] = [completed[c["condition_id"]] for c in conditions]
    write_json_atomic(output / "materialization_execution.json", {"devices": [str(d) for d in devices],
        "native_frame_chunk": frame_chunk, "cpu_threads_per_worker": cpu_threads, "conditions": statistics})


def _materializer_devices(gpus):
    physical = tuple(int(g) for g in gpus.split(",") if g)
    if not 1 <= len(physical) <= 6 or min(physical) < 0 or len(set(physical)) != len(physical):
        raise ValueError("P/Q materialization requires 1..6 distinct explicitly admitted GPUs")
    visible = ",".join(str(g) for g in physical)
    if torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise ValueError("materializer CUDA is already initialized on another physical mapping")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return execution_devices(devices=tuple(torch.device(f"cuda:{i}") for i in range(len(physical))))


def _shared_a0(checkpoint, path, shapes, arm):
    from ember.operator_writer.bank import BANK_SCHEMA, _factor_header

    metadata = {"schema_version": BANK_SCHEMA, "mode": f"{MODE}_{arm}"}
    a_shapes = {n: s for n, s in shapes.items() if n.endswith(LORA_A_SUFFIX)}
    if not path.exists():
        from ember.relation_input_compilation import load_G_weights

        weights = load_G_weights(checkpoint)
        values = {n: weights[f"common.values.{i}"].float().contiguous()
                  for i, n in enumerate(sorted(shapes)) if n.endswith(LORA_A_SUFFIX)}
        save_file(values, str(path), metadata=metadata)
    _factor_header(path, a_shapes, metadata=metadata)


def materialize(spec, args):
    from ember.operator_writer import bank as owner
    from ember.operator_writer.run import frozen_git

    devices = _materializer_devices(args.gpus)
    assets, arm, u = args.asset_root.resolve(), args.arm, args.u
    checkpoint, run, run_path = source_record(spec, arm, u)
    if read_json(args.spec) != spec or args.spec.resolve() != SPEC_PATH.resolve() or args.frame_chunk < 1 or args.cpu_threads < 1:
        raise ValueError("P/Q materialization spec or packing changed")
    tasks, conditions = scope.task_rows(assets, spec)
    lora, path = _lora(spec, assets), bank_path(arm, u)
    if args.output is not None and args.output.resolve() != path.parent.resolve():
        raise ValueError("P/Q materializer output is its canonical bank directory")
    if run["lora"] != lora.to_dict():
        raise ValueError("P/Q must retain the trained complete38/rank128 adapter")
    if path.exists():
        inspect(read_json(path), path, run["source"], tuple((t["suite"], t["task_id"]) for t in tasks),
                scope.ROLE, True, {(t["suite"], t["task_id"]): scope.STATES for t in tasks})
        return path
    inspect_registered_scenes(SCENES, tasks, states=scope.STATES, schema="ember_operator_seen_task_scenes_v1")
    contract = {"study_id": TASK, "relation_input_compilation_study": True, "mode": f"{MODE}_{arm}",
        "input_arm": arm, "diagnostic_update": u, "checkpoint": str(checkpoint), "spec": file_record(args.spec),
        "training_run": file_record(run_path), "training_git": run["git"]["commit"], "source": run["source"],
        "lora": lora.to_dict(), "materialization_git": frozen_git(),
        **_training_provenance(spec, run, u),
        "condition_factors": "complete_A0_plus_S_B0_plus_M", "shared_role": "public_A0_provenance_only_not_execution"}
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_git = None
    if args.resume is not None:
        if args.resume.resolve() != (path.parent / "materialization_contract.json").resolve():
            raise ValueError("materialization resume must name this bank's prior contract")
        previous_git = read_json(args.resume)["materialization_git"]["commit"]
    register_partial(path.parent, contract, previous_git)
    selector, selected = capture_path(arm, u), registration(arm, u, tasks)
    if selector.exists() and read_json(selector) != selected:
        raise ValueError("P/Q registered full/compact scope changed")
    write_json_atomic(selector, selected)
    shapes, shared = expected_lora_state_shapes(lora), path.parent / "shared.safetensors"
    _shared_a0(checkpoint, shared, shapes, arm)
    compile_conditions(assets, spec, arm, checkpoint, run["source"], path.parent, conditions, shapes,
                       devices=devices, frame_chunk=args.frame_chunk, cpu_threads=args.cpu_threads)
    write_json_atomic(path, {**contract, "schema_version": owner.BANK_SCHEMA, "kind": owner.KIND,
        "status": "sealed", "asset_root": str(assets), "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
        "shared": file_record(shared), "conditions": conditions, "tasks": tasks, "scene_root": str(SCENES),
        "information_wall": _wall(arm)})
    return path


def _inspect_conditions(bank, path, conditions, shapes, arm):
    from ember.operator_writer.bank import BANK_SCHEMA, _factor_header
    shared = path.parent / "shared.safetensors"
    if bank.get("shared") != file_record(shared) or len(bank.get("conditions", ())) != len(conditions):
        raise ValueError("P/Q full condition factors or public provenance changed")
    _factor_header(shared, {n: s for n, s in shapes.items() if n.endswith(LORA_A_SUFFIX)},
                   metadata={"schema_version": BANK_SCHEMA, "mode": f"{MODE}_{arm}"})
    for row, wanted in zip(bank["conditions"], conditions, strict=True):
        factor, record_path = path.parent / f"{wanted['condition_id']}.safetensors", path.parent / f"{wanted['condition_id']}.json"
        if (set(row) != set(wanted) | {"factors", "compilation_record", "raw_frames", "sampled_frames"}
                or any(row.get(k) != v for k, v in wanted.items()) or row["factors"] != file_record(factor)
                or row["compilation_record"] != file_record(record_path)):
            raise ValueError("P/Q factor/video/record lineage changed")
        _factor_header(factor, shapes, metadata={"schema_version": BANK_SCHEMA, "condition_id": wanted["condition_id"], "mode": f"{MODE}_{arm}"})
        record = read_json(record_path)
        if record.get("schema_version") != RECORD_SCHEMA or record.get("sampled_frames") != row["sampled_frames"]:
            raise ValueError("P/Q used-frame count changed")
        _validate_record(record, wanted, arm, row["raw_frames"], row["sampled_frames"] + 1, shapes)


def inspect(bank, path, source, task_keys, evaluation_role, require_formal, task_init_state_ids):
    from ember.operator_writer import bank as owner
    arm, u = bank["input_arm"], bank["diagnostic_update"]
    spec, assets = read_json(Path(bank["spec"]["path"])), Path(bank["asset_root"])
    checkpoint, run, run_path = source_record(spec, arm, u)
    tasks, conditions = scope.task_rows(assets, spec)
    lora = _lora(spec, assets)
    expected = {"study_id": TASK, "relation_input_compilation_study": True, "mode": f"{MODE}_{arm}",
        "kind": owner.KIND, "schema_version": owner.BANK_SCHEMA, "status": "sealed", "checkpoint": str(checkpoint),
        "spec": file_record(SPEC_PATH), "training_run": file_record(run_path), "training_git": run["git"]["commit"],
        **_training_provenance(spec, run, u),
        "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"), "source": source, "lora": lora.to_dict(),
        "tasks": tasks, "scene_root": str(SCENES), "information_wall": _wall(arm),
        "condition_factors": "complete_A0_plus_S_B0_plus_M", "shared_role": "public_A0_provenance_only_not_execution"}
    if (path.resolve() != bank_path(arm, u).resolve() or any(bank.get(k) != v for k, v in expected.items())
            or source != run["source"] or run["lora"] != lora.to_dict() or not owner.source_matches(source, run["source"])
            or evaluation_role != scope.ROLE or require_formal is not True
            or set(task_keys) != {(t["suite"], t["task_id"]) for t in tasks}
            or task_init_state_ids is None or set(task_init_state_ids) != set(task_keys)
            or any(tuple(v) != scope.STATES for v in task_init_state_ids.values())):
        raise ValueError("P/Q readout source, train36/state32..35 or adapter scope changed")
    _clean_git(bank.get("materialization_git", {}))
    contract = read_json(path.parent / "materialization_contract.json")
    if any(bank.get(k) != v for k, v in contract.items()):
        raise ValueError("P/Q materialization contract changed")
    _inspect_conditions(bank, path, conditions, expected_lora_state_shapes(lora), arm)
    inspect_registered_scenes(SCENES, tasks, states=scope.STATES, schema="ember_operator_seen_task_scenes_v1")
    return {**bank, "schema_version": owner.EVAL_SCHEMA, "arm": "correct", "manifest": file_record(path),
            "scene_manifest": file_record(SCENES / "manifest.json")}


def registered_capture(args, tasks, output_dir, path, manifest, task_subset):
    bank = read_json(Path(args.static_task_lora_manifest))
    arm, u = bank["input_arm"], bank["diagnostic_update"]
    expected_tasks, _conditions = scope.task_rows(Path(bank["asset_root"]), read_json(Path(bank["spec"]["path"])))
    expected_keys = {(t["suite"], t["task_id"]) for t in expected_tasks}
    if (bank.get("study_id") != TASK or bank.get("relation_input_compilation_study") is not True
            or manifest != registration(arm, u, expected_tasks) or task_subset is not None
            or Path(args.static_task_lora_manifest).resolve() != bank_path(arm, u).resolve()
            or path.resolve() != capture_path(arm, u).resolve() or output_dir.resolve() != evaluation_path(arm, u).resolve()
            or args.role != scope.ROLE or args.mode != "formal" or len(tasks) != 36
            or {(t.suite, t.task_id) for t in tasks} != expected_keys
            or any(tuple(t.init_state_ids) != scope.STATES for t in tasks)):
        raise Pi05EvaluationError("P/Q full36/compact108 capture or train-only pairing changed")
    capture = {"schema_version": manifest["schema_version"], "selection_path": str(path), "selection_bytes": path.stat().st_size,
        "mode": "compact", "full_conditions": manifest["full_conditions"], "trajectory_root": str((output_dir / "trajectories").resolve()),
        "passive_trace": {"schema_version": PASSIVE_TAG, "trace_root": str((output_dir / "continuous_traces").resolve())},
        **{key: False for key in ("training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use")}}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1", "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False, "training_gradient_use": False,
        "checkpoint_selection_use": False, "validation_action_reads": 0, "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def evaluate(spec, args):
    _specification(spec)
    if read_json(args.spec) != spec or args.spec.resolve() != SPEC_PATH.resolve():
        raise ValueError("P/Q evaluation must use its frozen diagnostic spec")
    arm, u, assets = args.arm, args.u, args.asset_root.resolve()
    path, output = bank_path(arm, u), evaluation_path(arm, u)
    bank = read_json(path)
    if args.output is not None and args.output.resolve() != output.resolve():
        raise ValueError("P/Q evaluator output is fixed by its registered panel")
    command = [sys.executable, str(REPO / "scripts/evaluate_pi05.py")]
    if args.resume is not None:
        if args.resume.resolve() != output.resolve():
            raise ValueError("evaluation resume must name its canonical evaluation root")
        contract = read_json(output / "run_contract.json")
        if (args.gpus != ",".join(str(i) for i in contract["parallel"]["physical_gpu_ids"])
                or args.replicas != contract["parallel"]["replicas_per_gpu"]):
            raise ValueError("P/Q evaluator resume requires the existing worker allocation")
        command += ["resume", "--output-dir", str(output)]
    else:
        source = bank["source"]
        command += ["run", "--config", str(assets / spec["source"]["evaluation_config"]),
            "--source-run", source["source_run"], "--checkpoint", source["checkpoint"],
            "--tokenizer-path", str(assets / spec["source"]["tokenizer"]), "--output-dir", str(output),
            "--role", scope.ROLE, "--mode", "formal", "--state-count", "4", "--init-state-ids", "32,33,34,35",
            "--static-task-lora-manifest", str(path), "--trajectory-capture-selection", str(capture_path(arm, u)),
            "--replicas-per-gpu", str(args.replicas), "--gpu-indices", args.gpus]
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)  # Canonical evaluator maps admitted physical IDs itself.
    subprocess.run(command, cwd=REPO, env=environment, check=True)
    return output / "results.json"
