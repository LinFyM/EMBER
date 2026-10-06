"""The four registered G/F panels on the canonical bank and PI05 evaluator."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from safetensors import safe_open
from safetensors.torch import save_file

from ember.lora import LORA_A_SUFFIX, expected_lora_state_shapes
from ember.operator_writer import scope
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.scene import inspect_registered_scenes
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record


REPO = Path(__file__).resolve().parents[3]
TASK = "relation_grounded_writer_20261006"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
SPEC_PATH = REPO / "configs/relation_grounded_writer_v1/spec.json"
MODE = "relation_grounded_writer"
F_KIND = "relation_grounded_feedback_diagnostic"
F_BANK_SCHEMA = "ember_relation_feedback_bank_v1"
F_EVAL_SCHEMA = "ember_relation_feedback_eval_v1"
F_EPISODE_SCHEMA = "ember_relation_feedback_episode_v1"
PASSIVE_TAG = "ember_relation_grounded_writer_passive_capture_v1"
OFFICIAL_SCENES = ROOT.parent / "demonstration_transfer_learning_20260927/scenes"
SEEN_SCENES = ROOT.parent / "operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/scenes"


def checkpoint_path(macro: int) -> Path:
    if macro not in (360, 450):
        raise ValueError("relation readouts only use fixed360 and450")
    return ROOT / "train/attempts/fresh/checkpoints" / f"macro_{macro:08d}"


def _macro(checkpoint: Path) -> int:
    checkpoint = checkpoint.resolve()
    for macro in (360, 450):
        if checkpoint == checkpoint_path(macro).resolve():
            return macro
    raise ValueError("relation readout checkpoint is outside its unique fresh attempt")


def bank_path(model: str, macro: int, panel: str) -> Path:
    if (model, macro, panel) not in {("G", 360, "correct400"), ("G", 450, "correct400"),
                                    ("G", 450, "seen144"), ("F", 450, "seen144")}:
        raise ValueError("relation readout model/checkpoint/panel is not registered")
    return ROOT / "readouts" / model / str(macro) / panel / "banks/manifest.json"


def evaluation_path(model: str, macro: int, panel: str) -> Path:
    return bank_path(model, macro, panel).parent.parent / "evaluation"


def capture_path(model: str, macro: int, panel: str) -> Path:
    return bank_path(model, macro, panel).parent.parent / "capture.json"


def _geometry(panel: str, spec: Mapping, asset_root: Path):
    from ember.operator_writer.bank import task_rows

    if panel == "seen144":
        tasks, conditions = scope.task_rows(asset_root, spec)
        return tasks, conditions, SEEN_SCENES, scope.STATES, scope.ROLE
    if panel != "correct400":
        raise ValueError("relation panel must be correct400 or seen144")
    tasks, conditions = task_rows(spec, asset_root)
    if (spec["evaluation"]["task_ids"] != [3, 6, 11, 16, 23, 26, 31, 39]
            or spec["evaluation"]["video_schedule_seed"] != 7):
        raise ValueError("relation validation task IDs or paired video schedule changed")
    return tasks, conditions, OFFICIAL_SCENES, tuple(range(50)), "validation"


def source_record(checkpoint: Path) -> tuple[dict, dict, Path]:
    """A complete360 can be consumed while the same fresh run continues to450."""
    macro = _macro(checkpoint)
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    path = Path(run["spec"])
    spec = read_json(path)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    world, files = manifest.get("world_size"), manifest.get("files", {})
    expected_files = ({"ecp.safetensors", "trainer_state.pt"}
                      | {f"rank_{rank:02d}_state.pt" for rank in range(world)}) if world in range(1, 7) else set()
    if (spec != read_json(SPEC_PATH) or spec.get("task") != TASK or spec.get("run_root") != str(ROOT)
            or run.get("source_trainable") != 0 or run.get("events") != spec["events"]
            or manifest.get("next_macro") != macro or set(files) != expected_files or not expected_files
            or manifest.get("stage") != run.get("stage")
            or manifest.get("run_contract_schema") != run.get("schema_version")
            or any(not (checkpoint / name).is_file()
                   or (checkpoint / name).stat().st_size != record.get("bytes")
                   for name, record in files.items())):
        raise ValueError("relation readout requires its complete frozen joint G/F checkpoint")
    _clean_git(run.get("git", {}))
    with safe_open(str(checkpoint / "ecp.safetensors"), framework="pt", device="cpu") as reader:
        if {key.split(".", 1)[0] for key in reader.keys()} != {"G", "F"}:
            raise ValueError("relation ECP does not contain the two independent learned owners")
    return spec, run, path


def _clean_git(git: Mapping) -> None:
    if (not git.get("commit") or git.get("branch") != "" or git.get("dirty_paths") != []
            or git.get("pushed_ref") != "origin/main"):
        raise ValueError("relation formal source/consumer must be clean pushed detached main")


def _wall(model: str, panel: str) -> dict:
    count = 144 if panel == "seen144" else 400
    return {"teacher_video_values_read": count if model == "G" else 0,
            "teacher_runtime_reads": 0, "deployment_adapters": 1 if model == "G" else 0,
            "validation_test_gradients": False, "privileged_training_diagnostic": model == "F",
            "teacher_geometry_conditions": count if model == "F" else 0,
            "current_environment_geometry": model == "F", "held_geometry_reads": 0}


def _registration(model: str, macro: int, panel: str, tasks: list) -> dict:
    state = 32 if panel == "seen144" else 0
    return {"schema_version": "ember_pi05_registered_trajectory_capture_v1", "study_id": TASK,
            "model": model, "macro": macro, "panel": panel, "task_subset_selection": None,
            "full_conditions": [{"suite": row["suite"], "task_id": row["task_id"], "init_state_id": state}
                                for row in tasks], "mode": "compact",
            "passive_control_trace": PASSIVE_TAG, "stage_predicates": True,
            "training_gradient_use": False, "checkpoint_selection_use": False,
            "validation_use": False, "test_use": False}


def _shared_a0(checkpoint: Path, shapes: dict) -> dict:
    """Export only public A0 provenance; execution always uses the full condition."""
    with safe_open(str(checkpoint / "ecp.safetensors"), framework="pt", device="cpu") as reader:
        return {name: reader.get_tensor(f"G.common.values.{index}").float().contiguous()
                for index, name in enumerate(sorted(shapes)) if name.endswith(LORA_A_SUFFIX)}


def materialize(model: str, checkpoint: Path, asset_root: Path, panel: str, devices=None,
                native_frame_chunk: int = 32, cpu_threads: int = 6, *, device=None,
                resume_materialization_git: str | None = None) -> Path:
    from ember.operator_writer import bank as owner
    from ember.operator_writer.materialization import register_partial
    from ember.operator_writer.run import frozen_git
    from ember.writer.materialization_workers import execution_devices
    from .materialization import compile_conditions

    checkpoint, asset_root = checkpoint.resolve(), asset_root.resolve()
    macro = _macro(checkpoint)
    path = bank_path(model, macro, panel)
    spec, run, spec_path = source_record(checkpoint)
    if cpu_threads < 1 or native_frame_chunk < 1 or path.exists():
        raise ValueError("relation readout needs positive packing and an unpublished bank")
    tasks, conditions, scenes, states, _role = _geometry(panel, spec, asset_root)
    inspect_registered_scenes(scenes, tasks, states=states, schema=(
        "ember_operator_seen_task_scenes_v1" if panel == "seen144" else "ember_demonstration_formal_scenes_v1"))
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset_root / spec["source"]["lora_contract"]), rank=128)
    if run["lora"] != lora.to_dict():
        raise ValueError("relation readout changed the trained complete38 rank128 LoRA")
    contract = {"mode": MODE, "study_id": TASK, "relation_grounded_study": True,
                "readout_model": model, "macro": macro, "panel": panel, "checkpoint": str(checkpoint),
                "spec": file_record(spec_path), "training_git": run["git"]["commit"],
                "training_run": file_record(checkpoint.parent.parent / "run_contract.json"),
                "source": run["source"], "lora": lora.to_dict(), "materialization_git": frozen_git()}
    if model == "G":
        contract.update(condition_factors="complete_A0_plus_S_B0_plus_M",
                        shared_role="public_A0_provenance_only_not_execution")
    path.parent.mkdir(parents=True, exist_ok=True)
    register_partial(path.parent, contract, resume_materialization_git)
    registration = _registration(model, macro, panel, tasks)
    selector = capture_path(model, macro, panel)
    if selector.exists() and read_json(selector) != registration:
        raise ValueError("registered relation full/compact selection changed")
    write_json_atomic(selector, registration)
    if model == "G":
        shapes, shared = expected_lora_state_shapes(lora), path.parent / "shared.safetensors"
        metadata = {"schema_version": owner.BANK_SCHEMA, "mode": MODE}
        if not shared.exists():
            save_file(_shared_a0(checkpoint, shapes), str(shared), metadata=metadata)
        owner._factor_header(shared, {name: shape for name, shape in shapes.items() if name.endswith(LORA_A_SUFFIX)},
                             metadata=metadata)
        compile_conditions(asset_root, spec, checkpoint, run["source"], path.parent, conditions, shapes,
                           devices=execution_devices(device, devices), frame_chunk=native_frame_chunk,
                           task_ids=tuple(row["global_task_id"] for row in tasks),
                           role="train" if panel == "seen144" else "validation", cpu_threads=cpu_threads)
    result = {**contract, "schema_version": owner.BANK_SCHEMA if model == "G" else F_BANK_SCHEMA,
              "kind": owner.KIND if model == "G" else F_KIND, "status": "sealed",
              "asset_root": str(asset_root), "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
              "conditions": conditions, "tasks": tasks, "scene_root": str(scenes),
              "information_wall": _wall(model, panel)}
    if model == "G":
        result["shared"] = file_record(shared)
    write_json_atomic(path, result)
    return path


def inspect(bank: Mapping, path: Path, source: Mapping, task_keys: tuple,
            evaluation_role: str, require_formal: bool, task_init_state_ids: Mapping | None) -> dict:
    from ember.operator_writer import bank as owner

    model, macro, panel = bank["readout_model"], bank["macro"], bank["panel"]
    checkpoint = Path(bank["checkpoint"])
    spec, run, spec_path = source_record(checkpoint)
    tasks, conditions, scenes, states, role = _geometry(panel, spec, Path(bank["asset_root"]))
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank["asset_root"]) / spec["source"]["lora_contract"]), rank=128)
    expected = ((path.resolve(), bank_path(model, macro, panel).resolve()), (_macro(checkpoint), macro),
                (bank.get("relation_grounded_study"), True), (bank.get("study_id"), TASK),
                (bank.get("mode"), MODE), (bank.get("status"), "sealed"),
                (bank.get("kind"), owner.KIND if model == "G" else F_KIND),
                (bank.get("schema_version"), owner.BANK_SCHEMA if model == "G" else F_BANK_SCHEMA),
                (bank.get("spec"), file_record(spec_path)), (bank.get("training_git"), run["git"]["commit"]),
                (bank.get("training_run"), file_record(checkpoint.parent.parent / "run_contract.json")),
                (bank.get("checkpoint_manifest"), file_record(checkpoint / "checkpoint_manifest.json")),
                (bank.get("source"), source), (source, run["source"]), (bank.get("lora"), lora.to_dict()),
                (run["lora"], lora.to_dict()), (bank.get("tasks"), tasks),
                (bank.get("scene_root"), str(scenes)), (bank.get("information_wall"), _wall(model, panel)),
                (evaluation_role, role), (require_formal, True),
                (set(task_keys), {(row["suite"], row["task_id"]) for row in tasks}))
    if (any(actual != wanted for actual, wanted in expected) or not owner.source_matches(source, run["source"])
            or task_init_state_ids is None or set(task_init_state_ids) != set(task_keys)
            or any(tuple(value) != states for value in task_init_state_ids.values())):
        raise ValueError("relation readout source/task/state/model scope changed")
    _clean_git(bank.get("materialization_git", {}))
    contract = read_json(path.parent / "materialization_contract.json")
    if any(contract.get(key) != bank.get(key) for key in contract):
        raise ValueError("relation bank materialization lineage changed")
    if model == "G":
        _inspect_factors(bank, path, conditions, expected_lora_state_shapes(lora))
    elif bank.get("conditions") != conditions:
        raise ValueError("privileged F teacher pairing changed")
    inspect_registered_scenes(scenes, tasks, states=states, schema=(
        "ember_operator_seen_task_scenes_v1" if panel == "seen144" else "ember_demonstration_formal_scenes_v1"))
    return {**bank, "schema_version": owner.EVAL_SCHEMA if model == "G" else F_EVAL_SCHEMA,
            "arm": "correct" if model == "G" else "privileged_training_feedback",
            "manifest": file_record(path), "scene_manifest": file_record(scenes / "manifest.json")}


def _inspect_factors(bank, path, conditions, shapes):
    from ember.operator_writer.bank import BANK_SCHEMA, _factor_header

    shared = path.parent / "shared.safetensors"
    if (bank.get("shared") != file_record(shared)
            or bank.get("condition_factors") != "complete_A0_plus_S_B0_plus_M"
            or bank.get("shared_role") != "public_A0_provenance_only_not_execution"
            or len(bank.get("conditions", ())) != len(conditions)):
        raise ValueError("relation G bank lost its complete condition factors")
    _factor_header(shared, {name: shape for name, shape in shapes.items() if name.endswith(LORA_A_SUFFIX)},
                   metadata={"schema_version": BANK_SCHEMA, "mode": MODE})
    for row, wanted in zip(bank["conditions"], conditions, strict=True):
        factor = path.parent / f"{wanted['condition_id']}.safetensors"
        if (set(row) != set(wanted) | {"factors", "raw_frames", "sampled_frames"}
                or any(row.get(key) != value for key, value in wanted.items())
                or row.get("factors") != file_record(factor)
                or not 0 < row["sampled_frames"] <= row["raw_frames"]):
            raise ValueError("relation G condition/video provenance changed")
        _factor_header(factor, shapes, metadata={"schema_version": BANK_SCHEMA,
                       "condition_id": wanted["condition_id"], "mode": MODE})


def registered_capture(args, tasks, output_dir, path, manifest, task_subset):
    bank = read_json(Path(args.static_task_lora_manifest))
    model, macro, panel = bank["readout_model"], bank["macro"], bank["panel"]
    _tasks, _conditions, _scenes, states, role = _geometry(panel, read_json(Path(bank["spec"]["path"])),
                                                       Path(bank["asset_root"]))
    registration = _registration(model, macro, panel, [vars(row) for row in tasks])
    if (bank.get("study_id") != TASK or task_subset is not None or manifest != registration
            or Path(args.static_task_lora_manifest).resolve() != bank_path(model, macro, panel).resolve()
            or path.resolve() != capture_path(model, macro, panel).resolve()
            or output_dir.resolve() != evaluation_path(model, macro, panel).resolve()
            or args.role != role or args.mode != "formal" or len(tasks) != len(_tasks)
            or any(tuple(row.init_state_ids) != states for row in tasks)):
        raise Pi05EvaluationError("registered relation full/compact capture scope changed")
    capture = {"schema_version": registration["schema_version"], "selection_path": str(path),
               "selection_bytes": path.stat().st_size, "mode": "compact",
               "full_conditions": registration["full_conditions"],
               "trajectory_root": str((output_dir / "trajectories").resolve()),
               "passive_trace": {"schema_version": PASSIVE_TAG,
                                 "trace_root": str((output_dir / "continuous_traces").resolve())},
               **{key: False for key in ("training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use")}}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
             "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False,
             "training_gradient_use": False, "checkpoint_selection_use": False,
             "validation_action_reads": 0, "validation_reward_reads": 0, "held_data_use": False,
             "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def attach_capture_provenance(contract, repo_root):
    del repo_root
    adapter, scene = contract.get("adapter") or {}, contract.get("operator_read_write_scene") or {}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    if (adapter.get("study_id") != TASK or scene.get("manifest") != adapter.get("scene_manifest")
            or adapter.get("kind") not in ("operator_read_write_lora_bank", F_KIND)
            or (capture.get("passive_trace") or {}).get("schema_version") != PASSIVE_TAG):
        raise Pi05EvaluationError("relation scene and adapter are not paired")
    contract["passive_capture_provenance"] = {
        "schema_version": PASSIVE_TAG, "bank": adapter["manifest"], "scene": adapter["scene_manifest"],
        "checkpoint": adapter["checkpoint"], "model": adapter["readout_model"],
        "evaluation_commit": contract["git"]["commit"]}


def validate_capture_contract(contract: Mapping, repo_root):
    adapter, capture = contract.get("adapter") or {}, contract.get("diagnostic_occupancy_capture") or {}
    if (capture.get("passive_trace") or {}).get("schema_version") != PASSIVE_TAG:
        raise Pi05EvaluationError("relation readouts require their all-row native control trace")
    path = Path(capture["selection_path"])
    args = SimpleNamespace(static_task_lora_manifest=Path(adapter["manifest"]["path"]),
                           role=contract["role"], mode=contract["mode"])
    expected, stage = registered_capture(args, [SimpleNamespace(**row) for row in contract["tasks"]],
                                         Path(contract["output_dir"]), path, read_json(path),
                                         contract.get("diagnostic_task_subset"))
    regenerated = dict(contract)
    attach_capture_provenance(regenerated, repo_root)
    if (capture != expected or contract.get("diagnostic_stage_predicates") != stage
            or contract.get("passive_capture_provenance") != regenerated["passive_capture_provenance"]):
        raise Pi05EvaluationError("relation capture/provenance changed after preparation")


def feedback_episode_evidence(bank, task, episode):
    return {"schema_version": F_EPISODE_SCHEMA, "privileged_training_diagnostic": True,
            "global_task_id": task["global_task_id"], "init_state_id": episode["init_state_id"],
            "condition_id": episode["condition_id"], "teacher_demo": episode["teacher_demo_indices"][0],
            "video_ordinal": episode["video_ordinal"], "checkpoint": bank["checkpoint"],
            "scene_manifest": bank["scene_manifest"]}


def validate_feedback_episode(bank, evidence, *, suite, task_id, init_state_id):
    task = next((row for row in bank["tasks"] if (row["suite"], row["task_id"]) == (suite, task_id)), None)
    episode = next((row for row in task["episodes"] if row["init_state_id"] == init_state_id), None) if task else None
    return (isinstance(evidence, Mapping) and episode is not None
            and dict(evidence) == feedback_episode_evidence(bank, task, episode))


def validate_feedback_episode_fields(bank, row, *, suite, task_id, init_state_id):
    return (all(row.get(name) is None for name in (
                "operator_read_write_lora", "horizon_writer_lora", "static_task_lora",
                "demonstration_comparison_lora", "conditional_velocity_lora", "task_expert",
                "policy_adapter_sha256"))
            and validate_feedback_episode(bank, row.get("relation_privileged_feedback"),
                                          suite=suite, task_id=task_id, init_state_id=init_state_id))
