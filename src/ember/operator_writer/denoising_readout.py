"""Fixed T2340/63/72 readouts on the canonical RGB compiler and evaluator."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from safetensors.torch import save_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, expected_lora_state_shapes
from ember.pi05_eval.scene import inspect_registered_scenes
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.materialization_workers import execution_devices

from . import bank as owner, scope
from .materialization import compile_conditions, register_partial
from .public_beta import public_state


TASK = "denoising_return_writer_20261006"
SCHEMA = "denoising_return_writer_v1"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
CONTRACT = ROOT / "contract.json"
PARENT = (ROOT.parent / "operator_read_write_learning_20260928/continuation2340/T/train"
          "/attempts/continuation/checkpoints/macro_00002340")
PARENT_BANK = PARENT.parents[4] / "banks/2340/manifest.json"
PARENT_SEEN = ROOT.parent / "query_conditioned_transition_read_20261005/parent/banks/seen12/manifest.json"
PARENT_ROWS = PARENT_SEEN.parents[2] / "evaluation/seen12/results.json"
SEEN_SCENES = Path(scope.registration()["run_root"]) / "attempts/scene_canonical144/scenes"
OLD_SEEN_IDS = (29, 34, 73)
OFFICIAL_IDS = (3, 6, 11, 16, 23, 26, 31, 39)


def node(checkpoint: Path, panel: str) -> str:
    checkpoint = Path(checkpoint).resolve()
    label = ("parent" if checkpoint == PARENT.resolve() else
             next((str(step) for step in (63, 72) if checkpoint ==
                   (ROOT / f"train/checkpoints/macro_{step:08d}").resolve()), None))
    if (label, panel) not in (("parent", "seen"), ("63", "official"),
                             ("72", "seen"), ("72", "official")):
        raise ValueError("denoising readout requires its fixed parent/63/72 panel")
    return label


def bank_path(checkpoint: Path, panel: str) -> Path:
    return ROOT / "readouts" / node(checkpoint, panel) / panel / "bank/manifest.json"


def evaluation_path(checkpoint: Path, panel: str, sampler: str) -> Path:
    label = node(checkpoint, panel)
    if sampler not in ("ODE", "SDE") or sampler == "SDE" and panel != "seen":
        raise ValueError("official denoising readout requires original ODE")
    return ROOT / "readouts" / label / panel / sampler / "evaluation"


def _lineage(checkpoint: Path, panel: str) -> tuple[dict, dict, dict, str]:
    label = node(checkpoint, panel)
    contract = read_json(CONTRACT)
    parent = read_json(PARENT.parent.parent / "run_contract.json")
    source_spec = file_record(owner.CONTINUATION2340_FROZEN_SPEC_PATH)
    if (contract.get("schema_version") != SCHEMA or contract.get("run_root") != str(ROOT)
            or contract.get("parent_checkpoint") != str(PARENT)
            or contract.get("source_spec") != source_spec
            or contract.get("source") != parent["source"]
            or parent.get("mode") != "T" or parent.get("source_trainable") != 0
            or parent.get("git") != owner.CONTINUATION2340_TRAINING_GIT
            or parent.get("spec") != source_spec["path"]):
        raise ValueError("denoising source must be the original frozen T2340 graph")
    git = contract.get("git") or {}
    if (not git.get("commit") or git.get("branch") != "" or git.get("dirty_paths") != []
            or git.get("pushed_ref") != "origin/main"):
        raise ValueError("denoising training source requires clean pushed detached code")
    spec = read_json(owner.CONTINUATION2340_FROZEN_SPEC_PATH)
    if (tuple(spec["evaluation"]["task_ids"]) != OFFICIAL_IDS
            or spec["evaluation"]["video_schedule_seed"] != 7
            or spec["operator"]["frame_stride"] != 5):
        raise ValueError("denoising parent video schedule or preprocessing changed")
    return contract, parent, spec, label


def _checkpoint(checkpoint: Path, contract: Mapping, parent: Mapping) -> str:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    old = checkpoint == PARENT.resolve()
    world = manifest.get("world_size", 0)
    files = manifest.get("files", {})
    expected = {"ecp.safetensors", "trainer_state.pt"} | {
        f"rank_{rank:02d}_state.pt" for rank in range(world) if world in range(1, 7)}
    if (world not in range(1, 7) or set(files) != expected
            or manifest.get("next_macro") != (2340 if old else int(checkpoint.name[6:]))
            or manifest.get("schema_version") != "ember_ecp_checkpoint_v1"
            or manifest.get("run_contract_schema") != (owner.SCHEMA if old else SCHEMA)
            or manifest.get("stage") != (owner.STAGE if old else TASK)
            or not old and (manifest.get("parent_checkpoint") != str(PARENT)
                            or manifest.get("source") != parent["source"]
                            or manifest.get("training_git") != contract["git"]["commit"])
            or any(not (checkpoint / name).is_file()
                   or (checkpoint / name).stat().st_size != row.get("bytes")
                   for name, row in files.items())):
        raise ValueError("denoising checkpoint is incomplete or changes training lineage")
    return parent["git"]["commit"] if old else contract["git"]["commit"]


def _tasks(asset_root: Path, spec: Mapping, panel: str) -> tuple[list, list]:
    return scope.task_rows(asset_root, spec) if panel == "seen" else owner.task_rows(spec, asset_root)


def _stage(checkpoint: Path, panel: str) -> dict:
    label = node(checkpoint, panel)
    result = {"schema_version": SCHEMA, "study": TASK, "node": label, "panel": panel,
              "runtime_mode": "T", "parent_checkpoint": str(PARENT),
              "video_schedule_seed": 20260928 if panel == "seen" else 7}
    if label == "parent":
        result["parent_ode_reused"] = {"bank": file_record(PARENT_SEEN),
            "results": file_record(PARENT_ROWS), "global_task_ids": list(OLD_SEEN_IDS), "rows": 12}
    return result


def _parent_conditions(tasks: list, conditions: list, panel: str, source: Mapping) -> dict:
    path = PARENT_SEEN if panel == "seen" else PARENT_BANK
    bank = read_json(path)
    expected_tasks = [task for task in tasks if panel != "seen" or task["global_task_id"] in OLD_SEEN_IDS]
    if (bank.get("kind") != owner.KIND or bank.get("schema_version") != owner.BANK_SCHEMA
            or bank.get("mode") != ("parent" if panel == "seen" else "T")
            or bank.get("checkpoint") != str(PARENT) or bank.get("source") != source
            or bank.get("spec") != file_record(owner.CONTINUATION2340_FROZEN_SPEC_PATH)
            or bank.get("tasks") != expected_tasks
            or bank.get("scene_root") != str(SEEN_SCENES if panel == "seen" else owner.SCENE_ROOT)
            or bank.get("shared") != read_json(PARENT_BANK)["shared"]):
        raise ValueError("original T2340 bank reuse changes source/teacher/scene")
    if panel == "seen":
        _parent_rows(expected_tasks, source)
    expected = {row["condition_id"]: row for row in conditions if panel != "seen"
                or row["global_task_id"] in OLD_SEEN_IDS}
    reused = {}
    for row in bank["conditions"]:
        key = row["condition_id"]
        if key in reused or key not in expected or any(row[k] != v for k, v in expected[key].items()):
            raise ValueError("original T2340 reused condition changed")
        reused[key] = {**row, "reused_bank": file_record(path)}
    if set(reused) != set(expected):
        raise ValueError("original T2340 reused conditions are incomplete")
    return reused


def _parent_rows(tasks: list, source: Mapping) -> None:
    results = read_json(PARENT_ROWS)
    adapter = results.get("adapter", {})
    expected = {(task["suite"], task["task_id"], episode["init_state_id"]): episode
                for task in tasks for episode in task["episodes"]}
    rows = results.get("rows", ())
    if (adapter.get("checkpoint") != str(PARENT) or adapter.get("source") != source
            or adapter.get("manifest") != file_record(PARENT_SEEN)
            or adapter.get("scene_manifest") != file_record(SEEN_SCENES / "manifest.json")
            or len(rows) != 12 or {(row["suite"], row["task_id"], row["init_state_id"])
                                   for row in rows} != set(expected)):
        raise ValueError("original parent ODE twelve rows change checkpoint/scene/scope")
    for row in rows:
        episode = expected[(row["suite"], row["task_id"], row["init_state_id"])]
        evidence = row.get("operator_read_write_lora", {})
        if (row.get("env_seed") != 7 or row.get("policy_seed_root") != 7
                or evidence.get("checkpoint") != str(PARENT)
                or evidence.get("condition_id") != episode["condition_id"]
                or evidence.get("teacher_demo") != episode["teacher_demo_indices"][0]
                or evidence.get("video_ordinal") != episode["video_ordinal"]
                or evidence.get("scene_manifest") != adapter["scene_manifest"]):
            raise ValueError("original parent ODE task/video/env/policy pairing changed")


def _write_shared(output: Path, checkpoint: Path, lora, label: str) -> dict:
    if label == "parent":
        return read_json(PARENT_BANK)["shared"]
    path = output / "shared.safetensors"
    shapes = {name: shape for name, shape in expected_lora_state_shapes(lora).items()
              if name.endswith(LORA_A_SUFFIX)}
    metadata = {"schema_version": owner.BANK_SCHEMA, "mode": "T"}
    if path.exists():
        owner._factor_header(path, shapes, metadata=metadata)
    else:
        shared = {name: value for name, value in public_state(checkpoint, lora).items()
                  if name.endswith(LORA_A_SUFFIX)}
        save_file(shared, str(path), metadata=metadata)
    return file_record(path)


def _capture_manifest(tasks: list, first: int) -> dict:
    return {"schema_version": "ember_pi05_registered_trajectory_capture_v1", "study_id": TASK,
            "task_subset_selection": None, "mode": "compact",
            "full_conditions": [{"suite": row["suite"], "task_id": row["task_id"],
                                 "init_state_id": first} for row in tasks],
            "passive_control_trace": owner.PASSIVE_TAG, "stage_predicates": True,
            "training_gradient_use": False, "checkpoint_selection_use": False,
            "validation_use": False, "test_use": False}


def materialize(checkpoint: Path, asset_root: Path, *, panel: str, devices=None,
                native_frame_chunk: int | None = None, cpu_threads: int = 6,
                resume_materialization_git: str | None = None) -> Path:
    """Compile only absent conditions; historical factors stay at their original paths."""
    checkpoint, asset_root = Path(checkpoint).resolve(), Path(asset_root).resolve()
    contract, parent, spec, label = _lineage(checkpoint, panel)
    training_git = _checkpoint(checkpoint, contract, parent)
    tasks, conditions = _tasks(asset_root, spec, panel)
    path = bank_path(checkpoint, panel)
    if path.exists():
        inspect(read_json(path), path, parent["source"],
                tuple((row["suite"], row["task_id"]) for row in tasks),
                scope.ROLE if panel == "seen" else "validation", True, None)
        return path
    if native_frame_chunk is not None and native_frame_chunk < 1 or cpu_threads < 1:
        raise ValueError("denoising materialization chunk and threads must be positive")
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset_root / spec["source"]["lora_contract"]), rank=128)
    if lora.to_dict() != parent["lora"]:
        raise ValueError("denoising materialization requires the parent's complete38 rank128 contract")
    mode, stage = f"denoising_return_T_{panel}", _stage(checkpoint, panel)
    registration = {"mode": mode, "checkpoint": str(checkpoint), "source": parent["source"],
        "spec": file_record(owner.CONTINUATION2340_FROZEN_SPEC_PATH),
        "training_spec": file_record(CONTRACT), "training_git": training_git,
        "materialization_git": owner.frozen_git(), "lora": lora.to_dict(), "denoising_return": stage}
    output = path.parent
    if output.exists() and not (output / "materialization_contract.json").is_file():
        raise ValueError("denoising bank output exists without registered lineage")
    output.mkdir(parents=True, exist_ok=True)
    register_partial(output, registration, resume_materialization_git)
    shared = _write_shared(output, checkpoint, lora, label)
    reused = _parent_conditions(tasks, conditions, panel, parent["source"]) if label == "parent" else {}
    pending = [row for row in conditions if row["condition_id"] not in reused]
    b_shapes = {name: shape for name, shape in expected_lora_state_shapes(lora).items()
                if name.endswith(LORA_B_SUFFIX)}
    compile_conditions(asset_root, spec, "T", checkpoint, parent["source"], output, pending, b_shapes,
        devices=execution_devices(None, devices), frame_chunk=native_frame_chunk or spec["operator"]["frame_chunk"],
        task_ids=tuple(row["global_task_id"] for row in tasks),
        role="train" if panel == "seen" else "validation", cpu_threads=cpu_threads)
    completed = {**reused, **{row["condition_id"]: row for row in pending}}
    bank = {**registration, "schema_version": owner.BANK_SCHEMA, "kind": owner.KIND,
        "asset_root": str(asset_root), "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
        "shared": shared, "conditions": [completed[row["condition_id"]] for row in conditions], "tasks": tasks,
        "scene_root": str(SEEN_SCENES if panel == "seen" else owner.SCENE_ROOT),
        "information_wall": {"teacher_video_values_read": len(conditions), "teacher_runtime_reads": 0,
                             "deployment_adapters": 1, "validation_test_gradients": False}}
    write_json_atomic(output.parent / "capture.json", _capture_manifest(tasks, 32 if panel == "seen" else 0))
    if label == "parent":
        remaining = [task for task in tasks if task["global_task_id"] not in OLD_SEEN_IDS]
        write_json_atomic(output.parent / "capture_ODE.json", _capture_manifest(remaining, 32))
    write_json_atomic(output / "manifest.json", bank)
    return path


def inspect(bank: Mapping, path: Path, source: Mapping, task_keys: tuple,
            evaluation_role: str, require_formal: bool, task_init_state_ids: Mapping | None) -> dict:
    checkpoint, panel = Path(bank["checkpoint"]).resolve(), bank["denoising_return"]["panel"]
    contract, parent, spec, label = _lineage(checkpoint, panel)
    tasks, planned = _tasks(Path(bank["asset_root"]), spec, panel)
    remaining = [row for row in tasks if row["global_task_id"] not in OLD_SEEN_IDS]
    subset = label == "parent" and set(task_keys) == {(row["suite"], row["task_id"]) for row in remaining}
    selected, states = remaining if subset else tasks, scope.STATES if panel == "seen" else tuple(range(50))
    expected = {"schema_version": owner.BANK_SCHEMA, "kind": owner.KIND,
        "mode": f"denoising_return_T_{panel}", "spec": file_record(owner.CONTINUATION2340_FROZEN_SPEC_PATH),
        "training_spec": file_record(CONTRACT), "training_git": _checkpoint(checkpoint, contract, parent),
        "source": parent["source"], "lora": parent["lora"], "tasks": tasks,
        "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
        "denoising_return": _stage(checkpoint, panel),
        "scene_root": str(SEEN_SCENES if panel == "seen" else owner.SCENE_ROOT),
        "information_wall": {"teacher_video_values_read": len(planned), "teacher_runtime_reads": 0,
                             "deployment_adapters": 1, "validation_test_gradients": False}}
    git = bank.get("materialization_git", {})
    if (path.resolve() != bank_path(checkpoint, panel) or not require_formal
            or source != parent["source"] or not owner.source_matches(source, parent["source"])
            or evaluation_role != (scope.ROLE if panel == "seen" else "validation")
            or any(bank.get(key) != value for key, value in expected.items())
            or set(task_keys) != {(row["suite"], row["task_id"]) for row in selected}
            or not git.get("commit") or git.get("branch") != "" or git.get("dirty_paths") != []
            or git.get("pushed_ref") != "origin/main"
            or task_init_state_ids is not None and any(tuple(task_init_state_ids.get(
                (row["suite"], row["task_id"]), ())) != states for row in selected)):
        raise ValueError("denoising bank/source/task/checkpoint scope changed")
    registration = read_json(path.parent / "materialization_contract.json")
    keys = ("mode", "checkpoint", "source", "spec", "training_spec", "training_git",
            "materialization_git", "lora", "denoising_return")
    if registration != {key: bank.get(key) for key in keys}:
        raise ValueError("denoising materialization lineage changed")
    shapes = expected_lora_state_shapes(derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank["asset_root"]) / spec["source"]["lora_contract"]), rank=128))
    _inspect_factors(bank, path, planned, tasks, shapes, label, panel, source)
    scene_root = Path(bank["scene_root"])
    inspect_registered_scenes(scene_root, tasks, states=states,
        schema="ember_operator_seen_task_scenes_v1" if panel == "seen" else "ember_demonstration_formal_scenes_v1")
    return {**bank, "tasks": selected, "schema_version": owner.EVAL_SCHEMA, "arm": "correct",
            "manifest": file_record(path), "scene_manifest": file_record(scene_root / "manifest.json"),
            "denoising_return_readout": {"new_rows": len(selected) * len(states),
                "reused_rows": 12 if subset else 0, "parent_ode_reused": _stage(checkpoint, panel).get("parent_ode_reused")}}


def _inspect_factors(bank, path, planned, tasks, shapes, label, panel, source):
    a_shapes = {key: value for key, value in shapes.items() if key.endswith(LORA_A_SUFFIX)}
    b_shapes = {key: value for key, value in shapes.items() if key.endswith(LORA_B_SUFFIX)}
    wanted_shared = read_json(PARENT_BANK)["shared"] if label == "parent" else file_record(path.parent / "shared.safetensors")
    if bank["shared"] != wanted_shared:
        raise ValueError("denoising shared A source changed")
    owner._factor_header(Path(wanted_shared["path"]), a_shapes,
                         metadata={"schema_version": owner.BANK_SCHEMA, "mode": "T"})
    reused = _parent_conditions(tasks, planned, panel, source) if label == "parent" else {}
    if len(bank["conditions"]) != len(planned):
        raise ValueError("denoising condition count changed")
    for row, expected in zip(bank["conditions"], planned, strict=True):
        key = expected["condition_id"]
        original = reused.get(key)
        if (any(row.get(name) != value for name, value in expected.items())
                or original is not None and row != original
                or original is None and row.get("reused_bank") is not None
                or row.get("factors") != file_record(Path(row["factors"]["path"]))
                or original is None and Path(row["factors"]["path"]) != path.parent / f"{key}.safetensors"
                or int(row.get("raw_frames", 0)) < 1
                or row.get("sampled_frames") != (int(row["raw_frames"]) - 1) // 5 + 1
                   + int((int(row["raw_frames"]) - 1) % 5 != 0)):
            raise ValueError("denoising exact task/video/factor record changed")
        metadata = ({"study": "query_conditioned_transition_read_20261005", "arm": "parent",
                     "condition_id": key, "checkpoint": str(PARENT)} if original is not None else
                    {"schema_version": owner.BANK_SCHEMA, "condition_id": key, "mode": "T"})
        owner._factor_header(Path(row["factors"]["path"]), b_shapes, metadata=metadata)


def remaining_tasks(args, tasks, *, output_dir: Path) -> tuple:
    """Register the 132 new parent ODE rows; retain the old twelve as separate evidence."""
    bank = read_json(Path(args.static_task_lora_manifest))
    stage = bank.get("denoising_return", {})
    if stage.get("node") != "parent":
        return tuple(tasks)
    path = evaluation_path(Path(bank["checkpoint"]), stage["panel"], "ODE")
    if output_dir.resolve() != path:
        return tuple(tasks)
    if getattr(args, "denoising_return_sampler", None) is not None:
        raise ValueError("parent ODE remaining rows cannot use SDE sampling")
    expected = set(scope.task_keys_from_ids(scope.registration()["global_task_ids"]))
    if {(task.suite, task.task_id) for task in tasks} != expected:
        raise ValueError("parent ODE remaining selection requires the complete original seen144 scope")
    excluded = set(scope.task_keys_from_ids(OLD_SEEN_IDS))
    return tuple(task for task in tasks if (task.suite, task.task_id) not in excluded)


def capture_expectations(bank: Mapping, manifest_path: Path, tasks: list,
                         output_dir: Path | None = None) -> dict:
    checkpoint, stage = Path(bank["checkpoint"]), bank["denoising_return"]
    panel, label = stage["panel"], node(checkpoint, stage["panel"])
    default = evaluation_path(checkpoint, panel, "ODE")
    output = Path(output_dir).resolve() if output_dir is not None else default
    sampler = next((mode for mode in ("ODE", "SDE") if output ==
                    evaluation_path(checkpoint, panel, mode)), None) if panel == "seen" else "ODE"
    if sampler is None or output != evaluation_path(checkpoint, panel, sampler):
        raise ValueError("denoising capture is outside the fixed sampler/panel output")
    ids = scope.registration()["global_task_ids"] if panel == "seen" else OFFICIAL_IDS
    subset = label == "parent" and sampler == "ODE"
    if subset:
        ids = [task for task in ids if task not in OLD_SEEN_IDS]
    if tuple((task.suite, task.task_id) for task in tasks) != scope.task_keys_from_ids(ids):
        raise ValueError("denoising capture task geometry changed")
    first = 32 if panel == "seen" else 0
    return {"full": [{"suite": task.suite, "task_id": task.task_id, "init_state_id": first} for task in tasks],
        "capture": manifest_path.parent.parent / ("capture_ODE.json" if subset else "capture.json"),
        "study": TASK, "output": output, "role": scope.ROLE if panel == "seen" else "validation",
        "states": scope.STATES if panel == "seen" else tuple(range(50)),
        "task_count": len(ids), "expected_bank": bank_path(checkpoint, panel)}
