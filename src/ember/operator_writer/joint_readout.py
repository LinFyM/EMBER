"""Registered fresh full/public study readouts on the canonical materializer and PI05 evaluator."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from safetensors.torch import save_file

from ember.lora import LORA_A_SUFFIX, expected_lora_state_shapes
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record

from . import scope
from .public_beta import factor_map, public_state


REPO = Path(__file__).resolve().parents[3]
TASK = "operator_joint_public_fresh_20260930"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
MODES = ("joint", "joint_public", "T450_public", "context", "context_public")
PUBLIC_MODES = ("joint_public", "T450_public", "context_public")
OLD_CHECKPOINT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928"
                      "/continuation900/T/train/attempts/continuation/checkpoints/macro_00000450")
PUBLIC_SCENES = Path(scope.registration()["run_root"]) / "attempts/scene_canonical144/scenes"
FIXED_PANEL = Path("/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929"
                   "/functional_credit_transport/group0")
TRAIN_TASKS = (0, 12, 20, 32)
TEACHERS = {0: (40, 11), 12: (25, 14), 20: (38, 42), 32: (17, 43)}


def study_root(mode: str) -> Path:
    from .joint_training import CONTEXT_ROOT

    if mode not in MODES:
        raise ValueError("unregistered fresh study readout mode")
    return CONTEXT_ROOT if mode in ("context", "context_public") else ROOT


def study_id(mode: str) -> str:
    from .joint_training import CONTEXT_TASK

    return CONTEXT_TASK if mode in ("context", "context_public") else TASK


def capture_path(mode: str) -> Path:
    prefix = "context" if mode in ("context", "context_public") else "joint"
    return REPO / "configs/operator_read_write_v1" / f"{prefix}_{'public' if mode in PUBLIC_MODES else 'official'}_capture.json"


def bank_path(mode: str) -> Path:
    if mode not in MODES:
        raise ValueError("unregistered joint readout mode")
    return study_root(mode) / mode / "banks/450/manifest.json"


def source_record(mode: str, checkpoint: Path) -> tuple[dict, dict, Path]:
    """Keep old training identity and new reading identity separate."""
    from . import bank, joint_training, run

    bank_path(mode)
    checkpoint = checkpoint.resolve()
    if checkpoint.name != "macro_00000450":
        raise ValueError("joint readouts use only the registered 450 endpoint")
    if mode == "T450_public":
        if checkpoint != OLD_CHECKPOINT.resolve():
            raise ValueError("T450 public readout changed its fixed old source")
        path = bank.CONTINUATION_FROZEN_SPEC_PATH
        spec = read_json(path)
        training = bank.inspect_training_source(spec, checkpoint, "T", sealed_evaluation=True)
    else:
        path = run.CONTEXT_SPEC_PATH if mode in ("context", "context_public") else run.JOINT_SPEC_PATH
        spec = run.specification(path)
        training = joint_training.inspect_source(spec, checkpoint)
    return spec, training, path


def _geometry(mode: str, spec: Mapping, asset_root: Path) -> tuple[list, list, Path, tuple]:
    from . import bank

    if mode in PUBLIC_MODES:
        tasks, conditions = scope.task_rows(asset_root, spec)
        return tasks, conditions, PUBLIC_SCENES, scope.STATES
    tasks, conditions = bank.task_rows(spec, asset_root)
    return tasks, conditions, bank.SCENE_ROOT, tuple(range(50))


def _wall(mode: str) -> dict:
    wall = {"teacher_video_values_read": 0 if mode in PUBLIC_MODES else 400,
            "teacher_runtime_reads": 0, "deployment_adapters": 1,
            "validation_test_gradients": False}
    if mode in PUBLIC_MODES:
        wall["video_id_role"] = "paired_metadata_only"
    return wall


def _public_intervention(lora) -> dict:
    return {"formula": "B0 A", "removed_term": "M(V,L) A",
            "factor_map": factor_map(lora)}


def materialize(mode: str, checkpoint: Path, asset_root: Path, devices=None,
                native_frame_chunk: int | None = None, cpu_threads: int = 6,
                *, device=None, resume_materialization_git: str | None = None) -> Path:
    """Public export is CPU-only; full export uses the existing condition queue."""
    from . import bank, run
    from .materialization import compile_conditions, register_partial
    from ember.pi05_eval.scene import inspect_registered_scenes
    from ember.writer.materialization_workers import execution_devices

    path = bank_path(mode)
    checkpoint, asset_root = checkpoint.resolve(), asset_root.resolve()
    spec, training, spec_path = source_record(mode, checkpoint)
    if cpu_threads < 1 or native_frame_chunk is not None and native_frame_chunk < 1:
        raise ValueError("joint readout packing must be positive")
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        asset_root / spec["source"]["lora_contract"]), rank=128)
    if training["lora"] != lora.to_dict():
        raise ValueError("joint readout changed the source LoRA contract")
    tasks, conditions, scenes, states = _geometry(mode, spec, asset_root)
    inspect_registered_scenes(scenes, tasks, states=states,
                              schema=("ember_operator_seen_task_scenes_v1" if mode in PUBLIC_MODES
                                      else "ember_demonstration_formal_scenes_v1"))
    if path.exists():
        raise ValueError("published joint readout bank already exists")
    contract = {"mode": mode, "joint_public_study": True, "checkpoint": str(checkpoint),
                "spec": file_record(spec_path), "training_git": training["git"]["commit"],
                "training_run": file_record(checkpoint.parent.parent / "run_contract.json"),
                "source": training["source"], "lora": lora.to_dict(),
                "materialization_git": run.frozen_git()}
    path.parent.mkdir(parents=True, exist_ok=True)
    register_partial(path.parent, contract, resume_materialization_git)
    state = public_state(checkpoint, lora)
    if mode not in PUBLIC_MODES:
        state = {key: value for key, value in state.items() if key.endswith(LORA_A_SUFFIX)}
    shared_path = path.parent / ("public_beta.safetensors" if mode in PUBLIC_MODES else "shared.safetensors")
    metadata = {"schema_version": bank.BANK_SCHEMA, "mode": mode}
    shapes = expected_lora_state_shapes(lora)
    selected_shapes = {key: shape for key, shape in shapes.items()
                       if mode in PUBLIC_MODES or key.endswith(LORA_A_SUFFIX)}
    if shared_path.exists():
        bank._factor_header(shared_path, selected_shapes, metadata=metadata)
    else:
        save_file(state, str(shared_path), metadata=metadata)
    if mode not in PUBLIC_MODES:
        compile_conditions(asset_root, spec, mode, checkpoint, training["source"], path.parent,
                           conditions, {key: shape for key, shape in shapes.items()
                                        if not key.endswith(LORA_A_SUFFIX)},
                           devices=execution_devices(device, devices),
                           frame_chunk=native_frame_chunk or spec["operator"]["frame_chunk"],
                           task_ids=tuple(spec["evaluation"]["task_ids"]), role="validation",
                           cpu_threads=cpu_threads)
    result = {**contract, "schema_version": bank.BANK_SCHEMA, "kind": bank.KIND, "status": "sealed",
              "asset_root": str(asset_root), "checkpoint_manifest": file_record(
                  checkpoint / "checkpoint_manifest.json"), "shared": file_record(shared_path),
              "conditions": conditions, "tasks": tasks, "scene_root": str(scenes),
              "information_wall": _wall(mode)}
    if mode in PUBLIC_MODES:
        result["public_intervention"] = _public_intervention(lora)
    write_json_atomic(path, result)
    return path


def inspect(bank: Mapping, path: Path, source: Mapping, task_keys: tuple,
            evaluation_role: str, require_formal: bool,
            task_init_state_ids: Mapping | None) -> dict:
    from . import bank as owner
    from ember.pi05_eval.scene import inspect_registered_scenes

    mode = bank["mode"]
    path = path.resolve()
    spec, training, spec_path = source_record(mode, Path(bank["checkpoint"]))
    tasks, conditions, scenes, states = _geometry(mode, spec, Path(bank["asset_root"]))
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank["asset_root"]) / spec["source"]["lora_contract"]), rank=128)
    shared = path.parent / ("public_beta.safetensors" if mode in PUBLIC_MODES else "shared.safetensors")
    expected = ((path, bank_path(mode).resolve()), (bank.get("joint_public_study"), True),
                (bank.get("schema_version"), owner.BANK_SCHEMA), (bank.get("kind"), owner.KIND),
                (bank.get("status"), "sealed"),
                (bank.get("spec"), file_record(spec_path)),
                (bank.get("training_git"), training["git"]["commit"]),
                (bank.get("training_run"), file_record(Path(bank["checkpoint"]).parent.parent / "run_contract.json")),
                (bank.get("checkpoint_manifest"), file_record(Path(bank["checkpoint"]) / "checkpoint_manifest.json")),
                (bank.get("source"), source), (source, training["source"]),
                (bank.get("lora"), lora.to_dict()), (training["lora"], lora.to_dict()),
                (bank.get("shared"), file_record(shared)), (bank.get("scene_root"), str(scenes)),
                (bank.get("tasks"), tasks), (bank.get("information_wall"), _wall(mode)),
                (evaluation_role, scope.ROLE if mode in PUBLIC_MODES else "validation"),
                (require_formal, True), (set(task_keys), {(row["suite"], row["task_id"]) for row in tasks}))
    git = bank.get("materialization_git", {})
    if (any(actual != wanted for actual, wanted in expected) or not git.get("commit")
            or git.get("branch") != "" or git.get("dirty_paths") != []
            or git.get("pushed_ref") != "origin/main"
            or task_init_state_ids is None or set(task_init_state_ids) != set(task_keys)
            or any(tuple(value) != states for value in task_init_state_ids.values())):
        raise ValueError("joint readout source, task, scene or state scope changed")
    contract = read_json(path.parent / "materialization_contract.json")
    if any(contract.get(key) != bank.get(key) for key in contract):
        raise ValueError("joint materialization lineage changed")
    shapes = expected_lora_state_shapes(lora)
    owner._factor_header(shared, {key: shape for key, shape in shapes.items()
                                 if mode in PUBLIC_MODES or key.endswith(LORA_A_SUFFIX)},
                         metadata={"schema_version": owner.BANK_SCHEMA, "mode": mode})
    if mode in PUBLIC_MODES:
        if bank.get("conditions") != conditions or bank.get("public_intervention") != _public_intervention(lora):
            raise ValueError("joint public teacher metadata or factor map changed")
    else:
        _inspect_conditions(bank, conditions, path, shapes)
    inspect_registered_scenes(scenes, tasks, states=states,
                              schema=("ember_operator_seen_task_scenes_v1" if mode in PUBLIC_MODES
                                      else "ember_demonstration_formal_scenes_v1"))
    return {**bank, "schema_version": owner.EVAL_SCHEMA, "arm": mode if mode in PUBLIC_MODES else "correct",
            "manifest": file_record(path), "scene_manifest": file_record(scenes / "manifest.json")}


def _inspect_conditions(bank: Mapping, conditions: list, path: Path, shapes: Mapping) -> None:
    from . import bank as owner

    if len(bank.get("conditions", ())) != 400:
        raise ValueError("joint full bank lost its 400 video conditions")
    b_shapes = {key: shape for key, shape in shapes.items() if not key.endswith(LORA_A_SUFFIX)}
    for row, wanted in zip(bank["conditions"], conditions, strict=True):
        factor = path.parent / f"{wanted['condition_id']}.safetensors"
        if (set(row) != set(wanted) | {"factors", "raw_frames", "sampled_frames"}
                or any(row.get(key) != value for key, value in wanted.items())
                or row["factors"] != file_record(factor)
                or not 0 < row["sampled_frames"] <= row["raw_frames"]):
            raise ValueError("joint full teacher factor provenance changed")
        owner._factor_header(factor, b_shapes, metadata={"schema_version": owner.BANK_SCHEMA,
                            "condition_id": wanted["condition_id"], "mode": bank["mode"]})


def capture_expectations(bank: Mapping, bank_path: Path, tasks: list,
                         output_dir: Path | None = None) -> dict:
    mode = bank["mode"]
    if mode not in MODES:
        raise ValueError("unregistered joint capture mode")
    expected_bank = study_root(mode) / mode / "banks/450/manifest.json"
    public = mode in PUBLIC_MODES
    expected_output = study_root(mode) / mode / "evaluation" / ("public144" if public else "correct400")
    states = scope.STATES if public else tuple(range(50))
    full_state = 32 if public else 0
    capture = capture_path(mode)
    registered = read_json(capture)
    full = [{"suite": row.suite, "task_id": row.task_id, "init_state_id": full_state} for row in tasks]
    if (bank.get("joint_public_study") is not True or bank_path.resolve() != expected_bank.resolve()
            or len(tasks) != (36 if public else 8) or full != registered["full_conditions"]
            or any(tuple(row.init_state_ids) != states for row in tasks)
            or output_dir is not None and output_dir.resolve() != expected_output.resolve()):
        raise ValueError("joint capture task/state/output scope changed")
    return dict(full=full, capture=capture, study=study_id(mode), output=expected_output,
                role=scope.ROLE if public else "validation", states=states,
                task_count=36 if public else 8, expected_bank=expected_bank)
