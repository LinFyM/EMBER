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
PUBLIC_VALIDATION_MODE = "context_public_validation"
MODES = ("joint", "joint_public", "T450_public", "context", "context_public", "context_seen", PUBLIC_VALIDATION_MODE)
PUBLIC_MODES = ("joint_public", "T450_public", "context_public", PUBLIC_VALIDATION_MODE)
OLD_CHECKPOINT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928"
                      "/continuation900/T/train/attempts/continuation/checkpoints/macro_00000450")
PUBLIC_SCENES = Path(scope.registration()["run_root"]) / "attempts/scene_canonical144/scenes"
FIXED_PANEL = Path("/data1/user/ymdai/ember_runs/operator_chain_diagnosis_20260929"
                   "/functional_credit_transport/group0")
SEEN_MODE = "context_seen"  # Evaluation-only identity; runtime model remains context.
SEEN_TASK = "operator_context900_seen_task_diagnosis_20261001"
SEEN_ROOT = ROOT.parent / SEEN_TASK
SEEN_TRAINING_GIT = "17ee3e387c778c189475898bcaf9f47e7bb79399"
SEEN_CHECKPOINT = (ROOT.parent / "operator_context_value_continuation900_20261001"
                   / "context/train/attempts/continuation/checkpoints/macro_00000900")
PUBLIC_VALIDATION_TASK = "operator_context900_public_validation_20261001"
PUBLIC_VALIDATION_ROOT = ROOT.parent / PUBLIC_VALIDATION_TASK
PUBLIC_FACTOR_MANIFEST = (ROOT.parent / "operator_context_value_continuation900_20261001"
                          / "context_public/banks/900/manifest.json")
TRAIN_TASKS = (0, 12, 20, 32)
TEACHERS = {0: (40, 11), 12: (25, 14), 20: (38, 42), 32: (17, 43)}


def _continuation_checkpoint(mode: str, checkpoint: Path | None) -> bool:
    from .joint_training import CONTEXT_CONTINUATION_ROOT

    return (mode in ("context", "context_public", SEEN_MODE, PUBLIC_VALIDATION_MODE) and checkpoint is not None
            and checkpoint.resolve().is_relative_to(CONTEXT_CONTINUATION_ROOT / "context/train/attempts"))


def study_root(mode: str, checkpoint: Path | None = None) -> Path:
    from .joint_training import CONTEXT_ROOT, CONTEXT_CONTINUATION_ROOT

    if mode not in MODES:
        raise ValueError("unregistered full/public study readout mode")
    if mode == SEEN_MODE:
        return SEEN_ROOT
    if mode == PUBLIC_VALIDATION_MODE:
        return PUBLIC_VALIDATION_ROOT
    if _continuation_checkpoint(mode, checkpoint):
        return CONTEXT_CONTINUATION_ROOT
    return CONTEXT_ROOT if mode in ("context", "context_public") else ROOT


def study_id(mode: str, checkpoint: Path | None = None) -> str:
    from .joint_training import CONTEXT_TASK, CONTEXT_CONTINUATION_TASK

    if mode == SEEN_MODE:
        return SEEN_TASK
    if mode == PUBLIC_VALIDATION_MODE:
        return PUBLIC_VALIDATION_TASK
    return CONTEXT_CONTINUATION_TASK if _continuation_checkpoint(mode, checkpoint) else (
        CONTEXT_TASK if mode in ("context", "context_public") else TASK)


def capture_path(mode: str, checkpoint: Path | None = None) -> Path:
    if mode == SEEN_MODE:
        return REPO / "configs/operator_read_write_v1/context900_seen_capture.json"
    if mode == PUBLIC_VALIDATION_MODE:
        return REPO / "configs/operator_read_write_v1/context900_public_validation_capture.json"
    prefix = "context_continuation" if _continuation_checkpoint(mode, checkpoint) else (
        "context" if mode in ("context", "context_public") else "joint")
    return REPO / "configs/operator_read_write_v1" / f"{prefix}_{'public' if mode in PUBLIC_MODES else 'official'}_capture.json"


def bank_path(mode: str, checkpoint: Path | None = None) -> Path:
    root = study_root(mode, checkpoint)
    macro = int(checkpoint.name.removeprefix("macro_")) if checkpoint is not None else 450
    if mode in (SEEN_MODE, PUBLIC_VALIDATION_MODE):
        if checkpoint is None or checkpoint.resolve() != SEEN_CHECKPOINT.resolve():
            raise ValueError("context seen144 requires the fixed actual Context900 checkpoint")
        return root / ("context" if mode == SEEN_MODE else "context_public") / "banks/900/manifest.json"
    allowed = (900,) if mode in PUBLIC_MODES else (810, 900)
    if macro not in (allowed if _continuation_checkpoint(mode, checkpoint) else (450,)):
        raise ValueError("readout checkpoint is outside registered main/conditional endpoint")
    return root / mode / f"banks/{macro}/manifest.json"


def evaluation_path(mode: str, checkpoint: Path) -> Path:
    if mode == SEEN_MODE:
        bank_path(mode, checkpoint)
        return SEEN_ROOT / "context/evaluation/correct144"
    if mode == PUBLIC_VALIDATION_MODE:
        bank_path(mode, checkpoint)
        return PUBLIC_VALIDATION_ROOT / "context_public/evaluation/public400"
    output = study_root(mode, checkpoint) / mode / "evaluation"
    if _continuation_checkpoint(mode, checkpoint):
        output /= checkpoint.name.removeprefix("macro_").lstrip("0")
    return output / ("public144" if mode in PUBLIC_MODES else "correct400")


def source_record(mode: str, checkpoint: Path) -> tuple[dict, dict, Path]:
    """Keep original training identity, bounded resume and current reader separate."""
    from . import bank, joint_training, run

    checkpoint = checkpoint.resolve()
    bank_path(mode, checkpoint)
    continuation = _continuation_checkpoint(mode, checkpoint)
    if continuation and checkpoint.name == "macro_00000810":
        trigger_path = joint_training.CONTEXT_CONTINUATION_ROOT / "launch/900_branch_trigger.json"
        if not trigger_path.is_file():
            raise ValueError("810 readout requires validated complete900 success strictly above153")
        trigger = read_json(trigger_path)
        expected = checkpoint.parent / "macro_00000900"
        if (trigger.get("status") != "validated" or trigger.get("successes", 0) <= 153
                or trigger.get("checkpoint") != str(expected)
                or trigger.get("results") != str(evaluation_path("context", expected) / "results.json")):
            raise ValueError("810 trigger is not the complete registered900 consumer")
    if mode == "T450_public":
        if checkpoint != OLD_CHECKPOINT.resolve():
            raise ValueError("T450 public readout changed its fixed old source")
        path = bank.CONTINUATION_FROZEN_SPEC_PATH
        spec = read_json(path)
        training = bank.inspect_training_source(spec, checkpoint, "T", sealed_evaluation=True)
    else:
        path = (run.CONTEXT_CONTINUATION_SPEC_PATH if continuation else
                run.CONTEXT_SPEC_PATH if mode in ("context", "context_public") else run.JOINT_SPEC_PATH)
        spec = run.specification(path)
        training = joint_training.inspect_source(spec, checkpoint)
        if mode in (SEEN_MODE, PUBLIC_VALIDATION_MODE) and training["git"]["commit"] != SEEN_TRAINING_GIT:
            raise ValueError("context900 diagnostic actual training identity changed")
    return spec, training, path


def _seen_geometry(mode: str) -> bool:
    return (mode in PUBLIC_MODES and mode != PUBLIC_VALIDATION_MODE) or mode == SEEN_MODE


def _geometry(mode: str, spec: Mapping, asset_root: Path) -> tuple[list, list, Path, tuple]:
    from . import bank

    if _seen_geometry(mode):
        tasks, conditions = scope.task_rows(asset_root, spec)
        return tasks, conditions, PUBLIC_SCENES, scope.STATES
    tasks, conditions = bank.task_rows(spec, asset_root)
    return tasks, conditions, bank.SCENE_ROOT, tuple(range(50))


def _wall(mode: str) -> dict:
    wall = {"teacher_video_values_read": 0 if mode in PUBLIC_MODES else 144 if mode == SEEN_MODE else 400,
            "teacher_runtime_reads": 0, "deployment_adapters": 1,
            "validation_test_gradients": False}
    if mode in PUBLIC_MODES:
        wall["video_id_role"] = "paired_metadata_only"
    return wall


def _public_intervention(lora) -> dict:
    return {"formula": "B0 A", "removed_term": "M(V,L) A",
            "factor_map": factor_map(lora)}


def _reused_public_factors(training: Mapping, lora) -> tuple[Path, dict]:
    """Reference the sealed76 factors without copying/exporting weights or videos."""
    from . import bank

    old = read_json(PUBLIC_FACTOR_MANIFEST)
    path = PUBLIC_FACTOR_MANIFEST.parent / "public_beta.safetensors"
    expected = ((old.get("status"), "sealed"), (old.get("mode"), "context_public"),
                (old.get("checkpoint"), str(SEEN_CHECKPOINT)),
                (old.get("training_git"), SEEN_TRAINING_GIT),
                (old.get("checkpoint_manifest"), file_record(SEEN_CHECKPOINT / "checkpoint_manifest.json")),
                (old.get("source"), training["source"]), (old.get("lora"), lora.to_dict()),
                (old.get("shared"), file_record(path)),
                (old.get("public_intervention"), _public_intervention(lora)),
                (old.get("information_wall"), _wall("context_public")))
    if any(actual != wanted for actual, wanted in expected):
        raise ValueError("fixed Context900 public factor source changed")
    bank._factor_header(path, expected_lora_state_shapes(lora),
                       metadata={"schema_version": bank.BANK_SCHEMA, "mode": "context_public"})
    return path, {"manifest": file_record(PUBLIC_FACTOR_MANIFEST),
                  "shared": file_record(path), "factor_mode": "context_public"}


def materialize(mode: str, checkpoint: Path, asset_root: Path, devices=None,
                native_frame_chunk: int | None = None, cpu_threads: int = 6,
                *, device=None, resume_materialization_git: str | None = None) -> Path:
    """Public export is CPU-only; full export uses the existing condition queue."""
    from . import bank, run
    from .materialization import compile_conditions, register_partial
    from ember.pi05_eval.scene import inspect_registered_scenes
    from ember.writer.materialization_workers import execution_devices

    path = bank_path(mode, checkpoint)
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
                              schema=("ember_operator_seen_task_scenes_v1" if _seen_geometry(mode)
                                      else "ember_demonstration_formal_scenes_v1"))
    if path.exists():
        raise ValueError("published joint readout bank already exists")
    contract = {"mode": mode, "joint_public_study": True, "checkpoint": str(checkpoint),
                "spec": file_record(spec_path), "training_git": training["git"]["commit"],
                "training_run": file_record(checkpoint.parent.parent / "run_contract.json"),
                "source": training["source"], "lora": lora.to_dict(),
                "materialization_git": run.frozen_git()}
    shared_path = path.parent / ("public_beta.safetensors" if mode in PUBLIC_MODES else "shared.safetensors")
    metadata = {"schema_version": bank.BANK_SCHEMA, "mode": mode}
    shapes = expected_lora_state_shapes(lora)
    selected_shapes = {key: shape for key, shape in shapes.items()
                       if mode in PUBLIC_MODES or key.endswith(LORA_A_SUFFIX)}
    if mode == PUBLIC_VALIDATION_MODE:
        shared_path, contract["public_factor_source"] = _reused_public_factors(training, lora)
    path.parent.mkdir(parents=True, exist_ok=True)
    register_partial(path.parent, contract, resume_materialization_git)
    if shared_path.exists():
        bank._factor_header(shared_path, selected_shapes, metadata={**metadata, "mode":
                           "context_public" if mode == PUBLIC_VALIDATION_MODE else mode})
    else:
        state = public_state(checkpoint, lora)
        if mode not in PUBLIC_MODES:
            state = {key: value for key, value in state.items() if key.endswith(LORA_A_SUFFIX)}
        save_file(state, str(shared_path), metadata=metadata)
    if mode not in PUBLIC_MODES:
        compile_conditions(asset_root, spec, mode, checkpoint, training["source"], path.parent,
                           conditions, {key: shape for key, shape in shapes.items()
                                        if not key.endswith(LORA_A_SUFFIX)},
                           devices=execution_devices(device, devices),
                           frame_chunk=native_frame_chunk or spec["operator"]["frame_chunk"],
                           task_ids=tuple(row["global_task_id"] for row in tasks),
                           role="train" if mode == SEEN_MODE else "validation",
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
    if mode == PUBLIC_VALIDATION_MODE:
        shared, provenance = _reused_public_factors(training, lora)
        if bank.get("public_factor_source") != provenance:
            raise ValueError("public validation reused factor provenance changed")
    expected = ((path, bank_path(mode, Path(bank["checkpoint"])).resolve()), (bank.get("joint_public_study"), True),
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
                (evaluation_role, scope.ROLE if _seen_geometry(mode) else "validation"),
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
                         metadata={"schema_version": owner.BANK_SCHEMA,
                                   "mode": "context_public" if mode == PUBLIC_VALIDATION_MODE else mode})
    if mode in PUBLIC_MODES:
        if bank.get("conditions") != conditions or bank.get("public_intervention") != _public_intervention(lora):
            raise ValueError("joint public teacher metadata or factor map changed")
    else:
        _inspect_conditions(bank, conditions, path, shapes)
    inspect_registered_scenes(scenes, tasks, states=states,
                              schema=("ember_operator_seen_task_scenes_v1" if _seen_geometry(mode)
                                      else "ember_demonstration_formal_scenes_v1"))
    return {**bank, "schema_version": owner.EVAL_SCHEMA, "arm": mode if mode in PUBLIC_MODES else "correct",
            "manifest": file_record(path), "scene_manifest": file_record(scenes / "manifest.json")}


def _inspect_conditions(bank: Mapping, conditions: list, path: Path, shapes: Mapping) -> None:
    from . import bank as owner

    if len(bank.get("conditions", ())) != len(conditions):
        raise ValueError("joint full bank lost its registered video conditions")
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


def capture_expectations(bank: Mapping, manifest_path: Path, tasks: list,
                         output_dir: Path | None = None) -> dict:
    mode = bank["mode"]
    if mode not in MODES:
        raise ValueError("unregistered joint capture mode")
    checkpoint = Path(bank["checkpoint"])
    expected_bank = bank_path(mode, checkpoint)
    expected_output = evaluation_path(mode, checkpoint)
    seen = _seen_geometry(mode)
    states = scope.STATES if seen else tuple(range(50))
    full_state = 32 if seen else 0
    capture = capture_path(mode, checkpoint)
    registered = read_json(capture)
    full = [{"suite": row.suite, "task_id": row.task_id, "init_state_id": full_state} for row in tasks]
    if (bank.get("joint_public_study") is not True or manifest_path.resolve() != expected_bank.resolve()
            or len(tasks) != (36 if seen else 8) or full != registered["full_conditions"]
            or any(tuple(row.init_state_ids) != states for row in tasks)
            or output_dir is not None and output_dir.resolve() != expected_output.resolve()):
        raise ValueError("joint capture task/state/output scope changed")
    return dict(full=full, capture=capture, study=study_id(mode, checkpoint), output=expected_output,
                role=scope.ROLE if seen else "validation", states=states,
                task_count=36 if seen else 8, expected_bank=expected_bank)
