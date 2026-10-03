"""Fixed T1800 public-factor export and bank inspection for the single beta diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from safetensors import safe_open
from safetensors.torch import save_file

from ember.lora import expected_lora_state_shapes, validate_lora_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import write_json_atomic
from ember.pi05_eval.scene import inspect_registered_scenes
from ember.writer.materialization import file_record

from . import bank as owner


def factor_map(lora) -> list[dict[str, str]]:
    return [{"lora_name": name, "ecp_name": f"common.values.{index}"}
            for index, name in enumerate(sorted(expected_lora_state_shapes(lora)))]


def intervention(lora) -> dict:
    return {"formula": "B0 A", "removed_term": "M(V,L) A",
            "weights": "same T1800 checkpoint public A/B0; no MT weights",
            "factor_map": factor_map(lora)}


def public_state(checkpoint: Path, lora) -> dict:
    shapes = expected_lora_state_shapes(lora)
    mapping = factor_map(lora)
    from .fixed_b_archive import weights_file

    with safe_open(str(weights_file(checkpoint)), framework="pt", device="cpu") as reader:
        if {name for name in reader.keys() if name.startswith("common.values.")} != {
                row["ecp_name"] for row in mapping}:
            raise ValueError("T1800 ECP public factor count changed")
        for row in mapping:
            part = reader.get_slice(row["ecp_name"])
            if (tuple(part.get_shape()) != tuple(shapes[row["lora_name"]])
                    or part.get_dtype() not in {"F32", "BF16"}):
                raise ValueError("T1800 ECP public factor shape or precision changed")
        state = {row["lora_name"]: reader.get_tensor(row["ecp_name"]).float().contiguous()
                 for row in mapping}
    validate_lora_state(state, lora)
    return state


def _fixed_checkpoint(spec: Mapping, checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    expected = Path(spec["run_root"]) / "T/train/attempts/continuation/checkpoints/macro_00001800"
    if checkpoint != expected.resolve():
        raise ValueError("public beta only reads the fixed complete T1800 ECP")
    return checkpoint


def materialize(checkpoint: Path, asset_root: Path) -> Path:
    spec = owner.specification(owner.CONTINUATION1800_SPEC_PATH)
    checkpoint = _fixed_checkpoint(spec, checkpoint)
    run = owner.inspect_training_source(spec, checkpoint, "T", sealed_evaluation=True)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        asset_root / spec["source"]["lora_contract"]), rank=128)
    if run["lora"] != lora.to_dict():
        raise ValueError("public beta LoRA contract differs from T1800")
    state = public_state(checkpoint, lora)
    tasks, conditions = owner.task_rows(spec, asset_root)
    output = owner.PUBLIC_BETA_ROOT / owner.PUBLIC_BETA_MODE / "banks/1800"
    output.mkdir(parents=True, exist_ok=False)
    shared_path = output / "public_beta.safetensors"
    save_file(state, str(shared_path), metadata={"schema_version": owner.PUBLIC_BETA_SCHEMA,
                                                "mode": owner.PUBLIC_BETA_MODE})
    bank = {"schema_version": owner.PUBLIC_BETA_SCHEMA, "kind": owner.KIND,
            "mode": owner.PUBLIC_BETA_MODE,
            "spec": file_record(owner.CONTINUATION1800_FROZEN_SPEC_PATH),
            "asset_root": str(asset_root.resolve()), "training_git": run["git"]["commit"],
            "checkpoint": str(checkpoint),
            "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
            "source": run["source"], "lora": lora.to_dict(),
            "shared": file_record(shared_path), "conditions": conditions,
            "tasks": tasks, "scene_root": str(owner.SCENE_ROOT),
            "public_intervention": intervention(lora),
            "information_wall": {"teacher_video_values_read": 0, "teacher_runtime_reads": 0,
                                 "video_id_role": "paired_metadata_only", "deployment_adapters": 1,
                                 "validation_test_gradients": False}}
    write_json_atomic(output / "manifest.json", bank)
    return output / "manifest.json"


def inspect(bank: Mapping, path: Path, source: Mapping, task_keys: tuple,
            evaluation_role: str, require_formal: bool,
            task_init_state_ids: Mapping | None) -> dict:
    spec = owner.specification(owner.CONTINUATION1800_SPEC_PATH)
    checkpoint = _fixed_checkpoint(spec, Path(bank["checkpoint"]))
    run = owner.inspect_training_source(spec, checkpoint, "T", sealed_evaluation=True)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank["asset_root"]) / spec["source"]["lora_contract"]), rank=128)
    tasks, conditions = owner.task_rows(spec, Path(bank["asset_root"]))
    expected_path = owner.PUBLIC_BETA_ROOT / owner.PUBLIC_BETA_MODE / "banks/1800/manifest.json"
    shared_path = path.parent / "public_beta.safetensors"
    expected = (
        (path, expected_path.resolve()), (bank.get("schema_version"), owner.PUBLIC_BETA_SCHEMA),
        (bank.get("kind"), owner.KIND), (bank.get("mode"), owner.PUBLIC_BETA_MODE),
        (bank.get("spec"), file_record(owner.CONTINUATION1800_FROZEN_SPEC_PATH)),
        (bank.get("training_git"), owner.CONTINUATION1800_TRAINING_GIT["commit"]),
        (bank.get("training_git"), run["git"]["commit"]),
        (bank.get("checkpoint_manifest"), file_record(checkpoint / "checkpoint_manifest.json")),
        (bank.get("source"), run["source"]), (bank.get("source"), source),
        (bank.get("lora"), run["lora"]), (bank.get("lora"), lora.to_dict()),
        (bank.get("shared"), file_record(shared_path)),
        (bank.get("conditions"), conditions), (bank.get("tasks"), tasks),
        (bank.get("scene_root"), str(owner.SCENE_ROOT)),
        (bank.get("public_intervention"), intervention(lora)),
        (bank.get("information_wall"), {"teacher_video_values_read": 0,
                                        "teacher_runtime_reads": 0,
                                        "video_id_role": "paired_metadata_only",
                                        "deployment_adapters": 1,
                                        "validation_test_gradients": False}),
        (evaluation_role, "validation"), (require_formal, True),
        (set(task_keys), {(row["suite"], row["task_id"]) for row in tasks}),
    )
    if (any(actual != wanted for actual, wanted in expected)
            or not owner.source_matches(bank["source"], source)
            or task_init_state_ids is not None and any(
                tuple(task_init_state_ids.get((row["suite"], row["task_id"]), ())) != tuple(range(50))
                for row in tasks)):
        raise ValueError("public beta T1800/source/scope identity changed")
    owner._factor_header(shared_path, expected_lora_state_shapes(lora),
                         metadata={"schema_version": owner.PUBLIC_BETA_SCHEMA,
                                   "mode": owner.PUBLIC_BETA_MODE})
    inspect_registered_scenes(owner.SCENE_ROOT, tasks)
    return {**bank, "schema_version": owner.EVAL_SCHEMA, "arm": owner.PUBLIC_BETA_MODE,
            "manifest": file_record(path),
            "scene_manifest": file_record(owner.SCENE_ROOT / "manifest.json")}
