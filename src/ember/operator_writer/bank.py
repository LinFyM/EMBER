"""Sealed T/U rank-128 banks and fixed MT step300 for official same-scene evaluation."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import (LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_,
                        expected_lora_state_shapes, identity_lora_state, inject_task_lora,
                        task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.scene import inspect_registered_scenes
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.task_protocol import load_task_authorities
from ember.writer.materialization import file_record, planned_episodes, selection_contract

from .data import (CONTINUATION1350_CHECKPOINTS, CONTINUATION1800_CHECKPOINTS,
                   CONTINUATION2340_CHECKPOINTS, PILOT_CHECKPOINTS, FormalData)
from .run import (CONTINUATION_SPEC_PATH, CONTINUATION1350_SPEC_PATH,
                  CONTINUATION1800_SPEC_PATH, CONTINUATION2340_SPEC_PATH,
                  PILOT_SPEC_PATH, PILOT_ROOT, PILOT_ARMS, REPO, SCHEMA,
                  SPEC_PATH, STAGE, build_runtime, complete_checkpoint, frozen_git, specification)
from . import scope as seen_scope


KIND = "operator_read_write_lora_bank"
BANK_SCHEMA = "ember_operator_read_write_bank_v1"
EVAL_SCHEMA = "ember_operator_read_write_eval_v1"
EPISODE_SCHEMA = "ember_operator_read_write_episode_v1"
PASSIVE_TAG = "ember_operator_read_write_passive_capture_v1"
PUBLIC_BETA_MODE = "public_beta"
PUBLIC_BETA_SCHEMA = "ember_operator_public_beta_bank_v1"
PUBLIC_BETA_STUDY = "operator_public_beta_diagnosis_20260929"
PUBLIC_BETA_ROOT = Path("/data1/user/ymdai/ember_runs/operator_public_beta_diagnosis_20260929")
SCENE_ROOT = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes")
SEALED_TRAINING_COMMIT = "784febbff32d991e53b9e5c6ba9f74683890425e"
SEALED_SPEC_PATH = Path("/data1/user/ymdai/projects/EMBER-operator-stage1-formal"
                        "/configs/operator_read_write_v1/learning_spec.json")
CONTINUATION_FROZEN_SPEC_PATH = Path(
    "/data1/user/ymdai/projects/EMBER-operator-continuation900-formal"
    "/configs/operator_read_write_v1/continuation900_spec.json")
CONTINUATION1350_FROZEN_SPEC_PATH = Path(
    "/data1/user/ymdai/projects/EMBER-operator-continuation1350-formal"
    "/configs/operator_read_write_v1/continuation1350_spec.json")
CONTINUATION1800_FROZEN_SPEC_PATH = Path(
    "/data1/user/ymdai/projects/EMBER-operator-continuation1800-formal"
    "/configs/operator_read_write_v1/continuation1800_spec.json")
PUBLIC_BETA_CAPTURE_PATH = REPO / "configs/operator_read_write_v1/public_beta_capture.json"
PILOT_CAPTURE_PATH = REPO / "configs/operator_read_write_v1/public_function_pilot_capture.json"
PILOT_FROZEN_SPEC_PATH = Path(
    "/data1/user/ymdai/projects/EMBER-operator-public-function-pilot-formal"
    "/configs/operator_read_write_v1/public_function_pilot_spec.json")
PILOT_TRAINING_GIT = {"commit": "9801641d0967e163d91474ff92e6fb6520be1084",
                      "branch": "", "dirty_paths": [],
                      "pushed_ref": "origin/codex/demonstration-transfer"}
CONTINUATION2340_FROZEN_SPEC_PATH = Path(
    "/data1/user/ymdai/projects/EMBER-operator-continuation2340-formal"
    "/configs/operator_read_write_v1/continuation2340_spec.json")
CONTINUATION2340_TRAINING_GIT = {
    "commit": "e2afbfd7c997e3f792921600608efa2fa3c1b25a",
    "branch": "", "dirty_paths": [],
    "pushed_ref": "origin/codex/demonstration-transfer"}
SEALED_TRAINING_GIT = {"commit": SEALED_TRAINING_COMMIT, "branch": "",
                       "dirty_paths": [], "pushed_ref": "origin/main"}
CONTINUATION_TRAINING_GIT = {"commit": "81846ed35933222b14ac693a0b760268ecff7f17",
                             "branch": "", "dirty_paths": [],
                             "pushed_ref": "origin/codex/demonstration-transfer"}
CONTINUATION1350_TRAINING_GIT = {
    "commit": "14bac4cdd6c27eee06f5574317da8257834e3884",
    "branch": "", "dirty_paths": [],
    "pushed_ref": "origin/codex/demonstration-transfer"}
CONTINUATION1800_TRAINING_GIT = {
    "commit": "fcc23cd15cc475530c385e354670efee6bacfa12",
    "branch": "", "dirty_paths": [],
    "pushed_ref": "origin/codex/demonstration-transfer"}
CONTINUATION_EVALUATION_MACROS = (450, 810, 900)
CONTINUATION1350_EVALUATION_MACROS = (1080, 1350)
CONTINUATION1800_EVALUATION_MACROS = (1710, 1800)
CONTINUATION2340_EVALUATION_MACROS = CONTINUATION2340_CHECKPOINTS
EVALUATION_SPEC_PATHS = {
    **{macro: CONTINUATION_SPEC_PATH for macro in CONTINUATION_EVALUATION_MACROS},
    **{macro: CONTINUATION1350_SPEC_PATH for macro in CONTINUATION1350_EVALUATION_MACROS},
    **{macro: CONTINUATION1800_SPEC_PATH for macro in CONTINUATION1800_EVALUATION_MACROS},
    **{macro: PILOT_SPEC_PATH for macro in PILOT_CHECKPOINTS},
    **{macro: CONTINUATION2340_SPEC_PATH for macro in CONTINUATION2340_EVALUATION_MACROS},
}


def source_matches(left: Mapping, right: Mapping) -> bool:
    return all(left.get(key) and right.get(key) and Path(left[key]).resolve() == Path(right[key]).resolve()
               for key in ("source_run", "checkpoint", "model_path"))


def selection(spec: Mapping) -> dict:
    return selection_contract(role="validation", task_ids=spec["evaluation"]["task_ids"],
                              cardinality=1, arm="correct", mode="per_init_ordinal",
                              seed=spec["evaluation"]["video_schedule_seed"],
                              init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))


def task_rows(spec: Mapping, asset_root: Path) -> tuple[list[dict], list[dict]]:
    _, authority = load_task_authorities(asset_root, spec["source"]["data_protocol"])
    entries = {int(row["global_task_id"]): row for row in authority["tasks"]}
    selected, conditions = [], []
    for task_id in spec["evaluation"]["task_ids"]:
        original = entries[task_id]
        if original["split_role"] != "validation":
            raise ValueError("bank task crosses the validation information wall")
        episodes = planned_episodes(selection(spec), task_id)
        if len(episodes) != 50 or len({row["teacher_demo_indices"][0] for row in episodes}) != 50:
            raise ValueError("official K1 schedule repeats a teacher video")
        selected.append({"global_task_id": task_id, "suite": original["suite"],
                         "task_id": original["task_id"], "language": original["language"],
                         "split_role": "validation", "episodes": episodes})
        for episode in episodes:
            conditions.append({"condition_id": episode["condition_id"],
                               "global_task_id": task_id,
                               "teacher_demo": episode["teacher_demo_indices"][0]})
    return selected, conditions


def _factor_header(path: Path, shapes: Mapping[str, tuple], *, metadata: dict | None = None,
                   dtype: str | None = "F32") -> None:
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        if (set(reader.keys()) != set(shapes)
                or metadata is not None and reader.metadata() != metadata
                or any(tuple(reader.get_slice(name).get_shape()) != tuple(shape)
                       or dtype is not None and reader.get_slice(name).get_dtype() != dtype
                       for name, shape in shapes.items())):
            raise ValueError("sealed operator factor header/shape/dtype changed")


def assemble_state(shared: Mapping[str, torch.Tensor], conditional: Mapping[str, torch.Tensor],
                   contract) -> dict[str, torch.Tensor]:
    """Install one full adapter: public A and this condition's final B0+M."""
    shapes = expected_lora_state_shapes(contract)
    if (set(shared) != {name for name in shapes if name.endswith(LORA_A_SUFFIX)}
            or set(conditional) != {name for name in shapes if name.endswith(LORA_B_SUFFIX)}
            or any(value.dtype != torch.float32 for value in (*shared.values(), *conditional.values()))):
        raise ValueError("operator bank must contain one complete FP32 A/B state")
    result = {**shared, **conditional}
    validate_lora_state(result, contract)
    return result


def _mt_source(spec: Mapping, source: Mapping) -> tuple[Path, dict, dict]:
    checkpoint = Path(spec["evaluation"]["mt_checkpoint"]).resolve()
    expected = Path("/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc"
                    "/checkpoints/step_00000300/lora.safetensors")
    if checkpoint != expected or not checkpoint.is_file():
        raise ValueError("fixed historical coverage36 MT step300 source changed")
    root = checkpoint.parent.parent.parent
    run = read_json(root / "run_contract.json")
    manifest = read_json(checkpoint.parent / "checkpoint_manifest.json")
    wanted_tasks = specification()["events"]["task_ids"]
    facts = (
        (run.get("schema_version"), "ember_pi05_source_sft_launch_v3"),
        (run.get("stage"), "development"),
        (run["adapter"].get("kind"), "one_shared_multitask_pi05_lora"),
        (run["adapter"].get("contract"), "configs/pi05_lora_rank128_aligned.json"),
        (run["adapter"].get("per_task_adapter"), False),
        (run["adapter"].get("stacked_shared_source_adapter"), False),
        (run["optimization"].get("normalization"), "frozen filtered-LIBERO-90 source normalization"),
        (run["optimization"].get("precision"), "bfloat16"),
        # Preserve the old MT authority even though today's config file has a later identity.
        (run["authorities"]["source_base_config"].get("sha256"),
         "be2c885b4f76fef68bc0d5e2da1a8aa0a47640673ff77c18a1bc2d19566f8dda"),
        (run["authorities"]["lora_contract"].get("sha256"),
         "379ee28989f55ef571f82d9999a5a1c0c6e53906c8297dd216b79d88b814da6f"),
        (run.get("source"), source),
        ([row["global_task_id"] for row in run["tasks"]], wanted_tasks),
        (run["stage_contract"].get("task_count"), 36),
        (run["data"].get("action_start_offset"), 1),
        (manifest.get("schema_version"), "ember_pi05_source_sft_checkpoint_v5"),
        (manifest.get("consumed", {}).get("next_optimizer_step"), 300),
        (manifest.get("files", {}).get("lora.safetensors", {}).get("bytes"), checkpoint.stat().st_size),
    )
    if any(actual != wanted for actual, wanted in facts) or any(
            row["split_role"] != "train" for row in run["tasks"]):
        raise ValueError("fixed MT run/step/source/normalization/task provenance changed")
    base = load_pi05_lora_contract(REPO / run["adapter"]["contract"])
    current = derive_pi05_lora_rank(load_pi05_lora_contract(
        REPO / spec["source"]["lora_contract"]), rank=128)
    if ((base.rank, base.alpha, len(base.targets)) != (128, 128, 38)
            or expected_lora_state_shapes(base) != expected_lora_state_shapes(current)
            or base.alpha != current.alpha):
        raise ValueError("historical MT LoRA rank/alpha/targets changed")
    _factor_header(checkpoint, expected_lora_state_shapes(base), dtype=None)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
        if any(reader.get_slice(name).get_dtype() != (
                "F32" if name.startswith(("model.action_in_proj.", "model.action_out_proj.")) else "BF16")
               for name in reader.keys()):
            raise ValueError("historical MT factor precision changed")
    return checkpoint, run, manifest


def inspect_training_source(spec: Mapping, checkpoint: Path, mode: str, *, sealed_evaluation: bool = False) -> dict:
    if checkpoint.name in {f"macro_{macro:08d}" for macro in
                           (*CONTINUATION_EVALUATION_MACROS, *CONTINUATION1350_EVALUATION_MACROS,
                            *CONTINUATION1800_EVALUATION_MACROS, *PILOT_CHECKPOINTS,
                            *CONTINUATION2340_EVALUATION_MACROS)}:
        return _inspect_continuation_source(spec, checkpoint, mode,
                                            sealed_evaluation=sealed_evaluation)
    if mode not in ("T", "U") or checkpoint.name != "macro_00000270":
        raise ValueError("only formal T/U270 may materialize")
    output = checkpoint.parent.parent
    expected_root = Path(spec["run_root"]) / mode / "train" / "attempts"
    if output.parent.resolve() != expected_root.resolve():
        raise ValueError("operator bank checkpoint is outside this arm's formal attempts")
    run = read_json(output / "run_contract.json")
    complete = read_json(output / "completion.json")
    ecp = read_json(checkpoint / "checkpoint_manifest.json")
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True,
                         weights_only=True)
    metrics = (output / "metrics.jsonl").read_text().splitlines()
    expected = (
        (run.get("schema_version"), SCHEMA), (run.get("stage"), STAGE),
        (run.get("mode"), mode),
        (run.get("git"), SEALED_TRAINING_GIT if sealed_evaluation else frozen_git()),
        (run.get("spec"), str(SEALED_SPEC_PATH if sealed_evaluation else SPEC_PATH)),
        (run.get("operator"), spec["operator"]),
        (run.get("events"), spec["events"]), (run.get("optimizer"), spec["optimization"]),
        (run.get("source_trainable"), 0),
        (complete.get("updates"), 270), (complete.get("metrics_rows"), 270),
        (complete.get("checkpoint"), str(checkpoint)),
        (ecp.get("stage"), STAGE), (ecp.get("run_contract_schema"), SCHEMA),
        (ecp.get("next_macro"), 270), (ecp.get("world_size"), 2),
        (trainer.get("schema_version"), "ember_ecp_checkpoint_v1"),
        (trainer.get("stage"), STAGE), (trainer.get("next_macro"), 270),
        (trainer.get("metrics_rows"), 270),
        (trainer.get("training_state"), {"updates": 270, "mode": mode}),
        (trainer.get("sampler_state", {}).get("next_step"), 270),
        ({k: v for k, v in trainer.get("sampler_state", {}).items() if k != "next_step"}, run["sampler"]),
    )
    if (not complete_checkpoint(checkpoint) or any(actual != wanted for actual, wanted in expected)
            or len(metrics) != 270
            or [row["update"] for row in map(json.loads, metrics)] != list(range(1, 271))):
        raise ValueError("operator bank training run/ECP/source numerical identity changed")
    return run


def _continuation_source_identity(spec: Mapping, macro: int, sealed_evaluation: bool) -> tuple:
    window = (4 if macro in CONTINUATION2340_EVALUATION_MACROS else
              3 if macro in PILOT_CHECKPOINTS else
              2 if macro in CONTINUATION1800_EVALUATION_MACROS else
              1 if macro in CONTINUATION1350_EVALUATION_MACROS else 0)
    current_specs = (CONTINUATION_SPEC_PATH, CONTINUATION1350_SPEC_PATH,
                     CONTINUATION1800_SPEC_PATH, PILOT_SPEC_PATH,
                     CONTINUATION2340_SPEC_PATH)
    source_specs = (CONTINUATION_FROZEN_SPEC_PATH, CONTINUATION1350_FROZEN_SPEC_PATH,
                    CONTINUATION1800_FROZEN_SPEC_PATH, PILOT_FROZEN_SPEC_PATH,
                    CONTINUATION2340_FROZEN_SPEC_PATH)
    wanted_spec_path = source_specs[window] if sealed_evaluation else current_specs[window]
    teacher_rounds = [[20260928, 1, "task"]] + [
        [20260928, 1, "task", index] for index in range(1, window + 2)]
    expected_sampler = {"schema_version": spec["events"]["schema_version"],
                        "seed": spec["events"]["seed"], "tasks": spec["events"]["task_ids"],
                        "demo_pool": [0, 49], "query_offset": 1, "queries_per_task": 28,
                        "teacher_rounds": teacher_rounds,
                        "teacher_visits_per_round": 50, "teacher_demo_pool": list(range(50))}
    allowed_parents = ((270, 360, 450, 540, 630, 720, 810),
                       (900, *CONTINUATION1350_CHECKPOINTS[:-1]),
                       (1350, *CONTINUATION1800_CHECKPOINTS[:-1]), (1800,),
                       (1890, *CONTINUATION2340_CHECKPOINTS[:-1]))[window]
    sealed_git = (CONTINUATION_TRAINING_GIT, CONTINUATION1350_TRAINING_GIT,
                  CONTINUATION1800_TRAINING_GIT, PILOT_TRAINING_GIT,
                  CONTINUATION2340_TRAINING_GIT)[window]
    wanted_git = sealed_git if sealed_evaluation else frozen_git(continuation=True)
    return window, wanted_spec_path, expected_sampler, allowed_parents, wanted_git


def _inspect_continuation_source(spec: Mapping, checkpoint: Path, mode: str, *,
                                 sealed_evaluation: bool) -> dict:
    macro = int(checkpoint.name.split("_")[-1])
    window, wanted_spec_path, expected_sampler, allowed_parents, wanted_git = (
        _continuation_source_identity(spec, macro, sealed_evaluation))
    output = checkpoint.parent.parent
    run = read_json(output / "run_contract.json")
    pilot = window == 3
    arm = run.get("pilot_arm") if pilot else mode
    expected_root = Path(spec["run_root"]) / arm / "train/attempts"
    if (mode not in spec["execution"]["modes"] or macro not in (*CONTINUATION_EVALUATION_MACROS,
                                             *CONTINUATION1350_EVALUATION_MACROS,
                                             *CONTINUATION1800_EVALUATION_MACROS,
                                             *PILOT_CHECKPOINTS,
                                             *CONTINUATION2340_EVALUATION_MACROS)
            or pilot and (arm not in PILOT_ARMS or run.get("loss_variant") != PILOT_ARMS[arm]
                          or run.get("pilot") != spec["pilot"])
            or output.parent.resolve() != expected_root.resolve()
            or not complete_checkpoint(checkpoint)):
        raise ValueError("continuation bank requires complete same-arm registered ECP")
    ecp = read_json(checkpoint / "checkpoint_manifest.json")
    trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True,
                         weights_only=True)
    parent = Path(run.get("parent_checkpoint", ""))
    resume = read_json(output / "resume_provenance.json")
    metrics = (output / "metrics.jsonl").read_text().splitlines()[:macro]
    parent_macro = int(parent.name.split("_")[-1]) if parent.name.startswith("macro_") else -1
    source_parent_macro = (270, 900, 1350, 1800, 1890)[window]
    parent_root = (Path(spec["continuation"]["parent_run_root"])
                   if parent_macro == source_parent_macro else Path(spec["run_root"]))
    parent_arm = (spec["continuation"].get("parent_arm", mode)
                  if parent_macro == source_parent_macro else arm)
    expected_parent = parent_root / parent_arm / "train/attempts"
    parent_run = read_json(parent.parent.parent / "run_contract.json")
    parent_training_git = (spec["continuation"]["parent_training_git"]
                           if parent_macro == source_parent_macro else wanted_git["commit"])
    expected = (
        (run.get("schema_version"), SCHEMA), (run.get("stage"), STAGE), (run.get("mode"), mode),
        (run.get("spec"), str(wanted_spec_path)), (run.get("source_trainable"), 0),
        (run.get("source"), parent_run.get("source")), (run.get("lora"), parent_run.get("lora")),
        (run.get("operator"), spec["operator"]), (run.get("optimizer"), spec["optimization"]),
        (run.get("events"), spec["events"]), (run.get("continuation"), spec["continuation"]),
        (run.get("git"), wanted_git),
        (ecp.get("stage"), STAGE), (ecp.get("run_contract_schema"), SCHEMA),
        (ecp.get("next_macro"), macro), (ecp.get("world_size") in (2, 3, 4), True),
        (trainer.get("schema_version"), "ember_ecp_checkpoint_v1"),
        (trainer.get("stage"), STAGE), (trainer.get("next_macro"), macro),
        (trainer.get("metrics_rows"), macro),
        (bool(trainer.get("optimizer", {}).get("param_groups")), True),
        (trainer.get("scheduler", {}).get("last_epoch"), macro),
        (trainer.get("scaler"), None),
        (trainer.get("training_state"), {"updates": macro, "mode": mode,
                                         **({"pilot_arm": arm, "loss_variant": PILOT_ARMS[arm]}
                                            if pilot else
                                            {"loss_variant": "full"} if window == 4 else {})}),
        (trainer.get("sampler_state", {}).get("next_step"), macro),
        ({k: v for k, v in trainer.get("sampler_state", {}).items() if k != "next_step"}, run["sampler"]),
        (run.get("sampler"), expected_sampler),
        (parent_macro in allowed_parents and parent_macro < macro, True),
        (parent.parent.parent.parent.resolve(), expected_parent.resolve()),
        (parent.is_dir() and complete_checkpoint(parent), True),
        (resume.get("checkpoint"), str(parent)),
        (resume.get("parent_git", {}).get("commit") if window >= 3 else None,
         parent_training_git if window >= 3 else None),
        (parent_run.get("git", {}).get("commit") if window >= 3 else None,
         parent_training_git if window >= 3 else None),
        (run.get("loss_variant") if window == 4 else None,
         "full" if window == 4 else None),
        (run.get("pilot_arm") if window == 4 else None, None),
    )
    parsed_metrics = list(map(json.loads, metrics))
    if (any(actual != wanted for actual, wanted in expected) or len(metrics) != macro
            or [row["update"] for row in parsed_metrics] != list(range(1, macro + 1))
            or pilot and any(row.get("pilot_arm") != arm or row.get("loss_variant") != PILOT_ARMS[arm]
                             for row in parsed_metrics[1800:])
            or window == 4 and any(row.get("loss_variant") != "full"
                                   for row in parsed_metrics[1890:])):
        raise ValueError("continuation bank source/ECP/optimizer/sampler provenance changed")
    if window and parent_macro == source_parent_macro:
        old_spec_path = (CONTINUATION_SPEC_PATH, CONTINUATION1350_SPEC_PATH,
                         CONTINUATION1800_SPEC_PATH, PILOT_SPEC_PATH)[window - 1]
        old = inspect_training_source(specification(old_spec_path), parent, mode,
                                      sealed_evaluation=True)
        if any(old.get(key) != run.get(key) for key in ("source", "lora", "operator", "optimizer",
                                                     "trainable_names", "source_trainable",
                                                     "information_wall")):
            raise ValueError("continuation bank parent scientific source changed")
    return run


def _require_pilot_source(mode: str, run: Mapping) -> None:
    if mode in PILOT_ARMS and (run.get("pilot_arm") != mode
                              or run.get("loss_variant") != PILOT_ARMS[mode]):
        raise ValueError("pilot bank arm or loss source changed")


def _write_condition_factors(runtime, data: FormalData, output: Path,
                             conditions: list[dict], b_shapes: Mapping,
                             *, mode: str, frame_chunk: int) -> None:
    try:
        for condition in conditions:
            path = output / f"{condition['condition_id']}.safetensors"
            raw, sampled = data.videos.frame_counts(condition["global_task_id"],
                                                     condition["teacher_demo"])
            if path.exists():
                _factor_header(path, b_shapes,
                               metadata={"schema_version": BANK_SCHEMA,
                                         "condition_id": condition["condition_id"], "mode": mode})
            else:
                pixels, raw, sampled = data.condition(runtime, condition["global_task_id"],
                                                      condition["teacher_demo"])
                with torch.no_grad():
                    state, _ = runtime.compile(pixels, frame_chunk=frame_chunk)
                factors = {name: value.detach().float().cpu().contiguous()
                           for name, value in state.items() if name.endswith(LORA_B_SUFFIX)}
                if len(factors) != 38:
                    raise ValueError("video value did not produce a complete B0+M")
                save_file(factors, str(path), metadata={"schema_version": BANK_SCHEMA,
                                                       "condition_id": condition["condition_id"],
                                                       "mode": mode})
            condition.update(factors=file_record(path), raw_frames=raw, sampled_frames=sampled)
    finally:
        data.close()


def materialize(mode: str, checkpoint: Path, asset_root: Path, device: torch.device,
                *, seen_task: bool = False) -> Path:
    checkpoint = checkpoint.resolve()
    macro = int(checkpoint.name.split("_")[-1]) if checkpoint.name.startswith("macro_") else -1
    pilot = mode in PILOT_ARMS and macro in PILOT_CHECKPOINTS
    if seen_task and (mode != "T" or checkpoint != Path(seen_scope.registration()["checkpoint_t"]).resolve()):
        raise ValueError("seen-task materialization requires the fixed T1800 ECP")
    next_window = mode == "T" and macro in CONTINUATION2340_EVALUATION_MACROS
    if not seen_task and not pilot and not next_window and (mode != "T" or macro not in CONTINUATION1800_EVALUATION_MACROS):
        raise ValueError("new materialization requires T1710/1800/1980..2340 or pilot1890 ECP")
    spec_path = (CONTINUATION1800_FROZEN_SPEC_PATH if seen_task else
                 CONTINUATION2340_FROZEN_SPEC_PATH if next_window else
                 EVALUATION_SPEC_PATHS.get(macro, SPEC_PATH))
    spec = read_json(spec_path) if seen_task or next_window else specification(spec_path)
    if next_window and spec != specification(CONTINUATION2340_SPEC_PATH):
        raise ValueError("2340 materialization differs from its frozen training spec")
    source_spec_path = spec_path
    run = inspect_training_source(spec, checkpoint, "T" if pilot else mode,
                                  sealed_evaluation=seen_task or next_window)
    _require_pilot_source(mode, run)
    output = (Path(seen_scope.registration()["run_root"]) if seen_task else
              Path(spec["run_root"])) / mode / "banks" / str(macro)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        asset_root / spec["source"]["lora_contract"]), rank=128)
    if run["lora"] != lora.to_dict():
        raise ValueError("operator bank LoRA contract changed")
    if (output / "manifest.json").exists() or (output.exists() and not
                                              (output / "materialization_contract.json").is_file()):
        raise ValueError("published or unregistered operator bank output exists")
    runtime = build_runtime(asset_root, spec, device, "T" if pilot else mode)
    if runtime.source != run["source"]:
        raise ValueError("materialization source differs from the formal training run")
    contract = {"mode": mode, "checkpoint": str(checkpoint), "spec": file_record(source_spec_path),
                "training_git": run["git"]["commit"], "source": run["source"],
                "lora": lora.to_dict(),
                **({"materialization_git": frozen_git(continuation=True)}
                   if seen_task or next_window else {}),
                **({"evaluation_scope": file_record(seen_scope.PATH)} if seen_task else {}),
                **({"loss_variant": "full"} if next_window else {}),
                **({"loss_variant": PILOT_ARMS[mode]} if pilot else {})}
    output.mkdir(parents=True, exist_ok=True)
    registration = output / "materialization_contract.json"
    if registration.exists():
        if read_json(registration) != contract:
            raise ValueError("partial bank belongs to a different formal source")
    else:
        write_json_atomic(registration, contract)
    runtime.writer.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(device)), strict=True)
    runtime.writer.eval()
    shared = {name: value.detach().float().cpu().contiguous()
              for name, value in runtime.writer.public_state().items() if name.endswith(LORA_A_SUFFIX)}
    if len(shared) != 38:
        raise ValueError("public A is not the complete 38-target shared factor")
    shapes = expected_lora_state_shapes(lora)
    a_shapes = {name: shape for name, shape in shapes.items() if name.endswith(LORA_A_SUFFIX)}
    b_shapes = {name: shape for name, shape in shapes.items() if name.endswith(LORA_B_SUFFIX)}
    shared_path = output / "shared.safetensors"
    if shared_path.exists():
        _factor_header(shared_path, a_shapes,
                       metadata={"schema_version": BANK_SCHEMA, "mode": mode})
    else:
        save_file(shared, str(shared_path), metadata={"schema_version": BANK_SCHEMA, "mode": mode})
    tasks, conditions = (seen_scope.task_rows(asset_root, spec) if seen_task else
                         task_rows(spec, asset_root))
    data = FormalData(asset_root, spec, query_labels=False,
                      task_ids=tuple(seen_scope.registration()["global_task_ids"] if seen_task else
                                     spec["evaluation"]["task_ids"]),
                      role="train" if seen_task else "validation")
    _write_condition_factors(runtime, data, output, conditions, b_shapes,
                             mode=mode, frame_chunk=spec["operator"]["frame_chunk"])
    bank = {"schema_version": BANK_SCHEMA, "kind": KIND, "mode": mode,
            "spec": file_record(source_spec_path), "asset_root": str(asset_root),
            "training_git": run["git"]["commit"], "checkpoint": str(checkpoint),
            "checkpoint_manifest": file_record(checkpoint / "checkpoint_manifest.json"),
            "source": run["source"], "lora": lora.to_dict(),
            "shared": file_record(output / "shared.safetensors"), "conditions": conditions,
            "tasks": tasks, "scene_root": str(Path(seen_scope.registration()["run_root"]) / "scenes")
            if seen_task else str(SCENE_ROOT),
            "information_wall": {"teacher_video_values_read": 144 if seen_task else 400,
                                 "teacher_runtime_reads": 0, "deployment_adapters": 1,
                                 "validation_test_gradients": False}}
    if seen_task:
        bank["evaluation_scope"] = file_record(seen_scope.PATH)
        bank["materialization_git"] = contract["materialization_git"]
    if next_window:
        bank["materialization_git"] = contract["materialization_git"]
        bank["loss_variant"] = "full"
    if pilot:
        bank.update(loss_variant=PILOT_ARMS[mode], pilot=spec["pilot"],
                    parent_checkpoint=run["parent_checkpoint"])
    write_json_atomic(output / "manifest.json", bank)
    return output / "manifest.json"


def materialize_public_beta(checkpoint: Path, asset_root: Path) -> Path:
    from .public_beta import materialize

    return materialize(checkpoint, asset_root)


def register_mt(asset_root: Path, source: Mapping, *, seen_task: bool = False) -> Path:
    spec = read_json(SEALED_SPEC_PATH) if seen_task else specification()
    checkpoint, run, _ = _mt_source(spec, source)
    if seen_task and checkpoint != Path(seen_scope.registration()["checkpoint_mt"]).resolve():
        raise ValueError("seen-task MT source changed")
    output = (Path(seen_scope.registration()["run_root"]) if seen_task else
              Path(spec["run_root"])) / "MT" / "banks" / "300"
    output.mkdir(parents=True, exist_ok=False)
    tasks, conditions = (seen_scope.task_rows(asset_root, spec) if seen_task else
                         task_rows(spec, asset_root))
    bank = {"schema_version": BANK_SCHEMA, "kind": KIND, "mode": "MT",
            "spec": file_record(SEALED_SPEC_PATH if seen_task else SPEC_PATH),
            "asset_root": str(asset_root),
            "training_git": run["git"]["commit"], "checkpoint": str(checkpoint),
            "checkpoint_manifest": file_record(checkpoint.parent / "checkpoint_manifest.json"),
            "source": source, "lora": load_pi05_lora_contract(REPO / run["adapter"]["contract"]).to_dict(),
            "shared": file_record(checkpoint), "conditions": conditions,
            "tasks": tasks, "scene_root": str(Path(seen_scope.registration()["run_root"]) / "scenes")
            if seen_task else str(SCENE_ROOT),
            "information_wall": {"teacher_video_values_read": 0,
                                 "teacher_runtime_reads": 0, "deployment_adapters": 1,
                                 "validation_test_gradients": False}}
    if seen_task:
        bank["evaluation_scope"] = file_record(seen_scope.PATH)
        bank["materialization_git"] = frozen_git(continuation=True)
    write_json_atomic(output / "manifest.json", bank)
    return output / "manifest.json"


def _inspect_scope(bank: Mapping, spec: Mapping, path: Path, source: Mapping, task_keys: tuple,
                   evaluation_role: str, require_formal: bool,
                   task_init_state_ids: Mapping | None) -> None:
    seen_scope.inspect_official_scope(bank, spec, path, source, task_keys,
                                      evaluation_role, require_formal,
                                      task_init_state_ids)


def _inspect_mt_bank(bank: Mapping, spec: Mapping, source: Mapping) -> None:
    checkpoint, run, _ = _mt_source(spec, source)
    expected = (
        (bank["checkpoint"], str(checkpoint)), (bank["shared"], file_record(checkpoint)),
        (bank["training_git"], run["git"]["commit"]),
        (bank["checkpoint_manifest"], file_record(checkpoint.parent / "checkpoint_manifest.json")),
        (bank["lora"], load_pi05_lora_contract(REPO / run["adapter"]["contract"]).to_dict()),
    )
    if (any(actual != wanted for actual, wanted in expected)
            or any(set(row) != {"condition_id", "global_task_id", "teacher_demo"}
                   for row in bank["conditions"])):
        raise ValueError("fixed MT source/bank changed")


def _inspect_tu_bank(bank: Mapping, spec: Mapping, path: Path) -> None:
    mode, checkpoint = bank["mode"], Path(bank["checkpoint"])
    run = inspect_training_source(spec, checkpoint, "T" if mode in PILOT_ARMS else mode,
                                  sealed_evaluation=True)
    _require_pilot_source(mode, run)
    base = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank["asset_root"]) / spec["source"]["lora_contract"]), rank=128)
    shapes = expected_lora_state_shapes(base)
    expected = (
        (bank["shared"], file_record(path.parent / "shared.safetensors")),
        (bank["checkpoint_manifest"], file_record(checkpoint / "checkpoint_manifest.json")),
        (bank["training_git"], run["git"]["commit"]), (bank["lora"], base.to_dict()),
        (run["lora"], base.to_dict()), (run["source"], bank["source"]),
    )
    if any(actual != wanted for actual, wanted in expected):
        raise ValueError("T/U shared factor or formal checkpoint changed")
    _factor_header(path.parent / "shared.safetensors",
                   {name: shape for name, shape in shapes.items() if name.endswith(LORA_A_SUFFIX)},
                   metadata={"schema_version": BANK_SCHEMA, "mode": mode})
    b_shapes = {name: shape for name, shape in shapes.items() if name.endswith(LORA_B_SUFFIX)}
    for row in bank["conditions"]:
        factor = path.parent / f"{row['condition_id']}.safetensors"
        if (set(row) != {"condition_id", "global_task_id", "teacher_demo", "factors",
                        "raw_frames", "sampled_frames"}
                or row["factors"] != file_record(factor)
                or not 0 < row["sampled_frames"] <= row["raw_frames"]):
            raise ValueError("operator B0+M file or video provenance changed")
        _factor_header(factor, b_shapes, metadata={"schema_version": BANK_SCHEMA,
                                                "condition_id": row["condition_id"], "mode": mode})


def inspect_bank(*, manifest_path: Path, source: Mapping, task_keys: tuple,
                 evaluation_role: str, require_formal: bool,
                 task_init_state_ids: Mapping | None = None) -> dict:
    try:
        path = manifest_path.resolve()
        bank = read_json(path)
        if bank.get("mode") == PUBLIC_BETA_MODE:
            from .public_beta import inspect

            return inspect(bank, path, source, task_keys, evaluation_role,
                           require_formal, task_init_state_ids)
        spec = read_json(Path(bank["spec"]["path"]))
        if bank.get("evaluation_scope") is not None:
            scene_root = seen_scope.inspect_bank_scope(bank, spec, path, source, task_keys,
                                                        evaluation_role, require_formal,
                                                        task_init_state_ids)
        else:
            _inspect_scope(bank, spec, path, source, task_keys, evaluation_role, require_formal,
                           task_init_state_ids)
            scene_root = SCENE_ROOT
        if bank["mode"] == "MT":
            _inspect_mt_bank(bank, spec, source)
        else:
            _inspect_tu_bank(bank, spec, path)
        return {**bank, "schema_version": EVAL_SCHEMA, "arm": "correct",
                "manifest": file_record(path),
                "scene_manifest": file_record(scene_root / "manifest.json")}
    except (KeyError, TypeError, ValueError, OSError, StopIteration) as error:
        raise Pi05EvaluationError(str(error)) from error


@dataclass(frozen=True)
class PreparedOperatorLoRA:
    key: str
    evidence: dict


class FrozenOperatorAdapter:
    """Batch a single FP32 rank128 adapter; T/U condition B, MT fixed complete state."""

    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        del device, require_formal
        bank = evaluation_adapter
        if (bank.get("kind") != KIND or bank.get("schema_version") != EVAL_SCHEMA
                or bank["source"] != source or not source_matches(bank["source"], source)):
            raise Pi05EvaluationError("operator worker bank/source changed")
        self.bank, self.policy = bank, policy
        self.tasks = {(row["suite"], row["task_id"]): row for row in bank["tasks"]}
        if set(self.tasks) != set(task_keys):
            raise Pi05EvaluationError("operator worker task keys changed")
        self.lora = load_pi05_lora_contract(REPO / "configs/pi05_lora_rank128_aligned.json") if bank["mode"] == "MT" else derive_pi05_lora_rank(
            load_pi05_lora_contract(Path(bank["asset_root"]) / read_json(Path(bank["spec"]["path"]))["source"]["lora_contract"]), rank=128)
        if self.lora.to_dict() != bank["lora"]:
            raise Pi05EvaluationError("operator worker LoRA rank/source changed")
        inject_task_lora(policy, self.lora)
        for value in task_lora_state_dict(policy).values():
            value.requires_grad_(False)
        policy.eval()
        self.batched = BatchedLoRAInference(policy, self.lora)
        self.identity = identity_lora_state(self.lora)
        self.common = load_file(bank["shared"]["path"], device="cpu")
        self.conditions = {row["condition_id"]: row for row in bank["conditions"]}
        self.states: OrderedDict[str, dict] = OrderedDict()

    def _state(self, key: str) -> dict:
        if self.bank["mode"] in ("MT", PUBLIC_BETA_MODE):
            return self.common
        if key in self.states:
            self.states.move_to_end(key)
            return self.states[key]
        row = self.conditions[key]
        if row["factors"] != file_record(Path(row["factors"]["path"])):
            raise Pi05EvaluationError("operator condition changed during evaluation")
        result = assemble_state(self.common, load_file(row["factors"]["path"], device="cpu"), self.lora)
        self.states[key] = result
        if len(self.states) > 8:
            self.states.popitem(last=False)
        return result

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedOperatorLoRA:
        task = self.tasks[(suite, task_id)]
        episode = next(row for row in task["episodes"] if row["init_state_id"] == init_state_id)
        return PreparedOperatorLoRA(episode["condition_id"], episode_evidence(self.bank, task, episode))

    @torch.no_grad()
    def install(self, prepared: PreparedOperatorLoRA) -> None:
        copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if not prepared or len(prepared) != noise.shape[0]:
            raise Pi05EvaluationError("operator LoRA batch lost paired conditions")
        copy_task_lora_state_(self.policy, self.identity, self.lora)
        with self.batched.activate([self._state(item.key) for item in prepared]):
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    def close(self) -> None:
        self.batched.close()
        self.states.clear()


def episode_evidence(bank: Mapping, task: Mapping, episode: Mapping) -> dict:
    evidence = {"schema_version": EPISODE_SCHEMA, "mode": bank["mode"],
            "global_task_id": task["global_task_id"], "init_state_id": episode["init_state_id"],
            "condition_id": episode["condition_id"], "teacher_demo": episode["teacher_demo_indices"][0],
            "video_ordinal": episode["video_ordinal"], "shared": bank["shared"],
            "checkpoint": bank["checkpoint"], "scene_manifest": bank["scene_manifest"]}
    if bank["mode"] == PUBLIC_BETA_MODE:
        evidence.update(intervention="public_B0_A", teacher_video_values_read=0,
                        video_id_role="paired_metadata_only")
    return evidence


def validate_episode(bank: Mapping, evidence, *, suite: str, task_id: int, init_state_id: int) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    task = next((row for row in bank["tasks"] if (row["suite"], row["task_id"]) == (suite, task_id)), None)
    if task is None:
        return False
    episode = next((row for row in task["episodes"] if row["init_state_id"] == init_state_id), None)
    return episode is not None and dict(evidence) == episode_evidence(bank, task, episode)


def registered_capture(args, tasks, output_dir: Path, path: Path, manifest: Mapping,
                       task_subset: Mapping | None) -> tuple[dict, dict]:
    bank_path = Path(args.static_task_lora_manifest).resolve()
    bank = read_json(bank_path)
    try:
        expected = seen_scope.capture_expectations(bank, bank_path, tasks, output_dir)
    except ValueError as error:
        raise Pi05EvaluationError(str(error)) from error
    facts = ((path.resolve(), expected["capture"].resolve()),
             (bank.get("kind"), KIND),
             (manifest.get("schema_version"), "ember_pi05_registered_trajectory_capture_v1"),
             (manifest.get("study_id"), expected["study"]),
             (manifest.get("task_subset_selection"), None),
             (manifest.get("full_conditions"), expected["full"]),
             (manifest.get("mode"), "compact"),
             (manifest.get("passive_control_trace"), PASSIVE_TAG),
             (manifest.get("stage_predicates"), True),
             (args.role, expected["role"]), (args.mode, "formal"),
             (len(tasks), expected["task_count"]),
             (output_dir.resolve(), expected["output"].resolve()))
    if (any(actual != wanted for actual, wanted in facts)
            or expected["expected_bank"] is not None and bank_path != expected["expected_bank"]
            or task_subset is not None
            or any(tuple(task.init_state_ids) != expected["states"] for task in tasks)
            or any(manifest.get(key) is not False for key in (
                "training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use"))):
        raise Pi05EvaluationError("operator official full/compact capture scope changed")
    capture = {"schema_version": "ember_pi05_registered_trajectory_capture_v1",
               "selection_path": str(path), "selection_bytes": path.stat().st_size,
               "mode": "compact", "full_conditions": expected["full"],
               "trajectory_root": str((output_dir / "trajectories").resolve()),
               "passive_trace": {"schema_version": PASSIVE_TAG,
                                 "trace_root": str((output_dir / "continuous_traces").resolve())},
               "training_gradient_use": False, "checkpoint_selection_use": False,
               "validation_use": False, "test_use": False}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
             "capture": "all_rows_post_settling_then_every_executed_control_step",
             "predicate_source": "installed_LIBERO_BDDL_goal_conjunction",
             "full_conditions_only": False, "training_gradient_use": False,
             "checkpoint_selection_use": False, "validation_action_reads": 0,
             "validation_reward_reads": 0, "held_data_use": False,
             "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def attach_capture_provenance(contract: dict, repo_root: Path) -> None:
    del repo_root
    adapter = contract.get("adapter") or {}
    scene = contract.get("operator_read_write_scene") or {}
    if adapter.get("kind") != KIND or scene.get("manifest") != adapter.get("scene_manifest"):
        raise Pi05EvaluationError("operator scene and adapter are not paired")
    contract["passive_capture_provenance"] = {
        "schema_version": PASSIVE_TAG, "bank": adapter["manifest"],
        "scene": adapter["scene_manifest"], "checkpoint": adapter["checkpoint"],
        "evaluation_commit": contract["git"]["commit"]}


def validate_capture_contract(contract: Mapping, repo_root: Path) -> None:
    adapter = contract.get("adapter") or {}
    capture = contract.get("diagnostic_occupancy_capture") or {}
    path = Path(capture["selection_path"])
    args = SimpleNamespace(static_task_lora_manifest=Path(adapter["manifest"]["path"]),
                           role=contract["role"], mode=contract["mode"])
    tasks = [SimpleNamespace(**row) for row in contract["tasks"]]
    expected, stage = registered_capture(args, tasks, Path(contract["output_dir"]),
                                         path, read_json(path), contract["diagnostic_task_subset"])
    regenerated = dict(contract)
    regenerated.pop("passive_capture_provenance", None)
    attach_capture_provenance(regenerated, repo_root)
    if (capture != expected or contract.get("diagnostic_stage_predicates") != stage
            or contract.get("passive_capture_provenance") != regenerated["passive_capture_provenance"]):
        raise Pi05EvaluationError("operator passive capture or scene provenance changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("materialize", "register-mt", "public-beta",
                                          "seen-materialize", "seen-mt"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("T", "U", *PILOT_ARMS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.phase in ("materialize", "seen-materialize"):
        if args.mode is None or args.checkpoint is None:
            parser.error("materialize requires a registered arm and completed selected ECP")
        print(materialize(args.mode, args.checkpoint, args.asset_root,
                          torch.device(args.device or "cuda:0"),
                          seen_task=args.phase == "seen-materialize"))
    elif args.phase == "public-beta":
        if args.mode is not None or args.device is not None or args.checkpoint is None:
            parser.error("public-beta takes only its fixed T1800 checkpoint and runs on CPU")
        print(materialize_public_beta(args.checkpoint, args.asset_root))
    elif args.mode is not None or args.checkpoint is not None:
        parser.error("fixed MT registration accepts no writer mode or checkpoint override")
    else:
        spec = specification()
        from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
        authorities = load_evaluation_authorities(args.asset_root / spec["source"]["evaluation_config"],
                                                  args.asset_root)
        ckpt = args.asset_root / spec["source"]["checkpoint"]
        source = inspect_source_checkpoint(authorities, ckpt.parent.parent, ckpt, evaluation_mode="formal")
        print(register_mt(args.asset_root, source, seen_task=args.phase == "seen-mt"))


if __name__ == "__main__":
    main()
