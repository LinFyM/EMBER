"""Frozen post-selection validation controls for one registered T checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_eval.scene import inspect_registered_scenes
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.lora import LORA_B_SUFFIX, expected_lora_state_shapes
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.expert_manifold.video_schedule import frame_order_seed
from ember.pi05_target_data import SUITE_ORDER
from ember.video_conditions import frame_control
from ember.writer.materialization import file_record, planned_episodes, selection_contract
from ember.writer.video_controls import controlled_frames


ROOT = Path("/data1/user/ymdai/ember_runs/operator_selected_validation_20260929")
SELECTION = ROOT / "selection.json"
SCHEMA = "ember_operator_selected_validation_v1"
STUDY = "operator_selected_validation_20260929"
CAPTURE = Path(__file__).resolve().parents[3] / "configs/operator_read_write_v1/selected_capture.json"
MACROS = (1890, 1980, 2070, 2160, 2250, 2340)
VIDEO_ARMS = ("same_task_other", "cross_suite_wrong", "shuffled")


def dispatch(args, parser) -> Path:
    """Keep one bank CLI while the selected sub-scope owns its fixed arguments."""
    if args.mode is not None or args.checkpoint is not None:
        parser.error("selected banks read only the frozen main selection")
    if args.phase == "selected-other":
        if args.arm is not None or args.device is not None:
            parser.error("selected-other needs no GPU or arm override")
        return register_other()
    if args.phase == "selected-public-beta":
        if args.arm is not None or args.device is not None:
            parser.error("selected-public-beta needs no GPU or arm override")
        return materialize_public_beta(args.asset_root)
    if args.phase == "selected-video":
        if args.arm is None or args.device is None:
            parser.error("selected-video requires wrong/shuffled arm and GPU device")
        return materialize_video(args.arm, args.asset_root, torch.device(args.device))
    raise ValueError("unregistered selected bank phase")


def _correct_paths(macro: int) -> tuple[Path, Path]:
    if macro not in MACROS:
        raise ValueError("selected checkpoint is outside the registered observation nodes")
    if macro == 1890:
        root = Path("/data1/user/ymdai/ember_runs/operator_public_function_pilot_20260929/control")
    else:
        root = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/continuation2340/T")
    return root / "banks" / str(macro) / "manifest.json", root / "evaluation" / str(macro) / "correct400"


def registration() -> tuple[dict, dict]:
    """A main-selected complete correct400 is the only input authority."""
    value = read_json(SELECTION)
    if (set(value) != {"schema_version", "selected_by", "selection_rule", "macro",
                       "correct_bank", "correct_results"}
            or value["schema_version"] != SCHEMA
            or value["selected_by"] != "science_main"
            or value["selection_rule"] != "highest_complete_correct400_tie_earlier"):
        raise ValueError("selected validation requires the main's frozen selection record")
    macro = value["macro"]
    bank_path, official = _correct_paths(macro)
    if (value["correct_bank"] != file_record(bank_path)
            or value["correct_results"] != file_record(official / "results.json")):
        raise ValueError("selected correct400 source changed")
    complete = read_json(official / "launcher_completion.json")
    if (complete.get("queue", {}).get("completed_rows") != 400
            or any(code != 0 for code in complete.get("return_codes", {}).values())
            or not complete.get("return_codes")):
        raise ValueError("selected checkpoint lacks a complete official400")
    source = read_json(bank_path)
    if (source.get("mode") != ("control" if macro == 1890 else "T")
            or Path(source.get("checkpoint", "")).name != f"macro_{macro:08d}"
            or len(source.get("conditions", ())) != 400):
        raise ValueError("selected control does not inherit the fixed T checkpoint")
    return value, source


def bank_path(arm: str, macro: int) -> Path:
    if arm not in (*VIDEO_ARMS, "public_beta") or macro not in MACROS:
        raise ValueError("unregistered selected validation arm or checkpoint")
    return ROOT / arm / "banks" / str(macro) / "manifest.json"


def output_path(arm: str) -> Path:
    if arm not in (*VIDEO_ARMS, "public_beta"):
        raise ValueError("unregistered selected validation arm")
    return ROOT / arm / "evaluation/correct400"


def task_rows(source: Mapping, arm: str) -> tuple[list[dict], list[dict]]:
    if arm not in VIDEO_ARMS:
        raise ValueError("selected video task rows require a registered control")
    spec = read_json(Path(source["spec"]["path"]))
    selection = selection_contract(
        role="validation", task_ids=spec["evaluation"]["task_ids"], cardinality=1,
        arm=arm, mode="per_init_ordinal", seed=spec["evaluation"]["video_schedule_seed"],
        init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    tasks = [{**task, "episodes": planned_episodes(selection, task["global_task_id"])}
             for task in source["tasks"]]
    conditions = [{"condition_id": episode["condition_id"],
                   "global_task_id": task["global_task_id"],
                   "teacher_demo": episode["teacher_demo_indices"][0]}
                  for task in tasks for episode in task["episodes"]]
    if (len(tasks) != 8 or len(conditions) != 400
            or len({row["condition_id"] for row in conditions}) != 400
            or any(len({episode["teacher_demo_indices"][0] for episode in task["episodes"]}) != 50
                   for task in tasks)):
        raise ValueError("selected video control must retain eight complete fifty-video rounds")
    return tasks, conditions


def _lineage(arm: str, source_path: Path) -> dict:
    return {"schema_version": SCHEMA, "arm": arm,
            "selection": file_record(SELECTION), "correct_bank": file_record(source_path),
            "checkpoint_selection_use": False, "training_feedback": False}


def register_other() -> Path:
    """Change the per-init video association; reuse the 400 already compiled LoRAs."""
    from .run import frozen_git

    selected, source = registration()
    macro = selected["macro"]
    source_path, _ = _correct_paths(macro)
    tasks, conditions = task_rows(source, "same_task_other")
    if {row["condition_id"] for row in conditions} != {
            row["condition_id"] for row in source["conditions"]}:
        raise ValueError("other-video association cannot reuse all original factor files")
    wanted = {**source, "tasks": tasks,
              "selected_control": _lineage("same_task_other", source_path),
              "materialization_git": frozen_git(continuation=True)}
    output = bank_path("same_task_other", macro)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if read_json(output) != wanted:
            raise ValueError("published other-video bank differs")
    else:
        write_json_atomic(output, wanted)
    return output


def materialize_video(arm: str, asset_root: Path, device: torch.device) -> Path:
    """Compile wrong or shuffled real RGB with the unchanged selected T Writer."""
    from . import bank as owner
    from .data import FormalData
    from .run import build_runtime, frozen_git

    if arm not in ("cross_suite_wrong", "shuffled"):
        raise ValueError("GPU video materialization requires wrong or shuffled")
    selected, source = registration()
    if asset_root.resolve() != Path(source["asset_root"]).resolve():
        raise ValueError("selected video asset root changed")
    macro = selected["macro"]
    correct_path, _ = _correct_paths(macro)
    checkpoint = Path(source["checkpoint"])
    spec = read_json(Path(source["spec"]["path"]))
    run = owner.inspect_training_source(spec, checkpoint, "T",
                                        sealed_evaluation=True)
    if (run["git"]["commit"] != source["training_git"]
            or checkpoint != Path(source["checkpoint"])):
        raise ValueError("selected video materialization changed training source")
    tasks, conditions = task_rows(source, arm)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        asset_root / spec["source"]["lora_contract"]), rank=128)
    if lora.to_dict() != source["lora"] or run["lora"] != source["lora"]:
        raise ValueError("selected video LoRA factor contract changed")
    output = bank_path(arm, macro)
    if output.exists():
        raise ValueError("selected video bank already published")
    output.parent.mkdir(parents=True, exist_ok=True)
    git = frozen_git(continuation=True)
    contract = {"schema_version": SCHEMA, "arm": arm, "selected": file_record(SELECTION),
                "correct_bank": file_record(correct_path), "checkpoint": str(checkpoint),
                "materialization_git": git, "source": source["source"]}
    contract_path = output.parent / "materialization_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError("partial selected video bank has another source")
    else:
        write_json_atomic(contract_path, contract)
    runtime = build_runtime(asset_root, spec, device, "T")
    if runtime.source != source["source"]:
        raise ValueError("selected video native/source identity changed")
    runtime.writer.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"),
                                             device=str(device)), strict=True)
    runtime.writer.eval()
    b_shapes = {name: shape for name, shape in expected_lora_state_shapes(lora).items()
                if name.endswith(LORA_B_SUFFIX)}
    data = FormalData(asset_root, spec, query_labels=False,
                      task_ids=tuple(spec["evaluation"]["task_ids"]), role="validation")
    episodes = {episode["condition_id"]: episode
                for task in tasks for episode in task["episodes"]}
    transforms = {}
    try:
        for condition in conditions:
            key = condition["condition_id"]
            target, demo = condition["global_task_id"], condition["teacher_demo"]
            episode = episodes[key]
            donor = episode["video_global_task_id"]
            video = data.videos.load(donor, demo)
            pixels = torch.from_numpy(video.frames).to(device, non_blocking=True)
            indices = torch.from_numpy(video.frame_indices).to(device, non_blocking=True)
            evidence = {"video_global_task_id": donor,
                        "source_frame_indices": video.frame_indices.tolist()}
            if arm == "shuffled":
                order, positions, moved = controlled_frames(
                    video.frame_indices,
                    control={"selection_seed": spec["evaluation"]["video_schedule_seed"],
                             "language_global_task_id": target, "arm": arm}, demo=demo)
                pixels = pixels.index_select(0, order.to(device))
                indices = positions.to(device)
                evidence.update(moved)
            transforms[key] = evidence
            tokens, mask, _ = runtime.tokenizer([data.tasks[target].authority.language])
            path = output.parent / f"{key}.safetensors"
            if path.exists():
                owner._factor_header(path, b_shapes,
                                     metadata={"schema_version": owner.BANK_SCHEMA,
                                               "condition_id": key, "mode": source["mode"]})
            else:
                with torch.no_grad():
                    state, _ = runtime.compile((pixels, indices, tokens, mask),
                                               frame_chunk=spec["operator"]["frame_chunk"])
                factors = {name: value.detach().float().cpu().contiguous()
                           for name, value in state.items() if name.endswith(LORA_B_SUFFIX)}
                if len(factors) != 38:
                    raise ValueError("selected video did not compile the complete B0+M")
                save_file(factors, str(path), metadata={"schema_version": owner.BANK_SCHEMA,
                                                       "condition_id": key, "mode": source["mode"]})
            condition.update(factors=file_record(path), raw_frames=video.raw_frame_count,
                             sampled_frames=len(video.frames))
    finally:
        data.close()
    transforms_path = output.parent / "video_transforms.json"
    if transforms_path.exists():
        if read_json(transforms_path) != transforms:
            raise ValueError("selected video transform record changed during resume")
    else:
        write_json_atomic(transforms_path, transforms)
    bank = {**source, "tasks": tasks, "conditions": conditions,
            "selected_control": _lineage(arm, correct_path),
            "materialization_git": git, "video_transforms": file_record(transforms_path)}
    write_json_atomic(output, bank)
    return output


def materialize_public_beta(asset_root: Path) -> Path:
    """Export one B0 A adapter from the selected ECP without reading teacher RGB."""
    from . import bank as owner, public_beta
    from .run import frozen_git

    selected, source = registration()
    if asset_root.resolve() != Path(source["asset_root"]).resolve():
        raise ValueError("selected public beta asset root changed")
    macro = selected["macro"]
    correct_path, _ = _correct_paths(macro)
    checkpoint = Path(source["checkpoint"])
    spec = read_json(Path(source["spec"]["path"]))
    run = owner.inspect_training_source(spec, checkpoint, "T",
                                        sealed_evaluation=True)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        asset_root / spec["source"]["lora_contract"]), rank=128)
    if run["git"]["commit"] != source["training_git"] or lora.to_dict() != source["lora"]:
        raise ValueError("selected beta source or LoRA contract changed")
    state = public_beta.public_state(checkpoint, lora)
    output = bank_path("public_beta", macro)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("selected public beta already published")
    shared_path = output.parent / "public_beta.safetensors"
    if shared_path.exists():
        owner._factor_header(shared_path, expected_lora_state_shapes(lora),
                             metadata={"schema_version": owner.PUBLIC_BETA_SCHEMA,
                                       "mode": owner.PUBLIC_BETA_MODE})
    else:
        save_file(state, str(shared_path),
                  metadata={"schema_version": owner.PUBLIC_BETA_SCHEMA,
                            "mode": owner.PUBLIC_BETA_MODE})
    wanted = {**source, "schema_version": owner.PUBLIC_BETA_SCHEMA,
              "mode": owner.PUBLIC_BETA_MODE, "shared": file_record(shared_path),
              "selected_control": _lineage("public_beta", correct_path),
              "materialization_git": frozen_git(continuation=True),
              "public_intervention": {
                  "formula": "B0 A", "removed_term": "M(V,L) A",
                  "weights": "same selected T checkpoint common A/B0; no MT weights",
                  "factor_map": public_beta.factor_map(lora)},
              "information_wall": {"teacher_video_values_read": 0,
                                   "teacher_runtime_reads": 0,
                                   "video_id_role": "paired_metadata_only",
                                   "deployment_adapters": 1,
                                   "validation_test_gradients": False}}
    write_json_atomic(output, wanted)
    return output


def _inspect_beta(bank: Mapping, path: Path, correct: Mapping, correct_path: Path) -> None:
    from . import bank as owner, public_beta

    spec = read_json(Path(correct["spec"]["path"]))
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(
        Path(bank["asset_root"]) / spec["source"]["lora_contract"]), rank=128)
    shared_path = path.parent / "public_beta.safetensors"
    wanted = {**correct, "schema_version": owner.PUBLIC_BETA_SCHEMA,
              "mode": owner.PUBLIC_BETA_MODE, "shared": file_record(shared_path),
              "selected_control": _lineage("public_beta", correct_path),
              "materialization_git": bank.get("materialization_git"),
              "public_intervention": {
                  "formula": "B0 A", "removed_term": "M(V,L) A",
                  "weights": "same selected T checkpoint common A/B0; no MT weights",
                  "factor_map": public_beta.factor_map(lora)},
              "information_wall": {"teacher_video_values_read": 0,
                                   "teacher_runtime_reads": 0,
                                   "video_id_role": "paired_metadata_only",
                                   "deployment_adapters": 1,
                                   "validation_test_gradients": False}}
    if bank != wanted:
        raise ValueError("selected public beta source changed")
    owner._factor_header(shared_path, expected_lora_state_shapes(lora),
                         metadata={"schema_version": owner.PUBLIC_BETA_SCHEMA,
                                   "mode": owner.PUBLIC_BETA_MODE})


def _inspect_transforms(bank: Mapping, path: Path, arm: str, correct: Mapping,
                        tasks: list, conditions: list, actual_conditions: list) -> Path:
    transform_path = path.parent / "video_transforms.json"
    if bank.get("video_transforms") != file_record(transform_path):
        raise ValueError("selected video transform record changed")
    transforms = read_json(transform_path)
    episode_map = {episode["condition_id"]: episode
                   for task in tasks for episode in task["episodes"]}
    if set(transforms) != {row["condition_id"] for row in conditions}:
        raise ValueError("selected video transform coverage changed")
    seed_value = read_json(Path(correct["spec"]["path"]))["evaluation"]["video_schedule_seed"]
    for row, actual in zip(conditions, actual_conditions, strict=True):
        evidence = transforms[row["condition_id"]]
        indices = evidence.get("source_frame_indices")
        if (evidence.get("video_global_task_id") !=
                episode_map[row["condition_id"]]["video_global_task_id"]
                or not isinstance(indices, list) or len(indices) != actual["sampled_frames"]
                or len(set(indices)) != len(indices)):
            raise ValueError("selected video source changed")
        if arm == "shuffled":
            suite = SUITE_ORDER[row["global_task_id"] // 10]
            seed = frame_order_seed(seed_value, suite, row["global_task_id"] % 10,
                                    row["teacher_demo"])
            order = frame_control(len(indices), condition="shuffled", order_seed=seed)
            if (evidence.get("frame_order_seed") != seed
                    or evidence.get("frame_permutation") != order.content.tolist()
                    or evidence.get("frame_indices") != sorted(indices)):
                raise ValueError("selected shuffled RGB/time mapping changed")
    return transform_path


def _inspect_video(bank: Mapping, path: Path, arm: str, correct: Mapping,
                   correct_path: Path) -> list:
    from . import bank as owner

    tasks, conditions = task_rows(correct, arm)
    wanted = {**correct, "tasks": tasks, "selected_control": _lineage(arm, correct_path),
              "materialization_git": bank.get("materialization_git")}
    if arm == "same_task_other":
        wanted["conditions"] = correct["conditions"]
        if {row["condition_id"] for row in wanted["conditions"]} != {
                row["condition_id"] for row in conditions}:
            raise ValueError("other-video bank cannot reuse the paired factors")
    else:
        actual = bank.get("conditions") or []
        if (len(actual) != 400 or any(
                {name: row.get(name) for name in
                 ("condition_id", "global_task_id", "teacher_demo")} != expected
                or row.get("factors") != file_record(
                    path.parent / f"{expected['condition_id']}.safetensors")
                or not 0 < row.get("sampled_frames", 0) <= row.get("raw_frames", 0)
                for row, expected in zip(actual, conditions, strict=True))):
            raise ValueError("selected compiled condition map changed")
        transforms = _inspect_transforms(bank, path, arm, correct, tasks, conditions, actual)
        materialization = read_json(path.parent / "materialization_contract.json")
        if materialization != {"schema_version": SCHEMA, "arm": arm,
                               "selected": file_record(SELECTION),
                               "correct_bank": file_record(correct_path),
                               "checkpoint": bank["checkpoint"],
                               "materialization_git": bank["materialization_git"],
                               "source": bank["source"]}:
            raise ValueError("selected materialization source changed")
        wanted.update(conditions=actual, video_transforms=file_record(transforms))
    if bank != wanted:
        raise ValueError("selected video bank/scientific source changed")
    owner._inspect_tu_bank(bank, read_json(Path(bank["spec"]["path"])), path)
    inspect_registered_scenes(owner.SCENE_ROOT, tasks)
    return tasks


def inspect(bank: Mapping, path: Path, source: Mapping, task_keys: tuple,
            evaluation_role: str, require_formal: bool,
            task_init_state_ids: Mapping | None) -> dict:
    """Accept evaluation-only Git while fixing numerical source and scene pairing."""
    from . import bank as owner

    selected, correct = registration()
    arm = (bank.get("selected_control") or {}).get("arm")
    correct_path, _ = _correct_paths(selected["macro"])
    if arm not in (*VIDEO_ARMS, "public_beta") or path != bank_path(arm, selected["macro"]).resolve():
        raise ValueError("selected validation bank/arm path changed")
    original = owner.inspect_bank(manifest_path=correct_path, source=source,
                                  task_keys=task_keys, evaluation_role=evaluation_role,
                                  require_formal=require_formal,
                                  task_init_state_ids=task_init_state_ids)
    git = bank.get("materialization_git") or {}
    if (bank.get("selected_control") != _lineage(arm, correct_path)
            or git.get("branch") != "" or git.get("dirty_paths") != []
            or git.get("pushed_ref") not in
            ("origin/main", "origin/codex/demonstration-transfer")
            or evaluation_role != "validation" or not require_formal):
        raise ValueError("selected validation source or evaluation Git changed")
    if arm == "public_beta":
        _inspect_beta(bank, path, correct, correct_path)
        tasks = correct["tasks"]
    else:
        tasks = _inspect_video(bank, path, arm, correct, correct_path)
    if ({tuple(key) for key in task_keys} !=
            {(row["suite"], row["task_id"]) for row in tasks}
            or task_init_state_ids is not None and any(
                tuple(task_init_state_ids.get((row["suite"], row["task_id"]), ()))
                != tuple(range(50)) for row in tasks)):
        raise ValueError("selected validation paired task/state map changed")
    return {**bank, "schema_version": owner.EVAL_SCHEMA, "arm": arm,
            "manifest": file_record(path), "scene_manifest": original["scene_manifest"]}


def capture_expectations(bank: Mapping, bank_path_value: Path, tasks: list,
                         output_dir: Path | None) -> dict:
    arm = bank["selected_control"]["arm"]
    selected, _ = registration()
    macro = selected["macro"]
    output = output_path(arm)
    if (bank_path_value != bank_path(arm, macro).resolve()
            or output_dir is not None and output_dir.resolve() != output.resolve()):
        raise ValueError("selected validation capture path changed")
    full = [{"suite": task.suite, "task_id": task.task_id, "init_state_id": 0}
            for task in tasks]
    return {"full": full, "capture": CAPTURE, "study": STUDY, "output": output,
            "role": "validation", "states": tuple(range(50)), "task_count": 8,
            "expected_bank": bank_path(arm, macro).resolve()}
