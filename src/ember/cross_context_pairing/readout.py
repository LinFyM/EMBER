"""Finite registration proposal; integrate as ember.cross_context_pairing.readout.

No second evaluator, environment creation, training runner, or CLI orchestration.
The build_banks function is a future authorized materialization consumer; this
proposal delivery invokes only metadata/schedule helpers on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ember.expert_manifold.video_schedule import reference_demo_index
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import condition_id, file_record, planned_episodes, selection_contract

STUDY = "cross_context_pairing_20261008"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY
PARENT = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928"
              "/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340")
PARENT_GIT = "e2afbfd7c997e3f792921600608efa2fa3c1b25a"
GROUPS = ((113, 118, 121), (114, 119, 122), (115, 120, 123))
SOURCE_IDS = tuple(sorted(task for group in GROUPS for task in group))
VALIDATION_IDS = (3, 6, 11, 16, 23, 26, 31, 39)
PASSIVE_TAG = "ember_operator_read_write_passive_capture_v1"
SOURCE_AUDIT = ROOT / "analysis/source_support_audit.json"


def identity(arm, slot, teacher_scene=None):
    if arm not in ("Parent", "Within", "Product") or slot not in (
            "source9", "correct", "same_task_other"):
        raise ValueError("unregistered cross-context endpoint or readout")
    if (slot == "source9" and teacher_scene not in (1, 2, 3)
            or slot != "source9" and (arm == "Parent" or teacher_scene is not None)):
        raise ValueError("only three registered source9 panels or two terminal validation arms")
    return {"study_id": STUDY, "arm": arm, "slot": slot, "teacher_scene": teacher_scene,
            "additional_updates": 0 if arm == "Parent" else 216,
            "cumulative_updates": 2340 if arm == "Parent" else 2556}


def paths(point):
    identity(point["arm"], point["slot"], point["teacher_scene"])
    root = ROOT / point["arm"]
    panel = root / "readouts" / point["slot"]
    if point["slot"] == "source9":
        panel /= f"teacher_scene_{point['teacher_scene']}"
    family = "source9" if point["slot"] == "source9" else "validation"
    checkpoint = (PARENT if point["arm"] == "Parent" else root / "train/attempts/formal"
                  / "checkpoints/macro_00000216")
    return {"panel": panel, "manifest": panel / "manifest.json", "capture": panel / "capture.json",
            "output": panel / "evaluation", "factors": root / "banks" / family,
            "shared": root / "banks/shared.safetensors", "checkpoint": checkpoint}


def official_plan(asset_root, spec, arm, slot):
    from ember.task_protocol import load_task_authorities

    point = identity(arm, slot)
    _, authority = load_task_authorities(asset_root, spec["source"]["data_protocol"])
    entries = {int(row["global_task_id"]): row for row in authority["tasks"]}
    selection = selection_contract(role="validation", task_ids=VALIDATION_IDS, cardinality=1,
        arm=slot, mode="per_init_ordinal", seed=7, init_state_ids=tuple(range(50)),
        video_pool=tuple(range(50)))
    tasks, conditions = [], {}
    for gid in VALIDATION_IDS:
        row = entries[gid]
        if row["split_role"] != "validation":
            raise ValueError("cross-context official bank crosses the held split")
        episodes = planned_episodes(selection, gid)
        tasks.append({"global_task_id": gid, "suite": row["suite"], "task_id": row["task_id"],
                      "language": row["language"], "split_role": "validation", "episodes": episodes})
        for episode in episodes:
            key = episode["condition_id"]
            conditions[key] = {"condition_id": key, "global_task_id": gid,
                               "teacher_demo": episode["teacher_demo_indices"][0]}
    return point, tasks, list(conditions.values())


def source_plan(arm, teacher_scene, metadata):
    point = identity(arm, "source9", teacher_scene)
    tasks, conditions = [], {}
    for group_index, group in enumerate(GROUPS):
        teacher = group[teacher_scene - 1]
        if len({metadata[task].authority.language for task in group}) != 1:
            raise ValueError("source readout crosses exact goal language")
        for receiver_scene, receiver in enumerate(group, 1):
            episodes = []
            for state in (0, 1):
                demo = reference_demo_index(20261008, "libero_90", teacher - 40, state,
                    demo_count=50, sampling_mode="without_replacement")
                key = condition_id(teacher, (demo,))
                episodes.append({"init_state_id": state, "video_ordinal": state,
                    "condition_id": key, "teacher_demo_indices": [demo],
                    "video_global_task_id": teacher, "teacher_scene": teacher_scene,
                    "receiver_scene": receiver_scene, "goal_index": group_index})
                conditions[key] = {"condition_id": key, "global_task_id": teacher, "teacher_demo": demo}
            tasks.append({"global_task_id": receiver, "suite": "libero_90", "task_id": receiver - 40,
                "language": metadata[receiver].authority.language, "split_role": "train",
                "episodes": episodes})
    tasks.sort(key=lambda row: row["global_task_id"])
    return point, tasks, list(conditions.values())


def materialization_data(asset_root, spec, task_ids, role):
    """Two existing compiler data-construction sites call this narrow dispatch."""
    from ember.operator_writer.data import FormalData

    if spec.get("task") == STUDY and role == "train" and tuple(task_ids) == SOURCE_IDS:
        from ember.cross_context_pairing.data import PairingData

        return PairingData(asset_root, spec, query_labels=False)
    return FormalData(asset_root, spec, query_labels=False, task_ids=task_ids, role=role)


def inspect_source(point, spec):
    """Terminal identity and complete ECP metadata; no optimizer tensor values."""
    import torch
    from ember.writer.materialization import frozen_authority

    checkpoint = paths(point)["checkpoint"]
    run = read_json(checkpoint.parent.parent / "run_contract.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    macro = 2340 if point["arm"] == "Parent" else 216
    stage = "operator_read_write_learning" if point["arm"] == "Parent" else "cross_context_pairing"
    schema = ("ember_operator_read_write_formal_run_v1" if point["arm"] == "Parent" else
              "ember_cross_context_pairing_run_v1")
    world = manifest["world_size"]
    files = {"ecp.safetensors", "trainer_state.pt", *(f"rank_{r:02d}_state.pt" for r in range(world))}
    if (not 1 <= world <= 6 or manifest["schema_version"] != "ember_ecp_checkpoint_v1"
            or manifest["next_macro"] != macro or manifest["stage"] != stage
            or manifest["run_contract_schema"] != schema or set(manifest["files"]) != files
            or run["schema_version"] != schema
            or point["arm"] != "Parent" and not frozen_authority(run["git"])
            or any(file_record(checkpoint / name)["bytes"] != record["bytes"]
                   for name, record in manifest["files"].items())):
        raise ValueError("pairing readout requires its registered full formal ECP")
    if point["arm"] == "Parent":
        historical_git = {"commit": PARENT_GIT, "branch": "", "dirty_paths": [],
                          "pushed_ref": "origin/codex/demonstration-transfer"}
        if run["git"] != historical_git or run["mode"] != "T":
            raise ValueError("parent T2340 source changed")
    else:
        trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True,
                             weights_only=True)
        expected = {"arm": point["arm"], "updates": 216, "cumulative_updates": 2556,
                    "loss_variant": "full", "scientific_distribution_fork": True}
        if (run["phase"] != "formal" or run["arm"] != point["arm"] or run["spec"] != spec
                or run["parent_training_git"] != PARENT_GIT or trainer["next_macro"] != 216
                or trainer["metrics_rows"] != 216 or trainer["training_state"] != expected
                or trainer["scheduler"]["last_epoch"] != 2556):
            raise ValueError("pairing terminal fork or inherited update clock changed")
    return run


def build_banks(asset_root, spec_path, arm, family, *, devices, frame_chunk, cpu_threads,
                materialization_git, source_audit, previous_materialization_git=None):
    """Future authorized GPU consumer: one generic compiler queue, one shared A."""
    from safetensors.torch import load_file, save_file
    from ember.cross_context_pairing.data import pairing_tasks
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, expected_lora_state_shapes, identity_lora_state
    from ember.operator_writer.bank import BANK_SCHEMA, KIND, _factor_header
    from ember.operator_writer.materialization import compile_conditions, register_partial
    from ember.operator_writer.model import OperatorReadWrite
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
    from ember.writer.materialization import frozen_authority

    spec_path, asset_root = Path(spec_path), Path(asset_root)
    spec = read_json(spec_path)
    if Path(source_audit).resolve() != SOURCE_AUDIT.resolve():
        raise ValueError("source9 requires the actual audited official asset registration")
    if spec["task"] != STUDY or family not in ("source9", "validation") or not frozen_authority(materialization_git):
        raise ValueError("unregistered pairing materialization identity")
    metadata = pairing_tasks(asset_root, spec["source"]["data_protocol"]) if family == "source9" else None
    plans = ([source_plan(arm, scene, metadata) for scene in (1, 2, 3)] if family == "source9" else
             [official_plan(asset_root, spec, arm, slot) for slot in ("correct", "same_task_other")])
    if any(paths(panel)["manifest"].exists() for panel, _, _ in plans):
        raise ValueError("pairing bank is already published")
    point, _, _ = plans[0]; location = paths(point); run = inspect_source(point, spec)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset_root / spec["source"]["lora_contract"]), rank=128)
    if lora.to_dict() != run["lora"]:
        raise ValueError("pairing must preserve full T rank128/38-target contract")
    unique = {row["condition_id"]: row for _, _, rows in plans for row in rows}
    if len(unique) != (18 if family == "source9" else 400):
        raise ValueError("pairing readout changed the registered number of compiled conditions")
    output = location["factors"]; output.mkdir(parents=True, exist_ok=True)
    register_partial(output, {"mode": "T", "checkpoint": str(location["checkpoint"]),
        "spec": file_record(spec_path), "source": run["source"], "lora": lora.to_dict(),
        "training_git": run["git"]["commit"], "materialization_git": materialization_git,
        "cross_context_pairing_family": family}, previous_materialization_git)
    shapes = expected_lora_state_shapes(lora)
    ashapes = {k: v for k, v in shapes.items() if k.endswith(LORA_A_SUFFIX)}
    bshapes = {k: v for k, v in shapes.items() if k.endswith(LORA_B_SUFFIX)}
    if location["shared"].exists():
        _factor_header(location["shared"], ashapes, metadata={"schema_version": BANK_SCHEMA, "mode": "T"})
    else:
        writer = OperatorReadWrite(lora, identity_lora_state(lora), "T")
        writer.load_state_dict(load_file(str(location["checkpoint"] / "ecp.safetensors")), strict=True)
        shared = {k: v.detach().float().cpu().contiguous() for k, v in writer.public_state().items() if k in ashapes}
        temporary = location["shared"].with_suffix(".safetensors.tmp")
        save_file(shared, str(temporary), metadata={"schema_version": BANK_SCHEMA, "mode": "T"})
        temporary.replace(location["shared"])
        del writer, shared
    conditions = list(unique.values())
    compile_conditions(asset_root, spec, "T", location["checkpoint"], run["source"], output,
        conditions, bshapes, devices=devices, frame_chunk=frame_chunk,
        task_ids=SOURCE_IDS if family == "source9" else VALIDATION_IDS,
        role="train" if family == "source9" else "validation", cpu_threads=cpu_threads)
    compiled = {row["condition_id"]: row for row in conditions}
    for panel, tasks, rows in plans:
        target = paths(panel); target["panel"].mkdir(parents=True, exist_ok=True)
        bank = {"schema_version": BANK_SCHEMA, "kind": KIND, "mode": "T", "cross_context_pairing": panel,
            "asset_root": str(asset_root), "spec": file_record(spec_path), "source": run["source"],
            "checkpoint": str(target["checkpoint"]), "checkpoint_manifest": file_record(target["checkpoint"] / "checkpoint_manifest.json"),
            "lora": lora.to_dict(), "training_git": run["git"]["commit"], "materialization_git": materialization_git,
            "shared": file_record(target["shared"]), "conditions": [compiled[row["condition_id"]] for row in rows],
            "tasks": tasks, "source_audit": file_record(source_audit), "scene_manifest": None,
            "information_wall": {"teacher_video_values_read": len(unique), "teacher_runtime_reads": 0,
                "deployment_adapters": 1, "validation_test_gradients": False}}
        if family == "validation":
            from ember.operator_writer.bank import SCENE_ROOT
            bank["scene_root"] = str(SCENE_ROOT)
        write_json_atomic(target["manifest"], bank)
        write_json_atomic(target["capture"], capture_selection(panel, tasks))
    return [paths(panel)["manifest"] for panel, _, _ in plans]


def capture_selection(point, tasks):
    full = [{"suite": task["suite"], "task_id": task["task_id"], "init_state_id": 0}
            for task in tasks if point["slot"] == "correct" or
            point["slot"] == "source9" and task["global_task_id"] in GROUPS[0]]
    return {"schema_version": "ember_pi05_registered_trajectory_capture_v1", "study_id": STUDY,
        "cross_context_pairing": point, "mode": "compact", "full_conditions": full,
        "task_subset_selection": None, "passive_control_trace": PASSIVE_TAG, "stage_predicates": True,
        **{key: False for key in ("training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use")}}


def inspect(bank, path, source, task_keys, role, require_formal, task_states):
    from ember.operator_writer.bank import BANK_SCHEMA, EVAL_SCHEMA, KIND, _factor_header, SCENE_ROOT
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, expected_lora_state_shapes
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
    from ember.cross_context_pairing.data import pairing_tasks
    from ember.pi05_eval.scene import inspect_registered_scenes

    point = bank["cross_context_pairing"]; wanted = identity(point["arm"], point["slot"], point["teacher_scene"])
    location = paths(wanted); spec = read_json(Path(bank["spec"]["path"])); asset_root = Path(bank["asset_root"])
    plan = (source_plan(point["arm"], point["teacher_scene"], pairing_tasks(asset_root, spec["source"]["data_protocol"]))
            if point["slot"] == "source9" else official_plan(asset_root, spec, point["arm"], point["slot"]))
    run = inspect_source(wanted, spec); _, tasks, rows = plan
    states = (0, 1) if point["slot"] == "source9" else tuple(range(50))
    keys = tuple((task["suite"], task["task_id"]) for task in tasks)
    expected_role = "nonheld_meta" if point["slot"] == "source9" else "validation"
    if (not require_formal or role != expected_role or tuple(task_keys) != keys or point != wanted
            or path != location["manifest"].resolve() or bank["tasks"] != tasks
            or bank["schema_version"] != BANK_SCHEMA or bank["kind"] != KIND or bank["mode"] != "T"
            or bank["source"] != source or source != run["source"] or bank["training_git"] != run["git"]["commit"]
            or bank["checkpoint"] != str(location["checkpoint"]) or bank["lora"] != run["lora"]
            or bank["checkpoint_manifest"] != file_record(location["checkpoint"] / "checkpoint_manifest.json")
            or bank["source_audit"] != file_record(SOURCE_AUDIT)
            or bank["shared"] != file_record(location["shared"])
            or task_states is None or {key: tuple(value) for key, value in task_states.items()}
            != {key: states for key in keys}):
        raise ValueError("cross-context bank/pairing/endpoint scope changed")
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset_root / spec["source"]["lora_contract"]), rank=128)
    shapes = expected_lora_state_shapes(lora)
    _factor_header(location["shared"], {k: v for k, v in shapes.items() if k.endswith(LORA_A_SUFFIX)},
                   metadata={"schema_version": BANK_SCHEMA, "mode": "T"})
    if len(bank["conditions"]) != len(rows):
        raise ValueError("registered conditions incomplete")
    for actual, expected in zip(bank["conditions"], rows, strict=True):
        factor = location["factors"] / f"{expected['condition_id']}.safetensors"
        if (set(actual) != {*expected, "factors", "raw_frames", "sampled_frames"}
                or any(actual[k] != v for k, v in expected.items()) or actual["factors"] != file_record(factor)
                or not 0 < actual["sampled_frames"] <= actual["raw_frames"]):
            raise ValueError("teacher condition factor provenance changed")
        _factor_header(factor, {k: v for k, v in shapes.items() if k.endswith(LORA_B_SUFFIX)},
            metadata={"schema_version": BANK_SCHEMA, "condition_id": expected["condition_id"], "mode": "T"})
    from ember.writer.materialization import frozen_authority
    if not frozen_authority(bank["materialization_git"]):
        raise ValueError("pairing bank lacks its frozen materialization code identity")
    scene = None
    if point["slot"] != "source9":
        if bank["scene_root"] != str(SCENE_ROOT):
            raise ValueError("official validation no longer reuses the T scenes")
        inspect_registered_scenes(SCENE_ROOT, tasks)
        scene = file_record(SCENE_ROOT / "manifest.json")
    return {**bank, "schema_version": EVAL_SCHEMA, "arm": point["slot"], "manifest": file_record(path),
            "scene_manifest": scene}


def source_request(args):
    if not getattr(args, "cross_context_pairing", False):
        return False
    path = getattr(args, "static_task_lora_manifest", None)
    if path is None:
        raise Pi05EvaluationError("cross-context registration requires an operator manifest")
    point = read_json(Path(path)).get("cross_context_pairing")
    if point is None or point != identity(point["arm"], point["slot"], point["teacher_scene"]):
        raise Pi05EvaluationError("missing cross-context finite registration")
    if (args.mode != "formal" or args.role != ("nonheld_meta" if point["slot"] == "source9" else "validation")
            or args.state_count != (2 if point["slot"] == "source9" else 50)
            or any(getattr(args, key, None) for key in ("init_state_ids", "exploration_sigma", "occupancy_capture_selection", "task_subset_selection"))):
        raise Pi05EvaluationError("cross-context official states/panel request changed")
    return point["slot"] == "source9"


def select_source_tasks(args, tasks):
    if not source_request(args):
        return tuple(tasks), None
    selected = tuple(task for task in tasks if task.suite == "libero_90" and task.task_id + 40 in SOURCE_IDS)
    if (tuple(task.task_id + 40 for task in selected) != SOURCE_IDS
            or any(tuple(task.init_state_ids) != (0, 1) or task.horizon != 400 for task in selected)):
        raise Pi05EvaluationError("actual official source9/init0/1/horizon assets changed")
    return selected, None


def registered_capture(args, tasks, output, path, manifest, subset, bank):
    point = bank["cross_context_pairing"]; location = paths(point)
    states = (0, 1) if point["slot"] == "source9" else tuple(range(50))
    if (path != location["capture"].resolve() or output.resolve() != location["output"].resolve()
            or manifest != capture_selection(point, bank["tasks"]) or subset is not None
            or args.mode != "formal" or args.role != ("nonheld_meta" if point["slot"] == "source9" else "validation")
            or tuple((t.suite, t.task_id) for t in tasks) != tuple((t["suite"], t["task_id"]) for t in bank["tasks"])
            or any(tuple(t.init_state_ids) != states for t in tasks)):
        raise Pi05EvaluationError("pairing passive capture scope changed")
    capture = {"schema_version": manifest["schema_version"], "cross_context_pairing": point, "selection_path": str(path),
        "selection_bytes": path.stat().st_size, "mode": "compact", "full_conditions": manifest["full_conditions"],
        "trajectory_root": str((output / "trajectories").resolve()),
        "passive_trace": {"schema_version": PASSIVE_TAG, "trace_root": str((output / "continuous_traces").resolve())},
        **{key: False for key in ("training_gradient_use", "checkpoint_selection_use", "validation_use", "test_use")}}
    stage = {"schema_version": "ember_pi05_stage_predicate_capture_v1",
        "capture": "all_rows_post_settling_then_every_executed_control_step",
        "predicate_source": "installed_LIBERO_BDDL_goal_conjunction", "full_conditions_only": False,
        "training_gradient_use": False, "checkpoint_selection_use": False,
        "validation_action_reads": 0, "validation_reward_reads": 0, "held_data_use": False,
        "claim_boundary": "BDDL predicates are partial progress signals"}
    return capture, stage


def evidence_fields(bank, task, episode):
    condition = next(row for row in bank["conditions"] if row["condition_id"] == episode["condition_id"])
    return {"cross_context_pairing": bank["cross_context_pairing"],
        "teacher_global_task_id": condition["global_task_id"], "receiver_global_task_id": task["global_task_id"],
        "teacher_scene": episode.get("teacher_scene"), "receiver_scene": episode.get("receiver_scene"),
        "factors": condition["factors"], "training_git": bank["training_git"],
        "materialization_git": bank["materialization_git"], "readout_spec": bank["spec"],
        "initialization": "official_source_init_dummy10" if bank["cross_context_pairing"]["slot"] == "source9" else "sealed_T_validation_scene"}


def attach_provenance(contract):
    adapter = contract["adapter"]; point = adapter["cross_context_pairing"]
    expected = None if point["slot"] == "source9" else adapter["scene_manifest"]
    scene = contract.get("operator_read_write_scene")
    if (point["slot"] == "source9" and scene is not None
            or point["slot"] != "source9" and (scene or {}).get("manifest") != expected):
        raise Pi05EvaluationError("pairing source initializer and T official scenes mixed")
    contract["passive_capture_provenance"] = {"schema_version": PASSIVE_TAG,
        "bank": adapter["manifest"], "scene": expected, "checkpoint": adapter["checkpoint"],
        "cross_context_pairing": point, "training_commit": adapter["training_git"],
        "bank_materialization_git": adapter["materialization_git"],
        "evaluation_commit": contract["git"]["commit"], "source_audit": adapter["source_audit"],
        "official_initialization_assets": [{k: t[k] for k in ("suite", "task_id", "bddl_file", "bddl_bytes",
            "init_states_file", "init_states_bytes", "installed_init_state_count", "init_state_ids", "horizon")}
            for t in contract["tasks"]] if point["slot"] == "source9" else None}


def actual_prefixes(slot):
    starts, prefixes = slot["replay_replan_steps"], slot["replay_executed_prefixes"]
    if len(starts) != len(prefixes):
        raise Pi05EvaluationError("pairing capture lost a real replan prefix")
    return tuple(prefix[:min(5, max(0, int(slot["steps"]) - int(start)))]
                 for start, prefix in zip(starts, prefixes, strict=True))


def fixed_review_cases():
    cases = []
    for arm in ("Parent", "Within", "Product"):
        for teacher_scene, receiver in ((1, 121), (3, 113)):
            point = identity(arm, "source9", teacher_scene)
            cases.append({"point": point, "suite": "libero_90", "task_id": receiver - 40,
                          "global_task_id": receiver, "init_state_id": 0})
    for arm in ("Within", "Product"):
        for gid, suite, local in ((23, "libero_goal", 3), (39, "libero_10", 9)):
            cases.append({"point": identity(arm, "correct"), "suite": suite,
                          "task_id": local, "global_task_id": gid, "init_state_id": 0})
    return cases
