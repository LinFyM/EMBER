"""Real-frame control wiring, frozen pairing, and source-identity execution."""

from dataclasses import (replace)
import json
from pathlib import Path

import pytest

from ember.lora import (LoRATarget)
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer import (evaluation, materialization)
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import (file_record, planned_episodes, selection_contract)
from ember.writer.runtime import MODEL_DEFAULTS
from ember.writer.video_controls import (CONTROL_ARMS, DIAGNOSTIC_DECLARATION, METHOD_FREEZE_DECLARATION, SEALED_TEST_DECLARATION, inspect_diagnostic_contract, video_task_id)
from test_horizon_evaluation import GIT, ROOT, SOURCE


VALIDATION = (1, 3, 11, 13, 23, 26, 31, 32)
TEST = (6, 8, 10, 17, 24, 27, 30, 33)


def selection(arm, **changes):
    values = dict(role="validation", task_ids=VALIDATION, cardinality=1, arm=arm,
                  mode="per_init_ordinal", seed=7, init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    return selection_contract(**(values | changes))


@pytest.mark.parametrize("arm", CONTROL_ARMS)
def test_control_rounds_keep_all_fifty_paired_ordinals_without_replacement(arm):
    selected, reference = selection(arm), selection("correct")
    donors = []
    for task in VALIDATION:
        observed, correct = planned_episodes(selected, task), planned_episodes(reference, task)
        assert [(r["init_state_id"], r["video_ordinal"], r["paired_correct_demos"], r["paired_other_demos"])
                for r in observed] == [(r["init_state_id"], r["video_ordinal"], r["paired_correct_demos"], r["paired_other_demos"])
                                      for r in correct]
        donor = video_task_id(selected, task)
        donors.append(donor)
        assert all(row["video_global_task_id"] == donor for row in observed)
        assert not {r["condition_id"] for r in observed} & {r["condition_id"] for r in correct}
        if arm == "no_video":
            assert donor is None and all(row["teacher_demo_indices"] == [] for row in observed)
            assert len({row["condition_id"] for row in observed}) == 1
        else:
            assert sorted(row["teacher_demo_indices"][0] for row in observed) == list(range(50))
            assert [row["teacher_demo_indices"] for row in observed] == [row["teacher_demo_indices"] for row in correct]
            assert donor // 10 != task // 10 if arm == "cross_suite_wrong" else donor == task
    if arm == "cross_suite_wrong":
        assert set(donors) == set(VALIDATION)


@pytest.mark.parametrize("changes", [dict(role="development_train"), dict(cardinality=2),
                                     dict(init_state_ids=tuple(range(10))), dict(video_pool=tuple(range(49)))])
def test_partial_or_train_control_rounds_are_rejected(changes):
    with pytest.raises(ValueError):
        selection("reversed", **changes)


@pytest.fixture
def frozen_control_assets(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    target = json.loads((ROOT / "configs/pi05_target_data_v1/manifest.json").read_text())
    for row in target["tasks"]:
        if row["global_task_id"] in VALIDATION:
            path = root / "data/datasets" / target["dataset"]["revision"] / row["hdf5"]["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not HDF5; no-video must never open this file")
            row["hdf5"]["bytes"] = path.stat().st_size
    for relative, value in (("configs/pi05_target_data_v1/manifest.json", target),
                            ("configs/libero_24_8_8_v1/protocol.json", json.loads((ROOT / "configs/libero_24_8_8_v1/protocol.json").read_text())),
                            ("configs/pi05_writer_data_v1.json", {"authorities": {"lora_contract": "configs/pi05_lora_v1.json"}})):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    lora_path = root / "configs/pi05_lora_v1.json"
    lora_path.write_text((ROOT / "configs/pi05_lora_v1.json").read_text())
    lora = replace(load_pi05_lora_contract(lora_path), targets=tuple(LoRATarget(f"layers.{index}", 3, 3) for index in range(38)))
    monkeypatch.setattr("ember.pi05_lora.load_pi05_lora_contract", lambda _path: lora)
    monkeypatch.setattr(evaluation, "load_pi05_lora_contract", lambda _path: lora)
    checkpoint = {"path": str(tmp_path / "macro_00000900"), "macro": 900}
    run = {"source": SOURCE, "model_config": dict(MODEL_DEFAULTS), "config": {
        "data": {}, "observer": {"camera_view": "dual", "frame_chunk": 4}, "update_version": materialization.UPDATE_VERSION,
        "execution_precision": "native_bf16_writer_fm_fp32_lora"}}
    monkeypatch.setattr(materialization, "inspect_writer_checkpoint", lambda _path: (run, checkpoint))
    monkeypatch.setattr(evaluation, "inspect_writer_checkpoint", lambda _path: (run, checkpoint))
    tasks = load_learning_tasks(root, VALIDATION, role="validation")
    selected = selection("correct")
    rows = [{"global_task_id": task, "suite": value.suite, "task_id": value.suite_task_id,
             "split_role": "validation", "language": value.authority.language,
             "teacher_source": file_record(value.authority.path), "episodes": planned_episodes(selected, task)}
            for task, value in tasks.items()]
    correct = {"schema_version": materialization.BANK_SCHEMA, "kind": materialization.BANK_KIND, "status": "sealed",
        "arm": "correct", "evaluation_role": "validation", "selection": selected,
        "asset_root": str(root), "source": SOURCE, "writer_checkpoint": checkpoint, "materialization_git": GIT,
        "lora_contract": file_record(lora_path), "method": materialization.method_metadata(run), "tasks": rows,
        "single_complete_rank16": True, "conditions": [{"condition_id": episode["condition_id"]}
            for row in rows for episode in row["episodes"]]}
    anchor = tmp_path / "correct400.json"
    anchor.write_text(json.dumps(correct))
    declaration = DIAGNOSTIC_DECLARATION | {"checkpoint_macro": 900, "paired_correct_manifest": str(anchor)}
    return root, run, checkpoint, declaration, rows, lora


@pytest.mark.parametrize("damage", ["missing", "step", "feedback", "seed", "anchor_checkpoint", "partial_anchor"])
def test_controls_require_explicit_selected_checkpoint_authority_and_the_same_correct400_map(frozen_control_assets, damage):
    root, run, checkpoint, declaration, _, _ = frozen_control_assets
    selected = selection("shuffled")
    if damage == "missing":
        declaration = None
    elif damage == "step":
        checkpoint["macro"] = 600
    elif damage == "feedback":
        declaration["training_feedback"] = True
    elif damage == "seed":
        selected["seed"] += 1
    else:
        path = Path(declaration["paired_correct_manifest"])
        anchor = json.loads(path.read_text())
        if damage == "anchor_checkpoint":
            anchor["writer_checkpoint"]["path"] += "_different"
        else:
            anchor["conditions"].pop()
        path.write_text(json.dumps(anchor))
    with pytest.raises(ValueError):
        inspect_diagnostic_contract(declaration, selection=selected, checkpoint=checkpoint, run=run, asset_root=root)


@pytest.fixture
def frozen_test_assets(frozen_control_assets, tmp_path, request):
    root, run, checkpoint, _, _, lora = frozen_control_assets
    target_path = root / "configs/pi05_target_data_v1/manifest.json"
    target = json.loads(target_path.read_text())
    for row in target["tasks"]:
        if row["global_task_id"] in TEST:
            path = root / "data/datasets" / target["dataset"]["revision"] / row["hdf5"]["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic fixture; no real Test episode is opened")
            row["hdf5"]["bytes"] = path.stat().st_size
    if getattr(request, "param", None) == "coverage":
        protocol_path = "configs/libero_24_8_8_coverage_v1/protocol.json"
        coverage = json.loads((ROOT / "configs/libero_24_8_8_coverage_v1/manifest.json").read_text())
        for relative in (protocol_path, "configs/pi05_source_corpus_v1/source_manifest.json"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((ROOT / relative).read_text())
        canonical = {row["global_task_id"]: row for row in target["tasks"]}
        for row in coverage["tasks"]:
            if row["global_task_id"] in canonical:
                if row["split_role"] == "test":
                    path = root / "data/datasets" / target["dataset"]["revision"] / row["hdf5"]["relative_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic coverage Test RGB fixture")
                    canonical[row["global_task_id"]]["hdf5"]["bytes"] = path.stat().st_size
                row["hdf5"]["bytes"] = canonical[row["global_task_id"]]["hdf5"]["bytes"]
        (root / "configs/libero_24_8_8_coverage_v1/manifest.json").write_text(json.dumps(coverage))
        run["config"]["data"]["protocol"] = protocol_path
    target_path.write_text(json.dumps(target))
    freeze = tmp_path / "method_freeze.json"
    freeze.write_text(json.dumps(METHOD_FREEZE_DECLARATION | {"terminal_macro": 900,
        "writer_checkpoint": checkpoint, "method": materialization.method_metadata(run)}))
    declaration = SEALED_TEST_DECLARATION | {"checkpoint_macro": 900, "method_freeze": str(freeze)}
    return root, run, checkpoint, declaration, lora


@pytest.mark.parametrize("damage", ["missing", "missing_freeze", "missing_file", "macro", "checkpoint", "method",
    "further_training", "further_architecture_changes", "test_gradient_use", "checkpoint_selection", "feedback"])
def test_test_requires_the_exact_frozen_method_and_terminal_checkpoint(frozen_test_assets, damage):
    root, run, checkpoint, declaration, _ = frozen_test_assets
    selected = selection("correct", role="test", task_ids=TEST)
    freeze_path = Path(declaration["method_freeze"])
    freeze = json.loads(freeze_path.read_text())
    if damage == "missing":
        declaration = None
    elif damage == "missing_freeze":
        declaration.pop("method_freeze")
    elif damage == "missing_file":
        declaration["method_freeze"] = str(freeze_path.with_name("absent.json"))
    elif damage == "macro":
        checkpoint = checkpoint | {"macro": 600}
    elif damage == "checkpoint":
        freeze["writer_checkpoint"]["path"] += "_different"
    elif damage == "method":
        freeze["method"]["frame_stride"] = 10
    elif damage == "feedback":
        declaration["training_feedback"] = True
    else:
        freeze[damage] = True
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises((ValueError, OSError)):
        inspect_diagnostic_contract(declaration, selection=selected, checkpoint=checkpoint, run=run, asset_root=root)


@pytest.mark.parametrize("changes", [dict(cardinality=2), dict(init_state_ids=tuple(range(49))),
    dict(video_pool=tuple(range(49))), dict(mode="fixed_per_task"),
    dict(arm="no_video")])
def test_test_cannot_select_partial_rounds_few_shot_or_unregistered_arms(changes):
    values = dict(role="test", task_ids=TEST, cardinality=1, arm="correct", mode="per_init_ordinal",
                  seed=7, init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    with pytest.raises(ValueError):
        selection_contract(**(values | changes))
