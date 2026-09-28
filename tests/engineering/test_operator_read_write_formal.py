"""CPU contract checks for the formal operator comparison, without model execution."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.lora import expected_lora_state_shapes
from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.operator_writer import bank as operator_bank
from ember.operator_writer.bank import (_inspect_continuation_source, _mt_source, assemble_state,
                                        registered_capture, task_rows)
from ember.operator_writer.run import (CHECKPOINTS, CONTINUATION_SPEC_PATH, TASKS,
                                       FormalData, audit, resume_contract_compatible,
                                       specification, train, validate_attempt)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.scene import inspect_registered_scenes, validate_scene_row
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
ASSET = Path("/data1/user/ymdai/projects/EMBER")
SCENES = Path("/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927/scenes")
SEALED = Path("/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1")


def test_full_270_event_stream_and_exact_resume_cursor():
    spec = specification()
    data = FormalData(ASSET, spec, query_labels=False)
    try:
        summary = audit(spec, ASSET)
        assert (summary["updates_per_mode"], summary["conditions_per_mode"],
                summary["queries_per_mode"]) == (270, 1080, 30240)
        assert set(map(int, summary["teacher_order"])) == set(TASKS)
        for task, teachers in summary["teacher_order"].items():
            assert len(teachers) == len(set(teachers)) == 30
        for step in (0, 1, 8, 9, 269):
            jobs = [data.event(step, task) for task in data.tasks_for_step(step)]
            assert len(jobs) == 4 and all(row["visit"] == step // 9 for row in jobs)
            assert all(row["query_offset"] == 1 and len(row["queries"]) == 28 for row in jobs)
            assert all(row == data.event(step, row["task"]) for row in jobs)
        for cursor in CHECKPOINTS:
            state = data.sampler_state() | {"next_step": cursor}
            data.restore(state)
            assert data.next_step == cursor
        with pytest.raises(ValueError):
            data.restore(data.sampler_state() | {"schema_version": "ember_operator_read_write_events_v1"})
    finally:
        data.close()


def test_resume_only_allows_physical_oom_repacking():
    parent = {"git": "one", "mode": "T", "source": {"checkpoint": "source1000"},
              "topology": {"ranks": ["uuid0", "uuid1"]}, "microbatch": 28, "frame_chunk": 8}
    current = deepcopy(parent)
    assert resume_contract_compatible(parent, current)
    current.update(microbatch=14, frame_chunk=4)
    assert resume_contract_compatible(parent, current)
    for changed in ({"mode": "U"}, {"git": "engineering"},
                    {"source": {"checkpoint": "wrong"}},
                    {"topology": {"ranks": ["uuid1", "uuid0"]}},
                    {"microbatch": 13}, {"frame_chunk": 2}):
        candidate = current | changed
        assert not resume_contract_compatible(parent, candidate)


def test_resume_requires_latest_complete_same_arm_ecp(tmp_path):
    spec = {"run_root": str(tmp_path)}
    contract = {"git": "formal", "mode": "T", "source": "source1000",
                "topology": ["uuid0", "uuid1"], "microbatch": 28, "frame_chunk": 8}
    parent = tmp_path / "T/train/attempts/fresh"
    checkpoint = parent / "checkpoints/macro_00000090"
    checkpoint.mkdir(parents=True)
    write_json_atomic(parent / "run_contract.json", contract)
    files = {name: {"bytes": 1} for name in (
        "ecp.safetensors", "trainer_state.pt", "rank_00_state.pt", "rank_01_state.pt")}
    for name in files:
        (checkpoint / name).write_bytes(b"x")
    write_json_atomic(checkpoint / "checkpoint_manifest.json", {
        "stage": "operator_read_write_learning", "run_contract_schema": "ember_operator_read_write_formal_run_v1",
        "next_macro": 90, "world_size": 2, "files": files})
    args = SimpleNamespace(mode="T", resume=checkpoint)
    output = tmp_path / "T/train/attempts/recover1"
    validate_attempt(spec, args, contract, output)
    with pytest.raises(ValueError):
        validate_attempt(spec, SimpleNamespace(mode="U", resume=checkpoint), contract,
                         tmp_path / "U/train/attempts/recover1")
    with pytest.raises(ValueError):
        validate_attempt(spec, args, contract | {"git": "engineering"}, output)
    later = parent / "checkpoints/macro_00000180"
    later.mkdir()
    for name in files:
        (later / name).write_bytes(b"x")
    write_json_atomic(later / "checkpoint_manifest.json", {
        "stage": "operator_read_write_learning", "run_contract_schema": "ember_operator_read_write_formal_run_v1",
        "next_macro": 180, "world_size": 2, "files": files})
    with pytest.raises(ValueError):
        validate_attempt(spec, args, contract, output)


def test_fixed_mt_source_400_scene_and_single_rank128_state():
    spec = specification()
    authority = load_evaluation_authorities(ASSET / spec["source"]["evaluation_config"], ASSET)
    source_checkpoint = ASSET / spec["source"]["checkpoint"]
    source = inspect_source_checkpoint(authority, source_checkpoint.parent.parent,
                                       source_checkpoint, evaluation_mode="formal")
    checkpoint, run, manifest = _mt_source(spec, source)
    assert checkpoint == Path(spec["evaluation"]["mt_checkpoint"])
    assert manifest["consumed"]["next_optimizer_step"] == 300
    assert len(run["tasks"]) == 36
    with pytest.raises(ValueError):
        _mt_source(spec, source | {"model_path": "/wrong/source/policy"})
    tasks, conditions = task_rows(spec, ASSET)
    assert len(tasks) == 8 and len(conditions) == 400
    held = FormalData(ASSET, spec, query_labels=False,
                      task_ids=tuple(spec["evaluation"]["task_ids"]), role="validation")
    try:
        assert held.queries is None and held.rows is None and len(held.tasks) == 8
        with pytest.raises(ValueError):
            held.batch({"task": spec["evaluation"]["task_ids"][0], "queries": []})
    finally:
        held.close()
    assert all(len({e["teacher_demo_indices"][0] for e in task["episodes"]}) == 50 for task in tasks)
    prior = read_json(Path("/data1/user/ymdai/ember_runs/demonstration_transfer_learning_20260927"
                           "/stage1/P/banks/288/manifest.json"))
    assert [[episode["teacher_demo_indices"][0] for episode in task["episodes"]] for task in tasks] == [
        [episode["teacher_demo"] for episode in task["episodes"]] for task in prior["tasks"]]
    inspect_registered_scenes(SCENES, tasks)

    # The materialized T/U state has one A and one B for every target, with no second adapter.
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(ASSET / spec["source"]["lora_contract"]), rank=128)
    shapes = expected_lora_state_shapes(lora)
    shared = {name: torch.empty(shape) for name, shape in shapes.items() if ".lora_A." in name}
    conditional = {name: torch.empty(shape) for name, shape in shapes.items() if ".lora_B." in name}
    assert len(assemble_state(shared, conditional, lora)) == 76
    with pytest.raises(ValueError):
        assemble_state(shared, {name: value for name, value in conditional.items() if name != next(iter(conditional))}, lora)


def test_official_capture_and_scene_route_are_registered(tmp_path):
    spec = specification()
    tasks, _ = task_rows(spec, ASSET)
    rows = [SimpleNamespace(**task, init_state_ids=tuple(range(50))) for task in tasks]
    bank_path = tmp_path / "T" / "banks" / "270" / "manifest.json"
    bank_path.parent.mkdir(parents=True)
    write_json_atomic(bank_path, {"kind": "operator_read_write_lora_bank"})
    capture_path = ROOT / "configs/operator_read_write_v1/official_capture.json"
    selector = read_json(capture_path)
    output = bank_path.parent.parent.parent / "evaluation/correct400"
    capture, stage = registered_capture(
        SimpleNamespace(static_task_lora_manifest=bank_path, role="validation", mode="formal"),
        rows, output, capture_path, selector, None)
    assert len(capture["full_conditions"]) == 8 and stage["full_conditions_only"] is False
    prepared, prepared_stage = _registered_trajectory_capture(
        SimpleNamespace(static_task_lora_manifest=bank_path, role="validation", mode="formal",
                        trajectory_capture_selection=capture_path), rows, output, None, ROOT)
    assert (prepared, prepared_stage) == (capture, stage)
    mt_bank = tmp_path / "MT/banks/300/manifest.json"
    mt_bank.parent.mkdir(parents=True)
    write_json_atomic(mt_bank, {"kind": "operator_read_write_lora_bank"})
    mt_output = tmp_path / "MT/evaluation/correct400"
    mt_capture, mt_stage = registered_capture(
        SimpleNamespace(static_task_lora_manifest=mt_bank, role="validation", mode="formal"),
        rows, mt_output, capture_path, selector, None)
    assert mt_capture["trajectory_root"] == str((mt_output / "trajectories").resolve())
    assert mt_stage == stage
    continuation_bank = tmp_path / "T/banks/450/manifest.json"
    continuation_bank.parent.mkdir(parents=True)
    write_json_atomic(continuation_bank, {"kind": "operator_read_write_lora_bank"})
    continuation_output = tmp_path / "T/evaluation/450/correct400"
    continuation_capture, continuation_stage = registered_capture(
        SimpleNamespace(static_task_lora_manifest=continuation_bank, role="validation", mode="formal"),
        rows, continuation_output, capture_path, selector, None)
    assert len(continuation_capture["full_conditions"]) == 8
    assert continuation_stage == stage
    observed_bank = tmp_path / "T/banks/810/manifest.json"
    observed_bank.parent.mkdir(parents=True)
    write_json_atomic(observed_bank, {"kind": "operator_read_write_lora_bank"})
    observed_output = tmp_path / "T/evaluation/810/correct400"
    observed_capture, observed_stage = registered_capture(
        SimpleNamespace(static_task_lora_manifest=observed_bank, role="validation", mode="formal"),
        rows, observed_output, capture_path, selector, None)
    assert observed_capture["trajectory_root"] == str((observed_output / "trajectories").resolve())
    assert observed_stage == stage
    with pytest.raises(Pi05EvaluationError):
        registered_capture(
            SimpleNamespace(static_task_lora_manifest=continuation_bank, role="validation", mode="formal"),
            rows, output, capture_path, selector, None)
    scene_file = SCENES / "libero_spatial_task_03_state_000.npz"
    contract = {"operator_read_write_scene": {"root": str(SCENES)}}
    row = {"init_state_id": 0, "scene_reference": {
        "path": str(scene_file), "bytes": scene_file.stat().st_size,
        "restoration": "full_model_body_pose_post_dummy_sim_controller_and_dual_rgb_verified"}}
    validate_scene_row(row, tasks[0], contract)
    with pytest.raises(ValueError):
        validate_scene_row({**row, "scene_reference": None}, tasks[0], contract)


def test_sealed_operator_banks_admit_only_their_original_training_source(monkeypatch):
    for mode in ("MT", "T", "U"):
        path = SEALED / mode / "banks" / ("300" if mode == "MT" else "270") / "manifest.json"
        bank = read_json(path)
        tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                                 init_state_ids=tuple(range(50))) for row in bank["tasks"]]
        request = dict(manifest_path=path, source=bank["source"], tasks=tasks,
                       evaluation_role="validation", require_formal=True)
        adapter = inspect_static_task_lora_adapter(**request)
        assert (adapter["arm"], adapter["mode"], len(adapter["conditions"])) == ("correct", mode, 400)
        with pytest.raises(Pi05EvaluationError):
            inspect_static_task_lora_adapter(**(request | {
                "source": bank["source"] | {"checkpoint": "/wrong/source/checkpoint"}}))
        with pytest.raises(ValueError, match="provenance/scope"):
            operator_bank._inspect_scope(bank | {"arm": "wrong"}, read_json(Path(bank["spec"]["path"])),
                                         path, bank["source"],
                                         tuple((row.suite, row.task_id) for row in tasks),
                                         "validation", True, None)

    checkpoint = SEALED / "T/train/attempts/fresh/checkpoints/macro_00000270"
    spec = read_json(operator_bank.SEALED_SPEC_PATH)
    with pytest.raises(ValueError, match="outside this arm"):
        operator_bank.inspect_training_source(spec, checkpoint, "U", sealed_evaluation=True)
    run_path = checkpoint.parent.parent / "run_contract.json"
    original_read = operator_bank.read_json
    monkeypatch.setattr(operator_bank, "read_json", lambda path: (
        original_read(path) | {"git": {"commit": "engineering"}}
        if Path(path) == run_path else original_read(path)))
    with pytest.raises(ValueError, match="numerical identity"):
        operator_bank.inspect_training_source(spec, checkpoint, "T", sealed_evaluation=True)


def test_900_event_stream_preserves_270_and_two_teacher_rounds():
    old_spec, new_spec = specification(), specification(CONTINUATION_SPEC_PATH)
    old = FormalData(ASSET, old_spec, query_labels=False)
    new = FormalData(ASSET, new_spec, query_labels=False)
    try:
        for step in range(270):
            assert old.tasks_for_step(step) == new.tasks_for_step(step)
            for task in old.tasks_for_step(step):
                assert old.event(step, task) == new.event(step, task)
        summary = audit(new_spec, ASSET)
        assert (summary["updates_per_mode"], summary["conditions_per_mode"],
                summary["queries_per_mode"], summary["visits_per_task"]) == (900, 3600, 100800, 100)
        for teachers in summary["teacher_order"].values():
            assert len(teachers[:50]) == len(set(teachers[:50])) == 50
            assert len(teachers[50:]) == len(set(teachers[50:])) == 50
        sealed_state = old.sampler_state() | {"next_step": 270}
        assert new.restore(sealed_state, migrate_sealed_270=True)["cursor"] == 270
        assert new.sampler_state()["teacher_demo_pool"] == list(range(50))
        with pytest.raises(ValueError, match="migration source"):
            new.restore(sealed_state | {"teacher_pool": list(range(29))}, migrate_sealed_270=True)
    finally:
        old.close()
        new.close()


def test_real_sealed_270_is_only_compatible_start_of_continuation(tmp_path):
    spec = specification(CONTINUATION_SPEC_PATH) | {"run_root": str(tmp_path)}
    checkpoint = SEALED / "T/train/attempts/fresh/checkpoints/macro_00000270"
    old = read_json(checkpoint.parent.parent / "run_contract.json")
    data = FormalData(ASSET, spec, query_labels=False)
    try:
        sampler = {key: value for key, value in data.sampler_state().items() if key != "next_step"}
        trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta", mmap=True,
                             weights_only=True)
        migrated = data.restore(trainer["sampler_state"], migrate_sealed_270=True)
        assert migrated["cursor"] == 270
        assert trainer["scheduler"]["last_epoch"] == 270
        assert trainer["optimizer"]["param_groups"]
    finally:
        data.close()
    contract = {**old, "spec": str(CONTINUATION_SPEC_PATH), "events": spec["events"],
                "sampler": sampler, "continuation": spec["continuation"],
                "parent_checkpoint": str(checkpoint),
                "git": {"commit": "new-frozen", "branch": "", "dirty_paths": [],
                        "pushed_ref": "origin/codex/demonstration-transfer"}}
    args = SimpleNamespace(mode="T", resume=checkpoint)
    output = tmp_path / "T/train/attempts/cpu-contract-only"
    validate_attempt(spec, args, contract, output)
    with pytest.raises(ValueError, match="source, parameters or optimization"):
        validate_attempt(spec, args, contract | {"source": {"checkpoint": "wrong"}}, output)
    with pytest.raises(ValueError, match="sealed same-arm"):
        validate_attempt(spec, SimpleNamespace(mode="U", resume=checkpoint), contract, output)
    later = tmp_path / "T/train/attempts/continuation/checkpoints/macro_00000360"
    later.mkdir(parents=True)
    files = {name: {"bytes": 1} for name in (
        "ecp.safetensors", "trainer_state.pt", "rank_00_state.pt", "rank_01_state.pt")}
    for name in files:
        (later / name).write_bytes(b"x")
    write_json_atomic(later / "checkpoint_manifest.json", {
        "stage": "operator_read_write_learning", "run_contract_schema": "ember_operator_read_write_formal_run_v1",
        "next_macro": 360, "world_size": 2, "files": files})
    with pytest.raises(ValueError, match="latest complete same-arm"):
        validate_attempt(spec, args, contract, output)
    with pytest.raises(ValueError, match="requires the continuation1350 spec"):
        train(specification(), SimpleNamespace())


def test_world3_assignment_retains_four_equal_global_condition_weights(monkeypatch):
    from ember.writer.replay import sum_writer_gradients
    from ember.writer.task_execution import condition_assignment
    import torch.distributed as dist

    assignment = condition_assignment(tuple(range(4)), {0: 5, 1: 4, 2: 3, 3: 2}, world_size=3)
    assert sorted(map(len, assignment)) == [1, 1, 2]
    gradients = [2.0, 4.0, 6.0, 8.0]
    locals_ = [sum(gradients[index] / 4 for index in group) for group in assignment]
    expected = sum(gradients) / 4
    assert sum(locals_) == expected
    calls = []
    def reduce_sum(value, *, op):
        calls.append(op)
        value.fill_(expected)
    monkeypatch.setattr(dist, "all_reduce", reduce_sum)
    for local in locals_:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        parameter.grad = torch.tensor(local)
        sum_writer_gradients((parameter,), world_size=3)
        assert parameter.grad.item() == expected
    assert calls == [dist.ReduceOp.SUM] * 3


def test_complete_450_ecp_can_feed_bank_before_train_completion(tmp_path):
    spec = specification(CONTINUATION_SPEC_PATH) | {"run_root": str(tmp_path)}
    data = FormalData(ASSET, spec, query_labels=False)
    try:
        sampler = {key: value for key, value in data.sampler_state().items() if key != "next_step"}
    finally:
        data.close()
    attempt = tmp_path / "T/train/attempts/continuation"
    parent = attempt / "checkpoints/macro_00000360"
    checkpoint = attempt / "checkpoints/macro_00000450"
    for macro, path in ((360, parent), (450, checkpoint)):
        path.mkdir(parents=True)
        files = {name: {"bytes": 1} for name in (
            "ecp.safetensors", "rank_00_state.pt", "rank_01_state.pt")}
        for name in files:
            (path / name).write_bytes(b"x")
        torch.save({"schema_version": "ember_ecp_checkpoint_v1", "stage": "operator_read_write_learning",
                    "next_macro": macro, "metrics_rows": macro,
                    "optimizer": {"param_groups": [{"lr": 0.0003}]},
                    "scheduler": {"last_epoch": macro}, "scaler": None,
                    "training_state": {"updates": macro, "mode": "T"},
                    "sampler_state": sampler | {"next_step": macro}}, path / "trainer_state.pt")
        files["trainer_state.pt"] = {"bytes": (path / "trainer_state.pt").stat().st_size}
        write_json_atomic(path / "checkpoint_manifest.json", {
            "stage": "operator_read_write_learning", "run_contract_schema": "ember_operator_read_write_formal_run_v1",
            "next_macro": macro, "world_size": 2, "files": files})
    run = {"schema_version": "ember_operator_read_write_formal_run_v1", "stage": "operator_read_write_learning",
           "mode": "T", "spec": str(operator_bank.CONTINUATION_FROZEN_SPEC_PATH),
           "source_trainable": 0, "operator": spec["operator"], "optimizer": spec["optimization"],
           "events": spec["events"], "continuation": spec["continuation"], "sampler": sampler,
           "parent_checkpoint": str(parent),
           "git": operator_bank.CONTINUATION_TRAINING_GIT}
    write_json_atomic(attempt / "run_contract.json", run)
    write_json_atomic(attempt / "resume_provenance.json", {"checkpoint": str(parent)})
    (attempt / "metrics.jsonl").write_text("".join(f'{{"update": {step}}}\n' for step in range(1, 461)))
    assert not (attempt / "completion.json").exists()
    assert _inspect_continuation_source(spec, checkpoint, "T", sealed_evaluation=True) == run
    write_json_atomic(attempt / "run_contract.json", run | {"events": {"seed": -1}})
    with pytest.raises(ValueError, match="source/ECP"):
        _inspect_continuation_source(spec, checkpoint, "T", sealed_evaluation=True)


def test_real_810_evaluation_reads_only_old_complete_same_arm_ecp(monkeypatch):
    spec = specification(CONTINUATION_SPEC_PATH)
    root = Path(spec["run_root"])
    for mode in ("T", "U"):
        checkpoint = root / mode / "train/attempts/continuation/checkpoints/macro_00000810"
        run_path = checkpoint.parent.parent / "run_contract.json"
        run = read_json(run_path)
        assert read_json(checkpoint / "checkpoint_manifest.json")["next_macro"] == 810
        assert operator_bank.inspect_training_source(spec, checkpoint, mode, sealed_evaluation=True) == run
        other = "U" if mode == "T" else "T"
        with pytest.raises(ValueError, match="same-arm"):
            operator_bank.inspect_training_source(spec, checkpoint, other, sealed_evaluation=True)
        with monkeypatch.context() as patch:
            original = operator_bank.read_json
            patch.setattr(operator_bank, "read_json", lambda path: (
                original(path) | {"source": {"model_path": "/wrong"}}
                if Path(path) == run_path else original(path)))
            with pytest.raises(ValueError, match="source/ECP"):
                operator_bank.inspect_training_source(spec, checkpoint, mode, sealed_evaluation=True)
        with monkeypatch.context() as patch:
            original = operator_bank.read_json
            patch.setattr(operator_bank, "read_json", lambda path: (
                original(path) | {"git": {"commit": "wrong"}}
                if Path(path) == run_path else original(path)))
            with pytest.raises(ValueError, match="source/ECP"):
                operator_bank.inspect_training_source(spec, checkpoint, mode, sealed_evaluation=True)


def test_existing_900_bank_still_consumes_original_training_source():
    spec = specification(CONTINUATION_SPEC_PATH)
    path = Path(spec["run_root"]) / "T/banks/900/manifest.json"
    bank = read_json(path)
    tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                             init_state_ids=tuple(range(50))) for row in bank["tasks"]]
    adapter = inspect_static_task_lora_adapter(
        manifest_path=path, source=bank["source"], tasks=tasks,
        evaluation_role="validation", require_formal=True)
    assert adapter["mode"] == "T" and len(adapter["conditions"]) == 400
