"""Real-frame control wiring, frozen pairing, and source-identity execution."""

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.eval_adapters import inspect_static_task_lora_adapter
from ember.lora import LoRATarget, expected_lora_state_shapes
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer import evaluation, materialization, materialization_workers, runtime
from ember.writer.learning_data import load_learning_tasks
from ember.writer.materialization import condition_id, file_record, planned_episodes, selection_contract
from ember.writer.runtime import MODEL_DEFAULTS
from ember.writer.video_controls import (CONTROL_ARMS, DIAGNOSTIC_DECLARATION, METHOD_FREEZE_DECLARATION,
    SEALED_TEST_DECLARATION, control_provenance, controlled_frames, inspect_diagnostic_contract, video_task_id)
from ember.pi05_eval.native_reader_transfer import select_tasks, validate_bank
from ember.pi05_source_checkpoint import read_json
from ember.writer.native_reader_transfer import (
    authority as native_authority, composed_state, expected_bank, parent_checkpoint, validate_materialization,
)
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


@pytest.mark.parametrize("arm", CONTROL_ARMS[:-1])
@pytest.mark.parametrize('camera_view', ['agentview', 'dual'])
@pytest.mark.parametrize("role", ["validation", "test"])
def test_real_declared_camera_pixels_are_reordered_before_complete_reads(tmp_path, arm, camera_view, role):
    ids = TEST if role == "test" else VALIDATION
    task_id = ids[0]
    selected = selection(arm, role=role, task_ids=ids)
    tasks = load_learning_tasks(ROOT, ids, role=role)
    rows = {task: {"suite": value.suite, "task_id": value.suite_task_id, "split_role": role,
                   "teacher_source": {"path": str(value.authority.path), "bytes": value.authority.expected_bytes}}
            for task, value in tasks.items()}
    control = control_provenance(selected, task_id, rows)
    donor, demo = tasks[control["video_global_task_id"]], 9
    donor = replace(donor, episode_lengths=(17,) * 50)
    indices = np.array([0, 5, 10, 15, 16])
    frames = np.arange(5 * 2 * 3 * 2 * 2).reshape(5, 2, 3, 2, 2).astype(np.float32)
    if camera_view == 'agentview':
        frames = frames[:, 0]
    reads, prepared = [], []

    def load(task, index):
        reads.append(("RGB", task, index))
        assert task == donor.authority.task_id and index == demo
        return SimpleNamespace(frames=frames, frame_indices=indices, raw_frame_count=17)

    def prepare(pixels, displayed, language):
        prepared.append((pixels, displayed, language))
        return prepared[-1]

    lora = replace(load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json"),
                   targets=(LoRATarget("linear", 3, 3),))
    def compile(value):
        assert value is prepared[-1] and not torch.is_grad_enabled()
        reads.append(("writer", value[2]))
        return {key: torch.zeros(shape) for key, shape in expected_lora_state_shapes(lora).items()}
    current = SimpleNamespace(lora=lora, prepare=prepare, compile=compile)
    # Exercise the actual resident-worker donor handoff and canonical compiler.
    worker = object.__new__(materialization_workers.ResidentCompiler)
    worker.runtime, worker.store, worker.tasks = current, SimpleNamespace(load=load), tasks | {donor.authority.task_id: donor}
    worker.output, worker.record = tmp_path, {"path": "/frozen/900", "macro": 900}
    record = worker.compile({"task": task_id, "demos": [demo], "control": control})
    content, displayed, evidence = controlled_frames(indices, control=control, demo=demo)
    assert evidence['transform_stage'] == 'declared_real_camera_RGB_before_complete_Writer_forward'
    assert torch.equal(prepared[0][0][0], torch.from_numpy(frames)[content])
    assert torch.equal(prepared[0][1][0], displayed)
    assert prepared[0][2] == tasks[task_id].authority.language
    assert [value[0] for value in reads] == ["RGB", "writer"]
    assert record["video_control"] == control
    assert all(record["teacher_videos"][0][key] == value for key, value in evidence.items())
    evaluation._validate_video_frames(record["teacher_videos"], [demo], donor.episode_lengths, control)
    for field in ("source_frame_indices", "frame_indices", "frame_permutation", "frame_order_seed"):
        damaged = copy.deepcopy(record["teacher_videos"])
        damaged[0][field] = -1 if field == "frame_order_seed" else [-1]
        with pytest.raises(ValueError):
            evaluation._validate_video_frames(damaged, [demo], donor.episode_lengths, control)
    wrong = replace(donor, authority=replace(donor.authority, task_id=99))
    with pytest.raises(ValueError, match="actual donor"):
        materialization._compile_condition(current, worker.store, tasks[task_id], [demo], tmp_path,
                                            worker.record, control=control, video_task=wrong)


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
    monkeypatch.setattr(materialization, "git_state", lambda _root: GIT)
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


@dataclass(frozen=True)
class _NativeTask:
    suite: str
    task_id: int
    init_state_ids: tuple[int, ...]


def test_native_reader_exact_parent_partition_and_mix_request():
    spec = native_authority()
    n, w = parent_checkpoint("1"), parent_checkpoint("0")
    state, roles = composed_state(n, w)
    assert len(state) == 622
    assert (len(roles["N_keys"]), len(roles["W_keys"]), len(roles["fixed_keys"])) == (432, 113, 77)
    run, _ = materialization.inspect_writer_checkpoint(w)
    selected = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=20260911,
        init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    output = Path(spec["resources"]["study_root"]) / "materialization" / "N1_W0"
    checked = validate_materialization(cell="N1_W0", checkpoint=w, output=output,
                                       selection=selected, run=run)
    assert checked["partition"] == roles
    with pytest.raises(ValueError):
        validate_materialization(cell="N1_W0", checkpoint=w, output=output,
                                 selection=selected | {"seed": 5}, run=run)
    with pytest.raises(ValueError):
        validate_materialization(cell="N0_W0", checkpoint=w, output=output,
                                 selection=selected, run=run)


def test_native_reader_endpoint_bank_and_pilot_remaining_cases():
    cell = "N0_W0"
    bank = expected_bank(cell)
    manifest = read_json(bank)
    validate_bank(manifest, bank, cell, "unused-current-commit")
    with pytest.raises(ValueError):
        validate_bank(manifest, bank, "N1_W1", "unused-current-commit")
    tasks = [_NativeTask("libero_object", 4, tuple(range(50))),
             _NativeTask("libero_goal", 1, tuple(range(50))),
             _NativeTask("libero_goal", 2, tuple(range(50)))]
    for stage, expected_states in (("pilot", (0,)), ("remaining", tuple(range(1, 50)))):
        output = Path(native_authority()["resources"]["study_root"]) / "evaluation" / stage / cell
        args = SimpleNamespace(native_reader_transfer_cell=cell, output_dir=output,
            config=ROOT / "configs/relational_support_causality_v1/evaluation.json",
            role="development_train", mode="formal", state_count=50,
            static_task_lora_manifest=bank)
        selected, capture, predicates = select_tasks(args, tasks, ROOT)
        assert len(selected) == 2 and all(task.init_state_ids == expected_states for task in selected)
        assert len(capture["full_conditions"]) == 2
        assert predicates["full_conditions_only"] is False


def test_no_video_materializes_without_runtime_or_pixels_and_executes_source_identity(frozen_control_assets, tmp_path, monkeypatch):
    root, _, checkpoint, declaration, rows, _ = frozen_control_assets

    def forbidden(*_args, **_kwargs):
        pytest.fail("no-video attempted to construct a model or read teacher RGB")

    monkeypatch.setattr(runtime, "build_runtime", forbidden)
    monkeypatch.setattr(materialization, "RawTeacherVideoStore", forbidden)
    monkeypatch.setattr(materialization, "MaterializationWorkers", forbidden)
    path, = materialization.materialize_requests(asset_root=root, device=torch.device("cpu"), requests=[{
        "checkpoint": checkpoint["path"], "output": str(tmp_path / "identity"), "role": "validation",
        "task_ids": VALIDATION, "k": 1, "arm": "no_video", "state_count": 50, "seed": 7,
        "diagnostic_contract": declaration}])
    keys = [(row["suite"], row["task_id"]) for row in rows]
    adapter = evaluation.inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=keys,
                                                     evaluation_role="validation", require_formal=True)
    assert len(adapter["conditions"]) == 8 and sum(len(row["episodes"]) for row in adapter["tasks"]) == 400
    assert adapter["information_wall"]["total_writer_invocations"] == 0
    assert adapter["information_wall"]["materialization_rgb_video_reads"] == 0
    assert adapter["materialization_execution"]["workers"] == 0
    assert adapter["method"]["deployment_frozen_source_vjp"] is False
    assert all(row["teacher_videos"] == [] for row in adapter["conditions"])

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Linear(3, 3) for _ in range(38))

        def predict_action_chunk(self, batch, *, noise, num_steps):
            assert num_steps == 10
            return sum(layer(batch["input"]) for layer in self.layers) / len(self.layers) + noise

    policy = Policy().eval()
    batch, noise = {"input": torch.randn(2, 3)}, torch.randn(2, 3)
    expected = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
    execution = evaluation.FrozenHorizonWriterAdapter(policy=policy, source=SOURCE, evaluation_adapter=adapter,
        task_keys=keys, device=torch.device("cpu"), require_formal=True)
    prepared = [execution.prepare_episode(suite=suite, task_id=task, init_state_id=0) for suite, task in keys[:2]]
    assert all(item.evidence["diagnostic_contract"] == adapter["diagnostic_contract"] for item in prepared)
    execution.install(prepared[0])
    torch.testing.assert_close(policy.predict_action_chunk(batch, noise=noise, num_steps=10), expected)
    torch.testing.assert_close(execution.predict_action_chunk(prepared, batch, noise=noise, num_steps=10), expected)
    execution.close()
    condition = adapter["conditions"][0]
    state = load_file(condition["adapter"]["path"])
    assert len(state) == 76 and all(value.count_nonzero() == 0 for value in state.values())
    next(iter(state.values())).fill_(1.)
    save_file(state, condition["adapter"]["path"], metadata=materialization.adapter_metadata(condition["condition_id"], checkpoint))
    with pytest.raises(ValueError, match="zero-delta identity"):
        evaluation._inspect_adapter_file(condition, checkpoint, execution.lora)


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


@pytest.fixture
def test_cpu_compiler(frozen_test_assets, monkeypatch):
    from ember.task_protocol import load_task_authorities
    root, run, _, _, lora = frozen_test_assets
    protocol = run["config"]["data"].get("protocol")
    _, manifest = load_task_authorities(root, protocol)
    test_ids = tuple(row["global_task_id"] for row in manifest["tasks"] if row["split_role"] == "test")
    tasks = load_learning_tasks(root, test_ids, role="test", protocol_path=protocol)
    reads, stages = [], []

    def load(task, demo):
        assert task in tasks and 0 <= demo < 50
        reads.append((task, demo))
        raw = tasks[task].episode_lengths[demo]
        indices = list(range(0, raw, 5))
        if indices[-1] != raw - 1:
            indices.append(raw - 1)
        return SimpleNamespace(raw_frame_count=raw, frame_indices=np.array(indices),
            frames=np.ones((len(indices), 2, 3, 2, 2), dtype=np.float32))

    def prepare(frames, indices, language):
        assert len(frames) == len(indices) == 1
        assert frames[0].shape[1:] == (2, 3, 2, 2)
        assert language in {task.authority.language for task in tasks.values()}
        stages.append("prepare_RGB_indices_language")
        return frames, indices, language

    def compile(condition):
        assert not torch.is_grad_enabled() and len(condition) == 3
        stages.append("writer")
        return {key: torch.full(shape, .01) for key, shape in expected_lora_state_shapes(lora).items()}
    current = SimpleNamespace(lora=lora, prepare=prepare, compile=compile)

    class CPUWorkers:
        def __init__(self, *, config, devices, **_kwargs):
            self.config, self.devices = config, devices

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def compile(self, request, jobs):
            _, _, record, selected_tasks, output, transfer = request
            assert transfer is None
            worker = object.__new__(materialization_workers.ResidentCompiler)
            worker.runtime, worker.store, worker.tasks = current, SimpleNamespace(load=load), selected_tasks
            worker.output, worker.record = output, record
            for job in jobs:
                yield job, worker.compile(job)

    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU contract check attempted to load a real model or teacher episode")

    monkeypatch.setattr(materialization, "MaterializationWorkers", CPUWorkers)
    monkeypatch.setattr(materialization, "RawTeacherVideoStore", forbidden)
    monkeypatch.setattr(runtime, "build_runtime", forbidden)
    return reads, stages


@pytest.mark.parametrize("ids", [TEST[:-1], VALIDATION, (0, *TEST[1:])])
def test_test_task_subsets_and_wrong_splits_fail_before_video_reads(frozen_test_assets, test_cpu_compiler, tmp_path, ids):
    root, _, checkpoint, declaration, _ = frozen_test_assets
    with pytest.raises(ValueError, match="fixed .*split|test8"):
        materialization.materialize_requests(asset_root=root, device=torch.device("cpu"), requests=[{
            "checkpoint": checkpoint["path"], "output": str(tmp_path / "rejected"), "role": "test",
            "task_ids": ids, "k": 1, "diagnostic_contract": declaration}])
    assert test_cpu_compiler == ([], [])
    assert not (tmp_path / "rejected").exists()


def test_complete_frozen_test400_uses_the_canonical_compiler_and_official_adapter(frozen_test_assets,
    test_cpu_compiler, tmp_path):
    root, _, checkpoint, declaration, _ = frozen_test_assets
    with pytest.raises(ValueError, match="fixed target split"):
        load_learning_tasks(root, TEST)
    path, = materialization.materialize_requests(asset_root=root, device=torch.device("cpu"), requests=[{
        "checkpoint": checkpoint["path"], "output": str(tmp_path / "test400"), "role": "test",
        "task_ids": TEST, "k": 1, "arm": "correct", "state_count": 50, "init_state_ids": list(range(50)), "seed": 7,
        "diagnostic_contract": declaration}])
    manifest = json.loads(path.read_text())
    requests = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"], init_state_ids=tuple(range(50)))
                for row in manifest["tasks"]]
    adapter = inspect_static_task_lora_adapter(manifest_path=path, source=SOURCE, tasks=requests,
        evaluation_role="test", require_formal=True)
    assert len(adapter["tasks"]) == 8 and len(adapter["conditions"]) == 400
    assert all(row["split_role"] == "test" and len(row["episodes"]) == 50 for row in adapter["tasks"])
    assert set(test_cpu_compiler[0]) == {(task, demo) for task in TEST for demo in range(50)}
    assert len(test_cpu_compiler[0]) == 400
    assert test_cpu_compiler[1] == ["prepare_RGB_indices_language", "writer"] * 400
    assert adapter["information_wall"]["total_writer_invocations"] == 400
    assert adapter["information_wall"]["materialization_rgb_video_reads"] == 400
    assert adapter["diagnostic_contract"] == SEALED_TEST_DECLARATION | {"checkpoint_macro": 900,
        "method_freeze": file_record(Path(declaration["method_freeze"]))}
    first = adapter["tasks"][0]
    evidence = evaluation.episode_evidence(adapter, first, first["episodes"][0])
    assert evidence["diagnostic_contract"] == adapter["diagnostic_contract"]
    assert len(load_file(adapter["conditions"][0]["adapter"]["path"])) == 76
    keys = [(row.suite, row.task_id) for row in requests]
    with pytest.raises(Pi05EvaluationError, match="fixed init states"):
        evaluation.inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=keys,
            evaluation_role="test", require_formal=False,
            task_init_state_ids={key: tuple(range(49)) for key in keys})
    for field in ("validation_test_gradients", "deployment_loss_or_optimizer"):
        damaged = copy.deepcopy(manifest)
        damaged["information_wall"][field] = True
        with pytest.raises(ValueError, match="information wall"):
            evaluation.validate_information_wall(damaged)
    freeze_path = Path(declaration["method_freeze"])
    freeze = json.loads(freeze_path.read_text())
    freeze["further_training"] = True
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(Pi05EvaluationError, match="freeze"):
        evaluation.inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=keys,
            evaluation_role="test", require_formal=True)


@pytest.fixture
def paired_test_bank(frozen_test_assets, test_cpu_compiler, tmp_path):
    from ember.task_protocol import load_task_authorities
    root, run, checkpoint, declaration, _ = frozen_test_assets
    _, manifest = load_task_authorities(root, run["config"]["data"].get("protocol"))
    ids = [row["global_task_id"] for row in manifest["tasks"] if row["split_role"] == "test"]
    path, = materialization.materialize_requests(asset_root=root, device=torch.device("cpu"), requests=[{
        "checkpoint": checkpoint["path"], "output": str(tmp_path / "paired_correct"), "role": "test",
        "task_ids": ids, "k": 1, "arm": "correct", "state_count": 50, "seed": 7,
        "diagnostic_contract": declaration}])
    return path, declaration | {"paired_correct_manifest": str(path)}, ids


@pytest.mark.parametrize("frozen_test_assets", ["coverage"], indirect=True)
@pytest.mark.parametrize("arm", ("same_task_other", "cross_suite_wrong", "shuffled", "reversed"))
def test_frozen_coverage_test_controls_use_same_checkpoint_full_pairing_and_real_compiler(
    frozen_test_assets, test_cpu_compiler, paired_test_bank, tmp_path, arm):
    root, _, checkpoint, _, _ = frozen_test_assets
    anchor, declaration, ids = paired_test_bank
    assert ids == [8, 9, 10, 18, 24, 27, 30, 33]
    reference = json.loads(anchor.read_text())
    path, = materialization.materialize_requests(asset_root=root, device=torch.device("cpu"), requests=[{
        "checkpoint": checkpoint["path"], "output": str(tmp_path / arm), "role": "test",
        "task_ids": ids, "k": 1, "arm": arm, "state_count": 50, "seed": 7,
        "diagnostic_contract": declaration}])
    keys = [(row["suite"], row["task_id"]) for row in reference["tasks"]]
    adapter = evaluation.inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=keys,
        evaluation_role="test", require_formal=True)
    assert len(adapter["conditions"]) == 400
    assert adapter["task_protocol"] == "configs/libero_24_8_8_coverage_v1/protocol.json"
    assert adapter["diagnostic_contract"]["paired_correct_manifest"] == file_record(anchor)
    assert len(test_cpu_compiler[0]) == 800
    for actual, correct in zip(adapter["tasks"], reference["tasks"], strict=True):
        assert actual["split_role"] == "test"
        assert sorted(row["teacher_demo_indices"][0] for row in actual["episodes"]) == list(range(50))
        for row, paired in zip(actual["episodes"], correct["episodes"], strict=True):
            assert row["init_state_id"] == paired["init_state_id"]
            assert row["paired_correct_demos"] == paired["paired_correct_demos"]
            assert row["paired_other_demos"] == paired["paired_other_demos"]
            assert (row["teacher_demo_indices"] != paired["teacher_demo_indices"]) == (arm == "same_task_other")
    if arm != "same_task_other":
        for condition in adapter["conditions"]:
            control = condition["video_control"]
            assert control["language_split_role"] == control["video_split_role"] == "test"
            if arm == "cross_suite_wrong":
                assert control["video_global_task_id"] // 10 != control["language_global_task_id"] // 10
            elif arm == "reversed":
                video = condition["teacher_videos"][0]
                assert video["source_frame_indices"] == list(reversed(video["frame_indices"]))
    # Evaluator rechecks the immutable pairing authority after materialization.
    damaged = json.loads(anchor.read_text())
    damaged["selection"]["seed"] += 1
    anchor.write_text(json.dumps(damaged))
    with pytest.raises(Pi05EvaluationError, match="manifest changed|checkpoint and map"):
        evaluation.inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=keys,
            evaluation_role="test", require_formal=True)


@pytest.mark.parametrize("damage", ["missing_pair", "unsealed_pair", "seed", "protocol", "partial", "feedback"])
def test_test_controls_reject_unsealed_or_changed_correct_reference(
    frozen_test_assets, paired_test_bank, damage):
    root, run, checkpoint, _, _ = frozen_test_assets
    path, declaration, ids = paired_test_bank
    selected = selection("same_task_other", role="test", task_ids=ids)
    reference = json.loads(path.read_text())
    if damage == "missing_pair":
        declaration.pop("paired_correct_manifest")
    elif damage == "feedback":
        declaration["training_feedback"] = True
    elif damage == "seed":
        selected["seed"] += 1
    elif damage == "protocol":
        reference["task_protocol"] = "another/protocol.json"
    elif damage == "partial":
        reference["conditions"].pop()
    else:
        reference["diagnostic_contract"] = None
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        inspect_diagnostic_contract(declaration, selection=selected, checkpoint=checkpoint, run=run, asset_root=root)
