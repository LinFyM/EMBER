from collections import Counter
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.source_sft.control import (
    checkpoint_declared, clamped_lr_multiplier, validate_coverage_manifest,
)
from ember.source_sft.contract import (
    _target_tasks, load_source_sft_config, reconcile_resume_contract,
    resolve_runtime,
)
from ember.writer.errors import WriterModelError
from ember.source_sft.sampler import HierarchicalMixedBatchSampler

CONTROL = dict(kind="validation_early_stopping", checkpoint_interval=25, validation_interval=50)


def test_dynamic_contract_extension_preserves_checkpoint_hash(tmp_path):
    contract = dict(git=dict(commit="same", branch=""), training_control=CONTROL,
                    runtime=dict(selected_stop_step=50, total_steps=50, checkpoint_steps=[25, 50], world_size=4))
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    args = SimpleNamespace(output_dir=tmp_path, resume=tmp_path / "checkpoints/step_00000050")
    candidate = deepcopy(contract)
    candidate["runtime"].update(selected_stop_step=1500, total_steps=1500,
                                checkpoint_steps=list(range(25, 1501, 25)))
    assert reconcile_resume_contract(args, candidate) == contract
    assert checkpoint_declared(contract, 1500)
    assert not checkpoint_declared(contract, 1501)
    candidate["runtime"]["world_size"] = 3
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, candidate)


def test_explicit_physical_resume_preserves_original_scientific_contract(tmp_path):
    existing = dict(git=dict(commit="old", branch=""), training_control=CONTROL,
                    runtime=dict(selected_stop_step=50, total_steps=50,
                                 checkpoint_steps=[25, 50], world_size=1,
                                 per_rank_batch_size=64, gradient_accumulation_steps=9,
                                 microbatch_sizes_by_rank=[[64] * 9],
                                 num_workers_per_rank=2, rank_topology={"0": "gpu01:0"},
                                 effective_global_batch_size=576))
    (tmp_path / "run_contract.json").write_text(json.dumps(existing))
    args = SimpleNamespace(output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000050", allow_physical_resume=True,
        allow_contract_compatible_code_resume=True)
    candidate = deepcopy(existing)
    candidate["git"]["commit"] = "new"
    candidate["runtime"].update(selected_stop_step=100, total_steps=100,
        checkpoint_steps=[25, 50, 75, 100], world_size=4,
        per_rank_batch_size=64, gradient_accumulation_steps=3,
        microbatch_sizes_by_rank=[[64, 64, 16]] * 4,
        rank_topology={str(rank): f"gpu02:{rank}" for rank in range(4)},
        physical_packing="task_striped")
    assert reconcile_resume_contract(args, candidate) == existing
    candidate["runtime"]["effective_global_batch_size"] = 575
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, candidate)


def test_code_compatible_resume_accepts_only_identical_tokenizer_manifest_move(tmp_path):
    old_manifest = tmp_path / "old" / "tokenizer.json"
    new_manifest = tmp_path / "new" / "tokenizer.json"
    for path in (old_manifest, new_manifest):
        path.parent.mkdir()
        path.write_text('{"tokenizer": "same"}')
    existing = dict(git=dict(commit="old", branch=""), training_control=CONTROL,
                    runtime=dict(selected_stop_step=50, total_steps=50,
                                 checkpoint_steps=[25, 50]),
                    tokenizer=dict(path="canonical.model", bytes=123,
                                   manifest_path=str(old_manifest)))
    (tmp_path / "run_contract.json").write_text(json.dumps(existing))
    args = SimpleNamespace(output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000050",
        allow_contract_compatible_code_resume=True)
    candidate = deepcopy(existing)
    candidate["git"]["commit"] = "new"
    candidate["runtime"].update(selected_stop_step=100, total_steps=100,
                                checkpoint_steps=[25, 50, 75, 100])
    candidate["tokenizer"]["manifest_path"] = str(new_manifest)
    assert reconcile_resume_contract(args, candidate) == existing
    new_manifest.write_text('{"tokenizer": "changed"}')
    with pytest.raises(WriterModelError, match="tokenizer authority changed"):
        reconcile_resume_contract(args, candidate)
    new_manifest.write_text(old_manifest.read_text())
    candidate["tokenizer"]["path"] = "different.model"
    with pytest.raises(WriterModelError, match="tokenizer authority changed"):
        reconcile_resume_contract(args, candidate)


def test_formal_mtbc_resolves_four_cards_only_for_explicit_checkpoint_migration(monkeypatch):
    import ember.source_sft.contract as module
    root = Path(__file__).resolve().parents[1]
    config = load_source_sft_config(root / "configs/libero_24_8_8_coverage_v1/mtbc.json")
    monkeypatch.setattr(module, "git_state", lambda _: {"branch": "", "commit": "new"})
    monkeypatch.setattr(module, "git_state_is_clean_pushed_or_frozen_authority", lambda _: True)
    args = SimpleNamespace(stage="development", mode="formal", total_steps=None,
        checkpoint_steps=None, stop_after_step=100, batch_size=None,
        gradient_accumulation_steps=None, resume=Path("step_00000050"),
        allow_contract_compatible_code_resume=True, allow_physical_resume=False,
        physical_packing="task_striped")
    context = SimpleNamespace(world_size=4, numa_node=0, cpu_affinity=[0])
    with pytest.raises(WriterModelError, match="task-striped packing requires physical resume"):
        resolve_runtime(args, config, context)
    args.allow_physical_resume = True
    assert resolve_runtime(args, config, context) == (100, 64, (25, 50, 75, 100))
    assert args.gradient_accumulation_steps == 3


def test_clamped_lr_and_exact_scheduler_resume():
    multiplier = partial(clamped_lr_multiplier, warmup=150, decay=1200, peak=1e-4, floor=1e-5)
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([p], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    for _ in range(50):
        optimizer.step(); scheduler.step()
    state = deepcopy(scheduler.state_dict())
    optim_state = deepcopy(optimizer.state_dict())
    expected = []
    for _ in range(50):
        optimizer.step(); scheduler.step(); expected.append(scheduler.get_last_lr())
    optimizer.load_state_dict(optim_state); scheduler.load_state_dict(state)
    actual = []
    for _ in range(50):
        optimizer.step(); scheduler.step(); actual.append(scheduler.get_last_lr())
    assert actual == expected
    assert multiplier(1200) == multiplier(1800) == multiplier(2400) == pytest.approx(0.1)
    assert all(multiplier(i) >= multiplier(i + 1) for i in range(150, 1500))


def test_36_tasks_equal_and_resume_independent_of_endpoint():
    frame_index = [(task, demo, 0) for task in range(36) for demo in range(50)]
    dataset = SimpleNamespace(frame_index=frame_index, task_episode_rows={
        task: {demo: (task * 50 + demo,) for demo in range(50)} for task in range(36)})
    def sampler(start, stop):
        return HierarchicalMixedBatchSampler(dataset, task_ids=range(36), logical_world_size=4,
            logical_per_rank_batch_size=144, per_rank_batch_size=144, gradient_accumulation_steps=1,
            start_step=start, stop_step=stop, rank=0, world_size=4, seed=7)
    initial, resumed = sampler(0, 50), sampler(50, 1600)
    rows = resumed.global_rows_for_step(1234)
    assert Counter(frame_index[row][0] for row in rows) == {task: 16 for task in range(36)}
    assert initial.global_rows_for_step(50) == resumed.global_rows_for_step(50)
    assert initial.resume_contract() == resumed.resume_contract()


def test_explicit_allowlist_and_held_paths_never_selected(tmp_path, monkeypatch):
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    aux = [2, 3, 11, 15, 16, 22, 24, 33, 55, 56, 57, 61]
    protocol = dict(version="libero_24_8_8_coverage_v1", split=dict(suites={
        suite: dict(train=list(range(6)), validation=[6, 7], test=[8, 9]) for suite in suites}),
        auxiliary_train=dict(suite="libero_90", task_ids=aux, global_task_id_offset=40))
    rows = [dict(global_task_id=i, suite=suites[i // 10], task_id=i % 10,
                 split_role="train" if i % 10 < 6 else "validation" if i % 10 < 8 else "test") for i in range(40)]
    rows += [dict(global_task_id=40+i, suite="libero_90", task_id=i, split_role="train") for i in aux]
    for row in rows:
        row.update(language="move", hdf5=dict(relative_path="train" if row["split_role"] == "train" else "DO_NOT_READ",
                                               bytes=0, sha256="historical"))
    manifest = dict(tasks=rows, summary=dict(roles={role: [r["global_task_id"] for r in rows if r["split_role"] == role]
                                                        for role in ("train", "validation", "test")}))
    validate_coverage_manifest(manifest, protocol)
    import ember.source_sft.contract as module
    monkeypatch.setattr(module, "read_json", lambda _: manifest)
    config = dict(authorities=dict(target_data_manifest=dict(path="unused")),
                  stages=dict(development=dict(action_split_roles=["train"], task_count=36)))
    tasks = _target_tasks(config, tmp_path, "development")
    assert len(tasks) == 36 and all(t.authority.path.name == "train" for t in tasks)
    rows[-1]["task_id"] = 62
    with pytest.raises(WriterModelError, match="identities"):
        validate_coverage_manifest(manifest, protocol)


def test_test_requires_final_selection_not_segment_summary(tmp_path):
    from ember.source_sft.inference import _formal_summary_sha
    contract = dict(mode="formal", training_control=CONTROL, runtime=dict(world_size=4))
    config = dict(stages=dict(development=dict(formal_run=dict(status="sealed", expected_world_size=4))))
    summary = dict(schema_version="ember_pi05_source_sft_run_summary_v1", contract_sha256="fixed",
                   stage="development", completed_optimizer_steps=100, test_action_reads=0,
                   training_complete=False, selected_checkpoint_step=None)
    path = tmp_path / "run_summary.json"
    path.write_text(json.dumps(summary))
    kwargs = dict(require_formal=True, run_root=tmp_path, run_contract=contract,
                  run_contract_sha="fixed", config=config, stage="development", step=75)
    _formal_summary_sha(**kwargs, evaluation_role="validation")
    with pytest.raises(WriterModelError, match="frozen selected checkpoint"):
        _formal_summary_sha(**kwargs, evaluation_role="test")
    summary.update(training_complete=True, selected_checkpoint_step=75)
    path.write_text(json.dumps(summary))
    _formal_summary_sha(**kwargs, evaluation_role="test")


def test_dynamic_runtime_endpoint_passes_lr_and_historical_caps():
    from ember.source_sft.contract import _runtime_steps
    args = SimpleNamespace(mode="formal", total_steps=None, checkpoint_steps=None, stop_after_step=1500)
    total, steps = _runtime_steps(args, dict(total_steps=None), CONTROL)
    assert total == 1500 and steps == tuple(range(25, 1501, 25))
    args.stop_after_step = 1525
    with pytest.raises(WriterModelError, match="validation interval endpoint"):
        _runtime_steps(args, dict(total_steps=None), CONTROL)
