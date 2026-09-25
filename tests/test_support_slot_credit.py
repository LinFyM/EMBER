"""Frozen 1155 fork identity, task-slot credit, and bounded bank/panel authority."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.support_slot_credit import attach, select_tasks, validate_bank, validate_contract
from ember.writer.materialization import inspect_writer_checkpoint, selection_contract
from ember.writer.support_slot_credit import (
    ARMS, ForkTrainingData, authority, execute_registered_event, inspect_fork,
    registered_bank_request, registration_for, root,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT.parent / "EMBER"
CONFIG = ROOT / "configs/relational_support_causality_v1/train_C_S00.json"


@pytest.mark.parametrize("arm,slot_task,gate", [
    ("KEEP77", 77, 1), ("SWAP76", 76, 1), ("DROP77", 77, 0),
])
def test_fork_preserves_108_common_events_and_registered_first_slot(arm, slot_task, gate):
    import json

    config = json.loads(CONFIG.read_text())
    scope = inspect_fork(arm, config, root() / "smoke" / "cpu_contract", mode="smoke")
    data = ForkTrainingData(ASSET_ROOT, config, scope)
    try:
        parent = torch.load(scope.parent_checkpoint / "trainer_state.pt", map_location="meta",
                            mmap=True, weights_only=True)["sampler_state"]
        data.restore_parent(parent)
        rows = [data.next_iteration() for _ in range(5)]
        slot = scope.mapping[4]["slot_position"]
        assert slot == 1 and rows[4][slot]["task"] == slot_task
        assert rows[4][slot]["credit_gate"] == gate
        assert rows[4][slot]["occurrence"] == 165
        donor_plan = scope.s10_plan if arm == "SWAP76" else scope.s00_plan
        donor_index = scope.mapping[4]["S10_event_indices" if arm == "SWAP76" else "S00_event_indices"][slot]
        donor_event = donor_plan["events"][donor_index]
        assert rows[4][slot]["video_demos"] == (donor_event["teacher_demo"],)
        assert rows[4][slot]["query_seed"] == donor_event["query_seed"]
        assert rows[4][slot]["source_event_index"] == donor_index
        assert [row["task"] for index, row in enumerate(rows[4]) if index != slot] == [
            scope.s00_plan["events"][event]["task"]
            for index, event in enumerate(scope.mapping[4]["S00_event_indices"]) if index != slot]
        saved = data.sampler_state()
        assert saved["branch_cursor"] == 5 and saved["next_global_macro"] == 1160
        resumed = ForkTrainingData(ASSET_ROOT, config, scope)
        try:
            resumed.restore_branch(saved)
            assert resumed.sampler_state() == saved
        finally:
            resumed.close()
    finally:
        data.close()


def test_drop_removes_only_this_events_gradient_and_keeps_optimizer_history():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    (parameter.square()).backward()
    optimizer.step()
    before_m = optimizer.state[parameter]["exp_avg"].clone()
    optimizer.zero_grad(set_to_none=True)
    (3 * parameter).backward()
    common = parameter.grad.clone()

    def backward(draw):
        assert draw["task"] == 77
        (7 * parameter).backward()
        return {"flow_loss": 7.0}

    engine = SimpleNamespace(runtime=SimpleNamespace(state=SimpleNamespace(parameters=lambda: (parameter,))),
                             backward=backward)
    data = SimpleNamespace(scope=SimpleNamespace(arm="DROP77", spec=authority()))
    execute_registered_event(engine, data, {"task": 77, "credit_gate": 0}, 1160)
    assert torch.equal(parameter.grad, common)
    assert torch.equal(optimizer.state[parameter]["exp_avg"], before_m)
    with pytest.raises(ValueError, match="outside registered"):
        execute_registered_event(engine, data, {"task": 77, "credit_gate": 0}, 1161)


def test_parent_bank_scope_rejects_unregistered_task_video_and_phase():
    spec = authority()
    run, checkpoint = inspect_writer_checkpoint(Path(spec["parent"]["checkpoint"]))
    selection = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=20260911,
        init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    request = {"support_slot_model": "P1155", "support_slot_phase": "final",
               "selection": selection, "checkpoint": Path(checkpoint["path"]),
               "output": root() / "materialization/final/P1155"}
    token = registered_bank_request(request, run, checkpoint, {"commit": "new_frozen_commit"})
    assert token["model"] == "P1155" and token["checkpoint"]["macro"] == 1155
    changed = {**request, "selection": {**selection, "task_ids": [14, 20]}}
    with pytest.raises(ValueError, match="selection changed|selection"):
        registered_bank_request(changed, run, checkpoint, {"commit": "new_frozen_commit"})
    for change in ({"support_slot_phase": "first_slot"},
                   {"output": root() / "materialization/final/unregistered"}):
        with pytest.raises(ValueError):
            registered_bank_request({**request, **change}, run, checkpoint, {"commit": "new_frozen_commit"})


def test_branch_first_slot_bank_requires_1160_and_its_own_generation_commit():
    import copy

    parent_run, _ = inspect_writer_checkpoint(Path(authority()["parent"]["checkpoint"]))
    run = copy.deepcopy(parent_run)
    run["support_slot_credit"] = registration_for("SWAP76")
    run["git"]["commit"] = "frozen_implementation"
    checkpoint_path = root() / "training/SWAP76/checkpoints/macro_00001160"
    checkpoint = {"path": str(checkpoint_path), "macro": 1160}
    selection = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=20260911,
        init_state_ids=(0, 10, 20, 30, 40), video_pool=tuple(range(50)))
    path = root() / "materialization/first_slot/SWAP76/manifest.json"
    request = {"support_slot_model": "SWAP76", "support_slot_phase": "first_slot",
               "selection": selection, "checkpoint": checkpoint_path, "output": path.parent}
    token = registered_bank_request(request, run, checkpoint, {"commit": "frozen_implementation"})
    manifest = {"selection": selection, "support_slot_credit": token,
                "materialization_git": {"commit": "frozen_implementation"},
                "writer_checkpoint": checkpoint, "compilation": {"new_conditions": 10,
                "reused_conditions": 0}, "conditions": [{} for _ in range(10)]}
    validate_bank(manifest, path, "SWAP76", "frozen_implementation", run, checkpoint,
                  phase="first_slot")
    with pytest.raises((ValueError, Pi05EvaluationError)):
        registered_bank_request(request, run, checkpoint, {"commit": "different_commit"})
    with pytest.raises(Pi05EvaluationError, match="generation"):
        validate_bank({**manifest, "compilation": {"new_conditions": 9,
                       "reused_conditions": 0}}, path, "SWAP76", "frozen_implementation",
                      run, checkpoint, phase="first_slot")


@dataclass(frozen=True)
class Task:
    suite: str
    task_id: int
    init_state_ids: tuple[int, ...]


def test_final_panels_are_exact_pilot_and_remaining_with_all_row_trace():
    installed = (Task("libero_object", 4, tuple(range(50))),
                 Task("libero_goal", 1, tuple(range(50))))
    for stage, states, full in (("pilot", (0,), 2), ("remaining", tuple(range(1, 50)), 2)):
        args = SimpleNamespace(support_slot_model="KEEP77", output_dir=root() / "evaluation" / stage / "KEEP77",
            role="development_train", mode="formal", state_count=50,
            config=ROOT / "configs/relational_support_causality_v1/evaluation.json",
            static_task_lora_manifest=root() / "materialization/final/KEEP77/manifest.json")
        tasks, capture, predicates = select_tasks(args, installed, ROOT)
        assert len(tasks) == 2 and all(task.init_state_ids == states for task in tasks)
        assert len(capture["full_conditions"]) == full
        assert predicates["full_conditions_only"] is False
        args.output_dir = root() / "evaluation" / stage / "DROP77"
        with pytest.raises(Pi05EvaluationError, match="outside"):
            select_tasks(args, installed, ROOT)


def test_registered_panel_resume_rejects_missing_all_row_predicates(tmp_path, monkeypatch):
    import copy
    import ember.pi05_eval.support_slot_credit as panel

    bank = tmp_path / "manifest.json"
    bank.write_text("{}")
    monkeypatch.setattr(panel, "bank_path", lambda model: bank)
    installed = (Task("libero_object", 4, tuple(range(50))),
                 Task("libero_goal", 1, tuple(range(50))))
    output = root() / "evaluation/pilot/KEEP77"
    args = SimpleNamespace(support_slot_model="KEEP77", output_dir=output,
        role="development_train", mode="formal", state_count=50,
        config=ROOT / "configs/relational_support_causality_v1/evaluation.json",
        static_task_lora_manifest=bank)
    tasks, capture, predicates = select_tasks(args, installed, ROOT)
    contract = {"output_dir": str(output), "role": "development_train", "mode": "formal",
                "arm": "correct", "git": {"commit": "frozen"},
                "tasks": [{"suite": task.suite, "task_id": task.task_id,
                           "init_state_ids": list(task.init_state_ids)} for task in tasks],
                "adapter": {"manifest": {"path": str(bank), "bytes": bank.stat().st_size}},
                "diagnostic_occupancy_capture": capture,
                "diagnostic_stage_predicates": predicates}
    attach(contract, model="KEEP77", repo_root=ROOT)
    validate_contract(contract, ROOT)
    changed = copy.deepcopy(contract)
    changed["diagnostic_stage_predicates"]["capture"] = "full_conditions_only"
    with pytest.raises(Pi05EvaluationError, match="all-row"):
        validate_contract(changed, ROOT)
