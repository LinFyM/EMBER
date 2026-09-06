from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.pi05_eval.launcher_evidence import launcher_attempt_summary
from ember.pi05_eval.recovery import worker_command_matches
from ember.pi05_eval_queue import publish_json_exclusive


def _invocation_events(
    *,
    contract_reference: str,
    worker_ids: list[str],
    failed_id: str,
    completed_id: str,
    failed_codes: dict[str, int],
) -> tuple[dict, ...]:
    return (
        {
            "event": "started",
            "unix": 1.0,
            "invocation_id": failed_id,
            "contract_reference": contract_reference,
            "worker_ids": worker_ids,
        },
        {
            "event": "failed",
            "unix": 5.0,
            "wall_seconds": 4.0,
            "invocation_id": failed_id,
            "return_codes": failed_codes,
        },
        {
            "event": "resume_started",
            "unix": 10.0,
            "invocation_id": completed_id,
            "contract_reference": contract_reference,
            "worker_ids": worker_ids,
        },
        {
            "event": "completed",
            "unix": 20.0,
            "wall_seconds": 10.0,
            "invocation_id": completed_id,
            "return_codes": {"4-r0": 0, "4-r1": 0},
        },
    )


def test_launcher_attempt_summary_counts_shards_across_resume(tmp_path: Path) -> None:
    contract = {"mode": "formal", "contract_reference": "contract-a"}
    worker_ids = ["4-r0", "4-r1"]
    failed_id = "b" * 32
    completed_id = "c" * 32
    failed_codes = {"4-r0": -15, "4-r1": 1}
    events = _invocation_events(
        contract_reference=contract["contract_reference"],
        worker_ids=worker_ids,
        failed_id=failed_id,
        completed_id=completed_id,
        failed_codes=failed_codes,
    )
    (tmp_path / "invocations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    publish_json_exclusive(
        tmp_path / "failures/launcher_1.json",
        {
            "schema_version": "ember_pi05_eval_launcher_failure_v1",
            "invocation_id": failed_id,
            "return_codes": failed_codes,
            "queue": {"status_counts": {"complete": 5}},
        },
    )
    launcher = {
        "invocation_id": completed_id,
        "started_unix": 10.0,
        "finished_unix": 20.0,
        "wall_seconds": 10.0,
        "worker_ids": worker_ids,
    }
    summary = launcher_attempt_summary(
        tmp_path,
        contract,
        launcher,
        [{"completed_shards": 19}, {"completed_shards": 12}],
        total_shards=36,
    )
    assert summary["active_wall_seconds"] == 14.0
    assert summary["completed_before_final_attempt"] == 5
    assert [row["completed_shards"] for row in summary["attempts"]] == [5, 31]


def test_worker_detection_uses_exact_argv_tokens(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    worker = b"\0".join(
        (
            b"python",
            b"/repo/scripts/evaluate_pi05.py",
            b"worker",
            b"--output-dir",
            str(output.resolve()).encode(),
            b"--worker-id",
            b"4-r0",
            b"",
        )
    )
    wrapper = b"\0".join(
        (
            b"bash",
            b"-c",
            (b"pgrep evaluate_pi05.py worker " + str(output.resolve()).encode()),
            b"",
        )
    )
    launcher = worker.replace(b"\0worker\0", b"\0start\0")

    assert worker_command_matches(worker, output)
    assert not worker_command_matches(wrapper, output)
    assert not worker_command_matches(launcher, output)
    assert not worker_command_matches(worker, tmp_path / "another")


@pytest.mark.parametrize("state_count", [5, 10])
@pytest.mark.parametrize("informed,scope,selection,role,accepted", [
    (False, None, False, "development_train", True),
    (True, "training_task_fitting_diagnostic", False, "development_train", True),
    (True, None, False, "development_train", False),
    (True, "training_task_fitting_diagnostic", True, "development_train", False),
    (True, "training_task_fitting_diagnostic", False, "validation", False),
    (True, "training_task_fitting_diagnostic", False, "test", False),
])
def test_outcome_informed_subset_is_explicit_training_diagnostic(
    tmp_path: Path, informed: bool, scope: str | None,
    selection: bool, role: str, accepted: bool, state_count: int,
) -> None:
    from ember.pi05_assets import Pi05EvaluationError
    from ember.pi05_eval.preparation import _task_subset_tasks

    path = tmp_path / "subset.json"
    path.write_text(json.dumps({
        "schema_version": "ember_pi05_task_subset_selection_v1",
        "role": role, "mode": "screen", "state_count": state_count,
        "task_ordinals": [0], "global_task_ids": [7],
        "tasks": [{"global_task_id": 7, "suite": "libero_spatial", "task_id": 7}],
        "outcome_dependence": informed, "selection_scope": scope,
        "checkpoint_selection_use": selection, "validation_use": False, "test_use": False,
    }))
    args = SimpleNamespace(
        task_subset_selection=path, role=role, mode="screen", state_count=state_count,
    )
    tasks = (SimpleNamespace(suite="libero_spatial", task_id=7),)
    if not accepted:
        with pytest.raises(Pi05EvaluationError):
            _task_subset_tasks(args, tasks, adapter_kind="static_task_lora")
        return
    selected, recorded = _task_subset_tasks(args, tasks, adapter_kind="static_task_lora")
    assert selected == tasks
    assert recorded["outcome_dependence"] is informed
    if informed:
        assert recorded["selection_scope"] == scope
        assert recorded["checkpoint_selection_use"] is False
    else:
        assert "selection_scope" not in recorded
