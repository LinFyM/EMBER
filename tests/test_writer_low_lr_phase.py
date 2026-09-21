import json

import pytest

from ember.pi05_eval_results import AGGREGATE_SCHEMA
from scripts.read_writer_low_lr_phase import read_phase


def test_read_phase_accepts_only_complete_registered_validation400(tmp_path):
    protocol = {"split": {"suites": {
        suite: {"validation": [2 * index, 2 * index + 1]}
        for index, suite in enumerate(("libero_spatial", "libero_object", "libero_goal", "libero_10"))
    }}}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    directory = tmp_path / "evaluation/writer_00001900"
    directory.mkdir(parents=True)
    rows = [
        {"suite": suite, "task_id": task, "init_state_id": state, "success": state == 0}
        for suite, roles in protocol["split"]["suites"].items()
        for task in roles["validation"]
        for state in range(50)
    ]
    reference = "sealed-panel"
    successes = sum(row["success"] for row in rows)
    (directory / "results.json").write_text(json.dumps({
        "schema_version": AGGREGATE_SCHEMA, "contract_reference": reference,
        "rows": rows, "overall": {"episodes": 400, "successes": successes},
    }))
    (directory / "run_contract.json").write_text(json.dumps({
        "contract_reference": reference, "role": "validation", "mode": "formal",
    }))
    (directory / "launcher_completion.json").write_text(json.dumps({
        "contract_reference": reference, "return_codes": {"worker": 0},
        "queue": {"completed_rows": 400, "successes": successes},
    }))

    result = read_phase(tmp_path, protocol_path, 1900)
    assert result["history"] == [{
        "step": 1900, "global_step": 1900, "phase_step": 100, "complete": True,
        "episodes": 400, "successes": 8,
        "results": str(directory / "results.json"),
    }]
    assert not result["decision"]["stop"]

    broken = json.loads((directory / "launcher_completion.json").read_text())
    broken["queue"]["completed_rows"] = 399
    (directory / "launcher_completion.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="incomplete"):
        read_phase(tmp_path, protocol_path, 1900)
