import json

import pytest

from scripts.export_writer_low_lr_report import _training_cost, paired_counts


def panel(values):
    return {"rows": [
        {"suite": "suite", "task_id": index // 2, "init_state_id": index % 2, "success": value}
        for index, value in enumerate(values)
    ]}


def test_paired_counts_reports_retained_gained_lost_and_churn():
    result = paired_counts(panel([True, True, False, False]), panel([True, False, True, False]))
    assert result == {"retained": 1, "gained": 1, "lost": 1, "churn": 2}


def test_training_cost_requires_contiguous_metrics_through_stop(tmp_path):
    root = tmp_path / "training/writer"
    root.mkdir(parents=True)
    rows = [{"step": step, "optimizer_updates": step, "seconds": 2.0, "lr_applied": 2.959936e-5,
             "peak_reserved_gib": 21.0} for step in range(1801, 1901)]
    (root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = _training_cost(tmp_path, expected_end_step=1900)
    assert result["phase_updates"] == 100 and result["total_queries"] == 11200
    with pytest.raises(ValueError, match="stopping node"):
        _training_cost(tmp_path, expected_end_step=2000)
