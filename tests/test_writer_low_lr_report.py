from scripts.export_writer_low_lr_report import paired_counts


def panel(values):
    return {"rows": [
        {"suite": "suite", "task_id": index // 2, "init_state_id": index % 2, "success": value}
        for index, value in enumerate(values)
    ]}


def test_paired_counts_reports_retained_gained_lost_and_churn():
    result = paired_counts(panel([True, True, False, False]), panel([True, False, True, False]))
    assert result == {"retained": 1, "gained": 1, "lost": 1, "churn": 2}
