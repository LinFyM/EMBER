from pathlib import Path

from ember.expert_manifold.inference import _training_source_matches_evaluation


def test_smoke_source_may_omit_formal_summary_descriptor(tmp_path: Path) -> None:
    summary = tmp_path / "run_summary.json"
    summary.write_text('{"schema_version":"ember_pi05_source_run_summary_v1"}\n')
    common = {
        "source_run": "/source",
        "checkpoint": "/source/checkpoints/step_00001000",
        "model_path": "/source/checkpoints/step_00001000/policy",
    }
    training = {
        **common,
        "source_run_summary": {
            "path": str(summary),
            "bytes": summary.stat().st_size,
            "schema_version": "ember_pi05_source_run_summary_v1",
        },
    }
    smoke = {**common, "source_run_summary": None}

    assert _training_source_matches_evaluation(training, smoke)
    assert not _training_source_matches_evaluation(
        training, {**smoke, "checkpoint": "/different/checkpoint"}
    )
    summary.write_text("changed\n")
    assert not _training_source_matches_evaluation(training, smoke)
