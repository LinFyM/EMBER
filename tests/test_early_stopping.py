import pytest

from ember.early_stopping import validation_decision


def decide(scores):
    return validation_decision([{"step": 200 * (i + 1), "episodes": 400,
        "complete": True, "successes": score} for i, score in enumerate(scores)], interval=200)


def test_decline_stops_and_recovers_historical_best():
    result = decide([152, 168, 180, 174, 168, 164])
    assert result["stop"] and result["reason"] == "sustained_decline"
    assert result["best_step"] == 600


def test_recovery_and_new_high_continue():
    assert not decide([152, 168, 180, 168, 176])["stop"]
    assert not decide([152, 168, 180, 168, 176, 184])["stop"]
    assert not decide([180, 164, 168, 172])["stop"]


def test_plateau_and_ties_preserve_earliest_best():
    result = decide([160] * 6)
    assert result["reason"] == "plateau" and result["best_step"] == 200
    assert len(result["tied_best_steps"]) == 6
    assert not decide([160, 155, 156, 157, 158, 159])["stop"]


def test_partial_or_skipped_panels_cannot_stop_training():
    for row in [{"step": 200, "episodes": 399, "complete": True, "successes": 80},
                {"step": 400, "episodes": 400, "complete": True, "successes": 80},
                {"step": 200, "episodes": 400, "complete": False, "successes": 80}]:
        with pytest.raises(ValueError):
            validation_decision([row], interval=200)
