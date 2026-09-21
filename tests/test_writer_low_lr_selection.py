import pytest

from scripts.finalize_writer_low_lr_phase import choose_phase_checkpoint


def phase(scores, *, stop=True):
    history = [
        {"step": 1900 + 100 * index, "episodes": 400, "complete": True, "successes": score}
        for index, score in enumerate(scores)
    ]
    best = max(scores)
    tied = [row["step"] for row in history if row["successes"] == best]
    return {
        "schema_version": "ember_writer_low_lr_phase_history_v1",
        "selection_reference": {"step": 1000, "successes": 117},
        "history": history,
        "decision": {"stop": stop, "reason": "plateau", "best_successes": best,
                     "tied_best_steps": tied},
    }


def test_phase_must_strictly_beat_original_n1000():
    for scores in ([110, 117], [116, 115]):
        result = choose_phase_checkpoint(phase(scores))
        assert result["outcome"] == "retain_original_n1000"
        assert result["selected_macro"] == 1000
        assert result["controls_required"] == []


def test_unique_strict_phase_best_is_selected_for_controls():
    result = choose_phase_checkpoint(phase([118, 121, 119]))
    assert result["outcome"] == "select_low_lr_phase"
    assert result["selected_macro"] == 2000
    assert result["controls_required"] == ["same_task_other", "cross_suite_wrong"]


def test_tied_correct_waits_for_every_other400_then_uses_other_and_earliest():
    value = phase([121, 119, 121])
    waiting = choose_phase_checkpoint(value, {1900: 100})
    assert waiting["status"] == "awaiting_tied_other400"
    assert waiting["required_other_steps"] == [2100]
    assert choose_phase_checkpoint(value, {1900: 100, 2100: 101})["selected_macro"] == 2100
    tied_other = choose_phase_checkpoint(value, {1900: 101, 2100: 101})
    assert tied_other["selected_macro"] == 1900
    assert tied_other["other_tiebreak_required"]


def test_selection_rejects_running_or_incomplete_phase():
    with pytest.raises(ValueError, match="stopped"):
        choose_phase_checkpoint(phase([120], stop=False))
    invalid = phase([120])
    invalid["history"][0]["episodes"] = 399
    with pytest.raises(ValueError, match="Validation400"):
        choose_phase_checkpoint(invalid)


def test_owner_stop_can_finalize_completed_nodes_without_claiming_registered_early_stop():
    value = phase([101, 92, 95, 98, 101], stop=False)
    result = choose_phase_checkpoint(value, owner_stop=True)
    assert result["outcome"] == "retain_original_n1000"
    assert result["phase_stop_reason"] == "owner_requested_stop"
