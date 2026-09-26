"""The fixed400 completion extends only the registered C0 evaluation scope."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember.pi05_eval.language_content_capture import _cases
from ember.pi05_eval.preparation import _explicit_diagnostic_states, _registered_subset_mode_allowed
from ember.pi05_source_checkpoint import read_json
from ember.writer.language_content_contract import (
    bank_panel, evaluation_panel, fixed400_spec,
    validate_evaluation_bank,
)
from ember.writer.materialization import request_init_state_ids, selection_contract


ROOT = Path(__file__).resolve().parents[1]
STUDY = fixed400_spec()
RUN = Path(STUDY["outputs"]["planned_run_root"])
PARENT = Path(STUDY["parent_study"])
STATES = list(range(10, 50))


def _panel(model: str, condition: str = "correct") -> dict:
    return next(row for row in STUDY["evaluation"]["panels"]
                if row["model"] == model and row["condition"] == condition)


def _selection(condition: str) -> dict:
    return selection_contract(
        role="development_train", task_ids=STUDY["evaluation"]["task_ids"],
        cardinality=1, arm=condition, mode="per_init_ordinal",
        seed=STUDY["evaluation"]["video_schedule_seed"],
        init_state_ids=STATES, video_pool=range(50))


def test_exact_new_scope_keeps_prior_panel_and_capture_separate():
    assert len(STUDY["evaluation"]["panels"]) == 4
    assert sum(row["rows"] for row in STUDY["evaluation"]["panels"]) == 1280
    for row in STUDY["evaluation"]["panels"]:
        panel = evaluation_panel(RUN / "evaluation" / row["id"])
        cases, full = _cases(panel)
        assert panel["state_ids"] == STATES and len(cases) == 320 and len(full) == 4
        assert all(state in STATES for _, _, state in cases)
        assert {state for _, _, state in full} == {10}
    prior = evaluation_panel(PARENT / "evaluation" / "C0_630_held_correct")
    assert prior["state_ids"] == list(range(10))
    assert len(_cases(prior)[0]) == 80
    with pytest.raises(ValueError, match="outside four registered"):
        evaluation_panel(RUN / "evaluation" / "Cplus_630_held_correct")


def test_only_registered_40_state_selectors_are_admitted():
    panel = _panel("Source")
    selectors = RUN / "launch" / "selectors"
    args = SimpleNamespace(
        role="development_train", mode="screen", state_count=40,
        init_state_ids=STATES,
        task_subset_selection=selectors / f"{panel['id']}_subset.json",
        trajectory_capture_selection=selectors / f"{panel['id']}_capture.json",
        occupancy_capture_selection=None, exploration_sigma=False,
        frozen_replay_registration=None)
    assert _explicit_diagnostic_states(args) == tuple(STATES)
    assert _registered_subset_mode_allowed(args)
    assert request_init_state_ids(role="development_train", init_state_ids=STATES,
                                  state_count=40, registered_fixed400=True) == tuple(STATES)
    with pytest.raises(ValueError, match="explicit Writer"):
        request_init_state_ids(role="development_train", init_state_ids=STATES,
                               state_count=40)
    for change in (dict(init_state_ids=list(range(40))),
                   dict(task_subset_selection=selectors / "Cplus_630_held_correct_subset.json"),
                   dict(trajectory_capture_selection=selectors / "B630_held_correct_capture.json"),
                   dict(exploration_sigma=True)):
        bad = SimpleNamespace(**(vars(args) | change))
        assert not _registered_subset_mode_allowed(bad)


@pytest.mark.parametrize("condition", ("correct", "same_task_other"))
def test_c0_bank_requires_frozen_parent_and_complete_mapping(condition):
    config = read_json(ROOT / "configs/language_content_path_causality_v1/train_C0.json")
    checkpoint = Path(STUDY["frozen_inputs"]["C0_checkpoint"])
    panel = _panel("C0", condition)
    output = RUN / "materialization" / panel["id"]
    selected = _selection(condition)
    assert bank_panel(config, selected, checkpoint=checkpoint, output=output) == evaluation_panel(
        RUN / "evaluation" / panel["id"])
    bad = deepcopy(selected)
    bad["init_state_ids"] = list(range(40))
    with pytest.raises(ValueError, match="fixed400 bank task"):
        bank_panel(config, bad, checkpoint=checkpoint, output=output)
    with pytest.raises(ValueError, match="complete 630 checkpoint"):
        bank_panel(config, selected, checkpoint=checkpoint.with_name("macro_00000525"), output=output)
    assert selected["schedule_state_origin"] == 0
    assert selected["video_ordinal_rule"] == "init_state_id"


def test_reference_and_generated_bank_commit_boundaries():
    reference = _panel("B630")
    bpanel = evaluation_panel(RUN / "evaluation" / reference["id"])
    path = Path(STUDY["frozen_inputs"]["B630_bank_root"]) / "manifest.json"
    manifest = read_json(path)
    run = read_json(Path(manifest["writer_checkpoint"]["path"]).parent.parent / "run_contract.json")
    assert validate_evaluation_bank(bpanel, path, manifest, run, "future_E") is True
    with pytest.raises(ValueError, match="B630 reference"):
        validate_evaluation_bank(bpanel, path.with_name("other.json"), manifest, run, "future_E")
    cpanel = evaluation_panel(RUN / "evaluation" / _panel("C0")["id"])
    cpath = RUN / "materialization" / cpanel["id"] / "manifest.json"
    crun = read_json(Path(STUDY["frozen_inputs"]["C0_checkpoint"]).parent.parent / "run_contract.json")
    cm = {"selection": _selection("correct"), "materialization_git": {"commit": "future_E"},
          "writer_checkpoint": {"macro": 630, "path": STUDY["frozen_inputs"]["C0_checkpoint"]}}
    assert validate_evaluation_bank(cpanel, cpath, cm, crun, "future_E") is False
    cm["materialization_git"]["commit"] = "wrong"
    with pytest.raises(ValueError, match="C0 training"):
        validate_evaluation_bank(cpanel, cpath, cm, crun, "future_E")
