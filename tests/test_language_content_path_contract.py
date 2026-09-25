"""The 630-prefix study admits only its registered learning and evaluation scope."""

from copy import deepcopy
from pathlib import Path

import pytest

from ember.pi05_source_checkpoint import read_json
from ember.writer.language_content_contract import (
    bank_panel, evaluation_panel, spec, validate_config, validate_evaluation_bank,
)
from ember.writer.materialization import selection_contract
from ember.writer.training import _checkpoint_nodes, _segment_limit


ROOT = Path(__file__).resolve().parents[1]
STUDY = spec()
RUN = Path(STUDY["outputs"]["planned_run_root"])


def _args(**changes):
    from types import SimpleNamespace

    return SimpleNamespace(**({"checkpoint_updates": None, "mode": "formal",
                               "stop_after_step": 630, "support_slot_arm": None} | changes))


@pytest.mark.parametrize("arm", ("C0", "Cplus"))
def test_training_is_exact_parent_recipe_with_six_prefix_checkpoints(arm):
    config = read_json(ROOT / f"configs/language_content_path_causality_v1/train_{arm}.json")
    assert validate_config(config) == config
    assert _checkpoint_nodes(_args(), config) == (105, 210, 315, 420, 525, 630)
    assert _segment_limit(_args(), config) == 630
    with pytest.raises(ValueError, match="630"):
        _segment_limit(_args(stop_after_step=735), config)
    with pytest.raises(ValueError, match="six registered"):
        _checkpoint_nodes(_args(checkpoint_updates="105,210,315,420,525,630,735"), config)
    changed = deepcopy(config)
    changed["experiment"]["language_content_path"] = not changed["experiment"]["language_content_path"]
    with pytest.raises(ValueError, match="scalar switch"):
        validate_config(changed)
    changed = deepcopy(config)
    changed["optimization"]["lr"] *= 2
    with pytest.raises(ValueError, match="optimizer"):
        validate_config(changed)


@pytest.mark.parametrize("arm", ("C0", "Cplus"))
@pytest.mark.parametrize("kind", ("held_correct", "seen_correct", "held_other"))
def test_bank_request_rejects_deferred_states_and_unregistered_controls(arm, kind):
    config = read_json(ROOT / f"configs/language_content_path_causality_v1/train_{arm}.json")
    panel = next(row for row in STUDY["evaluation"]["panels"]
                 if row["model"] == arm and row["kind"] == kind)
    held = kind != "seen_correct"
    selection = selection_contract(
        role="development_train", task_ids=panel["task_ids"], cardinality=1,
        arm="same_task_other" if kind == "held_other" else "correct",
        mode="per_init_ordinal", seed=STUDY["evaluation"]["video_schedule_seed"],
        init_state_ids=panel["state_ids"],
        video_pool=range(50) if held else range(46, 50),
    )
    checkpoint = RUN / "training" / arm / "checkpoints" / "macro_00000630"
    output = RUN / "materialization" / panel["id"]
    assert bank_panel(config, selection, checkpoint=checkpoint, output=output) == panel
    mutated = deepcopy(selection)
    mutated["init_state_ids"] = list(range(50)) if held else list(range(5))
    with pytest.raises(ValueError, match="bank task"):
        bank_panel(config, mutated, checkpoint=checkpoint, output=output)
    with pytest.raises(ValueError, match="exact registered panel"):
        bank_panel(config, selection, checkpoint=checkpoint, output=output.with_name("unregistered"))
    assert evaluation_panel(RUN / "evaluation" / panel["id"]) == panel


def test_old_language_reference_is_only_admitted_for_registered_subset():
    panel = next(row for row in STUDY["evaluation"]["panels"]
                 if row["id"] == "B630_held_correct")
    path = Path(STUDY["frozen_references"]["B630"]["held_bank_root"]) / "manifest.json"
    manifest = read_json(path)
    run = read_json(Path(manifest["writer_checkpoint"]["path"]).parent.parent / "run_contract.json")
    assert validate_evaluation_bank(panel, path, manifest, run, "future_frozen_commit") is True
    with pytest.raises(ValueError, match="reference bank"):
        validate_evaluation_bank(panel, path.with_name("different.json"), manifest, run, "future_frozen_commit")
