"""Contract checks for one bounded P/I correspondence engineering batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ember.demonstration_learning.data import SUPPORTED, TransferData, audit_full_cycle
from ember.demonstration_learning.model import concatenate_factors, split_identity_template


REPO = Path(__file__).resolve().parents[2]
ASSETS = Path("/data1/user/ymdai/projects/EMBER")


def test_complete_factor_concatenation_keeps_two_BA_terms_and_gradient() -> None:
    full = {}
    for index in range(38):
        stem = f"target_{index}"
        full[stem + ".lora_A.default.weight"] = torch.randn(144, 5)
        full[stem + ".lora_B.default.weight"] = torch.zeros(3, 144)
    common, video = split_identity_template(full)
    for name in common:
        common[name].requires_grad_()
        video[name].requires_grad_()
    compiled = concatenate_factors(common, video)
    for index in range(38):
        stem = f"target_{index}"
        a, b = stem + ".lora_A.default.weight", stem + ".lora_B.default.weight"
        assert compiled[b].shape == (3, 144)
        assert compiled[a].shape == (144, 5)
        assert torch.allclose(compiled[b] @ compiled[a], common[b] @ common[a] + video[b] @ video[a])
    a, b = "target_0.lora_A.default.weight", "target_0.lora_B.default.weight"
    with torch.no_grad():
        common[b].fill_(0.25)
        video[b].fill_(0.5)
    (concatenate_factors(common, video)[b] @ concatenate_factors(common, video)[a]).square().sum().backward()
    assert all(value.grad is not None and torch.count_nonzero(value.grad)
               for value in (common[a], common[b], video[a], video[b]))


def test_full_events_and_actual_new_query_offset() -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_engineering_spec.json").read_text())
    data = TransferData(ASSETS, spec)
    try:
        audit = audit_full_cycle(data.events)
        assert (audit["queries_total"], audit["new_queries"], audit["old_queries"]) == (32256, 8960, 23296)
        assert set(audit["supported"]) == set(SUPPORTED)
        new = next(row for step in range(9, 18) for row in data.events.event(step, "P")
                   if row["kind"] == "new")
        paired = next(row for step in range(9, 18) for row in data.events.event(step, "I")
                      if row["task"] == new["task"])
        assert new["queries"] == paired["queries"] and new["flow_seed"] == paired["flow_seed"]
        raw = data.batch(new)
        assert raw["action"].shape == (28, 50, 7)
        assert raw["observation.state"].shape == (28, 8)
        assert raw["observation.images.camera1"].shape == (28, 3, 256, 256)
        first = new["queries"][0]
        with np.load(first["path"], allow_pickle=False) as trace:
            assert torch.equal(raw["action"][0, 0], torch.from_numpy(trace["actions"][first["frame"]]))
            assert int(trace["rgb_steps"][first["rgb_ordinal"]]) == first["frame"]
    finally:
        data.close()


def test_crossing_resume_identity_rejects_changed_cursor() -> None:
    spec = json.loads((REPO / "configs/demonstration_transfer_v1/learning_engineering_spec.json").read_text())
    data = TransferData(ASSETS, spec)
    try:
        state = data.events.sampler_state()
        state["next_step"] = 2
        data.events.restore(state)
        assert data.events.next_step == 2
        with pytest.raises(ValueError):
            data.events.restore({**state, "event_seeds": {**state["event_seeds"], "flow": 0}})
        with pytest.raises(ValueError):
            data.events.restore({**state, "next_step": 3})
    finally:
        data.close()
