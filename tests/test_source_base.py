from __future__ import annotations

import json
from pathlib import Path

import torch

from ember.source_base import _build_policy_config
from ember.source_base_checkpoint import resolve_formal_segment


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_policy_config_uses_explicit_rank_local_cuda_device() -> None:
    config = json.loads((REPO_ROOT / "configs/source_base_v1.json").read_text())

    policy_config = _build_policy_config(
        config, Path("/tmp/pinned-vlm"), torch.device("cuda", 7)
    )

    assert policy_config.device == "cuda:7"


def test_formal_continuation_is_relative_thirds_with_exact_parent_scheduler() -> None:
    config = json.loads((REPO_ROOT / "configs/source_base_v1.json").read_text())
    initial = resolve_formal_segment(config, resuming=False)
    continuation = resolve_formal_segment(config, resuming=True)

    assert initial["start_step"] == 0
    assert initial["checkpoint_steps"] == (210, 420, 630)
    assert continuation["start_step"] == 630
    assert continuation["total_steps"] == 945
    assert continuation["checkpoint_steps"] == (735, 840, 945)
    assert continuation["scheduler_horizon_steps"] == 630
