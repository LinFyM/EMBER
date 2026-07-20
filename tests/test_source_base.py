from __future__ import annotations

import json
from pathlib import Path

import torch

from ember.source_base import _build_policy_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_policy_config_uses_explicit_rank_local_cuda_device() -> None:
    config = json.loads((REPO_ROOT / "configs/source_base_v1.json").read_text())

    policy_config = _build_policy_config(
        config, Path("/tmp/pinned-vlm"), torch.device("cuda", 7)
    )

    assert policy_config.device == "cuda:7"
