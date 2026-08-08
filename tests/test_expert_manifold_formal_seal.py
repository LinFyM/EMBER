from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_expert_manifold_config,
)
from ember.expert_manifold.writer_training import _runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"


def test_meta_writer_formal_is_blocked_until_new_profile_and_online_smoke() -> None:
    formal = load_expert_manifold_config(CONFIG)["meta_training"]["formal_run"]

    assert formal["status"] == (
        "blocked_until_live_a40_profile_and_online_generation_smoke"
    )
    assert formal["selected_expert_step"] == 2000
    assert "profile_evidence" not in formal
    assert "online_smoke_evidence" not in formal


def test_meta_writer_formal_runtime_rejects_unprofiled_address_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_expert_manifold_config(CONFIG)
    monkeypatch.setattr(
        "ember.expert_manifold.writer_training.git_state",
        lambda _root: {
            "dirty_paths": [],
            "commit": "sealed",
            "upstream_commit": "sealed",
        },
    )

    with pytest.raises(ExpertManifoldError, match="sealed contract"):
        _runtime(
            Namespace(
                mode="formal",
                microbatch=None,
                stop_after_macro=50,
                expert_step=2000,
            ),
            config,
            Namespace(world_size=6),
        )


def test_meta_writer_formal_seal_rejects_missing_smoke_evidence(
    tmp_path: Path,
) -> None:
    changed = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    changed["meta_training"]["formal_run"]["status"] = "sealed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_expert_manifold_config(path)
