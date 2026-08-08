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


def test_meta_writer_formal_is_sealed_by_address_binding_execution_evidence() -> None:
    formal = load_expert_manifold_config(CONFIG)["meta_training"]["formal_run"]

    assert formal["status"] == "sealed"
    assert formal["selected_expert_step"] == 2000
    assert formal["profile_evidence"]["topology_address_binding"] == (
        "normalized_dynamic_times_normalized_chunk_plus_rank_address"
    )
    assert formal["online_smoke_evidence"]["topology_address_binding"] == (
        "normalized_dynamic_times_normalized_chunk_plus_rank_address"
    )


def test_meta_writer_formal_runtime_accepts_profiled_address_binding(
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

    assert _runtime(
        Namespace(
            mode="formal",
            microbatch=None,
            stop_after_macro=50,
            expert_step=2000,
        ),
        config,
        Namespace(world_size=6),
    ) == (800, 1, (50, 100, 200, 400, 600, 800), 50)


def test_meta_writer_formal_seal_rejects_missing_smoke_evidence(
    tmp_path: Path,
) -> None:
    changed = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    del changed["meta_training"]["formal_run"]["online_smoke_evidence"]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_expert_manifold_config(path)


def test_meta_writer_formal_seal_rejects_old_decoder_evidence(
    tmp_path: Path,
) -> None:
    changed = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    changed["meta_training"]["formal_run"]["profile_evidence"][
        "topology_address_binding"
    ] = "phase_centered_dynamic_without_topology_address"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_expert_manifold_config(path)
