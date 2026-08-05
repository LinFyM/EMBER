from __future__ import annotations

import json
from pathlib import Path

import pytest

from ember.reward.protocol import RewardProtocolError
from ember.rl_writer.contract import load_rl_writer_config
from ember.writer.model import WriterModelError
from ember.writer.topology import visible_physical_cuda_index


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/pi05_rl_writer_development_v1.json"
)


def test_retired_program_credit_config_fails_closed() -> None:
    with pytest.raises(RewardProtocolError, match="retired"):
        load_rl_writer_config(CONFIG)


def test_retirement_gate_precedes_stale_coldstart_resolution(tmp_path: Path) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["authorities"]["as_writer_config"]["path"] = "missing.json"
    path = tmp_path / "retired.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RewardProtocolError, match="retired"):
        load_rl_writer_config(path)


def test_torchrun_local_rank_maps_to_the_physical_egl_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3,4,5,7")
    assert [visible_physical_cuda_index(rank) for rank in range(6)] == [
        1,
        2,
        3,
        4,
        5,
        7,
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    with pytest.raises(WriterModelError, match="numeric physical GPU"):
        visible_physical_cuda_index(0)
