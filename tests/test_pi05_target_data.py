from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ember.libero_data import AuditResult
from ember.pi05_assets import load_protocol
from ember.pi05_source_checkpoint import canonical_hash, read_json
import ember.pi05_target_data as target_data
from ember.pi05_target_data import (
    HubFileAuthority,
    Pi05TargetDataError,
    SUITE_ORDER,
    build_target_rows,
    seal_target_data,
    target_global_task_id,
    target_hdf5_relative_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _authorities() -> tuple[dict, dict, dict[str, HubFileAuthority]]:
    protocol = load_protocol(REPO_ROOT / "configs/libero_24_8_8_v1/protocol.json")
    overlap = read_json(REPO_ROOT / "configs/pi05_source_corpus_v1/overlap_audit.json")
    hub = {}
    for task in overlap["target_tasks"]:
        relative = target_hdf5_relative_path(task)
        global_id = target_global_task_id(str(task["suite"]), int(task["task_id"]))
        hub[relative] = HubFileAuthority(relative, global_id + 100, f"{global_id:064x}")
    return protocol, overlap, hub


def test_target_rows_map_four_local_namespaces_to_unique_global_ids() -> None:
    protocol, overlap, hub = _authorities()
    rows = build_target_rows(protocol=protocol, overlap_audit=overlap, hub_files=hub)
    assert len(rows) == 40
    assert [row["global_task_id"] for row in rows] == list(range(40))
    assert [row["suite"] for row in rows[::10]] == list(SUITE_ORDER)
    assert {role: sum(row["split_role"] == role for row in rows) for role in ("train", "validation", "test")} == {
        "train": 24,
        "validation": 8,
        "test": 8,
    }
    assert len({row["hdf5"]["relative_path"] for row in rows}) == 40


def test_target_rows_fail_closed_on_missing_or_extra_hdf5() -> None:
    protocol, overlap, hub = _authorities()
    hub.pop(next(iter(hub)))
    with pytest.raises(Pi05TargetDataError, match="missing target HDF5"):
        build_target_rows(protocol=protocol, overlap_audit=overlap, hub_files=hub)

    _, _, complete_hub = _authorities()
    complete_hub["libero_spatial/unexpected_demo.hdf5"] = HubFileAuthority(
        "libero_spatial/unexpected_demo.hdf5", 1, "f" * 64
    )
    with pytest.raises(Pi05TargetDataError, match="unexpected file"):
        build_target_rows(
            protocol=protocol, overlap_audit=overlap, hub_files=complete_hub
        )


def test_target_rows_reject_unsealed_overlap_schema() -> None:
    protocol, overlap, hub = _authorities()
    changed = copy.deepcopy(overlap)
    changed["schema_version"] = "untrusted"
    with pytest.raises(Pi05TargetDataError, match="sealed target-40"):
        build_target_rows(protocol=protocol, overlap_audit=changed, hub_files=hub)


def test_seal_target_data_keeps_trajectory_values_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, hub = _authorities()
    calls: list[dict] = []

    def fake_audit(_path: Path, **kwargs: object) -> AuditResult:
        calls.append(dict(kwargs))
        lengths = [2] * 50
        return AuditResult(
            record={
                "hdf5": {
                    "sha256": kwargs["expected_sha256"],
                },
                "demonstrations": {
                    "count": 50,
                    "steps": sum(lengths),
                    "min_steps": 2,
                    "max_steps": 2,
                    "episode_lengths": lengths,
                },
                "camera": {},
                "controller": {},
                "robot": "Panda",
                "quality": {"status": "pass", "warning_count": 0, "warnings": []},
            },
            state_samples=None,
            action_samples=None,
        )

    monkeypatch.setattr(target_data, "fetch_hub_authorities", lambda: hub)
    monkeypatch.setattr(target_data, "audit_demonstration_file", fake_audit)
    output = tmp_path / "manifest.json"
    manifest = seal_target_data(
        protocol_path=REPO_ROOT / "configs/libero_24_8_8_v1/protocol.json",
        overlap_audit_path=REPO_ROOT
        / "configs/pi05_source_corpus_v1/overlap_audit.json",
        data_root=tmp_path,
        output_path=output,
        sealed_utc="2026-07-21",
    )

    assert len(calls) == 40
    assert all(call["normalization_episodes"] == () for call in calls)
    assert manifest["summary"]["tasks"] == 40
    assert manifest["summary"]["episodes"] == 2000
    assert manifest["summary"]["frames"] == 4000
    assert manifest["dataset"]["revision"] == target_data.DATASET_REVISION
    assert manifest["information_wall"]["decoded_trajectory_or_video_values"] == 0
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256")
    assert canonical_hash(payload) == digest
    assert read_json(output) == manifest


def test_seen_panel_is_specification_only_and_recomputable() -> None:
    manifest = read_json(REPO_ROOT / "configs/pi05_target_data_v1/manifest.json")
    panel = read_json(REPO_ROOT / "configs/pi05_seen_panel_v1.json")
    selection = panel["selection"]
    assert selection["policy_outcome_reads"] == 0
    assert selection["trajectory_value_reads"] == 0
    assert selection["selection_changes_after_outcome"] == 0
    assert len(panel["tasks"]) == 8
    target_rows = {
        int(row["global_task_id"]): row for row in manifest["tasks"]
    }
    for suite in selection["suite_order"]:
        candidates = []
        for row in manifest["tasks"]:
            if row["suite"] != suite or row["split_role"] != "train":
                continue
            encoded = json.dumps(
                [
                    "ember_pi05_seen_panel_v1",
                    int(selection["seed"]),
                    suite,
                    int(row["task_id"]),
                    row["language"],
                    row["bddl"]["sha256"],
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            candidates.append((hashlib.sha256(encoded).hexdigest(), row))
        expected = sorted(candidates)[: int(selection["tasks_per_suite"])]
        observed = [row for row in panel["tasks"] if row["suite"] == suite]
        assert [row["selection_sha256"] for row in observed] == [
            digest for digest, _ in expected
        ]
        for rank, (row, (digest, source)) in enumerate(zip(observed, expected, strict=True)):
            assert row["selection_rank_within_suite"] == rank
            assert row["selection_sha256"] == digest
            assert row["global_task_id"] == source["global_task_id"]
            assert row["split_role"] == source["split_role"] == "train"
            assert row["language"] == source["language"]
            assert row["bddl_sha256"] == source["bddl"]["sha256"]
            assert target_rows[row["global_task_id"]] == source
