from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "libero90_70_10_10"


def _load(name: str) -> dict:
    return json.loads((PROTOCOL / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sealed_protocol_hashes_match_active_implementations() -> None:
    expected = {}
    for line in (PROTOCOL / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    assert expected == {
        name: _sha256(PROTOCOL / name)
        for name in (
            "data_manifest.json",
            "factor_table.json",
            "normalization_train_only.json",
            "split.json",
        )
    }
    split = _load("split.json")
    factors = _load("factor_table.json")
    assert split["generator_sha256"] == _sha256(ROOT / "scripts/seal_libero90_protocol.py")
    assert split["split_implementation_sha256"] == _sha256(
        ROOT / "src/ember/libero_split.py"
    )
    assert factors["specification_authority"]["factor_parser_sha256"] == _sha256(
        ROOT / "src/ember/libero_task_factors.py"
    )


def test_split_and_manifest_enforce_the_information_wall() -> None:
    factors = _load("factor_table.json")
    split = _load("split.json")
    manifest = _load("data_manifest.json")
    normalization = _load("normalization_train_only.json")
    ids = split["task_ids"]
    assert [len(ids[name]) for name in ("train", "validation", "test")] == [70, 10, 10]
    assert sorted(ids["train"] + ids["validation"] + ids["test"]) == list(range(90))
    assert split["audit"]["minimum_train_support_per_held_exact_role_atom"] >= 2
    for held in ("validation", "test"):
        summary = split["audit"]["summaries"][held]
        assert summary["scene_family"] == {"KITCHEN": 5, "LIVING_ROOM": 3, "STUDY": 2}
        assert summary["operation_count"] == {"1": 5, "2": 5}
        assert summary["task_family"] == {
            "actuation": 2,
            "compound": 1,
            "pick_place": 4,
            "single_place": 3,
        }
        assert len(summary["scene"]) == 10
    assert len(factors["rows"]) == 90
    forbidden = set(factors["forbidden_input_fields"])
    assert {"action", "proprio", "reward", "terminal", "normalization", "policy_outcome"} <= forbidden
    assert manifest["summary"]["split_counts"] == {
        "test": 10,
        "train": 70,
        "validation": 10,
    }
    assert manifest["summary"]["demonstrations"] == 4500
    for record in manifest["tasks"]:
        assert record["demonstrations"]["count"] == 50
        assert record["init_states"]["count"] == 50
        expected_access = (
            "source_normalization_values" if record["split"] == "train" else "metadata_only"
        )
        assert record["access_policy"] == expected_access
    assert normalization["authority"]["split"] == "train"
    assert normalization["authority"]["task_indices"] == ids["train"]
    assert normalization["authority"]["forbidden_surfaces"] == ["validation", "test"]
