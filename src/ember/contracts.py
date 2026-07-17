from __future__ import annotations

import argparse
import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SPLIT_SIZES = {"source": 60, "validation": 15, "held_out": 15}
EXPECTED_EPISODE_POOLS = {
    "writer_spec",
    "source_base_fit",
    "oracle_support",
    "functional_query",
    "locked_source_report",
}


class ContractError(ValueError):
    """Raised when the checked-in experiment contract violates an invariant."""


def load_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _require_sha(value: object, label: str, *, sha256: bool = False) -> None:
    pattern = SHA256_RE if sha256 else SHA_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ContractError(f"{label} must be an immutable hexadecimal hash")


def _validate_revisions(contract: Mapping[str, Any]) -> None:
    upstreams = contract["upstreams"]
    for name in ("lerobot", "libero_official", "libero_runtime"):
        _require_sha(upstreams[name]["commit"], f"upstreams.{name}.commit")
    for field in ("bddl_tree_sha", "init_states_tree_sha", "scenes_assets_tree_sha"):
        _require_sha(
            upstreams["libero_official"][field],
            f"upstreams.libero_official.{field}",
        )
    _require_sha(upstreams["libero_task_map"]["git_blob_sha"], "task-map git blob")
    _require_sha(upstreams["libero_runtime"]["wheel_sha256"], "hf-libero wheel", sha256=True)

    for name, model in contract["models"].items():
        _require_sha(model["revision"], f"models.{name}.revision")
        _require_sha(model["weight_sha256"], f"models.{name}.weight_sha256", sha256=True)
    for name, dataset in contract["datasets"].items():
        _require_sha(dataset["revision"], f"datasets.{name}.revision")


def _validate_resources(resources: Mapping[str, Any]) -> None:
    maximum = resources["max_concurrent_gpus"]
    if not isinstance(maximum, int) or maximum < 1 or maximum > 4:
        raise ContractError("EMBER has a hard four GPU concurrency ceiling")
    for key in ("default_smoke_gpus", "default_pilot_gpus"):
        value = resources[key]
        if not isinstance(value, int) or value < 1 or value > maximum:
            raise ContractError(f"resources.{key} exceeds the four GPU contract")
    if resources["phase0_peak_growth_budget_bytes"] >= resources["personal_storage_cap_bytes"]:
        raise ContractError("Phase 0 growth budget must be smaller than the personal storage cap")


def _validate_environment(environment: Mapping[str, Any]) -> None:
    if environment.get("video_decode_backend") != "pyav":
        raise ContractError(
            "Phase 0 video decoder must remain the explicitly validated PyAV backend"
        )
    pyav_version = environment.get("pyav")
    if not isinstance(pyav_version, str) or re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", pyav_version
    ) is None:
        raise ContractError("Phase 0 PyAV video decoder version must be pinned")


def _validate_splits(splits: Mapping[str, Sequence[int]]) -> None:
    selected: list[int] = []
    for name, expected_size in EXPECTED_SPLIT_SIZES.items():
        values = splits[name]
        if len(values) != expected_size or len(set(values)) != expected_size:
            raise ContractError(f"task split {name} must contain {expected_size} unique tasks")
        selected.extend(values)
    if len(set(selected)) != 90 or set(selected) != set(range(90)):
        raise ContractError("task split must be disjoint and cover exactly LIBERO-90 indices 0..89")


def _expand_inclusive(bounds: Sequence[int], label: str) -> set[int]:
    if len(bounds) != 2 or not all(isinstance(value, int) for value in bounds):
        raise ContractError(f"episode authority {label} must be an inclusive [start, end] pair")
    start, end = bounds
    if start > end:
        raise ContractError(f"episode authority {label} has reversed bounds")
    return set(range(start, end + 1))


def _validate_episode_authority(authority: Mapping[str, Sequence[int]]) -> None:
    if set(authority) != EXPECTED_EPISODE_POOLS:
        raise ContractError("episode authority must declare the five canonical source pools")
    seen: set[int] = set()
    for label, bounds in authority.items():
        episodes = _expand_inclusive(bounds, label)
        if seen.intersection(episodes):
            raise ContractError("episode authority pools overlap")
        seen.update(episodes)
    if seen != set(range(50)):
        raise ContractError("episode authority must cover exactly episodes 0..49")


def _validate_held_contract(held: Mapping[str, Sequence[str]]) -> None:
    visible = set(held["writer_visible"])
    forbidden = set(held["writer_forbidden"])
    if visible.intersection(forbidden):
        raise ContractError("Writer-visible held inputs include a privileged forbidden field")
    required_forbidden = {
        "action",
        "proprioceptive_trajectory",
        "reward",
        "terminal_flag",
        "task_id",
        "filename",
        "hidden_normalization_statistics",
    }
    if not required_forbidden.issubset(forbidden):
        raise ContractError("held Writer forbidden-field contract is incomplete")


def _validate_dataset_surface(datasets: Mapping[str, Mapping[str, Any]]) -> None:
    canonical = datasets["libero_90"]
    expected = {
        "task_count": 90,
        "file_count": 90,
        "total_bytes": 66_658_085_995,
        "demos_per_task": 50,
        "hdf5_tag": "libero-v1",
    }
    if any(canonical.get(key) != value for key, value in expected.items()):
        raise ContractError("canonical LIBERO-90 dataset surface does not match the pinned revision")
    if canonical.get("license") != "CC-BY-4.0":
        raise ContractError("canonical LIBERO demonstrations must retain their CC-BY-4.0 license")
    assets = datasets["libero_assets"]
    expected_assets = {"snapshot_file_count": 586, "snapshot_bytes": 422_320_936}
    if any(assets.get(key) != value for key, value in expected_assets.items()):
        raise ContractError("LIBERO asset snapshot surface does not match the pinned revision")
    if assets.get("hub_reported_used_storage_bytes") != 492_798_408:
        raise ContractError("LIBERO asset Hub storage accounting is not recorded")
    if assets["snapshot_bytes"] == assets["hub_reported_used_storage_bytes"]:
        raise ContractError("snapshot bytes cannot be replaced by Hub repository storage accounting")
    if datasets["official_smoke"].get("role") != "mechanics_only_not_libero_90_authority":
        raise ContractError("the 40-task smoke dataset cannot become LIBERO-90 authority")


def _validate_model_roles(models: Mapping[str, Mapping[str, Any]]) -> None:
    if models["smolvla_base"].get("role") != "development_base":
        raise ContractError("SmolVLA base must remain the canonical development base")
    expected_smoke_role = "official_mechanics_only_never_ember_shared_base"
    if models["smolvla_libero_smoke"].get("role") != expected_smoke_role:
        raise ContractError("the all-task LIBERO checkpoint must remain mechanics-only")


def validate_contract(contract: Mapping[str, Any]) -> None:
    _validate_revisions(contract)
    _validate_environment(contract["environment"])
    _validate_resources(contract["resources"])
    _validate_splits(contract["splits"])
    _validate_episode_authority(contract["episode_authority"])
    _validate_held_contract(contract["held_contract"])
    _validate_dataset_surface(contract["datasets"])
    _validate_model_roles(contract["models"])


def contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase": contract["phase"],
        "lerobot_commit": contract["upstreams"]["lerobot"]["commit"],
        "smolvla_revision": contract["models"]["smolvla_base"]["revision"],
        "libero_runtime_commit": contract["upstreams"]["libero_runtime"]["commit"],
        "video_decode_backend": contract["environment"]["video_decode_backend"],
        "task_map_git_blob": contract["upstreams"]["libero_task_map"]["git_blob_sha"],
        "split_sizes": {name: len(contract["splits"][name]) for name in EXPECTED_SPLIT_SIZES},
        "max_concurrent_gpus": contract["resources"]["max_concurrent_gpus"],
        "phase0_peak_growth_budget_bytes": contract["resources"]["phase0_peak_growth_budget_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the EMBER Phase 0 experiment contract")
    parser.add_argument("contract", type=Path, nargs="?", default=Path("configs/phase0.toml"))
    args = parser.parse_args()
    contract = load_contract(args.contract)
    validate_contract(contract)
    print(json.dumps(contract_summary(contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
