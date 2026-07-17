"""Materialize the pinned, offline SmolVLA/LIBERO Phase 0 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ember.contracts import load_contract, validate_contract


class Phase0RuntimeError(RuntimeError):
    """Raised when runtime assets do not satisfy the pinned contract."""


@dataclass(frozen=True)
class WeightSnapshotSpec:
    """Immutable identity of one required model weight snapshot."""

    revision: str
    weight_sha256: str
    weight_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    if not path.is_file():
        raise Phase0RuntimeError(f"Missing {label}: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise Phase0RuntimeError(
            f"{label} has {actual_bytes} bytes, expected {expected_bytes}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise Phase0RuntimeError(
            f"{label} SHA256 is {actual_sha256}, expected {expected_sha256}"
        )


def _write_if_identical_or_missing(path: Path, content: str) -> bool:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise Phase0RuntimeError(f"Refusing to overwrite unexpected file: {path}")
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _verify_policy_inputs(
    source_policy: Path,
    vlm_snapshot: Path,
    source_spec: WeightSnapshotSpec,
    vlm_spec: WeightSnapshotSpec,
) -> None:
    _verify_file(
        source_policy / "model.safetensors",
        source_spec.weight_bytes,
        source_spec.weight_sha256,
        "source policy weight",
    )
    _verify_file(
        vlm_snapshot / "model.safetensors",
        vlm_spec.weight_bytes,
        vlm_spec.weight_sha256,
        "VLM weight",
    )
    for required in ("config.json", "tokenizer.json"):
        if not (vlm_snapshot / required).is_file():
            raise Phase0RuntimeError(
                f"Pinned VLM snapshot is missing {required}: {vlm_snapshot}"
            )


def _derive_policy_configs(
    source_policy: Path, vlm_snapshot: Path, upstream_vlm_name: str
) -> tuple[str, str, Path, Path]:
    source_config_path = source_policy / "config.json"
    source_processor_path = source_policy / "policy_preprocessor.json"
    if not source_config_path.is_file() or not source_processor_path.is_file():
        raise Phase0RuntimeError("Source policy is missing its config or preprocessor")

    policy_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    processor_config = json.loads(
        source_processor_path.read_text(encoding="utf-8")
    )
    if policy_config.get("vlm_model_name") != upstream_vlm_name:
        raise Phase0RuntimeError("Source policy VLM name differs from the contract")
    tokenizer_steps = [
        step
        for step in processor_config.get("steps", [])
        if step.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1:
        raise Phase0RuntimeError("Expected exactly one tokenizer processor step")
    tokenizer_name = tokenizer_steps[0].get("config", {}).get("tokenizer_name")
    if tokenizer_name != upstream_vlm_name:
        raise Phase0RuntimeError("Source tokenizer name differs from the contract")

    policy_config["vlm_model_name"] = str(vlm_snapshot)
    tokenizer_steps[0]["config"]["tokenizer_name"] = str(vlm_snapshot)
    return (
        _json_text(policy_config),
        _json_text(processor_config),
        source_config_path,
        source_processor_path,
    )


def _prepare_runtime_output(output_policy: Path, expected_names: set[str]) -> bool:
    if not output_policy.exists():
        output_policy.mkdir(parents=True)
        return True
    if not output_policy.is_dir():
        raise Phase0RuntimeError(f"Runtime view is not a directory: {output_policy}")
    unexpected = {path.name for path in output_policy.iterdir()} - expected_names
    if unexpected:
        raise Phase0RuntimeError(
            f"Runtime view contains unexpected entries: {sorted(unexpected)}"
        )
    return False


def _link_runtime_files(source_files: dict[str, Path], output_policy: Path) -> bool:
    changed = False
    for name, source_path in source_files.items():
        destination = output_policy / name
        if os.path.lexists(destination):
            if not destination.is_symlink() or destination.resolve() != source_path:
                raise Phase0RuntimeError(
                    f"Refusing to replace unexpected runtime-view entry: {destination}"
                )
            continue
        destination.symlink_to(os.path.relpath(source_path, output_policy))
        changed = True
    return changed


def _build_runtime_manifest(
    *,
    source_policy: Path,
    output_policy: Path,
    source_spec: WeightSnapshotSpec,
    source_config_path: Path,
    derived_config_text: str,
    source_processor_path: Path,
    derived_processor_text: str,
    vlm_snapshot: Path,
    vlm_spec: WeightSnapshotSpec,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_policy": os.path.relpath(source_policy, output_policy),
        "source_revision": source_spec.revision,
        "source_weight_bytes": source_spec.weight_bytes,
        "source_weight_sha256": source_spec.weight_sha256,
        "source_config_sha256": _sha256(source_config_path),
        "derived_config_sha256": hashlib.sha256(
            derived_config_text.encode("utf-8")
        ).hexdigest(),
        "source_preprocessor_sha256": _sha256(source_processor_path),
        "derived_preprocessor_sha256": hashlib.sha256(
            derived_processor_text.encode("utf-8")
        ).hexdigest(),
        "vlm_snapshot": os.path.relpath(vlm_snapshot, output_policy),
        "vlm_revision": vlm_spec.revision,
        "vlm_weight_bytes": vlm_spec.weight_bytes,
        "vlm_weight_sha256": vlm_spec.weight_sha256,
    }


def materialize_policy_runtime_view(
    *,
    source_policy: Path,
    vlm_snapshot: Path,
    output_policy: Path,
    upstream_vlm_name: str,
    source_spec: WeightSnapshotSpec,
    vlm_spec: WeightSnapshotSpec,
) -> dict[str, Any]:
    """Create a no-weight-copy policy view whose VLM references are local."""

    source_policy = source_policy.resolve()
    vlm_snapshot = vlm_snapshot.resolve()
    output_policy = output_policy.resolve()
    _verify_policy_inputs(
        source_policy,
        vlm_snapshot,
        source_spec,
        vlm_spec,
    )
    (
        derived_config_text,
        derived_processor_text,
        source_config_path,
        source_processor_path,
    ) = _derive_policy_configs(source_policy, vlm_snapshot, upstream_vlm_name)

    source_files = {
        path.name: path
        for path in source_policy.iterdir()
        if path.is_file()
        and path.name not in {"config.json", "policy_preprocessor.json"}
    }
    expected_names = set(source_files) | {
        "config.json",
        "policy_preprocessor.json",
        "runtime_manifest.json",
    }
    changed = _prepare_runtime_output(output_policy, expected_names)
    changed |= _link_runtime_files(source_files, output_policy)
    changed |= _write_if_identical_or_missing(
        output_policy / "config.json", derived_config_text
    )
    changed |= _write_if_identical_or_missing(
        output_policy / "policy_preprocessor.json", derived_processor_text
    )
    manifest = _build_runtime_manifest(
        source_policy=source_policy,
        output_policy=output_policy,
        source_spec=source_spec,
        source_config_path=source_config_path,
        derived_config_text=derived_config_text,
        source_processor_path=source_processor_path,
        derived_processor_text=derived_processor_text,
        vlm_snapshot=vlm_snapshot,
        vlm_spec=vlm_spec,
    )
    changed |= _write_if_identical_or_missing(
        output_policy / "runtime_manifest.json", _json_text(manifest)
    )
    return {"created": changed, "manifest": manifest}


def _verify_libero_asset_snapshot(
    asset_snapshot: Path, expected_file_count: int, expected_bytes: int
) -> None:
    required_subdirectories = (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
        "scenes",
    )
    missing = [
        name for name in required_subdirectories if not (asset_snapshot / name).is_dir()
    ]
    if missing:
        raise Phase0RuntimeError(
            f"Pinned LIBERO asset snapshot is incomplete: {missing}"
        )
    asset_files = [
        path
        for path in asset_snapshot.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(asset_snapshot).parts
    ]
    asset_bytes = sum(path.stat().st_size for path in asset_files)
    if len(asset_files) != expected_file_count or asset_bytes != expected_bytes:
        raise Phase0RuntimeError(
            "Pinned LIBERO asset surface differs from the contract: "
            f"files={len(asset_files)}, bytes={asset_bytes}"
        )


def _ensure_libero_asset_link(benchmark_root: Path, asset_snapshot: Path) -> bool:
    package_assets = benchmark_root / "assets"
    if os.path.lexists(package_assets):
        if not package_assets.is_symlink() or package_assets.resolve() != asset_snapshot:
            raise Phase0RuntimeError(
                f"Refusing to replace existing LIBERO package assets: {package_assets}"
            )
        return False
    package_assets.symlink_to(os.path.relpath(asset_snapshot, benchmark_root))
    return True


def write_libero_config(
    *,
    config_root: Path,
    site_packages: Path,
    asset_snapshot: Path,
    data_root: Path,
    expected_asset_file_count: int,
    expected_asset_bytes: int,
) -> bool:
    """Write the isolated LIBERO path map without triggering import prompts."""

    config_root = config_root.resolve()
    site_packages = site_packages.resolve()
    asset_snapshot = asset_snapshot.resolve()
    data_root = data_root.resolve()
    benchmark_root = site_packages / "libero" / "libero"
    bddl_files = benchmark_root / "bddl_files"
    init_states = benchmark_root / "init_files"
    for required in (benchmark_root, bddl_files, init_states, data_root):
        if not required.is_dir():
            raise Phase0RuntimeError(f"Required LIBERO path is missing: {required}")
    _verify_libero_asset_snapshot(
        asset_snapshot, expected_asset_file_count, expected_asset_bytes
    )
    changed = _ensure_libero_asset_link(benchmark_root, asset_snapshot)

    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "assets": str(asset_snapshot),
        "bddl_files": str(bddl_files),
        "benchmark_root": str(benchmark_root),
        "datasets": str(data_root),
        "init_states": str(init_states),
    }
    changed |= _write_if_identical_or_missing(
        config_root / "config.yaml", _json_text(config)
    )
    return changed


def _required_path(value: str | None, environment_name: str) -> Path:
    if not value:
        raise Phase0RuntimeError(
            f"Provide the path argument or set {environment_name}"
        )
    return Path(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/phase0.toml"))
    parser.add_argument("--asset-root", default=os.environ.get("EMBER_ASSET_ROOT"))
    parser.add_argument("--data-root", default=os.environ.get("EMBER_DATA_ROOT"))
    parser.add_argument(
        "--libero-config-root",
        default=os.environ.get("LIBERO_CONFIG_PATH"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = load_contract(args.contract)
    validate_contract(contract)
    asset_root = _required_path(args.asset_root, "EMBER_ASSET_ROOT").resolve()
    data_root = _required_path(args.data_root, "EMBER_DATA_ROOT").resolve()
    config_root = _required_path(
        args.libero_config_root, "LIBERO_CONFIG_PATH"
    ).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    smoke = contract["models"]["smolvla_libero_smoke"]
    vlm = contract["models"]["smolvlm_constructor_dependency"]
    assets = contract["datasets"]["libero_assets"]
    source_policy = (
        asset_root / "models" / smoke["repo_id"].split("/")[-1] / smoke["revision"]
    )
    vlm_snapshot = (
        asset_root / "models" / vlm["repo_id"].split("/")[-1] / vlm["revision"]
    )
    output_policy = asset_root / "runtime" / "smolvla_libero" / smoke["revision"]
    asset_snapshot = (
        asset_root
        / "datasets"
        / assets["repo_id"].split("/")[-1]
        / assets["revision"]
    )

    policy_result = materialize_policy_runtime_view(
        source_policy=source_policy,
        vlm_snapshot=vlm_snapshot,
        output_policy=output_policy,
        upstream_vlm_name=vlm["repo_id"],
        source_spec=WeightSnapshotSpec(
            revision=smoke["revision"],
            weight_sha256=smoke["weight_sha256"],
            weight_bytes=smoke["weight_bytes"],
        ),
        vlm_spec=WeightSnapshotSpec(
            revision=vlm["revision"],
            weight_sha256=vlm["weight_sha256"],
            weight_bytes=vlm["weight_bytes"],
        ),
    )
    config_created = write_libero_config(
        config_root=config_root,
        site_packages=Path(sysconfig.get_paths()["purelib"]),
        asset_snapshot=asset_snapshot,
        data_root=data_root,
        expected_asset_file_count=assets["snapshot_file_count"],
        expected_asset_bytes=assets["snapshot_bytes"],
    )
    print(
        json.dumps(
            {
                "libero_config_created": config_created,
                "policy_view_created": policy_result["created"],
                "policy_view": str(output_policy),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
