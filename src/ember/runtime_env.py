"""Narrow compatibility repairs for the pinned Phase 0 runtime.

The repairs here are deliberately version- and content-guarded.  They address
two upstream packaging defects without changing simulator or policy behavior.
"""

from __future__ import annotations

import argparse
import json
import sysconfig
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any


BDDL_VERSION = "1.0.1"
LEROBOT_VERSION = "0.6.0"
ROBOSUITE_VERSION = "1.4.0"
ROBOSUITE_DEFAULT_FILE_LOGGING = 'FILE_LOGGING_LEVEL = "DEBUG"'
ROBOSUITE_PRIVATE_MACROS = '''"""EMBER override for robosuite 1.4.0 shared-/tmp compatibility."""

import robosuite.macros as _macros

# Upstream writes every process to the global /tmp/robosuite.log.  Durable
# experiment stdout/stderr logging remains enabled by the launch wrapper.
_macros.FILE_LOGGING_LEVEL = None
'''


class RuntimeEnvironmentError(RuntimeError):
    """Raised when a repair cannot be applied without guessing."""


def _require_distribution(
    site_packages: Path, distribution: str, version: str
) -> Path:
    metadata_path = site_packages / f"{distribution}-{version}.dist-info" / "METADATA"
    if not metadata_path.is_file():
        raise RuntimeEnvironmentError(
            f"Expected {distribution}=={version} metadata at {metadata_path}"
        )

    metadata = BytesParser(policy=default).parsebytes(metadata_path.read_bytes())
    if metadata.get("Name") != distribution or metadata.get("Version") != version:
        raise RuntimeEnvironmentError(
            f"Expected {distribution}=={version}, found "
            f"{metadata.get('Name')}=={metadata.get('Version')}"
        )
    return metadata_path


def _remove_duplicate_metadata(
    site_packages: Path, distribution: str, version: str
) -> bool:
    metadata_path = _require_distribution(site_packages, distribution, version)
    duplicate_path = site_packages / f"{distribution}-{version}.egg-info"
    if not duplicate_path.exists():
        return False
    if not duplicate_path.is_file():
        raise RuntimeEnvironmentError(
            f"Expected malformed {distribution} metadata to be a file: "
            f"{duplicate_path}"
        )
    if duplicate_path.read_bytes() != metadata_path.read_bytes():
        raise RuntimeEnvironmentError(
            f"{distribution} duplicate metadata does not match the installed "
            "distribution; refusing to remove it"
        )
    duplicate_path.unlink()
    return True


def _install_robosuite_private_macros(site_packages: Path) -> bool:
    _require_distribution(site_packages, "robosuite", ROBOSUITE_VERSION)
    package_path = site_packages / "robosuite"
    macros_path = package_path / "macros.py"
    if not macros_path.is_file():
        raise RuntimeEnvironmentError(f"Missing robosuite macros file: {macros_path}")
    if ROBOSUITE_DEFAULT_FILE_LOGGING not in macros_path.read_text(encoding="utf-8"):
        raise RuntimeEnvironmentError(
            "Pinned robosuite file-logging default was not found; refusing an "
            "unreviewed compatibility override"
        )

    private_path = package_path / "macros_private.py"
    if private_path.exists():
        if private_path.is_file() and private_path.read_text(
            encoding="utf-8"
        ) == ROBOSUITE_PRIVATE_MACROS:
            return False
        raise RuntimeEnvironmentError(
            f"Refusing to overwrite existing robosuite private macros: {private_path}"
        )
    private_path.write_text(ROBOSUITE_PRIVATE_MACROS, encoding="utf-8")
    return True


def repair_runtime_environment(site_packages: Path) -> dict[str, bool]:
    """Apply the audited repairs to one Python site-packages directory."""

    site_packages = site_packages.resolve()
    if not site_packages.is_dir():
        raise RuntimeEnvironmentError(
            f"site-packages directory does not exist: {site_packages}"
        )
    return {
        "bddl_metadata_removed": _remove_duplicate_metadata(
            site_packages, "bddl", BDDL_VERSION
        ),
        "lerobot_metadata_removed": _remove_duplicate_metadata(
            site_packages, "lerobot", LEROBOT_VERSION
        ),
        "robosuite_override_created": _install_robosuite_private_macros(
            site_packages
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path(sysconfig.get_paths()["purelib"]),
        help="site-packages directory to repair (defaults to the active interpreter)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result: dict[str, Any] = repair_runtime_environment(args.site_packages)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
