#!/usr/bin/env python3
"""Replay frozen 529da6b on the current host without changing its model graph.

This temporary operator owns only path relocation, current host admission, and
strict local weight loading. The archived evaluator still owns all generation,
video scheduling, queues, policy steps, and result evidence. Retire this entry
after the fixed step900 reference completes; retain both frozen code commits.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType


ARCHIVE_COMMIT = "529da6bbe290f7393422937aa7cc278cee732107"
OPERATOR_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_function(module: ModuleType, name: str, changes: dict[str, str]) -> None:
    """Apply exact, reviewable host-only substitutions to the archived function."""
    source = inspect.getsource(getattr(module, name))
    for old, new in changes.items():
        if source.count(old) != 1:
            raise RuntimeError(f"archived compatibility site changed: {name}: {old}")
        source = source.replace(old, new)
    exec(compile(source, f"{__file__}:compatibility:{name}", "exec"), module.__dict__)


def install_path_relocation(assets: Path) -> None:
    from ember import pi05_eval_contract
    from ember.writer import inference

    roots = {
        "/data/ymdai/outputs/ember": assets / "runs/outputs",
        "/data/ymdai/ember_data/LIBERO-datasets": assets / "data/datasets",
    }

    def relocated_path(value: object) -> str:
        text = str(value)
        for old, new in roots.items():
            if text == old or text.startswith(old + "/"):
                return str((new / text[len(old):].lstrip("/")).resolve())
        return str(Path(text).resolve())

    def relocated_source(value: object) -> object:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for key in ("source_run", "checkpoint", "model_path"):
            if key in result:
                result[key] = relocated_path(result[key])
        return result

    pi05_eval_contract._v52_relocated_path = relocated_path
    inference._v52_relocated_path = relocated_path
    inference._v52_relocated_source = relocated_source
    patch_function(pi05_eval_contract, "_validate_final_source_policy", {
        'Path(str(summary.get("final_checkpoint", ""))).resolve() == checkpoint':
        'Path(_v52_relocated_path(summary.get("final_checkpoint", ""))) == checkpoint',
    })
    patch_function(inference, "_inspect_training_checkpoint", {
        'training_source = training.get("source")':
        'training_source = _v52_relocated_source(training.get("source"))',
    })
    patch_function(inference, "inspect_as_writer_evaluation", {
        'training_video.get("root") != video_data.get("root")':
        '_v52_relocated_path(training_video.get("root")) != video_data.get("root")',
    })


def install_local_assets(assets: Path) -> None:
    from ember import pi05_assets, pi05_eval_contract
    from ember.pi05_eval import worker_setup

    current_assets = load_module("_v52_current_assets", OPERATOR_ROOT / "src/ember/pi05_assets.py")
    prepare = current_assets.prepare_libero_config

    def prepare_and_bind(config_dir: Path) -> dict[str, str]:
        paths = prepare(config_dir)
        current_assets.configure_libero_runtime_assets(Path(paths["assets"]))
        return paths

    pi05_assets.prepare_libero_config = prepare_and_bind
    pi05_eval_contract.prepare_libero_config = prepare_and_bind
    # The installed upstream from_pretrained catches load failures. Reuse the
    # canonical strict loader, preserving the evaluator's native-key remapping.
    setup = load_module("_v52_current_source_setup", OPERATOR_ROOT / "src/ember/pi05_source_setup.py")
    worker_setup._v52_strict_load = setup.load_pretrained_policy
    patch_function(worker_setup, "load_policy", {
        'PI05Policy.from_pretrained(\n        model_path,\n        config=config,\n        local_files_only=True,\n        strict=True,\n    )':
        '_v52_strict_load(model_path, config, remap_native_keys=True)',
    })
    os.environ["EMBER_LIBERO_ASSETS_ROOT"] = str(
        assets / "data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
    )
    if os.environ.get("LIBERO_CONFIG_PATH"):
        current_assets.configure_libero_runtime_assets(Path(os.environ["EMBER_LIBERO_ASSETS_ROOT"]))


def install_host_admission(evaluator: ModuleType, archive: Path) -> None:
    current = load_module("_v52_current_launcher", OPERATOR_ROOT / "src/ember/pi05_eval/launcher.py")
    original_spawn = evaluator.spawn_worker_processes

    def preflight(indices: tuple[int, ...]) -> dict:
        resources_path = Path(os.environ["EMBER_V52_RESOURCES"])
        resources = json.loads(resources_path.read_text())
        if time.time() - float(resources["unix"]) > 1800:
            raise RuntimeError("refresh the dual-node GPU and quota registration before launching")
        if len(indices) > 6:
            raise RuntimeError("v5.2 reference uses at most six GPUs on one node")
        result = current.gpu_preflight(indices, materialized_lora_replicas=3)
        if not current.evaluator_gpus_are_eligible(result):
            raise RuntimeError("selected GPUs no longer have the registered headroom")
        result.update(
            personal_bytes=int(resources["quota_used_bytes"]),
            personal_cap_bytes=int(resources["quota_soft_bytes"]),
            current_resource_registration=str(resources_path),
            archive_commit=ARCHIVE_COMMIT,
            operator_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=OPERATOR_ROOT, text=True
            ).strip(),
        )
        if result["personal_bytes"] + int(resources["projected_additional_bytes"]) >= result["personal_cap_bytes"]:
            raise RuntimeError("registered reference peak exceeds the data1 quota")
        return result

    def spawn(*args, **kwargs):
        kwargs["script_path"] = Path(__file__).resolve()
        kwargs["repo_root"] = archive
        return original_spawn(*args, **kwargs)

    evaluator._gpu_preflight = preflight
    evaluator.spawn_worker_processes = spawn


def install_runtime() -> ModuleType:
    archive = Path(os.environ["EMBER_V52_RUNTIME"]).resolve()
    assets = Path(os.environ["EMBER_ASSET_ROOT"]).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=archive, text=True).strip()
    if commit != ARCHIVE_COMMIT:
        raise RuntimeError("v5.2 reference must use the original frozen 529da6b tree")
    sys.path.insert(0, str(archive / "src"))
    install_path_relocation(assets)
    install_local_assets(assets)
    evaluator = load_module("_v52_archived_evaluator", archive / "scripts/evaluate_pi05.py")
    install_host_admission(evaluator, archive)
    return evaluator


def main() -> int:
    return install_runtime().main()


if __name__ == "__main__":
    raise SystemExit(main())
