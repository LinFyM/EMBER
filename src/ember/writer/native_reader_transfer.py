"""Exact parent partition and bank authority for the frozen N/W transfer study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json


SPEC = Path("configs/native_reader_transfer_causality_v1/experiment_spec.json")
REPO_ROOT = Path(__file__).resolve().parents[3]
NAMES = ("N0_W0", "N1_W0", "N0_W1", "N1_W1")


def authority() -> dict[str, Any]:
    spec = read_json(REPO_ROOT / SPEC)
    if (spec.get("schema_version") != "ember_native_reader_transfer_causality_spec_v1"
            or spec.get("study_id") != "native_reader_transfer_causality_20260925"
            or [spec["parents"][index]["arm"] for index in ("0", "1")] != ["C_S00", "C_S10"]
            or [cell.get("id") for cell in spec.get("cells", [])] != list(NAMES)
            or [(row["global_task_id"], row["suite"], row["task_id"])
                for row in spec["evaluation"]["tasks"]] != [
                    (14, "libero_object", 4), (21, "libero_goal", 1)]
            or spec["evaluation"]["init_state_ids"] != list(range(50))
            or spec["evaluation"]["full_states"] != [0, 25]
            or "seed 20260911" not in spec["evaluation"]["teacher_schedule"]
            or spec["evaluation"]["new_rollouts"] != 400
            or spec["evaluation"]["full_cases"] != 16):
        raise ValueError("native reader transfer registration changed")
    return spec


def cell_spec(name: str) -> dict[str, Any]:
    return next((cell for cell in authority()["cells"] if cell["id"] == name), None) or _bad_cell(name)


def _bad_cell(name: str) -> dict[str, Any]:
    raise ValueError(f"unregistered native reader transfer cell: {name}")


def parent_checkpoint(index: str) -> Path:
    return Path(authority()["parents"][index]["checkpoint"]).resolve()


def partition(states: Mapping[str, torch.Tensor], spec: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    rule = spec["parameter_partition"]
    names = set(states)
    native = {key for key in names if any(key.startswith(prefix) for prefix in rule["N_prefixes"])}
    fixed = {key for key in names if key.startswith(tuple(rule["fixed_buffer_prefixes"]))
             or key in rule["fixed_buffer_names"]}
    writer = names - native - fixed
    if (len(names) != rule["total_checkpoint_tensors"] or len(native) != rule["N_tensors"]
            or sum(states[key].numel() for key in native) != rule["N_elements"]
            or len(writer) != rule["W_tensors"] or len(fixed) != rule["fixed_buffer_tensors"]
            or native & fixed or not all(key.endswith((".a", ".b")) for key in native)
            or not all(key.startswith("writer.") for key in names)):
        raise ValueError("native/W/fixed checkpoint partition changed")
    return native, writer, fixed


def composed_state(n_path: Path, w_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load complete W parent then replace exactly the registered native A/B tensors."""
    spec = authority()
    actual = {str(n_path.resolve()), str(w_path.resolve())}
    allowed = {str(parent_checkpoint(index)) for index in ("0", "1")}
    if not actual <= allowed:
        raise ValueError("native transfer parent checkpoint is outside the two frozen endpoints")
    n = load_file(str(n_path / "ecp.safetensors"), device="cpu")
    w = load_file(str(w_path / "ecp.safetensors"), device="cpu")
    if set(n) != set(w) or any(n[key].shape != w[key].shape or n[key].dtype != w[key].dtype for key in n):
        raise ValueError("native transfer parents differ in checkpoint topology")
    native, writer, fixed = partition(w, spec)
    if partition(n, spec) != (native, writer, fixed):
        raise ValueError("native transfer parent roles differ")
    if any(not torch.equal(n[key], w[key]) for key in fixed):
        raise ValueError("native transfer source templates or fixed probe differ")
    if any(not torch.isfinite(value).all() for state in (n, w) for value in state.values()):
        raise ValueError("native transfer parent contains nonfinite values")
    for key in native:
        w[key] = n[key]
    return w, {"N_keys": sorted(native), "W_keys": sorted(writer), "fixed_keys": sorted(fixed),
               "N_elements": sum(w[key].numel() for key in native)}


def validate_materialization(*, cell: str, checkpoint: Path, output: Path,
                             selection: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    from ember.writer.materialization import selection_contract

    spec = authority()
    definition = cell_spec(cell)
    if definition["endpoint_bank_reuse"]:
        raise ValueError("self cells must reuse their sealed E3 endpoint bank")
    n_parent, w_parent = definition["N_parent"], definition["W_parent"]
    expected = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
        arm="correct", mode="per_init_ordinal", seed=20260911,
        init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
    study = Path(spec["resources"]["study_root"]).resolve()
    if (checkpoint.resolve() != parent_checkpoint(w_parent)
            or output.resolve() != study / "materialization" / cell
            or dict(selection) != expected
            or run.get("git", {}).get("commit") != spec["parents"][w_parent]["training_commit"]
            or run.get("config", {}).get("experiment", {}).get("arm_id") != spec["parents"][w_parent]["arm"]):
        raise ValueError("native transfer materialization differs from its exact frozen cell")
    from ember.writer.materialization import inspect_writer_checkpoint, source_matches

    n_run, n_record = inspect_writer_checkpoint(parent_checkpoint(n_parent))
    if (n_run["git"]["commit"] != spec["parents"][n_parent]["training_commit"]
            or n_run["config"]["experiment"]["arm_id"] != spec["parents"][n_parent]["arm"]
            or n_run["model_config"] != run["model_config"]
            or n_run["config"]["observer"] != run["config"]["observer"]
            or n_run["config"]["data"]["protocol"] != run["config"]["data"]["protocol"]
            or not source_matches(n_run["source"], run["source"])):
        raise ValueError("native transfer parents differ in source, topology or authority")
    # All 622 roles and the 77 immutable tensors are checked before GPU work.
    _, roles = composed_state(parent_checkpoint(n_parent), checkpoint)
    return {"schema_version": "ember_native_reader_transfer_bank_v1", "cell": cell,
            "N_parent": n_parent, "W_parent": w_parent, "N_checkpoint": n_record,
            "W_checkpoint": str(checkpoint.resolve()), "study_spec": str(SPEC),
            "partition": {key: roles[key] for key in ("N_keys", "W_keys", "fixed_keys", "N_elements")}}


def registered_transfers(requests, inspected) -> list[dict[str, Any] | None]:
    checked = []
    for request, (run, record) in zip(requests, inspected, strict=True):
        cell = request.get("native_transfer_cell")
        if not cell:
            checked.append(None)
            continue
        if request.get("reuse_manifest") is not None or request.get("diagnostic_contract") is not None:
            raise ValueError("mixed native transfer banks require 100 new complete Writer forwards")
        checked.append(validate_materialization(cell=cell, checkpoint=Path(record["path"]),
            output=Path(request["output"]), selection=request["selection"], run=run))
    return checked


def registered_panels(requests, inspected, transfers):
    from ember.writer.materialization import _registered_request_panel

    return [None if transfer else _registered_request_panel(request, run, record)
            for request, (run, record), transfer in zip(requests, inspected, transfers, strict=True)]


def attach_manifest(manifest: dict[str, Any], transfer: Mapping[str, Any] | None) -> None:
    if transfer is not None:
        manifest["native_reader_transfer"] = dict(transfer)


def expected_bank(name: str) -> Path:
    spec = authority()
    cell = cell_spec(name)
    if cell["endpoint_bank_reuse"]:
        return Path(spec["parents"][cell["W_parent"]]["stage1_correct_bank"]).resolve()
    return Path(spec["resources"]["study_root"]).resolve() / "materialization" / name / "manifest.json"
