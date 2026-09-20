"""One explicit task authority for training, video compilation and evaluation."""
from pathlib import Path

from ember.pi05_assets import load_protocol
from ember.pi05_source_checkpoint import read_json


DEFAULT_PROTOCOL = "configs/libero_24_8_8_v1/protocol.json"
DEFAULT_MANIFEST = "configs/pi05_target_data_v1/manifest.json"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _add_auxiliary_tasks(asset_root, protocol, canonical):
    auxiliary = protocol.get("auxiliary_train")
    allowed = set()
    if auxiliary:
        if auxiliary.get("suite") != "libero_90" or auxiliary.get("global_task_id_offset") != 40:
            raise ValueError("auxiliary tasks require a distinct registered LIBERO-90 identity")
        ids = auxiliary["task_ids"]
        if len(ids) != len(set(ids)) or any(type(i) is not int for i in ids):
            raise ValueError("auxiliary task allowlist must contain unique integer IDs")
        source = read_json(asset_root / "configs/pi05_source_corpus_v1/source_manifest.json")
        source_rows = {row["task_index"]: row for row in source["tasks"]}
        if set(ids) - source_rows.keys():
            raise ValueError("auxiliary task is outside the audited Source-71 allowlist")
        for task in ids:
            row = source_rows[task]
            canonical[40 + task] = {**row, "suite": "libero_90", "task_id": task,
                "hdf5": {**row["hdf5"], "relative_path": "libero_90/" + row["hdf5"]["filename"]}}
            allowed.add(40 + task)
    return allowed


def load_task_authorities(asset_root: Path, protocol_path: str | None = None):
    """Resolve roles without changing canonical data identities or reading labels."""
    protocol = load_protocol(asset_root / (protocol_path or DEFAULT_PROTOCOL))
    manifest = read_json(asset_root / protocol.get("data_manifest", DEFAULT_MANIFEST))
    targets = read_json(asset_root / DEFAULT_MANIFEST)
    canonical = {row["global_task_id"]: row for row in targets["tasks"]}
    allowed = _add_auxiliary_tasks(asset_root, protocol, canonical)
    rows = manifest["tasks"]
    if (len(rows) != len(canonical) or {row["global_task_id"] for row in rows} != set(canonical)
            or manifest["dataset"]["revision"] != targets["dataset"]["revision"]):
        raise ValueError("task manifest does not cover the registered data identities exactly")
    roles = {role: [] for role in ("train", "validation", "test")}
    for row in rows:
        key = row["global_task_id"]
        original = canonical[key]
        if (any(row[field] != original[field] for field in ("suite", "task_id", "language"))
                or any(row["hdf5"][field] != original["hdf5"][field] for field in ("relative_path", "bytes"))
                or row["demonstrations"] != original["demonstrations"]):
            raise ValueError("task manifest changed canonical task or episode metadata")
        role = ("train" if key in allowed else next(role for role in roles
            if row["task_id"] in protocol["split"]["suites"][row["suite"]][role]))
        if row["split_role"] != role:
            raise ValueError("task role differs between protocol and manifest")
        roles[role].append(key)
    if any(sorted(manifest["summary"]["roles"][role]) != sorted(ids) for role, ids in roles.items()):
        raise ValueError("task manifest role summary changed")
    return protocol, manifest
