from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.functional_adaptation.phase_alignment import (
    FunctionalWhitener,
    arc_length_phase_embedding,
    arc_length_phase_indices,
    fit_task_equal_whitener,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.reward.occupancy import load_successful_occupancy_trajectory
from ember.reward.rollout import query_successful_expert_occupancy_actions
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHARD_SCHEMA = "ember_ecp_stage1_mdco_source_response_shard_v1"
AUTHORITY_SCHEMA = "ember_ecp_stage1_mapping_diverse_authority_v1"
PHASE_TRANSFORM_SCHEMA = "ember_ecp_stage1_mdco_phase_transform_v1"
CAPTURE_SCHEMA = "ember_successful_expert_occupancy_capture_v1"
TARGET_ANALYSIS_SCHEMA = "ember_successful_expert_equivalence_phase_analysis_v1"
TARGET_SELECTION_SCHEMA = "ember_successful_expert_equivalence_selection_v1"

def _repo_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        parts = resolved.parts
        if "runs" in parts:
            return str(Path(*parts[parts.index("runs") :]))
    return str(resolved)

def _resolve_reference(asset_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()

def _save_torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)

def _save_safetensors_atomic(path: Path, value: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(dict(value), str(temporary))
    os.replace(temporary, path)

def _source_task_rows(source_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    active = tuple(
        int(value) for value in source_manifest["summary"]["active_source_task_ids"]
    )
    by_id = {int(row["task_index"]): dict(row) for row in source_manifest["tasks"]}
    if (
        source_manifest.get("schema_version") != "ember_pi05_source_manifest_v1"
        or len(active) != 71
        or len(set(active)) != 71
        or any(value not in by_id for value in active)
    ):
        raise ValueError("MDCO source-task authority changed")
    return tuple(by_id[value] for value in active)

def _load_source_capture(
    capture_run: Path, source_manifest: Mapping[str, Any]
):
    contract = read_json(capture_run / "run_contract.json")
    results = read_json(capture_run / "results.json")
    capture = contract.get("diagnostic_occupancy_capture", {})
    rows = tuple(dict(row) for row in results.get("rows", ()))
    adapters = tuple(dict(row) for row in contract.get("adapter", {}).get("tasks", ()))
    source_ids = tuple(
        int(row["task_index"]) for row in _source_task_rows(source_manifest)
    )
    valid_capture = (
        results.get("mode") == "formal",
        results.get("role") == "nonheld_meta",
        int(results.get("overall", {}).get("episodes", -1)) == 142,
        int(results.get("overall", {}).get("successes", -1)) == 141,
        len(rows) == 142,
        contract.get("git", {}).get("dirty_paths") == [],
        capture.get("schema_version") == CAPTURE_SCHEMA,
        int(capture.get("selected_rows", -1)) == 142,
        int(capture.get("selected_tasks", -1)) == 71,
        capture.get("training_gradient_use") is True,
        capture.get("held_data_use") is False,
        len(adapters) == 71,
    )
    if not all(valid_capture):
        raise ValueError("MDCO source successful-occupancy capture changed")
    rows_by_task: dict[int, list[dict[str, Any]]] = {}
    failed = [row for row in rows if not bool(row.get("success"))]
    if len(failed) != 1:
        raise ValueError("MDCO source capture failure count changed")
    for row in rows:
        task_id = int(row["task_id"])
        if str(row.get("suite")) != "libero_90":
            raise ValueError("MDCO source capture suite changed")
        if bool(row.get("success")):
            rows_by_task.setdefault(task_id, []).append(row)
    adapters_by_task = {int(row["task_id"]): row for row in adapters}
    valid_coverage = (
        set(rows_by_task) == set(source_ids),
        set(adapters_by_task) == set(source_ids),
        sum(len(values) for values in rows_by_task.values()) == 141,
        sum(len(values) == 1 for values in rows_by_task.values()) == 1,
        all(len(values) in {1, 2} for values in rows_by_task.values()),
    )
    if not all(valid_coverage):
        raise ValueError("MDCO source capture lost mapping coverage")
    for values in rows_by_task.values():
        values.sort(key=lambda row: int(row["init_state_id"]))
    return contract, results, rows_by_task, adapters_by_task

def _trajectory_action_deltas(
    actions: Mapping[str, Sequence[Sequence[torch.Tensor]]]
) -> tuple[torch.Tensor, ...]:
    result = []
    for expert, source in zip(actions["expert"], actions["student"], strict=True):
        value = torch.cat(
            [
                left.float() - right.float()
                for left, right in zip(expert, source, strict=True)
            ]
        ).contiguous()
        if (
            value.ndim != 3
            or value.shape[1:] != (50, 7)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("MDCO complete action response shape changed")
        result.append(value)
    return tuple(result)

def build_source_response_shard(args: Any) -> None:
    """Re-query one disjoint source-task shard at retained successful states."""

    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("MDCO source response requires a clean pushed authority")
    source_manifest = read_json(args.source_manifest)
    source_tasks = _source_task_rows(source_manifest)
    contract, _, rows_by_task, adapters_by_task = _load_source_capture(
        args.capture_run, source_manifest
    )
    selected = [
        (ordinal, row)
        for ordinal, row in enumerate(source_tasks)
        if ordinal % args.shard_count == args.shard_index
    ]
    if not selected:
        raise ValueError("MDCO source response shard is empty")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    source_config = read_json(REPO_ROOT / "configs/pi05_source_base_v1.json")
    policy = load_policy(args.source_checkpoint / "policy", source_config, device)
    lora = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    prepare_frozen_writer_policy(policy, lora)
    identity = identity_lora_state(lora, device=device)
    replan_steps = int(contract["policy"]["replan_steps"])
    members = []
    for ordinal, task in selected:
        task_id = int(task["task_index"])
        adapter = adapters_by_task[task_id]
        checkpoint = Path(str(adapter["checkpoint"])).resolve()
        if (
            int(adapter.get("global_task_id", -1)) != task_id
            or int(adapter.get("step", -1)) != 1000
            or str(adapter.get("language")) != str(task["language"])
            or not (checkpoint / "adapter.safetensors").is_file()
            or any(
                int(row.get("task_expert", {}).get("step", -1)) != 1000
                or int(row.get("task_expert", {}).get("global_task_id", -1)) != task_id
                or Path(str(row.get("task_expert", {}).get("checkpoint"))) != checkpoint
                for row in rows_by_task[task_id]
            )
        ):
            raise ValueError("MDCO source task expert changed")
        trajectory_rows = rows_by_task[task_id]
        trajectories = tuple(
            load_successful_occupancy_trajectory(
                row=row,
                suite="libero_90",
                task_id=task_id,
                global_task_id=task_id,
                replan_steps=replan_steps,
            )
            for row in trajectory_rows
        )
        expert = load_file(str(checkpoint / "adapter.safetensors"), device=str(device))
        actions, requery = query_successful_expert_occupancy_actions(
            policy=policy,
            lora_contract=lora,
            identity_state=identity,
            trajectories=trajectories,
            expert_lora=expert,
            student_lora=identity,
            device=device,
            microbatch_size=args.microbatch_size,
            num_inference_steps=10,
        )
        deltas = _trajectory_action_deltas(actions)
        members.append(
            {
                "ordinal": ordinal,
                "task_id": task_id,
                "global_task_id": task_id,
                "language": str(task["language"]),
                "expert_step": 1000,
                "checkpoint": _repo_reference(checkpoint),
                "requery": requery,
                "trajectories": [
                    {
                        "init_state_id": int(row["init_state_id"]),
                        "path": _repo_reference(
                            Path(str(row["occupancy_trajectory"]["path"]))
                        ),
                        "bytes": int(row["occupancy_trajectory"]["bytes"]),
                        "replans": int(row["occupancy_trajectory"]["replans"]),
                        "steps": int(row["steps"]),
                        "action_delta": delta,
                    }
                    for row, delta in zip(trajectory_rows, deltas, strict=True)
                ],
            }
        )
        print(
            {
                "ordinal": ordinal,
                "task_id": task_id,
                "replans": [int(value.shape[0]) for value in deltas],
                "response_rms": [
                    float(value.square().mean().sqrt()) for value in deltas
                ],
            },
            flush=True,
        )
    _save_torch_atomic(
        args.output,
        {
            "schema_version": SOURCE_SHARD_SCHEMA,
            "repository": repository,
            "capture_run": _repo_reference(args.capture_run),
            "source_manifest": _repo_reference(args.source_manifest),
            "source_checkpoint": _repo_reference(args.source_checkpoint),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "microbatch_size": args.microbatch_size,
            "members": members,
        },
    )

def _load_source_shards(
    paths: Sequence[Path], *, source_tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    shards = [
        torch.load(path, map_location="cpu", weights_only=False) for path in paths
    ]
    if not shards:
        raise ValueError("MDCO source response shards changed")
    count = len(shards)
    valid_shards = (
        all(row.get("schema_version") == SOURCE_SHARD_SCHEMA for row in shards),
        {int(row["shard_count"]) for row in shards} == {count},
        {int(row["shard_index"]) for row in shards} == set(range(count)),
        all(row["repository"] == shards[0]["repository"] for row in shards),
        all(row["capture_run"] == shards[0]["capture_run"] for row in shards),
    )
    if not all(valid_shards):
        raise ValueError("MDCO source response shards changed")
    members = [dict(member) for shard in shards for member in shard["members"]]
    members.sort(key=lambda row: int(row["ordinal"]))
    valid_members = (
        [int(row["ordinal"]) for row in members] == list(range(71)),
        [int(row["task_id"]) for row in members]
        == [int(row["task_index"]) for row in source_tasks],
        sum(len(row["trajectories"]) for row in members) == 141,
        sum(len(row["trajectories"]) == 1 for row in members) == 1,
        all(len(row["trajectories"]) in {1, 2} for row in members),
    )
    if not all(valid_members):
        raise ValueError("MDCO source response shards are incomplete")
    return members

def _source_reliability(
    capture_contract: Mapping[str, Any], *, asset_root: Path
) -> dict[int, float]:
    paths = tuple(
        _resolve_reference(asset_root, value)
        for value in capture_contract["diagnostic_occupancy_capture"]["direct_results"]
    )
    rows = [row for path in paths for row in read_json(path).get("rows", ())]
    grouped: dict[int, list[bool]] = {}
    for row in rows:
        grouped.setdefault(int(row["task_id"]), []).append(bool(row["success"]))
    if len(grouped) != 71 or any(len(values) != 50 for values in grouped.values()):
        raise ValueError("MDCO source expert reliability panel changed")
    result = {task_id: sum(values) / 50.0 for task_id, values in grouped.items()}
    if min(result.values()) <= 0:
        raise ValueError("MDCO selected source mapping has zero reliability")
    return result

def _load_target_members(
    *, analysis_path: Path, selection_path: Path, asset_root: Path
) -> tuple[list[dict[str, Any]], dict[int, Path]]:
    analysis = read_json(analysis_path)
    selection = read_json(selection_path)
    selected = [dict(row) for row in selection.get("rows", ())]
    if not all(
        (
            analysis.get("schema_version") == TARGET_ANALYSIS_SCHEMA,
            selection.get("schema_version") == TARGET_SELECTION_SCHEMA,
            len(selected) == 47,
        )
    ):
        raise ValueError("MDCO target successful-policy authority changed")
    raw = []
    for value in analysis["shards"]:
        shard = torch.load(
            _resolve_reference(asset_root, value),
            map_location="cpu",
            weights_only=False,
        )
        raw.extend(dict(row) for row in shard["members"])
    key = lambda row: (
        int(row["global_task_id"]),
        str(row["member"]),
        int(row["expert_step"]),
        int(row["init_state_id"]),
    )
    raw_by_key = {key(row): row for row in raw}
    if not all(
        (
            len(raw_by_key) == 47,
            set(raw_by_key) == {key(row) for row in selected},
        )
    ):
        raise ValueError("MDCO target action responses are incomplete")
    panel_roots = {
        int(step): _resolve_reference(asset_root, value)
        for step, value in analysis["panels"].items()
    }
    panel_rows: dict[int, dict[tuple[str, int, int], dict[str, Any]]] = {}
    panel_adapters: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    for step, root in panel_roots.items():
        results = read_json(root / "results.json")
        contract = read_json(root / "run_contract.json")
        panel_rows[step] = {
            (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): row
            for row in results["rows"]
        }
        panel_adapters[step] = {
            (str(row["suite"]), int(row["task_id"])): row
            for row in contract["adapter"]["tasks"]
        }
    result = []
    for row in selected:
        step = int(row["expert_step"])
        raw_row = raw_by_key[key(row)]
        task_key = (str(row["suite"]), int(row["task_id"]))
        result_key = (*task_key, int(row["init_state_id"]))
        result_row = panel_rows[step].get(result_key)
        adapter = panel_adapters[step].get(task_key)
        if result_row is None or adapter is None:
            raise ValueError("MDCO target successful member changed")
        trajectory_path = Path(
            str(result_row.get("occupancy_trajectory", {}).get("path", ""))
        )
        trajectory_bytes = (
            trajectory_path.stat().st_size if trajectory_path.is_file() else -1
        )
        valid_member = (
            bool(result_row.get("success")),
            Path(str(adapter["checkpoint"]))
            == Path(str(result_row["task_expert"]["checkpoint"])),
            int(adapter.get("global_task_id", -1)) == int(row["global_task_id"]),
            str(adapter.get("language")) == str(row["language"]),
            raw_row["action_delta"].ndim == 3,
            tuple(raw_row["action_delta"].shape)[1:] == (50, 7),
            bool(torch.isfinite(raw_row["action_delta"]).all()),
            trajectory_path.is_file(),
            trajectory_bytes
            == int(result_row["occupancy_trajectory"].get("bytes", -1)),
        )
        if not all(valid_member):
            raise ValueError("MDCO target successful member changed")
        occupancy = result_row["occupancy_trajectory"]
        result.append(
            row
            | {
                "action_delta": raw_row["action_delta"].float().contiguous(),
                "checkpoint": _repo_reference(Path(str(adapter["checkpoint"]))),
                "trajectory": {
                    "init_state_id": int(row["init_state_id"]),
                    "path": _repo_reference(Path(str(occupancy["path"]))),
                    "bytes": int(occupancy["bytes"]),
                    "replans": int(occupancy["replans"]),
                    "steps": int(result_row["steps"]),
                },
            }
        )
    return result, panel_roots

def _target_task_order(
    selection: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    by_global: dict[int, dict[str, Any]] = {}
    for row in selection:
        task_id = int(row["global_task_id"])
        previous = by_global.setdefault(task_id, dict(row))
        if (
            previous["fold_role"] != row["fold_role"]
            or previous["ordinal"] != row["ordinal"]
        ):
            raise ValueError("MDCO target fold ownership changed")
    fit = sorted(
        (row for row in by_global.values() if row["fold_role"] == "fit"),
        key=lambda row: int(row["ordinal"]),
    )
    held = sorted(
        (
            row
            for row in by_global.values()
            if row["fold_role"] == "held_transform_only"
        ),
        key=lambda row: int(row["ordinal"]),
    )
    if len(fit) != 19 or len(held) != 5:
        raise ValueError("MDCO target fold differs from fit19/held5")
    return tuple(fit + held)

def _fit_mapping_whitener(
    source_members: Sequence[Mapping[str, Any]],
    target_members: Sequence[Mapping[str, Any]],
    target_ordinal: Mapping[int, int],
) -> tuple[FunctionalWhitener, int]:
    sequences: list[torch.Tensor] = []
    ordinals: list[int] = []
    for member in source_members:
        for trajectory in member["trajectories"]:
            sequences.append(trajectory["action_delta"])
            ordinals.append(int(member["ordinal"]))
    for member in target_members:
        if member["fold_role"] == "fit":
            sequences.append(member["action_delta"])
            ordinals.append(target_ordinal[int(member["global_task_id"])])
    return fit_task_equal_whitener(sequences, ordinals, width=32), len(sequences)

def _mapping_tasks(
    *,
    source_tasks: Sequence[Mapping[str, Any]],
    target_order: Sequence[Mapping[str, Any]],
    target_by_global: Mapping[int, Mapping[str, Any]],
    target_ordinal: Mapping[int, int],
) -> list[dict[str, Any]]:
    tasks = [
        {
            "ordinal": ordinal,
            "global_task_id": int(row["task_index"]),
            "suite": "libero_90",
            "task_id": int(row["task_index"]),
            "language": str(row["language"]),
            "hdf5_relative_path": f"libero_90/{row['hdf5']['filename']}",
            "hdf5_bytes": int(row["hdf5"]["bytes"]),
            "episode_lengths": [
                int(value) for value in row["demonstrations"]["episode_lengths"]
            ],
            "fold_role": "fit",
            "asset_key": f"source90:{int(row['task_index'])}",
            "domain": "libero90_nonheld",
        }
        for ordinal, row in enumerate(source_tasks)
    ]
    for row in target_order:
        global_id = int(row["global_task_id"])
        task = target_by_global[global_id]
        tasks.append(
            {
                "ordinal": target_ordinal[global_id],
                "global_task_id": global_id,
                "suite": str(task["suite"]),
                "task_id": int(task["task_id"]),
                "language": str(task["language"]),
                "hdf5_relative_path": str(task["hdf5"]["relative_path"]),
                "hdf5_bytes": int(task["hdf5"]["bytes"]),
                "episode_lengths": [
                    int(value) for value in task["demonstrations"]["episode_lengths"]
                ],
                "fold_role": str(row["fold_role"]),
                "asset_key": f"target40:{global_id}",
                "domain": "target_train",
            }
        )
    if [int(row["ordinal"]) for row in tasks] != list(range(95)):
        raise ValueError("MDCO task namespace is not contiguous")
    return tasks


def _source_authority_members(
    *,
    sources: Sequence[Mapping[str, Any]],
    reliability: Mapping[int, float],
    whitener: FunctionalWhitener,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    members, phase_rows = [], []
    for source in sources:
        trajectories, embeddings = [], []
        for trajectory in source["trajectories"]:
            coordinates = whitener.transform(trajectory["action_delta"])
            indices = arc_length_phase_indices(coordinates, count=8)
            embeddings.append(arc_length_phase_embedding(coordinates, count=8).float())
            trajectories.append(
                {
                    key: trajectory[key]
                    for key in ("init_state_id", "path", "bytes", "replans", "steps")
                }
                | {"selected_replan_indices": [int(value) for value in indices]}
            )
        phase_rows.append(torch.stack(embeddings).mean(dim=0))
        task_id = int(source["task_id"])
        members.append(
            {
                "index": len(members),
                "ordinal": int(source["ordinal"]),
                "global_task_id": task_id,
                "suite": "libero_90",
                "task_id": task_id,
                "member": "step1000_pooled2",
                "expert_step": 1000,
                "init_state_id": int(trajectories[0]["init_state_id"]),
                "fold_role": "fit",
                "reliability": reliability[task_id],
                "checkpoint": str(source["checkpoint"]),
                "trajectories": trajectories,
                "asset_key": f"source90:{task_id}",
            }
        )
    return members, phase_rows


def _target_authority_members(
    *,
    targets: Sequence[Mapping[str, Any]],
    target_ordinal: Mapping[int, int],
    whitener: FunctionalWhitener,
    first_index: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    ordered = sorted(
        targets,
        key=lambda row: (
            target_ordinal[int(row["global_task_id"])],
            int(row["expert_step"]),
            str(row["member"]),
        ),
    )
    members, phase_rows = [], []
    for target in ordered:
        global_id = int(target["global_task_id"])
        coordinates = whitener.transform(target["action_delta"])
        indices = arc_length_phase_indices(coordinates, count=8)
        phase_rows.append(arc_length_phase_embedding(coordinates, count=8).float())
        reliability = (
            int(target["checkpoint_successes"][str(target["expert_step"])]) / 50.0
        )
        members.append(
            {
                "index": first_index + len(members),
                "ordinal": target_ordinal[global_id],
                "global_task_id": global_id,
                "suite": str(target["suite"]),
                "task_id": int(target["task_id"]),
                "member": str(target["member"]),
                "expert_step": int(target["expert_step"]),
                "init_state_id": int(target["init_state_id"]),
                "fold_role": str(target["fold_role"]),
                "reliability": reliability,
                "checkpoint": str(target["checkpoint"]),
                "trajectories": [
                    target["trajectory"]
                    | {"selected_replan_indices": [int(value) for value in indices]}
                ],
                "asset_key": f"target40:{global_id}",
            }
        )
    return members, phase_rows


def _publish_mapping_authority(
    *,
    args: Any,
    repository: Mapping[str, Any],
    target_panels: Mapping[int, Path],
    tasks: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
    phase_response: torch.Tensor,
    whitener: FunctionalWhitener,
    fit_sequence_count: int,
    dropped_source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    transform_path = args.output_dir / "phase_transform.pt"
    phase_path = args.output_dir / "member_phase_response.safetensors"
    if manifest_path.exists() or transform_path.exists() or phase_path.exists():
        raise ValueError("MDCO mapping-diverse authority already exists")
    _save_torch_atomic(
        transform_path,
        {
            "schema_version": PHASE_TRANSFORM_SCHEMA,
            "repository": repository,
            "response": "expert_minus_source_full_50x7_action_chunk_at_every_replan",
            "fit_role": "source71_plus_target_fit19_task_equal_sequence_equal_state_equal",
            "held_role": "target_fold0_transform_only",
            "width": 32,
            **whitener.state_dict(),
        },
    )
    _save_safetensors_atomic(phase_path, {"member_phase_response": phase_response})
    manifest = {
        "schema_version": AUTHORITY_SCHEMA,
        "status": "complete_mapping_diverse_authority",
        "repository": dict(repository),
        "sources": {
            "source_manifest": _repo_reference(args.source_manifest),
            "meta_protocol": _repo_reference(args.meta_protocol),
            "source_capture_run": _repo_reference(args.capture_run),
            "source_response_shards": [
                _repo_reference(path) for path in args.source_shard
            ],
            "target_manifest": _repo_reference(args.target_manifest),
            "target_selection": _repo_reference(args.target_selection),
            "target_analysis": _repo_reference(args.target_analysis),
            "target_panels": {
                str(step): _repo_reference(path)
                for step, path in sorted(target_panels.items())
            },
        },
        "validity": {
            "fit_tasks": 90,
            "held_transform_only_tasks": 5,
            "successful_policy_members": 118,
            "source_successful_trajectories": 141,
            "source_capture_dropped_unsuccessful_rows": [
                {
                    "suite": str(row["suite"]),
                    "task_id": int(row["task_id"]),
                    "init_state_id": int(row["init_state_id"]),
                    "steps": int(row["steps"]),
                }
                for row in dropped_source_rows
            ],
            "target_successful_trajectories": 47,
            "source71_training_gradient_use": True,
            "target_fit19_training_gradient_use": True,
            "target_held5_transform_only": True,
            "held5_shared_gradient_use": False,
            "validation_data_use": False,
            "test_data_use": False,
            "task_id_model_route": False,
        },
        "phase_transform": {
            "schema_version": PHASE_TRANSFORM_SCHEMA,
            "path": _repo_reference(transform_path),
            "input_width": 350,
            "output_width": 32,
            "fit_tasks": 90,
            "fit_sequences": fit_sequence_count,
            "explained_variance_ratio": whitener.explained_variance_ratio,
        },
        "phase_response": {
            "path": _repo_reference(phase_path),
            "tensor": "member_phase_response",
            "shape": [118, 8, 32],
        },
        "tasks": list(tasks),
        "members": list(members),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def assemble_mapping_diverse_authority(args: Any) -> dict[str, Any]:
    """Fit one fit90 coordinate system and publish all 95 mapping authorities."""

    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("MDCO authority assembly requires a clean pushed authority")
    source_manifest = read_json(args.source_manifest)
    source_tasks = _source_task_rows(source_manifest)
    meta_protocol = read_json(args.meta_protocol)
    if tuple(int(value) for value in meta_protocol["active_source_task_ids"]) != tuple(
        int(row["task_index"]) for row in source_tasks
    ):
        raise ValueError("MDCO meta protocol and source corpus differ")
    capture_contract, capture_results, _, _ = _load_source_capture(
        args.capture_run, source_manifest
    )
    source_members = _load_source_shards(args.source_shard, source_tasks=source_tasks)
    source_reliability = _source_reliability(
        capture_contract, asset_root=args.asset_root
    )

    target_selection = read_json(args.target_selection)
    target_order = _target_task_order(target_selection["rows"])
    target_members, target_panels = _load_target_members(
        analysis_path=args.target_analysis,
        selection_path=args.target_selection,
        asset_root=args.asset_root,
    )
    target_manifest = read_json(args.target_manifest)
    target_by_global = {
        int(row["global_task_id"]): dict(row) for row in target_manifest["tasks"]
    }
    target_ordinal = {
        int(row["global_task_id"]): 71 + index for index, row in enumerate(target_order)
    }
    if target_manifest.get(
        "schema_version"
    ) != "ember_pi05_target_data_manifest_v1" or any(
        target_by_global[int(row["global_task_id"])]["split_role"] != "train"
        or target_by_global[int(row["global_task_id"])]["suite"] != row["suite"]
        or int(target_by_global[int(row["global_task_id"])]["task_id"])
        != int(row["task_id"])
        or target_by_global[int(row["global_task_id"])]["language"] != row["language"]
        for row in target_order
    ):
        raise ValueError("MDCO target data authority changed")

    whitener, fit_sequence_count = _fit_mapping_whitener(
        source_members, target_members, target_ordinal
    )
    tasks = _mapping_tasks(
        source_tasks=source_tasks,
        target_order=target_order,
        target_by_global=target_by_global,
        target_ordinal=target_ordinal,
    )
    members, phase_rows = _source_authority_members(
        sources=source_members,
        reliability=source_reliability,
        whitener=whitener,
    )
    target_rows, target_phase = _target_authority_members(
        targets=target_members,
        target_ordinal=target_ordinal,
        whitener=whitener,
        first_index=len(members),
    )
    members.extend(target_rows)
    phase_rows.extend(target_phase)
    phase_response = torch.stack(phase_rows).float().contiguous()
    if (
        len(members) != 118
        or phase_response.shape != (118, 8, 32)
        or {int(row["ordinal"]) for row in members} != set(range(95))
    ):
        raise ValueError("MDCO member authority is incomplete")

    manifest = _publish_mapping_authority(
        args=args,
        repository=repository,
        target_panels=target_panels,
        tasks=tasks,
        members=members,
        phase_response=phase_response,
        whitener=whitener,
        fit_sequence_count=fit_sequence_count,
        dropped_source_rows=[
            row for row in capture_results["rows"] if not bool(row["success"])
        ],
    )
    print(
        {
            "status": manifest["status"],
            "tasks": len(tasks),
            "members": len(members),
            "fit_sequences": fit_sequence_count,
            "explained_variance_ratio": whitener.explained_variance_ratio,
            "manifest": str(args.output_dir / "manifest.json"),
        },
        flush=True,
    )
    return manifest
