"""Build one held-task occupancy-complete ECP policy-effect bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import build_target_owners
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectResponse,
    capture_policy_effect_response,
    prepare_execution_policy_prefix,
)
from ember.ecp.stage0_training import stage0_source_authority
from ember.ecp.stage1_data import load_stage1_tasks
from ember.ecp.stage1_equivalence import (
    OccupancyAnchors,
    Stage1EffectBank,
    load_initial_occupancy_anchor,
    load_occupancy_anchors,
    save_effect_bank,
    stack_observations,
    stage_progress_at_replans,
)
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval.occupancy_selection import (
    ECP_STAGE1_HELD_KEYS,
    ECP_STAGE1_PROFILE_KEYS,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy


INITIAL_IDS = (0, 7, 14, 21, 28, 35, 42, 49)
RECOVERY_IDS = (1, 26, 2, 27)


@dataclass(frozen=True)
class BankBuildInputs:
    config: Mapping[str, Any]
    task_ordinal: int
    source_support_run: Path
    independent_fixed50_run: Path
    independent_occupancy_run: Path
    candidate_occupancy_run: Path
    independent_adapter_root: Path
    source_run: Path
    checkpoint: Path
    data_root: Path
    output_dir: Path
    device: torch.device


def _authority_path(config: Mapping[str, Any], name: str, asset_root: Path) -> Path:
    path = Path(str(config["authorities"][name]))
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def _result_rows(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    result = read_json(root.resolve() / "results.json")
    rows = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): dict(row)
        for row in result.get("rows", ())
    }
    if len(rows) != len(result.get("rows", ())):
        raise ValueError("ECP occupancy result rows overlap")
    return rows


def _trajectory_result_row(path: Path) -> dict[str, Any]:
    root = path.resolve().parent.parent
    rows = _result_rows(root)
    matches = [
        row
        for row in rows.values()
        if Path(str(row.get("occupancy_trajectory", {}).get("path", ""))).resolve()
        == path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError("ECP successful-member trajectory lost its result row")
    return matches[0]


def _member_adapter(root: Path, global_id: int) -> Path:
    matches = list(
        root.resolve().glob(
            f"worker_*/task_*_global_{global_id:02d}/checkpoints/"
            "step_00002000/adapter.safetensors"
        )
    )
    if len(matches) != 1:
        raise ValueError("ECP independent member adapter is incomplete")
    return matches[0].resolve()


def _response_concat(values: Sequence[PolicyEffectResponse]) -> PolicyEffectResponse:
    return PolicyEffectResponse(
        owner=torch.cat([value.owner for value in values]),
        flow=torch.cat([value.flow for value in values]),
        action=torch.cat([value.action for value in values]),
    )


def _capture_response(
    *,
    policy: torch.nn.Module,
    observer: torch.nn.Module,
    lora: BatchedLoRAInference,
    state: Mapping[str, torch.Tensor],
    prefix: ExecutionPolicyPrefix,
    noise: torch.Tensor,
    microbatch: int,
) -> PolicyEffectResponse:
    rows = []
    for start in range(0, int(noise.shape[0]), microbatch):
        stop = min(start + microbatch, int(noise.shape[0]))
        rows.append(
            capture_policy_effect_response(
                policy=policy,
                observer=observer,
                lora=lora,
                state=state,
                prefix=ExecutionPolicyPrefix(
                    prefix.embeddings[start:stop], prefix.padding[start:stop]
                ),
                suffix_noise=noise[start:stop],
                denoising_steps=10,
            )
        )
    return _response_concat(rows)


def _prepare_prefix(
    policy: torch.nn.Module,
    observations: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    microbatch: int,
) -> ExecutionPolicyPrefix:
    batch = stack_observations(observations)
    embeddings, padding = [], []
    for start in range(0, len(observations), microbatch):
        stop = min(start + microbatch, len(observations))
        row = {
            name: value[start:stop].to(device, non_blocking=True)
            for name, value in batch.items()
        }
        encoded = prepare_execution_policy_prefix(policy, row)
        embeddings.append(encoded.embeddings)
        padding.append(encoded.padding)
    return ExecutionPolicyPrefix(torch.cat(embeddings), torch.cat(padding))


def _new_member_reliability(
    rows: Mapping[tuple[str, int, int], Mapping[str, Any]], suite: str, task_id: int
) -> float:
    task_rows = [row for key, row in rows.items() if key[:2] == (suite, task_id)]
    if len(task_rows) != 50:
        raise ValueError(
            "ECP independent particle did not receive fixed50 qualification"
        )
    return max(sum(bool(row["success"]) for row in task_rows) / 50.0, 0.02)


def _append_trajectory(
    observations: list[Mapping[str, torch.Tensor]],
    noises: list[torch.Tensor],
    progress: list[torch.Tensor],
    anchors: OccupancyAnchors,
) -> None:
    observations.extend(anchors.observations)
    noises.extend(anchors.suffix_noise.unbind())
    progress.extend(anchors.progress.unbind())


def build_effect_bank(inputs: BankBuildInputs, asset_root: Path) -> Path:
    config = inputs.config
    roles = config["roles"]
    held = tuple(int(value) for value in roles["held_task_ordinals"])
    profile = int(roles["profile_fit_task_ordinal"])
    if inputs.task_ordinal == profile:
        expected_key = ECP_STAGE1_PROFILE_KEYS[0]
        scientific_role = "fit_task_numerical_resource_profile"
    elif inputs.task_ordinal in held:
        expected_key = ECP_STAGE1_HELD_KEYS[held.index(inputs.task_ordinal)]
        scientific_role = "held5_privileged_realization_oracle"
    else:
        raise ValueError("ECP formal effect bank is outside profile/held roles")
    task_manifest = _authority_path(config, "task_evidence_manifest", asset_root)
    tasks = load_stage1_tasks(
        authority_manifest=task_manifest, data_root=inputs.data_root
    )
    task = tasks[inputs.task_ordinal]
    if (task.suite, task.task_id, task.global_task_id) != expected_key:
        raise ValueError("ECP profile/held task identity changed")

    manifest = read_json(task_manifest)
    existing = [
        dict(row)
        for row in manifest["members"]
        if int(row["ordinal"]) == inputs.task_ordinal
    ]
    by_name = {str(row["member"]): row for row in existing}
    if set(by_name) != {"earliest", "latest"}:
        raise ValueError("ECP held task lost earliest/latest successful members")

    source_rows = _result_rows(inputs.source_support_run)
    independent_rows = _result_rows(inputs.independent_occupancy_run)
    independent_fixed50 = _result_rows(inputs.independent_fixed50_run)
    candidate_rows = _result_rows(inputs.candidate_occupancy_run)
    key = (task.suite, task.task_id)
    observations: list[Mapping[str, torch.Tensor]] = []
    noises: list[torch.Tensor] = []
    progresses: list[torch.Tensor] = []
    anchor_metadata = []

    for init_id in INITIAL_IDS:
        row = source_rows[(*key, init_id)]
        anchor = load_initial_occupancy_anchor(row)
        observations.append(anchor.observation)
        noises.append(anchor.suffix_noise)
        progresses.append(
            stage_progress_at_replans(row["stage_predicates"], replans=1)[0]
        )
        anchor_metadata.append(
            {
                "category": "initial",
                "generator": "source",
                "init_state_id": init_id,
                "replan": 0,
            }
        )

    member_records = []
    member_specs = []
    for name in ("latest", "earliest"):
        record = by_name[name]
        trajectory = record["trajectories"][0]
        path = Path(str(trajectory["path"]))
        path = path.resolve() if path.is_absolute() else (asset_root / path).resolve()
        row = _trajectory_result_row(path)
        anchors = load_occupancy_anchors(row=row, require_success=True)
        adapter = Path(str(record["checkpoint"]))
        adapter = (
            adapter.resolve()
            if adapter.is_absolute()
            else (asset_root / adapter).resolve()
        )
        member_specs.append((name, anchors, adapter, float(record["reliability"])))

    independent_task_rows = [
        row for row_key, row in independent_rows.items() if row_key[:2] == key
    ]
    if len(independent_task_rows) != 1 or not bool(independent_task_rows[0]["success"]):
        raise ValueError("ECP independent particle occupancy is not one strict success")
    independent_anchors = load_occupancy_anchors(
        row=independent_task_rows[0], require_success=True
    )
    independent_adapter = _member_adapter(
        inputs.independent_adapter_root, task.global_task_id
    )
    independent_reliability = _new_member_reliability(
        independent_fixed50, task.suite, task.task_id
    )
    ordered_members = (
        member_specs[0],
        (
            "independent",
            independent_anchors,
            independent_adapter,
            independent_reliability,
        ),
        member_specs[1],
    )
    for name, anchors, adapter, reliability in ordered_members:
        _append_trajectory(observations, noises, progresses, anchors)
        member_records.append(
            {
                "name": name,
                "adapter": str(adapter),
                "reliability": reliability,
                "selected_replans": list(anchors.selected_replans),
            }
        )
        anchor_metadata.extend(
            {
                "category": "successful",
                "generator": name,
                "stage": stage,
                "replan": replan,
            }
            for stage, replan in enumerate(anchors.selected_replans)
        )

    candidate_row = candidate_rows[(*key, 0)]
    candidate = load_occupancy_anchors(row=candidate_row)
    _append_trajectory(observations, noises, progresses, candidate)
    anchor_metadata.extend(
        {
            "category": "candidate",
            "generator": "pecs_trajectory",
            "stage": stage,
            "replan": replan,
        }
        for stage, replan in enumerate(candidate.selected_replans)
    )

    recovery_row = next(
        (
            source_rows[(*key, init_id)]
            for init_id in RECOVERY_IDS
            if not bool(source_rows[(*key, init_id)]["success"])
        ),
        None,
    )
    if recovery_row is None:
        raise ValueError("ECP source recovery panel has no failed trajectory")
    recovery = load_occupancy_anchors(row=recovery_row, require_success=False)
    _append_trajectory(observations, noises, progresses, recovery)
    anchor_metadata.extend(
        {
            "category": "recovery",
            "generator": "source",
            "stage": stage,
            "replan": replan,
        }
        for stage, replan in enumerate(recovery.selected_replans)
    )
    if len(observations) != 48:
        raise ValueError("ECP Stage 1 anchor panel is incomplete")

    source_authority = stage0_source_authority(inputs)
    source_config = load_config(
        _authority_path(config, "source_base_config", asset_root)
    )
    policy = load_policy(
        Path(source_authority["model_path"]), source_config, inputs.device
    )
    contract: LoRAContract = load_pi05_lora_contract(
        _authority_path(config, "lora_contract", asset_root)
    )
    identity = prepare_frozen_writer_policy(policy, contract)
    carrier = load_file(
        str(_authority_path(config, "stable_carrier", asset_root)),
        device=str(inputs.device),
    )
    validate_lora_state(carrier, contract)
    states = []
    for record in member_records:
        path = Path(record["adapter"])
        if path.is_dir():
            path = path / "adapter.safetensors"
        state = load_file(str(path), device=str(inputs.device))
        validate_lora_state(state, contract)
        states.append(state)
    stage0_config = read_json(_authority_path(config, "stage0_config", asset_root))
    native = load_frozen_native_observer(
        stage0_config=stage0_config,
        owners=build_target_owners(contract),
        native_checkpoint=_authority_path(
            config, "native_observer_checkpoint", asset_root
        ),
        device=inputs.device,
    )
    prefix = _prepare_prefix(
        policy,
        observations,
        device=inputs.device,
        microbatch=int(config["effect_bank"].get("prefix_microbatch_size", 8)),
    )
    noise = torch.stack(noises).to(inputs.device)
    capture_microbatch = int(config["effect_bank"].get("capture_microbatch_size", 4))
    lora = BatchedLoRAInference(policy, contract)
    try:
        source_response = _capture_response(
            policy=policy,
            observer=native.encoder.observer,
            lora=lora,
            state=identity,
            prefix=prefix,
            noise=noise,
            microbatch=capture_microbatch,
        )
        carrier_response = _capture_response(
            policy=policy,
            observer=native.encoder.observer,
            lora=lora,
            state=carrier,
            prefix=prefix,
            noise=noise,
            microbatch=capture_microbatch,
        )
        member_responses = [
            _capture_response(
                policy=policy,
                observer=native.encoder.observer,
                lora=lora,
                state=state,
                prefix=prefix,
                noise=noise,
                microbatch=capture_microbatch,
            )
            for state in states
        ]
    finally:
        lora.close()
    member_response = PolicyEffectResponse(
        owner=torch.stack([value.owner for value in member_responses]),
        flow=torch.stack([value.flow for value in member_responses]),
        action=torch.stack([value.action for value in member_responses]),
    )
    bank = Stage1EffectBank(
        prefix=prefix,
        suffix_noise=noise,
        category_ids=torch.tensor(
            [0] * 8 + [1] * 24 + [2] * 8 + [3] * 8, device=inputs.device
        ),
        stage_ids=torch.tensor(
            [0] * 8 + list(range(8)) * 3 + list(range(8)) * 2, device=inputs.device
        ),
        progress=torch.stack(progresses).to(inputs.device),
        source=source_response,
        carrier=carrier_response,
        members=member_response,
        member_reliability=torch.tensor(
            [row["reliability"] for row in member_records], device=inputs.device
        ),
    )
    return save_effect_bank(
        inputs.output_dir,
        bank,
        {
            "task": {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "language": task.language,
            },
            "scientific_role": scientific_role,
            "members": member_records,
            "anchors": anchor_metadata,
            "action_meta_installed": False,
        },
    )
