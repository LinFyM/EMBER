"""Materialize and publish the first exact-effect EMBER-PECS oracle."""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import build_target_owners
from ember.ecp.effect_solver import (
    build_exact_policy_effect_targets,
    capture_policy_effect_response,
    evaluate_policy_effect_state,
    prepare_policy_effect_probe,
    relative_effective_update_distance,
    solve_policy_effects,
)
from ember.ecp.observer_authority import (
    FrozenObserverAuthority,
    load_frozen_observer_authority,
)
from ember.ecp.stage0_training import load_stage0_config, stage0_source_authority
from ember.ecp.stage1_data import (
    ECPStage1EvidenceBank,
    ECPStage1Task,
    build_stage1_video_store,
    load_stage1_evidence_bank,
    load_stage1_tasks,
    pack_stage1_videos,
    tokenize_stage1_languages,
)
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_config, load_policy
from ember.writer.functional import prepare_frozen_writer_policy


CONFIG_SCHEMA = "ember_ecp_policy_effect_solver_oracle_v1"
PROJECTION_SCHEMA = "ember_ecp_policy_effect_solver_oracle_projection_v1"
PROJECTION_KIND = "ecp_policy_effect_solver_exact_oracle"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_authority(config: Mapping[str, Any], name: str) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else REPO_ROOT / path


def _asset_authority(
    config: Mapping[str, Any], name: str, asset_root: Path
) -> Path:
    path = Path(str(config["authorities"][name]))
    return path if path.is_absolute() else asset_root / path


@dataclass(frozen=True)
class EffectOracleAuthorities:
    policy: torch.nn.Module
    source_config: Mapping[str, Any]
    contract: LoRAContract
    identity_state: Mapping[str, torch.Tensor]
    shared_state: Mapping[str, torch.Tensor]
    observer: FrozenObserverAuthority


@dataclass
class PreparedEffectTask:
    authorities: EffectOracleAuthorities
    task: ECPStage1Task
    evidence: ECPStage1EvidenceBank
    store: Any
    lora: BatchedLoRAInference
    packed: Any
    probe: Any
    targets: Any
    suffix_noise: torch.Tensor
    expert_states: tuple[dict[str, torch.Tensor], ...]
    expert_weights: torch.Tensor

    def response(self, state: Mapping[str, torch.Tensor], event: int):
        return capture_policy_effect_response(
            policy=self.authorities.policy,
            observer=self.authorities.observer.model.encoder.observer,
            lora=self.lora,
            state=state,
            prefix_embeddings=self.probe.prefix_embeddings[event],
            prefix_padding=self.probe.prefix_padding[event],
            suffix_noise=self.suffix_noise,
        )

    def close(self) -> None:
        self.lora.close()
        self.store.close()


def load_effect_oracle_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    roles = config.get("roles", {})
    data = config.get("data", {})
    solver = config.get("solver", {})
    wall = config.get("information_wall", {})
    valid = all(
        (
            config.get("schema_version") == CONFIG_SCHEMA,
            config.get("status") == "active_exact_effect_realizability_oracle",
            tuple(roles.get("held_task_ordinals", ())) == tuple(range(90, 95)),
            tuple(roles.get("held_target_global_task_ids", ())) == (0, 9, 18, 25, 36),
            int(data.get("frame_stride", -1)) == 5,
            int(data.get("visible_videos", -1)) == 2,
            int(data.get("event_slots", -1)) == 8,
            int(data.get("horizon_basis", -1)) == 4,
            int(solver.get("output_rank", -1)) == 16,
            int(solver.get("steps", -1)) > 0,
            float(solver.get("step_rms", -1)) > 0,
            float(solver.get("step_decay_power", -1)) >= 0,
            solver.get("per_task_early_stop") is False,
            solver.get("task_local_persistent_optimizer") is False,
            wall.get("teacher_action_forward_reads") == 0,
            wall.get("held_shared_parameter_updates") == 0,
            wall.get("validation_action_or_reward_reads") == 0,
            wall.get("test_action_or_reward_reads") == 0,
            wall.get("task_id_model_route") is False,
            wall.get("action_meta_effect_solver_installation") is False,
            wall.get("second_adapter_deployed") is False,
        )
    )
    if not valid:
        raise ValueError("unsupported PECS exact-effect oracle contract")
    return config


def _autocast(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _load_authorities(
    args: argparse.Namespace, config: Mapping[str, Any], device: torch.device
) -> EffectOracleAuthorities:
    source = stage0_source_authority(args)
    source_config = load_config(_repo_authority(config, "source_base_config"))
    policy = load_policy(Path(source["model_path"]), source_config, device)
    contract = load_pi05_lora_contract(_repo_authority(config, "lora_contract"))
    identity = prepare_frozen_writer_policy(policy, contract)
    stage0_config = load_stage0_config(_repo_authority(config, "stage0_config"))
    observer = load_frozen_observer_authority(
        stage0_config=stage0_config,
        owners=build_target_owners(contract),
        policy=policy,
        native_checkpoint=_asset_authority(
            config, "native_observer_checkpoint", args.asset_root
        ),
        action_meta_checkpoint=_asset_authority(
            config, "action_meta_checkpoint", args.asset_root
        ),
        device=device,
        max_frames_per_call=args.max_frames_per_call,
    )
    shared = load_file(
        str(_asset_authority(config, "stable_shared_prior", args.asset_root)),
        device=str(device),
    )
    validate_lora_state(shared, contract)
    return EffectOracleAuthorities(
        policy=policy,
        source_config=source_config,
        contract=contract,
        identity_state=identity,
        shared_state=shared,
        observer=observer,
    )


def _task_inputs(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    authorities: EffectOracleAuthorities,
    device: torch.device,
    ordinal: int,
) -> tuple[ECPStage1Task, ECPStage1EvidenceBank, Any, tuple[torch.Tensor, torch.Tensor]]:
    manifest = _asset_authority(
        config, "task_evidence_manifest", args.asset_root
    )
    tasks = load_stage1_tasks(authority_manifest=manifest, data_root=args.data_root)
    task = tasks[ordinal]
    evidence = load_stage1_evidence_bank(
        authority_manifest=manifest,
        asset_root=args.asset_root,
        contract=authorities.contract,
        device=device,
    )
    languages = tokenize_stage1_languages(
        (task,),
        tokenizer_path=args.tokenizer_path,
        max_length=int(authorities.source_config["features"]["tokenizer_max_length"]),
        device=device,
    )
    return (
        task,
        evidence,
        build_stage1_video_store((task,), frame_stride=int(config["data"]["frame_stride"])),
        languages[ordinal],
    )


def _member_states(
    evidence: ECPStage1EvidenceBank, ordinal: int
) -> tuple[tuple[dict[str, torch.Tensor], ...], torch.Tensor]:
    indices = evidence.member_indices(ordinal)
    states = tuple(
        {name: value[index] for name, value in evidence.member_states.items()}
        for index in indices
    )
    selected = torch.tensor(indices, device=evidence.reliability.device)
    return states, evidence.reliability.index_select(0, selected)


def _file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size}


def _prepare_effect_task(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    device: torch.device,
) -> PreparedEffectTask:
    authorities = _load_authorities(args, config, device)
    task, evidence, store, language = _task_inputs(
        args=args,
        config=config,
        authorities=authorities,
        device=device,
        ordinal=args.task_ordinal,
    )
    lora = BatchedLoRAInference(authorities.policy, authorities.contract)
    packed = pack_stage1_videos(
        store=store,
        ordinal=task.ordinal,
        visit=int(config["data"]["video_visit"]),
        seed=int(config["data"]["pair_seed"]),
        k=int(config["data"]["visible_videos"]),
        device=device,
    )
    tokens, mask = language
    expert_model = authorities.policy.model.paligemma_with_expert.gemma_expert.model
    with (
        torch.no_grad(),
        authorities.observer.action_meta.installed(expert_model),
        _autocast(device),
    ):
        encoded = authorities.observer.model.encoder(
            policy=authorities.policy,
            frames=packed.frames,
            video_offsets=packed.video_offsets,
            frame_condition_ids=packed.frame_condition_ids,
            language_tokens=tokens,
            language_mask=mask,
        )
    probe = prepare_policy_effect_probe(
        encoder=authorities.observer.model.encoder,
        policy=authorities.policy,
        packed=packed,
        encoded=encoded,
        language_tokens=tokens,
        language_mask=mask,
        prefix_batch_size=int(config["solver"]["prefix_batch_size"]),
    )
    expert_states, expert_weights = _member_states(evidence, task.ordinal)
    suffix_noise = authorities.observer.model.encoder.fixed_suffix_noise.detach()
    targets = build_exact_policy_effect_targets(
        policy=authorities.policy,
        observer=authorities.observer.model.encoder.observer,
        lora=lora,
        identity_state=authorities.identity_state,
        shared_state=authorities.shared_state,
        expert_states=expert_states,
        expert_weights=expert_weights,
        probe=probe,
        suffix_noise=suffix_noise,
    )
    return PreparedEffectTask(
        authorities=authorities,
        task=task,
        evidence=evidence,
        store=store,
        lora=lora,
        packed=packed,
        probe=probe,
        targets=targets,
        suffix_noise=suffix_noise,
        expert_states=expert_states,
        expert_weights=expert_weights,
    )


def _run_fixed_solver(prepared: PreparedEffectTask, solver: Mapping[str, Any]):
    candidate, history = solve_policy_effects(
        initial_state=prepared.authorities.shared_state,
        targets=prepared.targets,
        contract=prepared.authorities.contract,
        response=prepared.response,
        steps=int(solver["steps"]),
        step_rms=float(solver["step_rms"]),
        step_decay_power=float(solver["step_decay_power"]),
        owner_weight=float(solver["owner_weight"]),
        flow_weight=float(solver["flow_weight"]),
        shared_barrier_weight=float(solver["shared_barrier_weight"]),
        trust_region=float(solver["trust_region"]),
        trust_weight=float(solver["trust_weight"]),
    )
    final = evaluate_policy_effect_state(
        state=candidate,
        targets=prepared.targets,
        response=prepared.response,
        owner_weight=float(solver["owner_weight"]),
        flow_weight=float(solver["flow_weight"]),
        shared_barrier_weight=float(solver["shared_barrier_weight"]),
    )
    validate_lora_state(candidate, prepared.authorities.contract)
    return candidate, history, final


def _publish_task_result(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
    prepared: PreparedEffectTask,
    candidate: Mapping[str, torch.Tensor],
    history: Any,
    final: Mapping[str, float],
    started: float,
    device: torch.device,
) -> dict[str, Any]:
    task = prepared.task
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_dir / (
        f"task_{task.ordinal:03d}_global_{task.global_task_id:02d}.safetensors"
    )
    result_path = args.output_dir / f"task_{task.ordinal:03d}.json"
    if adapter_path.exists() or result_path.exists():
        raise ValueError("PECS task output already exists")
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in candidate.items()},
        str(adapter_path),
    )
    initial = history[0].effect
    sequence = [row.effect for row in history] + [final["effect"]]
    increasing = sum(right > left for left, right in zip(sequence, sequence[1:]))
    ratio = final["effect"] / max(initial, 1e-12)
    gate = config["profile_gate"]
    result = {
        "schema_version": "ember_ecp_policy_effect_solver_task_v1",
        "mode": args.mode,
        "repository": repository,
        "task": {
            "ordinal": task.ordinal,
            "global_task_id": task.global_task_id,
            "suite": task.suite,
            "task_id": task.task_id,
            "language": task.language,
            "asset_key": task.asset_key,
            "fold_role": task.fold_role,
        },
        "video": {
            "demo_indices": list(prepared.packed.demo_indices),
            "selected_flat_indices": prepared.probe.selected_flat_indices.tolist(),
            "selected_frame_positions": prepared.probe.selected_frame_positions.tolist(),
            "event_presence": prepared.probe.presence.tolist(),
        },
        "teacher": {
            "successful_member_indices": list(
                prepared.evidence.member_indices(task.ordinal)
            ),
            "successful_member_weights": prepared.expert_weights.cpu().tolist(),
            "teacher_action_forward_reads": 0,
            "action_meta_installed_for_effect_capture": False,
        },
        "solver": dict(config["solver"]),
        "history": [asdict(row) for row in history],
        "final": dict(final),
        "initial_to_final_effect_ratio": ratio,
        "increasing_steps": increasing,
        "profile_gate_pass": (
            ratio <= float(gate["maximum_final_to_initial_effect_ratio"])
            and increasing <= int(gate["maximum_increasing_steps"])
        ),
        "effective_distance": {
            "from_shared": float(
                relative_effective_update_distance(
                    candidate,
                    prepared.authorities.shared_state,
                    prepared.authorities.contract,
                )
            ),
            "to_expert_members": [
                float(
                    relative_effective_update_distance(
                        candidate, state, prepared.authorities.contract
                    )
                )
                for state in prepared.expert_states
            ],
        },
        "adapter": _file(adapter_path),
        "elapsed_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device.index or 0)
            if device.type == "cuda"
            else 0
        ),
    }
    write_json_atomic(result_path, result)
    print(result, flush=True)
    return result


def solve_task(args: argparse.Namespace) -> dict[str, Any]:
    config = load_effect_oracle_config(args.config)
    repository = git_state(REPO_ROOT)
    if args.mode == "formal" and not git_state_is_clean_pushed_or_frozen_authority(
        repository
    ):
        raise ValueError("formal PECS oracle requires a clean pushed authority")
    role = config["roles"]
    allowed = (
        {int(role["profile_fit_task_ordinal"])}
        if args.mode == "profile"
        else {int(value) for value in role["held_task_ordinals"]}
    )
    if args.task_ordinal not in allowed:
        raise ValueError("PECS task is outside the selected oracle role")
    device = torch.device(args.device)
    torch.manual_seed(int(config["data"]["pair_seed"]))
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.manual_seed_all(int(config["data"]["pair_seed"]))
        torch.cuda.reset_peak_memory_stats(device.index or 0)
    started = time.monotonic()
    prepared = _prepare_effect_task(args=args, config=config, device=device)
    try:
        candidate, history, final = _run_fixed_solver(
            prepared, config["solver"]
        )
        return _publish_task_result(
            args=args,
            config=config,
            repository=repository,
            prepared=prepared,
            candidate=candidate,
            history=history,
            final=final,
            started=started,
            device=device,
        )
    finally:
        prepared.close()


def _base_rows(
    path: Path, *, expert_bank_root: Path, expert_step: int
) -> dict[int, dict[str, Any]]:
    manifest = read_json(path)
    rows = {int(row["global_task_id"]): dict(row) for row in manifest.get("tasks", ())}
    if (
        manifest.get("schema_version")
        != "ember_phase_aligned_functional_decoder_train24_projection_v1"
        or manifest.get("projection_kind") != "stable_shared_prior_baseline"
        or Path(str(manifest.get("expert_bank_root", ""))).resolve()
        != expert_bank_root.resolve()
        or int(manifest.get("expert_step", -1)) != expert_step
        or len(rows) != 24
    ):
        raise ValueError("PECS base evaluation surface changed")
    return rows


def assemble_projection(args: argparse.Namespace) -> dict[str, Any]:
    config = load_effect_oracle_config(args.config)
    held = tuple(int(value) for value in config["roles"]["held_task_ordinals"])
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal PECS projection requires a clean pushed authority")
    base_manifest = _asset_authority(
        config, "base_projection_manifest", args.asset_root
    )
    base = _base_rows(
        base_manifest,
        expert_bank_root=args.expert_bank_root,
        expert_step=args.expert_step,
    )
    rows = []
    for ordinal in held:
        result = read_json(args.output_dir / f"task_{ordinal:03d}.json")
        task = result.get("task", {})
        global_task_id = int(task.get("global_task_id", -1))
        source = base.get(global_task_id)
        adapter = Path(str(result.get("adapter", {}).get("path", ""))).resolve()
        if (
            result.get("schema_version") != "ember_ecp_policy_effect_solver_task_v1"
            or result.get("mode") != "formal"
            or result.get("repository", {}).get("commit") != repository["commit"]
            or int(task.get("ordinal", -1)) != ordinal
            or source is None
            or not adapter.is_file()
            or adapter.stat().st_size != int(result["adapter"]["bytes"])
        ):
            raise ValueError("PECS task materialization changed")
        rows.append(
            {
                "suite": task["suite"],
                "task_id": int(task["task_id"]),
                "ordinal": int(source["ordinal"]),
                "stage1_ordinal": ordinal,
                "asset_key": task["asset_key"],
                "domain": "target_train",
                "global_task_id": global_task_id,
                "expert_checkpoint": source["expert_checkpoint"],
                "fold_role": task["fold_role"],
                "projected_adapter": str(adapter),
                "projected_adapter_bytes": adapter.stat().st_size,
                "video_demo_indices": result["video"]["demo_indices"],
                "successful_member_count": len(
                    result["teacher"]["successful_member_indices"]
                ),
                "initial_to_final_effect_ratio": result[
                    "initial_to_final_effect_ratio"
                ],
                "final_effect": result["final"]["effect"],
                "effective_distance_from_shared": result["effective_distance"][
                    "from_shared"
                ],
            }
        )
    solver = config["solver"]
    manifest = {
        "schema_version": PROJECTION_SCHEMA,
        "projection_kind": PROJECTION_KIND,
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "effect_oracle_config": _file(args.config),
        "base_projection_manifest": _file(base_manifest),
        "expert_bank_root": str(args.expert_bank_root.resolve()),
        "expert_step": int(args.expert_step),
        "optimization": {
            "fold": int(config["roles"]["fold"]),
            "fit_profile_task_count": 1,
            "held_task_count": 5,
            "held_shared_gradient_steps": 0,
            "solver_algorithm_frozen": True,
            "solver_steps": int(solver["steps"]),
            "per_task_early_stop": False,
            "task_local_persistent_optimizer": False,
            "single_complete_lora": True,
            "final_lora_averaging": False,
            "rank": 16,
            "second_adapter_deployed": False,
            "parameterization": "one complete rank16 LoRA solved from exact policy effects",
        },
        "information_wall": {
            "role": "development_train_leave_task_out_oracle_only",
            "deployment_carrier": False,
            "exact_privileged_effects": True,
            "teacher_action_forward_reads": 0,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "second_adapter_deployed": False,
        },
        "tasks": rows,
        "content_hash_policy": "disabled_by_owner",
    }
    path = args.output_dir / "projection_manifest.json"
    if path.exists():
        raise ValueError("PECS projection manifest already exists")
    write_json_atomic(path, manifest)
    print(manifest, flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve")
    solve.add_argument("--mode", choices=("profile", "formal"), required=True)
    solve.add_argument("--task-ordinal", type=int, required=True)
    solve.add_argument("--asset-root", type=Path, required=True)
    solve.add_argument("--source-run", type=Path, required=True)
    solve.add_argument("--checkpoint", type=Path, required=True)
    solve.add_argument("--tokenizer-path", type=Path, required=True)
    solve.add_argument("--data-root", type=Path, required=True)
    solve.add_argument("--output-dir", type=Path, required=True)
    solve.add_argument("--device", default="cuda:0")
    solve.add_argument("--max-frames-per-call", type=int)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--asset-root", type=Path, required=True)
    assemble.add_argument("--expert-bank-root", type=Path, required=True)
    assemble.add_argument("--expert-step", type=int, default=2000)
    assemble.add_argument("--output-dir", type=Path, required=True)
    for command in (solve, assemble):
        command.add_argument(
            "--config",
            type=Path,
            default=REPO_ROOT / "configs/pi05_ecp_policy_effect_solver_oracle.json",
        )
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "config",
        "asset_root",
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "output_dir",
        "expert_bank_root",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def main(args: argparse.Namespace) -> None:
    if args.command == "solve":
        solve_task(args)
    else:
        assemble_projection(args)
