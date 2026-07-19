"""Run the staged four-arm source-only task-local LoRA RL recovery."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy
from ember.gate_zero_checkpoint import CHECKPOINT_MANIFEST, validate_source_base_checkpoint
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_oracle_artifacts import (
    atomic_json,
    load_recovery_artifact,
    restore_trainable_state,
    save_candidate_artifact,
    save_recovery_artifact,
    sha256_file,
    validate_candidate_artifact,
    validate_recovery_artifact,
    write_output_checksums,
)
from ember.gate_zero_oracle_metrics import FixedQueryEvaluator
from ember.gate_zero_oracle_report_runtime import _closed_loop_metrics, _task_authority
from ember.gate_zero_oracle_session import (
    OracleModelSession,
    _load_task_datasets,
    build_oracle_optimizer,
    capture_trainable_state,
    configure_oracle_variant,
)
from ember.gate_zero_support.screen_runtime import (
    ParallelContext,
    _broadcast,
    _close_parallel,
    _gather,
    _initialize_parallel,
)
from ember.gate_zero_task_local_rl.contract import (
    assigned_task_local_rl_arm,
    decide_task_local_rl_node,
    load_task_local_rl_spec,
)
from ember.gate_zero_task_local_rl.runtime import (
    GateZeroTaskLocalRLRuntimeError,
    collect_training_round,
    train_signed_flow_ratio_round,
)


RESULT_NAME = "task_local_rl_recovery_result.json"


@dataclass
class LiveRLArm:
    session: OracleModelSession
    runtime: tuple[Any, Any, Any, Any, Any]
    language: str
    task_authority: dict[str, Any]
    initial_state_authority: dict[str, Any]

    def close(self) -> None:
        self.session.close()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroTaskLocalRLRuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise GateZeroTaskLocalRLRuntimeError(f"invalid {label}: {path}")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroTaskLocalRLRuntimeError(f"invalid TOML: {path}") from error


def _validate_result_authorities(
    spec: Mapping[str, Any],
    *,
    headroom_result: Path,
    diagnostic_result: Path,
    previous_awr_result: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority = spec["authority"]
    if sha256_file(headroom_result) != authority["headroom_result_sha256"]:
        raise GateZeroTaskLocalRLRuntimeError("Proposal-A result hash changed")
    if sha256_file(diagnostic_result) != authority["candidate_diagnostic_result_sha256"]:
        raise GateZeroTaskLocalRLRuntimeError("candidate diagnostic result hash changed")
    if sha256_file(previous_awr_result) != authority["previous_awr_result_sha256"]:
        raise GateZeroTaskLocalRLRuntimeError("previous AWR result hash changed")
    headroom = _load_json(headroom_result, "Proposal-A result")
    diagnostic = _load_json(diagnostic_result, "candidate diagnostic result")
    previous_awr = _load_json(previous_awr_result, "previous AWR result")
    counts = {
        (arm["task_id"], arm["condition"]): sum(arm["successes"])
        for arm in headroom.get("arms", [])
    }
    expected = {
        (3, "frozen_base"): 3,
        (4, "frozen_base"): 3,
        (3, authority["fit_variant"]): 2,
        (4, authority["fit_variant"]): 4,
    }
    if (
        headroom.get("status") != "mature_lora_headroom_control_failed_gate_recovery_required"
        or counts != expected
        or headroom.get("gate_zero_authorized") is not False
        or diagnostic.get("status") != "candidate_step_magnitude_recovery_not_supported"
        or diagnostic.get("selected_step") is not None
        or diagnostic.get("gate_zero_authorized") is not False
        or previous_awr.get("status") != "task_local_rl_early_check_not_supported"
        or previous_awr.get("interaction_episodes_per_task_initialization") != 16
        or previous_awr.get("gate_zero_authorized") is not False
        or previous_awr.get("writer_authorized") is not False
    ):
        raise GateZeroTaskLocalRLRuntimeError("upstream Gate-0 failure boundary changed")
    return headroom, diagnostic, previous_awr


def _initial_successes(
    headroom: Mapping[str, Any], *, task_id: int, initialization: str, variant: str
) -> list[bool]:
    condition = "frozen_base" if initialization == "zero_init" else variant
    matches = [
        arm
        for arm in headroom["arms"]
        if arm["task_id"] == task_id and arm["condition"] == condition
    ]
    if len(matches) != 1 or len(matches[0].get("successes", [])) != 8:
        raise GateZeroTaskLocalRLRuntimeError("initial closed-loop vector changed")
    return [bool(value) for value in matches[0]["successes"]]


def _supervised_candidate_path(fit_root: Path, spec: Mapping[str, Any], task_id: int) -> Path:
    return (
        fit_root
        / f"{spec['authority']['fit_variant']}_task{task_id}"
        / "candidates"
        / f"{spec['authority']['supervised_step']:06d}"
    )


def _load_supervised_state(
    path: Path, *, spec: Mapping[str, Any], task_id: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    manifest = validate_candidate_artifact(
        path,
        expected={
            "variant": spec["authority"]["fit_variant"],
            "task_id": task_id,
            "step": spec["authority"]["supervised_step"],
        },
    )
    prefix = f"task{task_id}_supervised"
    if (
        sha256_file(path / "candidate_manifest.json")
        != spec["authority"][f"{prefix}_manifest_sha256"]
        or sha256_file(path / "trainable_state.safetensors")
        != spec["authority"][f"{prefix}_state_sha256"]
    ):
        raise GateZeroTaskLocalRLRuntimeError("supervised LoRA state hash changed")
    return load_file(path / "trainable_state.safetensors"), manifest


def _open_live_arm(
    *,
    spec: dict[str, Any],
    parent: dict[str, Any],
    phase0: dict[str, Any],
    fit: dict[str, Any],
    checkpoint: dict[str, Any],
    manifest_path: Path,
    dataset_root: Path,
    source_base_checkpoint: Path,
    fit_root: Path,
    task_id: int,
    initialization: str,
) -> LiveRLArm:
    variant = spec["authority"]["fit_variant"]
    variant_spec = dict(fit["fit"][variant])
    support, query, task_authorities = _load_task_datasets(
        spec=fit,
        parent=parent,
        phase0=phase0,
        manifest=manifest_path,
        dataset_root=dataset_root,
        task_id=task_id,
    )
    evaluator = None
    try:
        loaded = list(
            _load_policy(
                source_base_checkpoint / "pretrained_model",
                {"task_suite": "libero_90", "task_id": task_id},
            )
        )
        model, summary = configure_oracle_variant(
            loaded[0],
            parent=parent,
            checkpoint=checkpoint,
            variant=variant,
            variant_spec=variant_spec,
        )
        initial_authority: dict[str, Any] = {
            "initialization": initialization,
            "physical_zero": initialization == "zero_init",
        }
        if initialization == "supervised_init":
            state, selected = _load_supervised_state(
                _supervised_candidate_path(fit_root, spec, task_id),
                spec=spec,
                task_id=task_id,
            )
            restore_trainable_state(model, state)
            initial_authority.update(
                {
                    "supervised_step": selected["step"],
                    "supervised_state_sha256": selected["files"]["trainable_state.safetensors"]["sha256"],
                }
            )
        selection = fit["selection"]
        evaluator = FixedQueryEvaluator(
            query,
            preprocessor=loaded[1],
            batch_size=selection["evaluation_batch_size"],
            num_workers=fit["fit"]["num_workers"],
            anchor_count_per_demo=selection["anchor_frames_per_demo"],
            action_chunk_size=parent["data"]["action_chunk_size"],
            fixed_noise_seed=selection["fixed_noise_seed"],
            fixed_time_seed=selection["fixed_time_seed"],
            inference_noise_seed=selection["inference_noise_seed"],
        )
        with model.disable_adapter():
            reference = evaluator.capture_base_reference(model)
        optimizer = build_oracle_optimizer(model, variant_spec)
        session = OracleModelSession(
            support,
            evaluator,
            model,
            loaded[1],
            optimizer,
            None,
            reference,
            summary,
            task_authorities,
        )
        loaded[0] = model
        language, task_authority = _task_authority(task_id, list(range(40, 48)))
        return LiveRLArm(session, tuple(loaded), language, task_authority, initial_authority)
    except BaseException:
        if evaluator is not None:
            evaluator.close()
        else:
            query.close()
        support.close()
        raise


def _recovery_authorities(
    *, spec_path: Path, spec: Mapping[str, Any], task_id: int, initialization: str
) -> dict[str, Any]:
    return {
        "task_local_rl_contract_sha256": sha256_file(spec_path),
        "candidate_diagnostic_result_sha256": spec["authority"]["candidate_diagnostic_result_sha256"],
        "previous_awr_result_sha256": spec["authority"]["previous_awr_result_sha256"],
        "task_id": task_id,
        "initialization": initialization,
        "fit_variant": spec["authority"]["fit_variant"],
        "trainable_parameters": spec["lora"]["trainable_parameters"],
        "interaction_budget_unit": "source_environment_episodes",
    }


def _resume_or_initialize(
    *,
    arm: LiveRLArm,
    arm_root: Path,
    authorities: dict[str, Any],
    resume: bool,
) -> int:
    if not resume:
        if any(arm_root.iterdir()):
            raise GateZeroTaskLocalRLRuntimeError("fresh RL arm output is not empty")
        return 0
    last = (arm_root / "recovery" / "last").resolve(strict=True)
    step = load_recovery_artifact(
        last,
        model=arm.session.model,
        optimizer=arm.session.optimizer,
        expected={"authorities": authorities},
    )
    if step != 8:
        raise GateZeroTaskLocalRLRuntimeError("RL resume step is not an atomic round boundary")
    return step


def _trackio_start(
    *, args: argparse.Namespace, spec: Mapping[str, Any], task_id: int, initialization: str
) -> Any:
    import trackio

    trackio.init(
        project=spec["resources"]["tracking_project"],
        name=f"{args.output_dir.name}_t{task_id}_{initialization}_to{args.stop_after_episodes}",
        group=spec["resources"]["tracking_group"],
        config={
            "task_id": task_id,
            "initialization": initialization,
            "stop_after_episodes": args.stop_after_episodes,
            "trainable_parameters": spec["lora"]["trainable_parameters"],
            "physical_gpus": args.physical_gpus,
        },
        auto_log_gpu=True,
        gpu_log_interval=1.0,
        auto_log_cpu=True,
        cpu_log_interval=1.0,
    )
    return trackio


def _run_rounds(
    *,
    arm: LiveRLArm,
    arm_root: Path,
    authorities: dict[str, Any],
    spec: Mapping[str, Any],
    task_id: int,
    start_episodes: int,
    stop_episodes: int,
    tracker: Any,
) -> list[dict[str, Any]]:
    rounds_root = arm_root / "rounds"
    rounds_root.mkdir(exist_ok=True)
    records = []
    episodes_per_round = spec["training_interaction"]["episodes_per_round_per_task_initialization"]
    optimizer_steps_per_round = spec["algorithm"]["optimizer_steps_per_rollout_round"]
    for round_index in range(start_episodes // episodes_per_round, stop_episodes // episodes_per_round):
        replay, collection = collect_training_round(
            runtime=arm.runtime,
            task_id=task_id,
            language=arm.language,
            round_index=round_index,
            spec=spec,
        )
        if (
            collection["mechanics_valid"] is not True
            or collection["saturation_fraction"] > spec["exploration"]["maximum_saturation_fraction"]
        ):
            failed = rounds_root / f"{(round_index + 1) * episodes_per_round:06d}_failed.json"
            atomic_json(failed, {"status": "collection_safeguard_failed", "collection": collection})
            raise GateZeroTaskLocalRLRuntimeError("RL collection failed mechanics or saturation guard")
        training = train_signed_flow_ratio_round(
            arm.session,
            replay,
            spec=spec,
            optimizer_step_start=round_index * optimizer_steps_per_round,
        )
        interaction_episodes = (round_index + 1) * episodes_per_round
        record = {
            "interaction_episodes": interaction_episodes,
            "collection": collection,
            "training": training,
        }
        atomic_json(rounds_root / f"{interaction_episodes:06d}.json", record)
        save_recovery_artifact(
            arm_root,
            variant=f"task_local_rl_{authorities['initialization']}",
            task_id=task_id,
            step=interaction_episodes,
            trainable_state=capture_trainable_state(arm.session.model),
            optimizer=arm.session.optimizer,
            authorities=authorities,
        )
        validate_recovery_artifact(
            (arm_root / "recovery" / "last").resolve(strict=True),
            expected={"step": interaction_episodes, "authorities": authorities},
        )
        tracker.log(
            {
                "interaction_episodes": interaction_episodes,
                "training_success_rate": collection["success_rate"],
                "training_environment_steps": collection["environment_steps"],
                "saturation_fraction": collection["saturation_fraction"],
                "signed_flow_ratio_loss": training[-1]["signed_flow_ratio_loss"],
                "ratio_clip_fraction": training[-1]["ratio_clip_fraction"],
                "gradient_norm": training[-1]["gradient_norm"],
            },
            step=interaction_episodes,
        )
        records.append(record)
        del replay
        gc.collect()
        torch.cuda.empty_cache()
    return records


def _stage_evaluation(
    *,
    arm: LiveRLArm,
    arm_root: Path,
    authorities: dict[str, Any],
    spec: Mapping[str, Any],
    headroom: Mapping[str, Any],
    task_id: int,
    initialization: str,
    interaction_episodes: int,
) -> dict[str, Any]:
    offline = arm.session.evaluator.evaluate_candidate(
        arm.session.model, arm.session.reference, step=interaction_episodes
    )
    development = spec["development_evaluation"]
    rollout_spec = {
        "report": {
            "rollout_batch_size": development["batch_size"],
            "official_rollout_init_state_indices": development["init_state_indices"],
            "seed_start": development["seed_start"],
            "warmup_seed_start": development["warmup_seed_start"],
            "policy_rng_seed": development["policy_rng_seed"],
        },
        "resources": {"retain_one_video_per_report_arm": True},
    }
    stage_root = arm_root / f"stage_{interaction_episodes:06d}"
    stage_root.mkdir(exist_ok=False)
    closed_loop = _closed_loop_metrics(
        runtime=arm.runtime,
        task_id=task_id,
        condition=f"{initialization}_rl_ep{interaction_episodes}",
        language=arm.language,
        spec=rollout_spec,
        output_dir=stage_root,
    )
    initial = _initial_successes(
        headroom,
        task_id=task_id,
        initialization=initialization,
        variant=spec["authority"]["fit_variant"],
    )
    current = [bool(value) for value in closed_loop["successes"]]
    paired_net = sum(int(right) - int(left) for left, right in zip(initial, current, strict=True))
    all_rounds = [
        _load_json(path, "RL round record")
        for path in sorted((arm_root / "rounds").glob("*.json"))
        if int(path.stem) <= interaction_episodes
    ]
    maximum_saturation = max(
        value["collection"]["saturation_fraction"] for value in all_rounds
    )
    record = {
        "step": interaction_episodes,
        "task_id": task_id,
        "initialization": initialization,
        "condition": f"{initialization}_rl",
        "initial_successes": initial,
        "successes": current,
        "paired_net_wins": paired_net,
        "mechanics_valid": closed_loop["mechanics_valid"]
        and all(value["collection"]["mechanics_valid"] for value in all_rounds),
        "maximum_saturation_fraction": maximum_saturation,
        "offline_query_flow_mse": offline["query_flow_mse"],
        "offline_query_reduction_fraction": (
            arm.session.reference.query_flow_mse - offline["query_flow_mse"]
        )
        / arm.session.reference.query_flow_mse,
        "action_drift_proxy": offline["action_drift_proxy"],
        "closed_loop": closed_loop,
        "rounds": all_rounds,
        "task_authority": arm.task_authority,
        "initial_state_authority": arm.initial_state_authority,
    }
    save_candidate_artifact(
        arm_root,
        variant=f"task_local_rl_{initialization}",
        task_id=task_id,
        step=interaction_episodes,
        trainable_state=capture_trainable_state(arm.session.model),
        metrics=record,
        authorities=authorities,
    )
    return record


def _relative_stage_videos(
    output_root: Path, arm_root: Path, record: Mapping[str, Any]
) -> list[str]:
    stage_root = arm_root / f"stage_{record['step']:06d}"
    return [
        (stage_root / value).resolve().relative_to(output_root.resolve()).as_posix()
        for value in record["closed_loop"]["video_paths"]
    ]


def _aggregate_stage(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = sorted(records, key=lambda value: (value["initialization"], value["task_id"]))
    if len(records) != 4 or len({(v["task_id"], v["initialization"]) for v in records}) != 4:
        raise GateZeroTaskLocalRLRuntimeError("aggregate RL stage lacks four unique arms")
    by_initialization = {
        initialization: sorted(
            [value for value in records if value["initialization"] == initialization],
            key=lambda value: value["task_id"],
        )
        for initialization in spec["initializations"]
    }
    metrics = {
        "mechanics_valid": all(value["mechanics_valid"] for value in records),
        "maximum_saturation_fraction": max(value["maximum_saturation_fraction"] for value in records),
        "nonfinite_count": 0,
        "action_drift_by_arm": {
            f"{initialization}_rl": max(
                value["action_drift_proxy"] for value in by_initialization[initialization]
            )
            for initialization in spec["initializations"]
        },
        "paired_net_wins_by_arm": {
            f"{initialization}_rl": [
                value["paired_net_wins"] for value in by_initialization[initialization]
            ]
            for initialization in spec["initializations"]
        },
        "supervised_init_advantage_paired_wins_by_task": [
            sum(
                int(supervised) - int(zero)
                for zero, supervised in zip(
                    by_initialization["zero_init"][task_index]["successes"],
                    by_initialization["supervised_init"][task_index]["successes"],
                    strict=True,
                )
            )
            for task_index in range(2)
        ],
    }
    decision = decide_task_local_rl_node(
        spec,
        interaction_episodes=args.stop_after_episodes,
        metrics=metrics,
    )
    return {
        "schema_version": 1,
        "status": decision["status"],
        "config_filename": args.config.name,
        "config_sha256": sha256_file(args.config),
        "surface": spec["surface"],
        "interaction_episodes_per_task_initialization": args.stop_after_episodes,
        "records": records,
        "aggregate_metrics": metrics,
        "decision": decision,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
        "resources": {
            "physical_gpus": args.physical_gpus,
            "world_size": spec["parallel"]["world_size"],
            "expected_peak_device_memory_mib": spec["resources"]["expected_peak_device_memory_mib"],
        },
        "interpretation": spec["interpretation"],
    }


def _publish_stage(
    *, args: argparse.Namespace, spec: Mapping[str, Any], stage: dict[str, Any]
) -> None:
    stage_root = args.output_dir / "stage_results"
    stage_root.mkdir(exist_ok=True)
    atomic_json(stage_root / f"{args.stop_after_episodes:06d}.json", stage)
    per_task = []
    for record in stage["records"]:
        arm_root = args.output_dir / "arms" / f"task{record['task_id']}_{record['initialization']}"
        per_task.append(
            {
                "task_group": f"task_local_rl:{record['initialization']}:ep{record['step']}",
                "task_id": record["task_id"],
                "metrics": {
                    "successes": record["successes"],
                    "sum_rewards": record["closed_loop"]["sum_rewards"],
                    "max_rewards": record["closed_loop"]["max_rewards"],
                    "video_paths": _relative_stage_videos(args.output_dir, arm_root, record),
                },
            }
        )
    eval_info = {
        "overall": {
            "surface": spec["surface"],
            "status": stage["status"],
            "interaction_episodes_per_task_initialization": args.stop_after_episodes,
            "gate_zero_authorized": False,
            "writer_authorized": False,
        },
        "per_task": per_task,
    }
    atomic_json(args.output_dir / "eval_info.json", eval_info)
    build_eval_gallery(args.output_dir)
    update_latest_link(args.output_dir, args.latest_link)
    if stage["status"] != "continue_same_rl_trajectories_to_32_episodes":
        atomic_json(args.output_dir / RESULT_NAME, stage)
        write_output_checksums(args.output_dir)


def _load_run_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], ...]:
    spec = load_task_local_rl_spec(
        args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        fit_path=args.fit_contract,
        headroom_path=args.headroom_contract,
        diagnostic_path=args.diagnostic_contract,
    )
    parent = load_gate_zero_contract(args.gate_zero_contract, args.phase0_contract)
    phase0 = _load_toml(args.phase0_contract)
    fit = _load_toml(args.fit_contract)
    checkpoint = validate_source_base_checkpoint(
        args.source_base_checkpoint,
        expected={
            "step": spec["authority"]["source_base_checkpoint_step"],
            "checkpoint_role": spec["authority"]["source_base_checkpoint_role"],
        },
    )
    if (
        sha256_file(args.source_base_checkpoint / CHECKPOINT_MANIFEST)
        != spec["authority"]["source_base_checkpoint_manifest_sha256"]
    ):
        raise GateZeroTaskLocalRLRuntimeError("source-base checkpoint hash changed")
    headroom, _, _ = _validate_result_authorities(
        spec,
        headroom_result=args.headroom_result,
        diagnostic_result=args.diagnostic_result,
        previous_awr_result=args.previous_awr_result,
    )
    return spec, parent, phase0, fit, checkpoint, headroom


def run_task_local_rl(args: argparse.Namespace) -> dict[str, Any]:
    context = _initialize_parallel()
    arm: LiveRLArm | None = None
    tracker = None
    try:
        spec, parent, phase0, fit, checkpoint, headroom = _load_run_inputs(args)
        task_id, initialization = assigned_task_local_rl_arm(
            rank=context.rank, world_size=context.world_size, spec=spec
        )
        arm_root = args.output_dir / "arms" / f"task{task_id}_{initialization}"
        arm_root.mkdir(parents=True, exist_ok=args.resume)
        arm = _open_live_arm(
            spec=spec,
            parent=parent,
            phase0=phase0,
            fit=fit,
            checkpoint=checkpoint,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            source_base_checkpoint=args.source_base_checkpoint,
            fit_root=args.fit_root,
            task_id=task_id,
            initialization=initialization,
        )
        authorities = _recovery_authorities(
            spec_path=args.config,
            spec=spec,
            task_id=task_id,
            initialization=initialization,
        )
        start_episodes = _resume_or_initialize(
            arm=arm,
            arm_root=arm_root,
            authorities=authorities,
            resume=args.resume,
        )
        if (
            args.stop_after_episodes not in spec["training_interaction"]["interaction_episode_nodes"]
            or args.stop_after_episodes <= start_episodes
        ):
            raise GateZeroTaskLocalRLRuntimeError("invalid staged RL stop")
        tracker = _trackio_start(
            args=args,
            spec=spec,
            task_id=task_id,
            initialization=initialization,
        )
        _run_rounds(
            arm=arm,
            arm_root=arm_root,
            authorities=authorities,
            spec=spec,
            task_id=task_id,
            start_episodes=start_episodes,
            stop_episodes=args.stop_after_episodes,
            tracker=tracker,
        )
        local_record = _stage_evaluation(
            arm=arm,
            arm_root=arm_root,
            authorities=authorities,
            spec=spec,
            headroom=headroom,
            task_id=task_id,
            initialization=initialization,
            interaction_episodes=args.stop_after_episodes,
        )
        gathered = _gather(context, local_record)
        stage = None
        if context.is_primary:
            stage = _aggregate_stage(args=args, spec=spec, records=gathered or [])
            _publish_stage(args=args, spec=spec, stage=stage)
        stage = _broadcast(context, stage)
        tracker.log(
            {
                "development_successes": sum(local_record["successes"]),
                "paired_net_wins": local_record["paired_net_wins"],
                "query_reduction_fraction": local_record["offline_query_reduction_fraction"],
                "action_drift_proxy": local_record["action_drift_proxy"],
            },
            step=args.stop_after_episodes,
        )
        tracker.finish()
        tracker = None
        return stage
    except BaseException as error:
        rank = context.rank
        failure = {
            "schema_version": 1,
            "status": "task_local_rl_runtime_failed",
            "rank": rank,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "gate_zero_authorized": False,
            "writer_authorized": False,
        }
        try:
            atomic_json(args.output_dir / f"failure_rank{rank}.json", failure)
        except Exception:
            pass
        if tracker is not None:
            tracker.finish()
        raise
    finally:
        if arm is not None:
            arm.close()
        _close_parallel(context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config gate-zero-contract phase0-contract fit-contract headroom-contract "
        "diagnostic-contract manifest dataset-root source-base-checkpoint fit-root "
        "headroom-result diagnostic-result previous-awr-result output-dir latest-link"
    ).split():
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--stop-after-episodes", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.output_dir.is_absolute() or not args.latest_link.is_absolute():
        raise GateZeroTaskLocalRLRuntimeError("RL output paths must be absolute")
    started = time.time()
    result = run_task_local_rl(args)
    payload = {
        "event": "gate_zero_task_local_rl_stage_complete",
        "status": result["status"],
        "interaction_episodes": args.stop_after_episodes,
        "wall_seconds": time.time() - started,
        "gate_zero_authorized": False,
        "writer_authorized": False,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
