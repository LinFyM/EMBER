"""Launch-only orchestration for rank-reserved profile, vertical, and seal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ember.eval_adapters import adapter_requests as _adapter_requests
from ember.expert_manifold.rank_reserved_contract import (
    RANK_RESERVED_CANONICAL_CONFIG,
    load_rank_reserved_config,
    load_rank_reserved_profile_evidence,
    rank_reserved_output_path,
    seal_rank_reserved_deployment,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.launcher import gpu_preflight as _gpu_preflight
from ember.pi05_eval.preparation import parse_gpu_indices as _parse_gpu_indices
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
    load_run_contract,
)
from ember.pi05_eval_queue import (
    publish_json_exclusive,
    read_json_with_size,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_SCRIPT = REPO_ROOT / "scripts/evaluate_pi05.py"
WRITER_PROFILE_PREFLIGHT = "writer_generation_preflight.json"
WRITER_PROFILE_WORKER_LOG = "writer_generation_profile_worker.log"


def _profile_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise Pi05EvaluationError(
            "Writer profile batch sizes must be comma-separated integers"
        ) from error
    if (
        result != tuple(sorted(set(result)))
        or len(result) < 3
        or result[0] < 8
        or not {8, 16, 32}.issubset(result)
        or any(item <= 0 for item in result)
    ):
        raise Pi05EvaluationError("Writer profile batch sizes are invalid")
    return result


def _profile_worker_launch(
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
    physical_gpu: int,
    batch_sizes: Sequence[int],
    warmup_runs: int,
    measured_runs: int,
) -> tuple[list[str], dict[str, str]]:
    replicas = int(contract["parallel"]["replicas_per_gpu"])
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(REPO_ROOT / "src"),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(physical_gpu),
        OMP_NUM_THREADS=str(
            contract["parallel"]["omp_threads_per_worker"][str(replicas)]
        ),
    )
    command = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        "profile-writer-worker",
        "--output-dir",
        str(output_dir.resolve()),
        "--worker-id",
        f"{physical_gpu}-r0",
        "--profile-batch-sizes",
        ",".join(str(value) for value in batch_sizes),
        "--profile-warmup-runs",
        str(warmup_runs),
        "--profile-measured-runs",
        str(measured_runs),
    ]
    return command, environment


def _preflight_gpu_identity(
    preflight: Mapping[str, Any]
) -> tuple[tuple[str, ...], ...]:
    identities = []
    for row in preflight.get("gpus", ()):
        fields = tuple(value.strip() for value in str(row).split(",")[:3])
        if len(fields) != 3:
            raise Pi05EvaluationError("Writer profile GPU identity is invalid")
        identities.append(fields)
    return tuple(identities)


def _tracked_authority_head(state: Mapping[str, Any]) -> bool:
    """Require the writable canonical branch, never a detached frozen checkout."""

    return bool(
        git_state_is_clean_pushed_or_frozen_authority(state)
        and state.get("branch") == "codex/bci-continuation"
        and state.get("upstream") == "origin/codex/bci-continuation"
        and state.get("commit") == state.get("authority_commit")
    )


def profile_writer_worker_run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    contract = load_run_contract(output_dir / "run_contract.json")
    contract_git = contract.get("git", {})
    live_git = git_state(REPO_ROOT)
    physical = tuple(int(value) for value in contract["parallel"]["physical_gpu_ids"])
    launcher_preflight, _ = read_json_with_size(output_dir / WRITER_PROFILE_PREFLIGHT)
    live_preflight = _gpu_preflight(physical)
    sizes = _profile_batch_sizes(args.profile_batch_sizes)
    if (
        len(physical) != 1
        or not git_state_is_clean_pushed_or_frozen_authority(live_git)
        or live_git.get("commit") != contract_git.get("commit")
        or not git_state_is_clean_pushed_or_frozen_authority(contract_git)
        or args.worker_id != f"{physical[0]}-r0"
        or launcher_preflight.get("physical_gpu_ids") != list(physical)
        or launcher_preflight.get("compute_applications") != []
        or launcher_preflight.get("device_names") != ["NVIDIA A40"]
        or live_preflight.get("physical_gpu_ids") != list(physical)
        or live_preflight.get("compute_applications") != []
        or live_preflight.get("device_names") != ["NVIDIA A40"]
        or _preflight_gpu_identity(live_preflight)
        != _preflight_gpu_identity(launcher_preflight)
    ):
        raise Pi05EvaluationError("Writer profile worker preflight changed")

    from ember.pi05_evaluation import _initialize_worker
    from ember.writer.generation_profile import profile_writer_generation

    runtime = _initialize_worker(
        output_dir,
        args.worker_id,
        writer_generation=True,
    )
    try:
        result = profile_writer_generation(
            runtime,
            batch_sizes=sizes,
            warmup_runs=int(args.profile_warmup_runs),
            measured_runs=int(args.profile_measured_runs),
            preflight=live_preflight,
        )
    finally:
        runtime.pool.close()
    print(
        json.dumps(
            {
                "event": "writer_generation_profile_worker_complete",
                "root": result["root"],
                "physical_gpu": result["physical_gpu"],
                "selected_writer_model_batch_size": result[
                    "selected_writer_model_batch_size"
                ],
            },
            sort_keys=True,
        )
    )
    return result


def _profile_launch_matches(
    args: argparse.Namespace,
    *,
    sizes: tuple[int, ...],
    writer_kind: str | None,
    source_sft_requested: bool,
    physical_args: Sequence[int] | None,
    state: Mapping[str, Any],
    profile_root: Path,
    cycle1: Path,
) -> bool:
    return all(
        (
            sizes == (8, 16, 32),
            args.mode == "smoke",
            args.role == "validation",
            args.state_count == 4,
            args.replicas_per_gpu == 1,
            args.writer_generators_per_gpu == 1,
            writer_kind == "expert_manifold_writer",
            not source_sft_requested,
            args.expert_manifold_video_condition == "correct",
            args.expert_manifold_video_sampling == "without_replacement",
            args.expert_manifold_config.resolve()
            == RANK_RESERVED_CANONICAL_CONFIG.resolve(),
            args.expert_manifold_checkpoint.resolve() == cycle1,
            args.output_dir.resolve() == profile_root,
            args.writer_lora_cache_root is None,
            physical_args is not None,
            physical_args is not None and len(physical_args) == 1,
            int(args.profile_warmup_runs) == 1,
            int(args.profile_measured_runs) == 2,
            git_state_is_clean_pushed_or_frozen_authority(state),
        )
    )


def _profile_result_matches(
    result: Mapping[str, Any],
    *,
    output_dir: Path,
    physical: tuple[int, ...],
    preflight: Mapping[str, Any],
) -> bool:
    return all(
        (
            result.get("root") == str(output_dir.resolve()),
            int(result.get("physical_gpu", -1)) == physical[0],
            result.get("preflight", {}).get("physical_gpu_ids") == list(physical),
            result.get("preflight", {}).get("compute_applications") == [],
            result.get("preflight", {}).get("device_names") == ["NVIDIA A40"],
            _preflight_gpu_identity(result.get("preflight", {}))
            == _preflight_gpu_identity(preflight),
        )
    )


def profile_writer_run(
    args: argparse.Namespace,
    *,
    prepare_run_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    sizes = _profile_batch_sizes(args.profile_batch_sizes)
    writer_kind, source_sft_requested = _adapter_requests(args)
    physical_args = _parse_gpu_indices(args.gpu_indices)
    state = git_state(REPO_ROOT)
    config = load_rank_reserved_config(args.expert_manifold_config.resolve())
    profile_root = rank_reserved_output_path(
        config["evaluation"]["registered_roots"]["profile"],
        label="rank-reserved profile root",
    )
    cycle1 = (REPO_ROOT / config["assets"]["cycle1"]["checkpoint"]).resolve()
    if not _profile_launch_matches(
        args,
        sizes=sizes,
        writer_kind=writer_kind,
        source_sft_requested=source_sft_requested,
        physical_args=physical_args,
        state=state,
        profile_root=profile_root,
        cycle1=cycle1,
    ):
        raise Pi05EvaluationError(
            "Writer generation profile requires the clean pushed validation/correct "
            "single-A40 smoke contract"
        )
    physical = tuple(physical_args)
    preflight = _gpu_preflight(physical)
    if preflight.get("compute_applications") != []:
        raise Pi05EvaluationError(
            "Writer generation profile requires an idle physical GPU"
        )
    if preflight.get("device_names") != ["NVIDIA A40"]:
        raise Pi05EvaluationError(
            "Writer generation profile requires one physical NVIDIA A40"
        )
    args.writer_generation_batch_size = sizes[-1]
    prepare_run_fn(args, create_evaluation_queue=False)
    contract = load_run_contract(args.output_dir.resolve() / "run_contract.json")
    if (
        tuple(int(value) for value in contract["parallel"]["physical_gpu_ids"])
        != physical
    ):
        raise Pi05EvaluationError("Writer generation profile requires one physical GPU")
    publish_json_exclusive(
        args.output_dir.resolve() / WRITER_PROFILE_PREFLIGHT,
        preflight,
    )
    command, environment = _profile_worker_launch(
        output_dir=args.output_dir,
        contract=contract,
        physical_gpu=physical[0],
        batch_sizes=sizes,
        warmup_runs=int(args.profile_warmup_runs),
        measured_runs=int(args.profile_measured_runs),
    )
    worker_log = args.output_dir.resolve() / WRITER_PROFILE_WORKER_LOG
    with worker_log.open("ab") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise Pi05EvaluationError(
            "Writer generation profile worker failed with return code "
            f"{completed.returncode}; inspect {worker_log}"
        )
    result, _ = read_json_with_size(
        args.output_dir.resolve() / "writer_generation_profile.json"
    )
    if not _profile_result_matches(
        result,
        output_dir=args.output_dir,
        physical=physical,
        preflight=preflight,
    ):
        raise Pi05EvaluationError("Writer generation profile worker result changed")
    print(
        json.dumps(
            {
                "event": "writer_generation_profile_complete",
                "root": result["root"],
                "selected_writer_model_batch_size": result[
                    "selected_writer_model_batch_size"
                ],
                "writer_generation_measurements": result[
                    "writer_generation_measurements"
                ],
            },
            sort_keys=True,
        )
    )
    return result


def rank_reserved_vertical_run(
    args: argparse.Namespace,
    *,
    prepare_run_fn: Callable[..., Mapping[str, Any]],
    start_workers_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    from ember.writer.rank_reserved_vertical import (
        RANK_RESERVED_VERTICAL_PREFLIGHT,
        RANK_RESERVED_VERTICAL_SCHEMA,
    )

    writer_kind, source_sft_requested = _adapter_requests(args)
    physical_args = _parse_gpu_indices(args.gpu_indices)
    state = git_state(REPO_ROOT)
    config = load_rank_reserved_config(args.expert_manifold_config.resolve())
    profile = load_rank_reserved_profile_evidence(
        config,
        require_run_commit=str(state.get("commit", "")),
    )
    selected_batch = int(profile["selected_writer_model_batch_size"])
    vertical_root = rank_reserved_output_path(
        config["evaluation"]["registered_roots"]["vertical"],
        label="rank-reserved vertical root",
    )
    cycle1 = (REPO_ROOT / config["assets"]["cycle1"]["checkpoint"]).resolve()
    if (
        args.mode != "smoke"
        or args.role != "validation"
        or args.state_count != 1
        or args.replicas_per_gpu != 1
        or args.writer_generators_per_gpu != 1
        or writer_kind != "expert_manifold_writer"
        or source_sft_requested
        or args.expert_manifold_video_condition != "correct"
        or args.expert_manifold_video_sampling != "without_replacement"
        or args.expert_manifold_config.resolve()
        != RANK_RESERVED_CANONICAL_CONFIG.resolve()
        or args.expert_manifold_checkpoint.resolve() != cycle1
        or args.output_dir.resolve() != vertical_root
        or args.writer_lora_cache_root is not None
        or physical_args is None
        or len(physical_args) != 1
        or not git_state_is_clean_pushed_or_frozen_authority(state)
    ):
        raise Pi05EvaluationError(
            "rank-reserved vertical requires its registered clean single-A40 "
            "validation/correct state0 contract"
        )
    physical = tuple(physical_args)
    preflight = _gpu_preflight(physical)
    if preflight.get("compute_applications") != [] or preflight.get("device_names") != [
        "NVIDIA A40"
    ]:
        raise Pi05EvaluationError(
            "rank-reserved vertical requires one live idle NVIDIA A40"
        )
    args.writer_generation_batch_size = selected_batch
    prepare_run_fn(args, create_evaluation_queue=True)
    contract = load_run_contract(vertical_root / "run_contract.json")
    if (
        tuple(int(value) for value in contract["parallel"]["physical_gpu_ids"])
        != physical
    ):
        raise Pi05EvaluationError(
            "rank-reserved vertical requires one live idle NVIDIA A40"
        )
    publish_json_exclusive(vertical_root / RANK_RESERVED_VERTICAL_PREFLIGHT, preflight)
    results = start_workers_fn(vertical_root, resume=False)
    mechanism, _ = read_json_with_size(vertical_root / "rank_reserved_vertical.json")
    if (
        mechanism.get("schema_version") != RANK_RESERVED_VERTICAL_SCHEMA
        or mechanism.get("passed") is not True
        or mechanism.get("contract_reference") != contract["contract_reference"]
        or int(results.get("overall", {}).get("episodes", -1)) != 8
    ):
        raise Pi05EvaluationError("rank-reserved vertical deployment smoke failed")
    print(
        json.dumps(
            {
                "event": "rank_reserved_vertical_complete",
                "root": str(vertical_root),
                "selected_writer_model_batch_size": selected_batch,
                "episodes": int(results["overall"]["episodes"]),
                "successes": int(results["overall"]["successes"]),
                "mechanism_passed": True,
            },
            sort_keys=True,
        )
    )
    return mechanism


def rank_reserved_seal_run(args: argparse.Namespace) -> dict[str, Any]:
    """Seal the active config from the two registered live artifacts only."""

    state = git_state(REPO_ROOT)
    config_path = args.expert_manifold_config.resolve()
    if (
        config_path != RANK_RESERVED_CANONICAL_CONFIG.resolve()
        or not _tracked_authority_head(state)
    ):
        raise Pi05EvaluationError(
            "rank-reserved sealing requires the clean pushed canonical branch head"
        )
    evidence = seal_rank_reserved_deployment(
        config_path,
        require_run_commit=str(state.get("commit", "")),
    )
    print(
        json.dumps(
            {
                "event": "rank_reserved_deployment_sealed",
                **evidence,
            },
            sort_keys=True,
        )
    )
    return evidence
