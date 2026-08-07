"""Immutable authorities and seed schedules for canonical PI05 target evaluation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.pi05_assets import (
    Pi05EvaluationError,
    load_protocol,
    prepare_libero_config,
)


EVALUATION_CONFIG_SCHEMA = "ember_pi05_target_evaluation_v1"
RUN_CONTRACT_SCHEMA = "ember_pi05_target_eval_launch_v2"
SUITE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ROLE_NAMES = {
    "all_targets",
    "development_train",
    "validation",
    "test",
    "final_source",
}
DERIVED_ROLE_NAMES = {"seen_panel"}
SEEN_PANEL_RELATIVE_PATH = Path("configs/pi05_seen_panel_v1.json")
FROZEN_SOURCE_POLICY_SUBDIR = "policy"
RUNTIME_REPLICA_PROFILES = (1, 2, 3, 4, 5, 6)
RUNTIME_OMP_THREADS = {"1": 8, "2": 4, "3": 2, "4": 1, "5": 1, "6": 1}


@dataclass(frozen=True)
class EvaluationAuthorities:
    repo_root: Path
    config_path: Path
    config: dict[str, Any]
    protocol: dict[str, Any]
    overlap_audit: dict[str, Any]
    normalization: dict[str, Any]
    source_base_config: dict[str, Any]
    tokenizer_manifest: dict[str, Any]
    seen_panel: dict[str, Any]
    paths: dict[str, str]


@dataclass(frozen=True)
class TargetTaskContract:
    suite: str
    task_id: int
    split_role: str
    language: str
    problem_folder: str
    bddl_file: str
    bddl_bytes: int
    init_states_file: str
    init_states_bytes: int
    installed_init_state_count: int
    horizon: int
    init_state_ids: tuple[int, ...]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Pi05EvaluationError(f"invalid evaluation JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise Pi05EvaluationError(f"evaluation authority is not an object: {path}")
    return value


def _validate_source_normalization(
    normalization: Mapping[str, Any], source_config: Mapping[str, Any]
) -> None:
    authority = normalization.get("authority", {})
    expected_ids = source_config.get("data", {}).get("active_task_ids")
    if (
        normalization.get("schema_version") != "ember_pi05_source_normalization_v1"
        or authority.get("active_task_ids") != expected_ids
        or authority.get("source_suite") != "libero_90"
        or int(authority.get("episodes_per_task", 0)) != 50
        or int(authority.get("validation_or_test_numeric_reads", -1)) != 0
        or set(normalization.get("stats", {})) != {"observation.state", "action"}
    ):
        raise Pi05EvaluationError("evaluation normalization is not the sealed source-only authority")


def _validate_recipe(config: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    feasibility = protocol["pi05_feasibility"]
    policy = config["policy"]
    environment = config["environment"]
    required_policy = {
        "arm": "frozen_pi05_source_base",
        "precision": "bfloat16",
        "chunk_size": feasibility["model_chunk_size"],
        "n_action_steps": feasibility["policy_n_action_steps"],
        "num_inference_steps": feasibility["num_inference_steps"],
        "replan_steps": feasibility["replan_steps"],
        "action_dim": 7,
        "state_dim": 8,
        "model_resolution": feasibility["model_resolution"],
        "camera_rotation_degrees": 180,
        "right_wrist_padding": "zero_image_with_false_mask_via_missing_feature_key",
    }
    required_environment = {
        "fixed_init_states": True,
        "fixed_init_state_count": feasibility["num_trials_per_task"],
        "render_resolution": feasibility["render_resolution"],
        "dummy_settling_steps": feasibility["num_steps_wait"],
        "dummy_action": feasibility["dummy_action"],
        "terminate_on_success": True,
        "horizons": feasibility["max_steps"],
    }
    required_rng = {
        "inference_seed": feasibility["seed"],
        "environment_schedule": (
            "reseed each fixed-state episode with inference_seed before reset and "
            "set_init_state"
        ),
        "policy_noise_schedule": (
            "sha256 first 63 bits of canonical JSON "
            "[seed,suite,task_id,init_state_id,replan_index]"
        ),
        "policy_noise_device": "cpu_generator_then_transfer",
    }
    if policy != required_policy:
        raise Pi05EvaluationError("PI05 target evaluator differs from the official policy recipe")
    if environment != required_environment:
        raise Pi05EvaluationError("PI05 target evaluator differs from the official environment recipe")
    if config.get("rng") != required_rng:
        raise Pi05EvaluationError("PI05 target evaluator differs from the sealed RNG recipe")
    parallel = config["parallel"]
    physical_gpu_count = int(parallel.get("physical_gpu_count", 0))
    if (
        not 0 < physical_gpu_count <= 8
        or parallel.get("allowed_replicas_per_gpu") != [1, 2, 3]
        or parallel.get("gpu0_extra_cuda_roles") != 0
        or int(parallel.get("envs_per_replica", 0)) <= 0
        or parallel.get("omp_threads_per_worker") != {"1": 8, "2": 4, "3": 2}
    ):
        raise Pi05EvaluationError("PI05 evaluation topology is not GPU symmetric")
    if set(config.get("roles", {})) != ROLE_NAMES:
        raise Pi05EvaluationError("PI05 evaluation role contract changed")


def load_evaluation_authorities(
    config_path: Path, repo_root: Path
) -> EvaluationAuthorities:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = _read_object(config_path)
    if config.get("schema_version") != EVALUATION_CONFIG_SCHEMA:
        raise Pi05EvaluationError("unsupported PI05 target evaluation config")
    values: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {"evaluation_config": str(config_path)}
    for name, reference in config.get("authorities", {}).items():
        path = repo_root / reference["path"]
        if not path.is_file():
            raise Pi05EvaluationError(f"missing evaluation authority: {name}")
        values[name] = _read_object(path)
        paths[name] = str(path.resolve())
    required = {
        "protocol",
        "overlap_audit",
        "normalization",
        "source_manifest",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(values) != required:
        raise Pi05EvaluationError("PI05 evaluation authority set changed")
    seen_panel_path = repo_root / SEEN_PANEL_RELATIVE_PATH
    seen_panel = _read_object(seen_panel_path)
    panel_authority = seen_panel.get("authority", {})
    panel_manifest = repo_root / str(panel_authority.get("target_data_manifest", ""))
    if not panel_manifest.is_file():
        raise Pi05EvaluationError("seen panel target-data authority changed")
    paths["seen_panel"] = str(seen_panel_path.resolve())
    protocol = load_protocol(repo_root / config["authorities"]["protocol"]["path"])
    audit = values["overlap_audit"]
    targets = audit.get("target_tasks", [])
    target_keys = {(row.get("suite"), int(row.get("task_id", -1))) for row in targets}
    if (
        audit.get("schema_version") != "ember_pi05_source_overlap_v1"
        or len(targets) != 40
        or target_keys != {(suite, task_id) for suite in SUITE_ORDER for task_id in range(10)}
    ):
        raise Pi05EvaluationError("overlap audit is not the complete target-40 authority")
    _validate_source_normalization(values["normalization"], values["source_base_config"])
    _validate_recipe(config, protocol)
    return EvaluationAuthorities(
        repo_root=repo_root,
        config_path=config_path,
        config=config,
        protocol=protocol,
        overlap_audit=audit,
        normalization=values["normalization"],
        source_base_config=values["source_base_config"],
        tokenizer_manifest=values["tokenizer_manifest"],
        seen_panel=seen_panel,
        paths=paths,
    )


def split_role(protocol: Mapping[str, Any], suite: str, task_id: int) -> str:
    suite_roles = protocol["split"]["suites"][suite]
    matches = [name for name in ("train", "validation", "test") if task_id in suite_roles[name]]
    if len(matches) != 1:
        raise Pi05EvaluationError(f"target task has ambiguous split role: {suite}/{task_id}")
    return matches[0]


def resolve_role_task_keys(
    protocol: Mapping[str, Any],
    role: str,
    seen_panel: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, int], ...]:
    if role not in ROLE_NAMES | DERIVED_ROLE_NAMES:
        raise Pi05EvaluationError(f"unsupported PI05 evaluation role: {role}")
    if role == "seen_panel":
        if seen_panel is None:
            raise Pi05EvaluationError("seen-panel evaluation lacks its sealed authority")
        tasks = seen_panel.get("tasks", [])
        keys = tuple(
            (str(row.get("suite")), int(row.get("task_id", -1))) for row in tasks
        )
        expected_global_ids = (
            [SUITE_ORDER.index(suite) * 10 + task_id for suite, task_id in keys]
            if all(suite in SUITE_ORDER for suite, _ in keys)
            else []
        )
        suite_counts = {
            suite: sum(name == suite for name, _ in keys) for suite in SUITE_ORDER
        }
        valid = (
            seen_panel.get("schema_version") == "ember_pi05_seen_panel_v1"
            and seen_panel.get("selection", {}).get("role") == "train"
            and int(
                seen_panel.get("selection", {}).get("policy_outcome_reads", -1)
            )
            == 0
            and int(
                seen_panel.get("selection", {}).get("trajectory_value_reads", -1)
            )
            == 0
            and len(keys) == len(set(keys)) == 8
            and suite_counts == {suite: 2 for suite in SUITE_ORDER}
            and all(
                task_id in protocol["split"]["suites"][suite]["train"]
                and row.get("split_role") == "train"
                for row, (suite, task_id) in zip(tasks, keys, strict=True)
            )
            and seen_panel.get("summary", {}).get("global_task_ids") == expected_global_ids
            and int(seen_panel.get("summary", {}).get("tasks", -1)) == 8
        )
        if not valid:
            raise Pi05EvaluationError("sealed seen-panel task contract changed")
        return keys
    keys: list[tuple[str, int]] = []
    for suite in SUITE_ORDER:
        roles = protocol["split"]["suites"][suite]
        if role == "all_targets":
            task_ids: Sequence[int] = range(10)
        elif role == "development_train":
            task_ids = roles["train"]
        elif role == "final_source":
            task_ids = (*roles["train"], *roles["validation"])
        else:
            task_ids = roles[role]
        keys.extend((suite, int(task_id)) for task_id in sorted(task_ids))
    if len(keys) != len(set(keys)):
        raise Pi05EvaluationError("PI05 evaluation role contains duplicate tasks")
    return tuple(keys)


def inspect_installed_target_tasks(
    authorities: EvaluationAuthorities,
    *,
    role: str,
    state_count: int,
    libero_config_dir: Path,
) -> tuple[tuple[TargetTaskContract, ...], dict[str, str]]:
    formal_count = int(authorities.config["environment"]["fixed_init_state_count"])
    if not 1 <= state_count <= formal_count:
        raise Pi05EvaluationError("fixed-state count must be within 1..50")
    paths = prepare_libero_config(libero_config_dir)
    from libero.libero import benchmark, get_libero_path

    audit_by_key = {
        (row["suite"], int(row["task_id"])): row
        for row in authorities.overlap_audit["target_tasks"]
    }
    sealed_test_by_key = {
        (row["suite"], int(row["task_id"])): row
        for row in authorities.protocol["test_tasks"]
    }
    horizons = authorities.config["environment"]["horizons"]
    suites = {
        suite_name: benchmark.get_benchmark_dict()[suite_name]() for suite_name in SUITE_ORDER
    }
    result: list[TargetTaskContract] = []
    for suite_name, task_id in resolve_role_task_keys(
        authorities.protocol,
        role,
        authorities.seen_panel if role == "seen_panel" else None,
    ):
        suite = suites[suite_name]
        task = suite.get_task(task_id)
        sealed = audit_by_key[(suite_name, task_id)]
        if (
            task.language != sealed["language"]
            or task.problem_folder != sealed["problem_folder"]
            or task.bddl_file != sealed["bddl_file"]
        ):
            raise Pi05EvaluationError(f"installed LIBERO task differs: {suite_name}/{task_id}")
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_path = (
            Path(get_libero_path("init_states"))
            / suite_name
            / f"{Path(task.bddl_file).stem}.pruned_init"
        )
        if (
            not bddl_path.is_file()
            or bddl_path.stat().st_size != int(sealed["bddl_bytes"])
        ):
            raise Pi05EvaluationError(f"installed BDDL size differs: {suite_name}/{task_id}")
        init_states = suite.get_task_init_states(task_id)
        if not init_path.is_file() or len(init_states) < formal_count:
            raise Pi05EvaluationError(f"installed fixed states incomplete: {suite_name}/{task_id}")
        if split_role(authorities.protocol, suite_name, task_id) == "test":
            sealed_test = sealed_test_by_key.get((suite_name, task_id))
            if (
                sealed_test is None
                or init_path.name != sealed_test.get("init_states_file")
            ):
                raise Pi05EvaluationError(
                    f"installed test fixed-state file differs: {suite_name}/{task_id}"
                )
        result.append(
            TargetTaskContract(
                suite=suite_name,
                task_id=task_id,
                split_role=split_role(authorities.protocol, suite_name, task_id),
                language=task.language,
                problem_folder=task.problem_folder,
                bddl_file=task.bddl_file,
                bddl_bytes=bddl_path.stat().st_size,
                init_states_file=init_path.name,
                init_states_bytes=init_path.stat().st_size,
                installed_init_state_count=len(init_states),
                horizon=int(horizons[suite_name]),
                init_state_ids=tuple(range(state_count)),
            )
        )
    return tuple(result), paths


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    upstream = run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "origin_main": run("rev-parse", "origin/main"),
        "upstream": upstream,
        "upstream_commit": run("rev-parse", upstream),
        "dirty_paths": run("status", "--porcelain").splitlines(),
    }


def _validate_source_checkpoint_provenance(
    authorities: EvaluationAuthorities,
    run_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    trainer: Mapping[str, Any],
) -> None:
    source_config = authorities.source_base_config
    observed = {
        "run_schema": run_contract.get("schema_version"),
        "models": run_contract.get("models"),
        "features": run_contract.get("features"),
        "optimization": run_contract.get("optimization"),
        "task_ids": run_contract.get("task_ids"),
        "manifest_schema": manifest.get("schema_version"),
        "trainer_schema": trainer.get("schema_version"),
        "manifest_step": manifest.get("optimizer_step"),
        "trainer_step": trainer.get("optimizer_step"),
        "manifest_micro_step": manifest.get("micro_step"),
        "trainer_micro_step": trainer.get("micro_step"),
        "ema_enabled": trainer.get("ema_enabled"),
    }
    expected = {
        "run_schema": "ember_pi05_source_launch_v1",
        "models": source_config["models"],
        "features": source_config["features"],
        "optimization": source_config["optimization"],
        "task_ids": source_config["data"]["active_task_ids"],
        "manifest_schema": "ember_pi05_source_checkpoint_v1",
        "trainer_schema": "ember_pi05_source_trainer_state_v1",
        "manifest_step": trainer.get("optimizer_step"),
        "trainer_step": trainer.get("optimizer_step"),
        "manifest_micro_step": trainer.get("micro_step"),
        "trainer_micro_step": trainer.get("micro_step"),
        "ema_enabled": True,
    }
    if observed != expected:
        raise Pi05EvaluationError("source checkpoint provenance contract changed")


def _verified_model_files(
    checkpoint: Path, manifest: Mapping[str, Any]
) -> tuple[Path, list[dict[str, Any]]]:
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise Pi05EvaluationError("source checkpoint manifest files changed")
    expected = {row["path"]: row for row in files}
    if len(expected) != len(files):
        raise Pi05EvaluationError("source checkpoint manifest contains duplicate paths")
    model_path = checkpoint / FROZEN_SOURCE_POLICY_SUBDIR
    observed_files: list[dict[str, Any]] = []
    for relative in (
        f"{FROZEN_SOURCE_POLICY_SUBDIR}/config.json",
        f"{FROZEN_SOURCE_POLICY_SUBDIR}/model.safetensors",
        "trainer_state.json",
    ):
        path = checkpoint / relative
        record = expected.get(relative)
        valid = (
            record is not None
            and path.is_file()
            and path.stat().st_size == int(record["bytes"])
        )
        if not valid:
            raise Pi05EvaluationError(f"source checkpoint model file changed: {relative}")
        if relative.startswith(f"{FROZEN_SOURCE_POLICY_SUBDIR}/"):
            observed_files.append(
                {"path": str(record["path"]), "bytes": int(record["bytes"])}
            )
    return model_path, observed_files


def _validate_model_config(model_path: Path) -> None:
    model_config = _read_object(model_path / "config.json")
    expected_scalars = {
        "type": "pi05",
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 10,
        "num_inference_steps": 10,
        "tokenizer_max_length": 200,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "image_resolution": [224, 224],
    }
    observed_scalars = {key: model_config.get(key) for key in expected_scalars}
    expected_inputs = {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {
            "type": "VISUAL",
            "shape": [3, 224, 224],
        },
        "observation.images.right_wrist_0_rgb": {
            "type": "VISUAL",
            "shape": [3, 224, 224],
        },
        "observation.state": {"type": "STATE", "shape": [8]},
    }
    expected_outputs = {"action": {"type": "ACTION", "shape": [7]}}
    if (
        observed_scalars != expected_scalars
        or model_config.get("input_features") != expected_inputs
        or model_config.get("output_features") != expected_outputs
    ):
        raise Pi05EvaluationError("source checkpoint PI05 interface changed")


def _validate_final_source_policy(
    authorities: EvaluationAuthorities,
    source_run: Path,
    checkpoint: Path,
    run_contract: Mapping[str, Any],
    trainer: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = source_run / "run_summary.json"
    summary = _read_object(summary_path)
    source_config = authorities.source_base_config
    formal = source_config["formal_run"]
    final_step = int(formal["optimizer_steps"])
    runtime = run_contract.get("runtime", {})
    expected_runtime = {
        "world_size": formal["world_size"],
        "micro_batch_size_per_rank": formal["micro_batch_size_per_rank"],
        "gradient_accumulation_steps": formal["gradient_accumulation_steps"],
        "optimizer_steps": final_step,
        "checkpoint_interval": formal["checkpoint_interval"],
        "ema_enabled": True,
        "task_limit": None,
    }
    observed_runtime = {key: runtime.get(key) for key in expected_runtime}
    source_corpus = run_contract.get("asset_validation", {}).get("source_corpus", {})
    expected_corpus = {
        "tasks_checked": len(source_config["data"]["active_task_ids"]),
    }
    observed_corpus = {key: source_corpus.get(key) for key in expected_corpus}
    expected_summary = {
        "schema_version": "ember_pi05_source_run_summary_v1",
        "completed_optimizer_steps": final_step,
        "requested_optimizer_steps": final_step,
        "stopped_early_for_resume_smoke": False,
        "frozen_policy_subdir": FROZEN_SOURCE_POLICY_SUBDIR,
    }
    observed_summary = {key: summary.get(key) for key in expected_summary}
    expected_micro_step = final_step * int(formal["gradient_accumulation_steps"])
    valid = (
        run_contract.get("mode") == "formal"
        and observed_runtime == expected_runtime
        and observed_corpus == expected_corpus
        and int(trainer.get("optimizer_step", -1)) == final_step
        and int(trainer.get("micro_step", -1)) == expected_micro_step
        and checkpoint.name == f"step_{final_step:08d}"
        and observed_summary == expected_summary
        and Path(str(summary.get("final_checkpoint", ""))).parts[-3:]
        == checkpoint.parts[-3:]
    )
    if not valid:
        raise Pi05EvaluationError(
            "screen/formal evaluation requires the selected final formal source policy"
        )
    return {
        "path": str(summary_path.resolve()),
        "bytes": summary_path.stat().st_size,
        "schema_version": str(summary["schema_version"]),
    }


def inspect_source_checkpoint(
    authorities: EvaluationAuthorities,
    source_run: Path,
    checkpoint: Path,
    *,
    evaluation_mode: str,
) -> dict[str, Any]:
    if evaluation_mode not in {"smoke", "screen", "formal"}:
        raise Pi05EvaluationError(f"unsupported checkpoint evaluation mode: {evaluation_mode}")
    source_run = source_run.resolve()
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.parent != source_run or checkpoint.parent.name != "checkpoints":
        raise Pi05EvaluationError("source checkpoint is not owned by the declared source run")
    run_contract = _read_object(source_run / "run_contract.json")
    manifest = _read_object(checkpoint / "checkpoint_manifest.json")
    trainer = _read_object(checkpoint / "trainer_state.json")
    _validate_source_checkpoint_provenance(
        authorities, run_contract, manifest, trainer
    )
    model_path, observed_files = _verified_model_files(checkpoint, manifest)
    _validate_model_config(model_path)
    summary_record = None
    if evaluation_mode in {"screen", "formal"}:
        summary_record = _validate_final_source_policy(
            authorities,
            source_run,
            checkpoint,
            run_contract,
            trainer,
        )
    return {
        "source_run": str(source_run),
        "source_run_contract": {
            "path": str((source_run / "run_contract.json").resolve()),
            "bytes": (source_run / "run_contract.json").stat().st_size,
            "schema_version": str(run_contract["schema_version"]),
        },
        "source_training_commit": run_contract["git"]["commit"],
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": {
            "path": str((checkpoint / "checkpoint_manifest.json").resolve()),
            "bytes": (checkpoint / "checkpoint_manifest.json").stat().st_size,
            "schema_version": str(manifest["schema_version"]),
        },
        "optimizer_step": int(trainer["optimizer_step"]),
        "source_run_summary": summary_record,
        "frozen_policy_subdir": FROZEN_SOURCE_POLICY_SUBDIR,
        "model_path": str(model_path),
        "model_files": observed_files,
    }


def inspect_tokenizer(authorities: EvaluationAuthorities, tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_path = tokenizer_path.resolve()
    manifest = authorities.tokenizer_manifest
    if (
        not tokenizer_path.is_file()
        or tokenizer_path.stat().st_size != int(manifest["bytes"])
    ):
        raise Pi05EvaluationError("OpenPI tokenizer differs from the sealed authority")
    return {
        "path": str(tokenizer_path),
        "bytes": tokenizer_path.stat().st_size,
        "manifest_path": authorities.paths["tokenizer_manifest"],
    }


from ember.pi05_eval.run_contract import (  # noqa: E402
    build_run_contract,
    load_run_contract,
    policy_noise_seed,
)
