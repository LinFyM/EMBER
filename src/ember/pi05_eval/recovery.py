"""Asset reinspection and launcher recovery helpers for PI05 evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.eval_adapters import (
    DYNAMIC_K_WRITER_KIND,
    FUNCTIONAL_CODE_WRITER_KIND,
    inspect_dynamic_k_writer_adapter,
    inspect_functional_code_writer_adapter,
    inspect_source_sft_adapter,
    inspect_task_expert_adapter,
    select_task_expert_adapter_tasks,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    SEEN_PANEL_RELATIVE_PATH,
    git_state,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_eval_queue import (
    failed_jobs,
    publish_json_exclusive,
    validate_worker_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_EXPERT_DIAGNOSTIC_SUBSETS = {
    "successful_on_policy_occupancy",
    "successful_expert_equivalence_occupancy",
    "phase_decoder_fit_projected_occupancy",
    "phase_aligned_decoder_held5",
}


def active_worker_pids(output_dir: Path) -> list[int]:
    needle = str(output_dir.resolve()).encode()
    active = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes()
        except OSError:
            continue
        if (
            b"evaluate_pi05.py" in command
            and b"worker" in command
            and needle in command
        ):
            active.append(int(path.parent.name))
    return sorted(active)


def _reinspect_adapter(
    adapter: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    model: Mapping[str, Any],
) -> Mapping[str, Any]:
    tasks = tuple(
        argparse.Namespace(suite=row["suite"], task_id=int(row["task_id"]))
        for row in contract["tasks"]
    )
    require_formal = contract["mode"] != "smoke"
    if adapter.get("kind") == "shared_source_sft_lora":
        return inspect_source_sft_adapter(
            config_path=Path(adapter["config"]["path"]),
            checkpoint=Path(adapter["checkpoint"]["path"]),
            source=model,
            tasks=tasks,
            evaluation_role=str(adapter["evaluation_role"]),
            require_formal=require_formal,
        )
    if adapter.get("kind") == "task_local_expert_bank":
        manifest_path = adapter.get("projection", {}).get("manifest_path")
        subset = adapter.get("information_wall", {}).get("diagnostic_subset")
        inspection_tasks = tasks
        if subset in TASK_EXPERT_DIAGNOSTIC_SUBSETS:
            inspection_tasks = tuple(
                argparse.Namespace(suite=row[0], task_id=int(row[1]))
                for row in adapter["information_wall"].get(
                    "inspection_task_keys", ()
                )
            )
        inspected = inspect_task_expert_adapter(
            config_path=Path(adapter["config"]["path"]),
            bank_root=Path(adapter["bank_root"]),
            step=int(adapter["step"]),
            source=model,
            tasks=inspection_tasks,
            evaluation_role=str(contract["role"]),
            require_formal=require_formal,
            projection_manifest=(
                Path(str(manifest_path)) if manifest_path is not None else None
            ),
        )
        if subset in TASK_EXPERT_DIAGNOSTIC_SUBSETS:
            return select_task_expert_adapter_tasks(
                inspected,
                tasks,
                diagnostic_subset=str(subset),
            )
        return inspected
    if adapter.get("kind") == DYNAMIC_K_WRITER_KIND:
        return inspect_dynamic_k_writer_adapter(
            config_path=Path(adapter["config"]["path"]),
            checkpoint=Path(adapter["writer_asset"]["checkpoint"]),
            video_data_root=Path(adapter["video_data"]["root"]),
            source=model,
            tasks=tasks,
            video_condition=str(adapter["video_condition"]),
            video_seed=int(adapter["video_schedule"]["seed"]),
            video_sampling_mode=str(adapter["video_schedule"]["sampling_mode"]),
            require_formal=require_formal,
            evaluation_k=int(
                adapter.get("information_wall", {}).get("evaluation_k", 1)
            ),
        )
    if adapter.get("kind") == FUNCTIONAL_CODE_WRITER_KIND:
        return inspect_functional_code_writer_adapter(
            config_path=Path(adapter["config"]["path"]),
            checkpoint=Path(adapter["writer_asset"]["checkpoint"]),
            video_data_root=Path(adapter["video_data"]["root"]),
            source=model,
            tasks=tasks,
            video_condition=str(adapter["video_condition"]),
            video_seed=int(adapter["video_schedule"]["seed"]),
            video_sampling_mode=str(adapter["video_schedule"]["sampling_mode"]),
            require_formal=require_formal,
            evaluation_k=int(
                adapter.get("information_wall", {}).get("evaluation_k", 1)
            ),
        )
    raise Pi05EvaluationError("evaluation adapter kind changed after prepare")


def validate_resume_inputs(contract: dict[str, Any]) -> None:
    authorities = load_evaluation_authorities(
        Path(contract["authorities"]["config_path"]), REPO_ROOT
    )
    current_git = git_state(REPO_ROOT)
    if (
        current_git["commit"] != contract["git"]["commit"]
        or contract["mode"] != "smoke"
        and current_git["dirty_paths"]
    ):
        raise Pi05EvaluationError(
            "evaluator checkout differs from the sealed run commit"
        )
    expected_role_authority = None
    if contract.get("role") == "seen_panel":
        path = REPO_ROOT / SEEN_PANEL_RELATIVE_PATH
        expected_role_authority = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "schema_version": authorities.seen_panel.get("schema_version"),
        }
    elif str(contract.get("role", "")).startswith("nonheld_meta"):
        path = Path(authorities.paths["meta_protocol"])
        expected_role_authority = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "schema_version": authorities.meta_protocol.get("schema_version"),
        }
    if contract.get("role_authority") != expected_role_authority:
        raise Pi05EvaluationError("evaluation role authority changed after prepare")
    model = inspect_source_checkpoint(
        authorities,
        Path(contract["model"]["source_run"]),
        Path(contract["model"]["checkpoint"]),
        evaluation_mode=contract["mode"],
    )
    tokenizer = inspect_tokenizer(authorities, Path(contract["tokenizer"]["path"]))
    if model != contract["model"] or tokenizer != contract["tokenizer"]:
        raise Pi05EvaluationError("evaluation model or tokenizer changed after prepare")
    normalization = Path(contract["normalization"]["path"])
    if not normalization.is_file() or normalization.stat().st_size != int(
        contract["normalization"]["bytes"]
    ):
        raise Pi05EvaluationError("evaluation normalization changed after prepare")
    task_subset = contract.get("diagnostic_task_subset")
    if task_subset is not None:
        path = Path(str(task_subset.get("selection_path", "")))
        if (
            task_subset.get("schema_version")
            != "ember_pi05_task_subset_selection_v1"
            or not path.is_file()
            or path.stat().st_size != int(task_subset.get("selection_bytes", -1))
        ):
            raise Pi05EvaluationError("evaluation task subset changed after prepare")
    adapter = contract.get("adapter")
    if (
        adapter is not None
        and _reinspect_adapter(adapter, contract=contract, model=model) != adapter
    ):
        raise Pi05EvaluationError("evaluation adapter assets changed after prepare")


def worker_ids(
    replicas_per_gpu: int, physical_gpu_ids: Sequence[int]
) -> tuple[str, ...]:
    values = tuple(
        f"{gpu}-r{replica}"
        for gpu in physical_gpu_ids
        for replica in range(replicas_per_gpu)
    )
    validate_worker_layout(
        values,
        replicas_per_gpu,
        physical_gpu_ids=physical_gpu_ids,
    )
    return values


def record_launcher_failure(
    output_dir: Path,
    *,
    return_codes: dict[str, int],
    queue: dict[str, Any],
    invocation_id: str,
    worker_pids: dict[str, int],
    error: str | None = None,
) -> Path:
    logs = [
        {"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size}
        for path in sorted((output_dir / "worker_logs").glob("*.log"))
    ]
    path = output_dir / "failures" / f"launcher_{time.time_ns()}.json"
    publish_json_exclusive(
        path,
        {
            "schema_version": "ember_pi05_eval_launcher_failure_v1",
            "unix": time.time(),
            "invocation_id": invocation_id,
            "error": error,
            "worker_pids": worker_pids,
            "return_codes": return_codes,
            "queue": queue,
            "failed_jobs": list(failed_jobs(output_dir / "queue.sqlite3")),
            "worker_logs": logs,
        },
    )
    return path
