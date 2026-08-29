"""Training-only initialization from validated task-local functional codes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.contracts import TargetOwner
from ember.ecp.native_factors import native_output_group_count, rms_normalize
from ember.ecp.natural_program import NaturalProgram
from ember.pi05_source_checkpoint import read_json


FUNCTIONAL_CODE_INITIALIZATION = (
    "fit_only_functional_positive_control_minimum_norm_heads"
)
FUNCTIONAL_CODE_INITIALIZATION_SCHEMA = (
    "ember_ecp_routing_functional_code_initialization_v1"
)
R5_SHARED_FUNCTIONAL_CHART = "r5_shared_functional_chart_step110"


def load_passed_r5_primal_scorer(
    config: Mapping[str, Any],
    compiler: torch.nn.Module,
    *,
    asset_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Restore the passed R5 shared scorer, never its fixed routing inputs."""

    checkpoint = (
        asset_root / str(config["authorities"]["r5_primal_scorer_checkpoint"])
    ).resolve()
    aggregate_path = (
        asset_root / str(config["authorities"]["r5_gate_aggregate"])
    ).resolve()
    run_root = checkpoint.parent.parent
    run_contract = read_json(run_root / "run_contract.json")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    aggregate = read_json(aggregate_path)
    files = manifest.get("files", {})
    expected_files = {
        "ecp.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(6)),
    }
    if (
        checkpoint_macro(checkpoint) != 110
        or aggregate.get("schema_version")
        != "ember_ecp_routing_token_control_gate_report_v1"
        or aggregate.get("status") != "complete"
        or aggregate.get("gate_pass") is not True
        or aggregate.get("primary_pass") is not True
        or not aggregate.get("checks")
        or not all(bool(value) for value in aggregate["checks"].values())
        or Path(str(aggregate.get("checkpoint", {}).get("path", ""))).resolve()
        != checkpoint
        or int(aggregate.get("checkpoint", {}).get("optimizer_step", -1)) != 110
        or run_contract.get("schema_version")
        != "ember_ecp_routing_token_control_run_v2"
        or run_contract.get("stage")
        != "g3_training_only_routing_token_grouped_decoder_control"
        or run_contract.get("phase") != "joint"
        or run_contract.get("mode") != "formal"
        or run_contract.get("model", {}).get("primal_scorer_trainable_partition")
        != "native_heads_only"
        or run_contract.get("information_wall", {}).get(
            "primal_scorer_feature_chart_frozen"
        )
        is not True
        or run_contract.get("inventory", {}).get("action_meta_module_count") != 0
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage")
        != "g3_training_only_routing_token_grouped_decoder_control"
        or int(manifest.get("next_macro", -1)) != 110
        or int(manifest.get("world_size", -1)) != 6
        or manifest.get("run_contract_schema")
        != "ember_ecp_routing_token_control_run_v2"
        or set(files) != expected_files
    ):
        raise ValueError("R5 shared functional-chart authority changed")
    for name, record in files.items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"R5 checkpoint file changed: {name}")
    payload = load_file(str(checkpoint / "ecp.safetensors"), device=str(device))
    prefix = "primal_scorer."
    scorer_state = {
        name[len(prefix) :]: tensor
        for name, tensor in payload.items()
        if name.startswith(prefix)
    }
    if len(scorer_state) != len(payload) or set(scorer_state) != set(
        compiler.primal_scorer.state_dict()
    ):
        raise ValueError("R5 scorer tensor inventory changed")
    compiler.primal_scorer.load_state_dict(scorer_state, strict=True)
    return {
        "kind": R5_SHARED_FUNCTIONAL_CHART,
        "checkpoint": str(checkpoint),
        "gate_aggregate": str(aggregate_path),
        "optimizer_step": 110,
        "training_commit": str(run_contract["git"]["commit"]),
        "tensor_bytes": int(files["ecp.safetensors"]["bytes"]),
        "fixed_routing_token_loaded": False,
        "task_lookup_parameters_loaded": False,
    }


def minimum_norm_head_solution(
    features: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Solve a full-row-rank linear head in FP64 with minimum parameter norm."""

    if (
        features.ndim != 2
        or labels.ndim != 2
        or features.shape[0] != labels.shape[0]
        or features.shape[0] > features.shape[1]
    ):
        raise ValueError("functional-code head fit schema changed")
    x = features.detach().double().cpu()
    y = labels.detach().double().cpu()
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    tolerance = (
        torch.finfo(singular.dtype).eps
        * max(x.shape)
        * singular.max().clamp_min(torch.finfo(singular.dtype).tiny)
    )
    rank = int((singular > tolerance).sum())
    if rank != x.shape[0]:
        raise ValueError(
            f"functional-code hidden rows are rank deficient: {rank}/{x.shape[0]}"
        )
    solution = vh.transpose(0, 1) @ (
        (u.transpose(0, 1) @ y) / singular[:, None]
    )
    prediction = x @ solution
    relative_error = float(
        (prediction - y).norm() / y.norm().clamp_min(1e-12)
    )
    if not bool(torch.isfinite(solution).all()) or relative_error > 1e-9:
        raise ValueError("functional-code minimum-norm solve is unstable")
    return solution.transpose(0, 1), {
        "rows": int(x.shape[0]),
        "columns": int(x.shape[1]),
        "rank": rank,
        "minimum_to_maximum_singular_ratio": float(singular[-1] / singular[0]),
        "fp64_relative_fit_error": relative_error,
    }


def _functional_code_authority(
    config: Mapping[str, Any],
    *,
    asset_root: Path,
    task_ids: tuple[int, ...],
) -> tuple[Path, tuple[Path, ...]]:
    root = (
        asset_root / str(config["authorities"]["positive_control_root"])
    ).resolve()
    aggregate_path = root / "aggregate.json"
    aggregate = read_json(aggregate_path)
    if (
        aggregate.get("schema_version")
        != "ember_ecp_j2_functional_positive_control_aggregate_v1"
        or aggregate.get("overall_gate")
        != "pass_after_runtime_microbatch_correction"
        or {int(row["task"]) for row in aggregate.get("tasks", ())}
        != set(task_ids)
    ):
        raise ValueError("functional-code aggregate authority changed")
    code_paths = []
    for task in task_ids:
        task_root = root / f"task_{task:03d}"
        result = read_json(task_root / "result.json")
        code_path = task_root / "task_local_primal.safetensors"
        checkpoint = result.get("checkpoint", {})
        if (
            result.get("schema_version")
            != "ember_ecp_j2_functional_positive_control_task_v1"
            or result.get("status") != "complete"
            or int(result.get("task", -1)) != task
            or checkpoint.get("deployment_candidate") is not False
            or Path(str(checkpoint.get("path", ""))).resolve() != code_path
            or not code_path.is_file()
            or code_path.stat().st_size != int(checkpoint.get("bytes", -1))
        ):
            raise ValueError(
                f"functional-code checkpoint authority changed for task {task}"
            )
        with safe_open(str(code_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if (
            metadata.get("schema_version")
            != "ember_ecp_j2_functional_positive_control_task_v1"
            or metadata.get("deployment_candidate") != "false"
            or metadata.get("task") != str(task)
        ):
            raise ValueError(
                f"functional-code tensor metadata changed for task {task}"
            )
        code_paths.append(code_path)
    return aggregate_path, tuple(code_paths)


@torch.no_grad()
def initialize_functional_code_heads(
    *,
    config: Mapping[str, Any],
    asset_root: Path,
    compiler: torch.nn.Module,
    owners: Sequence[TargetOwner],
    task_ids: tuple[int, ...],
    program_for_task: Callable[[int], NaturalProgram],
) -> dict[str, Any]:
    """Interpolate validated task-local primals into existing shared heads."""

    owners = tuple(owners)
    aggregate_path, code_paths = _functional_code_authority(
        config, asset_root=asset_root, task_ids=task_ids
    )
    scorer = compiler.primal_scorer
    input_features: list[list[torch.Tensor]] = [list() for _ in owners]
    output_features: list[list[list[torch.Tensor]]] = [
        [list() for _ in range(native_output_group_count(owner))]
        for owner in owners
    ]
    input_labels: list[list[torch.Tensor]] = [list() for _ in owners]
    output_labels: list[list[list[torch.Tensor]]] = [
        [list() for _ in range(native_output_group_count(owner))]
        for owner in owners
    ]
    expected_keys = {"fixed_scales"}
    expected_keys.update(f"input_code.{target}" for target in range(len(owners)))
    expected_keys.update(f"output_code.{target}" for target in range(len(owners)))
    for task, code_path in zip(task_ids, code_paths, strict=True):
        state = scorer.program_state(program_for_task(task))
        task_input_features = scorer.input_head_features(state)
        task_output_features = scorer.output_head_features(state)
        tensors = load_file(str(code_path), device="cpu")
        if set(tensors) != expected_keys or tensors["fixed_scales"].shape != (
            len(owners),
            4,
        ):
            raise ValueError(f"functional-code tensor schema changed for task {task}")
        for target, owner in enumerate(owners):
            input_label = rms_normalize(tensors[f"input_code.{target}"].float())
            output_label = rms_normalize(tensors[f"output_code.{target}"].float())
            groups = native_output_group_count(owner)
            if (
                input_label.shape != (4, owner.in_features)
                or output_label.shape
                != (groups, 4, owner.out_features // groups)
            ):
                raise ValueError(
                    f"functional-code native shape changed for task {task}, target {target}"
                )
            input_features[target].append(task_input_features[target].detach())
            input_labels[target].append(input_label)
            for group in range(groups):
                output_features[target][group].append(
                    task_output_features[target][group].detach()
                )
                output_labels[target][group].append(output_label[group])

    input_reports, output_reports = [], []
    maximum_fp32_error = 0.0
    for target, head in enumerate(scorer.input_primal_heads):
        features = torch.cat(input_features[target]).float()
        labels = torch.cat(input_labels[target]).to(features)
        solution, report = minimum_norm_head_solution(features, labels)
        head.weight.copy_(solution.to(head.weight))
        fp32_error = float(
            (head(features) - labels).norm() / labels.norm().clamp_min(1e-12)
        )
        maximum_fp32_error = max(maximum_fp32_error, fp32_error)
        input_reports.append(
            {
                "target": target,
                **report,
                "fp32_relative_fit_error": fp32_error,
            }
        )
    for target, heads in enumerate(scorer.output_primal_heads):
        for group, head in enumerate(heads):
            features = torch.cat(output_features[target][group]).float()
            labels = torch.cat(output_labels[target][group]).to(features)
            solution, report = minimum_norm_head_solution(features, labels)
            head.weight.copy_(solution.to(head.weight))
            fp32_error = float(
                (head(features) - labels).norm() / labels.norm().clamp_min(1e-12)
            )
            maximum_fp32_error = max(maximum_fp32_error, fp32_error)
            output_reports.append(
                {
                    "target": target,
                    "group": group,
                    **report,
                    "fp32_relative_fit_error": fp32_error,
                }
            )
    if maximum_fp32_error > 1e-2:
        raise ValueError(
            "functional-code FP32 head initialization is numerically unstable: "
            f"{maximum_fp32_error:.9g}"
        )
    all_reports = (*input_reports, *output_reports)
    return {
        "schema_version": FUNCTIONAL_CODE_INITIALIZATION_SCHEMA,
        "kind": FUNCTIONAL_CODE_INITIALIZATION,
        "task_ids": list(task_ids),
        "authority_files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in (aggregate_path, *code_paths)
        ],
        "head_count": len(all_reports),
        "minimum_hidden_rank": min(int(row["rank"]) for row in all_reports),
        "minimum_singular_ratio": min(
            float(row["minimum_to_maximum_singular_ratio"])
            for row in all_reports
        ),
        "maximum_fp64_relative_fit_error": max(
            float(row["fp64_relative_fit_error"]) for row in all_reports
        ),
        "maximum_fp32_relative_fit_error": maximum_fp32_error,
        "scale_source": "frozen_shared_scale_prior_not_task_local_fixed_scales",
        "deployment_candidate": False,
    }
