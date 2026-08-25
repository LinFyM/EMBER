"""Authority checks and frozen runtime for G3 held5 materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import ECP_CHECKPOINT_SCHEMA, checkpoint_macro
from ember.ecp.contracts import build_target_owners
from ember.ecp.natural_program_data import load_natural_program_tasks
from ember.ecp.shared_compiler import SharedNativeFactorCompiler
from ember.ecp.shared_compiler_assets import (
    authority_path,
    build_frozen_g2_program,
    load_shared_compiler_config,
    load_shared_rank_assets,
)
from ember.ecp.shared_compiler_authority import RUN_SCHEMA
from ember.ecp.stage0_training import (
    stage0_source_authority,
    tokenize_stage0_languages,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import (
    initialize_distributed,
    load_config,
    load_policy,
    seed_everything,
)
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


REPO_ROOT = Path(__file__).resolve().parents[3]
G3_GATE_CONFIG_SCHEMA = "ember_ecp_shared_compiler_g3_gate_v1"
CONDITIONS = ("correct_full", "first_final", "same_task_other")


@dataclass
class G3MaterializationRuntime:
    config: dict[str, Any]
    gate: dict[str, Any]
    state: dict[str, Any]
    source: dict[str, Any]
    source_config: dict[str, Any]
    held: tuple[Any, ...]
    target_keys: dict[int, tuple[str, int]]
    policy: torch.nn.Module
    ranks: Any
    owners: tuple[Any, ...]
    program: torch.nn.Module
    compiler: SharedNativeFactorCompiler
    compiler_macro: int
    shared_contract: dict[str, Any]
    wall: dict[str, Any]
    tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    video_store: RawTeacherVideoStore
    demos: tuple[int, ...]
    view: str
    query_points: int
    rank4_contract: Any
    device: torch.device

    def close(self) -> None:
        self.video_store.close()


def load_g3_gate_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    panel = config.get("video_panel", {})
    gate = config.get("gate", {})
    language = config.get("language_baseline", {})
    if (
        config.get("schema_version") != G3_GATE_CONFIG_SCHEMA
        or config.get("status") != "active_held5_closed_loop_gate"
        or config.get("conditions")
        != [
            "carrier",
            "learned_language_only",
            "correct_full",
            "first_final",
            "same_task_other",
        ]
        or config.get("shuffled_or_reversed_use") is not False
        or set(config.get("authorities", {})) != {"carrier_strict250"}
        or not isinstance(
            config.get("authorities", {}).get("carrier_strict250"), str
        )
        or int(panel.get("K", -1)) != 4
        or panel.get("correct_full") != [5, 6, 7, 8]
        or panel.get("first_final") != panel.get("correct_full")
        or panel.get("same_task_other") != [0, 1, 2, 3]
        or language
        != {
            "method": (
                "fit75_frozen_p_lang_linear_kernel_ridge_to_verified_rank4"
            ),
            "feature": (
                "row_normalized_owner_p_lang_then_centered_global_unit_vector"
            ),
            "relative_ridge": 0.01,
            "set_valued_task_target": (
                "reliability_weighted_member_update_then_rank4"
            ),
            "held_video_reads": 0,
            "held_action_or_reward_reads": 0,
        }
        or set(gate)
        != {
            "full_successes_minimum",
            "breadth_minimum",
            "carrier_retained_minimum",
            "goal_or_long_nonzero",
            "full_over_language_minimum",
            "full_over_first_final_minimum",
            "same_task_retention_minimum",
            "single_checkpoint_only",
        }
    ):
        raise ValueError("unsupported G3 held5 Gate config")
    return config


def load_g3_tasks(
    config: Mapping[str, Any], *, asset_root: Path, data_root: Path
) -> tuple[Any, ...]:
    fold = config["fold"]
    return load_natural_program_tasks(
        meta_protocol_path=authority_path(
            config, "meta_protocol", asset_root=asset_root
        ),
        source_manifest_path=authority_path(
            config, "source_manifest", asset_root=asset_root
        ),
        target_manifest_path=authority_path(
            config, "target_manifest", asset_root=asset_root
        ),
        data_root=data_root,
        target_fit_ids=fold["target_fit_task_ids"],
        target_held_ids=fold["target_held_task_ids"],
        held_meta_fold=int(fold["meta_held_fold"]),
    )


def _load_compiler_checkpoint(
    model: SharedNativeFactorCompiler,
    *,
    run_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    run_root = run_root.resolve()
    checkpoint = checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    contract_path = run_root / "run_contract.json"
    contract = read_json(contract_path)
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    tensor_record = manifest.get("files", {}).get("ecp.safetensors", {})
    tensor_path = checkpoint / "ecp.safetensors"
    completion = read_json(run_root / "segment_completion.json")
    if (
        checkpoint.parent.parent != run_root
        or contract.get("schema_version") != RUN_SCHEMA
        or contract.get("stage") != "g3_shared_compiler"
        or contract.get("mode") != "formal"
        or manifest.get("schema_version") != ECP_CHECKPOINT_SCHEMA
        or manifest.get("stage") != "g3_shared_compiler"
        or int(manifest.get("next_macro", -1)) != macro
        or manifest.get("run_contract_schema") != RUN_SCHEMA
        or int(completion.get("completed_macros", -1)) < macro
        or not tensor_path.is_file()
        or tensor_path.stat().st_size != int(tensor_record.get("bytes", -1))
    ):
        raise ValueError("G3 compiler checkpoint authority changed")
    model.load_state_dict(
        load_file(str(tensor_path), device=str(device)), strict=True
    )
    model.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("G3 materializer did not freeze the shared compiler")
    return macro, {
        "schema_version": contract["schema_version"],
        "stage": contract["stage"],
        "mode": contract["mode"],
        "path": str(contract_path),
        "bytes": contract_path.stat().st_size,
        "git": dict(contract["git"]),
        "frozen_program": dict(contract["frozen_program"]),
        "model": dict(contract["model"]),
        "information_wall": dict(contract["information_wall"]),
    }


def _materialization_wall(
    policy: torch.nn.Module,
    program: torch.nn.Module,
    compiler: torch.nn.Module,
) -> dict[str, Any]:
    action_meta = [
        f"{prefix}.{name}:{type(module).__name__}"
        for root, prefix in ((policy, "policy"), (program, "program"))
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
    ]
    trainable = [
        f"{prefix}.{name}"
        for root, prefix in (
            (policy, "policy"),
            (program, "program"),
            (compiler, "compiler"),
        )
        for name, value in root.named_parameters()
        if value.requires_grad
    ]
    if (
        action_meta
        or trainable
        or policy.training
        or program.training
        or compiler.training
    ):
        raise ValueError("G3 materialization information wall changed")
    return {
        "action_meta_module_instances": action_meta,
        "action_meta_module_count": 0,
        "action_meta_parameter_count": 0,
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": 0,
        "held_action_reads": 0,
        "held_reward_reads": 0,
        "held_state_reads": 0,
    }


def _condition_spec(
    gate: Mapping[str, Any], condition: str
) -> tuple[tuple[int, ...], str]:
    if condition not in CONDITIONS:
        raise ValueError("unsupported G3 materialization condition")
    demos = tuple(map(int, gate["video_panel"][condition]))
    view = "endpoints" if condition == "first_final" else "full"
    if len(demos) != 4 or len(set(demos)) != 4:
        raise ValueError("G3 held condition is not a fixed K4 video set")
    return demos, view


def prepare_g3_materialization_runtime(args: Any) -> G3MaterializationRuntime:
    context = initialize_distributed(require_numa=True)
    if context.world_size != 1:
        raise ValueError("G3 held materialization requires one GPU")
    state = git_state(REPO_ROOT)
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError("formal G3 materialization requires clean detached authority")
    config = load_shared_compiler_config(args.config)
    gate = load_g3_gate_config(args.gate_config)
    if args.config != (REPO_ROOT / gate["training_config"]).resolve():
        raise ValueError("G3 materializer training config changed")
    seed_everything(int(config["optimization"]["seed"]), context)
    expected_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    expected_tokenizer = authority_path(
        config, "tokenizer", asset_root=args.asset_root
    )
    if (
        args.checkpoint != expected_checkpoint
        or args.source_run != expected_checkpoint.parent.parent
        or args.tokenizer_path != expected_tokenizer
    ):
        raise ValueError("G3 materializer source authority changed")

    tasks = load_g3_tasks(
        config, asset_root=args.asset_root, data_root=args.data_root
    )
    held = tuple(task for task in tasks if task.role == "target_held")
    if len(held) != 5 or tuple(task.domain_task_id for task in held) != tuple(
        map(int, config["fold"]["target_held_task_ids"])
    ):
        raise ValueError("G3 materialization held5 changed")
    target_manifest = read_json(
        authority_path(config, "target_manifest", asset_root=args.asset_root)
    )
    target_keys = {
        int(row["global_task_id"]): (str(row["suite"]), int(row["task_id"]))
        for row in target_manifest["tasks"]
        if row["split_role"] == "train"
    }
    source = stage0_source_authority(args)
    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    policy = load_policy(
        Path(source["model_path"]), source_config, context.device
    ).requires_grad_(False).eval()
    ranks = load_shared_rank_assets(
        config,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, config["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    program = build_frozen_g2_program(
        config,
        asset_root=args.asset_root,
        owners=owners,
        device=context.device,
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    compiler = SharedNativeFactorCompiler(
        owners,
        program_width=int(config["model"]["program_width"]),
        event_slots=int(config["model"]["event_slots"]),
        key_width=int(config["model"]["key_width"]),
        maximum_video_correction=float(
            config["model"]["maximum_video_correction"]
        ),
        video_score_bound=float(config["model"]["video_score_bound"]),
    ).to(context.device)
    macro, shared_contract = _load_compiler_checkpoint(
        compiler,
        run_root=args.compiler_run,
        checkpoint=args.compiler_checkpoint,
        device=context.device,
    )
    wall = _materialization_wall(policy, program, compiler)
    tokens = tokenize_stage0_languages(
        held,
        tokenizer_path=args.tokenizer_path,
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    video_store = RawTeacherVideoStore(
        tuple(task.writer_authority() for task in held),
        frame_stride=int(config["data"]["frame_stride"]),
        max_open_files=8,
    )
    demos, view = _condition_spec(gate, args.condition)
    g2 = read_json(authority_path(config, "g2_config", asset_root=args.asset_root))
    return G3MaterializationRuntime(
        config=config,
        gate=gate,
        state=state,
        source=source,
        source_config=source_config,
        held=held,
        target_keys=target_keys,
        policy=policy,
        ranks=ranks,
        owners=owners,
        program=program,
        compiler=compiler,
        compiler_macro=macro,
        shared_contract=shared_contract,
        wall=wall,
        tokens=tokens,
        video_store=video_store,
        demos=demos,
        view=view,
        query_points=int(g2["data"]["query_points"]),
        rank4_contract=derive_pi05_lora_rank(ranks.contract, rank=4),
        device=context.device,
    )
