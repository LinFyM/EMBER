"""One-task model, data, and frozen-authority assembly for G1."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ember.batched_lora import BatchedLoRAInference
from ember.ecp.contracts import TargetOwner, build_target_owners
from ember.ecp.g1_assets import (
    G1_MEMBER_NAMES,
    G1RankAssets,
    G1TaskAssets,
    authority_path,
    load_g1_config,
    load_g1_rank_assets,
    load_g1_task_assets,
)
from ember.ecp.g1_initialization import (
    cache_native_video_readout,
    initialize_oracle_from_reference,
)
from ember.ecp.g1_objective import (
    G1EffectBank,
    VerifiedMemberObjective,
    build_verified_member_objective,
    load_g1_effect_bank,
    verified_member_validity,
)
from ember.ecp.g1_queries import calibrate_policy_sensitivity, functional_batch
from ember.ecp.g1_run_contract import build_run_contract, publish_run_contract
from ember.ecp.g1_video import (
    G1VideoRuntime,
    prepare_pass_a,
    prepare_pass_b_readout,
    pure_native_inventory,
)
from ember.ecp.native_factors import TaskLocalNativeFactorOracle, native_capture_modes
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    residual_lora_state,
)
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    PolicyEffectResponse,
    capture_policy_effect_response,
)
from ember.ecp.stage0 import ECPStage0Model
from ember.ecp.stage0_training import load_stage0_config
from ember.lora import LoRAContract
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_config, load_policy, load_stats
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
G1_RUN_SCHEMA = "ember_ecp_native_factor_g1_task_run_v1"
G1_CHECKPOINT_SCHEMA = "ember_ecp_native_factor_g1_checkpoint_v1"


@dataclass
class G1Runtime:
    args: argparse.Namespace
    config: dict[str, Any]
    task: G1TaskAssets
    owners: tuple[TargetOwner, ...]
    ranks: G1RankAssets
    policy: torch.nn.Module
    stage0: ECPStage0Model
    lora: BatchedLoRAInference
    video_store: RawTeacherVideoStore
    query_dataset: FunctionalQueryDataset
    query_processor: Pi05LiberoProcessor
    video: G1VideoRuntime
    effect_bank: G1EffectBank
    effect_objective: VerifiedMemberObjective
    oracle: TaskLocalNativeFactorOracle
    optimizer: torch.optim.Optimizer
    sensitivity_raw: torch.Tensor
    sensitivity_weights: torch.Tensor
    responsibilities: torch.Tensor
    start_step: int
    metrics_rows: int
    run_contract: dict[str, Any]

    def close(self) -> None:
        self.lora.close()
        self.video_store.close()
        self.query_dataset.close()


def _device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("G1 requires one explicitly selected CUDA device per task")
    torch.cuda.set_device(device)
    return device


def _optimizer(
    oracle: TaskLocalNativeFactorOracle, config: Mapping[str, Any]
) -> torch.optim.AdamW:
    cell = config["optimization"]["optimizer"]
    return torch.optim.AdamW(
        [
            {
                "params": [oracle.input_logits, oracle.output_logits],
                "lr": float(cell["selection_lr"]),
            },
            {
                "params": [oracle.rank_queries, oracle.event_logits],
                "lr": float(cell["program_lr"]),
            },
            {"params": [oracle.scale_logits], "lr": float(cell["scale_lr"])},
        ],
        betas=tuple(cell["betas"]),
        eps=float(cell["eps"]),
        weight_decay=float(cell["weight_decay"]),
    )


def capture_effect_response(
    *, runtime: G1Runtime, state: Mapping[str, torch.Tensor]
) -> PolicyEffectResponse:
    microbatch = int(runtime.config["optimization"]["effect_microbatch_states"])
    values: list[PolicyEffectResponse] = []
    for start in range(0, runtime.effect_bank.state_count, microbatch):
        stop = min(start + microbatch, runtime.effect_bank.state_count)
        indices = torch.arange(start, stop, device=runtime.args.torch_device)
        values.append(
            capture_policy_effect_response(
                policy=runtime.policy,
                observer=runtime.stage0.encoder.observer,
                lora=runtime.lora,
                state=state,
                prefix=ExecutionPolicyPrefix(
                    embeddings=runtime.effect_bank.prefix.embeddings.index_select(
                        0, indices
                    ),
                    padding=runtime.effect_bank.prefix.padding.index_select(0, indices),
                ),
                suffix_noise=runtime.effect_bank.suffix_noise.index_select(0, indices),
                denoising_steps=int(runtime.config["optimization"]["denoising_steps"]),
            )
        )
    response = PolicyEffectResponse(
        owner=torch.cat([value.owner for value in values]),
        flow=torch.cat([value.flow for value in values]),
        action=torch.cat([value.action for value in values]),
    )
    if response.owner.shape != runtime.effect_bank.source.owner.shape:
        raise ValueError("G1 microbatched policy-effect response changed")
    return response


def candidate_states(
    runtime: G1Runtime, *, canonicalize: bool
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    residual = runtime.oracle((runtime.video.readout,), s_ref=runtime.ranks.s_ref)
    residual_state = residual_lora_state(
        residual,
        derive_pi05_lora_rank(runtime.ranks.contract, rank=4),
        canonicalize=canonicalize,
    )
    complete_state = compose_rank12_plus_rank4(
        carrier_state=runtime.ranks.carrier_rank12,
        residual_state=residual_state,
        rank16_contract=runtime.ranks.contract,
    )
    return residual_state, complete_state


def prepare_runtime(args: argparse.Namespace) -> G1Runtime:
    device = _device(args)
    args.torch_device = device
    config = load_g1_config(args.config)
    repository = git_state(REPO_ROOT)
    if args.mode == "formal" and (
        not git_state_is_clean_pushed_or_frozen_authority(repository)
        or repository.get("branch") != ""
        or repository.get("upstream") is not None
    ):
        raise ValueError("formal G1 requires a clean detached origin/main authority")
    seed = int(config["optimization"]["seed"]) + args.task_ordinal
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tasks = load_g1_task_assets(
        config, asset_root=args.asset_root, data_root=args.data_root
    )
    selected = [task for task in tasks if task.ordinal == args.task_ordinal]
    if len(selected) != 1:
        raise ValueError("G1 task ordinal is outside held5")
    task = selected[0]
    ranks = load_g1_rank_assets(
        config, tasks, asset_root=args.asset_root, device=device
    )
    owners = build_target_owners(ranks.contract)

    source_config = load_config(
        authority_path(config, "source_base_config", asset_root=args.asset_root)
    )
    source_checkpoint = authority_path(
        config, "source_checkpoint", asset_root=args.asset_root
    )
    policy = load_policy(source_checkpoint / "policy", source_config, device)
    policy.requires_grad_(False).eval()
    stage0_config = load_stage0_config(
        authority_path(config, "stage0_config", asset_root=args.asset_root)
    )
    stage0 = load_frozen_native_observer(
        stage0_config=stage0_config,
        owners=owners,
        native_checkpoint=authority_path(
            config, "native_observer_checkpoint", asset_root=args.asset_root
        ),
        device=device,
        max_frames_per_call=int(config["video"]["frame_chunk_size"]),
    )
    video_store = RawTeacherVideoStore(
        (task.video_authority,), frame_stride=int(config["video"]["frame_stride"])
    )
    tokenizer = Pi05TeacherPrefixTokenizer(
        authority_path(config, "tokenizer", asset_root=args.asset_root),
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    teacher_demo = int(config["video"]["teacher_demo_index"])
    (
        frames,
        tokens,
        masks,
        process,
        posterior,
        frame_indices,
        raw_frames,
    ) = prepare_pass_a(
        policy=policy,
        stage0=stage0,
        store=video_store,
        task=task,
        tokenizer=tokenizer,
        teacher_demo=teacher_demo,
        device=device,
    )

    prepare_frozen_writer_policy(policy, ranks.contract)
    lora = BatchedLoRAInference(policy, ranks.contract)
    capture_modes = native_capture_modes(policy, owners)
    readout = cache_native_video_readout(
        prepare_pass_b_readout(
            policy=policy,
            stage0=stage0,
            owners=owners,
            frames=frames,
            tokens=tokens,
            masks=masks,
            process=process,
            posterior=posterior,
            chunk_size=int(config["video"]["frame_chunk_size"]),
        )
    )
    video = G1VideoRuntime(
        readout=readout,
        teacher_demo_index=teacher_demo,
        raw_frame_count=raw_frames,
        sampled_frame_indices=frame_indices,
    )
    oracle = TaskLocalNativeFactorOracle(
        owners,
        frame_counts=(readout.frame_count,),
        event_slots=8,
        program_width=128,
        initialization_seed=seed,
    ).to(device)
    initialization_cell = config["optimization"]["initialization"]
    if args.resume is None:
        reference_member = str(initialization_cell["reference_member"])
        reference_index = G1_MEMBER_NAMES.index(reference_member)
        initialization_report = initialize_oracle_from_reference(
            oracle=oracle,
            video=readout,
            owners=owners,
            contract=ranks.contract,
            reference=ranks.reference_rank4[task.ordinal][reference_index],
            s_ref=ranks.s_ref,
            relative_singular_threshold=float(
                initialization_cell["relative_singular_threshold"]
            ),
            probability_floor_mass=float(
                initialization_cell["probability_floor_mass"]
            ),
            reference_member=reference_member,
        )
    else:
        initialization_report = read_json(args.output_dir / "run_contract.json")[
            "native_factor_initialization"
        ]
    optimizer = _optimizer(oracle, config)

    effect_bank = load_g1_effect_bank(task.effect_manifest, device=device)
    validity = verified_member_validity(effect_bank, task.initial_success)
    effect_objective = build_verified_member_objective(effect_bank, validity)
    query_dataset = FunctionalQueryDataset(
        (task.video_authority,),
        demo_indices=tuple(map(int, config["functional_query"]["demo_indices"])),
        action_chunk_size=int(config["functional_query"]["action_chunk_size"]),
        max_open_files_per_worker=1,
    )
    query_processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        authority_path(config, "tokenizer", asset_root=args.asset_root),
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    references = ranks.reference_rank4[task.ordinal]
    if args.resume is None:
        calibration_batch = functional_batch(
            dataset=query_dataset,
            processor=query_processor,
            task=task,
            config=config,
            step=0xCA1B,
        )
        sensitivity_raw, sensitivity_weights = calibrate_policy_sensitivity(
            policy=policy,
            ranks=ranks,
            references=references,
            owners=owners,
            batch=calibration_batch,
            config=config,
            task=task,
        )
    else:
        retained = read_json(args.output_dir / "run_contract.json")[
            "policy_sensitivity"
        ]
        sensitivity_raw = torch.tensor(
            retained["raw"], dtype=torch.float32, device=device
        )
        sensitivity_weights = torch.tensor(
            retained["family_balanced_weights"],
            dtype=torch.float32,
            device=device,
        )
    pure_native = pure_native_inventory(
        policy=policy,
        stage0=stage0,
        oracle=oracle,
        capture_modes=capture_modes,
    )
    run_contract = build_run_contract(
        args=args,
        config=config,
        task=task,
        ranks=ranks,
        video=video,
        pure_native=pure_native,
        initialization=initialization_report,
        sensitivity_raw=sensitivity_raw,
        sensitivity_weights=sensitivity_weights,
        repo_root=REPO_ROOT,
        schema=G1_RUN_SCHEMA,
    )
    publish_run_contract(args=args, contract=run_contract)
    return G1Runtime(
        args=args,
        config=config,
        task=task,
        owners=owners,
        ranks=ranks,
        policy=policy,
        stage0=stage0,
        lora=lora,
        video_store=video_store,
        query_dataset=query_dataset,
        query_processor=query_processor,
        video=video,
        effect_bank=effect_bank,
        effect_objective=effect_objective,
        oracle=oracle,
        optimizer=optimizer,
        sensitivity_raw=sensitivity_raw,
        sensitivity_weights=sensitivity_weights,
        responsibilities=effect_objective.reliability.detach().clone(),
        start_step=0,
        metrics_rows=0,
        run_contract=run_contract,
    )
