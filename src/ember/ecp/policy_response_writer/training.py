"""Runtime and minimal real smoke for the canonical Policy-Response Writer."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from ember.ecp.contracts import build_target_owners
from ember.ecp.joint_program_primal.runtime import _load_panels, _tasks
from ember.ecp.observer_authority import load_frozen_native_observer
from ember.ecp.policy_effects import ExecutionPolicyPrefix
from ember.ecp.policy_response_writer.capture import (
    FrozenPolicyResponseVideo,
    capture_policy_response_chunk,
    merge_policy_response_chunks,
)
from ember.ecp.policy_response_writer.model import PolicyResponseEventToFactorWriter
from ember.ecp.policy_response_writer.shared_schedule import (
    _functional_panel_config,
    _selected_task_ids,
)
from ember.ecp.shared_compiler_assets import (
    authority_path,
    load_shared_compiler_config,
    load_shared_rank_assets,
)
from ember.ecp.stage0_training import load_stage0_config, tokenize_stage0_languages
from ember.pi05_lora import derive_pi05_lora_rank
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    read_json,
    write_json_atomic,
)
from ember.pi05_source_setup import (
    initialize_distributed,
    load_config as load_source_config,
    load_policy,
    load_stats,
    seed_everything,
)
from ember.writer.data import FunctionalQueryDataset, RawTeacherVideoStore
from ember.writer.functional import (
    ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    LATIN_BETA_TIME_SAMPLING_SCHEME,
    functional_lora_loss_gradient,
    prepare_frozen_writer_policy,
    writer_chain_rule_surrogate,
)


SCHEMA = "ember_ecp_policy_response_writer_v1"
RUN_SCHEMA = "ember_ecp_policy_response_writer_run_v1"
REPO_ROOT = Path(__file__).resolve().parents[4]
JOINT_PROCESS_COMPOSER_STAGE = "joint_process_composer"
COMPOSER_FUNCTIONAL_STAGE = "composer_functional_process_frozen"


@dataclass
class PolicyResponseRuntime:
    args: argparse.Namespace
    config: dict[str, Any]
    base: dict[str, Any]
    context: DistributedContext
    policy: torch.nn.Module
    stage0: torch.nn.Module
    writer: PolicyResponseEventToFactorWriter
    ranks: Any
    rank4_contract: Any
    owners: tuple[Any, ...]
    task_by_id: dict[int, Any]
    panels: dict[int, Any]
    language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]]
    video_store: RawTeacherVideoStore
    query_dataset: FunctionalQueryDataset | None
    query_processor: Pi05LiberoProcessor | None
    initialization: dict[str, object]

    def close(self) -> None:
        self.video_store.close()
        if self.query_dataset is not None:
            self.query_dataset.close()


def shared_training_stage(runtime: PolicyResponseRuntime) -> str:
    """Resolve the one configured optimizer ownership stage."""

    stage = str(
        runtime.config["optimization"]["shared"].get(
            "training_stage", JOINT_PROCESS_COMPOSER_STAGE
        )
    )
    if stage not in {JOINT_PROCESS_COMPOSER_STAGE, COMPOSER_FUNCTIONAL_STAGE}:
        raise ValueError("shared Writer training stage changed")
    return stage


def set_shared_training_mode(runtime: PolicyResponseRuntime) -> None:
    """Keep a frozen Process deterministic while training the same Composer."""

    runtime.writer.train()
    if shared_training_stage(runtime) == COMPOSER_FUNCTIONAL_STAGE:
        runtime.writer.process.eval()


def load_policy_response_config(path: Path) -> dict[str, Any]:
    config = read_json(path.resolve())
    model = config.get("model", {})
    data = config.get("data", {})
    optimization = config.get("optimization", {})
    shared = optimization.get("shared", {})
    training_stage = shared.get("training_stage", JOINT_PROCESS_COMPOSER_STAGE)
    process_weight = float(shared.get("process_weight", float("nan")))
    stage_contract = (
        training_stage == JOINT_PROCESS_COMPOSER_STAGE
        and math.isfinite(process_weight)
        and process_weight > 0.0
    ) or (
        training_stage == COMPOSER_FUNCTIONAL_STAGE
        and process_weight == 0.0
        and optimization.get("objective")
        == "composer_functional_stage_correct_cross_episode_plus_light_preservation"
    )
    training_k = tuple(map(int, data.get("training_K", (data.get("initial_K", -1),))))
    if not all(
        (
            config.get("schema_version") == SCHEMA,
            config.get("status") == "active_policy_response_event_to_factor_writer",
            model.get("target_owners") == 38,
            model.get("residual_rank") == 4,
            model.get("event_slots") == 8,
            model.get("relation_types") == 4,
            model.get("composer_relation_binding")
            == "soft_event_assignment_base_measure_signed_candidate_logits",
            model.get("composer_query_seed")
            == "parameter_free_pre_norm_rank_plus_variance_balanced_owner_family_common_language",
            model.get("composer_relative_gain_readout")
            == (
                "shared_within_target_set_relative_factor_gain_tokens_"
                "without_target_owned_output_rows"
            ),
            model.get("composer_gain_attention_scope")
            == "within_target_rank_by_ragged_group_only",
            model.get("composer_gain_initialization")
            == "g1_nonzero_relative_logit_0.1_for_first_step_direction_credit",
            model.get("composer_gain_blocks") == 1,
            model.get("process_consumer_boundary")
            == "parameter_free_pre_norm_common_and_innovation_at_prediction_memory_and_signed_score",
            model.get("causal_process_interval")
            == "deterministic_uniform_prefix_then_uniform_positive_future_offset_direct_sqrt_delta_standardized_target_with_parameter_free_encoding",
            model.get("causal_event_posterior")
            == "full_video_hard_first_final_and_prefix_first_anchored_forward_filter",
            model.get("representation_arms") == ["full"],
            data.get("frame_stride") == 5,
            data.get("supported_K") == [1, 2, 4],
            bool(training_k),
            tuple(sorted(set(training_k))) == training_k,
            set(training_k) <= {1, 2, 4},
            stage_contract,
            shared.get("process_normalizer")
            == "target_only_deterministic_8_pairs_per_fit_video",
            shared.get("process_normalizer_pairs_per_fit_video") == 8,
            shared.get("process_prediction_lr_multiplier") == 20.0,
            config.get("information_wall", {}).get("action_meta_installed") is False,
            config.get("information_wall", {}).get("wrong_training_loss") is False,
        )
    ):
        raise ValueError("invalid Policy-Response Writer config")
    return config


def _validate_launch_authority(args: argparse.Namespace) -> None:
    if args.mode != "formal":
        return
    state = git_state(REPO_ROOT)
    if (
        not git_state_is_clean_pushed_or_frozen_authority(state)
        or state.get("branch") != ""
        or state.get("upstream") is not None
    ):
        raise ValueError(
            "formal Policy-Response Writer requires detached pushed authority"
        )


def _initialize_writer(
    writer: PolicyResponseEventToFactorWriter,
    stage0: torch.nn.Module,
    kind: str,
) -> dict[str, object]:
    if kind == "component":
        return writer.initialize_from_stage0(stage0)
    if kind == "random":
        return {
            "kind": "fully_random_same_topology",
            "reused": [],
            "fresh": ["process", "causal_predictor", "composer"],
        }
    raise ValueError("unknown Policy-Response Writer initialization")


def _runtime_task_ids(
    args: argparse.Namespace, selected_ids: tuple[int, ...]
) -> tuple[int, ...]:
    if args.phase in {"smoke", "task-local"}:
        if args.task is None:
            raise ValueError(f"{args.phase} requires one task")
        return (int(args.task),)
    if args.mode == "profile" and args.task is not None:
        return (int(args.task),)
    return selected_ids


def _runtime_tasks_and_panels(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tasks: tuple[Any, ...],
    *,
    deployment_global_ids: tuple[int, ...] | None,
) -> tuple[tuple[Any, ...], dict[int, Any]]:
    task_by_id = {task.authority_id: task for task in tasks}
    if deployment_global_ids is not None:
        expected = set(deployment_global_ids)
        selected = tuple(
            sorted(
                (
                    task
                    for task in tasks
                    if task.role == "target_held" and task.domain_task_id in expected
                ),
                key=lambda task: task.domain_task_id,
            )
        )
        if tuple(task.domain_task_id for task in selected) != deployment_global_ids:
            raise ValueError("Policy-Response Writer deployment tasks changed")
        return selected, {}

    selected_ids = _selected_task_ids(config)
    runtime_ids = _runtime_task_ids(args, selected_ids)
    if not set(runtime_ids) <= set(selected_ids):
        raise ValueError("Policy-Response Writer task escaped its registered split")
    panel_config = _functional_panel_config(config, asset_root=args.asset_root)
    panels = _load_panels(panel_config, asset_root=args.asset_root)
    if not set(runtime_ids) <= set(panels):
        raise ValueError("Policy-Response Writer functional panels changed")
    return tuple(task_by_id[value] for value in runtime_ids), panels


def _functional_runtime_inputs(
    *,
    authorities: tuple[Any, ...],
    source_config: Mapping[str, Any],
    base: Mapping[str, Any],
    args: argparse.Namespace,
    context: DistributedContext,
    enabled: bool,
) -> tuple[FunctionalQueryDataset | None, Pi05LiberoProcessor | None]:
    if not enabled:
        return None, None
    dataset = FunctionalQueryDataset(
        authorities,
        demo_indices=range(50),
        action_chunk_size=int(source_config["features"]["chunk_size"]),
        max_open_files_per_worker=4,
    )
    processor = Pi05LiberoProcessor(
        load_stats(source_config, source_config["data"]["active_task_ids"]),
        authority_path(base, "tokenizer", asset_root=args.asset_root),
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    return dataset, processor


def prepare_runtime(
    args: argparse.Namespace,
    context: DistributedContext,
    *,
    deployment_global_ids: tuple[int, ...] | None = None,
) -> PolicyResponseRuntime:
    config = load_policy_response_config(args.config)
    _validate_launch_authority(args)
    base_path = (
        args.asset_root / str(config["authorities"]["base_g3_config"])
    ).resolve()
    base = load_shared_compiler_config(base_path)
    seed_everything(int(config["optimization"]["seed"]), context)
    source_config = load_source_config(
        authority_path(base, "source_base_config", asset_root=args.asset_root)
    )
    source_checkpoint = authority_path(
        base, "source_checkpoint", asset_root=args.asset_root
    )
    policy = (
        load_policy(source_checkpoint / "policy", source_config, context.device)
        .requires_grad_(False)
        .eval()
    )
    ranks = load_shared_rank_assets(
        base,
        asset_root=args.asset_root,
        held_global_ids=set(map(int, base["fold"]["target_held_task_ids"])),
        device=context.device,
    )
    owners = build_target_owners(ranks.contract)
    rank4_contract = derive_pi05_lora_rank(ranks.contract, rank=4)
    stage0 = load_frozen_native_observer(
        stage0_config=load_stage0_config(
            authority_path(base, "stage0_config", asset_root=args.asset_root)
        ),
        owners=owners,
        native_checkpoint=authority_path(
            base, "native_observer_checkpoint", asset_root=args.asset_root
        ),
        device=context.device,
        max_frames_per_call=int(config["model"]["capture_frame_chunk"]),
    )
    prepare_frozen_writer_policy(policy, ranks.contract)
    model = config["model"]
    writer = PolicyResponseEventToFactorWriter(
        owners,
        prefix_width=int(model["prefix_width"]),
        expert_width=int(model["expert_width"]),
        width=int(model["width"]),
        event_slots=int(model["event_slots"]),
        heads=int(model["attention_heads"]),
        frame_blocks=int(model["frame_blocks"]),
        event_blocks=int(model["event_blocks"]),
        composer_blocks=int(model["composer_blocks"]),
        composer_gain_blocks=int(model["composer_gain_blocks"]),
        pooling_frame_chunk=int(model["pooling_frame_chunk"]),
        task_local=args.phase == "task-local",
    ).to(context.device)
    initialization = _initialize_writer(writer, stage0, args.initialization)
    tasks = _tasks(base, args.data_root, args.asset_root)
    task_by_id = {task.authority_id: task for task in tasks}
    selected_tasks, panels = _runtime_tasks_and_panels(
        args,
        config,
        tasks,
        deployment_global_ids=deployment_global_ids,
    )
    authorities = tuple(task.writer_authority() for task in selected_tasks)
    video_store = RawTeacherVideoStore(
        authorities,
        frame_stride=int(config["data"]["frame_stride"]),
        max_open_files=4,
    )
    query_dataset, query_processor = _functional_runtime_inputs(
        authorities=authorities,
        source_config=source_config,
        base=base,
        args=args,
        context=context,
        enabled=deployment_global_ids is None,
    )
    language_tokens = tokenize_stage0_languages(
        selected_tasks,
        tokenizer_path=authority_path(base, "tokenizer", asset_root=args.asset_root),
        max_length=int(source_config["features"]["tokenizer_max_length"]),
        device=context.device,
    )
    return PolicyResponseRuntime(
        args=args,
        config=config,
        base=base,
        context=context,
        policy=policy,
        stage0=stage0,
        writer=writer,
        ranks=ranks,
        rank4_contract=rank4_contract,
        owners=owners,
        task_by_id=task_by_id,
        panels=panels,
        language_tokens=language_tokens,
        video_store=video_store,
        query_dataset=query_dataset,
        query_processor=query_processor,
        initialization=initialization,
    )


def capture_video(
    runtime: PolicyResponseRuntime, *, task_id: int, video_demo: int
) -> tuple[FrozenPolicyResponseVideo, dict[str, Any]]:
    raw = runtime.video_store.load(task_id, video_demo)
    frames = torch.from_numpy(raw.frames).to(
        device=runtime.context.device, non_blocking=True
    )
    tokens, masks = runtime.language_tokens[task_id]
    encoder = runtime.stage0.encoder
    language = encoder.embed_language_conditions(runtime.policy, tokens)
    chunk_size = int(runtime.config["model"]["capture_frame_chunk"])
    chunks = []
    started = time.monotonic()
    for start in range(0, frames.shape[0], chunk_size):
        stop = min(start + chunk_size, frames.shape[0])
        embeddings, padding = encoder.prepare_frame_prefix(
            policy=runtime.policy,
            frames=frames[start:stop],
            frame_condition_ids=torch.zeros(
                stop - start,
                dtype=torch.long,
                device=runtime.context.device,
            ),
            language_embeddings=language,
            language_mask=masks,
        )
        chunks.append(
            capture_policy_response_chunk(
                policy=runtime.policy,
                owners=runtime.owners,
                prefix=ExecutionPolicyPrefix(embeddings, padding),
                fixed_probe=encoder.fixed_suffix_noise,
                start_frame=start,
            )
        )
    positions = torch.from_numpy(raw.frame_indices).to(
        device=runtime.context.device, dtype=torch.float32
    ) / max(raw.raw_frame_count - 1, 1)
    result = merge_policy_response_chunks(chunks, frame_positions=positions)
    return result, {
        "task_id": task_id,
        "video_demo": video_demo,
        "raw_frames": raw.raw_frame_count,
        "sampled_frames": result.frame_count,
        "capture_seconds": time.monotonic() - started,
        "capture_chunks": len(chunks),
    }


def functional_panel_batch(
    runtime: PolicyResponseRuntime,
    *,
    task_id: int,
    panel_name: str,
    visit_index: int,
    rows: int | None = None,
) -> tuple[dict[str, Any], Any]:
    if runtime.query_dataset is None or runtime.query_processor is None:
        raise ValueError("Policy-Response Writer deployment has no functional data")
    panel = runtime.panels[task_id]
    visits = panel.panel_a if panel_name == "a" else panel.panel_b
    visit = visits[visit_index % len(visits)]
    count = len(visit.action_demos) if rows is None else int(rows)
    if not 0 < count <= len(visit.action_demos):
        raise ValueError("Policy-Response Writer functional row count changed")
    selected = []
    episode_rows = runtime.query_dataset.task_episode_rows[task_id]
    frame_index = runtime.query_dataset.frame_index
    for demo, frame in zip(
        visit.action_demos[:count], visit.action_frames[:count], strict=True
    ):
        index = int(episode_rows[demo][frame])
        if frame_index[index] != (task_id, demo, frame):
            raise ValueError("Policy-Response Writer panel row pairing changed")
        selected.append(index)
    batch = runtime.query_processor.training_batch(
        default_collate([runtime.query_dataset[index] for index in selected])
    )
    return batch, visit


def _gradient_norms(module: torch.nn.Module) -> dict[str, float]:
    groups = {
        "frame": "process.frame_blocks",
        "event": "process.events",
        "composer": "composer",
        "composer_relation": "composer.relation_embedding",
        "process_prediction": "process.prediction",
    }
    output = {}
    for label, prefix in groups.items():
        squares = [
            parameter.grad.detach().float().square().sum()
            for name, parameter in module.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        output[label] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    direction_squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith("composer.")
        and not name.startswith("composer.gain_readout.")
        and parameter.grad is not None
    ]
    gain_squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith("composer.gain_readout.") and parameter.grad is not None
    ]
    output["composer_direction"] = (
        float(torch.stack(direction_squares).sum().sqrt())
        if direction_squares
        else 0.0
    )
    output["composer_gain"] = (
        float(torch.stack(gain_squares).sum().sqrt()) if gain_squares else 0.0
    )
    output_squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith("composer.gain_readout.output.")
        and parameter.grad is not None
    ]
    conditioner_squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith("composer.gain_readout.")
        and not name.startswith("composer.gain_readout.output.")
        and parameter.grad is not None
    ]
    output["composer_gain_output"] = (
        float(torch.stack(output_squares).sum().sqrt()) if output_squares else 0.0
    )
    output["composer_gain_conditioner"] = (
        float(torch.stack(conditioner_squares).sum().sqrt())
        if conditioner_squares
        else 0.0
    )
    return output


def _cuda_peak(runtime: PolicyResponseRuntime) -> dict[str, int]:
    torch.cuda.synchronize(runtime.context.device)
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(runtime.context.device),
        "reserved_bytes": torch.cuda.max_memory_reserved(runtime.context.device),
    }


def _validate_smoke_graph(
    runtime: PolicyResponseRuntime,
    *,
    functional: torch.Tensor,
    process_loss: torch.Tensor | None,
    functional_gradients: Mapping[str, float],
    process_gradients: Mapping[str, float],
    generated_tensors: int,
) -> None:
    stage = shared_training_stage(runtime)
    required = [
        functional_gradients["composer"],
        functional_gradients["composer_direction"],
        functional_gradients["composer_gain"],
    ]
    if stage == JOINT_PROCESS_COMPOSER_STAGE:
        required.extend(
            (
                functional_gradients["frame"],
                functional_gradients["event"],
                process_gradients["frame"],
                process_gradients["event"],
                process_gradients["process_prediction"],
            )
        )
    frozen_process_changed = stage == COMPOSER_FUNCTIONAL_STAGE and (
        runtime.writer.process.training
        or any(value.requires_grad for value in runtime.writer.process.parameters())
        or any(
            value != 0.0
            for value in (
                functional_gradients["frame"],
                functional_gradients["event"],
                process_gradients["frame"],
                process_gradients["event"],
                process_gradients["process_prediction"],
            )
        )
    )
    process_contract_changed = (
        stage == JOINT_PROCESS_COMPOSER_STAGE and process_loss is None
    ) or (stage == COMPOSER_FUNCTIONAL_STAGE and process_loss is not None)
    invalid = any(
        (
            not math.isfinite(float(functional)),
            process_loss is not None
            and not math.isfinite(float(process_loss.detach())),
            min(required) <= 0,
            frozen_process_changed,
            process_contract_changed,
            any(
                parameter.grad is not None for parameter in runtime.policy.parameters()
            ),
            any(
                parameter.grad is not None for parameter in runtime.stage0.parameters()
            ),
            generated_tensors != 76,
        )
    )
    if invalid:
        raise RuntimeError(
            "Policy-Response Writer real smoke did not connect the graph"
        )


def _writer_chain_backward(
    runtime: PolicyResponseRuntime,
    *,
    video: FrozenPolicyResponseVideo,
    leaf_gradients: Mapping[str, torch.Tensor],
    weight: float = 1.0,
) -> int:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.writer(
            (video,),
            s_ref=runtime.ranks.s_ref,
            representation=runtime.args.representation,
        )
        state = runtime.writer.materialize(
            output,
            carrier_state=runtime.ranks.carrier_rank12,
            rank4_contract=runtime.rank4_contract,
            rank16_contract=runtime.ranks.contract,
            canonicalize=False,
        )
        surrogate = writer_chain_rule_surrogate(state, leaf_gradients) * float(weight)
    surrogate.backward()
    return len(state)


def _initial_factor_state(output: Any) -> dict[str, bool]:
    return {
        "input_factor_nonzero": any(
            torch.count_nonzero(value).item() > 0 for value in output.residual.a
        ),
        "output_factor_nonzero": any(
            torch.count_nonzero(value).item() > 0 for value in output.residual.b
        ),
    }


def run_smoke(runtime: PolicyResponseRuntime) -> dict[str, Any]:
    if runtime.context.world_size != 1:
        raise ValueError("Policy-Response Writer smoke is single-GPU")
    task_id = int(runtime.args.task)
    demo = (
        int(runtime.args.video_demo)
        if runtime.args.video_demo is not None
        else int(runtime.panels[task_id].program_video_demos[0])
    )
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    video, capture = capture_video(runtime, task_id=task_id, video_demo=demo)
    phase_memory = {"capture": _cuda_peak(runtime)}
    stage = shared_training_stage(runtime)
    runtime.writer.requires_grad_(False)
    if stage == JOINT_PROCESS_COMPOSER_STAGE:
        runtime.writer.requires_grad_(True)
    else:
        runtime.writer.composer.requires_grad_(True)
    set_shared_training_mode(runtime)
    # The frozen policy only needs the generated LoRA values.  Keeping the
    # Writer graph alive during its much larger functional forward needlessly
    # overlaps two graphs.  Recompute the deterministic Writer once after the
    # detached leaf gradient is known, which preserves the exact first-order
    # chain rule while materially lowering peak memory.
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        leaf_output = runtime.writer(
            (video,),
            s_ref=runtime.ranks.s_ref,
            representation=runtime.args.representation,
        )
        leaf_state = runtime.writer.materialize(
            leaf_output,
            carrier_state=runtime.ranks.carrier_rank12,
            rank4_contract=runtime.rank4_contract,
            rank16_contract=runtime.ranks.contract,
            canonicalize=False,
        )
    initial_factor_state = _initial_factor_state(leaf_output)
    phase_memory["writer_leaf_forward"] = _cuda_peak(runtime)
    batch, visit = functional_panel_batch(
        runtime,
        task_id=task_id,
        panel_name="a",
        visit_index=0,
        rows=int(runtime.config["smoke"]["functional_rows"]),
    )
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    functional, details, leaf_gradients = functional_lora_loss_gradient(
        runtime.policy,
        leaf_state,
        runtime.ranks.contract,
        batch=batch,
        policy_rng_seed=visit.policy_rng_seed,
        policy_rng_device=runtime.context.device,
        flow_time_sampling_scheme=LATIN_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=ANTITHETIC_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_microbatch_size=int(runtime.config["smoke"]["functional_microbatch"]),
        collect_policy_details=False,
    )
    phase_memory["functional_policy_gradient"] = _cuda_peak(runtime)
    if details:
        raise RuntimeError("Policy-Response Writer smoke collected policy diagnostics")
    del leaf_output, leaf_state
    torch.cuda.reset_peak_memory_stats(runtime.context.device)
    generated_tensors = _writer_chain_backward(
        runtime, video=video, leaf_gradients=leaf_gradients
    )
    phase_memory["initial_chain_rule_backward"] = _cuda_peak(runtime)
    functional_gradients = _gradient_norms(runtime.writer)
    runtime.writer.zero_grad(set_to_none=True)
    if stage == JOINT_PROCESS_COMPOSER_STAGE:
        cutoff = max(
            int(runtime.config["model"]["event_slots"]),
            min(video.frame_count - 1, video.frame_count // 2),
        )
        future_offset = min(8, video.frame_count - cutoff)
        torch.cuda.reset_peak_memory_stats(runtime.context.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            process_loss = runtime.writer.causal_prediction_loss(
                (video,),
                cutoffs=((cutoff,),),
                future_offsets=(future_offset,),
                representation=runtime.args.representation,
            )
        process_loss.backward()
        phase_memory["causal_process_backward"] = _cuda_peak(runtime)
        process_gradients = _gradient_norms(runtime.writer)
    else:
        cutoff = None
        future_offset = None
        process_loss = None
        process_gradients = _gradient_norms(runtime.writer)
    _validate_smoke_graph(
        runtime,
        functional=functional,
        process_loss=process_loss,
        functional_gradients=functional_gradients,
        process_gradients=process_gradients,
        generated_tensors=generated_tensors,
    )
    return {
        "schema_version": RUN_SCHEMA,
        "phase": "smoke",
        "representation": runtime.args.representation,
        "training_stage": stage,
        "process_objective_active": stage == JOINT_PROCESS_COMPOSER_STAGE,
        "capture": capture,
        "frozen_evidence_tensor_bytes": video.tensor_bytes,
        "functional_loss": float(functional),
        "process_loss": (
            float(process_loss.detach()) if process_loss is not None else None
        ),
        "causal_future_offset": future_offset,
        "functional_gradient_norms": functional_gradients,
        "initial_factor_state": initial_factor_state,
        "process_gradient_norms": process_gradients,
        "generated_tensors": generated_tensors,
        "targets": len(runtime.owners),
        "mobile_rank": runtime.rank4_contract.rank,
        "complete_rank": runtime.ranks.contract.rank,
        "initialization": runtime.initialization,
        "phase_cuda_memory": phase_memory,
        "max_phase_cuda_allocated_bytes": max(
            row["allocated_bytes"] for row in phase_memory.values()
        ),
        "max_phase_cuda_reserved_bytes": max(
            row["reserved_bytes"] for row in phase_memory.values()
        ),
    }


def run(args: argparse.Namespace) -> None:
    if args.phase == "materialize":
        from ember.ecp.policy_response_writer.materialization import (
            materialize_writer_evaluation_bank,
        )

        payload = materialize_writer_evaluation_bank(args)
        print(
            f"sealed {len(payload['tasks'])} {payload['arm']} rank16 adapters",
            flush=True,
        )
        return
    context = initialize_distributed(
        require_numa=args.mode == "formal",
        defer_process_group=True,
    )
    runtime: PolicyResponseRuntime | None = None
    try:
        runtime = prepare_runtime(args, context)
        torch.cuda.reset_peak_memory_stats(context.device)
        if args.phase == "smoke":
            result = run_smoke(runtime)
            filename = "smoke.json"
        elif args.phase == "task-local":
            from ember.ecp.policy_response_writer.tasklocal import run_task_local

            result = run_task_local(runtime)
            filename = "result.json"
        else:
            from ember.ecp.policy_response_writer.shared import run_shared

            result = run_shared(runtime)
            filename = "result.json"
        if context.is_main:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(args.output_dir / filename, result)
            if args.mode == "formal":
                write_json_atomic(
                    args.output_dir / "completion.json",
                    {
                        "schema_version": result["schema_version"],
                        "status": result.get("status", "complete"),
                        "phase": result["phase"],
                        "task": result.get("task"),
                        "optimizer_steps": result.get("optimizer_steps"),
                        "result": filename,
                    },
                )
            print(result, flush=True)
        barrier(context)
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("smoke", "task-local", "shared", "materialize"),
        required=True,
    )
    parser.add_argument("--task", type=int)
    parser.add_argument("--video-demo", type=int)
    parser.add_argument("--representation", choices=("full",), default="full")
    parser.add_argument(
        "--initialization", choices=("component", "random"), default="component"
    )
    parser.add_argument("--mode", choices=("profile", "formal"), default="profile")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--cache-replication-budget-gib",
        type=float,
        default=0.0,
        help=(
            "maximum additional host memory for outcome-independent frozen "
            "evidence replicas used by shared task scheduling"
        ),
    )
    parser.add_argument(
        "--shared-evidence-cache-root",
        type=Path,
        help=(
            "task-scoped node-local safetensors mmap root that lets every "
            "same-node rank access one physical copy of frozen video evidence"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--writer-run", type=Path)
    parser.add_argument("--writer-checkpoint", type=Path)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("config", "asset_root", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for name in (
        "resume",
        "evaluation_config",
        "writer_run",
        "writer_checkpoint",
        "shared_evidence_cache_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args
