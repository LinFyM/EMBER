"""Pure-Native Pass A and chunked target-native Pass B for G1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ember.ecp.contracts import TargetOwner
from ember.ecp.g1_assets import G1TaskAssets
from ember.ecp.native_factors import (
    NativeVideoReadout,
    TaskLocalNativeFactorOracle,
    capture_native_target_chunk,
)
from ember.ecp.policy_effects import ExecutionPolicyPrefix
from ember.ecp.stage0 import ECPStage0Model
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.writer.data import RawTeacherVideoStore
from ember.writer.meta_lora import MetaLoRAProjection, MetaLoRAStack


@dataclass(frozen=True)
class G1VideoRuntime:
    readout: NativeVideoReadout
    teacher_demo_index: int
    raw_frame_count: int
    sampled_frame_indices: tuple[int, ...]


def pure_native_inventory(
    *,
    policy: torch.nn.Module,
    stage0: ECPStage0Model,
    oracle: TaskLocalNativeFactorOracle,
    capture_modes: Sequence[str],
) -> dict[str, Any]:
    action_meta_modules = [
        f"{name}:{type(module).__name__}"
        for root, prefix in ((policy, "policy"), (stage0, "stage0"))
        for name, module in root.named_modules()
        if isinstance(module, (MetaLoRAStack, MetaLoRAProjection))
        for name in (f"{prefix}.{name}",)
    ]
    policy_trainable = [
        name for name, value in policy.named_parameters() if value.requires_grad
    ]
    stage0_trainable = [
        name for name, value in stage0.named_parameters() if value.requires_grad
    ]
    oracle_trainable = [
        name for name, value in oracle.named_parameters() if value.requires_grad
    ]
    if (
        action_meta_modules
        or policy_trainable
        or stage0_trainable
        or set(capture_modes) != {"identity_lora_base_layer"}
        or set(oracle_trainable)
        != {
            "rank_queries",
            "event_logits",
            "input_logits",
            "output_logits",
            "scale_logits",
        }
    ):
        raise ValueError("G1 pure-Native Stage 0 or trainable wall changed")
    return {
        "loader": "load_frozen_native_observer",
        "action_meta_argument": None,
        "install_action_meta_lora": False,
        "action_meta_module_instances": action_meta_modules,
        "action_meta_module_count": len(action_meta_modules),
        "action_meta_parameter_count": 0,
        "policy_trainable_parameters": policy_trainable,
        "policy_trainable_parameter_count": 0,
        "stage0_trainable_parameters": stage0_trainable,
        "stage0_trainable_parameter_count": 0,
        "oracle_trainable_parameters": oracle_trainable,
        "oracle_trainable_parameter_count": sum(
            value.numel() for value in oracle.parameters() if value.requires_grad
        ),
        "native_capture_modes": list(capture_modes),
    }


def prepare_pass_a(
    *,
    policy: torch.nn.Module,
    stage0: ECPStage0Model,
    store: RawTeacherVideoStore,
    task: G1TaskAssets,
    tokenizer: Pi05TeacherPrefixTokenizer,
    teacher_demo: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    int,
]:
    video = store.load(task.global_task_id, teacher_demo)
    frames = torch.from_numpy(video.frames).to(device=device, non_blocking=True)
    tokens, masks, _ = tokenizer([task.language])
    offsets = torch.tensor([0, frames.shape[0]], dtype=torch.long, device=device)
    condition_ids = torch.zeros(frames.shape[0], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = stage0.encoder(
            policy=policy,
            frames=frames,
            video_offsets=offsets,
            frame_condition_ids=condition_ids,
            language_tokens=tokens,
            language_mask=masks,
            action_meta_lora=None,
            install_action_meta_lora=False,
        )
    process = encoded.process[0].detach().float().clone()
    posterior = encoded.state_posterior[0, : frames.shape[0]].detach().float().clone()
    if process.shape != (8, 38, 128) or posterior.shape != (frames.shape[0], 8):
        raise ValueError("G1 Pass-A Program or canonical assignment changed")
    return (
        frames,
        tokens,
        masks,
        process,
        posterior,
        tuple(map(int, video.frame_indices.tolist())),
        int(video.raw_frame_count),
    )


def prepare_pass_b_readout(
    *,
    policy: torch.nn.Module,
    stage0: ECPStage0Model,
    owners: tuple[TargetOwner, ...],
    frames: torch.Tensor,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    process: torch.Tensor,
    posterior: torch.Tensor,
    chunk_size: int,
) -> NativeVideoReadout:
    language = stage0.encoder.embed_language_conditions(policy, tokens)
    fixed_probe = stage0.encoder.fixed_suffix_noise.detach()

    def prefix(start: int, stop: int) -> ExecutionPolicyPrefix:
        embeddings, padding = stage0.encoder.prepare_frame_prefix(
            policy=policy,
            frames=frames[start:stop],
            frame_condition_ids=torch.zeros(
                stop - start, dtype=torch.long, device=frames.device
            ),
            language_embeddings=language,
            language_mask=masks,
        )
        return ExecutionPolicyPrefix(embeddings=embeddings, padding=padding)

    last = int(frames.shape[0]) - 1
    final_chunk = capture_native_target_chunk(
        policy=policy,
        owners=owners,
        prefix=prefix(last, last + 1),
        fixed_probe=fixed_probe,
        start_frame=last,
    )
    final_outputs = tuple(value[0].detach() for value in final_chunk.outputs)

    def chunks():
        for start in range(0, int(frames.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(frames.shape[0]))
            yield capture_native_target_chunk(
                policy=policy,
                owners=owners,
                prefix=prefix(start, stop),
                fixed_probe=fixed_probe,
                start_frame=start,
            )

    return NativeVideoReadout(
        frame_count=int(frames.shape[0]),
        process=process,
        state_posterior=posterior,
        final_outputs=final_outputs,
        chunks=chunks,
    )
