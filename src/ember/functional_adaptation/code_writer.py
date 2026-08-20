"""One-pass language/video Writer over a frozen functional LoRA decoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.functional_adaptation.decoder import FunctionalAdapterDecoder
from ember.functional_adaptation.inference import (
    FunctionalCodePosterior,
    LanguageVideoCodeInference,
)
from ember.lora import LoRAContract, identity_lora_state
from ember.writer.video_program import (
    LanguageAxialProcessFeatures,
    Pi05LanguageAxialEncoder,
)


class FunctionalCodeWriterError(RuntimeError):
    """Raised when the one-pass code Writer leaves its frozen-decoder contract."""


@dataclass(frozen=True)
class FunctionalCodeWriterOutput:
    posterior: FunctionalCodePosterior
    language_adapter: Mapping[str, torch.Tensor]
    video_adapter: Mapping[str, torch.Tensor]
    combined_adapter: Mapping[str, torch.Tensor]


def load_fixed_decoder(
    *,
    checkpoint: Path,
    contract: LoRAContract,
    config: Mapping[str, Any],
    device: torch.device,
) -> FunctionalAdapterDecoder:
    decoder_config = config["decoder"]
    decoder = FunctionalAdapterDecoder(
        contract,
        identity_lora_state(contract, device=device),
        code_width=int(decoder_config["production_code_width"]),
        address_width=int(decoder_config["address_width"]),
        hidden_width=int(decoder_config["hidden_width"]),
        initialization_seed=int(decoder_config["initialization_seed"]),
    ).to(device)
    state = load_file(str(checkpoint.resolve()), device=str(device))
    prefix = "decoder."
    decoder_state = {
        name.removeprefix(prefix): value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not decoder_state:
        raise FunctionalCodeWriterError("functional decoder checkpoint is incomplete")
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.requires_grad_(False)
    decoder.eval()
    return decoder


def build_process_feature_encoder(
    policy: torch.nn.Module, config: Mapping[str, Any]
) -> Pi05LanguageAxialEncoder:
    bridge = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
    if bridge is None:
        raise FunctionalCodeWriterError("PI0.5 policy lost its joint backbone")
    settings = config["code_inference"]["feature_encoder"]
    if int(settings["text_meta_lora_rank"]) or int(settings["vl_meta_lora_rank"]):
        raise FunctionalCodeWriterError(
            "functional Writer must read frozen native text/VLM features"
        )
    return Pi05LanguageAxialEncoder(
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        image_width=int(settings["image_width"]),
        expert_width=int(settings["expert_width"]),
        program_width=int(config["code_inference"]["feature_width"]),
        text_meta_lora_rank=int(settings["text_meta_lora_rank"]),
        vl_meta_lora_rank=int(settings["vl_meta_lora_rank"]),
        action_meta_lora_rank=int(settings["action_meta_lora_rank"]),
        patch_grounding_heads=int(settings["patch_grounding_heads"]),
        max_frames_per_encoder_call=int(settings["max_frames_per_encoder_call"]),
        action_horizon=50,
        padded_action_dim=32,
        initialization_seed=int(config["code_inference"]["initialization_seed"]),
        activation_checkpointing=bool(settings["activation_checkpointing"]),
        raw_visual_projection=True,
    )


class FunctionalCodeWriter(torch.nn.Module):
    """Map exact language and ordered action-hidden videos to one complete LoRA."""

    def __init__(
        self,
        *,
        feature_encoder: Pi05LanguageAxialEncoder,
        code_inference: LanguageVideoCodeInference,
        fixed_decoder: FunctionalAdapterDecoder,
    ) -> None:
        super().__init__()
        self.feature_encoder = feature_encoder
        self.code_inference = code_inference
        self.fixed_decoder = fixed_decoder
        if any(parameter.requires_grad for parameter in fixed_decoder.parameters()):
            raise FunctionalCodeWriterError("functional decoder must be frozen")

    @classmethod
    def from_policy(
        cls,
        *,
        policy: torch.nn.Module,
        config: Mapping[str, Any],
        contract: LoRAContract,
        decoder_checkpoint: Path,
        device: torch.device,
    ) -> FunctionalCodeWriter:
        settings = config["code_inference"]
        inference = LanguageVideoCodeInference(
            feature_width=int(settings["feature_width"]),
            hidden_width=int(settings["hidden_width"]),
            code_width=int(settings["code_width"]),
            attention_heads=int(settings["attention_heads"]),
            temporal_layers=int(settings["temporal_layers"]),
            phase_queries=int(settings["phase_queries"]),
            event_queries=int(settings["event_queries"]),
            dropout=float(settings["dropout"]),
            initialization_seed=int(settings["initialization_seed"]),
        )
        writer = cls(
            feature_encoder=build_process_feature_encoder(policy, config),
            code_inference=inference,
            fixed_decoder=load_fixed_decoder(
                checkpoint=decoder_checkpoint,
                contract=contract,
                config=config,
                device=device,
            ),
        )
        return writer.to(device)

    @staticmethod
    def _frame_condition_ids(
        *,
        condition_count: int,
        frame_count: int,
        device: torch.device,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> torch.Tensor:
        videos_per_condition = (
            condition_video_offsets[1:] - condition_video_offsets[:-1]
        )
        video_condition_ids = torch.repeat_interleave(
            torch.arange(condition_count, device=device),
            videos_per_condition.to(device),
        )
        frames_per_video = video_offsets[1:] - video_offsets[:-1]
        frame_video_ids = torch.repeat_interleave(
            torch.arange(video_condition_ids.shape[0], device=device),
            frames_per_video.to(device),
        )
        if frame_video_ids.numel() != frame_count:
            raise FunctionalCodeWriterError("video offsets do not own every frame")
        return video_condition_ids.index_select(0, frame_video_ids)

    def encode_features(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[LanguageAxialProcessFeatures, torch.Tensor]:
        frame_condition_ids = self._frame_condition_ids(
            condition_count=language_tokens.shape[0],
            frame_count=frames.shape[0],
            device=frames.device,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
        )
        features = self.feature_encoder.forward_process_features(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        return features, frame_condition_ids

    def infer_features(
        self,
        *,
        features: LanguageAxialProcessFeatures,
        frame_condition_ids: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> FunctionalCodePosterior:
        return self.code_inference(
            language_tokens=features.text_queries,
            valid_task_tokens=features.valid_task_tokens,
            frame_tokens=features.frame_evidence,
            visual_patch_tokens=features.visual_patch_tokens,
            action_probe_tokens=features.action_probe_tokens,
            frame_condition_ids=frame_condition_ids,
            frame_positions=frame_indices,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
        )

    def language_only_adapter(
        self,
        *,
        policy: torch.nn.Module,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """Generate a complete LoRA without reading teacher-video frames."""

        text, valid = self.feature_encoder.forward_text_features(
            policy,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        return self.fixed_decoder(self.code_inference.infer_language_code(text, valid))

    def video_only_adapter(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """Generate a complete LoRA without language or Action-probe reads."""

        visual = self.feature_encoder.forward_visual_features(policy, frames)
        code = self.code_inference.infer_video_code(
            visual,
            frame_indices,
            video_offsets,
            condition_video_offsets,
        )
        return self.fixed_decoder(code)

    def forward(
        self,
        *,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_indices: torch.Tensor,
        video_offsets: torch.Tensor,
        condition_video_offsets: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> FunctionalCodeWriterOutput:
        features, frame_condition_ids = self.encode_features(
            policy=policy,
            frames=frames,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
            language_tokens=language_tokens,
            language_mask=language_mask,
            task_span_mask=task_span_mask,
        )
        posterior = self.infer_features(
            features=features,
            frame_condition_ids=frame_condition_ids,
            frame_indices=frame_indices,
            video_offsets=video_offsets,
            condition_video_offsets=condition_video_offsets,
        )
        return FunctionalCodeWriterOutput(
            posterior=posterior,
            language_adapter=self.fixed_decoder(posterior.language_code),
            video_adapter=self.fixed_decoder(posterior.video_code),
            combined_adapter=self.fixed_decoder(posterior.combined_code),
        )
