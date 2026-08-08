"""Online one-shot video-to-LoRA generation for Expert-Manifold evaluation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    authority_path,
    load_barycentric_writer_config,
)
from ember.expert_manifold.inference import (
    expected_expert_manifold_episode_evidence,
    inspect_expert_manifold_writer_evaluation,
)
from ember.expert_manifold.model import (
    PolicyEffectiveBarycentricWriter,
    phase_centered_causal_memory,
)
from ember.expert_manifold.video_schedule import shuffled_frame_permutation
from ember.expert_manifold.video_features import FrozenPi05VideoInnovationEncoder
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.lora_rollout import PreparedWriterLoRA, WriterLoRARolloutAdapter


def _build_barycentric_writer(
    *,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
    lora: Any,
    template: Mapping[str, torch.Tensor],
    device: torch.device,
) -> PolicyEffectiveBarycentricWriter:
    template_cpu = {
        name: value.detach().to(device="cpu", dtype=torch.float32)
        for name, value in template.items()
    }
    expert_rows = sorted(
        observed["expert_basis"]["tasks"], key=lambda row: int(row["ordinal"])
    )
    cache_rows = sorted(
        observed["feature_cache"]["tasks"],
        key=lambda row: int(row["task_ordinal"]),
    )
    expert_states = []
    task_centroids = []
    expected_shape = (
        int(config["expert_basis"]["centroid_videos_per_task"]),
        int(config["video_features"]["phase_slots"]),
        int(config["video_features"]["feature_width"]),
    )
    for ordinal, (expert_row, cache_row) in enumerate(
        zip(expert_rows, cache_rows, strict=True)
    ):
        if (
            int(expert_row["ordinal"]) != ordinal
            or int(cache_row["task_ordinal"]) != ordinal
        ):
            raise ExpertManifoldError("barycentric basis ordering changed")
        state = load_file(
            str(Path(expert_row["checkpoint"]) / "adapter.safetensors"), device="cpu"
        )
        validate_lora_state(state, lora)
        expert_states.append(state)
        features = load_file(str(cache_row["features"]["path"]), device="cpu")[
            "video_innovation"
        ].float()
        if tuple(features.shape) != expected_shape:
            raise ExpertManifoldError("barycentric centroid feature shape changed")
        task_centroids.append(
            phase_centered_causal_memory(features).mean(dim=1).mean(dim=0)
        )
    topology = config["barycentric_writer"]
    writer = PolicyEffectiveBarycentricWriter(
        contract=lora,
        template_state=template_cpu,
        expert_states=expert_states,
        task_centroids=torch.stack(task_centroids),
        phase_slots=int(config["video_features"]["phase_slots"]),
        feature_width=int(config["video_features"]["feature_width"]),
        ridge=float(topology["ridge"]),
        effective_basis_rank=int(topology["effective_basis_rank"]),
        identity_epsilon=float(topology["identity_epsilon"]),
    ).to(device)
    writer.eval()
    if tuple(writer.parameters()):
        raise ExpertManifoldError("closed-form barycentric Writer became trainable")
    return writer


def _build_video_runtime(
    *,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[
    FrozenPi05VideoInnovationEncoder,
    RawTeacherVideoStore,
    dict[int, str],
    Pi05TeacherPrefixTokenizer,
]:
    video = config["video_features"]
    extraction = video["extraction"]
    encoder = (
        FrozenPi05VideoInnovationEncoder(
            image_width=int(video["image_hidden_width"]),
            expert_width=int(video["expert_hidden_width"]),
            feature_width=int(video["feature_width"]),
            phase_slots=int(video["phase_slots"]),
            max_frames_per_encoder_call=int(extraction["max_frames_per_encoder_call"]),
            action_horizon=int(extraction["action_horizon"]),
            padded_action_dim=int(extraction["padded_action_dim"]),
            initialization_seed=int(extraction["initialization_seed"]),
        )
        .to(device)
        .eval()
    )
    root = Path(observed["video_data"]["root"])
    authorities = [
        WriterTaskAuthority(
            task_id=int(row["global_task_id"]),
            language=str(row["language"]),
            path=root / str(row["relative_path"]),
            expected_bytes=int(row["bytes"]),
        )
        for row in observed["video_data"]["tasks"]
    ]
    store = RawTeacherVideoStore(
        authorities, frame_stride=int(video["frame_stride"]), max_open_files=2
    )
    languages = {authority.task_id: authority.language for authority in authorities}
    source_config = read_json(authority_path(config, "source_base_config"))
    tokenizer = Pi05TeacherPrefixTokenizer(
        tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    return encoder, store, languages, tokenizer


class FrozenExpertManifoldTaskAdapter(WriterLoRARolloutAdapter):
    """Generate one complete public LoRA from one action-hidden teacher video."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        source: Mapping[str, Any],
        evaluation_adapter: Mapping[str, Any],
        task_keys: Sequence[tuple[str, int]],
        device: torch.device,
        tokenizer_path: Path,
        require_formal: bool,
    ) -> None:
        observed = inspect_expert_manifold_writer_evaluation(
            config_path=Path(str(evaluation_adapter["config"]["path"])),
            expert_bank_root=Path(str(evaluation_adapter["expert_basis"]["root"])),
            feature_cache_root=Path(str(evaluation_adapter["feature_cache"]["root"])),
            video_data_root=Path(str(evaluation_adapter["video_data"]["root"])),
            source=source,
            task_keys=task_keys,
            video_condition=str(evaluation_adapter["video_condition"]),
            video_seed=int(evaluation_adapter["video_schedule"]["seed"]),
            video_sampling_mode=str(
                evaluation_adapter["video_schedule"]["sampling_mode"]
            ),
            require_formal=require_formal,
        )
        if observed != dict(evaluation_adapter):
            raise ExpertManifoldError("Expert-Manifold evaluation assets changed")
        config = load_barycentric_writer_config(Path(observed["config"]["path"]))
        lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        template = prepare_frozen_writer_policy(policy, lora)
        self.writer = _build_barycentric_writer(
            config=config,
            observed=observed,
            lora=lora,
            template=template,
            device=device,
        )
        (
            self.encoder,
            self.store,
            self.language_by_id,
            self.tokenizer,
        ) = _build_video_runtime(
            config=config,
            observed=observed,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        self._initialize_rollout(
            policy=policy,
            lora_contract=lora,
            identity_state=template,
            evaluation_adapter=observed,
            device=device,
        )

    def _episode_input(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> tuple[dict[str, Any], torch.Tensor | None, str]:
        reference = (
            f"{self.evaluation_adapter['writer_asset']['reference']}:"
            f"{suite}:{task_id}:{init_state_id}"
        )
        row = expected_expert_manifold_episode_evidence(
            self.evaluation_adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=reference,
        )
        condition = str(self.evaluation_adapter["video_condition"])
        language = self.language_by_id[int(row["language_global_task_id"])]
        if condition == "no_video":
            return row, None, language
        teacher = self.store.load(
            int(row["video_global_task_id"]), int(row["teacher_demo_indices"][0])
        )
        frames = torch.from_numpy(teacher.frames).to(self.device, non_blocking=True)
        if condition == "reversed":
            frames = frames.flip(0)
        elif condition in {"shuffled", "shuffled_keep_first"}:
            permutation = shuffled_frame_permutation(
                frames.shape[0],
                int(row["teacher_video_order_seeds"][0]),
                keep_first=condition == "shuffled_keep_first",
            ).to(self.device)
            frames = frames.index_select(0, permutation)
        return row, frames, language

    @torch.inference_mode()
    def prepare_episodes(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[PreparedWriterLoRA, ...]:
        if not identities:
            raise ExpertManifoldError("Expert-Manifold generation batch is empty")
        inputs = [
            self._episode_input(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            for identity in identities
        ]
        rows, frame_batches, languages = zip(*inputs, strict=True)
        tokens, masks, spans = self.tokenizer(list(languages))
        batch_size = len(identities)
        started = time.monotonic()
        copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
        self._physical_lora_is_identity = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.evaluation_adapter["video_condition"] == "no_video":
                if any(batch is not None for batch in frame_batches):
                    raise ExpertManifoldError("no-video counterfactual read frames")
                features = torch.zeros(
                    batch_size,
                    self.encoder.phase_slots,
                    self.encoder.feature_width,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                if any(batch is None for batch in frame_batches):
                    raise ExpertManifoldError("video-conditioned episode lost frames")
                concrete_batches = tuple(
                    batch for batch in frame_batches if batch is not None
                )
                frames = torch.cat(concrete_batches)
                lengths = torch.tensor(
                    [batch.shape[0] for batch in concrete_batches],
                    dtype=torch.long,
                    device=self.device,
                )
                frame_video_ids = torch.repeat_interleave(
                    torch.arange(batch_size, device=self.device), lengths
                )
                video_offsets = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long, device=self.device),
                        lengths.cumsum(dim=0),
                    )
                )
                features = self.encoder(
                    self.policy,
                    frames,
                    frame_video_ids,
                    video_offsets,
                    tokens,
                    masks,
                    spans,
                )
            generated = self.writer(features)
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ExpertManifoldError("Expert-Manifold generation timing changed")
        result = []
        for index, row in enumerate(rows):
            state = {name: value[index].detach() for name, value in generated.items()}
            validate_lora_state(state, self.lora_contract)
            result.append(
                PreparedWriterLoRA(
                    state=state,
                    evidence={
                        **row,
                        "writer_generation_seconds": elapsed / batch_size,
                    },
                )
            )
        return tuple(result)

    @torch.inference_mode()
    def prepare_episode(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> PreparedWriterLoRA:
        return self.prepare_episodes(
            ({"suite": suite, "task_id": task_id, "init_state_id": init_state_id},)
        )[0]

    def release_to_cache(self, cache_contract: Mapping[str, Any]) -> Any:
        from ember.writer.evaluation_runtime import FrozenCachedWriterTaskAdapter

        cached = FrozenCachedWriterTaskAdapter.from_live(
            self,
            cache_contract=cache_contract,
        )
        self.store.close()
        del self.store
        del self.language_by_id
        del self.tokenizer
        del self.writer
        del self.encoder
        return cached
