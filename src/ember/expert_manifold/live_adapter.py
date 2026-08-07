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
    load_expert_manifold_config,
)
from ember.expert_manifold.inference import (
    expected_expert_manifold_episode_evidence,
    inspect_expert_manifold_writer_evaluation,
)
from ember.expert_manifold.model import VideoConditionedTopologicalWriter
from ember.expert_manifold.video_features import FrozenPi05VideoInnovationEncoder
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.inference import writer_shuffled_frame_permutation
from ember.writer.live_adapter import PreparedWriterLoRA
from ember.writer.lora_rollout import WriterLoRARolloutAdapter


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
            checkpoint=Path(str(evaluation_adapter["checkpoint"]["path"])),
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
        config = load_expert_manifold_config(Path(observed["config"]["path"]))
        lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        template = prepare_frozen_writer_policy(policy, lora)
        topology = config["topological_writer"]
        writer = VideoConditionedTopologicalWriter(
            contract=lora,
            template_state=template,
            phase_slots=int(config["video_features"]["phase_slots"]),
            feature_width=int(config["video_features"]["feature_width"]),
            memory_width=int(topology["memory_width"]),
            attention_heads=int(topology["attention_heads"]),
            axial_blocks=int(topology["axial_blocks"]),
            chunk_width=int(topology["chunk_width"]),
        ).to(device)
        writer.load_state_dict(
            load_file(
                str(Path(observed["checkpoint"]["path"]) / "writer.safetensors"),
                device=str(device),
            ),
            strict=True,
        )
        writer.eval()
        for parameter in writer.parameters():
            parameter.requires_grad_(False)
        video = config["video_features"]
        extraction = video["extraction"]
        encoder = FrozenPi05VideoInnovationEncoder(
            image_width=int(video["image_hidden_width"]),
            expert_width=int(video["expert_hidden_width"]),
            feature_width=int(video["feature_width"]),
            phase_slots=int(video["phase_slots"]),
            max_frames_per_encoder_call=int(extraction["max_frames_per_encoder_call"]),
            action_horizon=int(extraction["action_horizon"]),
            padded_action_dim=int(extraction["padded_action_dim"]),
            initialization_seed=int(extraction["initialization_seed"]),
        ).to(device).eval()
        records = observed["video_data"]["tasks"]
        root = Path(observed["video_data"]["root"])
        authorities = [
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=root / str(row["relative_path"]),
                expected_bytes=int(row["bytes"]),
            )
            for row in records
        ]
        self.store = RawTeacherVideoStore(
            authorities,
            frame_stride=int(video["frame_stride"]),
            max_open_files=2,
        )
        self.language_by_id = {
            authority.task_id: authority.language for authority in authorities
        }
        source_config = read_json(authority_path(config, "source_base_config"))
        self.tokenizer = Pi05TeacherPrefixTokenizer(
            tokenizer_path,
            int(source_config["features"]["tokenizer_max_length"]),
            str(device),
        )
        self.writer = writer
        self.encoder = encoder
        self._initialize_rollout(
            policy=policy,
            lora_contract=lora,
            identity_state=template,
            evaluation_adapter=observed,
            device=device,
        )

    def _episode_input(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> tuple[dict[str, Any], torch.Tensor, str]:
        reference = (
            f"{self.evaluation_adapter['checkpoint']['reference']}:"
            f"{suite}:{task_id}:{init_state_id}"
        )
        row = expected_expert_manifold_episode_evidence(
            self.evaluation_adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=reference,
        )
        teacher = self.store.load(
            int(row["video_global_task_id"]), int(row["teacher_demo_indices"][0])
        )
        frames = torch.from_numpy(teacher.frames).to(self.device, non_blocking=True)
        condition = str(self.evaluation_adapter["video_condition"])
        if condition == "reversed":
            frames = frames.flip(0)
        elif condition in {"shuffled", "shuffled_keep_first"}:
            permutation = writer_shuffled_frame_permutation(
                frames.shape[0],
                int(row["teacher_video_order_seeds"][0]),
                keep_first=condition == "shuffled_keep_first",
            ).to(self.device)
            frames = frames.index_select(0, permutation)
        language = self.language_by_id[int(row["language_global_task_id"])]
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
        frames = torch.cat(frame_batches)
        lengths = torch.tensor(
            [batch.shape[0] for batch in frame_batches],
            dtype=torch.long,
            device=self.device,
        )
        frame_video_ids = torch.repeat_interleave(
            torch.arange(len(frame_batches), device=self.device), lengths
        )
        video_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=self.device),
                lengths.cumsum(dim=0),
            )
        )
        tokens, masks, spans = self.tokenizer(list(languages))
        started = time.monotonic()
        copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
        self._physical_lora_is_identity = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        batch_size = len(identities)
        for index, row in enumerate(rows):
            state = {
                name: value[index].detach() for name, value in generated.items()
            }
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
