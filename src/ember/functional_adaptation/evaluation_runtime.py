"""Live LoRA generation for the fixed-decoder functional-code Writer."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.functional_adaptation.code_writer import FunctionalCodeWriter
from ember.functional_adaptation.decoder_training import (
    authority_path,
    load_functional_adapter_config,
)
from ember.functional_adaptation.evaluation import (
    expected_functional_code_writer_episode,
    inspect_functional_code_writer_evaluation,
)
from ember.functional_adaptation.process_controls import (
    TEMPORAL_PROCESS_VIDEO_CONDITIONS,
    frame_control,
)
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_eval_contract import load_evaluation_authorities
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.writer.data import RawTeacherVideo, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.errors import WriterModelError
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.lora_rollout import PreparedWriterLoRA, WriterLoRARolloutAdapter


REPO_ROOT = Path(__file__).resolve().parents[3]


def _condition_video_offsets(batch_size: int, evaluation_k: int) -> torch.Tensor:
    if batch_size <= 0 or evaluation_k <= 0:
        raise WriterModelError("functional-code generation batch or K is invalid")
    return torch.arange(
        0,
        (batch_size + 1) * evaluation_k,
        evaluation_k,
        dtype=torch.long,
        device="cpu",
    )


class FrozenFunctionalCodeTaskAdapter(WriterLoRARolloutAdapter):
    """Generate one complete LoRA through the frozen functional decoder."""

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
        observed = inspect_functional_code_writer_evaluation(
            config_path=Path(str(evaluation_adapter["config"]["path"])),
            checkpoint=Path(str(evaluation_adapter["writer_asset"]["checkpoint"])),
            video_data_root=Path(str(evaluation_adapter["video_data"]["root"])),
            source=source,
            task_keys=task_keys,
            video_condition=str(evaluation_adapter["video_condition"]),
            video_seed=int(evaluation_adapter["video_schedule"]["seed"]),
            video_sampling_mode=str(
                evaluation_adapter["video_schedule"]["sampling_mode"]
            ),
            require_formal=require_formal,
            evaluation_k=int(evaluation_adapter["information_wall"]["evaluation_k"]),
        )
        if observed != dict(evaluation_adapter):
            raise WriterModelError("functional-code evaluation assets changed")
        config = load_functional_adapter_config(
            Path(str(observed["config"]["path"])), REPO_ROOT
        )
        lora = load_pi05_lora_contract(
            authority_path(config, "lora_contract", REPO_ROOT)
        )
        identity = prepare_frozen_writer_policy(policy, lora)
        decoder_checkpoint = (
            Path(str(observed["writer_asset"]["decoder_profile"]["root"]))
            / "decoder.safetensors"
        )
        self.writer = FunctionalCodeWriter.from_policy(
            policy=policy,
            config=config,
            contract=lora,
            decoder_checkpoint=decoder_checkpoint,
            device=device,
        )
        self.writer.load_state_dict(
            load_file(
                str(Path(observed["writer_asset"]["writer_state"]["path"])),
                device=str(device),
            ),
            strict=True,
        )
        self.writer.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in self.writer.parameters()):
            raise WriterModelError("functional-code deployment Writer became trainable")
        self.evaluation_k = int(observed["information_wall"]["evaluation_k"])
        self.condition = str(observed["video_condition"])
        self.store, self.language_by_id = self._video_runtime(config, observed)
        self.tokenizer = self._tokenizer_runtime(
            config,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        self._initialize_rollout(
            policy=policy,
            lora_contract=lora,
            identity_state=identity,
            evaluation_adapter=observed,
            device=device,
        )
        self._last_generation_batch_profile: tuple[dict[str, Any], ...] = ()

    def _video_runtime(
        self, config: Mapping[str, Any], observed: Mapping[str, Any]
    ) -> tuple[RawTeacherVideoStore | None, dict[int, str]]:
        root = Path(str(observed["video_data"]["root"]))
        authorities = tuple(
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=root / str(row["relative_path"]),
                expected_bytes=int(row["bytes"]),
            )
            for row in observed["video_data"]["tasks"]
        )
        store = (
            None
            if self.condition == "language_only"
            else RawTeacherVideoStore(
                authorities,
                frame_stride=int(config["code_inference"]["training"]["frame_stride"]),
                max_open_files=int(
                    config["code_inference"]["training"]["video_open_files_per_rank"]
                ),
            )
        )
        return store, {row.task_id: row.language for row in authorities}

    def _tokenizer_runtime(
        self,
        config: Mapping[str, Any],
        *,
        tokenizer_path: Path,
        device: torch.device,
    ) -> Pi05TeacherPrefixTokenizer | None:
        if self.condition == "video_only":
            return None
        authorities = load_evaluation_authorities(
            authority_path(config, "evaluation_config", REPO_ROOT), REPO_ROOT
        )
        return Pi05TeacherPrefixTokenizer(
            tokenizer_path,
            int(authorities.source_base_config["features"]["tokenizer_max_length"]),
            str(device),
        )

    def _episode_input(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> tuple[dict[str, Any], tuple[RawTeacherVideo, ...], str | None]:
        row = self._episode_evidence(
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
        )
        language = (
            None
            if self.condition == "video_only"
            else self.language_by_id[int(row["language_global_task_id"])]
        )
        if self.condition == "language_only":
            return row, (), language
        if self.store is None:
            raise WriterModelError("functional-code video store is unavailable")
        videos = tuple(
            self.store.load(int(row["video_global_task_id"]), int(demo_index))
            for demo_index in row["teacher_demo_indices"]
        )
        return row, videos, language

    def _episode_evidence(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> dict[str, Any]:
        reference = (
            f"{self.evaluation_adapter['writer_asset']['reference']}:"
            f"{suite}:{task_id}:{init_state_id}"
        )
        return expected_functional_code_writer_episode(
            self.evaluation_adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=reference,
        )

    def _controlled_video_tensors(
        self,
        rows: Sequence[Mapping[str, Any]],
        video_sets: Sequence[Sequence[RawTeacherVideo]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if any(len(videos) != self.evaluation_k for videos in video_sets):
            raise WriterModelError("functional-code episode lost teacher videos")
        flattened = tuple(video for videos in video_sets for video in videos)
        seeds = tuple(
            int(seed) for row in rows for seed in row["teacher_video_order_seeds"]
        )
        content_condition = (
            self.condition
            if self.condition in TEMPORAL_PROCESS_VIDEO_CONDITIONS
            else "correct"
        )
        frames, indices = [], []
        for video, seed in zip(flattened, seeds, strict=True):
            control = frame_control(
                int(video.frames.shape[0]),
                condition=content_condition,
                order_seed=seed,
            )
            frames.append(
                torch.from_numpy(video.frames)
                .to(self.device, non_blocking=True)
                .index_select(0, control.content.to(self.device))
            )
            indices.append(
                torch.from_numpy(video.frame_indices)
                .to(self.device, non_blocking=True)
                .index_select(0, control.positions.to(self.device))
            )
        return frames, indices

    def generation_request_profiles(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        rows = []
        for identity in identities:
            evidence = self._episode_evidence(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            if self.condition == "language_only":
                counts: tuple[tuple[int, int], ...] = ()
            else:
                if self.store is None:
                    raise WriterModelError("functional-code video store is unavailable")
                counts = tuple(
                    self.store.frame_counts(
                        int(evidence["video_global_task_id"]), int(demo_index)
                    )
                    for demo_index in evidence["teacher_demo_indices"]
                )
            content_condition = (
                self.condition
                if self.condition in TEMPORAL_PROCESS_VIDEO_CONDITIONS
                else "correct"
            )
            sampled = sum(
                int(
                    frame_control(
                        available,
                        condition=content_condition,
                        order_seed=int(seed),
                    ).content.numel()
                )
                for (_, available), seed in zip(
                    counts, evidence["teacher_video_order_seeds"], strict=True
                )
            )
            rows.append(
                {
                    "suite": str(identity["suite"]),
                    "task_id": int(identity["task_id"]),
                    "init_state_id": int(identity["init_state_id"]),
                    "raw_frames": sum(raw for raw, _ in counts),
                    "available_stride5_frames": sum(
                        available for _, available in counts
                    ),
                    "sampled_frames": sampled,
                }
            )
        return tuple(rows)

    def last_generation_batch_profile(self) -> tuple[dict[str, Any], ...]:
        if not self._last_generation_batch_profile:
            raise WriterModelError("functional-code generation profile is unavailable")
        return self._last_generation_batch_profile

    @torch.inference_mode()
    def prepare_episodes(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[PreparedWriterLoRA, ...]:
        if not identities:
            raise WriterModelError("functional-code generation batch is empty")
        inputs = tuple(
            self._episode_input(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            for identity in identities
        )
        rows, video_sets, languages = zip(*inputs, strict=True)
        if not self._physical_lora_is_identity:
            copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
            self._physical_lora_is_identity = True
        started = time.monotonic()
        frames: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        if self.condition != "language_only":
            frames, indices = self._controlled_video_tensors(rows, video_sets)
        self._last_generation_batch_profile = tuple(
            {
                "suite": str(identity["suite"]),
                "task_id": int(identity["task_id"]),
                "init_state_id": int(identity["init_state_id"]),
                "raw_frames": sum(int(video.raw_frame_count) for video in videos),
                "available_stride5_frames": sum(
                    int(video.frames.shape[0]) for video in videos
                ),
                "sampled_frames": sum(
                    int(value.shape[0])
                    for value in frames[
                        index * self.evaluation_k : (index + 1) * self.evaluation_k
                    ]
                ),
            }
            for index, (identity, videos) in enumerate(
                zip(identities, video_sets, strict=True)
            )
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            if self.condition == "language_only":
                if self.tokenizer is None or any(
                    language is None for language in languages
                ):
                    raise WriterModelError("language-only input is unavailable")
                tokens, masks, spans = self.tokenizer(list(languages))
                generated = self.writer.language_only_adapter(
                    policy=self.policy,
                    language_tokens=tokens,
                    language_mask=masks,
                    task_span_mask=spans,
                )
            else:
                offsets = [0]
                for value in frames:
                    offsets.append(offsets[-1] + int(value.shape[0]))
                video_offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
                ownership = _condition_video_offsets(len(rows), self.evaluation_k)
                if self.condition == "video_only":
                    generated = self.writer.video_only_adapter(
                        policy=self.policy,
                        frames=torch.cat(frames),
                        frame_indices=torch.cat(indices),
                        video_offsets=video_offsets,
                        condition_video_offsets=ownership,
                    )
                else:
                    if self.tokenizer is None or any(
                        language is None for language in languages
                    ):
                        raise WriterModelError(
                            "combined Writer language is unavailable"
                        )
                    tokens, masks, spans = self.tokenizer(list(languages))
                    generated = self.writer(
                        policy=self.policy,
                        frames=torch.cat(frames),
                        frame_indices=torch.cat(indices),
                        video_offsets=video_offsets,
                        condition_video_offsets=ownership,
                        language_tokens=tokens,
                        language_mask=masks,
                        task_span_mask=spans,
                    ).combined_adapter
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WriterModelError("functional-code generation timing changed")
        result = []
        for index, row in enumerate(rows):
            state = {name: value[index].detach() for name, value in generated.items()}
            validate_lora_state(state, self.lora_contract)
            result.append(
                PreparedWriterLoRA(
                    state=state,
                    evidence={
                        **dict(row),
                        "writer_generation_seconds": elapsed / len(rows),
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
        self.release_generation_assets()
        return cached

    def release_generation_assets(self) -> None:
        if self.store is not None:
            self.store.close()
        del self.store
        del self.language_by_id
        del self.tokenizer
        del self.evaluation_k
        del self.writer
