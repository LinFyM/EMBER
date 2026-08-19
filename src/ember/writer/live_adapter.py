"""Online K1--K4 generation for the Dynamic-K Backbone-Memory Writer."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.lora import (
    copy_task_lora_state_,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.functional_adaptation.process_controls import frame_control
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import authority_path, load_writer_config
from ember.writer.checkpoint import load_writer_deployment_state_
from ember.writer.data import RawTeacherVideo, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.errors import WriterModelError
from ember.writer.evaluation import (
    expected_dynamic_k_episode_evidence,
    inspect_dynamic_k_writer_evaluation,
)
from ember.writer.frame_budget import apply_condition_frame_budget
from ember.writer.lora_rollout import PreparedWriterLoRA, WriterLoRARolloutAdapter
from ember.writer.training import build_writer


def condition_video_offsets(batch_size: int, evaluation_k: int) -> torch.Tensor:
    """Return canonical fixed-K ragged ownership for one condition batch."""

    if batch_size <= 0 or evaluation_k <= 0:
        raise WriterModelError("dynamic-K generation batch or K is invalid")
    return torch.arange(
        0,
        (batch_size + 1) * evaluation_k,
        evaluation_k,
        dtype=torch.long,
        device="cpu",
    )


def _video_runtime(
    *,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[RawTeacherVideoStore, dict[int, str], Pi05TeacherPrefixTokenizer]:
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
    store = RawTeacherVideoStore(
        authorities,
        frame_stride=int(config["writer"]["frame_stride"]),
        max_open_files=int(config["data"]["video_open_files_per_rank"]),
    )
    source_config = read_json(authority_path(config, "source_base_config"))
    tokenizer = Pi05TeacherPrefixTokenizer(
        tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    return store, {item.task_id: item.language for item in authorities}, tokenizer


def _ordered_video_tensors(
    video: RawTeacherVideo,
    *,
    condition: str,
    order_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = torch.from_numpy(video.frames).to(device, non_blocking=True)
    indices = torch.from_numpy(video.frame_indices).to(device, non_blocking=True)
    control = frame_control(
        frames.shape[0],
        condition=condition,
        order_seed=order_seed,
    )
    return (
        frames.index_select(0, control.content.to(device)),
        indices.index_select(0, control.positions.to(device)),
    )


class FrozenDynamicKTaskAdapter(WriterLoRARolloutAdapter):
    """Generate one complete deployment LoRA from a paired video condition."""

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
        observed = inspect_dynamic_k_writer_evaluation(
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
            raise WriterModelError("dynamic-K evaluation assets changed")
        config = load_writer_config(Path(str(observed["config"]["path"])))
        self.total_frame_budget = int(
            config["writer"]["backbone_total_frames_per_condition"]
        )
        self.evaluation_k = int(observed["information_wall"]["evaluation_k"])
        self.writer, lora = build_writer(
            config,
            policy,
            asset_root=Path(str(source["source_run"])).resolve().parents[2],
            deployment_rank=int(observed["lora_contract"]["rank"]),
        )
        self.writer.to(device)
        load_writer_deployment_state_(
            writer=self.writer,
            writer_asset=observed["writer_asset"],
            device=device,
        )
        self.writer.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in self.writer.parameters()):
            raise WriterModelError("dynamic-K deployment Writer became trainable")
        self.store, self.language_by_id, self.tokenizer = _video_runtime(
            config=config,
            observed=observed,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        self._initialize_rollout(
            policy=policy,
            lora_contract=lora,
            identity_state=task_lora_state_dict(policy, clone=True),
            evaluation_adapter=observed,
            device=device,
        )
        self._last_generation_batch_profile: tuple[dict[str, Any], ...] = ()

    def _video_condition(self) -> str:
        return str(
            getattr(self, "evaluation_adapter", {}).get("video_condition", "correct")
        )

    def _episode_input(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> tuple[
        dict[str, Any], tuple[RawTeacherVideo, ...], str, tuple[int, ...]
    ]:
        reference = (
            f"{self.evaluation_adapter['writer_asset']['reference']}:"
            f"{suite}:{task_id}:{init_state_id}"
        )
        row = expected_dynamic_k_episode_evidence(
            self.evaluation_adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=reference,
        )
        if self._video_condition() == "no_video":
            language = self.language_by_id[int(row["language_global_task_id"])]
            return row, (), language, ()
        global_task_id = int(row["video_global_task_id"])
        available = tuple(
            self.store.load(global_task_id, int(demo_index))
            for demo_index in row["teacher_demo_indices"]
        )
        videos = apply_condition_frame_budget(
            available,
            self.total_frame_budget,
        )
        return (
            row,
            videos,
            self.language_by_id[int(row["language_global_task_id"])],
            tuple(int(video.frames.shape[0]) for video in available),
        )

    def _identity_results(
        self,
        rows: Sequence[Mapping[str, Any]],
        elapsed: float,
    ) -> tuple[PreparedWriterLoRA, ...]:
        return tuple(
            PreparedWriterLoRA(
                state={
                    name: value.detach().to(device=self.device)
                    for name, value in self.identity_state.items()
                },
                evidence={
                    **dict(row),
                    "writer_generation_seconds": elapsed / len(rows),
                },
            )
            for row in rows
        )

    def generation_request_profiles(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], ...]:
        rows = []
        for identity in identities:
            suite = str(identity["suite"])
            task_id = int(identity["task_id"])
            init_state_id = int(identity["init_state_id"])
            reference = (
                f"{self.evaluation_adapter['writer_asset']['reference']}:"
                f"{suite}:{task_id}:{init_state_id}"
            )
            evidence = expected_dynamic_k_episode_evidence(
                self.evaluation_adapter,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                lora_reference=reference,
            )
            if self._video_condition() == "no_video":
                counts: tuple[tuple[int, int], ...] = ()
            else:
                counts = tuple(
                    self.store.frame_counts(
                        int(evidence["video_global_task_id"]), int(demo_index)
                    )
                    for demo_index in evidence["teacher_demo_indices"]
                )
            per_video_budget = (
                self.total_frame_budget // len(counts) if counts else 0
            )
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "init_state_id": init_state_id,
                    "raw_frames": sum(raw for raw, _ in counts),
                    "available_stride5_frames": sum(
                        available for _, available in counts
                    ),
                    "sampled_frames": sum(
                        min(available, per_video_budget)
                        for _, available in counts
                    ),
                }
            )
        return tuple(rows)

    def last_generation_batch_profile(self) -> tuple[dict[str, Any], ...]:
        if not self._last_generation_batch_profile:
            raise WriterModelError("dynamic-K generation profile is unavailable")
        return self._last_generation_batch_profile

    def _conditioned_video_tensors(
        self,
        rows: Sequence[Mapping[str, Any]],
        video_sets: Sequence[Sequence[RawTeacherVideo]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if any(len(videos) != self.evaluation_k for videos in video_sets):
            raise WriterModelError("video-conditioned dynamic-K episode lost frames")
        flattened = tuple(video for videos in video_sets for video in videos)
        order_seeds = tuple(
            int(seed)
            for row, videos in zip(rows, video_sets, strict=True)
            for seed in row.get("teacher_video_order_seeds", (0,) * len(videos))
        )
        ordered = tuple(
            _ordered_video_tensors(
                video,
                condition=self._video_condition(),
                order_seed=seed,
                device=self.device,
            )
            for video, seed in zip(flattened, order_seeds, strict=True)
        )
        return [value[0] for value in ordered], [value[1] for value in ordered]

    @torch.inference_mode()
    def prepare_episodes(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[PreparedWriterLoRA, ...]:
        if not identities:
            raise WriterModelError("dynamic-K generation batch is empty")
        inputs = tuple(
            self._episode_input(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            for identity in identities
        )
        rows, video_sets, languages, available_count_sets = zip(*inputs, strict=True)
        self._last_generation_batch_profile = tuple(
            {
                "suite": str(identity["suite"]),
                "task_id": int(identity["task_id"]),
                "init_state_id": int(identity["init_state_id"]),
                "raw_frames": sum(
                    int(video.raw_frame_count) for video in videos
                ),
                "available_stride5_frames": sum(available_counts),
                "sampled_frames": sum(
                    int(video.frames.shape[0]) for video in videos
                ),
            }
            for identity, videos, available_counts in zip(
                identities, video_sets, available_count_sets, strict=True
            )
        )
        started = time.monotonic()
        if not self._physical_lora_is_identity:
            copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
            self._physical_lora_is_identity = True
        condition = self._video_condition()
        if condition == "no_video":
            if any(videos for videos in video_sets):
                raise WriterModelError("no-video counterfactual read frames")
            return self._identity_results(rows, time.monotonic() - started)
        tokens, masks, spans = self.tokenizer(list(languages))
        frames, indices = self._conditioned_video_tensors(rows, video_sets)
        offsets = [0]
        for value in frames:
            offsets.append(offsets[-1] + int(value.shape[0]))
        video_offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
        ownership = condition_video_offsets(len(rows), self.evaluation_k)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            generated = self.writer(
                torch.cat(frames),
                torch.cat(indices),
                video_offsets,
                ownership,
                tokens,
                masks,
                spans,
                policy=self.policy,
            )
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WriterModelError("dynamic-K generation timing changed")
        batch_size = len(rows)
        result = []
        for index, row in enumerate(rows):
            state = {
                name: (value.detach() if batch_size == 1 else value[index].detach())
                for name, value in generated.items()
            }
            validate_lora_state(state, self.lora_contract)
            result.append(
                PreparedWriterLoRA(
                    state=state,
                    evidence={
                        **dict(row),
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
        self.release_generation_assets()
        return cached

    def release_generation_assets(self) -> None:
        self.store.close()
        del self.store
        del self.language_by_id
        del self.tokenizer
        del self.total_frame_budget
        del self.evaluation_k
        del self.writer
