"""Online one-shot v6 video-to-LoRA generation for canonical evaluation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.inference import (
    expected_expert_manifold_episode_evidence,
    inspect_expert_manifold_writer_evaluation,
    load_expert_manifold_deployment_config,
)
from ember.expert_manifold.v6_prior import (
    freeze_v6_prior_writer,
    load_v6_prior_warm_start_,
)
from ember.expert_manifold.v6_prior_checkpoint import PROGRAM_MEMORY_KEY
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    authority_path,
)
from ember.expert_manifold.video_schedule import shuffled_frame_permutation
from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.writer.architecture import LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
from ember.writer.data import RawTeacherVideo, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.lora_rollout import PreparedWriterLoRA, WriterLoRARolloutAdapter
from ember.writer.rank_reserved_compiler import (
    FrozenV6RankReservedRewardWriter,
    validate_frozen_v6_rank_reserved_writer,
)
from ember.writer.model import CompleteLoRAWriter, build_lora_tensor_specs


def _build_v6_writer(
    *,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
    policy: torch.nn.Module,
    template: Mapping[str, torch.Tensor],
    device: torch.device,
) -> FrozenV6RankReservedRewardWriter:
    bridge = policy.model.paligemma_with_expert
    writer_config = {
        name: value
        for name, value in config["writer"].items()
        if name in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS
    }
    base_writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model,
        **writer_config,
    )
    load_v6_prior_warm_start_(
        base_writer,
        (REPO_ROOT / str(config["initialization"]["checkpoint"])).resolve(),
    )
    freeze_v6_prior_writer(base_writer)
    asset = observed["writer_asset"]
    if type(asset.get("enable_program_residual")) is not bool:
        raise ExpertManifoldError(
            "retired v6 deployment assets cannot enter the rank-reserved compiler"
        )
    expected_assets = {
        "v6_qv_rank14_zero_program_load_only": (0, False),
        "v6_qv_rank14_plus2_reward_program_load_only": (1, True),
    }
    observed_identity = (
        int(asset.get("method_macro", -1)),
        asset["enable_program_residual"],
    )
    if expected_assets.get(str(asset.get("kind"))) != observed_identity:
        raise ExpertManifoldError("rank-reserved Writer asset identity changed")
    enable_program_residual = asset["enable_program_residual"]
    writer = FrozenV6RankReservedRewardWriter(
        base_writer,
        feature_width=int(config["condition_feature"]["feature_width"]),
        feature_seed=int(config["condition_feature"]["projection_seed"]),
        enable_program_residual=enable_program_residual,
    )
    if enable_program_residual:
        state = load_file(
            str(observed["writer_asset"]["residual_state"]["path"]),
            device="cpu",
        )
        if (
            set(state) != {PROGRAM_MEMORY_KEY}
            or state[PROGRAM_MEMORY_KEY].dtype != torch.float32
            or list(state[PROGRAM_MEMORY_KEY].shape)
            != list(observed["writer_asset"]["residual_state"]["shape"])
            or not bool(torch.isfinite(state[PROGRAM_MEMORY_KEY]).all())
        ):
            raise ExpertManifoldError("v6-prior residual evaluation state changed")
        writer.program_memory.value.copy_(state[PROGRAM_MEMORY_KEY])
    if any(
        not torch.equal(
            value.detach().cpu(),
            template[name].detach().cpu().to(value.dtype),
        )
        for name, value in base_writer.template_state().items()
    ):
        raise ExpertManifoldError("v6-prior evaluation template identity changed")
    writer.requires_grad_(False)
    validate_frozen_v6_rank_reserved_writer(
        writer, require_zero_memory=not enable_program_residual
    )
    writer.to(device).eval()
    if any(parameter.requires_grad for parameter in writer.parameters()):
        raise ExpertManifoldError("v6-prior evaluation Writer became trainable")
    return writer


def _build_video_runtime(
    *,
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
    tokenizer_path: Path,
    device: torch.device,
) -> tuple[
    RawTeacherVideoStore,
    dict[int, str],
    Pi05TeacherPrefixTokenizer,
]:
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
        authorities,
        frame_stride=int(config["writer"]["frame_stride"]),
        max_open_files=2,
    )
    languages = {authority.task_id: authority.language for authority in authorities}
    source_config = read_json(authority_path(config, "source_base_config"))
    tokenizer = Pi05TeacherPrefixTokenizer(
        tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(device),
    )
    return store, languages, tokenizer


def _ordered_video_tensors(
    video: RawTeacherVideo,
    *,
    condition: str,
    order_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames = torch.from_numpy(video.frames).to(device, non_blocking=True)
    indices = torch.from_numpy(video.frame_indices).to(device, non_blocking=True)
    if condition == "reversed":
        frames = frames.flip(0)
    elif condition in {"shuffled", "shuffled_keep_first"}:
        permutation = shuffled_frame_permutation(
            frames.shape[0],
            order_seed,
            keep_first=condition == "shuffled_keep_first",
        ).to(device)
        frames = frames.index_select(0, permutation)
    return frames, indices


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
        )
        if observed != dict(evaluation_adapter):
            raise ExpertManifoldError("v6-prior evaluation assets changed")
        config = load_expert_manifold_deployment_config(
            Path(observed["config"]["path"])
        )
        lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
        template = prepare_frozen_writer_policy(policy, lora)
        self.writer = _build_v6_writer(
            config=config,
            observed=observed,
            policy=policy,
            template=template,
            device=device,
        )
        self.store, self.language_by_id, self.tokenizer = _build_video_runtime(
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
        self._last_generation_batch_profile: tuple[dict[str, Any], ...] = ()
        self._last_diagnostic_five_arm_profile: tuple[dict[str, Any], ...] = ()

    def _episode_input(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> tuple[dict[str, Any], RawTeacherVideo | None, str]:
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
        language = self.language_by_id[int(row["language_global_task_id"])]
        if self.evaluation_adapter["video_condition"] == "no_video":
            return row, None, language
        teacher = self.store.load(
            int(row["video_global_task_id"]),
            int(row["teacher_demo_indices"][0]),
        )
        return row, teacher, language

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
        self,
        identities: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        """Return action-hidden video lengths for exact profile requests."""

        rows = []
        for identity in identities:
            suite = str(identity["suite"])
            task_id = int(identity["task_id"])
            init_state_id = int(identity["init_state_id"])
            reference = (
                f"{self.evaluation_adapter['writer_asset']['reference']}:"
                f"{suite}:{task_id}:{init_state_id}"
            )
            evidence = expected_expert_manifold_episode_evidence(
                self.evaluation_adapter,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                lora_reference=reference,
            )
            if self.evaluation_adapter["video_condition"] == "no_video":
                raw_frames, sampled_frames = 0, 0
            else:
                raw_frames, sampled_frames = self.store.frame_counts(
                    int(evidence["video_global_task_id"]),
                    int(evidence["teacher_demo_indices"][0]),
                )
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "init_state_id": init_state_id,
                    "raw_frames": raw_frames,
                    "sampled_frames": sampled_frames,
                }
            )
        return tuple(rows)

    def last_generation_batch_profile(self) -> tuple[dict[str, Any], ...]:
        if not self._last_generation_batch_profile:
            raise ExpertManifoldError("v6-prior generation profile is unavailable")
        return self._last_generation_batch_profile

    @torch.inference_mode()
    def prepare_episodes(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[PreparedWriterLoRA, ...]:
        if not identities:
            raise ExpertManifoldError("v6-prior generation batch is empty")
        inputs = [
            self._episode_input(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            for identity in identities
        ]
        rows, videos, languages = zip(*inputs, strict=True)
        self._last_generation_batch_profile = tuple(
            {
                "suite": str(identity["suite"]),
                "task_id": int(identity["task_id"]),
                "init_state_id": int(identity["init_state_id"]),
                "raw_frames": 0 if video is None else int(video.raw_frame_count),
                "sampled_frames": 0 if video is None else int(video.frames.shape[0]),
            }
            for identity, video in zip(identities, videos, strict=True)
        )
        started = time.monotonic()
        copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
        self._physical_lora_is_identity = True
        if self.evaluation_adapter["video_condition"] == "no_video":
            if any(video is not None for video in videos):
                raise ExpertManifoldError("no-video counterfactual read frames")
            return self._identity_results(rows, time.monotonic() - started)
        if any(video is None for video in videos):
            raise ExpertManifoldError("video-conditioned episode lost frames")
        tokens, masks, spans = self.tokenizer(list(languages))
        frame_batches = []
        index_batches = []
        condition = str(self.evaluation_adapter["video_condition"])
        for row, video in zip(rows, videos, strict=True):
            if video is None:
                raise ExpertManifoldError("v6-prior video batch changed")
            frames, indices = _ordered_video_tensors(
                video,
                condition=condition,
                order_seed=int(row["teacher_video_order_seeds"][0]),
                device=self.device,
            )
            frame_batches.append(frames)
            index_batches.append(indices)
        offsets_list = [0]
        for batch in frame_batches:
            offsets_list.append(offsets_list[-1] + int(batch.shape[0]))
        offsets = torch.tensor(offsets_list, dtype=torch.long, device="cpu")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = self.writer(
                torch.cat(frame_batches),
                torch.cat(index_batches),
                offsets,
                tokens,
                masks,
                spans,
                policy=self.policy,
            )
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ExpertManifoldError("v6-prior generation timing changed")
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

    @torch.inference_mode()
    def prepare_diagnostic_five_arms(
        self,
        identities: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[dict[str, torch.Tensor], ...]]:
        """Generate Gate-A numeric references from one shared cycle1 video batch."""

        asset = self.evaluation_adapter["writer_asset"]
        if (
            not identities
            or self.evaluation_adapter["video_condition"] != "correct"
            or asset.get("kind") != "v6_qv_rank14_plus2_reward_program_load_only"
            or int(asset.get("method_macro", -1)) != 1
            or asset.get("enable_program_residual") is not True
        ):
            raise ExpertManifoldError(
                "rank-reserved vertical requires the correct cycle1 asset"
            )
        inputs = [
            self._episode_input(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            for identity in identities
        ]
        rows, videos, languages = zip(*inputs, strict=True)
        if any(video is None for video in videos):
            raise ExpertManifoldError("rank-reserved vertical lost its video")
        copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
        self._physical_lora_is_identity = True
        tokens, masks, spans = self.tokenizer(list(languages))
        frame_batches = []
        index_batches = []
        for row, video in zip(rows, videos, strict=True):
            if video is None:
                raise ExpertManifoldError("rank-reserved vertical lost its video")
            frames, indices = _ordered_video_tensors(
                video,
                condition="correct",
                order_seed=int(row["teacher_video_order_seeds"][0]),
                device=self.device,
            )
            frame_batches.append(frames)
            index_batches.append(indices)
        offsets_list = [0]
        for frames in frame_batches:
            offsets_list.append(offsets_list[-1] + int(frames.shape[0]))
        offsets = torch.tensor(offsets_list, dtype=torch.long, device="cpu")
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            generated = self.writer.forward_diagnostic_five_arms(
                torch.cat(frame_batches),
                torch.cat(index_batches),
                offsets,
                tokens,
                masks,
                spans,
                policy=self.policy,
            )
        batch_size = len(rows)
        self._last_diagnostic_five_arm_profile = tuple(
            {
                **dict(row),
                "suite": str(identity["suite"]),
                "task_id": int(identity["task_id"]),
                "init_state_id": int(identity["init_state_id"]),
                "raw_frames": int(video.raw_frame_count),
                "sampled_frames": int(video.frames.shape[0]),
            }
            for identity, row, video in zip(identities, rows, videos, strict=True)
            if video is not None
        )
        result: dict[str, tuple[dict[str, torch.Tensor], ...]] = {}
        for arm, state in generated.items():
            split = tuple(
                {
                    name: (value.detach() if batch_size == 1 else value[index].detach())
                    for name, value in state.items()
                }
                for index in range(batch_size)
            )
            for item in split:
                validate_lora_state(item, self.lora_contract)
            result[arm] = split
        return result

    def last_diagnostic_five_arm_profile(self) -> tuple[dict[str, Any], ...]:
        if not self._last_diagnostic_five_arm_profile:
            raise ExpertManifoldError("rank-reserved diagnostic profile is unavailable")
        return self._last_diagnostic_five_arm_profile

    def release_to_cache(self, cache_contract: Mapping[str, Any]) -> Any:
        from ember.writer.evaluation_runtime import FrozenCachedWriterTaskAdapter

        cached = FrozenCachedWriterTaskAdapter.from_live(
            self,
            cache_contract=cache_contract,
        )
        self.release_generation_assets()
        return cached

    def release_generation_assets(self) -> None:
        """Release video/Writer modules while retaining the source policy."""

        self.store.close()
        del self.store
        del self.language_by_id
        del self.tokenizer
        del self.writer
