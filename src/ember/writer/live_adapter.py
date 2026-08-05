"""Live Writer LoRA generation for rollout evaluation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.lora import (
    copy_task_lora_state_,
    lora_state_sha256,
    validate_lora_state,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.writer.architecture import CONDITION_KERNEL_WRITER_CONSTRUCTOR_KEYS
from ember.writer.condition_kernel import load_condition_authority
from ember.writer.as_contract import REPO_ROOT, load_writer_config
from ember.writer.data import RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.inference import (
    expected_writer_episode_evidence,
    inspect_as_writer_evaluation,
    writer_shuffled_frame_permutation,
)
from ember.writer.lora_rollout import WriterLoRARolloutAdapter
from ember.writer.model import (
    CompleteLoRAWriter,
    WriterModelError,
    build_lora_tensor_specs,
)


@dataclass(frozen=True)
class PreparedWriterLoRA:
    state: Mapping[str, torch.Tensor]
    evidence: dict[str, Any]


class FrozenWriterTaskAdapter(WriterLoRARolloutAdapter):
    """Generate batches of PI05 task LoRAs before rollout begins."""

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
        kind = str(evaluation_adapter.get("kind", "as_writer"))
        if kind not in {"as_writer", "rl_writer"}:
            raise WriterModelError("unknown PI05 Writer evaluation kind")
        common = {
            "config_path": Path(evaluation_adapter["config"]["path"]),
            "checkpoint": Path(evaluation_adapter["checkpoint"]["path"]),
            "video_data_root": Path(evaluation_adapter["video_data"]["root"]),
            "source": source,
            "task_keys": task_keys,
            "video_condition": str(evaluation_adapter["video_condition"]),
            "video_seed": int(evaluation_adapter["video_schedule"]["seed"]),
            "video_sampling_mode": (
                str(evaluation_adapter["video_schedule"]["sampling_mode"])
                if "sampling_mode" in evaluation_adapter["video_schedule"]
                else None
            ),
            "require_formal": require_formal,
        }
        if kind == "as_writer":
            observed = inspect_as_writer_evaluation(**common)
            config = load_writer_config(Path(observed["config"]["path"]))
        else:
            from ember.rl_writer.contract import authority_path, load_rl_writer_config
            from ember.rl_writer.inference import inspect_rl_writer_evaluation

            observed = inspect_rl_writer_evaluation(**common)
            rl_config = load_rl_writer_config(Path(observed["config"]["path"]))
            config = load_writer_config(authority_path(rl_config, "as_writer_config"))
        if observed != dict(evaluation_adapter):
            raise WriterModelError("PI05 Writer evaluation artifacts changed after prepare")
        lora = load_pi05_lora_contract(
            REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
        )
        template = prepare_frozen_writer_policy(policy, lora)
        writer_values = {
            key: value
            for key, value in config["writer"].items()
            if key in CONDITION_KERNEL_WRITER_CONSTRUCTOR_KEYS
        }
        bridge = policy.model.paligemma_with_expert
        writer = CompleteLoRAWriter(
            build_lora_tensor_specs(template),
            template_state=template,
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model,
            condition_authority=load_condition_authority(
                str(REPO_ROOT / config["authorities"]["condition_address"]["path"])
            ),
            **writer_values,
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
        needed = tuple(
            sorted(
                {int(row["language_global_task_id"]) for row in observed["task_video_mapping"]}
                | {int(row["video_global_task_id"]) for row in observed["task_video_mapping"]}
            )
        )
        target = read_json(
            REPO_ROOT / str(config["authorities"]["target_data_manifest"]["path"])
        )
        root = Path(observed["video_data"]["root"])
        by_id = {
            int(record["global_task_id"]): record for record in target["tasks"]
        }
        authorities = []
        for global_task_id in needed:
            record = by_id[global_task_id]
            authorities.append(
                WriterTaskAuthority(
                    task_id=global_task_id,
                    language=str(record["language"]),
                    path=root / str(record["hdf5"]["relative_path"]),
                    expected_bytes=int(record["hdf5"]["bytes"]),
                )
            )
        self.store = RawTeacherVideoStore(
            authorities,
            frame_stride=int(config["writer"]["frame_stride"]),
            max_open_files=2,
        )
        self.language_by_id = {
            authority.task_id: authority.language for authority in authorities
        }
        source_config = read_json(
            REPO_ROOT / str(config["authorities"]["source_base_config"]["path"])
        )
        self.tokenizer = Pi05TeacherPrefixTokenizer(
            tokenizer_path,
            int(source_config["features"]["tokenizer_max_length"]),
            str(device),
        )
        self.writer = writer
        self._initialize_rollout(
            policy=policy,
            lora_contract=lora,
            identity_state=template,
            evaluation_adapter=observed,
            device=device,
        )

    def _episode_inputs(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, str]:
        placeholder = "0" * 64
        row = expected_writer_episode_evidence(
            self.evaluation_adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_sha256=placeholder,
        )
        teacher = self.store.load(
            int(row["video_global_task_id"]),
            int(row["teacher_demo_index"]),
        )
        writer_language = self.language_by_id[int(row["language_global_task_id"])]
        frames = torch.from_numpy(teacher.frames).to(
            self.device, non_blocking=True
        )
        frame_indices = torch.from_numpy(teacher.frame_indices).to(
            self.device, non_blocking=True
        )
        condition = str(self.evaluation_adapter["video_condition"])
        if condition == "reversed":
            frames = frames.flip(0)
        elif condition in {"shuffled", "shuffled_keep_first"}:
            permutation = writer_shuffled_frame_permutation(
                frames.shape[0],
                int(row["teacher_video_order_seed"]),
                keep_first=condition == "shuffled_keep_first",
            ).to(self.device)
            frames = frames.index_select(0, permutation)
        return row, frames, frame_indices, writer_language

    @torch.inference_mode()
    def prepare_episodes(
        self, identities: Sequence[Mapping[str, Any]]
    ) -> tuple[PreparedWriterLoRA, ...]:
        if not identities:
            raise WriterModelError("Writer generation batch is empty")
        inputs = [
            self._episode_inputs(
                suite=str(identity["suite"]),
                task_id=int(identity["task_id"]),
                init_state_id=int(identity["init_state_id"]),
            )
            for identity in identities
        ]
        rows, frame_batches, index_batches, languages = zip(*inputs, strict=True)
        language_tokens, language_mask, task_span_mask = self.tokenizer(list(languages))
        frames = torch.cat(frame_batches, dim=0)
        frame_indices = torch.cat(index_batches, dim=0)
        offsets = [0]
        for batch in frame_batches:
            offsets.append(offsets[-1] + int(batch.shape[0]))
        video_offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        started = time.monotonic()
        copy_task_lora_state_(self.policy, self.identity_state, self.lora_contract)
        self._physical_lora_is_identity = True
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            generated = self.writer(
                frames,
                frame_indices,
                video_offsets,
                language_tokens,
                language_mask,
                task_span_mask,
                policy=self.policy,
            )
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise WriterModelError("PI05 Writer generation timing is invalid")
        batch_size = len(identities)
        prepared: list[PreparedWriterLoRA] = []
        for row_index, row in enumerate(rows):
            state: dict[str, torch.Tensor] = {}
            for name, value in generated.items():
                expected_shape = tuple(self.identity_state[name].shape)
                if batch_size == 1 and tuple(value.shape) == expected_shape:
                    selected = value
                elif tuple(value.shape) == (batch_size, *expected_shape):
                    selected = value[row_index]
                else:
                    raise WriterModelError(
                        f"Writer generated an invalid batched LoRA tensor: {name}"
                    )
                state[name] = selected.detach()
            validate_lora_state(state, self.lora_contract)
            evidence = {
                **row,
                "lora_sha256": lora_state_sha256(state),
                "writer_generation_seconds": elapsed / batch_size,
            }
            prepared.append(PreparedWriterLoRA(state=state, evidence=evidence))
        return tuple(prepared)

    @torch.inference_mode()
    def prepare_episode(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> PreparedWriterLoRA:
        return self.prepare_episodes(
            ({"suite": suite, "task_id": task_id, "init_state_id": init_state_id},)
        )[0]

    def release_to_cache(self, cache_contract: Mapping[str, Any]) -> Any:
        """Release only Writer-owned modules while retaining the source policy."""

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
        return cached
