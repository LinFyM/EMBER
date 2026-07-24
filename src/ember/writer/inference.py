"""PI05 AS-Writer evaluation authority and per-sample batched LoRA execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import (
    canonical_contract_sha256,
    copy_task_lora_state_,
    lora_state_sha256,
    validate_lora_state,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file
from ember.pi05_target_data import SUITE_ORDER, target_global_task_id
from ember.writer.as_contract import (
    AS_WRITER_LAUNCH_SCHEMA,
    REPO_ROOT,
    inspect_video_data,
    load_writer_config,
    writer_split_roles,
    writer_stage,
)
from ember.writer.checkpoint import validate_writer_checkpoint_files
from ember.pi05_processing import Pi05ForecastPrefixTokenizer
from ember.writer.data import (
    RawTeacherVideoStore,
    WriterFlowNoiseSchedule,
    WriterTaskAuthority,
)
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.model import (
    ACTION_FORECAST_WRITER_CONSTRUCTOR_KEYS,
    CompleteLoRAWriter,
    WriterModelError,
    build_lora_tensor_specs,
)


WRITER_ADAPTER_SCHEMA = "ember_pi05_action_forecast_writer_eval_adapter_v1"
RL_WRITER_ADAPTER_SCHEMA = "ember_pi05_action_forecast_rl_writer_eval_adapter_v1"
WRITER_ADAPTER_SCHEMAS = {WRITER_ADAPTER_SCHEMA, RL_WRITER_ADAPTER_SCHEMA}
WRITER_VIDEO_CONDITIONS = {
    "correct",
    "cross_suite_wrong",
    "shuffled",
    "reversed",
}
WRONG_VIDEO_CONDITIONS = {"cross_suite_wrong"}
WRITER_VIDEO_SCHEDULE = (
    "sha256 first 63 bits of canonical JSON "
    "[ember_pi05_writer_video_v1,seed,suite,task_id,init_state_id] modulo 50"
)


def writer_video_selection_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> int:
    if root_seed < 0 or suite not in SUITE_ORDER or not 0 <= task_id < 10:
        raise WriterModelError("invalid AS-Writer evaluation video seed key")
    if init_state_id < 0:
        raise WriterModelError("invalid AS-Writer evaluation video range")
    encoded = json.dumps(
        ["ember_pi05_writer_video_v1", root_seed, suite, task_id, init_state_id],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def writer_video_demo_index(
    root_seed: int,
    suite: str,
    task_id: int,
    init_state_id: int,
    *,
    demo_count: int = 50,
) -> int:
    """Choose one teacher video independently of queue/worker execution order."""

    if demo_count <= 0:
        raise WriterModelError("invalid AS-Writer evaluation video count")
    return writer_video_selection_seed(root_seed, suite, task_id, init_state_id) % demo_count


def _task_video_mapping(
    task_keys: Sequence[tuple[str, int]],
    task_roles: Mapping[tuple[str, int], str],
    condition: str,
) -> tuple[dict[str, Any], ...]:
    if condition not in WRITER_VIDEO_CONDITIONS or not task_keys:
        raise WriterModelError("invalid AS-Writer evaluation video condition")
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if len(set(normalized)) != len(normalized):
        raise WriterModelError("AS-Writer evaluation tasks are duplicated")
    selected = set(normalized)
    result: list[dict[str, Any]] = []
    roles = sorted({str(task_roles.get(key, "")) for key in normalized})
    if not roles or "" in roles or set(task_roles) != selected:
        raise WriterModelError("AS-Writer evaluation split-role mapping changed")
    for role in roles:
        by_suite = {
            suite: tuple(
                sorted(
                    task_id
                    for name, task_id in normalized
                    if name == suite and task_roles[(name, task_id)] == role
                )
            )
            for suite in SUITE_ORDER
        }
        if any(not values for values in by_suite.values()) or len(
            {len(values) for values in by_suite.values()}
        ) != 1:
            raise WriterModelError(
                "cross-suite Writer control requires equal per-suite panels within each role"
            )
        for suite in SUITE_ORDER:
            for ordinal, task_id in enumerate(by_suite[suite]):
                video_suite = suite
                video_task_id = task_id
                if condition in WRONG_VIDEO_CONDITIONS:
                    video_suite = SUITE_ORDER[
                        (SUITE_ORDER.index(suite) + 1) % len(SUITE_ORDER)
                    ]
                    video_task_id = by_suite[video_suite][ordinal]
                result.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "language_global_task_id": target_global_task_id(suite, task_id),
                        "language_split_role": role,
                        "video_suite": video_suite,
                        "video_task_id": video_task_id,
                        "video_global_task_id": target_global_task_id(
                            video_suite, video_task_id
                        ),
                        "video_split_role": role,
                    }
                )
    return tuple(sorted(result, key=lambda row: (SUITE_ORDER.index(row["suite"]), row["task_id"])))


task_video_mapping = _task_video_mapping


def expected_writer_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_sha256: str,
) -> dict[str, Any]:
    """Build the exact dynamic row fields implied by a sealed adapter contract."""

    if re.fullmatch(r"[0-9a-f]{64}", lora_sha256) is None:
        raise WriterModelError("AS-Writer row lacks a valid LoRA hash")
    if adapter.get("schema_version") not in WRITER_ADAPTER_SCHEMAS:
        raise WriterModelError("unsupported PI05 Writer evaluation adapter")
    matches = [
        row
        for row in adapter.get("task_video_mapping", [])
        if row.get("suite") == suite and int(row.get("task_id", -1)) == task_id
    ]
    if len(matches) != 1:
        raise WriterModelError("AS-Writer row task is outside its video mapping")
    mapping = matches[0]
    schedule = adapter.get("video_schedule", {})
    seed = int(schedule.get("seed", -1))
    count = int(schedule.get("demo_count", -1))
    demo_index = writer_video_demo_index(
        seed, suite, task_id, init_state_id, demo_count=count
    )
    selection_seed = writer_video_selection_seed(seed, suite, task_id, init_state_id)
    return {
        "schema_version": "ember_pi05_writer_episode_evidence_v3",
        "writer_method": adapter.get("writer_method", "as_writer"),
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "writer_checkpoint_axis": adapter["checkpoint"].get(
            "cursor_axis", "optimizer_step"
        ),
        "writer_checkpoint_cursor": int(adapter["checkpoint"]["cursor"]),
        "writer_checkpoint_manifest_sha256": adapter["checkpoint"][
            "manifest_file_sha256"
        ],
        "writer_state_sha256": adapter["checkpoint"]["writer_state_sha256"],
        "lora_contract_sha256": adapter["lora_contract_sha256"],
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_seed_root": seed,
        "teacher_video_selection_seed": selection_seed,
        "video_suite": str(mapping["video_suite"]),
        "video_task_id": int(mapping["video_task_id"]),
        "video_global_task_id": int(mapping["video_global_task_id"]),
        "video_split_role": str(mapping["video_split_role"]),
        "teacher_demo_index": demo_index,
        "wrong_video_map_sha256": adapter["task_video_mapping_sha256"],
        "pairing_sha256": adapter["pairing_sha256"],
        "lora_sha256": lora_sha256,
    }


def validate_writer_episode_evidence(
    adapter: Mapping[str, Any] | None,
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    """Validate all recomputable per-rollout Writer evidence without model access."""

    if adapter is None:
        return row is None
    if not isinstance(row, Mapping):
        return False
    try:
        generation_seconds = float(row.get("writer_generation_seconds", float("nan")))
        expected = expected_writer_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_sha256=str(row.get("lora_sha256", "")),
        )
    except (WriterModelError, TypeError, ValueError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return (
        observed == expected
        and math.isfinite(generation_seconds)
        and generation_seconds >= 0
    )


def _inspect_training_checkpoint(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "checkpoints":
        raise WriterModelError("AS-Writer checkpoint is outside a training run")
    run_root = checkpoint.parent.parent
    contract_path = run_root / "run_contract.json"
    training = read_json(contract_path)
    contract_sha256 = canonical_hash(training)
    world_size = int(training.get("runtime", {}).get("world_size", -1))
    manifest = validate_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=contract_sha256,
    )
    cursor = int(manifest.get("consumed", {}).get("next_step", -1))
    target_manifest = read_json(REPO_ROOT / config["authorities"]["target_data_manifest"]["path"])
    role_ids = target_manifest.get("summary", {}).get("roles", {})
    source_ids = [
        int(task_id)
        for role in writer_split_roles(config)
        for task_id in role_ids.get(role, [])
    ]
    lora = load_pi05_lora_contract(
        REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
    )
    valid = (
        training.get("schema_version") == AS_WRITER_LAUNCH_SCHEMA
        and training.get("stage", "development") == writer_stage(config)
        and training.get("config_sha256") == sha256_file(config_path)
        and training.get("source") == dict(source)
        and training.get("authorities") == config["authorities"]
        and training.get("information_wall") == config["information_wall"]
        and training.get("writer") == config["writer"]
        and training.get("data") == config["data"]
        and training.get("task_ids") == sorted(source_ids)
        and training.get("trainable", {}).get("object")
        == "shared_action_supervised_writer_only"
        and training.get("trainable", {}).get("lora_contract_sha256")
        == canonical_contract_sha256(lora)
        and world_size
        == int(config.get("formal_run", {}).get("expected_world_size", -1))
        and cursor > 0
        and cursor in training.get("runtime", {}).get("checkpoint_steps", [])
        and checkpoint.name == f"step_{cursor:08d}"
    )
    if require_formal:
        valid = (
            valid
            and training.get("mode") == "formal"
            and config.get("formal_run", {}).get("status") == "sealed"
        )
    elif training.get("mode") not in {"profile", "formal"}:
        valid = False
    if not valid:
        raise WriterModelError("AS-Writer training checkpoint authority changed")
    return training, manifest, cursor


def build_writer_evaluation_adapter(
    *,
    schema_version: str,
    writer_method: str,
    config_path: Path,
    checkpoint: Path,
    training: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cursor: int,
    cursor_axis: str,
    video_data: Mapping[str, Any],
    lora_contract_sha256: str,
    mapping: Sequence[Mapping[str, Any]],
    task_keys: Sequence[tuple[str, int]],
    source: Mapping[str, Any],
    video_condition: str,
    video_seed: int,
    forbidden_inputs: Sequence[str],
) -> dict[str, Any]:
    if schema_version not in WRITER_ADAPTER_SCHEMAS or writer_method not in {
        "as_writer",
        "rl_writer",
    }:
        raise WriterModelError("invalid PI05 Writer evaluation method")
    writer_record = manifest.get("files", {}).get("writer.safetensors", {})
    if re.fullmatch(r"[0-9a-f]{64}", str(writer_record.get("sha256", ""))) is None:
        raise WriterModelError("PI05 Writer checkpoint lacks a sealed Writer state")
    mapping_sha256 = canonical_hash(list(mapping))
    pairing_sha256 = canonical_hash(
        {
            "schema_version": "ember_pi05_writer_eval_pairing_v2",
            "writer_method": writer_method,
            "source_run_contract_sha256": source.get("source_run_contract_sha256"),
            "source_checkpoint_manifest_sha256": source.get(
                "checkpoint_manifest_sha256"
            ),
            "writer_checkpoint_manifest_sha256": sha256_file(
                checkpoint / "checkpoint_manifest.json"
            ),
            "task_keys": [list(key) for key in task_keys],
            "video_schedule": WRITER_VIDEO_SCHEDULE,
            "video_seed": video_seed,
        }
    )
    result = {
        "schema_version": schema_version,
        "kind": writer_method,
        "writer_method": writer_method,
        "arm": f"{writer_method}_{video_condition}_video",
        "execution_backend": "per_sample_lora_batched_replan",
        "video_condition": video_condition,
        "writer_input": "task language plus exactly one raw action-hidden teacher video",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "training_run": {
            "path": str(checkpoint.parent.parent),
            "run_contract_file_sha256": sha256_file(
                checkpoint.parent.parent / "run_contract.json"
            ),
            "run_contract_sha256": canonical_hash(training),
            "mode": training["mode"],
            "git_commit": training["git"]["commit"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "cursor": cursor,
            "cursor_axis": cursor_axis,
            "manifest_file_sha256": sha256_file(
                checkpoint / "checkpoint_manifest.json"
            ),
            "manifest_payload_sha256": manifest["canonical_payload_sha256"],
            "writer_state_sha256": writer_record["sha256"],
        },
        "video_data": dict(video_data),
        "lora_contract_sha256": lora_contract_sha256,
        "video_schedule": {
            "algorithm": WRITER_VIDEO_SCHEDULE,
            "seed": video_seed,
            "demo_count": 50,
            "queue_order_independent": True,
            "paired_between_correct_and_wrong": True,
        },
        "wrong_video_mapping": (
            "identity"
            if video_condition not in WRONG_VIDEO_CONDITIONS
            else "same role-panel ordinal in the next suite cyclically"
        ),
        "task_video_mapping_sha256": mapping_sha256,
        "task_video_mapping": list(mapping),
        "pairing_sha256": pairing_sha256,
        "writer_forbidden_tensor_inputs": list(forbidden_inputs),
        "teacher_action_values_read_by_evaluator": 0,
    }
    return result


def inspect_as_writer_evaluation(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_seed: int,
    require_formal: bool,
) -> dict[str, Any]:
    """Seal a Writer checkpoint and raw-video authority before queue creation."""

    config_path = config_path.resolve()
    checkpoint = checkpoint.resolve()
    video_data_root = video_data_root.resolve()
    config = load_writer_config(config_path)
    target_manifest = read_json(
        REPO_ROOT / str(config["authorities"]["target_data_manifest"]["path"])
    )
    target_by_key = {
        (str(row["suite"]), int(row["task_id"])): row
        for row in target_manifest.get("tasks", [])
    }
    normalized_keys = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if set(normalized_keys) - set(target_by_key):
        raise WriterModelError("AS-Writer evaluation task is outside target40")
    task_roles = {key: str(target_by_key[key]["split_role"]) for key in normalized_keys}
    mapping = _task_video_mapping(normalized_keys, task_roles, video_condition)
    needed_task_ids = tuple(
        sorted(
            {
                int(row["language_global_task_id"])
                for row in mapping
            }
            | {int(row["video_global_task_id"]) for row in mapping}
        )
    )
    training, manifest, cursor = _inspect_training_checkpoint(
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        source=source,
        require_formal=require_formal,
    )
    video_data = inspect_video_data(
        video_data_root, config, needed_task_ids, verify_hashes=False
    )
    training_video = training.get("video_data", {})
    if (
        training_video.get("root") != video_data.get("root")
        or training_video.get("dataset") != video_data.get("dataset")
        or training_video.get("target_data_manifest_file_sha256")
        != video_data.get("target_data_manifest_file_sha256")
    ):
        raise WriterModelError("AS-Writer checkpoint and video data disagree")
    lora = load_pi05_lora_contract(
        REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
    )
    return build_writer_evaluation_adapter(
        schema_version=WRITER_ADAPTER_SCHEMA,
        writer_method="as_writer",
        config_path=config_path,
        checkpoint=checkpoint,
        training=training,
        manifest=manifest,
        cursor=cursor,
        cursor_axis="optimizer_step",
        video_data=video_data,
        lora_contract_sha256=canonical_contract_sha256(lora),
        mapping=mapping,
        task_keys=normalized_keys,
        source=source,
        video_condition=video_condition,
        video_seed=video_seed,
        forbidden_inputs=config["information_wall"]["writer_forbidden_inputs"],
    )


@dataclass(frozen=True)
class PreparedWriterLoRA:
    state: Mapping[str, torch.Tensor]
    evidence: dict[str, Any]


class FrozenWriterTaskAdapter:
    """Generate one PI05 task LoRA per rollout and batch distinct policy adapters."""

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
        if kind == "rl_writer":
            raise WriterModelError(
                "RL-Writer has not yet been trained under Action-Forecast and must "
                "be retrained before evaluation"
            )
        if kind != "as_writer":
            raise WriterModelError("unknown PI05 Writer evaluation kind")
        observed = inspect_as_writer_evaluation(
            config_path=Path(evaluation_adapter["config"]["path"]),
            checkpoint=Path(evaluation_adapter["checkpoint"]["path"]),
            video_data_root=Path(evaluation_adapter["video_data"]["root"]),
            source=source,
            task_keys=task_keys,
            video_condition=str(evaluation_adapter["video_condition"]),
            video_seed=int(evaluation_adapter["video_schedule"]["seed"]),
            require_formal=require_formal,
        )
        config = load_writer_config(Path(observed["config"]["path"]))
        if observed != dict(evaluation_adapter):
            raise WriterModelError("PI05 Writer evaluation artifacts changed after prepare")
        lora = load_pi05_lora_contract(
            REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
        )
        template = prepare_frozen_writer_policy(policy, lora)
        writer_values = {
            key: value
            for key, value in config["writer"].items()
            if key in ACTION_FORECAST_WRITER_CONSTRUCTOR_KEYS
        }
        bridge = policy.model.paligemma_with_expert
        writer = CompleteLoRAWriter(
            build_lora_tensor_specs(template),
            template_state=template,
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model,
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
        self.tokenizer = Pi05ForecastPrefixTokenizer(
            tokenizer_path,
            int(source_config["features"]["tokenizer_max_length"]),
            str(device),
        )
        self.policy = policy
        self.writer = writer
        self.lora_contract = lora
        self.identity_state = {
            name: value.detach().clone() for name, value in template.items()
        }
        self.device = device
        self.evaluation_adapter = dict(observed)
        self.batched_lora = BatchedLoRAInference(policy, lora)
        self._physical_lora_is_identity = True
        self._state_cache: dict[tuple[Any, ...], Mapping[str, torch.Tensor]] = {}

    @torch.inference_mode()
    def prepare_episode(
        self, *, suite: str, task_id: int, init_state_id: int
    ) -> PreparedWriterLoRA:
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
        language_tokens, language_mask, state_positions = self.tokenizer(
            [writer_language]
        )
        frames = torch.from_numpy(teacher.frames).to(
            self.device, non_blocking=True
        )
        frame_indices = torch.from_numpy(teacher.frame_indices).to(
            self.device, non_blocking=True
        )
        condition = str(self.evaluation_adapter["video_condition"])
        if condition == "reversed":
            frames = frames.flip(0)
        elif condition == "shuffled":
            generator = torch.Generator(device="cpu").manual_seed(
                int(row["teacher_video_selection_seed"])
            )
            permutation = torch.randperm(
                frames.shape[0],
                generator=generator,
            ).to(self.device)
            frames = frames.index_select(0, permutation)
        video_offsets = torch.tensor(
            [0, frames.shape[0]], dtype=torch.long, device=self.device
        )
        flow_noise = WriterFlowNoiseSchedule(
            seed=int(row["teacher_video_selection_seed"])
        ).noise_for_visit(0, device=self.device)[None]
        cache_key = (
            int(row["language_global_task_id"]),
            int(row["video_global_task_id"]),
            int(row["teacher_demo_index"]),
            condition,
            int(row["teacher_video_selection_seed"]),
        )
        started = time.monotonic()
        # The forecast path must always see the frozen source policy,
        # never the task adapter left installed by the preceding rollout.
        copy_task_lora_state_(
            self.policy, self.identity_state, self.lora_contract
        )
        self._physical_lora_is_identity = True
        state = self._state_cache.get(cache_key)
        if state is None:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                state = self.writer(
                    frames,
                    frame_indices,
                    video_offsets,
                    language_tokens,
                    language_mask,
                    state_positions,
                    flow_noise,
                    policy=self.policy,
                )
            self._state_cache[cache_key] = state
        validate_lora_state(state, self.lora_contract)
        digest = lora_state_sha256(state)
        generation_seconds = time.monotonic() - started
        if not math.isfinite(generation_seconds) or generation_seconds < 0:
            raise WriterModelError("PI05 Writer generation timing is invalid")
        evidence = {
            **row,
            "lora_sha256": digest,
            "writer_generation_seconds": generation_seconds,
        }
        return PreparedWriterLoRA(state=state, evidence=evidence)

    @torch.inference_mode()
    def install(self, prepared: PreparedWriterLoRA) -> None:
        validate_lora_state(prepared.state, self.lora_contract)
        copy_task_lora_state_(self.policy, prepared.state, self.lora_contract)
        self._physical_lora_is_identity = False

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        prepared: Sequence[PreparedWriterLoRA],
        batch: Mapping[str, torch.Tensor],
        *,
        noise: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Run one native policy batch with a distinct Writer LoRA per sample."""

        if len(prepared) != int(noise.shape[0]):
            raise WriterModelError("Writer LoRA batch and policy noise batch differ")
        if not self._physical_lora_is_identity:
            copy_task_lora_state_(
                self.policy, self.identity_state, self.lora_contract
            )
            self._physical_lora_is_identity = True
        with self.batched_lora.activate([item.state for item in prepared]):
            return self.policy.predict_action_chunk(
                dict(batch),
                noise=noise,
                num_steps=num_steps,
            )
