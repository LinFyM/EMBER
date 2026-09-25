"""Inspect and execute independently compiled, strictly paired episode LoRAs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import (copy_task_lora_state_, expected_lora_state_shapes, identity_lora_state,
                        inject_task_lora, task_lora_state_dict, validate_lora_state)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.task_protocol import load_task_authorities
from ember.writer.materialization import (BANK_KIND, BANK_SCHEMA, adapter_metadata, condition_id,
    file_record, frozen_authority, inspect_writer_checkpoint, method_metadata, planned_episodes,
    selection_contract, source_matches)
from ember.writer.video_controls import (CONTROL_ARMS, control_provenance, controlled_frames,
                                         inspect_diagnostic_contract)
from ember.writer.relational_contract import CONFIG_SCHEMA as RELATIONAL_CONFIG_SCHEMA, registered_stage1_bank_panel
from ember.writer.language_content_contract import validate_evaluation_bank


EVALUATION_SCHEMA = "ember_video_writer_eval_adapter_v1"
EPISODE_SCHEMA = "ember_video_writer_episode_v1"


def validate_task_scope(rows: Sequence[Mapping[str, Any]], role: str, asset_root: Path,
                        protocol_path: str | None = None,
                        support_slot_credit: Mapping[str, Any] | None = None) -> None:
    if role not in {"development_train", "nonheld_meta", "validation", "test"}:
        raise ValueError("video Writer evaluation requires a registered target split")
    protocol, manifest = load_task_authorities(asset_root, protocol_path)
    canonical = {int(row["global_task_id"]): row for row in manifest["tasks"]}
    split = "train" if role in {"development_train", "nonheld_meta"} else role
    if role == "nonheld_meta":
        authority = protocol.get("study_authority")
        if authority != "configs/relational_support_causality_v1/experiment_spec.json":
            raise ValueError("support bank requires the registered study protocol")
        spec = read_json(asset_root / authority)
        allowed_sets = {tuple(arm["support_eval_global_ids"]) for arm in spec["arms"]}
        if support_slot_credit is not None:
            from ember.writer.support_slot_credit import SPEC_PATH as SLOT_SPEC

            if (support_slot_credit.get("schema_version") != "ember_support_slot_bank_v1"
                    or support_slot_credit.get("phase") != "donor_fm"
                    or support_slot_credit.get("study_spec") != str(Path(__file__).resolve().parents[3] / SLOT_SPEC)
                    or tuple(row["global_task_id"] for row in rows) != (76, 77)):
                raise ValueError("support-slot FM bank is outside its exact nonheld task pair")
            allowed_sets.add((76, 77))
        if tuple(row["global_task_id"] for row in rows) not in allowed_sets:
            raise ValueError("support bank is outside the four-task registered arm subset")
        expected = {("libero_90", task - 40) for group in allowed_sets for task in group}
    else:
        expected = {(suite, task) for suite, roles in protocol["split"]["suites"].items() for task in roles[split]}
    keys = [(str(row["suite"]), int(row["task_id"])) for row in rows]
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= expected or (split != "train" and set(keys) != expected):
        raise ValueError("task bank crosses the fixed split or omits validation8/test8 tasks")
    for row in rows:
        actual = canonical.get(int(row["global_task_id"]))
        if actual is None or any(row.get(key) != actual[key] for key in ("suite", "task_id", "language", "split_role")):
            raise ValueError("task language/identity differs from the fixed target authority")
        source = asset_root / "data/datasets" / manifest["dataset"]["revision"] / actual["hdf5"]["relative_path"]
        if row["teacher_source"] != {"path": str(source.resolve()), "bytes": int(actual["hdf5"]["bytes"])}:
            raise ValueError("teacher source provenance differs from the fixed task")


def _selection(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = selection_contract(role=value["evaluation_role"], task_ids=value["task_ids"],
        cardinality=value["K"], arm=value["arm"], mode=value["mode"], seed=value["seed"],
        init_state_ids=value["init_state_ids"], video_pool=value["video_pool"], fixed_videos=value["fixed_videos"])
    if dict(value) != normalized:
        raise ValueError("video selection contract changed")
    return normalized


def _inspect_adapter_file(condition: Mapping[str, Any], checkpoint: Mapping[str, Any], lora) -> None:
    path = Path(condition["adapter"]["path"])
    if not path.is_file() or condition["adapter"] != file_record(path):
        raise ValueError("materialized adapter file changed")
    expected = expected_lora_state_shapes(lora)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if handle.metadata() != adapter_metadata(condition["condition_id"], checkpoint) or set(handle.keys()) != set(expected):
            raise ValueError("adapter condition/checkpoint identity or complete target set changed")
        for name, shape in expected.items():
            value = handle.get_slice(name)
            if tuple(value.get_shape()) != shape or value.get_dtype() != "F32":
                raise ValueError("materialized adapter must retain complete FP32 native shapes")
            if condition.get("video_control", {}).get("identity_zero_delta") and torch.count_nonzero(handle.get_tensor(name)):
                raise ValueError("no-video adapter must be the actual complete zero-delta identity")


def _validate_video_frames(videos, demos, lengths, control=None) -> None:
    if [video["demo_index"] for video in videos] != demos:
        raise ValueError("actual teacher videos differ from the selected K-set")
    for video in videos:
        raw = int(video["raw_frame_count"])
        if raw <= 0 or raw != lengths[video["demo_index"]]:
            raise ValueError("teacher full-frame count differs from the data authority")
        indices = list(range(0, raw, 5))
        if indices[-1] != raw - 1:
            indices.append(raw - 1)
        if video["frame_indices"] != indices or video["sampled_frame_count"] != len(indices):
            raise ValueError("teacher stride5/full-frame provenance changed")
        if control is not None:
            _, _, expected = controlled_frames(indices, control=control, demo=video["demo_index"])
            if any(video.get(key) != value for key, value in expected.items()):
                raise ValueError("real source frames, displayed order or complete-forward control provenance changed")


def _inspect_conditions(manifest: Mapping[str, Any], root: Path, lora) -> None:
    conditions = {row["condition_id"]: row for row in manifest["conditions"]}
    tasks = {row["global_task_id"]: row for row in manifest["tasks"]}
    _, authority = load_task_authorities(Path(manifest["asset_root"]), manifest.get("task_protocol"))
    lengths = {row["global_task_id"]: row["demonstrations"]["episode_lengths"] for row in authority["tasks"]}
    conditional = manifest.get("method", {}).get("schema_version") == "ember_conditional_compilation_method_v1"
    parameterization = manifest.get("method", {}).get("parameterization", "video_writer")
    uses_video = parameterization == "video_writer"
    referenced = {episode["condition_id"] for row in tasks.values() for episode in row["episodes"]}
    if len(conditions) != len(manifest["conditions"]) or set(conditions) != referenced:
        raise ValueError("condition bank contains duplicates, missing rows, or unused adapters")
    for key, condition in conditions.items():
        task = tasks.get(condition["global_task_id"])
        if task is None:
            raise ValueError("materialized condition task is outside its registered bank")
        demos = condition["teacher_demo_indices"]
        arm = manifest["arm"]
        control = control_provenance(manifest["selection"], condition["global_task_id"], tasks) if arm in CONTROL_ARMS else None
        no_video = arm == "no_video"
        donor = control["video_global_task_id"] if control else condition["global_task_id"]
        identity = {field: task[field] for field in ("suite", "task_id", "language")}
        expected_invocations = (0 if no_video or parameterization == "direct_lora" else
                                condition.get("writer_invocations", 0) if conditional and parameterization == "language_writer" else
                                1)
        identity.update(condition_id=condition_id(task["global_task_id"], demos, arm=arm, video_task=donor),
                        teacher_demo_indices=sorted(set(demos)), video_control=control,
                        writer_invocations=expected_invocations, single_complete_rank16=True)
        if conditional:
            identity.update(parameterization=parameterization,
                            teacher_video_values_read=len(demos) if uses_video and not no_video else 0)
        if (any(condition.get(field) != value for field, value in identity.items())
                or len(demos) != (0 if no_video else manifest["selection"]["K"])
                or Path(condition["adapter"]["path"]).resolve() != root / f"{key}.safetensors"):
            raise ValueError("materialized condition task/video/adapter provenance changed")
        if uses_video and not no_video:
            _validate_video_frames(condition["teacher_videos"], demos,
                                   lengths[donor] if donor is not None else (), control)
        elif condition.get("teacher_videos") != []:
            raise ValueError("non-video parameterization must not claim teacher RGB evidence")
        if conditional and parameterization == "language_writer" and condition.get("writer_invocations") not in (0, 1):
            raise ValueError("cached language Writer invocation count must be zero or one per condition")
        for episode in task["episodes"]:
            if episode["condition_id"] == key and episode["teacher_demo_indices"] != demos:
                raise ValueError("episode mapping changed its actual teacher K-set")
        _inspect_adapter_file(condition, manifest["writer_checkpoint"], lora)


def _validate_round(selection, rows, require_formal) -> None:
    if require_formal and selection["evaluation_role"] == "validation" and (
            selection["init_state_ids"] != list(range(50)) or selection["video_pool"] != list(range(50))):
        raise ValueError("formal validation requires 50 init states and all 50 teacher videos per task")
    if selection["K"] != 1 or selection["mode"] != "per_init_ordinal":
        return
    for row in rows:
        episodes = row["episodes"]
        fields = ("paired_correct_demos", "paired_other_demos") if selection["arm"] == "no_video" else (
            "teacher_demo_indices", "paired_correct_demos", "paired_other_demos")
        if selection["arm"] == "no_video" and any(episode["teacher_demo_indices"] for episode in episodes):
            raise ValueError("no-video episodes retain paired ordinals but cannot claim actual teacher reads")
        for field in fields:
            videos = [episode[field] for episode in episodes]
            if any(len(value) != 1 for value in videos):
                raise ValueError("K1 episode must identify exactly one actual teacher video")
            ids = [value[0] for value in videos]
            if len(set(ids)) != len(ids) or not set(ids) <= set(selection["video_pool"]):
                raise ValueError("teacher videos repeat within a task/arm evaluation round")
            if len(episodes) == len(selection["video_pool"]) and set(ids) != set(selection["video_pool"]):
                raise ValueError("evaluation round omits an allowed teacher video")
        if any(episode["paired_correct_demos"] == episode["paired_other_demos"] for episode in episodes):
            raise ValueError("same-task-other must differ from correct for every init state")


def _inspect_scope(manifest, source, task_keys, evaluation_role, task_init_state_ids, require_formal,
                   native_reader_transfer_cell=None, support_slot_model=None,
                   language_content_panel=None) -> None:
    role = manifest["evaluation_role"]
    selection = _selection(manifest["selection"])
    rows = manifest["tasks"]
    keys = [(str(row["suite"]), int(row["task_id"])) for row in rows]
    if (manifest.get("schema_version") != BANK_SCHEMA or manifest.get("kind") != BANK_KIND
            or manifest.get("status") != "sealed" or role != evaluation_role
            or manifest.get("arm") != selection["arm"] or role != selection["evaluation_role"]
            or len(task_keys) != len(set(task_keys)) or set(task_keys) != set(keys)
            or [row["global_task_id"] for row in rows] != selection["task_ids"]
            or manifest.get("single_complete_rank16") is not True
            or not frozen_authority(manifest["materialization_git"])
            or not source_matches(manifest["source"], source)):
        raise ValueError("video Writer bank scope/source/commit changed")
    scope_args = (rows, role, Path(manifest["asset_root"]), manifest.get("task_protocol"))
    if manifest.get("support_slot_credit") is None:
        validate_task_scope(*scope_args)
    else:
        validate_task_scope(*scope_args, support_slot_credit=manifest["support_slot_credit"])
    _validate_round(selection, rows, require_formal)
    for row in rows:
        if row["episodes"] != planned_episodes(selection, row["global_task_id"]):
            raise ValueError("episode video ordinal or deterministic pairing changed")
        if task_init_state_ids is not None:
            requested = tuple(task_init_state_ids.get((row["suite"], row["task_id"]), ()))
            if (requested != tuple(selection["init_state_ids"])
                    and not ((native_reader_transfer_cell or support_slot_model)
                             and requested in ((0,), tuple(range(1, 50))))
                    and not (language_content_panel is not None
                             and language_content_panel["model"] == "B630"
                             and language_content_panel["kind"] == "held_correct"
                             and requested == tuple(range(10)))):
                raise ValueError("bank and evaluator must use the same exact fixed init states")


def validate_information_wall(manifest) -> None:
    wall = manifest["information_wall"]
    no_video = manifest["arm"] == "no_video"
    if manifest.get("method", {}).get("schema_version") == "ember_conditional_compilation_method_v1":
        parameterization = manifest["method"]["parameterization"]
        uses_video = parameterization == "video_writer" and not no_video
        conditions = manifest["conditions"]
        expected_inputs = ([] if no_video or parameterization == "direct_lora" else
                           ["exact task language"] if parameterization == "language_writer" else
                           ["exact task language", "ordered RGB videos", "original frame indices"])
        invocation_count = sum(int(row.get("writer_invocations", 0)) for row in conditions)
        video_read_count = sum(int(row.get("teacher_video_values_read", 0)) for row in conditions)
        required = {"parameterization": parameterization, "deployment_inputs": expected_inputs,
                    "teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
                    "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
                    "materialization_rgb_video_reads": len(conditions) if uses_video else 0,
                    "teacher_video_values_read": video_read_count,
                    "parameterization_invocations": invocation_count,
                    "deployment_frozen_source_vjp": False, "deployment_loss_or_optimizer": False,
                    "outcome_dependent_video_selection": False,
                    "shuffled_reversed_wrong_no_video": manifest["arm"] in CONTROL_ARMS}
        if (any(wall.get(key) != value for key, value in required.items())
                or video_read_count != (len(conditions) if uses_video else 0)
                or (parameterization == "direct_lora" and invocation_count != 0)
                or (parameterization == "video_writer" and invocation_count != (0 if no_video else len(conditions)))
                or (parameterization == "language_writer" and not 0 < invocation_count <= len(conditions))):
            raise ValueError("conditional Writer information wall changed")
        return
    required = {"teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
                "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
                "writer_invocations_per_unique_condition": 0 if no_video else 1,
                "total_writer_invocations": 0 if no_video else len(manifest["conditions"]),
                "outcome_dependent_video_selection": False,
                "shuffled_reversed_wrong_no_video": manifest["arm"] in CONTROL_ARMS}
    if manifest["arm"] in CONTROL_ARMS or manifest.get("evaluation_role") == "test":
        required.update(materialization_rgb_video_reads=0 if no_video else len(manifest["conditions"]),
                        deployment_frozen_source_vjp=False, deployment_loss_or_optimizer=False)
    if no_video:
        required["deployment_inputs"] = []
    if any(wall.get(key) != value for key, value in required.items()):
        raise ValueError("video Writer information wall changed")


def _validate_registered_bank_origin(manifest, path, run, checkpoint,
                                     native_reader_transfer_cell, support_slot_model,
                                     support_slot_phase="final"):
    if support_slot_model is not None:
        from ember.pi05_eval.support_slot_credit import validate_bank

        current = git_state(Path(__file__).resolve().parents[3])
        validate_bank(manifest, path, support_slot_model, current["commit"], run, checkpoint,
                      phase=support_slot_phase)
        return None
    if native_reader_transfer_cell is not None:
        from ember.pi05_eval.native_reader_transfer import validate_bank

        current = git_state(Path(__file__).resolve().parents[3])
        validate_bank(manifest, path, native_reader_transfer_cell, current["commit"])
        return None
    if run["config"].get("schema_version") != RELATIONAL_CONFIG_SCHEMA:
        return None
    from ember.writer.relational_contract import stage1_bank_materialization_commit

    panel = registered_stage1_bank_panel(run["config"], manifest["selection"],
        checkpoint=Path(checkpoint["path"]), output=path.parent)
    current = git_state(Path(__file__).resolve().parents[3])
    bank_commit = stage1_bank_materialization_commit(
        panel_id=panel["id"], manifest_path=path, evaluation_commit=current["commit"])
    if (run["git"]["commit"] != "7dc95edbba00cf61439700d77fb321eb8df95c07"
            or manifest.get("registered_stage1_panel_id") != panel["id"]
            or manifest["materialization_git"]["commit"] != bank_commit):
        raise ValueError("stage1 bank training, panel or E materialization identity changed")
    return panel


def inspect_horizon_writer_bank(
    *, manifest_path: Path, source: Mapping[str, Any], task_keys: Sequence[tuple[str, int]],
    evaluation_role: str, require_formal: bool,
    task_init_state_ids: Mapping[tuple[str, int], Sequence[int]] | None = None,
    native_reader_transfer_cell: str | None = None,
    support_slot_model: str | None = None,
    support_slot_phase: str = "final",
    language_content_panel: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate condition provenance and paired row coverage before workers start."""
    try:
        path = manifest_path.resolve()
        manifest = read_json(path)
        _inspect_scope(manifest, source, task_keys, evaluation_role, task_init_state_ids, require_formal,
                       native_reader_transfer_cell, support_slot_model, language_content_panel)
        run, checkpoint = inspect_writer_checkpoint(Path(manifest["writer_checkpoint"]["path"]))
        if language_content_panel is not None:
            validate_evaluation_bank(language_content_panel, path, manifest, run,
                                     git_state(Path(__file__).resolve().parents[3])["commit"])
        panel = _validate_registered_bank_origin(manifest, path, run, checkpoint,
                                                  native_reader_transfer_cell, support_slot_model,
                                                  support_slot_phase)
        if manifest.get("task_protocol") != run["config"]["data"].get("protocol"):
            raise ValueError("bank task protocol differs from its trained Writer")
        if checkpoint != manifest["writer_checkpoint"] or manifest["method"] != method_metadata(run, manifest["arm"]) or not source_matches(run["source"], source):
            raise ValueError("Writer checkpoint or method provenance changed")
        diagnostic = inspect_diagnostic_contract(manifest.get("diagnostic_contract"), selection=manifest["selection"],
                                                checkpoint=checkpoint, run=run, asset_root=Path(manifest["asset_root"]))
        if manifest.get("diagnostic_contract") != diagnostic:
            raise ValueError("frozen diagnostic contract changed")
        if (native_reader_transfer_cell is None and support_slot_model is None
                and run["config"].get("schema_version") == RELATIONAL_CONFIG_SCHEMA
                and panel["kind"] == "target_other"):
            expected_ids = {episode["condition_id"] for episode in planned_episodes(manifest["selection"], 21)}
            compilation = manifest.get("compilation", {})
            if (len(expected_ids) != 50 or compilation.get("new_conditions") != 0
                    or compilation.get("reused_conditions") != 50
                    or set(compilation.get("reused_condition_ids", [])) != expected_ids
                    or compilation.get("reuse_manifest") != diagnostic["paired_correct_manifest"]):
                raise ValueError("stage1 Goal21 other must only reuse its paired correct LoRAs")
        lora_path = Path(manifest["lora_contract"]["path"])
        if manifest["lora_contract"] != file_record(lora_path):
            raise ValueError("LoRA topology authority changed")
        lora = load_pi05_lora_contract(lora_path)
        if lora.rank != lora.alpha or lora.rank != 16 or len(lora.targets) != 38 or lora.dropout != 0:
            raise ValueError("evaluation requires one complete 38-target rank16 LoRA")
        validate_information_wall(manifest)
        _inspect_conditions(manifest, path.parent, lora)
        return {**manifest, "schema_version": EVALUATION_SCHEMA, "manifest": file_record(path)}
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise Pi05EvaluationError(str(error)) from error


def episode_evidence(adapter: Mapping[str, Any], task: Mapping[str, Any], episode: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {row["condition_id"]: row for row in adapter["conditions"]}
    condition = conditions[episode["condition_id"]]
    evidence = {"schema_version": EPISODE_SCHEMA, **dict(condition), **dict(episode),
            "selection_seed": adapter["selection"]["seed"], "selection_mode": adapter["selection"]["mode"],
            "K": adapter["selection"]["K"], "arm": adapter["arm"],
            "writer_checkpoint": dict(adapter["writer_checkpoint"]), "method": dict(adapter["method"]),
            "source_checkpoint": adapter["source"]["checkpoint"], "global_task_id": task["global_task_id"]}
    if "diagnostic_contract" in adapter:
        evidence["diagnostic_contract"] = dict(adapter["diagnostic_contract"])
    return evidence


def validate_horizon_writer_episode(adapter, evidence, *, suite: str, task_id: int, init_state_id: int) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    for task in adapter.get("tasks", ()):
        if (task["suite"], task["task_id"]) == (suite, task_id):
            for episode in task["episodes"]:
                if episode["init_state_id"] == init_state_id:
                    return dict(evidence) == episode_evidence(adapter, task, episode)
    return False


@dataclass(frozen=True)
class PreparedHorizonLoRA:
    key: str
    evidence: dict[str, Any]


class FrozenHorizonWriterAdapter:
    """Execution only: no observer, video, Meta, or learned Writer is loaded."""

    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal,
                 readout_intervention=None) -> None:
        del device, require_formal
        adapter = evaluation_adapter
        self.records = {(row["suite"], row["task_id"]): row for row in adapter["tasks"]}
        if (adapter.get("kind") != BANK_KIND or adapter.get("schema_version") != EVALUATION_SCHEMA
                or not source_matches(adapter["source"], source) or set(self.records) != set(task_keys)
                or adapter.get("single_complete_rank16") is not True):
            raise Pi05EvaluationError("video Writer runtime bank changed")
        self.adapter, self.policy = adapter, policy
        self.conditions = {row["condition_id"]: row for row in adapter["conditions"]}
        self.lora = load_pi05_lora_contract(Path(adapter["lora_contract"]["path"]))
        inject_task_lora(policy, self.lora)
        for parameter in task_lora_state_dict(policy).values():
            parameter.requires_grad_(False)
        policy.eval()
        self.batched = BatchedLoRAInference(policy, self.lora)
        self.identity = identity_lora_state(self.lora)
        self.readout_intervention = readout_intervention
        self.masked_manifest = None
        if readout_intervention is not None:
            from ember.pi05_eval.readout_state import SCHEMA

            path = Path(readout_intervention["derived_manifest"]["path"])
            if not path.is_file() or path.stat().st_size != int(readout_intervention["derived_manifest"]["bytes"]):
                raise Pi05EvaluationError("readout derived manifest changed")
            self.masked_manifest = read_json(path)
            if self.masked_manifest.get("schema_version") != SCHEMA:
                raise Pi05EvaluationError("readout derived manifest schema changed")
        self._states: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._installed: str | None = None

    def _state(self, key: str) -> dict[str, torch.Tensor]:
        group = (self.readout_intervention or {}).get("group", "11")
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key]
        condition = self.conditions[key]
        _inspect_adapter_file(condition, self.adapter["writer_checkpoint"], self.lora)
        if group in ("01", "10"):
            from ember.pi05_eval.readout_state import load_masked_state

            if self.masked_manifest["conditions"][key]["original"] != condition["adapter"]:
                raise Pi05EvaluationError("readout derived state original provenance changed")
            state = load_masked_state(self.masked_manifest, condition_id=key,
                                      group=group, lora=self.lora)
        else:
            state = load_file(condition["adapter"]["path"], device="cpu")
            if group == "00":
                from ember.pi05_eval.readout_state import mask_state

                state = mask_state(state, self.lora, "00")
        validate_lora_state(state, self.lora)
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
            raise Pi05EvaluationError("runtime adapter has nonfinite or non-FP32 values")
        self._states[key] = state
        if len(self._states) > 16:
            self._states.popitem(last=False)
        return state

    def prepare_episode(self, *, suite: str, task_id: int, init_state_id: int) -> PreparedHorizonLoRA:
        task = self.records.get((str(suite), int(task_id)))
        if task is not None:
            for episode in task["episodes"]:
                if episode["init_state_id"] == init_state_id:
                    return PreparedHorizonLoRA(episode["condition_id"], episode_evidence(self.adapter, task, episode))
        raise Pi05EvaluationError("rollout task/init state is absent from the paired Writer bank")

    @torch.no_grad()
    def install(self, prepared: PreparedHorizonLoRA) -> None:
        if prepared.key != self._installed:
            copy_task_lora_state_(self.policy, self._state(prepared.key), self.lora)
            self._installed = prepared.key

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if not prepared or len(prepared) != noise.shape[0] or any(item.key not in self.conditions for item in prepared):
            raise Pi05EvaluationError("policy batch and paired Writer conditions differ")
        if self._installed is not None:
            copy_task_lora_state_(self.policy, self.identity, self.lora)
            self._installed = None
        with self.batched.activate([self._state(item.key) for item in prepared]):
            return self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)

    def close(self) -> None:
        self.batched.close()
        self._states.clear()
