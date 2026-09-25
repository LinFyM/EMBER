"""Registered short fork of one relational Writer training task slot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ember.ecp.checkpoint import load_ecp_checkpoint
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_contract import reconcile_metrics
from ember.writer.learning_data import WriterTrainingData


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = Path("configs/support_slot_credit_causality_v1/experiment_spec.json")
SAMPLER_SCHEMA = "ember_support_slot_branch_sampler_v1"
ARMS = ("KEEP77", "SWAP76", "DROP77")
PARENT_MACRO = 1155
FINAL_MACRO = 1183


def authority() -> dict[str, Any]:
    spec = read_json(REPO_ROOT / SPEC_PATH)
    training = spec.get("training", {})
    if (spec.get("schema_version") != "ember_support_slot_credit_causality_v1"
            or spec.get("study_id") != "support_slot_credit_causality_20260925"
            or spec.get("status") != "registered_for_implementation_and_bounded_execution"
            or spec.get("parent", {}).get("macro") != PARENT_MACRO
            or training.get("first_macro") != PARENT_MACRO + 1
            or training.get("last_macro") != FINAL_MACRO
            or training.get("full_checkpoint_macros") != [1160, FINAL_MACRO]
            or training.get("world_size") != 2
            or [arm.get("id") for arm in training.get("arms", [])] != list(ARMS)
            or len(training.get("event_mapping", [])) != 28
            or spec["evaluation"]["model_order"] != ["P1155", *ARMS]
            or spec["evaluation"]["new_rollouts"] != 400):
        raise ValueError("support-slot study registration changed")
    return spec


def root(spec: Mapping[str, Any] | None = None) -> Path:
    return Path((spec or authority())["resources"]["study_root"]).resolve()


def registration_for(arm: str, spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = spec or authority()
    if arm not in ARMS:
        raise ValueError("unregistered support-slot branch")
    return {"schema_version": SAMPLER_SCHEMA, "study_spec": str(REPO_ROOT / SPEC_PATH),
            "study_spec_bytes": (REPO_ROOT / SPEC_PATH).stat().st_size,
            "arm": arm, "parent_checkpoint": str(Path(spec["parent"]["checkpoint"]).resolve()),
            "parent_training_commit": spec["parent"]["training_commit"],
            "parent_macro": PARENT_MACRO, "first_macro": PARENT_MACRO + 1,
            "last_macro": FINAL_MACRO, "checkpoint_macros": [1160, FINAL_MACRO],
            "event_sources": {"S00": spec["data"]["S00_event_plan"],
                              "S10": spec["data"]["S10_event_plan"]},
            "gradient_gate": spec["training"]["gradient_gate"],
            "not_original_exact_resume": True}


@dataclass(frozen=True)
class ForkScope:
    spec: Mapping[str, Any]
    arm: str
    parent_checkpoint: Path
    s00_plan: Mapping[str, Any]
    s10_plan: Mapping[str, Any]

    @property
    def definition(self) -> Mapping[str, Any]:
        return next(item for item in self.spec["training"]["arms"] if item["id"] == self.arm)

    @property
    def mapping(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.spec["training"]["event_mapping"])

    def training_state(self, config: Mapping[str, Any], updates: int) -> dict[str, Any]:
        if updates not in (1160, FINAL_MACRO):
            raise ValueError("branch checkpoint is outside its two registered nodes")
        return {"schema_version": "ember_video_teaching_training_state_v1", "updates": updates,
                "update_version": config["update_version"], "data_version": config["data"]["version"],
                "support_slot_credit": {"study_id": self.spec["study_id"], "arm": self.arm,
                                        "parent_checkpoint": str(self.parent_checkpoint),
                                        "parent_macro": PARENT_MACRO,
                                        "branch_cursor": updates - PARENT_MACRO}}

    def registration(self) -> dict[str, Any]:
        return registration_for(self.arm, self.spec)


def inspect_fork(arm: str, config: Mapping[str, Any], output: Path, *, mode: str) -> ForkScope:
    """Validate both original event streams before starting distributed/GPU work."""
    from ember.writer.materialization import inspect_writer_checkpoint

    spec = authority()
    if arm not in ARMS or mode not in ("smoke", "formal"):
        raise ValueError("unregistered support-slot arm or execution mode")
    study = root(spec)
    output = output.resolve()
    if ((mode == "formal" and output != study / "training" / arm)
            or (mode == "smoke" and not output.is_relative_to(study / "smoke"))):
        raise ValueError("support-slot output is outside its registered phase")
    parent = Path(spec["parent"]["checkpoint"]).resolve()
    run, checkpoint = inspect_writer_checkpoint(parent)
    if (config != read_json(REPO_ROOT / spec["parent"]["base_config"])
            or run["config"] != config or run["git"]["commit"] != spec["parent"]["training_commit"]
            or checkpoint["macro"] != PARENT_MACRO or parent != Path(checkpoint["path"])
            or run["topology"]["world_size"] != 2):
        raise ValueError("support-slot fork requires the complete registered S00@1155 parent")
    donor_run = read_json(Path(spec["data"]["S10_event_plan"]).parent / "run_contract.json")
    if (donor_run["git"]["commit"] != spec["parent"]["training_commit"]
            or donor_run["config"]["experiment"]["arm_id"] != "C_S10"
            or donor_run["source"] != run["source"]
            or donor_run["model_config"] != run["model_config"]
            or donor_run["config"]["optimization"] != config["optimization"]):
        raise ValueError("support-slot donor run differs from the registered parent recipe")
    s00, s10 = (read_json(Path(spec["data"][key])) for key in ("S00_event_plan", "S10_event_plan"))
    mapping = spec["training"]["event_mapping"]
    slot_macros = []
    common = 0
    for offset, row in enumerate(mapping):
        macro = PARENT_MACRO + offset + 1
        if (row["macro"] != macro or row["S00_event_indices"] != s00["groups"][macro - 1]
                or row["S10_event_indices"] != s10["groups"][macro - 1]):
            raise ValueError("registered fork event indices differ from the two sealed streams")
        slot = row["slot_position"]
        if slot is not None:
            slot_macros.append(macro)
        for position, (left, right) in enumerate(zip(row["S00_event_indices"], row["S10_event_indices"], strict=True)):
            first, second = s00["events"][left], s10["events"][right]
            if position == slot:
                if (first["task"], second["task"]) != (77, 76):
                    raise ValueError("registered 77/76 donor slot changed")
            elif first != second:
                raise ValueError("common task event differs across sealed parent streams")
            else:
                common += 1
    if (common != 108 or slot_macros != spec["training"]["donor_macros"]
            or spec["training"]["donor_occurrences"] != [
                s00["events"][row["S00_event_indices"][row["slot_position"]]]["occurrence"]
                for row in mapping if row["slot_position"] is not None]):
        raise ValueError("registered fork common events or donor visits changed")
    return ForkScope(spec, arm, parent, s00, s10)


class _VideoSources:
    def __init__(self, base: WriterTrainingData, donor: WriterTrainingData | None) -> None:
        self.base, self.donor = base, donor
        self.camera_view = base.videos.camera_view
        if donor is not None and donor.videos.camera_view != self.camera_view:
            raise ValueError("donor and parent video camera modes differ")

    def frame_counts(self, task: int, demo: int):
        source = self.donor if task == 76 and self.donor is not None else self.base
        return source.videos.frame_counts(task, demo)


class ForkTrainingData:
    """Use S00 scheduling and exact S10 slot draws without claiming prior 76 exposure."""

    def __init__(self, asset_root: Path, config: Mapping[str, Any], scope: ForkScope) -> None:
        self.scope = scope
        self.base = WriterTrainingData(asset_root, config["data"],
                                       camera_view=config["observer"]["camera_view"])
        if self.base.event_plan() != scope.s00_plan:
            raise ValueError("regenerated S00 training events differ from the sealed parent plan")
        self.donor = None
        if scope.arm == "SWAP76":
            donor_config = read_json(REPO_ROOT / "configs/relational_support_causality_v1/train_C_S10.json")
            self.donor = WriterTrainingData(asset_root, donor_config["data"],
                                            camera_view=donor_config["observer"]["camera_view"])
            if self.donor.event_plan() != scope.s10_plan:
                raise ValueError("regenerated S10 donor events differ from their sealed plan")
        self.tasks = {**self.base.tasks, **(self.donor.tasks if self.donor else {})}
        self.videos = _VideoSources(self.base, self.donor)
        self.branch_cursor = 0
        self.parent_sampler: dict[str, Any] | None = None

    def event_plan(self) -> dict[str, Any]:
        return {**self.scope.registration(), "mapping": list(self.scope.mapping),
                "credit_gate_by_arm": {"KEEP77": 1, "SWAP76": 1, "DROP77": 0},
                "branch_cursor_range": [0, 28], "source_occurrence_is_donor_index_not_exposure": True}

    def restore_parent(self, sampler: Mapping[str, Any]) -> None:
        if sampler.get("next_step") != PARENT_MACRO:
            raise ValueError("fork parent sampler is not at macro1155")
        self.base.restore_sampler(sampler)
        self.parent_sampler = dict(sampler)
        self.branch_cursor = 0

    def restore_branch(self, sampler: Mapping[str, Any]) -> None:
        if (sampler.get("schema_version") != SAMPLER_SCHEMA
                or sampler.get("registration") != self.scope.registration()
                or type(sampler.get("branch_cursor")) is not int
                or not 0 < sampler["branch_cursor"] <= 28
                or sampler.get("next_global_macro") != PARENT_MACRO + sampler["branch_cursor"]):
            raise ValueError("branch sampler identity or local cursor changed")
        self.restore_parent(sampler["frozen_parent_sampler"])
        for _ in range(sampler["branch_cursor"]):
            self.next_iteration()
        if self.sampler_state() != dict(sampler):
            raise ValueError("branch sampler event references or effective exposures changed")

    def sampler_state(self) -> dict[str, Any]:
        if self.parent_sampler is None:
            raise ValueError("branch sampler needs its real parent state")
        return {"schema_version": SAMPLER_SCHEMA, "registration": self.scope.registration(),
                "frozen_parent_sampler": self.parent_sampler, "branch_cursor": self.branch_cursor,
                "next_global_macro": PARENT_MACRO + self.branch_cursor,
                "effective_event_counts": self._effective_counts()}

    def _effective_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.scope.mapping[:self.branch_cursor]:
            for position, event_index in enumerate(row["S00_event_indices"]):
                slot = row["slot_position"]
                if position == slot and self.scope.arm == "DROP77":
                    continue
                event = (self.scope.s10_plan["events"][row["S10_event_indices"][position]]
                         if position == slot and self.scope.arm == "SWAP76" else
                         self.scope.s00_plan["events"][event_index])
                key = str(event["task"])
                counts[key] = counts.get(key, 0) + 1
        return counts

    def next_iteration(self) -> tuple[dict[str, Any], ...]:
        if self.parent_sampler is None or self.branch_cursor >= 28:
            raise StopIteration("registered support-slot branch is exhausted")
        row = self.scope.mapping[self.branch_cursor]
        base = [dict(draw) for draw in self.base.next_iteration()]
        if self.base.next_step != row["macro"]:
            raise ValueError("S00 event cursor left the registered fork window")
        slot = row["slot_position"]
        if slot is not None and self.scope.arm == "SWAP76":
            event = self.scope.s10_plan["events"][row["S10_event_indices"][slot]]
            base[slot].update(task=76, occurrence=event["occurrence"],
                              video_demos=(event["teacher_demo"],), query_seed=event["query_seed"],
                              frames=event["frames"])
        for position, draw in enumerate(base):
            draw["source_event_plan"] = "S10" if position == slot and self.scope.arm == "SWAP76" else "S00"
            draw["source_event_index"] = row[f'{draw["source_event_plan"]}_event_indices'][position]
            draw["credit_gate"] = 0 if position == slot and self.scope.arm == "DROP77" else 1
            draw["branch_cursor"] = self.branch_cursor + 1
        self.branch_cursor += 1
        return tuple(base)

    def action_batch(self, task: int, occurrence: int, demos, **kwargs):
        source = self.donor if task == 76 and self.donor is not None else self.base
        return source.action_batch(task, occurrence, demos, **kwargs)

    def load_videos(self, task: int, demos):
        source = self.donor if task == 76 and self.donor is not None else self.base
        return source.load_videos(task, demos)

    def close(self) -> None:
        self.base.close()
        if self.donor is not None:
            self.donor.close()


def restore_event_gradients(parameters, execute) -> Any:
    """Execute the whole slot for RNG/provenance, then remove only its current gradient."""
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    before = tuple(None if parameter.grad is None else parameter.grad.detach().clone()
                   for parameter in parameters)
    result = execute()
    for parameter, previous in zip(parameters, before, strict=True):
        if previous is None:
            parameter.grad = None
        elif parameter.grad is None:
            parameter.grad = previous
        else:
            parameter.grad.copy_(previous)
    return result


def execute_registered_event(engine, data, draw: Mapping[str, Any], step: int) -> Any:
    gate = draw.get("credit_gate", 1)
    if gate == 1:
        return engine.backward(draw)
    if (gate != 0 or getattr(data, "scope", None) is None or data.scope.arm != "DROP77"
            or draw["task"] != 77 or step not in data.scope.spec["training"]["donor_macros"]):
        raise ValueError("zero task credit is outside registered DROP77 slots")
    return restore_event_gradients(engine.runtime.state.parameters(), lambda: engine.backward(draw))


def attach_branch_contract(contract: dict[str, Any], scope: ForkScope) -> None:
    training = contract["training"]
    if (training["source_trainable_parameters"] != 0
            or any(training[key] <= 0 for key in (
                "writer_parameters", "meta_parameters", "vl_meta_parameters", "text_meta_parameters"))):
        raise ValueError("support-slot fork must train the complete Writer and native Meta on frozen Source")
    contract["support_slot_credit"] = scope.registration()
    training.update(
        optimizer="parent1155 complete AdamW m/v, scheduler and rank RNG; one unchanged grouped update per macro",
        branch_kind="controlled_full_state_fork_not_original_exact_resume",
        branch_window=[1156, 1183], branch_checkpoint_updates=[1160, 1183],
    )


def attach_branch_completion(completed: dict[str, Any], data: ForkTrainingData, scope: ForkScope) -> None:
    completed.update(support_slot_credit=scope.registration(), branch_cursor=data.branch_cursor,
                     branch_active_queries=sum(data.sampler_state()["effective_event_counts"].values()) * 28,
                     branch_executed_query_slots=data.branch_cursor * 4 * 28,
                     next="registered frozen first-slot and terminal diagnostics")


def restore_fork(args, context, runtime, data: ForkTrainingData, optimizer, scheduler,
                 config: Mapping[str, Any], scope: ForkScope) -> tuple[int, int]:
    """Restore exact full state; start a new branch ledger at the shared parent."""
    from ember.writer.training import RUN_SCHEMA, STAGE, _training_state

    checkpoint = args.resume.resolve() if args.resume else scope.parent_checkpoint
    if args.resume and checkpoint.parent.parent != args.output.resolve():
        raise ValueError("branch resume checkpoint must belong to this fork run")
    restored: dict[str, Any] = {}
    updates, saved_rows = load_ecp_checkpoint(
        checkpoint=checkpoint, stage=STAGE, context=context, model=runtime.state,
        optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA,
        restored_state=restored,
    )
    if args.resume:
        if restored["training_state"] != scope.training_state(config, updates):
            raise ValueError("branch model/optimizer checkpoint has a different fork identity")
        data.restore_branch(restored["sampler_state"])
        branch_rows = saved_rows
        if context.is_main:
            reconcile_metrics(args.output / "exposures.jsonl", updates, branch_rows,
                              cursor_key="step", packet_label="exposures")
            reconcile_metrics(args.output / "metrics.jsonl", updates, data.branch_cursor,
                              cursor_key="step", packet_label="metrics")
    else:
        if updates != PARENT_MACRO or restored["training_state"] != _training_state(config, updates):
            raise ValueError("full parent1155 state or original training recipe changed")
        data.restore_parent(restored["sampler_state"])
        branch_rows = 0
    if (updates != PARENT_MACRO + data.branch_cursor or scheduler.last_epoch != updates
            or any(int(state["step"]) != updates for state in optimizer.state.values() if "step" in state)):
        raise ValueError("branch optimizer, scheduler or sampler cursor changed")
    return updates, branch_rows


def validate_branch_checkpoint(run: Mapping[str, Any], trainer: Mapping[str, Any],
                               checkpoint: Path, macro: int, world_size: int) -> None:
    """Distinguish a full controlled fork from the original C_S00 training run."""
    spec = authority()
    registration = run.get("support_slot_credit") or {}
    arm = registration.get("arm")
    if arm not in ARMS:
        raise ValueError("unregistered branch checkpoint identity")
    expected = {"schema_version": "ember_video_teaching_training_state_v1", "updates": macro,
                "update_version": run["config"]["update_version"],
                "data_version": run["config"]["data"]["version"],
                "support_slot_credit": {"study_id": spec["study_id"], "arm": arm,
                    "parent_checkpoint": str(Path(spec["parent"]["checkpoint"]).resolve()),
                    "parent_macro": PARENT_MACRO, "branch_cursor": macro - PARENT_MACRO}}
    sampler = trainer.get("sampler_state") or {}
    if (registration != registration_for(arm, spec)
            or checkpoint.resolve() != root(spec) / "training" / arm / "checkpoints" / f"macro_{macro:08d}"
            or macro not in (1160, FINAL_MACRO) or world_size != 2
            or run["config"] != read_json(REPO_ROOT / spec["parent"]["base_config"])
            or trainer.get("training_state") != expected
            or sampler.get("schema_version") != SAMPLER_SCHEMA
            or sampler.get("registration") != registration
            or sampler.get("branch_cursor") != macro - PARENT_MACRO
            or sampler.get("next_global_macro") != macro
            or sampler.get("frozen_parent_sampler", {}).get("next_step") != PARENT_MACRO
            or trainer.get("scheduler", {}).get("last_epoch") != macro):
        raise ValueError("branch checkpoint model, optimizer or sampler provenance changed")


def registered_bank_request(request: Mapping[str, Any], run: Mapping[str, Any],
                            checkpoint: Mapping[str, Any], repository: Mapping[str, Any]) -> dict[str, Any]:
    """Permit only the three registered bank phases before compiler workers start."""
    from ember.writer.materialization import selection_contract

    spec = authority()
    model, phase = request.get("support_slot_model"), request.get("support_slot_phase")
    if model not in ("P1155", *ARMS) or phase not in ("final", "first_slot", "donor_fm"):
        raise ValueError("unregistered support-slot bank model or phase")
    if phase == "first_slot" and model == "P1155":
        raise ValueError("first-slot prediction only uses the three updated branches")
    if phase == "final":
        expected = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
            arm="correct", mode="per_init_ordinal", seed=20260911,
            init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
        macro = 1155 if model == "P1155" else FINAL_MACRO
    elif phase == "first_slot":
        expected = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
            arm="correct", mode="per_init_ordinal", seed=20260911,
            init_state_ids=(0, 10, 20, 30, 40), video_pool=tuple(range(50)))
        macro = 1160
    else:
        expected = selection_contract(role="nonheld_meta", task_ids=[76, 77], cardinality=1,
            arm="correct", mode="fixed_per_task", seed=20260925,
            init_state_ids=(0,), video_pool=(46,), fixed_videos={"76": [46], "77": [46]})
        macro = 1155 if model == "P1155" else FINAL_MACRO
    path = (Path(spec["parent"]["checkpoint"]).resolve() if model == "P1155" else
            root(spec) / "training" / model / "checkpoints" / f"macro_{macro:08d}")
    if (request["selection"] != expected or Path(request["checkpoint"]).resolve() != path
            or Path(request["output"]).resolve() != root(spec) / "materialization" / phase / model
            or checkpoint["macro"] != macro or checkpoint["path"] != str(path)
            or run["config"] != read_json(REPO_ROOT / spec["parent"]["base_config"])
            or request.get("native_transfer_cell") is not None
            or request.get("reuse_manifest") is not None or request.get("diagnostic_contract") is not None
            or (model != "P1155" and repository["commit"] != run["git"]["commit"])):
        raise ValueError("support-slot bank checkpoint, phase or video selection changed")
    if model == "P1155":
        if run.get("support_slot_credit") is not None or run["git"]["commit"] != spec["parent"]["training_commit"]:
            raise ValueError("parent bank must use the original complete C_S00@1155 checkpoint")
    elif run.get("support_slot_credit") != registration_for(model, spec):
        raise ValueError("branch bank differs from its own trained fork")
    return {"schema_version": "ember_support_slot_bank_v1", "study_spec": str(REPO_ROOT / SPEC_PATH),
            "study_spec_bytes": (REPO_ROOT / SPEC_PATH).stat().st_size,
            "model": model, "phase": phase, "checkpoint": dict(checkpoint),
            "parent_training_commit": spec["parent"]["training_commit"],
            "bank_generation_commit": repository["commit"]}
