"""Registered full-state Writer continuations from sealed formal parents."""
from copy import deepcopy
import shutil

from ember.ecp.checkpoint import checkpoint_macro
from ember.pi05_source_checkpoint import read_json


CONTINUATION = {
    "parent_updates": 1500, "maximum_updates": 2100,
    "learning_rate": "continue_existing_tail_floor",
    "optimizer_scheduler_rng": "preserve_complete_state",
    "checkpoint_selection": False,
}

LOW_LR_REPAIR = {
    "kind": "coverage_writer_low_lr_repair_v1",
    "parent_updates": 1800,
    "first_global_update": 1801,
    "fixed_lr": 2.959936e-5,
    "parent_applied_lr": 1.627965011517147e-4,
    "optimizer_scheduler_sampler_rng": "preserve_complete_parent_state",
    "checkpoint_interval": 100,
    "validation_interval": 100,
    "selection_reference_successes": 117,
}


def require_continuation_config(config):
    declaration = config.get("continuation")
    phase = config.get("phase_continuation")
    budget = config["data"].get("maximum_updates")
    control = config.get("training_control")
    if phase is not None:
        if (phase != LOW_LR_REPAIR or declaration is not None or budget is not None
                or control != {"kind": "validation_early_stopping", "checkpoint_interval": 100,
                               "validation_interval": 100}):
            raise ValueError("low-LR repair requires its registered phase continuation contract")
        return
    if control is not None:
        if (declaration is not None or budget is not None
                or control != {"kind": "validation_early_stopping", "checkpoint_interval": 100,
                               "validation_interval": 200}):
            raise ValueError("dynamic training requires the registered validation control and no fixed budget")
        return
    if (type(budget) is not int or budget != (2100 if declaration else 1500)
            or (declaration is not None and declaration != CONTINUATION)):
        raise ValueError("canonical training budget or continuation scientific contract changed")


def require_continuation_start(args, config):
    parent = getattr(args, "extend_from", None)
    phase_parent = getattr(args, "phase_from", None)
    resume = getattr(args, "resume", None)
    phase = config.get("phase_continuation")
    if phase_parent and (parent or resume or phase != LOW_LR_REPAIR):
        raise ValueError("low-LR phase needs its registered config and cannot also exact-resume")
    if phase and not (phase_parent or resume):
        raise ValueError("low-LR phase cannot start fresh; restore the complete N1800 checkpoint")
    if phase_parent is None and phase is None and getattr(args, "phase_from", None):
        raise ValueError("phase parent requires the registered low-LR repair config")
    if phase and parent:
        raise ValueError("low-LR phase cannot use the historical bounded continuation interface")
    if parent and (resume or config.get("continuation") != CONTINUATION):
        raise ValueError("continuation needs its registered config and cannot also exact-resume")
    if config.get("continuation") and not (parent or resume):
        raise ValueError("continued budget cannot start fresh; restore the complete parent checkpoint")


def require_extended_prefix(previous, current, *, parent_updates=1500, child_updates=2100):
    if previous.get("maximum_updates") is None or current.get("maximum_updates") is None:
        if previous != current:
            raise ValueError("dynamic continuation must preserve its complete event algorithm contract")
        return
    if previous.get("maximum_updates") != parent_updates or current.get("maximum_updates") != child_updates:
        raise ValueError("bounded continuation event limits differ from the registered phase")
    prefix = deepcopy(current)
    prefix["maximum_updates"] = parent_updates
    prefix["groups"] = prefix["groups"][:parent_updates]
    if "events" in prefix:
        prefix["events"] = prefix["events"][:parent_updates * 4]
    if previous != prefix:
        raise ValueError("continued sampler must preserve every parent event, RNG and group")


def prepare_continuation(args, contract):
    checkpoint = args.extend_from.resolve()
    parent_root = checkpoint.parent.parent
    if checkpoint_macro(checkpoint) != 1500 or args.output.resolve() == parent_root:
        raise ValueError("continuation requires parent1500 and a distinct output root")
    parent = read_json(parent_root / "run_contract.json")
    before, after = deepcopy(parent["config"]), deepcopy(contract["config"])
    before.pop("evidence")
    after.pop("evidence")
    if after.pop("continuation", None) != CONTINUATION:
        raise ValueError("continuation declaration changed")
    after["data"]["maximum_updates"] = 1500
    if before != after:
        raise ValueError("continuation may change only the budget and evidence registration")
    for field in ("schema_version", "stage", "mode", "model_config", "topology", "source", "execution"):
        if parent[field] != contract[field]:
            raise ValueError(f"continuation parent contract differs: {field}")
    contract["continuation"] = {
        **CONTINUATION, "parent_checkpoint": str(checkpoint),
        "parent_run_contract": str(parent_root / "run_contract.json"),
        "parent_training_commit": parent["git"]["commit"],
        "history": "copy parent metrics/exposures/diagnostics; append only new updates",
    }


def prepare_phase_continuation(args, contract):
    """Validate that the new run changes only the registered schedule intervention."""
    checkpoint = args.phase_from.resolve()
    parent_root = checkpoint.parent.parent
    phase = contract["config"].get("phase_continuation")
    if checkpoint_macro(checkpoint) != LOW_LR_REPAIR["parent_updates"]:
        raise ValueError("low-LR repair requires the complete formal N1800 checkpoint")
    if args.output.resolve() == parent_root:
        raise ValueError("low-LR repair requires a distinct output root")
    if phase != LOW_LR_REPAIR:
        raise ValueError("low-LR repair declaration changed")
    parent = read_json(parent_root / "run_contract.json")
    before, after = deepcopy(parent["config"]), deepcopy(contract["config"])
    before.pop("evidence")
    after.pop("evidence")
    before.pop("design", None)
    after.pop("design", None)
    after.pop("phase_continuation", None)
    after["training_control"] = before["training_control"]
    if before != after:
        raise ValueError("low-LR repair may change only its phase, evidence, design and validation cadence")
    for field in ("schema_version", "stage", "mode", "model_config", "topology", "source", "execution"):
        if parent[field] != contract[field]:
            raise ValueError(f"low-LR repair parent contract differs: {field}")
    contract["phase_continuation"] = {
        **LOW_LR_REPAIR,
        "parent_checkpoint": str(checkpoint),
        "parent_run_contract": str(parent_root / "run_contract.json"),
        "parent_training_commit": parent["git"]["commit"],
        "schedule_semantics": "parent state restored first; fixed LR applies beginning at global update 1801",
        "history": "copy parent metrics/exposures/diagnostics; append global and phase cursors",
    }


def inherit_history(checkpoint, output):
    for name in ("metrics.jsonl", "exposures.jsonl", "diagnostics.jsonl"):
        with (checkpoint.parent.parent / name).open("rb") as source:
            with (output / name).open("xb") as target:
                shutil.copyfileobj(source, target)
