"""Bounded full-state continuation from a sealed, completed 1500-update run."""
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


def require_continuation_config(config):
    declaration = config.get("continuation")
    budget = config["data"].get("maximum_updates")
    if (type(budget) is not int or budget != (2100 if declaration else 1500)
            or (declaration is not None and declaration != CONTINUATION)):
        raise ValueError("canonical training budget or continuation scientific contract changed")


def require_continuation_start(args, config):
    parent = getattr(args, "extend_from", None)
    resume = getattr(args, "resume", None)
    if parent and (resume or config.get("continuation") != CONTINUATION):
        raise ValueError("continuation needs its registered config and cannot also exact-resume")
    if config.get("continuation") and not (parent or resume):
        raise ValueError("continued budget cannot start fresh; restore the complete parent checkpoint")


def require_extended_prefix(previous, current):
    if previous.get("maximum_updates") != 1500 or current.get("maximum_updates") != 2100:
        raise ValueError("only the registered 1500-to-2100 event continuation is allowed")
    prefix = deepcopy(current)
    prefix["maximum_updates"] = 1500
    prefix["groups"] = prefix["groups"][:1500]
    if "events" in prefix:
        prefix["events"] = prefix["events"][:6000]
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


def inherit_history(checkpoint, output):
    for name in ("metrics.jsonl", "exposures.jsonl", "diagnostics.jsonl"):
        with (checkpoint.parent.parent / name).open("rb") as source:
            with (output / name).open("xb") as target:
                shutil.copyfileobj(source, target)
