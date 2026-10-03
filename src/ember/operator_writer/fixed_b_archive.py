"""Single registered A64 weights-only transfer readback; retire after its400 rows."""
from pathlib import Path

from ember.pi05_source_checkpoint import read_json
from ember.writer.materialization import file_record

TASK = "fixed_B_transfer_readback_20261003"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
CHECKPOINT = ROOT.parent / "joint_action_effect_credit_20261003/A/checkpoint64"
PARENT_ROOT = ROOT.parent / "conditional_read_write_continuation900_20261002/conditional_read_write"
PARENT_RUN = PARENT_ROOT / "train/attempts/continuation/run_contract.json"
TRAINING = CHECKPOINT.parent / "training_contract.json"
CAPTURE = Path(__file__).resolve().parents[3] / "configs/operator_read_write_v1/fixed_b_transfer_capture.json"
LEARNING_GIT = "b0df34ee1bc3fd09ff48bc603f08a272dd8ae359"
PARENT_GIT = "85919994aef11c17b49b7d0e70a2c110158bff61"


def registered(mode, checkpoint):
    return mode == "conditional_read_write" and checkpoint is not None and checkpoint.resolve() == CHECKPOINT


def weights_file(checkpoint):
    return checkpoint / ("writer.safetensors" if checkpoint.resolve() == CHECKPOINT else "ecp.safetensors")


def training_record(checkpoint):
    return TRAINING if checkpoint.resolve() == CHECKPOINT else checkpoint.parent.parent / "run_contract.json"


def source_record(asset_root):
    """Validate archival identity without loading optimizer or any action dataset."""
    from .bank import task_rows

    manifest = read_json(CHECKPOINT / "checkpoint_manifest.json")
    learning, parent = read_json(TRAINING), read_json(PARENT_RUN)
    expected_names = {f"conditional_targets.{i}.b_{part}.weight" for i in range(38)
                      for part in ("key", "delta", "context", "dynamic", "out")}
    if (manifest.get("complete") is not True or manifest.get("next_update") != 64
            or manifest["files"]["writer.safetensors"] != file_record(weights_file(CHECKPOINT))
            or learning.get("arm") != "A" or learning.get("updates") != 64
            or learning.get("query_offset") != 1 or learning.get("effect_weight") != 0
            or learning.get("source_trainable") != 0
            or set(learning["trainable_names"]) != expected_names
            or learning["git"]["commit"] != LEARNING_GIT
            or parent["git"]["commit"] != PARENT_GIT
            or learning["parent"] != file_record(CHECKPOINT.parent.parent.parent /
                "conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors")):
        raise ValueError("fixed A64 transfer archive or actual learning source changed")
    path = Path(parent["spec"])
    spec = read_json(path)
    tasks, conditions = task_rows(spec, asset_root)
    old = read_json(PARENT_ROOT / "banks/900/manifest.json")
    fields = ("condition_id", "global_task_id", "teacher_demo")
    if (old["tasks"] != tasks or len(conditions) != 400
            or [{key: row[key] for key in fields} for row in old["conditions"]] != conditions
            or parent["source"] != old["source"]):
        raise ValueError("fixed transfer changed parent400 task/video or source geometry")
    correction = {
        "archive": file_record(weights_file(CHECKPOINT)), "learning_contract": file_record(TRAINING),
        "parent_training_contract": file_record(PARENT_RUN), "parent900_training_and_correct400_git": PARENT_GIT,
        "parent450_training_git": "a0e0248d96568e42b24a3d1c4102e2ca6e35a40c",
        "Original32_bank_reading_git": "923ff89b60f4f8a269e5352d8d82189d72a9cd0a",
        "A64_learning_git": LEARNING_GIT, "A64_original_environment_git": "a270fb01",
        "sealed_parent_shortname_is_450": True, "sealed_originals_unchanged": True,
        "training_updates": 0, "optimizer_loaded": False, "held_actions_read": False,
        "fixed_parent400": str(PARENT_ROOT / "evaluation/900/correct400/results.json")}
    training = {"git": {"commit": LEARNING_GIT}, "source": parent["source"], "lora": parent["lora"],
                "fixed_B_transfer_readback": correction}
    return spec, training, path
