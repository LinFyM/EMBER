"""One bounded native-memory diagnostic contract, with existing asset owners."""

from __future__ import annotations

from pathlib import Path

from ember.operator_writer.data import TASKS
from ember.pi05_source_checkpoint import read_json


REPO = Path(__file__).resolve().parents[3]
STUDY_ID = TASK = "native_video_control_diagnostic_20261007"
SCHEMA = "ember_native_video_control_run_v1"
EVENT_SCHEMA = "ember_native_video_control_events_v1"
ROOT = Path("/data1/user/ymdai/ember_runs") / STUDY_ID
HELDOUTS = (12, 29, 32, 38)
FIT32 = tuple(task for task in TASKS if task not in HELDOUTS)
UPDATES = 128
CHECKPOINTS = (0, 64, 128)
SMOKE_CHECKPOINTS = (2, 4)
MT_WEIGHTS = Path("/data0/user/ymdai/ember_runs/coverage_retraining_20260920"
                  "/training/mtbc/checkpoints/step_00000300/lora.safetensors")
OPTIMIZATION = {
    "seed": 7, "name": "AdamW", "lr": 1e-4, "betas": [0.9, 0.95],
    "eps": 1e-8, "weight_decay": 1e-4, "grad_clip": 1.0,
    "parameter_dtype": "float32", "optimizer_dtype": "float32", "scheduler": None,
}


def specification() -> dict:
    """Reuse original paths; never copy datasets, models or their manifests."""
    source = read_json(REPO / "configs/operator_read_write_v1/learning_spec.json")["source"]
    source["lora_contract"] = "configs/pi05_lora_rank128_aligned.json"
    source["base_config"] = "configs/pi05_source_aligned.json"
    return {
        "schema_version": SCHEMA, "task": TASK, "study_id": STUDY_ID,
        "design": f"docs/designs/{STUDY_ID}.md", "run_root": str(ROOT),
        "source": source, "mt_weights": str(MT_WEIGHTS),
        "operator": {
            "rank": 128, "alpha": 128, "targets": 38, "module_seed": 7,
            "probe_seed": 1729, "probe_shape": [50, 32], "teaching_camera": "dual",
            "frame_stride": 5, "frame_chunk": 8, "source_frozen": True,
        },
        "model": {
            "hidden_width": 1024, "width": 256, "heads": 8, "encoder_layers": 2,
            "ffn_width": 1024, "dropout": 0.0, "activation": "gelu", "pre_norm": True,
            "reader_layers": 18, "rms_eps": 1e-6, "reader_bias": False,
            "reader_output_initialization": "zero", "positional_base": 10000,
            "time_position": "original_frame_index/5", "horizon_slots": 50,
        },
        "events": {
            "schema_version": EVENT_SCHEMA, "seed": 20261007, "task_ids": list(FIT32),
            "readback_task_ids": list(TASKS), "new_training_heldout_ids": list(HELDOUTS),
            "task_permutation_seed": [20261007, 0, "visit"],
            "teacher_permutation_seed": [20261007, 1, "task"],
            "query_seed": [20261007, 2, "task", "visit"],
            "teacher_demo_pool": list(range(50)), "visits_per_task": 16,
            "macros_per_visit": 8, "query_demo_pool": [0, 49],
            "queries_per_task": 28, "tasks_per_update": 4, "query_action_offset": 1,
            "logical_queries_per_macro": 112, "same_events_both_modes": True,
        },
        "optimization": {**OPTIMIZATION, "betas": list(OPTIMIZATION["betas"])},
        "execution": {
            "modes": ["M", "V"], "updates_per_mode": UPDATES,
            "queries_per_mode": 14336, "checkpoints": list(CHECKPOINTS),
            "only_scientific_endpoint": 128, "smoke_mode": "V",
            "smoke_checkpoints": list(SMOKE_CHECKPOINTS), "smoke_max_queries": 448,
            "profile_max_full_conditions": 3, "profile_max_queries": 84,
        },
        "evaluation": {
            "panel": "original_seen144", "task_ids": list(TASKS), "init_states": [32, 33, 34, 35],
            "mapping_seed": 20260928, "other_task_ids": list(HELDOUTS), "other_offset": 17,
            "new_episodes": 304, "full_rows": 76, "compact_rows": 228,
            "official_validation": False, "test": False,
        },
        "budget": {
            "estimated_wall_hours": [4, 7], "hard_wall_hours": 10,
            "gpu_hours_hard": 12, "run_root_peak_gib": 48,
            "project_gpu_cap": 8, "scarce_free_gpu_cap": 6, "single_node_gpu_cap": 6,
        },
    }
