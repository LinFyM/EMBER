import copy
import json
from pathlib import Path

import pytest

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_task_expert_config,
)
from ember.expert_manifold.inference import inspect_v6_prior_writer_asset
from ember.expert_manifold.v6_prior import (
    V6_PRIOR_FROZEN_PARAMETER_COUNT,
    V6_PRIOR_TRAINABLE_PARAMETER_COUNT,
)
from ember.expert_manifold.v6_prior_checkpoint import V6_PRIOR_CHECKPOINT_SCHEMA
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
    load_v6_prior_config,
)
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = V6_PRIOR_CANONICAL_CONFIG


def test_tangent_tube_inherits_only_the_unchanged_v6_deployment_smoke() -> None:
    evaluation = load_v6_prior_config(CONFIG)["evaluation"]
    assert evaluation["throughput_policy"] == (
        "highest_measured_throughput_with_device_memory_headroom"
    )
    assert evaluation["minimum_smoke_writer_model_batch_size"] == 8
    assert evaluation["formal_status"] == "sealed_from_unchanged_v6_deployment_graph"
    evidence = evaluation["inherited_online_smoke_evidence"]
    assert evidence["writer_model_batch_size"] == 8
    assert evidence["source_family"] == "legacy_v6_prior_v1"
    assert evidence["scientific_rows"] == 8
    assert evidence["failure_count"] == 0
    assert evidence["dynamic_anchor_deployment_owned"] is False
    assert evidence["deployment_graph_change"] == (
        "family_identity_and_trainable_state_restore_scope_only"
    )


def test_inherited_deployment_seal_rejects_dynamic_anchor_ownership(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["evaluation"]["inherited_online_smoke_evidence"][
        "dynamic_anchor_deployment_owned"
    ] = True
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_inherited_deployment_seal_rejects_changed_runtime_graph(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["evaluation"]["inherited_online_smoke_evidence"][
        "deployment_graph_change"
    ] = "training_dynamic_anchor_added_to_deployment"
    path = tmp_path / "stable-selection.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_tangent_tube_config_rejects_broad_resume_writer_scope(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["initialization"]["resume_writer_load_scope"] = "all_writer_tensors"
    path = tmp_path / "stable-tie.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_old_expert_asset_config_cannot_enter_canonical_runtime() -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(old)


def test_v6_task_expert_authority_ignores_retired_writer_seals(
    tmp_path: Path,
) -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    config = json.loads(old.read_text(encoding="utf-8"))
    config["topological_writer"] = {"retired": True}
    config["meta_training"] = {"retired": True}
    path = tmp_path / "task_experts.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_task_expert_config(path)["task_experts"]["task_count"] == 24

    config["task_experts"]["formal_run"]["selected_stop_step"] = 1000
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="task-expert scientific boundary"):
        load_task_expert_config(path)


def test_historical_v6_warm_start_is_a_real_load_only_evaluation_asset() -> None:
    config = load_v6_prior_config(CONFIG)
    checkpoint = (REPO_ROOT / config["initialization"]["checkpoint"]).resolve()
    historical_source = read_json(checkpoint.parent.parent / "run_contract.json")[
        "source"
    ]

    asset = inspect_v6_prior_writer_asset(
        config,
        checkpoint,
        historical_source,
        require_formal=False,
    )

    assert asset["kind"] == "historical_v6_macro400_load_only"
    assert asset["source_macro"] == 400
    assert asset["method_macro"] == 0
    assert asset["writer_state"]["state_tensor_count"] == 600
    assert asset["writer_state"]["state_value_count"] == 12_064_064
    storage = asset["writer_state"]["template_lora_storage"]
    assert storage["tensor_count"] == 76
    assert storage["parameter_count"] == 1_287_168
    assert storage["tensor_bytes"] == 2_641_920
    assert storage["dtype_tensor_counts"] == {"BF16": 72, "F32": 4}
    assert storage["dtype_parameter_counts"] == {
        "BF16": 1_253_376,
        "F32": 33_792,
    }
    assert len(storage["dtype_by_name"]) == 76
    assert (
        storage["dtype_by_name"]["model.action_in_proj.lora_A.default.weight"] == "F32"
    )


def test_trained_writer_asset_fails_closed_on_dynamic_anchor_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_v6_prior_config(CONFIG)
    checkpoint = tmp_path / "checkpoints" / "macro_00000001"
    checkpoint.mkdir(parents=True)
    writer_path = checkpoint / "writer.safetensors"
    writer_path.write_bytes(b"synthetic-writer")
    source = {"checkpoint": "/synthetic/source"}
    manifest = {
        "schema_version": V6_PRIOR_CHECKPOINT_SCHEMA,
        "next_macro": 1,
        "metrics_rows": 1,
        "world_size": 6,
        "content_hash_policy": "disabled_by_owner",
        "files": {"writer.safetensors": writer_path.stat().st_size},
        "checkpoint_contract": {
            "run_schema": V6_PRIOR_RUN_SCHEMA,
            "mode": "formal",
            "source": source,
            "objective": config["objective"],
            "config": {"schema": V6_PRIOR_CONFIG_SCHEMA},
            "initialization": {
                "checkpoint": str(
                    (
                        REPO_ROOT
                        / str(config["initialization"]["checkpoint"])
                        / "writer.safetensors"
                    ).resolve()
                ),
                "dynamic_anchor": (
                    "training_only_frozen_macro0_compiler_and_factor_heads"
                ),
                "resume_writer_load_scope": (
                    "trainable_compiler_and_factor_heads_only"
                ),
            },
            "ownership": {
                "frozen_parameter_count": V6_PRIOR_FROZEN_PARAMETER_COUNT,
                "trainable_parameter_count": V6_PRIOR_TRAINABLE_PARAMETER_COUNT,
                "dynamic_anchor": {
                    "parameter_count": V6_PRIOR_TRAINABLE_PARAMETER_COUNT,
                    "tensor_count": 41,
                    "optimizer_owned": False,
                    "checkpoint_owned": False,
                    "deployment_owned": False,
                },
            },
        },
    }
    monkeypatch.setattr(
        "ember.expert_manifold.inference._writer_state_record",
        lambda *_args, **_kwargs: {"state_tensor_count": 600},
    )

    def publish(value: dict) -> None:
        (checkpoint / "manifest.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    publish(manifest)
    asset = inspect_v6_prior_writer_asset(
        config,
        checkpoint,
        source,
        require_formal=True,
    )
    assert asset["kind"] == "v6_tangent_tube_trained_checkpoint"

    for path, changed in (
        (("initialization", "dynamic_anchor"), "deployment_anchor"),
        (("initialization", "resume_writer_load_scope"), "all_writer_tensors"),
        (("ownership", "dynamic_anchor", "optimizer_owned"), True),
        (("ownership", "dynamic_anchor", "checkpoint_owned"), True),
        (("ownership", "dynamic_anchor", "deployment_owned"), True),
    ):
        invalid = copy.deepcopy(manifest)
        cursor = invalid["checkpoint_contract"]
        for name in path[:-1]:
            cursor = cursor[name]
        cursor[path[-1]] = changed
        publish(invalid)
        with pytest.raises(
            ExpertManifoldError,
            match="trained Writer checkpoint changed",
        ):
            inspect_v6_prior_writer_asset(
                config,
                checkpoint,
                source,
                require_formal=True,
            )
