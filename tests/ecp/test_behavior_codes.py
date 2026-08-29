import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from ember.ecp.behavior.codes import (
    BEHAVIOR_CODE_SCHEMA,
    fixed_internal_behavior_fold,
    load_behavior_code_authority,
    load_program_model_initialization,
)
from ember.ecp.behavior.kernel import (
    distributed_behavior_kernel_loss,
    program_behavior_features,
)
from ember.ecp.natural_program import NaturalProgram
from ember.ecp.natural_program_gate import _behavior_kernel_qualification


def _program(*, process: torch.Tensor, language: torch.Tensor) -> NaturalProgram:
    batch, events, owners, width = process.shape
    return NaturalProgram(
        p_lang=language,
        p_scene=torch.randn(batch, owners, width),
        p_process=process,
        rho=torch.rand(batch, events).clamp_min(0.1),
        tau=torch.rand(batch, events, 2),
        sigma=torch.rand(batch, events, owners, width).clamp_min(0.1),
    )


def test_behavior_kernel_credit_stays_in_deployed_program_fields():
    torch.manual_seed(7)
    process = torch.randn(2, 4, 38, 8, requires_grad=True)
    language = torch.randn(2, 38, 8, requires_grad=True)
    program = _program(process=process, language=language)
    selected = (0, 16, 34, 1, 17, 35, 36, 37)
    feature = program_behavior_features(program, selected)
    assert feature.shape == (2, 8, 92)
    (feature * torch.randn_like(feature)).sum().backward()
    assert process.grad is not None
    assert language.grad is not None
    assert torch.isfinite(process.grad).all()
    assert process.grad.abs().sum() > 0


def test_behavior_kernel_loss_has_no_task_decoder_and_backpropagates():
    class Authority:
        meta_gradient_task_ids = frozenset((1, 2, 3))
        target_gradient_task_ids = frozenset((72, 73, 74))
        fit_task_ids = (1, 2, 3, 72, 73, 74)

        def __init__(self):
            self.latent = torch.nn.functional.normalize(torch.randn(6, 8, 5), dim=-1)
            self.value = torch.einsum("ntd,mtd->tnm", self.latent, self.latent)
            self.index = {task: row for row, task in enumerate((1, 2, 3, 72, 73, 74))}

        def kernel(self, task_ids, *, kind):
            assert kind in {"panel_a", "consensus"}
            rows = torch.tensor([self.index[int(task)] for task in task_ids])
            return self.value.index_select(1, rows).index_select(2, rows)

    torch.manual_seed(8)
    features = torch.randn(6, 2, 8, 12, requires_grad=True)
    authority = Authority()
    loss, metrics = distributed_behavior_kernel_loss(
        local_features=features,
        local_task_ids=torch.tensor([1, 2, 3, 72, 73, 74]),
        authority=authority,
        world_size=1,
        cross_view_weight=0.5,
        scope_weights={"joint": 0.5, "meta": 0.25, "target": 0.25},
    )
    cross_flipped = Authority()
    cross_flipped.value = authority.value.clone()
    cross_flipped.value[:, :3, 3:] *= -1
    cross_flipped.value[:, 3:, :3] *= -1
    flipped_loss, _ = distributed_behavior_kernel_loss(
        local_features=features,
        local_task_ids=torch.tensor([1, 2, 3, 72, 73, 74]),
        authority=cross_flipped,
        world_size=1,
        cross_view_weight=0.5,
        scope_weights={"joint": 0.5, "meta": 0.25, "target": 0.25},
    )
    lifted = torch.cat(
        (
            torch.full((6, 8, 1), 2**-0.5),
            authority.latent * (2**-0.5),
        ),
        dim=-1,
    )
    exact_loss, _ = distributed_behavior_kernel_loss(
        local_features=torch.stack((lifted, lifted), dim=1),
        local_task_ids=torch.tensor([1, 2, 3, 72, 73, 74]),
        authority=authority,
        world_size=1,
        cross_view_weight=0.5,
        scope_weights={"joint": 0.5, "meta": 0.25, "target": 0.25},
    )
    collapsed = torch.ones(6, 2, 8, 6)
    collapsed = torch.nn.functional.normalize(collapsed, dim=-1)
    collapsed_loss, _ = distributed_behavior_kernel_loss(
        local_features=collapsed,
        local_task_ids=torch.tensor([1, 2, 3, 72, 73, 74]),
        authority=authority,
        world_size=1,
        cross_view_weight=0.5,
        scope_weights={"joint": 0.5, "meta": 0.25, "target": 0.25},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert not torch.allclose(loss, flipped_loss)
    assert exact_loss < 1e-12
    assert collapsed_loss > 0.01
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert set(metrics) == {
        "behavior_kernel_alignment_loss",
        "behavior_kernel_cross_view_loss",
        "behavior_kernel_correlation_a",
        "behavior_kernel_correlation_b",
        "behavior_kernel_joint_correlation_a",
        "behavior_kernel_joint_correlation_b",
        "behavior_kernel_meta_correlation_a",
        "behavior_kernel_meta_correlation_b",
        "behavior_kernel_target_correlation_a",
        "behavior_kernel_target_correlation_b",
        "behavior_kernel_joint_program_std_a",
        "behavior_kernel_joint_program_std_b",
        "behavior_kernel_joint_teacher_std",
        "behavior_kernel_meta_program_std_a",
        "behavior_kernel_meta_program_std_b",
        "behavior_kernel_meta_teacher_std",
        "behavior_kernel_target_program_std_a",
        "behavior_kernel_target_program_std_b",
        "behavior_kernel_target_teacher_std",
    }


def test_behavior_kernel_gate_fits_on_gradient_tasks_and_populates_internal_holdout():
    torch.manual_seed(9)
    train_ids = (1, 2, 3, 72, 73, 74)
    held_ids = (4, 5, 6, 75, 76, 77)
    task_ids = train_ids + held_ids
    latent = torch.randn(len(task_ids), 8, 7)
    coordinates = torch.randn(len(task_ids), 8, 4)

    class Authority:
        selected_targets = (0, 16, 34, 1, 17, 35, 36, 37)
        fit_task_ids = train_ids
        held_task_ids = held_ids
        meta_gradient_task_ids = frozenset(train_ids[:3])
        target_gradient_task_ids = frozenset(train_ids[3:])

        def __init__(self):
            self.index = {task: row for row, task in enumerate(task_ids)}

        def kernel(self, ids, *, kind):
            assert kind == "panel_b"
            rows = torch.tensor([self.index[int(task)] for task in ids])
            values = latent.index_select(0, rows)
            return torch.einsum("ntd,mtd->tnm", values, values)

        def target(self, task, *, standardized):
            assert standardized
            return coordinates[self.index[int(task)]]

        @staticmethod
        def decode(value):
            return value

    def records(ids, *, held):
        rows = []
        for task in ids:
            feature = latent[task_ids.index(task)]
            views = {"same_a": feature, "same_b": feature}
            if held:
                views.update({"k1": feature, "k4": feature})
            rows.append(
                {
                    "authority_id": task,
                    "behavior_features": views,
                    "behavior_predictions": None,
                }
            )
        return rows

    runtime = SimpleNamespace(
        behavior_codes=Authority(),
        context=SimpleNamespace(device=torch.device("cpu")),
        config={"behavior_alignment": {"evaluator_ridge": 0.01}},
    )
    held_records = records(held_ids, held=True)
    topology = _behavior_kernel_qualification(
        runtime, records(train_ids, held=False), held_records
    )
    assert topology["train"]["role_equal_program_to_behavior_a"] > 0.99
    assert topology["held"]["role_equal_program_to_behavior_b"] > 0.99
    assert all(
        set(row["behavior_predictions"]) == {"same_a", "same_b", "k1", "k4"}
        for row in held_records
    )


def test_behavior_authority_loads_and_decodes(tmp_path):
    first = tmp_path / "factors_a"
    second = tmp_path / "factors_b"
    first.mkdir()
    second.mkdir()
    official = tuple(range(0, 71, 5)) + (71, 76, 81, 86, 91)
    fit75 = tuple(task for task in range(95) if task not in official)
    fit, held = fixed_internal_behavior_fold(fit75)
    authority_tasks = tuple(sorted((*fit, *held)))
    task_index = {task: row for row, task in enumerate(authority_tasks)}
    coordinates = torch.randn(75, 8, 16)
    mean = torch.randn(8, 16)
    scale = torch.rand(8, 16).clamp_min(0.1)
    tensors = {
        "task_ids": torch.tensor(authority_tasks),
        "fit_task_ids": torch.tensor(fit),
        "held_task_ids": torch.tensor(held),
        "official_held_task_ids": torch.tensor(official),
        "selected_targets": torch.tensor([0, 16, 34, 1, 17, 35, 36, 37]),
        "coordinates": coordinates,
        "mean": mean,
        "scale": scale,
        "eigenvectors": torch.randn(8, 60, 16),
        "eigenvalues": torch.rand(8, 16).clamp_min(0.1),
        "norms": torch.rand(8, 75).clamp_min(0.1),
        "train_sqrt_weights": torch.full((60,), 60**-0.5),
        "panel_a_gram": torch.eye(75).expand(8, -1, -1).clone(),
        "panel_b_gram": torch.eye(75).expand(8, -1, -1).clone(),
        "consensus_gram": torch.eye(75).expand(8, -1, -1).clone(),
    }
    tensor_path = tmp_path / "codes.safetensors"
    save_file(tensors, tensor_path)
    manifest = {
        "schema_version": BEHAVIOR_CODE_SCHEMA,
        "status": "complete",
        "tensor_file": tensor_path.name,
        "tensor_bytes": tensor_path.stat().st_size,
        "dimension": 16,
        "factor_roots": [first.name, second.name],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    authority = load_behavior_code_authority(
        manifest_path, asset_root=tmp_path, device=torch.device("cpu")
    )
    standardized = authority.target(90, standardized=True)
    torch.testing.assert_close(
        authority.decode(standardized),
        coordinates[task_index[90]],
        atol=1e-6,
        rtol=1e-6,
    )
    assert authority.task_ids == authority_tasks
    assert authority.fit_task_ids == fit
    assert authority.held_task_ids == held
    assert authority.official_held_task_ids == official


def test_model_only_initialization_is_strict_when_kernel_adds_no_parameters(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(3, 4)

    source = Model()
    target = Model()
    checkpoint = tmp_path / "checkpoints" / "macro_00000020"
    checkpoint.mkdir(parents=True)
    source_state = source.state_dict()
    weights = checkpoint / "ecp.safetensors"
    save_file(source_state, weights)
    manifest = {
        "schema_version": "ember_ecp_checkpoint_v1",
        "stage": "g2_natural_program",
        "next_macro": 20,
        "run_contract_schema": "ember_ecp_natural_program_g2_run_v2",
        "world_size": 4,
        "files": {"ecp.safetensors": {"bytes": weights.stat().st_size}},
    }
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    report = load_program_model_initialization(
        target,
        checkpoint,
        device=torch.device("cpu"),
        allowed_new_prefix=None,
        expected_macro=20,
    )
    assert torch.equal(target.base.weight, source.base.weight)
    assert report["fresh_tensors"] == 0
    assert report["optimizer_loaded"] is False
