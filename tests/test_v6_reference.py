"""CPU checks of the isolated v6 protocol, complete replay and evidence boundary."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.eval_adapters import episode_adapter_fields, validate_episode_adapter_fields
from ember.pi05_source_checkpoint import DistributedContext
from ember.v6_reference import contract, runtime
from ember.v6_reference.artifacts import inspect_checkpoint
from ember.writer.materialization import adapter_metadata, planned_episodes, selection_contract
from ember.writer.training import _config, _optimization, _training_state


ROOT = Path(__file__).resolve().parents[1]
GIT = {"branch": "", "commit": "a" * 40, "upstream": None, "dirty_paths": [],
       "authority_ref": "origin/codex/v6-causal-reference", "authority_contains_commit": True}


@pytest.fixture
def config():
    return _config(ROOT / "configs/pi05_v6_causal_reference.json")


@pytest.mark.parametrize("section,key,value", [
    ("data", "action_demos", list(range(50))), ("data", "task_ids", [1] * 24),
    ("optimization", "lr", 3e-4), ("model", "text_meta_lora_rank", 0),
    ("model", "action_horizon", 25), ("observer", "probe_seed", 1729),
    ("evidence", "final_checkpoint_selection", True),
])
def test_reference_rejects_recipe_or_information_wall_drift(config, section, key, value):
    changed = deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ValueError):
        contract.validate_config(changed)


def test_reference_authority_does_not_admit_dirty_unpushed_or_unrelated_code():
    assert contract.execution_authority(GIT)
    for changed in (GIT | {"dirty_paths": [" M source.py"]}, GIT | {"authority_contains_commit": False},
                    GIT | {"authority_ref": "origin/another-experiment"}, GIT | {"branch": "main"}):
        assert not contract.execution_authority(changed)


def test_full_replay_preserves_all_three_meta_gradients_and_task_weight(monkeypatch, config):
    class ToyWriter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.meta = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
            self.head = torch.nn.Parameter(torch.tensor(0.5))
            self.calls = []

        def forward(self, value, *, policy):
            self.calls.append(torch.is_grad_enabled())
            return {"factor": self.head * (self.meta * value).sum()}

    writer = ToyWriter()
    value = torch.tensor([2., 3., 5.])
    trace = {"action_demos": [16, 17] * 32, "action_frames": [0] * 64, "policy_rng_seed": 937}
    data = SimpleNamespace(
        tasks={0: SimpleNamespace(authority=SimpleNamespace(language="exact task"))},
        load_videos=lambda *args: ((value,), (torch.arange(3),)),
        action_batch=lambda *args, **kwargs: ({}, trace),
    )
    target_runtime = SimpleNamespace(
        policy=object(), lora=object(), state=SimpleNamespace(writer=writer),
        processor=SimpleNamespace(training_batch=lambda batch: batch),
        pack=lambda *args: (value,),
    )
    observed = {}

    def functional_gradient(policy, state, lora, **kwargs):
        observed.update(kwargs)
        assert not state["factor"].requires_grad
        return (state["factor"] - 1).square(), {}, {"factor": 2 * (state["factor"] - 1)}

    monkeypatch.setattr(runtime, "functional_lora_loss_gradient", functional_gradient)
    context = SimpleNamespace(device=torch.device("cpu"))
    engine = runtime.V6SupervisedEngine(target_runtime, data, context, config)
    result = engine.backward({"task": 0, "video_demos": (0,), "occurrence": 0, "query_seed": 17})
    assert writer.calls == [False, True]
    expected_scale = 0.25 * 2 * (float(writer.head.detach() * (writer.meta.detach() * value).sum()) - 1)
    torch.testing.assert_close(writer.meta.grad, expected_scale * writer.head.detach() * value)
    torch.testing.assert_close(writer.head.grad, expected_scale * (writer.meta.detach() * value).sum())
    assert result["queries"] == 64 and result["task_weight"] == 0.25 and result["prefix_cache_bytes"] == 0
    assert observed["policy_rng_seed"] == trace["policy_rng_seed"]
    assert observed["flow_time_sampling_scheme"] == runtime.INDEPENDENT_BETA_TIME_SAMPLING_SCHEME
    assert observed["flow_noise_sampling_scheme"] == runtime.INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME


@pytest.fixture
def checkpoint(tmp_path, config):
    model = torch.nn.Module()
    model.writer = torch.nn.Module()
    model.writer.semantic_encoder = torch.nn.Module()
    model.writer.semantic_encoder.register_buffer("fixed_suffix_noise", torch.zeros(50, 32))
    model.writer.parameter = torch.nn.Parameter(torch.ones(2))
    optimizer, scheduler = _optimization(model, config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    path = save_ecp_checkpoint(
        output_dir=tmp_path, macro=200, stage=contract.STAGE, context=context,
        model=model, optimizer=optimizer, scheduler=scheduler, run_contract_schema=contract.RUN_SCHEMA,
        metrics_rows=800, sampler_state={"next_step": 200}, training_state=_training_state(config, 200),
    )
    run = {"schema_version": contract.RUN_SCHEMA, "stage": contract.STAGE, "mode": "exploratory",
           "git": GIT, "config": config, "model_config": config["model"], "topology": {"world_size": 1}}
    (tmp_path / "run_contract.json").write_text(json.dumps(run))
    return path, run


def test_checkpoint_inspector_keeps_exploratory_schema_and_complete_state(checkpoint):
    path, run = checkpoint
    checked, record = inspect_checkpoint(path)
    assert checked == run and record["macro"] == 200 and record["kind"] == contract.BANK_KIND
    assert adapter_metadata("condition", record)["schema_version"] == contract.ADAPTER_SCHEMA
    for changed in (run | {"mode": "formal"}, run | {"mode": "profile"},
                    run | {"model_config": run["model_config"] | {"action_horizon": 25}}):
        with pytest.raises(ValueError):
            inspect_checkpoint(path, changed)
    trainer_path = path / "trainer_state.pt"
    trainer = torch.load(trainer_path, weights_only=True)
    trainer["sampler_state"]["next_step"] = 199
    torch.save(trainer, trainer_path)
    manifest_path = path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["trainer_state.pt"]["bytes"] = trainer_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sampling cursor"):
        inspect_checkpoint(path)


def test_v6_uses_current_strict400_schedule_and_excludes_controls():
    selection = selection_contract(role="validation", task_ids=[1, 3, 11, 13, 23, 26, 31, 32],
        cardinality=1, arm="correct", mode="per_init_ordinal", seed=20260907,
        init_state_ids=list(range(50)), video_pool=list(range(50)))
    contract.validate_selection(selection)
    rows = [episode for task in selection["task_ids"] for episode in planned_episodes(selection, task)]
    assert len(rows) == 400
    for task in selection["task_ids"]:
        assert {row["teacher_demo_indices"][0] for row in planned_episodes(selection, task)} == set(range(50))
    for changed in (selection | {"K": 4}, selection | {"arm": "same_task_other"},
                    selection | {"init_state_ids": list(range(10))}):
        with pytest.raises(ValueError):
            contract.validate_selection(changed)


def test_episode_dispatch_distinguishes_v6_from_horizon():
    from ember.writer.evaluation import episode_evidence

    episode = {"init_state_id": 0, "condition_id": "condition", "teacher_demo_indices": [4]}
    task = {"suite": "libero_spatial", "task_id": 1, "global_task_id": 1, "episodes": [episode]}
    adapter = {"kind": contract.BANK_KIND, "tasks": [task], "conditions": [{"condition_id": "condition"}],
               "selection": {"seed": 20260907, "mode": "per_init_ordinal", "K": 1}, "arm": "correct",
               "writer_checkpoint": {}, "method": {}, "source": {"checkpoint": "/source"}}
    evidence = episode_evidence(adapter, task, episode)
    assert evidence["schema_version"] == contract.EPISODE_SCHEMA
    row = episode_adapter_fields({"adapter": adapter}, object(), SimpleNamespace(evidence=evidence))
    assert set(row) == {"v6_reference_lora"}
    assert validate_episode_adapter_fields(adapter, row, suite="libero_spatial", task_id=1, init_state_id=0)
    assert not validate_episode_adapter_fields(adapter, row | {"horizon_writer_lora": evidence},
                                               suite="libero_spatial", task_id=1, init_state_id=0)


def test_materialization_calls_v6_once_without_observer_cache(tmp_path):
    import numpy as np
    from safetensors import safe_open
    from ember.lora import identity_lora_state
    from ember.pi05_lora import load_pi05_lora_contract
    from ember.writer.materialization import _compile_condition

    lora = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    calls = []

    def generate(frames, indices, language):
        calls.append((len(frames), indices[0].tolist(), language, torch.is_grad_enabled()))
        return identity_lora_state(lora)

    video = SimpleNamespace(raw_frame_count=6, frames=np.zeros((2, 3, 2, 2), dtype=np.uint8),
                            frame_indices=np.array([0, 5]))
    task = SimpleNamespace(authority=SimpleNamespace(task_id=0, language="exact task"),
                           episode_lengths=[6], suite="libero_spatial", suite_task_id=0)
    reference = {"kind": contract.BANK_KIND, "path": "/run/checkpoints/macro_00000200", "macro": 200}
    # Deliberately no observer/cache member: only the historical full Writer owns this forward.
    record = _compile_condition(SimpleNamespace(generate=generate, lora=lora),
        SimpleNamespace(load=lambda *args: video), task, (0,), tmp_path, reference)
    assert calls == [(1, [0, 5], "exact task", False)]
    assert record["writer_invocations"] == 1 and record["single_complete_rank16"] is True
    with safe_open(record["adapter"]["path"], framework="pt", device="cpu") as handle:
        assert len(handle.keys()) == 76
        assert handle.metadata()["schema_version"] == contract.ADAPTER_SCHEMA


def test_cross_architecture_pairing_preserves_video_schedule():
    from ember.pi05_eval_results import paired_success_comparison
    from ember.pi05_assets import Pi05EvaluationError

    common = {"suite": "libero_goal", "task_id": 0, "init_state_id": 0, "language": "open drawer",
              "env_seed": 7, "policy_seed_root": 7, "split_role": "train", "policy_noise_seeds": [8, 9],
              "success": True}
    video = {"video_ordinal": 32, "selection_seed": 20260907, "selection_mode": "per_init_ordinal",
             "K": 1, "paired_correct_demos": [4], "paired_other_demos": [17]}
    old = {"rows": [common | {"v6_reference_lora": video}]}
    new = {"rows": [common | {"horizon_writer_lora": dict(video)}]}
    assert paired_success_comparison(old, new)["retained"] == 1
    new["rows"][0]["horizon_writer_lora"]["paired_correct_demos"] = [5]
    with pytest.raises(Pi05EvaluationError, match="video ordinal or schedule"):
        paired_success_comparison(old, new)
