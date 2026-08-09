from __future__ import annotations

import json
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_COMPLETION_SCHEMA,
    V6_PRIOR_GRADIENT_PROFILE_SCHEMA,
    load_v6_prior_config,
    suggest_auxiliary_weight,
)
from ember.expert_manifold.v6_prior_runtime import (
    RuntimeSegment,
    _resolve_segment,
    _sampled_video_cost,
    _scheduler,
    _validate_collective_environment,
)
from ember.expert_manifold.v6_prior_run_contract import (
    build_run_contract,
    cursor_contract,
    rank_topology,
)
from ember.expert_manifold.v6_prior_policy_batch import (
    policy_rng_seed_for_logical_batch,
)
from ember.expert_manifold.v6_prior_training import (
    _mean_trainable_gradients,
    _run_gradient_profile,
    _task_objective,
    finalize_args,
)
from ember.expert_manifold.v6_prior_step import GeneratedCounterfactualPair
from ember.pi05_source_checkpoint import DistributedContext


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/pi05_v6_condition_local_tangent_tube_writer_v3.json"
)


def test_formal_segment_requires_a_clean_pushed_strict_profile_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(load_v6_prior_config(CONFIG))
    config["formal_run"]["status"] = "sealed_from_live_a40_resume_profile_evidence"
    config["formal_run"]["formal_result"] = None
    resume = config["profile_run"]["artifact_evidence"]
    gradient_commit = resume["gradient_commit"]
    profile_commit = resume["profile_git"]["commit"]
    formal_commit = "formal-seal-descendant"
    state = {
        "branch": "codex/formal-seal",
        "commit": formal_commit,
        "origin_main": "main",
        "upstream": "origin/codex/formal-seal",
        "upstream_commit": formal_commit,
        "dirty_paths": [],
    }
    expected_lineage = {
        (gradient_commit, profile_commit),
        (profile_commit, formal_commit),
    }
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime.git_state",
        lambda _root: state,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime.git_commit_is_strict_ancestor",
        lambda ancestor, descendant: (ancestor, descendant) in expected_lineage,
    )
    args = SimpleNamespace(
        mode="formal",
        resume=None,
        stop_after_macro=10,
        num_workers=2,
    )
    segment = _resolve_segment(
        args,
        config,
        DistributedContext(0, 0, 6, torch.device("cuda:0")),
    )
    assert (segment.start_macro, segment.stop_macro) == (0, 10)

    state["commit"] = profile_commit
    state["upstream_commit"] = profile_commit
    with pytest.raises(
        ExpertManifoldError,
        match="runtime differs from its sealed segment",
    ):
        _resolve_segment(
            args,
            config,
            DistributedContext(0, 0, 6, torch.device("cuda:0")),
        )


def test_v6_prior_video_cost_includes_true_final_frame() -> None:
    assert _sampled_video_cost(1, 5) == 1
    assert _sampled_video_cost(6, 5) == 2
    assert _sampled_video_cost(7, 5) == 3
    assert _sampled_video_cost(105, 5) == 22


def test_v6_prior_auxiliary_weight_obeys_both_trainable_groups() -> None:
    positive = {"compiler": 2.0, "factor_heads": 4.0}
    auxiliary = {"compiler": 8.0, "factor_heads": 1.0}
    assert suggest_auxiliary_weight(
        positive,
        auxiliary,
        maximum_fraction=0.25,
    ) == pytest.approx(0.0625)
    assert (
        suggest_auxiliary_weight(
            positive,
            {"compiler": 0.0, "factor_heads": 0.0},
            maximum_fraction=0.25,
        )
        == 0.0
    )
    assert (
        suggest_auxiliary_weight(
            positive,
            {"compiler": 0.25, "factor_heads": 0.0},
            maximum_fraction=0.25,
        )
        == 1.0
    )
    assert (
        suggest_auxiliary_weight(
            positive,
            {"compiler": -1.0, "factor_heads": 1.0},
            maximum_fraction=0.25,
        )
        == 0.0
    )
    with pytest.raises(
        ExpertManifoldError,
        match="invalid v6-prior gradient fraction",
    ):
        suggest_auxiliary_weight(
            positive,
            {"compiler": 1.0, "factor_heads": 1.0},
            maximum_fraction=0.0,
        )


def test_v6_prior_rank_topology_records_per_rank_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_run_contract.socket.gethostname",
        lambda: "gpu02",
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_run_contract.torch.cuda.get_device_name",
        lambda _device: "NVIDIA A40",
    )
    context = DistributedContext(
        0,
        0,
        1,
        torch.device("cuda:0"),
        numa_node=1,
        cpu_affinity=(16, 17),
    )
    assert rank_topology(context, physical_index=lambda _local_rank: 5) == [
        {
            "rank": 0,
            "local_rank": 0,
            "host": "gpu02",
            "cuda_visible_devices": "5",
            "device_name": "NVIDIA A40",
            "physical_gpu": 5,
            "device": "cuda:0",
            "numa_node": 1,
            "cpu_affinity": [16, 17],
        }
    ]


def test_v6_prior_collective_environment_does_not_mandate_an_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "Simple",
    }.items():
        monkeypatch.setenv(name, value)
    context = DistributedContext(0, 0, 6, torch.device("cuda:0"))
    _validate_collective_environment(context)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _validate_collective_environment(context)


def test_v6_prior_run_contract_retains_full_git_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    state = {
        "branch": "codex/frozen",
        "commit": "abc123",
        "origin_main": "main123",
        "upstream": "origin/codex/frozen",
        "upstream_commit": "abc123",
        "dirty_paths": [],
    }
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_run_contract.torch.cuda.get_device_name",
        lambda _device: "NVIDIA A40",
    )
    task = SimpleNamespace(
        ordinal=0,
        global_task_id=0,
        suite="libero_spatial",
        task_id=0,
        language="do the task",
        authority=SimpleNamespace(
            path=tmp_path / "task.hdf5",
            expected_bytes=123,
        ),
    )
    segment = RuntimeSegment(1, (), 0, 1, 49, 50)
    config = {
        "gradient_profile": {"schedule_macro": 49},
        "data": {"action_queries_per_task": 20},
        "method": {},
        "information_wall": {},
        "initialization": {
            "dynamic_anchor": (
                "training_only_frozen_macro0_compiler_and_factor_heads"
            ),
            "resume_writer_load_scope": "trainable_compiler_and_factor_heads_only",
        },
        "writer": {"activation_checkpointing": True},
        "expert_basis": {"expert_step": 2000},
        "objective": {},
        "optimization": {"functional_policy_microbatch_size": 16},
    }
    args = SimpleNamespace(
        mode="gradient-profile",
        config=config_path,
        expert_bank_root=tmp_path / "experts",
        data_root=tmp_path / "data",
        num_workers=2,
    )
    schedule = SimpleNamespace(
        consumed_identity_summary=lambda *_args: {"task_visits": 1}
    )
    warm_start = SimpleNamespace(
        checkpoint=tmp_path / "warm",
        state_tensor_count=600,
        state_value_count=1,
    )
    ownership = SimpleNamespace(
        frozen_parameter_count=7_060_992,
        trainable_parameter_count=3_714_304,
        frozen_tensor_count=482,
        trainable_tensor_count=41,
    )
    contract = build_run_contract(
        args=args,
        config=config,
        context=DistributedContext(0, 0, 1, torch.device("cuda:0")),
        segment=segment,
        source={},
        tokenizer={},
        tasks=(task,),
        sampler=object(),
        video_schedule=schedule,
        expert={"training_commit": "expert", "tasks": []},
        warm_start=warm_start,
        ownership=ownership,
        dynamic_anchor=SimpleNamespace(parameter_count=3_714_304, tensor_count=41),
        trainable_names=("compiler.weight",),
        git_state_fn=lambda _root: state,
        rank_topology_fn=lambda _context: [{"rank": 0}],
    )
    assert contract["git"] == state
    assert contract["ownership"]["dynamic_anchor"] == {
        "parameter_count": 3_714_304,
        "tensor_count": 41,
        "optimizer_owned": False,
        "checkpoint_owned": False,
        "deployment_owned": False,
    }


def test_v6_prior_gradient_profile_writes_sealed_panel_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    compiler = torch.nn.Parameter(torch.zeros(2))
    factor = torch.nn.Parameter(torch.zeros(1))
    batches = iter({"ordinal": ordinal} for ordinal in range(24))

    def task_objective(_runtime, *, macro, microtask, batch):
        assert macro == 49
        assert microtask == batch["ordinal"]
        return SimpleNamespace(
            ordinal=batch["ordinal"],
            pair=None,
            functional_gradients=None,
            auxiliary=None,
        )

    def components(**_kwargs):
        return SimpleNamespace(
            positive=(torch.tensor([2.0, 0.0]), torch.tensor([4.0])),
            projection=(torch.tensor([8.0, 0.0]), torch.tensor([1.0])),
            ranking=(torch.tensor([1.0, 0.0]), torch.tensor([8.0])),
        )

    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training._task_objective",
        task_objective,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.parameter_gradient_components",
        components,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training._task_record",
        lambda value: {
            "task_ordinal": value.ordinal,
            "task_visit": 49,
            "counterfactual_kind": ("reversed", "shuffled", "wrong")[value.ordinal % 3],
        },
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training._runtime_maximums",
        lambda _context, _started, input_wait: (12.5, 1_000, 2_000, input_wait),
    )
    runtime = SimpleNamespace(
        context=DistributedContext(0, 0, 1, torch.device("cpu")),
        trainable_names=("compiler.weight", "factor_heads.weight"),
        trainable_parameters=(compiler, factor),
        iterator=batches,
        segment=RuntimeSegment(1, (), 0, 1, 49, 50),
        config={
            "data": {"action_queries_per_task": 20},
            "objective": {
                "auxiliary_weights": {
                    "maximum_fraction_of_positive_gradient_per_auxiliary": 0.25
                }
            },
            "gradient_profile": {"seal_rule": "bounded"},
        },
        policy=torch.nn.Identity(),
        args=SimpleNamespace(output_dir=tmp_path),
        run_contract={
            "data": {
                "consumed_schedule": {
                    "query": {
                        "global_examples": 480,
                        "unique_query_rows": 480,
                    }
                }
            }
        },
    )
    _run_gradient_profile(runtime)
    profile = json.loads(
        (tmp_path / "gradient_profile.json").read_text(encoding="utf-8")
    )
    completion = json.loads((tmp_path / "completion.json").read_text(encoding="utf-8"))
    assert profile["schema_version"] == V6_PRIOR_GRADIENT_PROFILE_SCHEMA
    assert profile["schedule_macro"] == 49
    assert profile["task_count"] == 24
    assert profile["action_queries_per_task"] == 20
    assert profile["total_action_queries"] == 480
    assert profile["unique_action_queries"] == 480
    assert profile["counterfactual_counts"] == {
        "reversed": 8,
        "shuffled": 8,
        "wrong": 8,
    }
    assert len(profile["task_records"]) == 24
    assert completion == {
        "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
        "mode": "gradient-profile",
        "completed_diagnostic_macros": 1,
        "schedule_start_macro": 49,
        "schedule_stop_macro": 50,
        "gradient_profile_complete": True,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def test_v6_prior_scheduler_is_low_lr_warmup_then_fifty_macro_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW((parameter,), lr=3e-5)
    config = {
        "optimization": {
            "optimizer": {"peak_lr": 3e-5},
            "scheduler": {
                "warmup_macros": 2,
                "total_macros": 50,
                "decay_lr": 3e-6,
            },
        }
    }
    scheduler = _scheduler(optimizer, config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    for _ in range(49):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-6)


def test_v6_prior_flat_gradient_reduction_is_one_global_task_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([1.0, 2.0])
    second.grad = torch.tensor([3.0])

    def all_reduce(value: torch.Tensor, *, op: object) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert torch.equal(value, torch.tensor([1.0, 2.0, 3.0]))
        value.mul_(2).add_(torch.tensor([2.0, 4.0, 6.0]))

    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.dist.all_reduce",
        all_reduce,
    )
    runtime = SimpleNamespace(
        trainable_parameters=(first, second),
        context=DistributedContext(0, 0, 2, torch.device("cpu")),
    )
    _mean_trainable_gradients(runtime)
    assert torch.equal(first.grad, torch.tensor([2.0, 4.0]))
    assert torch.equal(second.grad, torch.tensor([6.0]))


def test_v6_prior_policy_randomness_is_keyed_by_the_complete_logical_batch() -> None:
    runtime = SimpleNamespace(
        config={
            "data": {"action_queries_per_task": 20},
            "optimization": {"seed": 7},
            "objective": {
                "positive_policy_randomness": {
                    "scope": "one_independent_flow_noise_and_time_per_action_query",
                    "seed_scheme": (
                        "task_logical_batch_keyed_stateless_policy_cpu_cuda_splitmix64_v3"
                    ),
                    "flow_time_sampling_scheme": (
                        "task_logical_batch_keyed_independent_beta15_time_v2"
                    ),
                    "flow_noise_sampling_scheme": (
                        "task_logical_batch_keyed_independent_gaussian_v2"
                    ),
                }
            },
        }
    )
    batch = {
        "demo_index": torch.arange(20) % 5,
        "frame_index": torch.arange(20) * 3,
    }
    seed = policy_rng_seed_for_logical_batch(
        runtime.config, batch, task_id=4, task_visit=9
    )
    assert seed == 3_295_656_931_063_255_022
    assert seed == policy_rng_seed_for_logical_batch(
        runtime.config,
        batch,
        task_id=4,
        task_visit=9,
    )
    changed = dict(batch)
    changed["frame_index"] = batch["frame_index"].clone()
    changed["frame_index"][0] += 1
    assert seed != policy_rng_seed_for_logical_batch(
        runtime.config,
        changed,
        task_id=4,
        task_visit=9,
    )


def test_v6_prior_task_objective_wires_logical_b20_into_physical_b16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    randomness = {
        "scope": "one_independent_flow_noise_and_time_per_action_query",
        "seed_scheme": (
            "task_logical_batch_keyed_stateless_policy_cpu_cuda_splitmix64_v3"
        ),
        "flow_time_sampling_scheme": (
            "task_logical_batch_keyed_independent_beta15_time_v2"
        ),
        "flow_noise_sampling_scheme": (
            "task_logical_batch_keyed_independent_gaussian_v2"
        ),
    }
    task = SimpleNamespace(ordinal=0, global_task_id=4)
    batch = {
        "task_id": torch.full((20,), 4),
        "demo_index": torch.arange(20) % 5,
        "frame_index": torch.arange(20) * 3,
    }
    pair = GeneratedCounterfactualPair(
        correct={"adapter": torch.tensor([1.0])},
        counterfactual={"adapter": torch.tensor([0.0])},
        correct_anchor={"adapter": torch.tensor([1.0])},
        counterfactual_anchor={"adapter": torch.tensor([0.0])},
        correct_raw_frames=10,
        correct_sampled_frames=3,
        counterfactual_raw_frames=10,
        counterfactual_sampled_frames=3,
    )
    captured: dict[str, object] = {}

    def functional_stub(
        policy: object,
        state: object,
        contract: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict, dict[str, torch.Tensor]]:
        captured.update(kwargs)
        assert policy is runtime.policy
        assert state is pair.correct
        assert contract is runtime.lora_contract
        return torch.tensor(0.5), {}, {"adapter": torch.tensor([0.25])}

    auxiliary = object()
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.torch.autocast",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.generate_counterfactual_pair",
        lambda **_kwargs: pair,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.functional_lora_loss_gradient",
        functional_stub,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_training.effective_auxiliary_output_gradients",
        lambda *_args, **_kwargs: auxiliary,
    )
    runtime = SimpleNamespace(
        config={
            "data": {
                "action_queries_per_task": 20,
                "counterfactual_seed": 23,
            },
            "optimization": {
                "seed": 7,
                "functional_policy_microbatch_size": 16,
            },
            "objective": {
                "positive_policy_randomness": randomness,
                "projection": {"smooth_l1_beta": 0.1},
                "ranking": {"required_margin": 0.1, "temperature": 0.1},
            },
        },
        sampler=SimpleNamespace(
            task_visit_for_step=lambda _macro, _microtask: (4, 0),
            action_demo_indices_for_task_visit=lambda _task, _visit: (1, 2),
        ),
        video_schedule=SimpleNamespace(
            demos_for_task_visit=lambda _task, _visit, **_kwargs: (3,)
        ),
        video_store=SimpleNamespace(load=lambda _task, _demo: object()),
        task_by_global_id={4: task},
        writer=object(),
        dynamic_anchor=object(),
        comparison_decoder=None,
        policy=object(),
        language_tokens={4: object()},
        context=SimpleNamespace(device=torch.device("cpu")),
        lora_contract=object(),
        processor=SimpleNamespace(training_batch=lambda _batch: {"policy": True}),
        expert_targets={"adapter": torch.zeros(1, 1)},
        args=SimpleNamespace(mode="gradient-profile"),
    )

    result = _task_objective(runtime, macro=0, microtask=0, batch=batch)
    assert result.functional_loss == torch.tensor(0.5)
    assert result.functional_gradients == {"adapter": torch.tensor([0.25])}
    assert result.auxiliary is auxiliary
    assert captured["batch"] == {"policy": True}
    assert captured["policy_microbatch_size"] == 16
    assert captured["collect_policy_details"] is False
    assert (
        captured["flow_time_sampling_scheme"] == randomness["flow_time_sampling_scheme"]
    )
    assert (
        captured["flow_noise_sampling_scheme"]
        == randomness["flow_noise_sampling_scheme"]
    )
    assert captured["policy_rng_seed"] == policy_rng_seed_for_logical_batch(
        runtime.config,
        batch,
        task_id=4,
        task_visit=0,
    )


def test_v6_prior_cursor_records_all_stateless_schedules() -> None:
    config = {
        "data": {
            "sampler_seed": 11,
            "teacher_video_seed": 13,
            "counterfactual_seed": 17,
            "action_queries_per_task": 20,
        }
    }
    assert cursor_contract(config, 25) == {
        "next_macro": 25,
        "task_visits_per_task": 25,
        "sampler_seed": 11,
        "teacher_video_seed": 13,
        "counterfactual_seed": 17,
        "counterfactual_phase": 1,
        "videos_per_task_visit": 1,
        "action_queries_per_task": 20,
    }


def test_v6_prior_runtime_rejects_external_config_copy(tmp_path) -> None:
    paths = {}
    for name in (
        "source_run",
        "checkpoint",
        "tokenizer_path",
        "data_root",
        "expert_bank_root",
        "warm_start",
    ):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    external = tmp_path / "pi05_v6_condition_local_tangent_tube_writer_v3.json"
    external.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        config=external,
        output_dir=tmp_path / "output",
        resume=None,
        stop_after_macro=1,
        num_workers=2,
        **paths,
    )
    with pytest.raises(
        ExpertManifoldError,
        match="tracked canonical config",
    ):
        finalize_args(args)
