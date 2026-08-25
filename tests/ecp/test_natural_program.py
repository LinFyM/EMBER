from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.ecp.natural_program_data import (
    NaturalProgramSample,
    NaturalProgramSchedule,
    _pack_cross_episode_supervision,
    _query_indices,
    _windowed_rising,
)
from ember.ecp.natural_program_gate import _build_report
from ember.ecp.natural_program_labels import _predicate_rising
from ember.ecp.natural_program import NaturalProgram, NaturalProgramModel
from ember.ecp.natural_program_authority import (
    _pure_native_inventory,
    _run_authority_commit,
)
from ember.ecp.natural_program_objective import _temporal_residual_mse
from ember.ecp.natural_program_training import _scheduler
from ember.ecp.stage0 import ECPVideoEncoderOutput


class _FakeEncoder(torch.nn.Module):
    def __init__(self, width: int, events: int) -> None:
        super().__init__()
        self.width = width
        self.events = events
        self.language = torch.nn.Embedding(64, width)
        self.frame_projection = torch.nn.Linear(1, width)
        self.register_buffer("fixed_suffix_noise", torch.ones(50, 32))

    def embed_language_conditions(self, _policy, tokens):
        return self.language(tokens)

    def forward(
        self,
        *,
        frames,
        video_offsets,
        suffix_noise,
        **_kwargs,
    ):
        boundaries = video_offsets.tolist()
        lengths = [stop - start for start, stop in zip(boundaries, boundaries[1:])]
        maximum = max(lengths)
        video_count = len(lengths)
        mask = torch.zeros(video_count, maximum, dtype=torch.bool)
        frame = torch.zeros(video_count, maximum, self.width)
        scale = suffix_noise.mean()
        for video, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            value = frames[start:stop].float().mean(dim=(1, 2, 3), keepdim=False)
            value = self.frame_projection(value[:, None] / 255.0)
            frame[video, : stop - start] = value
            mask[video, : stop - start] = True
        slot = torch.linspace(0.25, 1.0, self.events)[None, :, None, None]
        summary = frame.sum(1) / mask.sum(1, keepdim=True)
        process = (summary[:, None, None] + 0.01 * scale) * slot
        process = process.expand(-1, -1, 38, -1)
        presence = torch.sigmoid(
            torch.linspace(-1.0, 1.0, self.events)[None].expand(video_count, -1)
        )
        posterior = torch.zeros(video_count, maximum, self.events)
        for video, length in enumerate(lengths):
            ids = torch.linspace(0, self.events - 1, length).round().long()
            posterior[video, torch.arange(length), ids] = 1.0
        assignment = posterior.transpose(1, 2)[..., None]
        patch = frame[:, :, None].expand(-1, -1, 4, -1)
        return ECPVideoEncoderOutput(
            process=process,
            presence=presence,
            uncertainty=torch.full_like(process, 0.1),
            assignment=assignment,
            state_posterior=posterior,
            confidence=torch.zeros(video_count, maximum, 1),
            frame_mask=mask,
            program_summary=summary,
            frame_owner_evidence=frame[:, :, None].expand(-1, -1, 38, -1),
            patch_states=patch,
            language_summary=summary,
            scene_transition=torch.cat((summary, summary, summary), dim=-1),
        )


def _model(width: int = 8, events: int = 4) -> NaturalProgramModel:
    return NaturalProgramModel(
        _FakeEncoder(width, events),
        prefix_width=width,
        width=width,
        owners=38,
        event_slots=events,
        action_phases=5,
        predicate_slots=3,
    )


def test_formal_run_authority_stays_on_its_frozen_commit() -> None:
    repository = {
        "commit": "frozen-formal-commit",
        "authority_commit": "newer-origin-main-tip",
    }
    assert _run_authority_commit(repository, "formal") == "frozen-formal-commit"
    assert _run_authority_commit(repository, "profile") == "newer-origin-main-tip"


def test_k1_is_exact_identity_and_k2_is_permutation_invariant() -> None:
    torch.manual_seed(3)
    model = _model()
    p_lang = torch.randn(1, 38, 8)
    scene = torch.randn(2, 38, 8)
    process = torch.randn(2, 4, 38, 8)
    presence = torch.sigmoid(torch.randn(2, 4))
    tau = torch.stack(
        (
            torch.linspace(0.0, 1.0, 4).expand(2, -1),
            torch.full((2, 4), 0.1),
        ),
        dim=-1,
    )
    sigma = torch.rand_like(process).add_(0.1)

    single, alignment = model._aggregate(
        p_lang=p_lang,
        local_scene=scene[:1],
        local_process=process[:1],
        local_presence=presence[:1],
        local_tau=tau[:1],
        local_sigma=sigma[:1],
        video_set_offsets=torch.tensor([0, 1]),
    )
    assert torch.equal(alignment[0], torch.eye(4))
    assert torch.equal(single.p_scene[0], scene[0])
    assert torch.equal(single.p_process[0], process[0])
    assert torch.equal(single.rho[0], presence[0])
    assert torch.equal(single.tau[0], tau[0])
    assert torch.equal(single.sigma[0], sigma[0])

    first, _ = model._aggregate(
        p_lang=p_lang,
        local_scene=scene,
        local_process=process,
        local_presence=presence,
        local_tau=tau,
        local_sigma=sigma,
        video_set_offsets=torch.tensor([0, 2]),
    )
    second, _ = model._aggregate(
        p_lang=p_lang,
        local_scene=scene.flip(0),
        local_process=process.flip(0),
        local_presence=presence.flip(0),
        local_tau=tau.flip(0),
        local_sigma=sigma.flip(0),
        video_set_offsets=torch.tensor([0, 2]),
    )
    for name in ("p_scene", "p_process", "rho", "tau", "sigma"):
        assert torch.allclose(getattr(first, name), getattr(second, name), atol=1e-6)


def test_alignment_is_monotone_and_two_probe_forward_has_gradients() -> None:
    torch.manual_seed(4)
    model = _model()
    process = torch.randn(2, 4, 38, 8)
    presence = torch.rand(2, 4)
    tau = torch.stack(
        (
            torch.linspace(0.0, 1.0, 4).expand(2, -1),
            torch.full((2, 4), 0.1),
        ),
        dim=-1,
    )
    alignment = model.aligner(process, presence, tau)
    expected = torch.einsum(
        "vlc,c->vl",
        alignment.transpose(1, 2),
        torch.arange(4, dtype=alignment.dtype),
    )
    assert torch.all(expected[:, 1:] >= expected[:, :-1] - 1e-5)

    frames = torch.randint(0, 256, (5, 3, 8, 8), dtype=torch.uint8)
    output = model(
        policy=SimpleNamespace(),
        frames=frames,
        frame_indices=torch.tensor([0, 5, 10, 0, 9]),
        raw_frame_counts=torch.tensor([11, 10]),
        video_offsets=torch.tensor([0, 3, 5]),
        video_set_offsets=torch.tensor([0, 2]),
        frame_condition_ids=torch.zeros(5, dtype=torch.long),
        language_tokens=torch.randint(0, 64, (1, 6)),
        language_mask=torch.tensor([[True, True, True, True, False, False]]),
        query_times=torch.linspace(0.0, 1.0, 6)[None],
    )
    loss = output.program.p_process.square().mean() + output.predictions.action_phases.square().mean()
    loss.backward()

    assert output.program.p_lang.shape == (1, 38, 8)
    assert output.program.p_scene.shape == (1, 38, 8)
    assert output.program.p_process.shape == (1, 4, 38, 8)
    assert output.program.rho.shape == (1, 4)
    assert output.program.tau.shape == (1, 4, 2)
    assert output.program.sigma.shape == (1, 4, 38, 8)
    assert output.probe_process.shape == (2, 2, 4, 38, 8)
    assert output.predictions.action_phases.shape == (1, 6, 5, 7)
    assert output.predictions.scene_predicate_logits.shape == (1, 2, 3)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_temporal_heads_have_no_direct_language_or_scene_bypass() -> None:
    torch.manual_seed(5)
    model = _model()
    process = torch.randn(1, 4, 38, 8)
    rho = torch.rand(1, 4)
    tau = torch.stack(
        (
            torch.linspace(0.0, 1.0, 4)[None],
            torch.full((1, 4), 0.1),
        ),
        dim=-1,
    )
    sigma = torch.rand_like(process)
    first = NaturalProgram(
        p_lang=torch.randn(1, 38, 8),
        p_scene=torch.randn(1, 38, 8),
        p_process=process,
        rho=rho,
        tau=tau,
        sigma=sigma,
    )
    second = NaturalProgram(
        p_lang=torch.randn(1, 38, 8),
        p_scene=torch.randn(1, 38, 8),
        p_process=process,
        rho=rho,
        tau=tau,
        sigma=sigma,
    )
    query_times = torch.linspace(0.0, 1.0, 6)[None]
    first_prediction = model.decoder(first, query_times)
    second_prediction = model.decoder(second, query_times)

    for name in (
        "action_phases",
        "progress",
        "rising_logits",
        "contact_logits",
        "predicate_logits",
    ):
        assert torch.equal(
            getattr(first_prediction, name), getattr(second_prediction, name)
        )
    assert not torch.equal(
        first_prediction.scene_predicate_logits,
        second_prediction.scene_predicate_logits,
    )
    assert model.process_fusion[0].in_features == 2 * model.width


def test_temporal_readout_preserves_fixed_owner_identity() -> None:
    torch.manual_seed(6)
    model = _model()
    process = torch.randn(1, 4, 38, 8)
    rho = torch.rand(1, 4)
    tau = torch.stack(
        (
            torch.linspace(0.0, 1.0, 4)[None],
            torch.full((1, 4), 0.1),
        ),
        dim=-1,
    )
    common = {
        "p_lang": torch.randn(1, 38, 8),
        "p_scene": torch.randn(1, 38, 8),
        "rho": rho,
        "tau": tau,
        "sigma": torch.rand_like(process),
    }
    query_times = torch.linspace(0.0, 1.0, 6)[None]
    shared_query = torch.randn(8)
    shared_weights = torch.einsum(
        "cejd,d->cej", torch.tanh(process), shared_query
    ).softmax(-1)
    shared_event = torch.einsum("cej,cejd->ced", shared_weights, process)
    permuted_process = process.roll(1, dims=2)
    permuted_weights = torch.einsum(
        "cejd,d->cej", torch.tanh(permuted_process), shared_query
    ).softmax(-1)
    permuted_event = torch.einsum(
        "cej,cejd->ced", permuted_weights, permuted_process
    )
    assert torch.allclose(shared_event, permuted_event)

    assert torch.equal(
        model.decoder.owner_queries,
        model.decoder.owner_queries[:1].expand_as(model.decoder.owner_queries),
    )
    with torch.no_grad():
        model.decoder.owner_queries[0].add_(0.25)
    original = model.decoder(
        NaturalProgram(p_process=process, **common), query_times
    )
    permuted = model.decoder(
        NaturalProgram(p_process=permuted_process, **common), query_times
    )

    assert model.decoder.owner_queries.shape == (38, 8)
    assert not torch.equal(
        original.action_phases, permuted.action_phases
    )


def test_temporal_residual_loss_rejects_a_constant_mean_prediction() -> None:
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]]])
    constant = target.mean(1, keepdim=True).expand_as(target)
    shifted_match = target + 7.0

    assert _temporal_residual_mse(constant, target) > 0
    assert torch.equal(
        _temporal_residual_mse(shifted_match, target), torch.zeros(())
    )


def test_g2_inventory_requires_the_native_observer_to_remain_frozen() -> None:
    policy = torch.nn.Linear(2, 2).requires_grad_(False)
    model = _model().requires_grad_(True)
    model.encoder.requires_grad_(False).eval()

    inventory = _pure_native_inventory(policy, model)
    assert inventory["native_observer_trainable_parameter_count"] == 0
    assert inventory["native_observer_training"] is False
    assert inventory["natural_program_trainable_parameter_count"] > 0

    model.encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable frozen authority"):
        _pure_native_inventory(policy, model)

    model.encoder.requires_grad_(False).train()
    with pytest.raises(ValueError, match="trainable frozen authority"):
        _pure_native_inventory(policy, model)


def test_sparse_rising_windows_preserve_transitions() -> None:
    values = torch.tensor([0, 0, 1, 0, 0, 1, 0], dtype=torch.uint8).numpy()
    indices = torch.tensor([0, 3, 6]).numpy()
    assert _windowed_rising(values, indices).tolist() == [0.0, 1.0, 1.0]


def test_predicate_rising_includes_initial_state_boundary() -> None:
    predicates = torch.tensor(
        [[1, 0], [1, 1], [1, 1]], dtype=torch.uint8
    ).numpy()
    initial = torch.tensor([0, 0], dtype=torch.uint8).numpy()
    assert _predicate_rising(
        predicates, initial, goal_count=2
    ).tolist() == [1, 1, 0]


def test_schedule_contrastive_negatives_are_fixed_fit_only() -> None:
    roles = ["meta_fit"] * 56 + ["target_fit"] * 19
    roles += ["meta_held"] * 15 + ["target_held"] * 5
    tasks = tuple(
        SimpleNamespace(
            authority_id=index,
            role=role,
            episode_lengths=(100,) * 50,
        )
        for index, role in enumerate(roles)
    )
    schedule = NaturalProgramSchedule(tasks, seed=7, query_points=4)
    negatives = schedule.contrastive_task_ids(0, 3, count=8)
    assert len(negatives) == len(set(negatives)) == 8
    assert 0 not in negatives
    assert all(tasks[index].role in {"meta_fit", "target_fit"} for index in negatives)
    assert sum(tasks[index].role == "meta_fit" for index in negatives) == 4
    assert sum(tasks[index].role == "target_fit" for index in negatives) == 4
    expected = set(schedule.training_task_ids(3))
    optimizer_groups = schedule.optimizer_task_groups(3, tasks_per_role=2)
    assert len(optimizer_groups) == 10
    assert [len(group) for group in optimizer_groups] == [4] * 9 + [2]
    assert len({task for group in optimizer_groups for task in group}) == 38
    for group in optimizer_groups:
        assert sum(tasks[index].role == "meta_fit" for index in group) * 2 == len(
            group
        )
        assert sum(tasks[index].role == "target_fit" for index in group) * 2 == len(
            group
        )
    rotating_target_tails = {
        schedule.optimizer_task_groups(macro, tasks_per_role=2)[-1][0]
        for macro in range(19)
    }
    assert rotating_target_tails == set(schedule.target_fit)
    for world_size in range(1, 7):
        assigned = {
            task_id
            for group in schedule.assignments(3, world_size)
            for task_id in group
        }
        assert assigned == expected
        optimizer_assignments = schedule.optimizer_assignments(
            3, world_size, tasks_per_role=2
        )
        assert tuple(
            task
            for assignments in optimizer_assignments
            for rank in assignments
            for task in rank
        ) != ()
        for expected_group, assignments in zip(
            optimizer_groups, optimizer_assignments, strict=True
        ):
            assert {
                task for rank in assignments for task in rank
            } == set(expected_group)


def test_scheduler_uses_optimizer_step_cursor() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    config = {
        "optimization": {
            "scheduler": {"peak_lr": 1e-4, "decay_lr": 1e-6}
        }
    }
    scheduler = _scheduler(
        optimizer,
        config,
        total_optimizer_steps=600,
        warmup_optimizer_steps=30,
    )
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 / 30)
    for _ in range(30):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 30
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    for _ in range(570):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 600
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)


def test_cross_episode_supervision_uses_one_action_index_grid() -> None:
    lengths = tuple([9, 10, 7, 11] + [12] * 46)
    task = SimpleNamespace(authority_id=4, episode_lengths=lengths)
    sample = NaturalProgramSample(
        video_demos=(0, 1),
        action_demos=(2, 3),
        k=2,
        robustness_view="speed2",
    )

    class ActionStore:
        indices = None

        def phase_targets(self, **kwargs):
            self.indices = kwargs["frame_indices"].clone()
            return torch.zeros(self.indices.numel(), 2, 7)

    class LabelStore:
        def load(self, _task_id, demo):
            size = lengths[demo]
            values = torch.arange(size, dtype=torch.float32).numpy()
            return SimpleNamespace(
                progress=values,
                rising=(values == 2).astype("uint8"),
                contact=(values % 2).astype("uint8"),
                contact_mask=torch.ones(size, dtype=torch.uint8).numpy(),
                predicates=torch.stack(
                    (torch.arange(size) > 0, torch.arange(size) > 3), dim=-1
                ).numpy(),
                predicate_mask=torch.tensor([True, True]).numpy(),
            )

    action_store = ActionStore()
    packed = _pack_cross_episode_supervision(
        task=task,
        sample=sample,
        action_store=action_store,
        label_store=LabelStore(),
        query_points=4,
        predicate_slots=2,
        device=torch.device("cpu"),
    )
    expected_rows = [_query_indices(lengths[index], 4) for index in (2, 3)]
    expected = torch.from_numpy(np.concatenate(expected_rows))
    assert torch.equal(action_store.indices, expected)
    expected_progress = torch.tensor(
        [
            sum(expected_rows[video][query] for video in range(2)) / 2
            for query in range(4)
        ]
    )
    assert torch.equal(packed.progress_targets[0], expected_progress)


def test_g2_gate_uses_cross_task_margins_and_all_contract_checks() -> None:
    records = []
    for task in range(20):
        anchor = torch.zeros(20)
        anchor[task] = 1.0
        records.append(
            {
                "authority_id": task,
                "domain": "meta" if task < 15 else "target",
                "domain_task_id": task,
                "role": "meta_held" if task < 15 else "target_held",
                "embedding_a": anchor,
                "embedding_b": anchor + 0.001,
                "probe_delta_a": 0.001,
                "probe_delta_b": 0.001,
                "active_events": [3, 3],
                "one_event_rows": 0,
                "full_losses": [
                    {"action": 0.4, "progress": 0.4, "combined": 0.8},
                    {"action": 0.4, "progress": 0.4, "combined": 0.8},
                ],
                "endpoint_losses": [
                    {"action": 0.5, "progress": 0.5, "combined": 1.0},
                    {"action": 0.5, "progress": 0.5, "combined": 1.0},
                ],
                "K1_exact_identity": True,
                "K4_permutation_max_abs": 0.0,
                "K4_permutation_invariant": True,
                "mean_sigma": 0.1,
                "tau_order_violation_fraction": 0.0,
            }
        )
    report = _build_report(
        records,
        macro=10,
        thresholds={
            "same_task_nearer_fraction": 0.9,
            "probe_delta_below_half_cross_margin_fraction": 0.75,
            "maximum_one_event_fraction": 0.25,
            "median_active_events_min": 2,
            "median_active_events_max": 6,
            "full_vs_endpoints_action_progress_improvement": 0.1,
            "K1_exact_identity": True,
            "K_permutation_invariance": True,
            "shuffled_or_reversed_use": False,
        },
    )
    assert report["passed"]
    assert report["metrics"]["same_task_nearer_fraction"] == 1.0
    assert report["metrics"]["full_vs_endpoints_action_progress_improvement"] > 0.1
