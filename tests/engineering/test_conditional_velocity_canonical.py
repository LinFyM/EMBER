"""CPU contract checks only: no policy/model forward, data tensors or environment."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.eval_adapters import validate_episode_adapter_fields
from ember.pi05_eval_contract import git_state
from ember.pi05_eval.registered_passive_capture import (
    attach_requested_capture as attach_registered_capture,
    validate_contract as validate_registered_capture,
)
from ember.lora import identity_lora_state, validate_lora_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.source_sft.control import clamped_lr_multiplier
from ember.writer.conditional_velocity import (ConditionalVelocityOperator,
                                               VelocityTeachingEncoder, compile_velocity_state)
from ember.writer.conditional_velocity_bank import (_expected_episodes, canonical_selection,
                                                    episode_evidence, registered_capture,
                                                    PASSIVE_TAG, BANK_KIND)
from ember.writer.conditional_velocity_data import VelocityEvents
from ember.writer.conditional_velocity_training import VelocityRuntime, _lr_multiplier, _resume_prefix
from ember.writer.runtime import MODEL_DEFAULTS


ROOT = Path(__file__).resolve().parents[2]


def test_nonzero_common_operator_is_preserved_in_exact_single_rank135_state(tmp_path):
    base = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    common = identity_lora_state(derive_pi05_lora_rank(base, rank=128))
    for name in common:
        if name.endswith("lora_B.default.weight"):
            common[name].fill_(0.01)
    R, U = torch.randn(7, 256), torch.randn(256, 1024)
    compiled = compile_velocity_state(common, R, U)
    validate_lora_state(compiled, derive_pi05_lora_rank(base, rank=135))
    for name in common:
        if not name.endswith("lora_A.default.weight"):
            continue
        prefix = name.removesuffix("lora_A.default.weight")
        b_name = prefix + "lora_B.default.weight"
        expected = common[b_name] @ common[name]
        if prefix == "model.action_out_proj.":
            expected = expected.clone()
            expected[:7] += R @ U
        assert torch.allclose(compiled[b_name] @ compiled[name], expected, atol=1e-4, rtol=1e-5)
    shared_path, coefficient_path = tmp_path / "shared.safetensors", tmp_path / "R.safetensors"
    save_file({**common, "velocity_U": U}, str(shared_path))
    save_file({"R": R}, str(coefficient_path))
    restored = load_file(str(shared_path))
    U_restored = restored.pop("velocity_U")
    rebuilt = compile_velocity_state(restored, load_file(str(coefficient_path))["R"], U_restored)
    assert all(torch.equal(compiled[key], rebuilt[key]) for key in compiled)


def _policy_stub():
    def layers():
        attention = SimpleNamespace(**{name: torch.nn.Linear(4, 4, bias=False)
                                       for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
        return [SimpleNamespace(self_attn=attention)]
    bridge = SimpleNamespace(paligemma=SimpleNamespace(model=SimpleNamespace(
        language_model=SimpleNamespace(layers=layers()))),
        gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=layers())))
    return SimpleNamespace(model=SimpleNamespace(paligemma_with_expert=bridge))


def test_fresh_modes_share_common_readout_u_text_core_and_exclude_l_unused_parameters():
    policy = _policy_stub()
    template = {"test": torch.ones(1)}
    torch.manual_seed(7)
    v = ConditionalVelocityOperator(policy, MODEL_DEFAULTS, template, mode="V")
    torch.manual_seed(7)
    l = ConditionalVelocityOperator(policy, MODEL_DEFAULTS, template, mode="L")
    for name in ("common", "readout", "U"):
        assert all(torch.equal(left, right) for left, right in zip(
            getattr(v, name).state_dict().values(), getattr(l, name).state_dict().values(), strict=True))
    for name in ("language_projection", "text_meta_lora"):
        assert all(torch.equal(left, right) for left, right in zip(
            getattr(v.teaching.semantic_encoder, name).state_dict().values(),
            getattr(l.teaching.semantic_encoder, name).state_dict().values(), strict=True))
    assert all(torch.equal(left, right) for left, right in zip(
        v.teaching.semantic_core.state_dict().values(),
        l.teaching.semantic_core.state_dict().values(), strict=True))
    assert l.teaching.procedure is None
    assert not any(p.requires_grad for p in l.teaching.semantic_encoder.vl_meta_lora.parameters())
    assert not any(p.requires_grad for p in l.teaching.semantic_encoder.action_meta_lora.parameters())
    assert all(p.requires_grad for p in l.teaching.semantic_encoder.text_meta_lora.parameters())
    no_video = SimpleNamespace(load=lambda *_: (_ for _ in ()).throw(AssertionError("video read")))
    language_runtime = VelocityRuntime(None, l, lambda _: ("tokens", "mask", "span"),
                                       None, None, {}, torch.device("cpu"), "L")
    condition, raw, sampled = language_runtime.condition(no_video, 3, 7, "exact language")
    assert condition == ("tokens", "mask", "span") and raw is None and sampled == 0


def test_language_reads_same_core_memory_twice_with_native_token_positions():
    text = torch.randn(1, 3, 256)
    core = text + 1
    valid = torch.tensor([[True, True, False]])
    owner = SimpleNamespace(
        mode="L", procedure=None,
        semantic_encoder=SimpleNamespace(encode_text_only=lambda *_: (text, valid)),
        semantic_core=SimpleNamespace(language_only=lambda values, mask: core),
    )
    first, first_mask, second, positions, second_mask = VelocityTeachingEncoder.encode_language(
        owner, None, None, None, None,
    )
    assert first is second is core
    assert first_mask is second_mask is valid
    assert positions.tolist() == [[1, 2, 3]]


def test_coverage36_events_teacher_query_and_resume_identity(tmp_path):
    manifest = json.loads((ROOT / "configs/libero_24_8_8_coverage_v1/manifest.json").read_text())
    spec = json.loads((ROOT / "configs/conditional_velocity_operator_v1/learning_spec.json").read_text())
    lengths = {row["global_task_id"]: tuple(row["demonstrations"]["episode_lengths"])
               for row in manifest["tasks"] if row["global_task_id"] in spec["train_tasks"]}
    events = VelocityEvents(lengths)
    assert [(job["task"], job["teacher_demo"]) for job in events.event(0)] == [
        (32, 18), (20, 37), (96, 25), (35, 49)]
    seen = {task: [] for task in spec["train_tasks"]}
    longest_selected = 0
    for update in range(270):
        jobs = events.event(update)
        assert len(jobs) == 4 and len({job["task"] for job in jobs}) == 4
        for job in jobs:
            raw = lengths[job["task"]][job["teacher_demo"]]
            longest_selected = max(longest_selected, len(range(0, raw, 5)) + int((raw - 1) % 5 != 0))
            seen[job["task"]].append(job["teacher_demo"])
            assert len(job["action_demos"]) == len(set(job["action_demos"])) == 28
            assert job["teacher_demo"] not in job["action_demos"]
            assert all(start == frame + 1 and start < lengths[job["task"]][demo]
                       for demo, frame, start in zip(job["action_demos"], job["action_frames"],
                                                      job["action_start_indices"], strict=True))
    assert all(len(demos) == 30 and len(set(demos)) == 30 for demos in seen.values())
    assert longest_selected == 102
    events.next_step = 180
    restored = VelocityEvents(lengths)
    restored.restore(events.sampler_state())
    assert restored.event(180) == events.event(180)
    with pytest.raises(ValueError):
        restored.restore({**events.sampler_state(), "seed": 0})
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"update": n}) + "\n" for n in range(1, 5)))
    assert len(_resume_prefix(tmp_path, 2)) == 2
    assert len(path.read_text().splitlines()) == 4


def test_official_pairing_and_clock(tmp_path):
    spec = json.loads((ROOT / "configs/conditional_velocity_operator_v1/learning_spec.json").read_text())
    selection = canonical_selection(spec)
    assert len(spec["validation_tasks"]) == 8
    capture = json.loads((ROOT / "configs/conditional_velocity_operator_v1/official_capture.json").read_text())
    assert capture["mode"] == "compact"
    assert len(capture["full_conditions"]) == 8
    assert all(row["init_state_id"] == 0 for row in capture["full_conditions"])
    authority = json.loads((ROOT / "configs/libero_24_8_8_coverage_v1/manifest.json").read_text())
    validation = [(row["suite"], row["task_id"], 0) for row in authority["tasks"]
                  if row["global_task_id"] in spec["validation_tasks"]]
    assert [(row["suite"], row["task_id"], row["init_state_id"])
            for row in capture["full_conditions"]] == validation
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({"kind": BANK_KIND, "mode": "V"}))
    tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"],
                             init_state_ids=tuple(range(50))) for row in capture["full_conditions"]]
    selection_path = ROOT / "configs/conditional_velocity_operator_v1/official_capture.json"
    output_dir = Path(spec["run_root"]) / "V/evaluation/correct400"
    captured, stage = registered_capture(
        SimpleNamespace(role="validation", static_task_lora_manifest=bank_path),
        tasks, output_dir, selection_path,
        capture, None, ROOT, capture["schema_version"],
    )
    assert captured["passive_trace"]["schema_version"] == PASSIVE_TAG
    assert stage["full_conditions_only"] is False
    eval_contract = {"role": "validation", "output_dir": str(output_dir),
                     "tasks": [vars(task) for task in tasks],
                     "adapter": {"kind": BANK_KIND,
                                 "manifest": {"path": str(bank_path), "bytes": bank_path.stat().st_size},
                                 "checkpoint": {}},
                     "git": {"commit": git_state(ROOT)["commit"]},
                     "diagnostic_occupancy_capture": captured,
                     "diagnostic_stage_predicates": stage}
    attach_registered_capture(SimpleNamespace(), eval_contract, ROOT, output_dir)
    validate_registered_capture(eval_contract, ROOT)
    for task in spec["validation_tasks"]:
        v = _expected_episodes(selection, task, "V")
        l = _expected_episodes(selection, task, "L")
        assert len(v) == len(l) == 50
        assert len({row["teacher_demo_indices"][0] for row in v}) == 50
        assert [row["paired_correct_demos"] for row in v] == [row["paired_correct_demos"] for row in l]
        assert len({row["condition_id"] for row in v}) == 50
        assert len({row["condition_id"] for row in l}) == 1
        assert all("teacher_demo_indices" not in row for row in l)
    first = _expected_episodes(selection, 3, "V")[0]
    task_row = {"suite": "libero_spatial", "task_id": 3, "global_task_id": 3,
                "episodes": [first]}
    adapter = {"kind": BANK_KIND, "mode": "V", "selection": selection,
               "checkpoint": {}, "shared": {}, "tasks": [task_row],
               "conditions": [{"condition_id": first["condition_id"], "coefficient": {}}]}
    evidence = episode_evidence(adapter, task_row, first)
    assert validate_episode_adapter_fields(adapter, {"conditional_velocity_lora": evidence},
                                           suite="libero_spatial", task_id=3,
                                           init_state_id=first["init_state_id"])
    setting = spec["optimizer"]
    assert setting["decay_updates"] == 1200 and "cosine_updates" not in setting
    for step in (0, 149, 150, 675, 1200, 1201, 1349):
        assert _lr_multiplier(step, setting) == clamped_lr_multiplier(
            step, warmup=150, decay=1200, peak=3e-4, floor=1e-5,
        )
    assert _lr_multiplier(149, setting) == pytest.approx(150 / 151)
    assert _lr_multiplier(150, setting) == 1.0
    assert _lr_multiplier(1200, setting) == pytest.approx(1 / 30)
