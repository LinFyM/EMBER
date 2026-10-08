"""CPU queue/panel checks; fake episodes exercise bookkeeping, never science."""
from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.experience_compiler import evaluation as ev
from ember.pi05_eval_queue import claim_next, complete_job, publish_json_exclusive, queue_summary


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520)))


def panel(formal=False):
    result, tasks = [], []
    for suite_index, suite in enumerate(SUITES):
        for local in range(2):
            global_id = suite_index * 10 + local
            tasks.append({"suite": suite, "task_id": local, "language": f"task {global_id}",
                          "split_role": "validation" if formal else "train"})
            for demo in range(50 if formal else 2):
                c = {"condition_id": f"task{global_id}_demo{demo}", "task_id": global_id,
                     "suite": suite, "suite_task_id": local, "teacher_demo": demo,
                     "final_state_ids": [demo] if formal else [32, 33, 34], "seed": global_id * 50 + demo}
                if formal:
                    c.update(video_ordinal=demo, init_state_id=demo)
                result.append(c)
    return result, {"environment": {"horizons": HORIZONS}, "tasks": tasks,
                    "libero_paths": {}, "policy": {}, "rng": {}, "role": "validation" if formal else "train"}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    conditions, environment = panel()
    monkeypatch.setattr(ev, "panel_contract", lambda **kwargs: {"conditions": conditions})
    monkeypatch.setattr(ev, "learning_environment", lambda **kwargs: environment)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    publish_json_exclusive(checkpoint / 'checkpoint_manifest.json', {'stage': ev.STAGE, 'next_macro': 182})
    output = tmp_path / "meta54"
    ev.prepare(output, checkpoint, "meta54")
    return output, checkpoint


def row(condition, state, arm):
    return {"suite": condition["suite"], "task_id": condition["suite_task_id"], "init_state_id": state,
            "condition_id": condition["condition_id"], "teacher_demo": condition["teacher_demo"], "arm": arm,
            "language": f"task {condition['task_id']}", "success": arm == "end", "env_seed": 7,
            "policy_seed_root": 7, "policy_noise_seeds": [10, 20], "split_role": "train"}


def finish_all(output):
    contract = ev.read_json(output / "evaluation_contract.json")
    by_id = {f"{contract['stage']}-{c['condition_id']}": c for c in contract["conditions"]}
    while (claim := claim_next(output / "queue.sqlite3", worker_id="cpu-test")) is not None:
        c = by_id[claim.shard.job_id]
        paths = {}
        for arm in contract["arms"]:
            path = f"shards/{claim.shard.job_id}-{arm}.json"
            publish_json_exclusive(output / path, {"condition": c, "arm": arm,
                "rows": [row(c, state, arm) for state in c["final_state_ids"]]})
            paths[arm] = path
        primary = ev.read_json(output / paths["end"])
        primary.update(arm_paths=paths, artifact_root="conditions/cpu-test", actual_J=1,
                       metrics={"reads": 2, "steps": 100}, wall_seconds=1.0,
                       weights={"end": "end.safetensors"}, chain="chain.pt.gz")
        path = f"shards/{claim.shard.job_id}.json"
        size = publish_json_exclusive(output / path, primary)
        complete_job(output / "queue.sqlite3", job_id=claim.shard.job_id, worker_id="cpu-test",
                     claim_token=claim.claim_token, rows_path=path, rows_bytes=size,
                     row_count=len(primary["rows"]), successes=sum(r["success"] for r in primary["rows"]))


def test_prepare_is_immutable_long_first_and_counts_conditions(prepared):
    output, checkpoint = prepared
    assert ev.prepare(output, checkpoint, "meta54") == output / "evaluation_contract.json"
    contract = ev.read_json(output / "evaluation_contract.json")
    assert len(contract["conditions"]) == 16 and contract["expected_rows_per_arm"] == 48
    assert contract["environment_contract"]["parallel"]["envs_per_replica"] == 3
    assert "operator_read_write_scene" in contract["environment_contract"]
    assert "operator_read_write_scene" not in contract["adaptation_environment_contract"]
    claim = claim_next(output / "queue.sqlite3", worker_id="cpu-test")
    assert claim.shard.suite == "libero_10" and len(claim.shard.init_state_ids) == 3
    with pytest.raises(ValueError, match="checkpoint"):
        ev.prepare(output, checkpoint, "meta27")


def test_formal400_freezes_without_replacement(tmp_path, monkeypatch):
    conditions, environment = panel(True)
    monkeypatch.setattr(ev, "formal400_mapping", lambda **kwargs: conditions)
    monkeypatch.setattr(ev, "formal_environment", lambda **kwargs: environment)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    publish_json_exclusive(checkpoint / 'checkpoint_manifest.json', {'stage': ev.STAGE, 'next_macro': 182})
    path = ev.prepare(tmp_path / "formal", checkpoint, "formal")
    contract = ev.read_json(path)
    assert len(contract["conditions"]) == 400 and contract["arms"] == ["end"]
    contract["conditions"][1]["teacher_demo"] = 0
    with pytest.raises(ValueError, match="without replacement"):
        ev._validate_panel(contract)


def test_aggregate_arm_counts_and_paired_exchange(prepared):
    output, _ = prepared
    finish_all(output)
    result = ev.aggregate(output)
    assert queue_summary(output / "queue.sqlite3")["completed_rows"] == 48
    assert result["arms"]["end"]["row_count"] == 48
    assert result["arms"]["end"]["successes"] == 48 and result["arms"]["end"]["breadth"] == 8
    assert len(result["conditions"]) == 16
    assert result["comparisons"]["initial_to_end"]["gained"] == 48
    assert result["comparisons"]["null_to_end"]["lost"] == 0
    assert result["comparisons"]["initial_to_end"]["candidate_breadth"] == 8
    assert ev.aggregate(output) == result


def test_condition_executes_one_chain_then_fixed_three_state_readouts(prepared):
    output, _ = prepared
    contract = ev.read_json(output / "evaluation_contract.json")
    c = contract["conditions"][0]
    loras = [{"a": torch.zeros(1)}, {"a": torch.ones(1)}]
    calls = []
    chain = SimpleNamespace(states=loras, endpoints=[1], metrics={"reads": 2})
    chain.to_record = lambda: {"records": [{"images": torch.zeros(1, dtype=torch.uint8)}], "endpoints": [1]}
    runtime = SimpleNamespace(teacher=lambda *args, **kwargs: {"real_frames": True})
    runtime.replay = lambda teaching, actual, masked: (calls.append(("null", actual is chain, masked)) or [loras[0]])
    def adapt(task, teaching, condition_seed, excluded_states):
        assert "operator_read_write_scene" not in runner.contract
        calls.append(("adapt", condition_seed, excluded_states))
        return chain
    def final_many(task, states, lora, noise_root):
        assert runner.contract["operator_read_write_scene"] == contract["environment_contract"]["operator_read_write_scene"]
        calls.append(("final", id(lora), tuple(states), noise_root))
        arm = "end" if lora is loras[1] else "initial"
        return [row(c, state, arm) for state in states]
    runner = SimpleNamespace(adapt=adapt, final_many=final_many)
    destination = output / "condition-test"
    result = ev.execute_condition(runtime, runner, contract, c, destination)
    assert sum(call[0] == "adapt" for call in calls) == 1
    assert calls[0][2] == (32, 33, 34) and ("null", True, True) in calls
    assert all(len(rows) == 3 for rows in result["rows"].values())
    assert result["actual_J"] == 1 and len(set(result["weights"].values())) == 2
    with gzip.open(destination / "chain.pt.gz", "rb") as stream:
        assert torch.load(stream, map_location="cpu", weights_only=False)["endpoints"] == [1]


def test_pairing_rejects_changed_noise_and_gpu_cost_unions(prepared):
    output, _ = prepared
    c = ev.read_json(output / "evaluation_contract.json")["conditions"][0]
    before, after = row(c, 32, "initial"), row(c, 32, "end")
    after["policy_noise_seeds"] = [11, 20]
    with pytest.raises(ValueError, match="RNG pairing"):
        ev.compare_train([before], [after])
    for index, (gpu, start, stop) in enumerate(((0, 0, 100), (0, 50, 150), (1, 20, 80))):
        ev.immutable(output / "workers" / f"{index}.json", {"host": "test-host", "physical_gpu": gpu,
                     "started_unix": start, "finished_unix": stop})
    assert ev._cost(output, [])["allocated_physical_GPUh"] == pytest.approx(210 / 3600)


def test_formal_comparison_requires_scene_and_teaching_identity():
    c = panel(True)[0][0]
    before, after = row(c, 0, "end"), row(c, 0, "end")
    before["scene_reference"] = after["scene_reference"] = {"state": "canonical"}
    before["operator_read_write_lora"] = {"condition_id": c["condition_id"],
                                           "teacher_demo": c["teacher_demo"], "video_ordinal": 0}
    after["video_ordinal"] = 0
    assert ev.compare_formal({"rows": [before]}, {"rows": [after]})["retained"] == 1
    after["scene_reference"] = {"state": "different"}
    with pytest.raises(ValueError, match="scene reference"):
        ev.compare_formal({"rows": [before]}, {"rows": [after]})
    after["scene_reference"] = before["scene_reference"]
    after["teacher_demo"] += 1
    with pytest.raises(ValueError, match="teaching condition"):
        ev.compare_formal({"rows": [before]}, {"rows": [after]})
