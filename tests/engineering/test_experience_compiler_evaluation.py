"""CPU queue/panel checks; fake episodes exercise bookkeeping, never science."""
from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from ember.experience_compiler import evaluation as ev
from ember.pi05_eval_queue import (claim_next, complete_job, completed_jobs, fail_job,
                                  publish_json_exclusive, queue_summary)


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
    checkpoint = make_checkpoint(tmp_path / "checkpoint", 180)
    output = tmp_path / "train180"
    ev.prepare(output, checkpoint, "train180")
    return output, checkpoint


def make_checkpoint(path, update):
    path.mkdir()
    publish_json_exclusive(path / 'checkpoint_manifest.json', {
        'stage': ev.STAGE, 'next_macro': update, 'run_contract_schema': ev.SCHEMA})
    return path


def row(condition, state, arm):
    return {"suite": condition["suite"], "task_id": condition["suite_task_id"], "init_state_id": state,
            "condition_id": condition["condition_id"], "teacher_demo": condition["teacher_demo"], "arm": arm,
            "language": f"task {condition['task_id']}", "success": arm == "end", "env_seed": 7,
            "policy_seed_root": 7, "policy_noise_seeds": [10, 20],
            "split_role": "validation" if "video_ordinal" in condition else "train",
            **({"video_ordinal": condition["video_ordinal"]} if "video_ordinal" in condition else {})}


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
                       weights={arm: "canonical_MT_reference" if arm == "MT" else f"{arm}.safetensors"
                                for arm in contract["arms"]}, chain="experience.pt")
        path = f"shards/{claim.shard.job_id}.json"
        size = publish_json_exclusive(output / path, primary)
        complete_job(output / "queue.sqlite3", job_id=claim.shard.job_id, worker_id="cpu-test",
                     claim_token=claim.claim_token, rows_path=path, rows_bytes=size,
                     row_count=len(primary["rows"]), successes=sum(r["success"] for r in primary["rows"]))


def test_prepare_is_immutable_long_first_and_counts_conditions(prepared):
    output, checkpoint = prepared
    assert ev.prepare(output, checkpoint, "train180") == output / "evaluation_contract.json"
    contract = ev.read_json(output / "evaluation_contract.json")
    assert len(contract["conditions"]) == 16 and contract["expected_rows_per_arm"] == 48
    assert contract["arms"] == ["MT", "end", "null"]
    assert contract["environment_contract"]["parallel"]["envs_per_replica"] == 1
    assert "operator_read_write_scene" in contract["environment_contract"]
    assert "operator_read_write_scene" not in contract["adaptation_environment_contract"]
    claim = claim_next(output / "queue.sqlite3", worker_id="cpu-test")
    assert claim.shard.suite == "libero_10" and len(claim.shard.init_state_ids) == 3
    with pytest.raises(ValueError, match="checkpoint"):
        ev.prepare(output, checkpoint, "train360")


@pytest.mark.parametrize("stage,update", [("formal180", 180), ("formal360", 360)])
def test_formal400_freezes_without_replacement(tmp_path, monkeypatch, stage, update):
    conditions, environment = panel(True)
    monkeypatch.setattr(ev, "formal400_mapping", lambda **kwargs: conditions)
    monkeypatch.setattr(ev, "formal_environment", lambda **kwargs: environment)
    checkpoint = make_checkpoint(tmp_path / "checkpoint", update)
    path = ev.prepare(tmp_path / stage, checkpoint, stage)
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
    assert result["comparisons"]["MT_to_end"]["gained"] == 48
    assert result["comparisons"]["null_to_end"]["lost"] == 0
    assert result["comparisons"]["MT_to_end"]["candidate_breadth"] == 8
    assert ev.aggregate(output) == result


def test_train360_reuses_complete180_MT_and_compares_adjacent_readouts(prepared):
    output, _ = prepared
    finish_all(output)
    previous = ev.aggregate(output)
    checkpoint = make_checkpoint(output.parent / "checkpoint360", 360)
    next_output = output.parent / "train360"
    contract = ev.read_json(ev.prepare(next_output, checkpoint, "train360"))
    assert contract["arms"] == ["end", "null"]
    assert contract["MT_reuse"] == str(output / "results.json")
    finish_all(next_output)
    result = ev.aggregate(next_output)
    assert result["arms"]["MT"]["rows"] == previous["arms"]["MT"]["rows"]
    assert result["comparisons"]["MT_to_end"]["gained"] == 48
    assert result["comparisons"]["180_to_360"]["retained"] == 48


@pytest.fixture
def consumer(prepared, monkeypatch):
    """Toy CPU factors/experience exercise only _Cases persistence and scheduling."""
    from ember.experience_compiler.interaction import Chain
    from ember.experience_compiler import run

    monkeypatch.setattr(run, "check_budget", lambda root: None)
    output, checkpoint = prepared
    contract = ev.read_json(output / "evaluation_contract.json")
    states = [{"a": torch.tensor([value])} for value in (0., 1., 2.)]
    chain = Chain(states=states, records=[{"episode": 0, "feedback": torch.zeros(3)},
                                         {"episode": 0, "feedback": torch.tensor([1., 1., 1.])}],
                  endpoints=[1, 2], episodes=[{"success": True}],
                  metrics={"actual_J": 2, "wall_seconds": 1., "practice_success": True, "reads": 2},
                  observations={"cpu-observation": {"images": torch.zeros(1, dtype=torch.uint8)}},
                  behavior_versions=["train180", "train180"])
    saves, replay_calls, future = [], [], Future()
    def submit(save):
        saves.append(save)
        return future
    null = {"a": torch.tensor([-1.])}
    def replay(teacher, actual):
        replay_calls.append(actual)
        return null
    runtime = SimpleNamespace(mt=states[0], teacher=lambda *args: {"indices": torch.arange(2)},
        null_replay=replay, last_teacher_cost={"frames": 2}, io=SimpleNamespace(submit=submit))
    cases = ev._Cases(runtime, contract, output, "cpu-test", 0, "cpu-consumer", output.parent)
    request = cases.next_request()
    assert request["kind"] == "adapt"
    assert "operator_read_write_scene" not in request["environment_contract"]
    cases.adapted({"request": request, "chain": chain})
    return SimpleNamespace(cases=cases, chain=chain, request=request, future=future, saves=saves,
                           runtime=runtime, replay_calls=replay_calls, null=null,
                           output=output, checkpoint=checkpoint, contract=contract)


def test_cases_freeze_success_chain_last_edit_and_wait_for_originals(consumer):
    from safetensors.torch import load_file

    cases, request = consumer.cases, consumer.request
    c = request["condition"]
    assert consumer.replay_calls == [consumer.chain]
    assert len(cases.finals) == 9
    for _ in range(9):
        final = cases.next_request()
        assert final["kind"] == "final" and final["noise_root"] == 7
        assert final["environment_contract"] == consumer.contract["environment_contract"]
        expected = {"MT": consumer.runtime.mt, "end": consumer.chain.states[-1], "null": consumer.null}
        assert torch.equal(final["state"]["a"], expected[final["arm"]]["a"])
        cases.final({"request": final, "row": row(c, final["state_id"], final["arm"])})
    cases.publish_ready()
    assert queue_summary(consumer.output / "queue.sqlite3")["completed_rows"] == 0
    record = consumer.saves.pop()()
    consumer.future.set_result(record)
    cases.publish_ready(flush=True)
    job = completed_jobs(consumer.output / "queue.sqlite3")[0]
    primary = ev.read_json(consumer.output / job["rows_path"])
    destination = consumer.output / primary["artifact_root"]
    assert primary["actual_J"] == 2 and primary["chain"] == "experience.pt"
    assert primary["weights"]["MT"] == "canonical_MT_reference"
    assert record["metrics"]["practice_success"] and record["actual_incoming_parameters"]
    assert [e["incoming"] for e in record["events"]] == ["MT", "incoming_001.safetensors"]
    assert torch.equal(load_file(str(destination / "end.safetensors"))["a"], consumer.chain.states[-1]["a"])
    assert torch.equal(load_file(str(destination / "incoming_001.safetensors"))["a"], consumer.chain.states[1]["a"])
    actual = torch.load(destination / "experience.pt", map_location="cpu", weights_only=False)
    assert actual["endpoints"] == [1, 2] and actual["episodes"][0]["success"]


def test_cases_recovery_reuses_actual_adaptation_and_negative_final_rows(consumer):
    cases, request = consumer.cases, consumer.request
    consumer.future.set_result(consumer.saves.pop()())
    case = cases.cases[request["case_id"]]
    negative_request = next(r for r in cases.finals if r["arm"] == "end" and r["state_id"] == 32)
    negative = row(request["condition"], 32, "end")
    negative["success"] = False
    cases.final({"request": negative_request, "row": negative})
    record_path = case["destination"] / "record.json"
    negative_path = case["destination"] / "final_end_32.json"
    old_negative = negative_path.read_text()
    claim = case["claim"]
    fail_job(consumer.output / "queue.sqlite3", job_id=claim.shard.job_id, worker_id="cpu-test",
             claim_token=claim.claim_token, error="CPU consumer interruption")
    ev.prepare(consumer.output, consumer.checkpoint, "train180", recover_claims=True, retry_failed=True)
    def forbidden(*args):
        pytest.fail("recovery must reuse the saved adaptation and null factors")
    runtime = SimpleNamespace(mt=consumer.runtime.mt, teacher=forbidden, null_replay=forbidden,
                              io=SimpleNamespace(submit=forbidden))
    recovered = ev._Cases(runtime, consumer.contract, consumer.output, "cpu-recovery", 0,
                          "cpu-recovery", consumer.output.parent)
    first = recovered.next_request()
    assert first["kind"] == "final" and first["case_id"] == request["case_id"]
    restored = recovered.cases[first["case_id"]]
    assert restored["actual_J"] == 2 and restored["recovered_from"] == str(record_path)
    finals = [first, *recovered.finals]
    recovered.finals.clear()
    assert len(finals) == 8
    assert not any(r["arm"] == "end" and r["state_id"] == 32 for r in finals)
    for final in finals:
        recovered.final({"request": final, "row": row(request["condition"], final["state_id"], final["arm"])})
    recovered.publish_ready(flush=True)
    primary = ev.read_json(consumer.output / completed_jobs(consumer.output / "queue.sqlite3")[0]["rows_path"])
    assert primary["recovered_from"] == str(record_path) and primary["actual_J"] == 2
    assert next(r for r in primary["rows"] if r["init_state_id"] == 32)["success"] is False
    assert negative_path.read_text() == old_negative


def test_pairing_rejects_changed_noise_and_gpu_cost_unions(prepared):
    output, _ = prepared
    c = ev.read_json(output / "evaluation_contract.json")["conditions"][0]
    before, after = row(c, 32, "MT"), row(c, 32, "end")
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
