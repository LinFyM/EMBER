"""CPU checks of queue dependencies, recovery, early stop and child ownership."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

import pytest


spec = importlib.util.spec_from_file_location("compiler_batch", Path(__file__).resolve().parents[2] / "scripts/experience_compiler_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def arguments(tmp_path):
    args = batch.parser().parse_args(["--code", str(tmp_path), "--root", str(tmp_path / "run"),
        "--python", sys.executable, "--local-node", "gpu01", "--train-node", "gpu01",
        "--train-gpus", "0,1", "--eval-node", "gpu02", "--eval-gpus", "0,1",
        "--minimum-free-mib", "40000"])
    args.train_minimum_free_mib = args.eval_minimum_free_mib = args.minimum_free_mib
    return args


def snapshots():
    return {node: [dict(index=gpu, uuid=f"{node}-{gpu}", name="NVIDIA A40", used=0,
        free=45000, utilization=0, processes=[], hostname=f"BCI-{node}") for gpu in range(8)]
        for node in ("gpu01", "gpu02")}


def write_checkpoint(runner, macro, world=2):
    path = runner.checkpoint(macro)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoint_manifest.json").write_text(json.dumps(dict(
        schema_version="ember_ecp_checkpoint_v1", stage=batch.STAGE,
        run_contract_schema=batch.SCHEMA, next_macro=macro, world_size=world)))
    return path


class QueueCLI(batch.Supervisor):
    """Mock only the CLI boundary; production scheduling and metadata stay real."""
    def __init__(self, args, mode="pass"):
        super().__init__(args)
        self.mode, self.queues, self.timeline = mode, {}, []
        self.resumed, self.chi360, self.tail, self.stop_notice = (asyncio.Event() for _ in range(4))
        self.formal_done = asyncio.Event()
        self.formal_claims = 0

    def counts(self, name):
        return {k: v for k, v in self.queues[name].items() if v}

    def check_stop(self):
        super().check_stop()
        if self.stopped:
            self.stop_notice.set()

    def emit_result(self, name):
        spec = self.spec(name)
        row = dict(schema_version=batch.SCHEMA, stage=name, checkpoint=str(self.checkpoint(spec["point"])),
                   aggregated_complete=True)
        if spec["pool"]:
            row = dict(schema_version=batch.EVENT_SCHEMA, pool=name, version=f"{name}_20261009_v1",
                       complete=True, conditions=[dict(condition_id=i) for i in range(spec["count"])])
        elif name.startswith("train"):
            row["arms"] = {k: dict(successes=v) for k, v in dict(MT=25, end=20 if self.mode != "pass" else 30, null=24).items()}
            if name == "train360":
                row["MT_reference"] = str(self.spec("train180")["result"])
        else:
            row["arms"] = dict(end=dict(successes=135 if self.mode != "pass" else 160))
        spec["output"].mkdir(parents=True, exist_ok=True)
        spec["result"].write_text(json.dumps(row))

    async def train_cli(self, argv):
        stop = int(argv[argv.index("--stop-update") + 1])
        if stop == 360:
            assert self.result("refresh180") and self.result("train180")
            self.timeline.append(("resume360", self.counts("formal180")))
            self.resumed.set()
            if self.mode == "late_stop":
                await self.stop_notice.wait()
                point = 181
            else:
                point = 360
        else:
            point = stop
        checkpoint = write_checkpoint(self, point, len(self.train_cards))
        output = self.root / "training"
        (output / f"completion_{stop}.json").write_text(json.dumps(dict(updates=point,
            checkpoint=str(checkpoint), requested_stop_update=stop, stage_complete=point == stop,
            training_complete=point == 360, scientific_early_stop=point != stop)))
        if point == 360:
            self.chi360.set()

    async def queue_cli(self, name):
        queue = self.queues[name]
        count = min(100, queue["pending"]) if name == "formal180" else queue["pending"]
        queue["pending"] -= count
        queue["claimed"] += count
        if name == "formal180" and count:
            ordinal, self.formal_claims = self.formal_claims, self.formal_claims + 1
            if self.mode == "pass":
                await (self.tail.wait() if ordinal == 0 else self.chi360.wait() if ordinal == 1 else asyncio.sleep(0))
            elif self.mode == "late_stop":
                await self.resumed.wait()
        if name == "formal360":
            self.tail.set()
        if name == "train180" and self.mode == "early_stop":
            # Complete the formal arm first, so the registered non-pass cancels an unstarted segment.
            await self.formal_done.wait()
        await asyncio.sleep(0)
        queue["claimed"] -= count
        queue["complete"] += count
        if name == "formal180" and queue["complete"] == 400:
            self.formal_done.set()

    async def child(self, name, node, argv, targets=(), *, training=False):
        command = argv[argv.index(batch.MODULE) + 1]
        output = Path(argv[argv.index("--output") + 1])
        stage = output.name
        assert not self.leased.intersection(targets), "two different pools overlap a physical GPU"
        self.leased.update(targets)
        entry = dict(name=name, command=command, stage=stage, argv=argv,
                     physical_gpus=list(targets), started_unix=time.monotonic())
        self.record["processes"].append(entry)
        self.timeline.append(("start", command, stage, tuple(targets)))
        try:
            if command.endswith("prepare"):
                self.queues.setdefault(stage, dict(pending=self.spec(stage)["count"], claimed=0, complete=0))
            elif command.endswith("aggregate"):
                assert self.counts(stage) == {"complete": self.spec(stage)["count"]}
                self.emit_result(stage)
            elif command == "train":
                await self.train_cli(argv)
            else:
                await self.queue_cli(stage)
            return entry
        finally:
            self.leased.difference_update(targets)
            entry.update(status="complete", return_code=0, finished_unix=time.monotonic())
            self.timeline.append(("end", command, stage, tuple(targets)))


def test_full_batch_resumes_before_formal_tail_and_prioritizes_pending180(tmp_path):
    runner = QueueCLI(arguments(tmp_path))
    asyncio.run(asyncio.wait_for(runner.run(), 5))
    assert runner.record["status"] == "complete" and not runner.leased
    assert set(runner.results) == {"pool0", "refresh180", "train180", "formal180", "train360", "formal360"}
    assert next(t for t in runner.timeline if t[0] == "resume360")[1]["pending"] > 0
    processes = runner.record["processes"]
    train = [p for p in processes if p["command"] == "train"]
    assert len(train) == 2 and "--resume" not in train[0]["argv"]
    assert train[1]["argv"][train[1]["argv"].index("--resume") + 1] == str(runner.checkpoint(180))
    formal = [p for p in processes if p["stage"] == "formal180" and p["command"] == "worker"]
    later = [p for p in processes if p["stage"] in ("train360", "formal360") and p["command"] == "worker"]
    assert train[1]["started_unix"] < min(p["finished_unix"] for p in formal)
    assert min(p["started_unix"] for p in later) < max(p["finished_unix"] for p in formal)
    borrowed = [p for p in formal if p["started_unix"] > train[1]["finished_unix"]]
    assert borrowed and all(p["started_unix"] < min(q["started_unix"] for q in later) for p in borrowed)
    assert runner.results["train360"]["MT_reference"] == str(runner.spec("train180")["result"])
    assert len({tuple(p["physical_gpus"][0]) for p in processes if p["stage"] == "pool0" and p["command"] == "collect"}) == 4


@pytest.mark.parametrize("mode", ["early_stop", "late_stop"])
def test_registered_early_stop_skips_or_finishes_active360_without_qualification(tmp_path, mode):
    runner = QueueCLI(arguments(tmp_path), mode)
    asyncio.run(asyncio.wait_for(runner.run(), 5))
    marker = json.loads((runner.root / "early_stop_requested.json").read_text())
    assert marker["formal180"] == 135 and marker["train180"] == dict(MT=25, end=20, null=24)
    assert runner.record["status"] == "scientific_stopped" and runner.record["candidate"] is None
    train360 = [p for p in runner.record["processes"] if p["name"] == "train-360"]
    assert bool(train360) == (mode == "late_stop")
    assert not ({"train360", "formal360"} & runner.prepared) and not runner.leased
    assert runner.results["formal180"]["aggregated_complete"] is True
    if mode == "late_stop":
        assert runner.latest_checkpoint()[0] == 181
        assert json.loads((runner.root / "training/completion_360.json").read_text())["scientific_early_stop"]


@pytest.mark.parametrize("pending", [0, 7])
def test_resume_retains_receipts_and_reuses_complete_artifacts_and_partial_queue(tmp_path, pending):
    original = QueueCLI(arguments(tmp_path))
    asyncio.run(asyncio.wait_for(original.run(), 5))
    original.spec("formal360")["result"].unlink()
    receipt = original.root / "batch_execution.json"
    failed = json.loads(receipt.read_text())
    failed["status"] = "failed"
    receipt.write_text(json.dumps(failed))
    protected = {p: p.read_bytes() for p in (receipt, original.spec("pool0")["result"], original.spec("train180")["result"])}
    args = arguments(tmp_path)
    args.resume, args.execution_output = True, args.root / "batch_attempts/repair1"
    resumed = QueueCLI(args)
    resumed.queues["formal360"] = dict(pending=pending, claimed=0, complete=400-pending)
    asyncio.run(asyncio.wait_for(resumed.run(), 5))
    assert resumed.record["status"] == "complete" and all(p.read_bytes() == data for p, data in protected.items())
    assert {p["stage"] for p in resumed.record["processes"]} == {"formal360"}
    assert {p["command"] for p in resumed.record["processes"]} == ({"prepare", "worker", "aggregate"} if pending else {"prepare", "aggregate"})
    prepared = next(p for p in resumed.record["processes"] if p["command"] == "prepare")
    assert {"--recover-claims", "--retry-failed"} <= set(prepared["argv"])
    assert str(receipt) in resumed.record["preserved_executions"]


def test_resume_topology_change_requires_explicit_flag_and_uses_full_checkpoint(tmp_path):
    args = arguments(tmp_path)
    runner = QueueCLI(args)
    runner.root.mkdir()
    runner.logs.mkdir()
    write_checkpoint(runner, 45, world=1)
    with pytest.raises(ValueError, match="world changed"):
        asyncio.run(runner.train(180))
    args.allow_topology_change = True
    asyncio.run(runner.train(180))
    command = runner.record["processes"][0]["argv"]
    assert command[command.index("--resume") + 1] == str(runner.checkpoint(45))
    assert "--allow-topology-change" in command and "--nproc-per-node=2" in command


def test_partial_aggregate_or_changed_scientific_metadata_cannot_qualify(tmp_path):
    runner = QueueCLI(arguments(tmp_path))
    runner.root.mkdir()
    runner.seal()
    runner.emit_result("train180")
    runner.emit_result("formal180")
    path = runner.spec("formal180")["result"]
    row = json.loads(path.read_text())
    row["aggregated_complete"] = False
    path.write_text(json.dumps(row))
    runner.check_stop()
    assert runner.result("formal180") is None and not runner.stopped
    row["checkpoint"] = str(runner.checkpoint(360))
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="registered checkpoint"):
        runner.result("formal180")
    runner.args.resume = True
    runner.args.asset_root = tmp_path / "different_assets"
    with pytest.raises(ValueError, match="scientific/asset"):
        runner.seal()
    runner.emit_result("pool0")
    path = runner.spec("pool0")["result"]
    row = json.loads(path.read_text())
    row["version"] = "new_behavior"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="registered pool"):
        runner.result("pool0")


class LocalChildren(batch.Supervisor):
    async def snapshot(self, node):
        return snapshots()[node]

    async def collect_initial(self):
        async with asyncio.TaskGroup() as group:
            group.create_task(self.child("pool0-long", "gpu01", [sys.executable, "-u", "-c",
                "import time; print('owned long child',flush=True); time.sleep(60)"], (("gpu01", 0),)))
            group.create_task(self.child("pool0-fail", "gpu01", [sys.executable, "-u", "-c",
                "import time,sys; time.sleep(.2); print('original failure',flush=True); sys.exit(7)"], (("gpu01", 1),)))


def test_nonzero_child_cancels_owned_group_and_records_transport_cost_and_logs(tmp_path):
    runner, start = LocalChildren(arguments(tmp_path)), time.monotonic()
    with pytest.raises(ExceptionGroup):
        asyncio.run(runner.run())
    assert time.monotonic() - start < 10 and not runner.leased
    assert runner.record["status"] == "failed"
    failed, cancelled = next(p for p in runner.record["processes"] if p["name"] == "pool0-fail"), runner.record["processes"][0]
    assert failed["return_code"] == 7 and cancelled["status"] == "cancelled"
    assert "original failure" in Path(failed["log"]).read_text()
    assert '"batch_child_returncode": -15' in Path(cancelled["log"]).read_text()
    costs = [json.loads(line) for line in (runner.root / "costs.jsonl").read_text().splitlines()]
    assert len(costs) == 4 and {r["event"] for r in costs} == {"start", "end"}
    assert all(r["allocation_window"] and r["node"] == "BCI-gpu01" and r["phase"] == "collect" for r in costs)
    assert all(r["end"] >= r["start"] for r in costs if r["event"] == "end")


def test_gpu_admission_checks_current_cap_process_ownership_and_actual_headroom():
    live = snapshots()
    assert batch.admission(live, [("gpu02", 2)], {("gpu02", 0)}, "ymdai", 40000, 10)["total_cap"] == 8
    for row in live["gpu01"][:6]:
        row["processes"] = [dict(owner="ymdai")]
    with pytest.raises(ValueError, match="total6/node6"):
        batch.admission(live, [("gpu02", 2)], set(), "ymdai", 40000, 10)
    live = snapshots()
    live["gpu02"][1]["free"] = 39999
    with pytest.raises(ValueError, match="admission refused"):
        batch.admission(live, [("gpu02", 1)], set(), "ymdai", 40000, 10)
    live["gpu02"][1]["processes"] = [dict(owner="unknown")]
    with pytest.raises(ValueError, match="attribute"):
        batch.admission(live, [("gpu02", 2)], set(), "ymdai", 40000, 10)
    with pytest.raises(ValueError, match="already leased"):
        batch.admission(snapshots(), [("gpu02", 1)], {("gpu02", 1)}, "ymdai", 40000, 10)


def test_commands_keep_new_physical_flags_and_shell_arguments(tmp_path):
    runner = batch.Supervisor(arguments(tmp_path))
    command = runner.program("train", runner.root / "training", stop_update=360, resume=runner.checkpoint(180))
    assert {"CUDA_VISIBLE_DEVICES=0,1", "NCCL_P2P_DISABLE=1", "torch.distributed.run", "--nproc-per-node=2"} <= set(command)
    assert {"--slot-batch", "--stop-update", "--resume"} <= set(command) and "--pg-microbatch" not in command
    remote = runner.transport("gpu02", [sys.executable, "-c", "print('$(must not run)`')"])
    assert shlex.split(remote[-1]) == [sys.executable, "-c", "print('$(must not run)`')"]
    with pytest.raises(SystemExit):
        batch.parser().parse_args(["--resume-stage", "formal"])
