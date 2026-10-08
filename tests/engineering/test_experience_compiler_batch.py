"""CPU subprocess checks for events, failure cleanup and live-admission policy."""
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


def snapshots():
    return {node: [dict(index=gpu, uuid=f"{node}-{gpu}", name="NVIDIA A40", used=0,
                       free=45000, utilization=0, processes=[]) for gpu in range(8)]
            for node in ("gpu01", "gpu02")}


def arguments(tmp_path):
    args = batch.parser().parse_args(["--code", str(tmp_path), "--root", str(tmp_path / "run"),
        "--python", sys.executable, "--local-node", "gpu02", "--train-node", "gpu02",
        "--train-gpus", "0", "--eval-node", "gpu02", "--eval-gpus", "1",
        "--minimum-free-mib", "40000", "--borrow-training-gpus"])
    args.train_minimum_free_mib = args.eval_minimum_free_mib = args.minimum_free_mib
    return args


class CPUChildren(batch.Supervisor):
    def __init__(self, args, fail=False):
        super().__init__(args)
        self.fail, self.queries, self.overlap = fail, [], False

    async def snapshot(self, node):
        self.queries.append(node)
        return snapshots()[node]

    def program(self, command, output, *, checkpoint=None, stage=None, gpu=None):
        if command == "train":
            first = self.root / "training/checkpoints/macro_00000155"
            last = self.root / "training/checkpoints/macro_00000182"
            text = f"print({{'checkpoint_ready': str({str(first)!r}), 'meta_update': 27}}, flush=True)\n"
            text += f"time.sleep({60 if self.fail else 1})\n"
            text += f"print({{'checkpoint_ready': str({str(last)!r}), 'meta_update': 54}}, flush=True)\n"
        elif command == "prepare":
            text = f"Path({str(output)!r}).mkdir(parents=True, exist_ok=True)\nprint('prepared')\n"
        elif command == "worker":
            text = "print('worker raw evidence', flush=True)\n"
            text += "sys.exit(7)\n" if self.fail else "time.sleep(.03)\n"
        else:
            text = f"Path({str(output / 'results.json')!r}).write_text('{{}}')\nprint('X' * 100000)\n"
        return [sys.executable, "-u", "-c", "from pathlib import Path\nimport sys,time\n" + text]

    async def panel(self, stage, checkpoint, targets):
        if stage == "meta27":
            self.overlap = not any(p["name"] == "train" and "finished_unix" in p for p in self.record["processes"])
        await super().panel(stage, checkpoint, targets)


def test_events_overlap_training_then_reuse_released_cards_and_aggregate(tmp_path):
    runner = CPUChildren(arguments(tmp_path))
    asyncio.run(runner.run())
    record = json.loads((runner.root / "batch_execution.json").read_text())
    assert record["status"] == "complete" and runner.overlap and not runner.leased
    assert [(s["stage"], s["conditions"]) for s in record["stages"]] == [("meta27", 16), ("meta54", 16), ("formal", 400)]
    assert all(p["return_code"] == 0 for p in record["processes"])
    train = next(p for p in record["processes"] if p["name"] == "train")
    later = [p for p in record["processes"] if p["name"].startswith("meta54-")]
    assert all(train["finished_unix"] <= p["started_unix"] for p in later)
    assert len(record["admissions"]) == 6 and len(runner.queries) == 12
    assert (runner.logs / "formal-aggregate.log").stat().st_size > 100000
    assert all(Path(s["results"]).exists() for s in record["stages"])


def test_child_failure_cancels_only_our_live_groups_and_retains_exit_evidence(tmp_path):
    runner, start = CPUChildren(arguments(tmp_path), fail=True), time.monotonic()
    with pytest.raises(ExceptionGroup):
        asyncio.run(runner.run())
    assert time.monotonic() - start < 10 and not runner.leased
    record = json.loads((runner.root / "batch_execution.json").read_text())
    assert record["status"] == "failed"
    worker = next(p for p in record["processes"] if p["name"] == "meta27-gpu02-gpu1")
    train = next(p for p in record["processes"] if p["name"] == "train")
    assert worker["return_code"] == 7 and train["status"] == "cancelled"
    assert '"batch_child_returncode": -15' in Path(train["log"]).read_text()
    assert "worker raw evidence" in Path(worker["log"]).read_text()
    assert "formal" not in [s["stage"] for s in record["stages"]]


def test_gpu_limits_include_other_owner_jobs_leases_and_actual_headroom():
    live = snapshots()
    decision = batch.admission(live, [("gpu02", 2)], {("gpu02", 0)}, "ymdai", 40000, 10)
    assert decision["total_cap"] == 8 and len(decision["owned_or_allocated"]) == 2
    for row in live["gpu01"][:6]:
        row["processes"] = [dict(owner="ymdai")]
    with pytest.raises(ValueError, match="total6/node6"):
        batch.admission(live, [("gpu02", 2)], set(), "ymdai", 40000, 10)
    live = snapshots()
    with pytest.raises(ValueError, match="/node6"):
        batch.admission(live, [("gpu02", 6)], {("gpu02", i) for i in range(6)}, "ymdai", 40000, 10)
    live["gpu02"][1]["free"] = 39999
    with pytest.raises(ValueError, match="admission refused"):
        batch.admission(live, [("gpu02", 1)], set(), "ymdai", 40000, 10)
    with pytest.raises(ValueError, match="already leased"):
        batch.admission(snapshots(), [("gpu02", 1)], {("gpu02", 1)}, "ymdai", 40000, 10)


def test_registered_events_and_remote_shell_arguments_preserve_physical_flags(tmp_path):
    args = arguments(tmp_path)
    runner = batch.Supervisor(args)
    checkpoint = runner.root / "training/checkpoints/macro_00000155"
    event = repr(dict(checkpoint_ready=str(checkpoint), meta_update=27)).encode()
    assert batch.checkpoint_event(event, runner.root) == (27, checkpoint)
    assert batch.checkpoint_event(b"{'checkpoint_ready': 'ignored', 'meta_update': 9}", runner.root) is None
    with pytest.raises(ValueError, match="optimizer node"):
        batch.checkpoint_event(event.replace(b"00000155", b"00000154"), runner.root)
    argv = runner.program("train", runner.root / "training")
    assert "CUDA_VISIBLE_DEVICES=0" in argv and "NCCL_P2P_DISABLE=1" in argv
    assert "torch.distributed.run" in argv and "--physical-gpus" in argv
    from ember.experience_compiler.run import parser as native_parser
    parsed = native_parser().parse_args(argv[argv.index(batch.MODULE) + 1:])
    assert parsed.command == "train" and parsed.physical_gpus == [0]
    remote = runner.transport("gpu01", [sys.executable, "-c", "print('$(must not run)`')"])
    assert shlex.split(remote[-1]) == [sys.executable, "-c", "print('$(must not run)`')"]
