"""Event-driven supervisor for the registered Compiler batch; no score selection.

Run this script in ordinary tmux. Every child uses the existing Compiler CLI.
Both node snapshots and all child stdout/exit evidence remain in the run root.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time
import traceback


MODULE = "ember.experience_compiler.run"
SCHEMA = "ember_parameter_conditioned_compiler_v1"
EVENT_SCHEMA = "ember_parameter_compiler_events_v1"
STAGE = "parameter_conditioned_compiler_20261009"
PHYSICAL = ("microbatch", "frame_chunk", "native_frame_chunk",
            "decoder_chunk", "experience_chunk", "cpu_threads", "slot_batch")
SNAPSHOT = r'''
import json, pathlib, pwd, socket, subprocess
def query(fields, kind):
    return subprocess.check_output(['nvidia-smi', '--query-'+kind+'='+fields,
           '--format=csv,noheader,nounits'], text=True).splitlines()
gpus = {}
for row in query('index,uuid,name,memory.used,memory.free,utilization.gpu', 'gpu'):
    index, uuid, name, used, free, util = map(str.strip, row.split(','))
    gpus[uuid] = dict(index=int(index), uuid=uuid, name=name, used=int(used),
                      free=int(free), utilization=int(util), processes=[], hostname=socket.gethostname())
for row in query('gpu_uuid,pid,used_memory', 'compute-apps'):
    uuid, pid, memory = map(str.strip, row.split(','))
    owner, command = 'unknown', ''
    try:
        proc = pathlib.Path('/proc') / pid
        owner = pwd.getpwuid(proc.stat().st_uid).pw_name
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')[:512]
    except FileNotFoundError:
        owner = 'gone'
    except (OSError, KeyError):
        pass
    gpus[uuid]['processes'].append(dict(pid=int(pid), owner=owner, memory=memory, command=command))
print(json.dumps(list(gpus.values())))
'''
CHILD = r'''
import json, os, signal, subprocess, sys, threading
os.chdir(sys.argv[1])
child = subprocess.Popen(sys.argv[2:], start_new_session=True, stdin=subprocess.DEVNULL)
print(json.dumps({'batch_child_pid': child.pid}), flush=True)
def stop(sig):
    try: os.killpg(child.pid, sig)
    except ProcessLookupError: pass
def control():
    while os.read(0, 4096): pass
    stop(signal.SIGTERM)
    timer = threading.Timer(10, stop, args=(signal.SIGKILL,))
    timer.daemon = True
    timer.start()
threading.Thread(target=control, daemon=True).start()
code = child.wait()
print(json.dumps({'batch_child_returncode': code}), flush=True)
sys.exit(code if code >= 0 else 128 - code)
'''


def admission(snapshots, requested, leased, owner, minimum_free, maximum_utilization):
    devices = {(node, row["index"]): row for node, rows in snapshots.items() for row in rows}
    if len({row["uuid"] for row in devices.values()}) != len(devices):
        raise ValueError("node aliases duplicate physical GPU UUIDs")
    unknown = [key for key, row in devices.items() if any(p["owner"] == "unknown" for p in row["processes"])]
    if unknown:
        raise ValueError(f"cannot attribute GPU processes on {unknown}")
    owned = {key for key, row in devices.items() if any(p["owner"] == owner for p in row["processes"])}
    free = sum(not row["processes"] and row["used"] <= 1000 and row["utilization"] <= 5
               and key not in leased for key, row in devices.items())
    cap = 6 if free <= 10 else 8
    occupied = owned | set(leased) | set(requested)
    if len(occupied) > cap or any(sum(key[0] == node for key in occupied) > 6 for node in snapshots):
        raise ValueError(f"GPU allocation exceeds total{cap}/node6: {sorted(occupied)}; idle={free}")
    for key in requested:
        row = devices.get(key)
        if key in leased or row is None or "A40" not in row["name"]:
            raise ValueError(f"GPU is unavailable, already leased, or not A40: {key}")
        if row["free"] < minimum_free or row["utilization"] > maximum_utilization:
            raise ValueError(f"GPU admission refused {key}: {row}; require free>={minimum_free}MiB, util<={maximum_utilization}")
    return {"idle_devices": free, "total_cap": cap, "owned_or_allocated": sorted(occupied)}


class Supervisor:
    def __init__(self, args):
        self.args, self.leased, self.lock = args, set(), asyncio.Lock()
        self.released = {}
        self.root, self.execution_root = args.root, args.execution_output or args.root
        self.logs = self.execution_root / "batch_logs"
        self.prepared, self.results = set(), {}
        self.train_cards = tuple((args.train_node, g) for g in args.train_gpus)
        self.eval_cards = tuple((args.eval_node, g) for g in args.eval_gpus)
        self.cards = self.train_cards + self.eval_cards
        self.record = {"schema_version": SCHEMA, "status": "running", "started_unix": time.time(),
                       "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       "processes": [], "admissions": [], "stages": {}}

    def save(self):
        path = self.execution_root / "batch_execution.json"
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(self.record, indent=2) + "\n")
        partial.replace(path)

    @property
    def stopped(self):
        return (self.root / "early_stop_requested.json").is_file()

    def transport(self, node, argv):
        if node in (self.args.local_node, "localhost"):
            return argv
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, shlex.join(argv)]

    async def snapshot(self, node):
        argv = self.transport(node, [str(self.args.python), "-c", SNAPSHOT])
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        if proc.returncode:
            raise RuntimeError(f"node snapshot failed {node}: exit{proc.returncode}: {stderr.decode(errors='replace')}")
        return json.loads(stdout)

    async def admit(self, name, targets):
        # NVIDIA utilization describes the preceding sample window. Allow our
        # just-exited child to leave that window before taking the next live
        # snapshot; busy/unknown owners are still rejected by admission.
        age = min((time.monotonic() - self.released.get(card, 0.) for card in targets), default=2.)
        if age < 1.5:
            await asyncio.sleep(1.5 - age)
        snapshots = dict(zip(self.args.nodes, await asyncio.gather(*(self.snapshot(n) for n in self.args.nodes))))
        minimum = (self.args.train_minimum_free_mib if name.startswith("train-") else self.args.eval_minimum_free_mib)
        receipt = {"child": name, "time": time.time(), "snapshots": snapshots, "minimum_free_mib": minimum}
        self.record["admissions"].append(receipt)
        try:
            receipt["decision"] = admission(snapshots, targets, self.leased, self.args.owner,
                                             minimum, self.args.maximum_utilization)
            return snapshots[targets[0][0]][0].get("hostname", targets[0][0])
        except BaseException as error:
            receipt["error"] = repr(error)
            self.save()
            raise

    def program(self, command, output, *, checkpoint=None, stage=None, pool=None,
                gpu=None, stop_update=None, resume=None):
        args = self.args
        cli = ["--asset-root", str(args.asset_root), "--run-root", str(self.root), "--output", str(output)]
        for key in PHYSICAL:
            cli += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        for key, value in (("checkpoint", checkpoint), ("stage", stage), ("pool", pool),
                           ("stop-update", stop_update), ("resume", resume)):
            if value is not None:
                cli += ["--" + key, str(value)]
        if command in ("prepare", "collect-prepare") and args.resume:
            cli += ["--recover-claims", "--retry-failed"]
        prefix = [str(args.python), "-u"]
        devices = args.train_gpus if command == "train" else ([] if gpu is None else [gpu])
        if command == "train":
            prefix += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       f"--nproc-per-node={len(devices)}"]
            cli += ["--physical-gpus", ",".join(map(str, devices))]
            if args.allow_topology_change:
                cli += ["--allow-topology-change"]
        if gpu is not None:
            cli += ["--physical-gpu", str(gpu)]
        environment = ["env", f"PYTHONPATH={args.code / 'src'}", "PYTHONUNBUFFERED=1",
                       "PYTHONDONTWRITEBYTECODE=1", "NCCL_P2P_DISABLE=1", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                       "MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl",
                       f"OMP_NUM_THREADS={args.cpu_threads}", f"MKL_NUM_THREADS={args.cpu_threads}"]
        if devices:
            environment += ["CUDA_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
                            f"MUJOCO_EGL_DEVICE_ID={devices[0]}"]
        return environment + prefix + ["-m", MODULE, command] + cli

    async def stop(self, proc):
        if proc.returncode is None:
            proc.stdin.close()  # EOF terminates only the remote group we created.
            try:
                await asyncio.wait_for(proc.wait(), timeout=20)
            except asyncio.TimeoutError:
                proc.kill()  # Only our transport; the remote EOF handler has its own kill timer.
                await proc.wait()

    async def consume(self, proc, log):
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                return
            log.write(chunk)
            log.flush()

    def allocation(self, entry, event):
        row = dict(identity=entry["allocation_identity"], phase=entry["phase"], node=entry["canonical_node"],
                   physical_gpus=[gpu for _, gpu in entry["physical_gpus"]], pid=os.getpid(),
                   start=entry["started_unix"], event=event, allocation_window=True, scientific=True)
        if event == "end":
            row["end"] = time.time()
        with (self.root / "costs.jsonl").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    async def child(self, name, node, argv, targets=(), *, training=False):
        proc, log, allocated = None, None, False
        entry = {"name": name, "node": node, "argv": argv, "physical_gpus": list(targets),
                             "log": str(self.logs / f"{name}.log"), "started_unix": time.time()}
        self.record["processes"].append(entry)
        try:
            log = Path(entry["log"]).open("xb")
            async with self.lock:
                if targets:
                    canonical_node = await self.admit(name, targets)
                    entry.update(canonical_node=canonical_node, started_unix=time.time(),
                        allocation_identity=f"batch-{os.getpid()}-{name}-{time.time_ns()}",
                        phase="train" if training else "collect" if name.startswith(("pool0-", "refresh180-")) else "worker")
                    self.allocation(entry, "start")
                transport = self.transport(node, [str(self.args.python), "-u", "-c", CHILD, str(self.args.code), *argv])
                proc = await asyncio.create_subprocess_exec(*transport, stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=16 * 1024**2)
                self.leased.update(targets)
                allocated = True
                entry.update(transport_pid=proc.pid, transport=transport)
                self.save()
            await self.consume(proc, log)
            code = await proc.wait()
            if code:
                raise RuntimeError(f"{name} exited{code}; original log {entry['log']}")
            entry["status"] = "complete"
            return entry
        except BaseException as error:
            entry.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=repr(error))
            if proc is not None:
                await asyncio.gather(self.stop(proc), self.consume(proc, log))
            raise
        finally:
            if log is not None:
                log.close()
            if allocated:
                self.leased.difference_update(targets)
                self.released.update({card: time.monotonic() for card in targets})
            if "allocation_identity" in entry:
                self.allocation(entry, "end")
            entry.update(finished_unix=time.time(), return_code=None if proc is None else proc.returncode)
            self.save()

    def spec(self, name):
        pool = name in ("pool0", "refresh180")
        point = 180 if name in ("refresh180", "train180", "formal180") else 360
        output = self.root / ("pools" if pool else "evaluation") / name
        return dict(output=output, pool=pool, point=point,
                    count=(144 if name == "pool0" else 72) if pool else (400 if name.startswith("formal") else 16),
                    result=output / ("manifest.json" if pool else "results.json"))

    def result(self, name):
        if name in self.results:
            return self.results[name]
        spec = self.spec(name)
        if not spec["result"].is_file():
            return None
        row = json.loads(spec["result"].read_text())
        if spec["pool"]:
            if (row.get("schema_version") != EVENT_SCHEMA or row.get("pool") != name
                    or row.get("version") != f"{name}_20261009_v1"
                    or not isinstance(row.get("conditions"), list) or len(row["conditions"]) != spec["count"]):
                raise ValueError(f"collection manifest differs from its registered pool: {name}")
            complete = row.get("complete") is True
        else:
            if (row.get("schema_version") != SCHEMA or row.get("stage") != name
                    or Path(row["checkpoint"]).resolve() != self.checkpoint(spec["point"])):
                raise ValueError(f"readout differs from its registered checkpoint: {name}")
            complete = row.get("aggregated_complete") is True
        if complete:
            self.results[name] = row
            self.record["stages"][name] = dict(status="complete", artifact=str(spec["result"]), reused=True)
            return row
        return None

    def checkpoint(self, macro):
        return self.root / "training/checkpoints" / f"macro_{macro:08d}"

    def checkpoint_metadata(self, path):
        row = json.loads((path / "checkpoint_manifest.json").read_text())
        if (row.get("schema_version") != "ember_ecp_checkpoint_v1" or row.get("stage") != STAGE
                or row.get("run_contract_schema") != SCHEMA
                or path != self.checkpoint(int(row["next_macro"]))):
            raise ValueError(f"incompatible complete training checkpoint: {path}")
        return row

    def latest_checkpoint(self):
        paths = list((self.root / "training/checkpoints").glob("macro_*/checkpoint_manifest.json"))
        complete = [(self.checkpoint_metadata(p.parent)["next_macro"], p.parent) for p in paths]
        return max(complete, default=(0, None))

    def counts(self, name):
        # Read only after prepare/child-exit events; never periodic queue polling.
        from ember.pi05_eval_queue import queue_summary
        return queue_summary(self.spec(name)["output"] / "queue.sqlite3")["status_counts"]

    async def prepare(self, name):
        if self.result(name) is not None or name in self.prepared:
            return
        spec = self.spec(name)
        await self.child(f"{name}-prepare", self.args.local_node, self.program(
            "collect-prepare" if spec["pool"] else "prepare", spec["output"],
            pool=name if spec["pool"] else None, stage=None if spec["pool"] else name,
            checkpoint=None if name == "pool0" else self.checkpoint(spec["point"])))
        self.prepared.add(name)
        self.record["stages"][name] = dict(status="prepared", conditions=spec["count"])

    async def aggregate(self, name):
        spec = self.spec(name)
        await self.child(f"{name}-aggregate", self.args.local_node,
                         self.program("collect-aggregate" if spec["pool"] else "aggregate", spec["output"]))
        if self.result(name) is None:
            raise RuntimeError(f"whole-queue aggregate produced no complete artifact: {name}")
        self.record["stages"][name]["reused"] = False
        self.check_stop()
        self.save()

    async def worker(self, name, card):
        spec, (node, gpu) = self.spec(name), card
        return await self.child(f"{name}-{node}-gpu{gpu}-{len(self.record['processes']):04d}", node,
            self.program("collect" if spec["pool"] else "worker", spec["output"], gpu=gpu,
                         pool=name if spec["pool"] else None, stage=None if spec["pool"] else name,
                         checkpoint=None if name == "pool0" else self.checkpoint(spec["point"])), (card,))

    def check_stop(self):
        formal, train = self.result("formal180"), self.result("train180")
        if self.stopped or formal is None or train is None:
            return
        successes = {arm: int(train["arms"][arm]["successes"]) for arm in ("MT", "end", "null")}
        final = int(formal["arms"]["end"]["successes"])
        if not (0 <= final <= 400 and all(0 <= value <= 48 for value in successes.values())):
            raise ValueError("readout counts exceed the registered complete panels")
        self.record["early_stop_evidence"] = dict(formal180=final, train180=successes)
        if final <= 135 and successes["end"] <= successes["MT"] and successes["end"] <= successes["null"]:
            marker = self.root / "early_stop_requested.json"
            with marker.open("x") as handle:
                json.dump(dict(schema_version=SCHEMA, reason="preregistered180_severe_nonpass",
                    formal180=final, train180=successes,
                    references=[str(self.spec(name)["result"]) for name in ("formal180", "train180")]), handle)

    async def collect_initial(self):
        await self.prepare("pool0")
        if self.result("pool0") is not None or self.stopped:
            return
        if self.counts("pool0") == {"complete": self.spec("pool0")["count"]}:
            await self.aggregate("pool0")
            return
        async with asyncio.TaskGroup() as pool:
            for card in self.cards:
                pool.create_task(self.worker("pool0", card))
        if not self.stopped:
            await self.aggregate("pool0")

    async def train(self, stop):
        checkpoint = self.checkpoint(stop)
        if checkpoint.is_dir():
            self.checkpoint_metadata(checkpoint)
            return checkpoint
        if self.stopped:
            return None
        macro, resume = self.latest_checkpoint()
        if macro >= stop or (self.root / "training/run_contract.json").exists() and resume is None:
            raise ValueError("training resume requires an existing full compatible checkpoint")
        if resume is not None and self.checkpoint_metadata(resume)["world_size"] != len(self.train_cards):
            if not self.args.allow_topology_change:
                raise ValueError("training world changed without explicit --allow-topology-change")
        if stop == 360 and macro < 180:
            raise ValueError("second segment requires the original complete180 checkpoint")
        await self.child(f"train-{stop}", self.args.train_node,
            self.program("train", self.root / "training", stop_update=stop, resume=resume),
            self.train_cards, training=True)
        row = json.loads((self.root / "training" / f"completion_{stop}.json").read_text())
        if row.get("scientific_early_stop") and self.stopped:
            saved = self.checkpoint_metadata(Path(row["checkpoint"]).resolve())
            if saved["next_macro"] != row["updates"] or not macro <= row["updates"] <= stop:
                raise ValueError("scientific stop did not retain its complete training boundary")
            return None
        if (not row.get("stage_complete") or row.get("updates") != stop
                or row.get("requested_stop_update") != stop or Path(row["checkpoint"]).resolve() != checkpoint):
            raise ValueError(f"training segment exited without its registered full checkpoint: {stop}")
        self.checkpoint_metadata(checkpoint)
        return checkpoint

    async def settle_queues(self, names, active):
        for name in names:
            if self.result(name) is None and not any(info[0] == name for info in active.values()):
                if self.counts(name) == {"complete": self.spec(name)["count"]}:
                    await self.aggregate(name)
        self.check_stop()

    def next_work(self, card, names, active, started360, finished360):
        if finished360:
            choices = ["formal180", "train360", "formal360"]
        elif card in self.train_cards:
            if started360:
                return None
            choices = sorted(("refresh180", "train180"),
                             key=lambda n: sum(info[0] == n for info in active.values()))
        else:
            choices = ["formal180"]
        for name in choices:
            if (name in names and self.result(name) is None
                    and not (self.stopped and name in ("refresh180", "train360", "formal360"))
                    and self.counts(name).get("pending", 0)):
                return name
        return None

    async def post180(self):
        names, active, free = ["refresh180", "train180", "formal180"], {}, set(self.cards)
        for name in names:
            await self.prepare(name)
        _, latest = self.latest_checkpoint()
        finished360 = latest is not None and self.checkpoint_metadata(latest)["next_macro"] == 360
        started360 = finished360
        async with asyncio.TaskGroup() as group:
            while True:
                if finished360 and not self.stopped:
                    for name in ("train360", "formal360"):
                        if name not in names:
                            names.append(name)
                            await self.prepare(name)
                await self.settle_queues(names, active)
                if not started360 and not self.stopped and all(self.result(n) is not None for n in ("refresh180", "train180")):
                    if not set(self.train_cards) <= free:
                        raise RuntimeError("training cards must be released before resume360")
                    task = group.create_task(self.train(360))
                    active[task] = ("training", self.train_cards)
                    free.difference_update(self.train_cards)
                    started360 = True
                for card in sorted(free):
                    chosen = self.next_work(card, names, active, started360, finished360)
                    if chosen:
                        task = group.create_task(self.worker(chosen, card))
                        active[task] = (chosen, (card,))
                        free.remove(card)
                if not active:
                    if self.stopped or all(self.result(n) is not None for n in ("refresh180", "train180", "formal180", "train360", "formal360")):
                        return
                    raise RuntimeError("no runnable child; incomplete queue or training dependency")
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    name, cards = active.pop(task)
                    outcome = task.result()
                    free.update(cards)
                    if name == "training":
                        finished360 = outcome is not None

    def seal(self):
        contract = dict(schema_version=SCHEMA, stage=STAGE, asset_root=str(self.args.asset_root),
                        seed=20261009, tasks=36, logical_batch=4, queries=28,
                        pools={"pool0": 144, "refresh180": 72}, checkpoints=[180, 360],
                        budget=dict(GPUh=40, scientific_elapsed_hours=16, peak_GiB=224), candidate=360)
        path = self.root / "batch_contract.json"
        if self.args.resume:
            if json.loads(path.read_text()) != contract:
                raise ValueError("resume changed the sealed scientific/asset contract")
            prior = [self.root / "batch_execution.json", *self.root.glob("batch_attempts/*/batch_execution.json")]
            for receipt in prior:
                if receipt.exists():
                    row = json.loads(receipt.read_text())
                    if row.get("status") == "running" or any("finished_unix" not in p for p in row["processes"]):
                        raise ValueError(f"resume requires previous children to have exited: {receipt}")
            self.record["preserved_executions"] = [str(p) for p in prior if p.exists()]
        else:
            if path.exists() or any((self.root / part).exists() for part in ("training", "pools", "evaluation")):
                raise ValueError("existing scientific artifacts require explicit --resume")
            with path.open("x") as handle:
                json.dump(contract, handle)

    async def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        if (self.execution_root / "batch_execution.json").exists():
            raise ValueError("execution-output already exists; retain it and use a new attempt")
        self.seal()
        self.save()
        try:
            if not self.stopped:
                await self.collect_initial()
                checkpoint = await self.train(180)
                if checkpoint is not None and not self.stopped:
                    await self.post180()
            self.record.update(status="scientific_stopped" if self.stopped else "complete",
                               candidate=None if self.stopped else str(self.checkpoint(360)))
        except BaseException:
            self.record.update(status="failed", traceback=traceback.format_exc())
            raise
        finally:
            self.record["finished_unix"] = time.time()
            self.save()


def gpu_list(value):
    result = [int(part) for part in value.split(",")]
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("provide distinct nonnegative physical GPU IDs")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "root"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--allow-topology-change", action="store_true")
    result.add_argument("--execution-output", type=Path, help="new receipt directory: ROOT/batch_attempts/ATTEMPT")
    result.add_argument("--python", type=Path, default=Path("/data1/user/ymdai/projects/EMBER/.venv/bin/python"))
    result.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    result.add_argument("--nodes", nargs=2, default=["gpu01", "gpu02"])
    result.add_argument("--local-node", default=socket.gethostname())
    for kind in ("train", "eval"):
        result.add_argument("--" + kind + "-node", required=True)
        result.add_argument("--" + kind + "-gpus", type=gpu_list, required=True)
        result.add_argument("--" + kind + "-minimum-free-mib", type=int)
    result.add_argument("--owner", default=getpass.getuser())
    result.add_argument("--minimum-free-mib", type=int, required=True)
    result.add_argument("--maximum-utilization", type=int, default=10)
    for name, default in zip(PHYSICAL, (28, 16, 16, 16384, 16, 8, 8)):
        result.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    return result


def main():
    args = parser().parse_args()
    args.code, args.root = args.code.resolve(), args.root.resolve()
    args.python, args.asset_root = args.python.absolute(), args.asset_root.resolve()
    sys.path.insert(0, str(args.code / "src"))
    if args.execution_output is not None:
        args.execution_output = args.execution_output.resolve()
        if args.execution_output.parent != args.root / "batch_attempts":
            raise ValueError("execution-output must be ROOT/batch_attempts/ATTEMPT so prior executions remain discoverable")
    if args.resume and (args.execution_output is None or args.execution_output == args.root):
        raise ValueError("resume requires a new execution-output to preserve failed evidence")
    for kind in ("train", "eval"):
        if getattr(args, kind + "_minimum_free_mib") is None:
            setattr(args, kind + "_minimum_free_mib", args.minimum_free_mib)
    if (len(set(args.nodes)) != 2 or args.train_node not in args.nodes or args.eval_node not in args.nodes
            or not 1 <= len(args.train_gpus) <= 4 or not 0 <= args.maximum_utilization <= 100
            or min(args.minimum_free_mib, args.train_minimum_free_mib, args.eval_minimum_free_mib,
                   *(getattr(args, key) for key in PHYSICAL)) <= 0):
        raise ValueError("invalid physical execution matrix or profile flags")
    matrix = [(args.train_node, g) for g in args.train_gpus] + [(args.eval_node, g) for g in args.eval_gpus]
    if len(set(matrix)) != len(matrix) or len(matrix) > 8 or any(sum(n == node for n, _ in matrix) > 6 for node in args.nodes):
        raise ValueError("GPU input matrix must be disjoint and within total8/node6")
    if not args.root.is_relative_to(Path("/data1/user/ymdai")):
        raise ValueError("new Compiler outputs must be under /data1/user/ymdai")
    asyncio.run(Supervisor(args).run())


if __name__ == "__main__":
    main()
