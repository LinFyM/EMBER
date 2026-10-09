"""Event-driven supervisor for the registered Compiler batch; no score selection.

Run this script in ordinary tmux. Every child uses the existing Compiler CLI.
Both node snapshots and all child stdout/exit evidence remain in the run root.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import getpass
import json
from pathlib import Path
import shlex
import socket
import sys
import time
import traceback


MODULE = "ember.experience_compiler.run"
PHYSICAL = ("microbatch", "pg_microbatch", "frame_chunk", "native_frame_chunk",
            "decoder_chunk", "experience_chunk", "cpu_threads")
SNAPSHOT = r'''
import json, pathlib, pwd, subprocess
def query(fields, kind):
    return subprocess.check_output(['nvidia-smi', '--query-'+kind+'='+fields,
           '--format=csv,noheader,nounits'], text=True).splitlines()
gpus = {}
for row in query('index,uuid,name,memory.used,memory.free,utilization.gpu', 'gpu'):
    index, uuid, name, used, free, util = map(str.strip, row.split(','))
    gpus[uuid] = dict(index=int(index), uuid=uuid, name=name, used=int(used),
                      free=int(free), utilization=int(util), processes=[])
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


def checkpoint_event(line: bytes, root: Path):
    if b"checkpoint_ready" not in line:
        return None
    try:
        row = ast.literal_eval(line.decode().strip())
    except (ValueError, SyntaxError):
        return None
    if not isinstance(row, dict) or "checkpoint_ready" not in row:
        return None
    meta = row.get("meta_update")
    if meta not in (27, 54):
        return None
    path = Path(row["checkpoint_ready"]).resolve()
    expected = root / "training" / "checkpoints" / f"macro_{128 + meta:08d}"
    if path != expected:
        raise ValueError(f"checkpoint event changed registered optimizer node: {row}")
    return meta, path


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
        self.args, self.events = args, asyncio.Queue()
        self.leased, self.lock = set(), asyncio.Lock()
        self.root = args.root
        self.execution_root = args.execution_output or self.root
        self.logs = self.execution_root / "batch_logs"
        self.record = {"schema_version": "ember_experience_batch_execution_v1", "status": "running",
                       "started_unix": time.time(), "arguments": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()}, "processes": [], "admissions": [], "stages": []}

    def save(self):
        path = self.execution_root / "batch_execution.json"
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(self.record, indent=2) + "\n")
        partial.replace(path)

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
        snapshots = dict(zip(self.args.nodes, await asyncio.gather(*(self.snapshot(n) for n in self.args.nodes))))
        minimum = (self.args.train_minimum_free_mib if name == "train" else self.args.eval_minimum_free_mib)
        receipt = {"child": name, "time": time.time(), "snapshots": snapshots, "minimum_free_mib": minimum}
        self.record["admissions"].append(receipt)
        try:
            receipt["decision"] = admission(snapshots, targets, self.leased, self.args.owner,
                                             minimum, self.args.maximum_utilization)
        except BaseException as error:
            receipt["error"] = repr(error)
            self.save()
            raise

    def program(self, command, output, *, checkpoint=None, stage=None, gpu=None):
        args = self.args
        cli = ["--asset-root", str(args.asset_root), "--run-root", str(self.root), "--output", str(output)]
        for key in PHYSICAL:
            cli += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        if checkpoint is not None:
            cli += ["--checkpoint", str(checkpoint)]
        if stage is not None:
            cli += ["--stage", stage]
        if command == "prepare" and stage == args.resume_stage == "formal":
            cli += ["--recover-claims", "--retry-failed"]
        prefix = [str(args.python), "-u"]
        devices = args.train_gpus if command == "train" else ([] if gpu is None else [gpu])
        if command == "train":
            prefix += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       f"--nproc-per-node={len(devices)}"]
            cli += ["--physical-gpus", ",".join(map(str, devices))]
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

    async def consume(self, proc, log, training):
        while True:
            chunk = await (proc.stdout.readline() if training else proc.stdout.read(65536))
            if not chunk:
                return
            log.write(chunk)
            log.flush()
            if training and (event := checkpoint_event(chunk, self.root)) is not None:
                await self.events.put(event)

    async def child(self, name, node, argv, targets=(), *, training=False):
        proc, log, allocated = None, None, False
        entry = {"name": name, "node": node, "argv": argv, "physical_gpus": list(targets),
                             "log": str(self.logs / f"{name}.log"), "started_unix": time.time()}
        self.record["processes"].append(entry)
        try:
            log = Path(entry["log"]).open("xb")
            async with self.lock:
                if targets:
                    await self.admit(name, targets)
                transport = self.transport(node, [str(self.args.python), "-u", "-c", CHILD, str(self.args.code), *argv])
                proc = await asyncio.create_subprocess_exec(*transport, stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=16 * 1024**2)
                self.leased.update(targets)
                allocated = True
                entry.update(transport_pid=proc.pid, transport=transport)
                self.save()
            await self.consume(proc, log, training)
            code = await proc.wait()
            if code:
                raise RuntimeError(f"{name} exited{code}; original log {entry['log']}")
            entry["status"] = "complete"
            if training:
                await self.events.put(None)
        except BaseException as error:
            entry.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=repr(error))
            if proc is not None:
                await asyncio.gather(self.stop(proc), self.consume(proc, log, False))
            raise
        finally:
            if log is not None:
                log.close()
            if allocated:
                self.leased.difference_update(targets)
            entry.update(finished_unix=time.time(), return_code=None if proc is None else proc.returncode)
            self.save()

    async def panel(self, stage, checkpoint, targets):
        output = self.root / "evaluation" / stage
        record = {"stage": stage, "checkpoint": str(checkpoint), "conditions": 400 if stage == "formal" else 16}
        self.record["stages"].append(record)
        await self.child(f"{stage}-prepare", self.args.local_node,
                         self.program("prepare", output, checkpoint=checkpoint, stage=stage))
        async with asyncio.TaskGroup() as pool:
            for node, gpu in targets:
                pool.create_task(self.child(f"{stage}-{node}-gpu{gpu}", node,
                    self.program("worker", output, checkpoint=checkpoint, stage=stage, gpu=gpu), ((node, gpu),)))
        await self.child(f"{stage}-aggregate", self.args.local_node, self.program("aggregate", output))
        if not (output / "results.json").is_file():
            raise RuntimeError(f"aggregate exited without results: {output}")
        record.update(status="complete", results=str(output / "results.json"))
        self.save()

    async def readouts(self, train):
        targets = [(self.args.eval_node, gpu) for gpu in self.args.eval_gpus]
        for expected in (27, 54):
            event = await self.events.get()
            if event is None or event[0] != expected:
                raise RuntimeError(f"missing/out-of-order meta{expected} checkpoint event: {event}")
            if expected == 54 and self.args.borrow_training_gpus:
                await train
                targets += [(self.args.train_node, gpu) for gpu in self.args.train_gpus]
            await self.panel(f"meta{expected}", event[1], targets)
        await self.panel("formal", event[1], targets)

    async def resume_formal(self):
        """Consume the original completed learning/panels; never restart them."""
        original = self.root / "batch_execution.json"
        previous = json.loads(original.read_text())
        if previous["status"] != "failed" or any("finished_unix" not in p for p in previous["processes"]):
            raise ValueError("formal recovery requires a terminated failed execution receipt")
        completion = json.loads((self.root / "training/completion.json").read_text())
        checkpoint = self.root / "training/checkpoints/macro_00000182"
        if (not completion.get("training_complete") or completion.get("updates") != 182
                or completion.get("warm") != 128 or completion.get("meta") != 54
                or Path(completion["checkpoint"]).resolve() != checkpoint):
            raise ValueError("formal recovery requires the original complete meta54 training")
        for stage, macro in (("meta27", 155), ("meta54", 182)):
            results = json.loads((self.root / f"evaluation/{stage}/results.json").read_text())
            if Path(results["checkpoint"]).resolve() != self.root / f"training/checkpoints/macro_{macro:08d}":
                raise ValueError("completed train panel changed its registered optimizer node")
        self.record["preserved_execution"] = str(original)
        self.record["training_completion"] = str(self.root / "training/completion.json")
        targets = [(self.args.eval_node, gpu) for gpu in self.args.eval_gpus]
        if self.args.borrow_training_gpus:
            targets += [(self.args.train_node, gpu) for gpu in self.args.train_gpus]
        await self.panel("formal", checkpoint, targets)

    async def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        if (self.execution_root / "batch_execution.json").exists():
            raise ValueError("batch_execution.json already exists; preserve its evidence before an explicit repair")
        self.save()
        try:
            if self.args.resume_stage == "formal":
                await self.resume_formal()
            else:
                async with asyncio.TaskGroup() as group:
                    train = group.create_task(self.child("train", self.args.train_node,
                        self.program("train", self.root / "training"),
                        tuple((self.args.train_node, gpu) for gpu in self.args.train_gpus), training=True))
                    group.create_task(self.readouts(train))
            self.record["status"] = "complete"
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
    result.add_argument("--resume-stage", choices=["formal"])
    result.add_argument("--execution-output", type=Path)
    result.add_argument("--python", type=Path, default=Path("/data1/user/ymdai/projects/EMBER/.venv/bin/python"))
    result.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    result.add_argument("--nodes", nargs=2, default=["gpu01", "gpu02"])
    result.add_argument("--local-node", default=socket.gethostname())
    result.add_argument("--train-node", required=True)
    result.add_argument("--train-gpus", type=gpu_list, required=True)
    result.add_argument("--eval-node", required=True)
    result.add_argument("--eval-gpus", type=gpu_list, required=True)
    result.add_argument("--owner", default=getpass.getuser())
    result.add_argument("--minimum-free-mib", type=int, required=True)
    result.add_argument("--train-minimum-free-mib", type=int)
    result.add_argument("--eval-minimum-free-mib", type=int)
    result.add_argument("--maximum-utilization", type=int, default=10)
    result.add_argument("--borrow-training-gpus", action="store_true")
    for name, default in zip(PHYSICAL, (28, 16, 16, 16, 16384, 16, 8)):
        result.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    return result


def main():
    args = parser().parse_args()
    args.code, args.root = args.code.resolve(), args.root.resolve()
    args.python, args.asset_root = args.python.absolute(), args.asset_root.resolve()
    if args.execution_output is not None:
        args.execution_output = args.execution_output.resolve()
        if not args.execution_output.is_relative_to(args.root):
            raise ValueError("execution receipts must remain within the owned run root")
    if args.resume_stage and (args.execution_output is None or args.execution_output == args.root):
        raise ValueError("formal recovery requires a new execution-output to retain the failed receipt")
    args.train_minimum_free_mib = args.minimum_free_mib if args.train_minimum_free_mib is None else args.train_minimum_free_mib
    args.eval_minimum_free_mib = args.minimum_free_mib if args.eval_minimum_free_mib is None else args.eval_minimum_free_mib
    if (len(set(args.nodes)) != 2 or args.train_node not in args.nodes or args.eval_node not in args.nodes
            or not 1 <= len(args.train_gpus) <= 4 or not 0 <= args.maximum_utilization <= 100
            or min(args.minimum_free_mib, args.train_minimum_free_mib, args.eval_minimum_free_mib,
                   *(getattr(args, key) for key in PHYSICAL)) <= 0):
        raise ValueError("invalid physical execution matrix or profile flags")
    if args.train_node == args.eval_node and set(args.train_gpus) & set(args.eval_gpus):
        raise ValueError("training and parallel evaluation must use disjoint GPUs")
    matrix = [(args.train_node, g) for g in args.train_gpus] + [(args.eval_node, g) for g in args.eval_gpus]
    if len(matrix) > 8 or any(sum(n == node for n, _ in matrix) > 6 for node in args.nodes):
        raise ValueError("input allocation matrix exceeds total8/node6 before live admission")
    if not args.root.is_relative_to(Path("/data1/user/ymdai")):
        raise ValueError("new Compiler outputs must be under /data1/user/ymdai")
    asyncio.run(Supervisor(args).run())


if __name__ == "__main__":
    main()
