"""Strict400 and fixed train panels with dynamic independent condition slots."""
from __future__ import annotations

import json
import os
import socket
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from ember.pi05_eval_queue import (EvaluationShard, claim_next, complete_job, completed_jobs,
                                   fail_job, initialize_queue, publish_json_exclusive,
                                   queue_summary, read_json_with_size)
from ember.pi05_eval_results import paired_success_comparison
from .contract import (ASSET_ROOT, RUN_ROOT, SCHEMA, MT_RESULTS, T_RESULTS, _LEARNING_CONTRACT, formal_environment,
                       learning_environment, formal400_mapping, panel_contract, STAGE)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def immutable(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError(f"immutable evaluation record changed: {path}")
    else:
        publish_json_exclusive(path, value)


def prepare(output: Path, checkpoint: Path, stage: str = "train180", asset_root: Path = ASSET_ROOT,
            *, recover_claims: bool = False, retry_failed: bool = False) -> Path:
    if stage not in {"train180", "train360", "formal180", "formal360"}:
        raise ValueError("only the preregistered180/360 complete panels are allowed")
    output, checkpoint, asset_root = Path(output).resolve(), Path(checkpoint).resolve(), Path(asset_root).resolve()
    manifest = read_json(checkpoint / 'checkpoint_manifest.json')
    expected_macro = 180 if stage.endswith('180') else 360
    if (manifest.get('stage') != STAGE or manifest.get('next_macro') != expected_macro
            or manifest.get('run_contract_schema') != SCHEMA):
        raise ValueError('evaluation checkpoint differs from its shared optimizer node')
    formal = stage.startswith('formal')
    conditions = list(formal400_mapping(asset_root=asset_root) if formal
                      else panel_contract(asset_root=asset_root)["conditions"])
    environment = (formal_environment if formal else learning_environment)(asset_root=asset_root)
    environment["parallel"] = {"envs_per_replica": 1}
    adaptation_environment = deepcopy(environment)
    adaptation_environment.pop("operator_read_write_scene", None)
    if not formal:
        environment["operator_read_write_scene"] = read_json(_LEARNING_CONTRACT)["operator_read_write_scene"]
    arms = ["end"] if formal else (["MT", "end", "null"] if stage == 'train180' else ["end", "null"])
    horizon = environment["environment"]["horizons"]
    contract = dict(schema_version=SCHEMA, stage=stage, checkpoint=str(checkpoint),
        asset_root=str(asset_root), environment_contract=environment,
        adaptation_environment_contract=adaptation_environment, arms=arms, conditions=conditions,
        adaptation_step_budget=1024, policy_seed_root=7, expected_rows_per_arm=400 if formal else 48,
        MT_reuse=None if stage != 'train360' else str(output.parent / 'train180/results.json'),
        references={"MT": str(MT_RESULTS), "T2340": str(T_RESULTS),
            "experience135": "/data1/user/ymdai/ember_runs/experience_conditioned_compiler_20261009/evaluation/formal/results.json"} if formal else {})
    _validate_panel(contract)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_contract.json"
    immutable(path, contract)
    shards = tuple(EvaluationShard(job_id=f"{stage}-{c['condition_id']}", ordinal=index,
        suite=c['suite'], task_id=int(c['suite_task_id']), horizon=int(horizon[c['suite']]),
        init_state_ids=tuple(c['final_state_ids']),
        estimated_cost=1024 + len(arms) * len(c['final_state_ids']) * int(horizon[c['suite']]))
        for index, c in enumerate(conditions))
    initialize_queue(output / 'queue.sqlite3', shards, contract_reference=str(path),
                     recover_claims=recover_claims, retry_failed=retry_failed)
    return path


def _validate_panel(contract: dict) -> None:
    conditions, formal = contract["conditions"], contract["stage"].startswith("formal")
    expected_count = 400 if formal else 16
    if len(conditions) != expected_count or len({c["condition_id"] for c in conditions}) != expected_count:
        raise ValueError("condition panel has missing or duplicate teaching conditions")
    tasks = defaultdict(list)
    for c in conditions:
        if not c["condition_id"] or "/" in c["condition_id"] or ".." in c["condition_id"]:
            raise ValueError("condition identity is not a safe queue/artifact component")
        states = c["final_state_ids"]
        if len(states) != (1 if formal else 3) or len(set(states)) != len(states) or not all(0 <= s < 50 for s in states):
            raise ValueError("condition final states differ from the fixed panel")
        if not 0 <= c["teacher_demo"] < 50:
            raise ValueError("teacher ordinal is outside the legal 50-video pool")
        tasks[c["task_id"]].append(c)
    if len(tasks) != 8:
        raise ValueError("readout must cover the registered eight tasks")
    for rows in tasks.values():
        _validate_task_conditions(rows, formal)


def _validate_task_conditions(rows, formal):
    if formal:
        if ({c['teacher_demo'] for c in rows} != set(range(50))
                or {c['final_state_ids'][0] for c in rows} != set(range(50)) or len(rows) != 50):
            raise ValueError('formal task/video/state conditions must be without replacement')
    elif len(rows) != 2 or len({c['teacher_demo'] for c in rows}) != 2:
        raise ValueError('train readout requires two distinct videos per task')


def _validate_rows(rows: list[dict], condition: dict, arm: str, task: dict) -> None:
    if len(rows) != len(condition["final_state_ids"]):
        raise ValueError("raw arm rows differ from the condition shard state count")
    if {row["init_state_id"] for row in rows} != set(condition["final_state_ids"]):
        raise ValueError("raw arm changed final state coverage")
    for row in rows:
        expected = {"suite": condition["suite"], "task_id": condition["suite_task_id"],
                    "condition_id": condition["condition_id"], "teacher_demo": condition["teacher_demo"],
                    "arm": arm, "language": task["language"], "split_role": task["split_role"],
                    "env_seed": 7, "policy_seed_root": 7}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("raw row changed condition, language, final RNG or task authority")
        if not isinstance(row.get("success"), bool) or not isinstance(row.get("policy_noise_seeds"), list):
            raise ValueError("raw final row must preserve success and actual policy-noise evidence")
        if "video_ordinal" in condition and row.get("video_ordinal") != condition["video_ordinal"]:
            raise ValueError("raw formal row changed the frozen teacher schedule")


def _task(contract: dict, condition: dict) -> dict:
    tasks = contract["environment_contract"]["tasks"]
    matches = [task for task in tasks if task["suite"] == condition["suite"]
               and int(task["task_id"]) == int(condition["suite_task_id"])]
    if len(matches) != 1:
        raise ValueError("condition does not identify exactly one registered environment task")
    return matches[0]


class _Cases:
    """One claim owns one frozen adaptation and every registered final row."""
    def __init__(self, runtime, contract, output, worker_id, gpu, code_git, run_root):
        from collections import deque
        self.runtime, self.contract, self.output = runtime, contract, output
        self.worker_id, self.gpu, self.code_git, self.run_root = worker_id, gpu, code_git, run_root
        self.cases, self.finals, self.completed = {}, deque(), []
        self.by_id = {f"{contract['stage']}-{c['condition_id']}": c for c in contract['conditions']}

    def stopped(self):
        return self.contract['stage'].endswith('360') and (self.run_root / 'early_stop_requested.json').is_file()

    def next_request(self):
        from .run import check_budget
        while True:
            if self.finals:
                return self.finals.popleft()
            if self.stopped():
                return None
            check_budget(self.run_root)
            claim = claim_next(self.output / 'queue.sqlite3', worker_id=self.worker_id, physical_gpu=self.gpu)
            if claim is None:
                return None
            condition = self.by_id[claim.shard.job_id]
            root = self.output / 'conditions' / claim.shard.job_id
            case = dict(claim=claim, condition=condition, destination=root / claim.claim_token,
                        rows={arm: [] for arm in self.contract['arms']}, future=None)
            self.cases[claim.shard.job_id] = case
            reused = self._recover(case, root)
            if reused:
                self._final_requests(case, reused)
                self.publish_ready()
                continue
            return dict(kind='adapt', condition=condition, task=_task(self.contract, condition),
                role='validation' if self.contract['stage'].startswith('formal') else 'train',
                case_id=claim.shard.job_id, behavior_version=self.contract['stage'],
                environment_contract=self.contract['adaptation_environment_contract'])

    def _recover(self, case, root):
        """Reuse valid frozen adaptation/negative rows after a same-contract repair."""
        import torch
        from safetensors.torch import load_file
        for path in sorted(root.glob('*/record.json')):
            record = read_json(path)
            if (not record.get('complete') or record.get('compiler_checkpoint') != self.contract['checkpoint']
                    or any(record.get(k) != v for k, v in case['condition'].items())):
                continue
            end = path.parent / 'end.safetensors'
            if not end.is_file():
                continue
            case.update(destination=path.parent, metrics=record['metrics'],
                        actual_J=record['metrics']['actual_J'], recovered_from=str(path))
            states = {'end': load_file(str(end))}
            if 'MT' in self.contract['arms']:
                states['MT'] = self.runtime.mt
            if 'null' in self.contract['arms']:
                null = path.parent / 'null.safetensors'
                if null.is_file():
                    states['null'] = load_file(str(null))
                else:
                    from .interaction import Chain
                    chain = Chain.from_record(torch.load(path.parent / 'experience.pt', map_location='cpu', weights_only=False))
                    teacher = self.runtime.teacher(case['condition']['task_id'], case['condition']['teacher_demo'])
                    states['null'] = self.runtime.null_replay(teacher, chain)
                    from safetensors.torch import save_file
                    save_file({k: v.detach().cpu().contiguous() for k, v in states['null'].items()}, str(null))
            return states
        return None

    def adapted(self, result):
        from .interaction import cpu_state
        request, chain = result['request'], result['chain']
        case = self.cases[request['case_id']]
        states = {'end': chain.states[-1]}
        if 'MT' in self.contract['arms']:
            states['MT'] = self.runtime.mt
        if 'null' in self.contract['arms']:
            condition = case['condition']
            teacher = self.runtime.teacher(condition['task_id'], condition['teacher_demo'])
            tick = time.monotonic()
            states['null'] = cpu_state(self.runtime.null_replay(teacher, chain))
            chain.metrics.update(null_replay_reads=len(chain.endpoints),
                null_full_video_equivalent_reads=float(len(chain.endpoints)),
                null_read_frames=len(chain.endpoints) * len(teacher['indices']),
                null_replay_seconds=time.monotonic() - tick,
                null_native_feature_cost=dict(self.runtime.last_teacher_cost))
        case.update(metrics=chain.metrics, actual_J=chain.metrics['actual_J'])
        def save():
            from .storage import save_condition
            from safetensors.torch import save_file
            record = save_condition(case['destination'], dict(**case['condition'],
                compiler_checkpoint=self.contract['checkpoint'], compilation_code_git=self.code_git), chain)
            if 'null' in states:
                save_file({k: v.detach().cpu().contiguous() for k, v in states['null'].items()},
                          str(case['destination'] / 'null.safetensors'))
            return record
        case['future'] = self.runtime.io.submit(save)
        self._final_requests(case, states)

    def _final_requests(self, case, states):
        condition = case['condition']
        for arm in self.contract['arms']:
            for state in condition['final_state_ids']:
                path = case['destination'] / f'final_{arm}_{state:02d}.json'
                if path.exists():
                    row = read_json(path)
                    _validate_rows([row], {**condition, 'final_state_ids': [state]}, arm, _task(self.contract, condition))
                    case['rows'][arm].append(row)
                else:
                    self.finals.append(dict(kind='final', case_id=case['claim'].shard.job_id, arm=arm,
                        task=_task(self.contract, condition), state_id=state, state=states[arm], noise_root=7,
                        environment_contract=self.contract['environment_contract']))

    def final(self, result):
        request, row = result['request'], result['row']
        case, arm = self.cases[request['case_id']], request['arm']
        condition = case['condition']
        row.update(condition_id=condition['condition_id'], teacher_demo=condition['teacher_demo'], arm=arm,
                   evaluation_code_git=self.code_git, compiler_checkpoint=self.contract['checkpoint'])
        if 'video_ordinal' in condition:
            row['video_ordinal'] = condition['video_ordinal']
        path = case['destination'] / f"final_{arm}_{row['init_state_id']:02d}.json"
        immutable(path, row)
        case['rows'][arm].append(row)

    def publish_ready(self, *, flush=False):
        for job_id, case in list(self.cases.items()):
            if not all(len(rows) == len(case['condition']['final_state_ids']) for rows in case['rows'].values()):
                continue
            future = case['future']
            if future is not None:
                if not flush and not future.done():
                    continue
                future.result()
            self._publish(case)
            self.completed.append(job_id)
            del self.cases[job_id]

    def _publish(self, case):
        claim, condition = case['claim'], case['condition']
        paths = {}
        for arm, rows in case['rows'].items():
            rows.sort(key=lambda r: r['init_state_id'])
            _validate_rows(rows, condition, arm, _task(self.contract, condition))
            path = f'shards/{claim.shard.job_id}-{claim.claim_token}-{arm}.json'
            publish_json_exclusive(self.output / path, dict(condition=condition, arm=arm, rows=rows))
            paths[arm] = path
        primary = read_json(self.output / paths['end'])
        primary.update(arm_paths=paths, artifact_root=str(case['destination'].relative_to(self.output)),
            metrics=case['metrics'], actual_J=case['actual_J'], chain='experience.pt',
            weights={arm: 'canonical_MT_reference' if arm == 'MT' else arm + '.safetensors' for arm in paths},
            wall_seconds=case['metrics']['wall_seconds'], recovered_from=case.get('recovered_from'))
        path = f'shards/{claim.shard.job_id}-{claim.claim_token}.json'
        size = publish_json_exclusive(self.output / path, primary)
        complete_job(self.output / 'queue.sqlite3', job_id=claim.shard.job_id, worker_id=self.worker_id,
            claim_token=claim.claim_token, rows_path=path, rows_bytes=size, row_count=len(primary['rows']),
            successes=sum(r['success'] for r in primary['rows']))


def worker(args):
    import torch
    from ember.writer.topology import bind_current_process_to_cuda_numa
    from .runtime import Runtime
    from .interaction import Runner

    output = Path(args.output).resolve()
    contract = read_json(output / 'evaluation_contract.json')
    if Path(args.checkpoint).resolve() != Path(contract['checkpoint']):
        raise ValueError('worker changed its frozen checkpoint')
    gpu = int(args.physical_gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu) or torch.cuda.device_count() != 1:
        raise ValueError('worker must expose exactly its admitted physical GPU')
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if affinity is None:
        raise ValueError('worker requires GPU-local NUMA affinity')
    torch.set_grad_enabled(False)
    identity = f'{socket.gethostname()}-{gpu}-{os.getpid()}'
    started, runtime, runner, cases, status, error = time.time(), None, None, None, 'failed', None
    try:
        runtime = Runtime(contract['asset_root'], args.device, frame_chunk=args.frame_chunk,
            decoder_chunk=args.decoder_chunk, experience_chunk=args.experience_chunk,
            native_frame_chunk=args.native_frame_chunk, cache_root=args.run_root / 'frozen_features')
        runtime.load_checkpoint(contract['checkpoint'])
        runtime.compiler.eval()
        runner = Runner(runtime, contract['adaptation_environment_contract'], gpu, slot_batch=args.slot_batch)
        cases = _Cases(runtime, contract, output, identity, gpu, args.code_git, args.run_root)
        for result in runner.run(cases.next_request):
            cases.adapted(result) if 'chain' in result else cases.final(result)
            cases.publish_ready()
        cases.publish_ready(flush=True)
        if cases.cases:
            raise ValueError('worker exited with incomplete claimed condition rows')
        status = 'scientific_stopped' if cases.stopped() else 'complete'
    except BaseException:
        error = traceback.format_exc()
        if cases is not None:
            for case in cases.cases.values():
                claim = case['claim']
                fail_job(output / 'queue.sqlite3', job_id=claim.shard.job_id, worker_id=identity,
                         claim_token=claim.claim_token, error=error)
        raise
    finally:
        if runner is not None:
            runner.close()
            if status == 'failed':
                runner.preserve_partial(output / 'failures' / identity)
        if runtime is not None:
            runtime.close()
        receipt = dict(worker_id=identity, host=socket.gethostname(), physical_gpu=gpu,
            code_git=args.code_git, checkpoint=str(args.checkpoint), started_unix=started,
            finished_unix=time.time(), status=status, cpu_affinity=affinity,
            completed_jobs=[] if cases is None else cases.completed, error=error,
            environment_steps=0 if runner is None else runner.total_environment_steps,
            components=None if runner is None else runner.components)
        immutable(output / 'workers' / f'{identity}.json', receipt)
    pending = queue_summary(output / 'queue.sqlite3')['status_counts'].get('pending', 0)
    print(dict(scientific_stopped=status == 'scientific_stopped', queue_pending=pending, status=status), flush=True)
    return receipt


def _indexed(rows: list[dict]) -> dict:
    result = {(r["suite"], r["task_id"], r["condition_id"], r["init_state_id"]): r for r in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("paired train rows are empty or duplicated")
    return result


def compare_train(reference: list[dict], candidate: list[dict]) -> dict:
    left, right = _indexed(reference), _indexed(candidate)
    if left.keys() != right.keys():
        raise ValueError("train comparison changed condition/state coverage")
    for key, before in left.items():
        after = right[key]
        fields = ("language", "env_seed", "policy_seed_root", "split_role", "teacher_demo", "scene_reference")
        common = min(len(before["policy_noise_seeds"]), len(after["policy_noise_seeds"]))
        if any(before.get(f) != after.get(f) for f in fields) or common < 1 or before["policy_noise_seeds"][:common] != after["policy_noise_seeds"][:common]:
            raise ValueError("train comparison changed language/video/final RNG pairing")
    def counts(keys):
        before, after = {k for k in keys if left[k]["success"]}, {k for k in keys if right[k]["success"]}
        return {"rows": len(keys), "reference_successes": len(before), "candidate_successes": len(after),
                "retained": len(before & after), "gained": len(after - before), "lost": len(before - after),
                "churn_count": len(before ^ after), "churn_fraction": len(before ^ after) / len(keys),
                "success_set_jaccard": len(before & after) / len(before | after) if before | after else None,
                "retained_keys": sorted(before & after), "gained_keys": sorted(after - before), "lost_keys": sorted(before - after)}
    tasks = sorted({k[:2] for k in left})
    per_task = [{"suite": suite, "task_id": task, **counts([k for k in left if k[:2] == (suite, task)])}
                for suite, task in tasks]
    return {**counts(list(left)), "per_task": per_task,
            "per_suite": [{"suite": suite, **counts([k for k in left if k[0] == suite])}
                          for suite in sorted({k[0] for k in left})],
            "reference_breadth": sum(p["reference_successes"] > 0 for p in per_task),
            "candidate_breadth": sum(p["candidate_successes"] > 0 for p in per_task)}


def _summary(rows: list[dict]) -> dict:
    tasks, suites = defaultdict(list), defaultdict(list)
    for row in rows:
        tasks[(row["suite"], row["task_id"])].append(row)
        suites[row["suite"]].append(row)
    def counts(values):
        return {"row_count": len(values), "successes": sum(r["success"] for r in values)}
    return {**counts(rows), "breadth": sum(any(r["success"] for r in values) for values in tasks.values()),
            "per_task": [{"suite": k[0], "task_id": k[1], **counts(v)} for k, v in sorted(tasks.items())],
            "per_suite": [{"suite": k, **counts(v)} for k, v in sorted(suites.items())]}


def compare_formal(reference: dict, candidate: dict) -> dict:
    result = paired_success_comparison(reference, candidate)
    key = lambda row: (row["suite"], row["task_id"], row["init_state_id"])
    previous = {key(row): row for row in reference["rows"]}
    for row in candidate["rows"]:
        before = previous[key(row)]
        if row.get("scene_reference") != before.get("scene_reference"):
            raise ValueError("formal comparison changed the actual canonical scene reference")
        teacher = before.get("operator_read_write_lora")
        if teacher and any(row.get(field) != teacher[field] for field in ("condition_id", "teacher_demo", "video_ordinal")):
            raise ValueError("formal comparison changed the original T teaching condition")
    return result


def _cost(output: Path, conditions: list[dict]) -> dict:
    intervals, receipts = defaultdict(list), []
    for path in sorted((output / "workers").glob("*.json")):
        row = read_json(path)
        receipts.append(row)
        intervals[(row["host"], row["physical_gpu"])].append((row["started_unix"], row["finished_unix"]))
    allocated = 0.0
    for periods in intervals.values():
        start, stop = None, None
        for a, b in sorted(periods):
            if start is None:
                start, stop = a, b
            elif a <= stop:
                stop = max(stop, b)
            else:
                allocated += stop - start
                start, stop = a, b
        if start is not None:
            allocated += stop - start
    metrics = defaultdict(list)
    for row in conditions:
        for key, value in {**row["metrics"], "actual_J": row["actual_J"], "condition_wall_seconds": row["wall_seconds"]}.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key].append(value)
    distributions = {}
    for key, values in metrics.items():
        ordered = sorted(values)
        distributions[key] = {"count": len(values), "sum": sum(values), "mean": sum(values) / len(values),
                              "p50": ordered[int(.5 * (len(values) - 1))], "p90": ordered[int(.9 * (len(values) - 1))], "max": ordered[-1]}
    return {"allocated_physical_GPUh": allocated / 3600, "worker_receipts": receipts,
            "condition_cost_distributions": distributions}


def _panel_comparisons(output, contract, panels, references):
    comparison = {}
    if contract['stage'] == 'train360':
        prior = read_json(Path(contract['MT_reuse']))
        if not prior.get('aggregated_complete') or prior['stage'] != 'train180':
            raise ValueError('strongMT reuse requires the original complete train180 panel')
        panels['MT'] = prior['arms']['MT']['rows']
    if 'MT' in panels:
        comparison['MT_to_end'] = compare_train(panels['MT'], panels['end'])
    if 'null' in panels:
        comparison['null_to_end'] = compare_train(panels['null'], panels['end'])
    if contract['stage'].endswith('360'):
        previous = output.parent / contract['stage'].replace('360', '180') / 'results.json'
        if previous.exists():
            prior = read_json(previous)
            rows = prior['arms']['end']['rows']
            comparison['180_to_360'] = (compare_formal({'rows': rows}, {'rows': panels['end']})
                if contract['stage'].startswith('formal') else compare_train(rows, panels['end']))
    for name, path in (contract['references'] if references is None else references).items():
        reference = read_json(Path(path))
        if 'rows' not in reference:
            reference = {'rows': reference['arms']['end']['rows']}
        comparison[name] = compare_formal(reference, {'rows': panels['end']})
    return comparison


def aggregate(output: Path, references: dict | None = None) -> dict:
    output = Path(output).resolve()
    contract = read_json(output / "evaluation_contract.json")
    _validate_panel(contract)
    queue = queue_summary(output / "queue.sqlite3")
    if queue["status_counts"] != {"complete": len(contract["conditions"])}:
        raise ValueError("aggregation requires the whole condition queue to complete")
    panels, conditions = {arm: [] for arm in contract["arms"]}, []
    expected = {c["condition_id"]: c for c in contract["conditions"]}
    seen = set()
    for job in completed_jobs(output / "queue.sqlite3"):
        primary, size = read_json_with_size(output / job["rows_path"])
        c = primary["condition"]
        if c != expected.get(c["condition_id"]) or c["condition_id"] in seen or size != job["rows_bytes"]:
            raise ValueError("completed artifact changed the frozen condition or queue identity")
        seen.add(c["condition_id"])
        for arm in contract["arms"]:
            raw = read_json(output / primary["arm_paths"][arm])
            if raw["condition"] != c or raw["arm"] != arm:
                raise ValueError("raw sidecar belongs to another condition/arm")
            _validate_rows(raw["rows"], c, arm, _task(contract, c))
            panels[arm].extend(raw["rows"])
        if primary["rows"] != read_json(output / primary["arm_paths"]["end"])["rows"] or len(primary["rows"]) != job["row_count"]:
            raise ValueError("primary queue rows disagree with the actual end-arm raw rows")
        conditions.append({key: primary[key] for key in ("condition", "metrics", "actual_J", "weights", "chain", "artifact_root", "wall_seconds")})
    if seen != expected.keys() or any(len(rows) != contract["expected_rows_per_arm"] for rows in panels.values()):
        raise ValueError("aggregate arm coverage does not match the preregistered panel")
    comparison = _panel_comparisons(output, contract, panels, references)
    result = {"schema_version": SCHEMA, "stage": contract["stage"], "checkpoint": contract["checkpoint"],
              "aggregated_complete": True, "queue": queue, "arms": {arm: {**_summary(rows), "rows": rows} for arm, rows in panels.items()},
              "conditions": conditions, "comparisons": comparison, "cost": _cost(output, conditions)}
    immutable(output / "results.json", result)
    return result
