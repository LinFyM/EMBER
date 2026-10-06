#!/usr/bin/env python3
"""Thin, budgeted dispatcher for the one authorized fixed continuation panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import time

from ember.pi05_eval.post_open_language import ROOT, TEXTS, run_worker, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('launch', 'worker'))
    parser.add_argument('--phase', choices=('admission','remaining'))
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--worker-id')
    parser.add_argument('--gpu-id', type=int)
    parser.add_argument('--max-pairs', type=int, default=4)
    args = parser.parse_args()
    label=args.phase if args.attempt==1 else f'{args.phase}_retry{args.attempt}'
    if args.command == 'worker':
        os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(args.gpu_id))
        run_worker(args.max_pairs, args.worker_id, args.gpu_id)
        return
    if (ROOT/'launch/retired.json').exists(): raise RuntimeError('diagnostic is sealed')
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise RuntimeError('formal consumer must be clean')
    if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip():
        raise RuntimeError('formal consumer must be detached')
    if subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=repo).returncode:
        raise RuntimeError('formal consumer must be pushed')
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or len(gpus)>6: raise RuntimeError('physical GPU scope invalid')
    preflight_path = ROOT/'launch'/f'{label}_preflight.json'
    with preflight_path.open('x') as f:
        subprocess.run(['/data0/soft/anaconda3/bin/python',
            '/data0/user/ymdai/.codex/skills/gpu-preflight/scripts/gpu_preflight.py',
            '--nodes','gpu01,gpu02','--json'],stdout=f,check=True)
    preflight=json.loads(preflight_path.read_text())
    nodes=preflight['node_results']
    own={ (node,g['index']) for node,r in nodes.items() for g in r['gpus']
          if any(p.get('owner')=='ymdai' for p in g['processes']) }
    idle=sum(g['state']=='idle' for r in nodes.values() for g in r['gpus'])
    cap=6 if idle<=10 else 8
    if len(own | {('gpu02',g) for g in gpus})>cap: raise RuntimeError('global project GPU cap exceeded')
    for index in gpus:
        gpu=next(g for g in nodes['gpu02']['gpus'] if g['index']==index)
        if gpu['mem_free_mib']<30000 or gpu['util_gpu_pct']>5:
            raise RuntimeError(f'GPU {index} lost co-resident eligibility')
    acceptance=json.loads((ROOT/'launch/acceptance.json').read_text())
    previous=[json.loads(p.read_text()) for p in (ROOT/'launch').glob('*_launch_exit.json')]
    used_gpu=sum(r['complete_gpu_seconds'] for r in previous)
    used_cpu=sum(r['process_tree_cpu_seconds'] for r in previous)
    deadline=datetime.fromisoformat(acceptance['hard_deadline_utc'].replace('Z','+00:00')).timestamp()
    timeout=min(deadline-time.time()-180,(7200-used_gpu)/len(gpus), (57600-used_cpu)/(2*len(gpus)))
    if timeout<180: raise RuntimeError('insufficient remaining wall/GPU/CPU budget')
    cohort=json.loads((ROOT/'cohort.json').read_text())
    indices=[i for i,r in enumerate(cohort['rows']) if (r['model'],r['state']) in {('T2340',8),('C900',8)}]
    with sqlite3.connect(ROOT/'launch/pairs.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS pairs(cohort_index INTEGER PRIMARY KEY, remaining INTEGER, status TEXT, worker TEXT)')
        if args.phase=='remaining':
            for model,state in [('T2340',8),('C900',8)]:
                for arm in TEXTS:
                    p=ROOT/'rows'/f'{model}_state{state:03d}_{arm}'/'row.json'
                    if not p.exists() or not json.loads(p.read_text())['admitted']:
                        raise RuntimeError('four first consumer rows not admitted/complete')
        for i,r in enumerate(cohort['rows']):
            if (i in indices)==(args.phase=='admission'):
                existing=db.execute('SELECT status FROM pairs WHERE cohort_index=?',(i,)).fetchone()
                if existing is None:
                    db.execute('INSERT INTO pairs VALUES(?,?,"pending",NULL)',(i,r['remaining_control_steps']))
                elif existing[0]!='pending':
                    completed=[(ROOT/'rows'/f"{r['model']}_state{r['state']:03d}_{arm}"/'row.json').exists() for arm in TEXTS]
                    if all(completed):
                        db.execute('UPDATE pairs SET status="done" WHERE cohort_index=?',(i,))
                    elif args.attempt>1 and existing[0]=='running':
                        db.execute('UPDATE pairs SET status="pending", worker=NULL WHERE cohort_index=?',(i,))
                    else: raise RuntimeError('resume only unfinished arms after an explicit engineering repair')
    started=time.time(); cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    processes=[]; logs=[]; commands=[]; active_times=[]
    try:
        for gpu in gpus:
            wid=f'{label}_gpu{gpu}'
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
                TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'tmp/cache'),
                MPLCONFIGDIR=str(ROOT/'tmp/mpl'),TRITON_CACHE_DIR=str(ROOT/'tmp/triton'),
                TORCHINDUCTOR_CACHE_DIR=str(ROOT/'tmp/inductor'))
            env['LIBERO_CONFIG_PATH']=str(ROOT/'libero_config')
            (ROOT/'tmp').mkdir(exist_ok=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--worker-id',wid,'--gpu-id',str(gpu),
                 '--max-pairs','1' if args.phase=='admission' else '3']
            log=(ROOT/'launch'/f'{wid}.log').open('xb');logs.append(log)
            stamp=time.time();p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append(p);commands.append(cmd);active_times.append(stamp)
        write(ROOT/'launch'/f'{label}_launch.json',dict(started_unix=started,git=commit,
            gpus=gpus,preflight=str(preflight_path),whole_project_before=sorted(own),project_cap=cap,
            commands=commands,pids=[p.pid for p in processes],timeout_seconds=timeout))
        def wait_exit(item):
            p,stamp=item
            code=p.wait(timeout=max(1,started+timeout-time.time()))
            return code,time.time()-stamp
        with ThreadPoolExecutor(max_workers=len(processes)) as pool:
            exits=list(pool.map(wait_exit,zip(processes,active_times)))
        codes=[x[0] for x in exits]; durations=[x[1] for x in exits]
    except BaseException:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=20)
            except subprocess.TimeoutExpired: p.kill();p.wait()
        raise
    finally:
        for log in logs: log.close()
        cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN)
        if 'durations' not in locals():
            durations=[time.time()-stamp for stamp in active_times]
        write(ROOT/'launch'/f'{label}_launch_exit.json',dict(started_unix=started,finished_unix=time.time(),
            exit_codes=[p.returncode for p in processes],complete_gpu_seconds=sum(durations),
            physical_gpu_seconds=dict(zip(map(str,gpus),durations)),
            process_tree_cpu_seconds=cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime))
    if any(codes): raise RuntimeError(f'actual worker failure: {codes}')


if __name__=='__main__': main()
