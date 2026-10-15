"""Explicit one-time expansion of the completed first window; never replay it."""
from contextlib import closing
from pathlib import Path
import datetime
import json
import shutil
import sqlite3

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import (SCHEMA,FIRST_TASKS,NEW_TASKS,TASKS,WINDOW,LIMITS,
                       environment_contract,checkpoint,queue_path)


def prepare_window(root):
    root=Path(root);path=root/'run_contract.json';previous=read_json(path)
    if previous.get('revision')==2:
        if (previous['window'],previous['tasks'],previous['limits'])!=(WINDOW,list(TASKS),LIMITS):
            raise ValueError('reference completion window contract changed')
        return previous
    if (previous['schema'],previous['tasks'])!=(SCHEMA,list(FIRST_TASKS)):
        raise ValueError('completion window requires the original eight-task contract')
    accepted=read_json(root/'analysis/engineering_completion_20261015/acceptance.json')
    if accepted['FM_queries']!=309120 or not accepted['all_recorded_PIDs_released']:
        raise ValueError('first window has not been completely accepted/released')
    manifest=read_json(checkpoint(root,0,1920)/'manifest.json')
    if not manifest['complete'] or not manifest['full_optimizer_rng_sampler']:
        raise ValueError('task0 must continue its complete last1920 checkpoint')
    frozen={str(t):read_json(root/'selection'/f'task_{t:04d}'/'selected.json')
            for t in FIRST_TASKS if t!=0}
    environment=environment_contract(root)
    for old in previous['environment']['tasks']:
        new=next(t for t in environment['tasks'] if t['global_task_id']==old['global_task_id'])
        if any(new[k]!=v for k,v in old.items()):
            raise ValueError('expansion changed a first-window environment identity')
    window=root/'windows'/WINDOW;archive=window/'previous';archive.mkdir(parents=True,exist_ok=True)
    names=('run_contract.json','queue_plan.json','work.sqlite','completion.json',
           'owner_launch.json','owner_exit.json')
    for name in names:
        destination=archive/name
        if not destination.exists():shutil.copy2(root/name,destination)
    (archive/'analysis').mkdir(exist_ok=True)
    for name in ('results.json','selection_rows.json','report_rows.json','producer_videos.json','report.md'):
        destination=archive/'analysis'/name
        if not destination.exists():shutil.copy2(root/'analysis'/name,destination)
    dispatch=read_json(root/'inputs'/f'{WINDOW}_dispatch.json')
    run=dict(previous,revision=2,window=WINDOW,tasks=list(TASKS),limits=LIMITS,environment=environment,
        scope='train24 plus audited source-seen support12 references only; no D/q/RL/held/Test',
        original_first_window=str(archive),original_contract_commit='44c0ee3a5de19313aafc77f8236377a61fa81fb5',
        frozen_first_seven=frozen,fresh_tasks=list(NEW_TASKS),task0_continuation=dict(start=1920,stop=2160),
        completion_dispatch=dispatch,revised_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        physical_execution='reuse first-window useful microbatch28 and evaluation8/16; per-node CPU admission',
        cumulative_prior_actual=dict(optimizer_updates=11040,FM_queries=309120,physical_submissions=2492,
            control_steps=587803,dummy_steps=24920,total_steps=612723,engineering_extra_attempts_used=0),
        additional_maximum=dict(optimizer_updates=54000,FM_queries=1512000,physical_submissions=10948,
            control_steps=3934320,dummy_steps=109480,total_steps=4043800),
        queue=str(queue_path(root)),plan=str(window/'queue_plan.json'))
    write_json_atomic(path,run)
    return run


def import_first_window_receipts(root):
    """Copy completed receipts into the revised queue, preserving the old DB."""
    root=Path(root);source=root/'windows'/WINDOW/'previous/work.sqlite'
    with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(queue_path(root))) as new:
        old.row_factory=sqlite3.Row;new.row_factory=sqlite3.Row
        records=list(old.execute('SELECT * FROM jobs'))
        if any(r['status'] not in ('pending','complete') for r in records):
            raise ValueError('first-window queue has an unresolved physical owner')
        fields=('status','worker_id','claimed_unix','claim_token','attempt','finished_unix',
                'rows_path','rows_bytes','row_count','successes','error')
        with new:
            for row in records:
                current=new.execute('SELECT * FROM jobs WHERE job_id=?',(row['job_id'],)).fetchone()
                if current is None or any(current[k]!=row[k] for k in ('ordinal','payload')):
                    raise ValueError('revised queue changed a historical work identity')
                if row['status']=='complete' and current['status']=='pending':
                    new.execute('UPDATE jobs SET '+','.join(k+'=?' for k in fields)+' WHERE job_id=?',
                                (*[row[k] for k in fields],row['job_id']))
    return dict(original_jobs=len(records),reused_complete=sum(r['status']=='complete' for r in records))
