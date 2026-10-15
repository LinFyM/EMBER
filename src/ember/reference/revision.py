"""Explicit bounded-window migration; never replay completed scientific work."""
from contextlib import closing
from pathlib import Path
import datetime
import shutil
import sqlite3

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import (SCHEMA,ACTIVE_TASKS,TASKS,WINDOW,REVISION,LIMITS,checkpoint,queue_path)


def prepare_window(root):
    root=Path(root);path=root/'run_contract.json';previous=read_json(path)
    if previous.get('revision')==REVISION:
        if (previous['window'],previous['tasks'],previous['limits'],previous['active_tasks'])!=(WINDOW,list(TASKS),LIMITS,list(ACTIVE_TASKS)):
            raise ValueError('reference final-four contract changed')
        return previous
    if (previous['schema'],previous['tasks'],previous.get('revision'))!=(SCHEMA,list(TASKS),2):
        raise ValueError('final-four window requires the accepted complete second window')
    accepted=read_json(root/'analysis/engineering_completion_window_20261015/acceptance.json')
    if (accepted['training_updates'],accepted['physical_submissions'],accepted['unsettled_tasks'])!=(45360,10566,list(ACTIVE_TASKS)) or not accepted['all_recorded_PIDs_released']:
        raise ValueError('previous window has not been accepted/released with exactly four unresolved tasks')
    for task in ACTIVE_TASKS:
        manifest=read_json(checkpoint(root,task,1920)/'manifest.json')
        if not manifest['complete'] or not manifest['full_optimizer_rng_sampler']:
            raise ValueError('each active task must continue complete last1920')
    frozen={str(t):read_json(root/'selection'/f'task_{t:04d}'/'selected.json')
            for t in TASKS if t not in ACTIVE_TASKS}
    environment=previous['environment']
    window=root/'windows'/WINDOW;archive=window/'previous';archive.mkdir(parents=True,exist_ok=True)
    names={'run_contract.json':path,'queue_plan.json':Path(previous['plan']),
           'work.sqlite':Path(previous['queue']),'completion.json':root/'completion.json',
           'owner_launch.json':root/'owner_launch.json','owner_exit.json':root/'owner_exit.json'}
    for name,source in names.items():
        destination=archive/name
        if not destination.exists():shutil.copy2(source,destination)
    (archive/'analysis').mkdir(exist_ok=True)
    for name in ('results.json','selection_rows.json','report_rows.json','producer_videos.json','report.md'):
        destination=archive/'analysis'/name
        if not destination.exists():shutil.copy2(root/'analysis'/name,destination)
    dispatch=read_json(root/'inputs'/f'{WINDOW}_dispatch.json')
    run=dict(previous,revision=REVISION,window=WINDOW,tasks=list(TASKS),limits=LIMITS,environment=environment,
        scope='train24 plus audited source-seen support12 references only; no D/q/RL/held/Test',
        previous_window_archive=str(archive),previous_training_code='0f8e00230968920e2b2bf544008b22d076b04d00',
        frozen_references=frozen,active_tasks=list(ACTIVE_TASKS),fresh_tasks=[],task0_continuation=None,
        task_continuations={str(t):dict(start=1920,maximum=2880) for t in ACTIVE_TASKS},
        current_dispatch=dispatch,revised_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        physical_execution='reuse accepted-window useful microbatch28 and evaluation8/16; per-node CPU admission',
        cumulative_prior_actual=dict(optimizer_updates=45360,FM_queries=1270080,physical_submissions=10566,
            control_steps=2282661,dummy_steps=105660,total_steps=2388321,engineering_extra_attempts_used=0),
        additional_maximum=dict(optimizer_updates=3840,FM_queries=107520,physical_submissions=672,
            control_steps=244480,dummy_steps=6720,total_steps=251200),
        queue=str(queue_path(root)),plan=str(window/'queue_plan.json'))
    write_json_atomic(path,run)
    return run


def import_previous_receipts(root):
    """Copy prior completed receipts into the revised queue, preserving its DB."""
    root=Path(root);source=root/'windows'/WINDOW/'previous/work.sqlite'
    with closing(sqlite3.connect(source)) as old, closing(sqlite3.connect(queue_path(root))) as new:
        old.row_factory=sqlite3.Row;new.row_factory=sqlite3.Row
        records=list(old.execute('SELECT * FROM jobs'))
        if any(r['status'] not in ('pending','complete') for r in records):
            raise ValueError('previous queue has an unresolved physical owner')
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
