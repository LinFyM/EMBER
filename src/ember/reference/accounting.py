"""One transactional interaction budget; unfinished attempts retain full cost."""
from pathlib import Path
from contextlib import contextmanager
import os
import sqlite3
import time

from .contract import LIMITS


@contextmanager
def connect(root):
    db=sqlite3.connect(Path(root)/'interactions.sqlite',timeout=60)
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('CREATE TABLE IF NOT EXISTS attempts (identity TEXT PRIMARY KEY,case_id TEXT NOT NULL,status TEXT NOT NULL,control INTEGER NOT NULL,dummy INTEGER NOT NULL,started REAL NOT NULL,finished REAL,row_path TEXT)')
    try:
        with db:yield db
    finally:db.close()


def reserve(root, case_id, horizon):
    with connect(root) as db:
        db.execute('BEGIN IMMEDIATE')
        count,unique,control,dummy=db.execute('SELECT count(*),count(distinct case_id),coalesce(sum(control),0),coalesce(sum(dummy),0) FROM attempts').fetchone()
        previous=db.execute('SELECT count(*) FROM attempts WHERE case_id=?',(case_id,)).fetchone()[0]
        if db.execute("SELECT count(*) FROM attempts WHERE case_id=? AND status='complete'",(case_id,)).fetchone()[0]:
            raise ValueError('valid saved interaction must never be rerun')
        if (count+1>LIMITS['physical_submissions'] or control+horizon>LIMITS['control_steps']
                or dummy+10>LIMITS['dummy_steps'] or control+dummy+horizon+10>LIMITS['total_steps']
                or count-unique+(previous>0)>LIMITS['engineering_extra_attempts']):
            raise RuntimeError('global reference interaction boundary reached')
        identity=f'{case_id}_attempt_{previous+1:03d}'
        db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,NULL,NULL)',
                   (identity,case_id,'reserved_unknown',horizon,10,time.time()))
    return identity


def finish(root, identity, row, path):
    control=int(row['environment_steps']-row['settling_steps']);dummy=int(row['settling_steps'])
    with connect(root) as db:
        changed=db.execute("UPDATE attempts SET status='complete',control=?,dummy=?,finished=?,row_path=? WHERE identity=? AND status='reserved_unknown'",
                           (control,dummy,time.time(),str(path),identity)).rowcount
        if changed!=1:raise ValueError('interaction receipt lost ownership')


def summary(root):
    with connect(root) as db:
        counts=db.execute('SELECT status,count(*),sum(control),sum(dummy) FROM attempts GROUP BY status').fetchall()
    return {status:dict(submissions=n,control=c,dummy=d,total=c+d) for status,n,c,d in counts}


def host_resources():
    """Current worker plus its descendants; no unrelated process mutation."""
    descendants={os.getpid()};processes={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            processes[int(path.name)]=(int(fields[1]),path)
        except (OSError,ValueError):continue
    while True:
        added={pid for pid,(parent,_) in processes.items() if parent in descendants}-descendants
        if not added:break
        descendants.update(added)
    pss=threads=0;observed=[];details=[]
    for pid in descendants:
        try:
            path=Path('/proc')/str(pid)
            process_pss=next(int(line.split()[1]) for line in (path/'smaps_rollup').read_text().splitlines() if line.startswith('Pss:'))
            process_threads=next(int(line.split()[1]) for line in (path/'status').read_text().splitlines() if line.startswith('Threads:'))
            pss+=process_pss;threads+=process_threads
            observed.append(pid)
            details.append(dict(PID=pid,PSS_KiB=process_pss,threads=process_threads))
        except (OSError,StopIteration):continue
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:','SwapTotal:'))}
    return dict(PID=os.getpid(),PIDs=sorted(observed),processes=len(observed),threads=threads,PSS_KiB=pss,per_process=details,host_memory_KiB=memory)
