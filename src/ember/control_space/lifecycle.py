"""Condition-scoped, post-consumption raw-H retirement with permanent facts."""
from pathlib import Path
import datetime,os
import torch
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .practice import load_history,H_SCHEMA


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def receipt_path(root,identity):return Path(root)/'archives'/identity/'retirement.json'


def tensor_identity(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())


def archive_facts(raw_path,destination,row):
    """Retain every event and observation identity; only RGB/hidden are temporary."""
    if destination.exists():
        saved=torch.load(destination,map_location='cpu',weights_only=False,mmap=True)
        if (not saved['complete'] or saved['raw_path']!=str(raw_path) or saved['attempt_id']!=row['attempt_id']):
            raise ValueError('raw retirement facts differ from their actual episode')
        return saved['raw_identity']
    history=load_history(raw_path)
    if history._archive_row['attempt_id']!=row['attempt_id']:raise ValueError('raw H changed its physical episode')
    records=[]
    for record in history.records:
        value={k:v for k,v in record.items() if k!='hidden'}
        value['native_hidden_identity']=tensor_identity(record['hidden']);records.append(value)
    observations={key:dict(proprio=value['proprio'],RGB_identity=tensor_identity(value['images']),
        raw_archive=str(raw_path),observation_key=key) for key,value in history.observations.items()}
    identity=dict(schema=H_SCHEMA,bytes=raw_path.stat().st_size,records=len(records),observations=len(observations),
        native_hidden_shape=[2,50,1024],RGB_shape=[2,3,256,256])
    value=dict(schema='ember_feedback_permanent_event_facts_v1',complete=True,raw_path=str(raw_path),
        attempt_id=row['attempt_id'],row=row,raw_identity=identity,records=records,observations=observations,
        episodes=history.episodes,behavior_codes=history.states,raw_RGB_hidden_in_event_facts=False)
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.partial')
    with temporary.open('wb') as handle:
        torch.save(value,handle);handle.flush();os.fsync(handle.fileno())
    temporary.replace(destination);return identity


def retire_condition(root,condition,features,*,environments_closed):
    root=Path(root).resolve();identity=condition['id'];destination=receipt_path(root,identity)
    if destination.exists():return read_json(destination)
    state=read_json(root/'conditions'/identity/'path.json')
    if (not environments_closed or not state['frozen'] or not state['query_complete'] or not state['complete']
            or state.get('pending') or state.get('pending_proposal') or len(state['query_rows'])!=1):
        raise ValueError('raw H retirement requires frozen final, no pending consumer and full runner.close')
    if condition['retain_raw_H']!=(condition['final_init']==0):raise ValueError('raw witness set must be fixed before results')
    rows=[row for panel in state['panels'] for row in panel['rows']]
    paths=[Path(row['history_path']).resolve() for row in rows]
    if len(paths)!=len(set(paths)) or any(not path.is_relative_to(root/'episodes') for path in paths):
        raise ValueError('retirement may own only this condition unique new raw episode archives')
    directory=destination.parent;directory.mkdir(parents=True,exist_ok=True);entries=[]
    # All small facts are atomic before any raw archive is removed. Partial
    # retirement resumes from these facts, never by reopening adaptation.
    for row,path in zip(rows,paths,strict=True):
        facts=directory/(row['attempt_id']+'.pt')
        raw_identity=archive_facts(path,facts,row)
        entries.append(dict(raw_path=str(path),facts=str(facts),facts_bytes=facts.stat().st_size,
            attempt_id=row['attempt_id'],raw_identity=raw_identity,retained=condition['retain_raw_H']))
    manifest=directory/'facts_manifest.json'
    value=dict(schema='ember_feedback_raw_H_lifecycle_v1',condition_id=identity,final_init=condition['final_init'],
        complete_facts=True,final_frozen=True,final_complete=True,no_pending=True,environments_closed=True,
        entries=entries,query_reused=state['query_reused'],selected_LoRA=state['selected_LoRA'],
        input_H_was_complete=True,permanent_raw_witness=condition['retain_raw_H'])
    if manifest.exists() and read_json(manifest)!=value:raise ValueError('retirement audit identity changed')
    write_json_atomic(manifest,value)
    eviction=features.evict_history([str(p) for p in paths])
    removed=0
    if not condition['retain_raw_H']:
        for entry,path in zip(entries,paths,strict=True):
            if path.exists():
                if path.stat().st_size!=entry['raw_identity']['bytes']:raise ValueError('raw payload bytes changed before retirement')
                removed+=path.stat().st_size;path.unlink()
    receipt=dict(value,UTC=utc(),facts_manifest=str(manifest),cache_eviction=eviction,deleted_raw_bytes=removed,
        full_H_replay_available=condition['retain_raw_H'],adaptation_resume_available=False,
        failure_unknown_fragments='preserved; only named complete valid row archives were eligible')
    write_json_atomic(destination,receipt)
    return receipt
