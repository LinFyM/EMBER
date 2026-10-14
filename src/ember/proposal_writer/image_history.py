"""Exact MT-prefix input wall and shared frozen vision cache, without hidden."""
from pathlib import Path
import time

import torch

from ember.lora import validate_lora_state
from ember.pi05_eval_queue import claim_next, fail_job
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .path import load_history, history_prefix
from .learning import memory_peak

HISTORY_SCHEMA = 'ember_actual_MT_prefix_without_execution_hidden_v1'
MODEL_FIELDS = ('images', 'prepost', 'facts', 'record_indices')


def history_file(root, task):
    return Path(root)/'history_features'/f'task_{task:04d}.pt'


def actual_prefix(descriptor, lora):
    """Use the registered first output's exact prefix, never later G experience."""
    spec = descriptor['H']
    if spec['episodes'] != 1 or descriptor['producer'] != 'MT300' or spec['records'] < 1:
        raise ValueError('only the registered nonempty single MT episode is authorized')
    history = history_prefix(load_history(Path(spec['path'])), spec['records'], 1)
    if (len(history.records) != spec['records'] or len(history.episodes) != 1
            or set(history.states) != {'MT300'}
            or any(r['parameter_ref'] != 'MT300' or r['episode'] != 0 for r in history.records)):
        raise ValueError('H includes an incorrect producer, length or later episode')
    episode = history.episodes[0]
    if (episode['actual_parameter_ref'] != 'MT300'
            or episode['init_state_id'] != descriptor['practice_state_id']):
        raise ValueError('actual MT producer or practice initial state changed')
    if any(episode[k] != descriptor['bootstrap_rng'][k] for k in ('env_seed','policy_seed_root')):
        raise ValueError('registered MT environment or complete policy RNG changed')
    validate_lora_state(history.states['MT300'], lora)
    observations, tensors = event_tensors(history)
    return observations, tensors, episode


def event_tensors(history):
    """Tensorize only actual prefix RGB and the complete63 event facts."""
    keys = list(dict.fromkeys(k for r in history.records for k in (r['pre'], r['post'])))
    if set(keys) != set(history.observations):
        raise ValueError('only actual prefix pre/post observations may enter H')
    for raw in history.observations.values():
        if (raw['images'].shape != (2, 3, 256, 256) or raw['images'].dtype != torch.uint8
                or raw['proprio'].shape != (8,) or not torch.isfinite(raw['proprio']).all()):
            raise ValueError('actual RGB/proprio shape or finite contract changed')
    facts = []
    for row in history.records:
        if (row['proprio'].shape != (2,8) or row['actions'].shape != (5,7)
                or row['executed'].shape != (5,) or row['feedback'].shape != (5,)):
            raise ValueError('all actual pre/post, action, execution and feedback facts required')
        facts.append(torch.cat((row['proprio'].flatten(), row['actions'].flatten(),
            row['executed'].float(), row['feedback'], torch.tensor([row['episode'],row['step']],dtype=torch.float32))))
    facts = torch.stack(facts).float()
    if not torch.isfinite(facts).all() or any(b['step'] < a['step'] for a,b in zip(history.records,history.records[1:])):
        raise ValueError('actual ordered63 facts are nonfinite or out of order')
    index = {key:i for i,key in enumerate(keys)}
    prepost = torch.tensor([[index[r['pre']],index[r['post']]] for r in history.records],dtype=torch.long)
    # Do not access hidden or reuse features embedded in the old full path.
    observations = {key:history.observations[key] for key in keys}
    return observations, dict(prepost=prepost, facts=facts, record_indices=torch.arange(len(facts)))


@torch.no_grad()
def encode_history(runtime, descriptor):
    started = time.monotonic()
    observations, model, episode = actual_prefix(descriptor, runtime.lora)
    if runtime._generator is not None or runtime.execution._active_state is not None:
        raise ValueError('H images must use frozen vision/projector without old Meta or execution context')
    features = runtime.observation_features(observations)
    images = torch.stack([features[k] for k in observations]).to(torch.bfloat16).contiguous()
    if images.shape != (len(observations),512,2048) or not torch.isfinite(images).all():
        raise ValueError('all actual prefix images require finite native vision/projector features')
    return dict(**model, images=images, complete=True, schema_version=HISTORY_SCHEMA,
        source=dict(H=descriptor['H'],producer='MT300',parent_weights=descriptor['parent_weights'],
            records=len(model['facts']),observations=len(observations),episode_steps=episode.get('steps'),
            raw_RGB=True,vision_projector_only=True,Gemma=False,language=False,execution_hidden=False,
            only_registered_prefix=True,real_MT_state_validated=True,seconds=time.monotonic()-started))


def model_history(value, device=None):
    if (not value['complete'] or value['schema_version'] != HISTORY_SCHEMA
            or not value['source']['only_registered_prefix'] or value['source']['execution_hidden']
            or value['source']['Gemma'] or value['source']['language']
            or value['images'].ndim != 3 or value['images'].shape[1:] != (512,2048)
            or value['facts'].shape != (len(value['prepost']),63)
            or any(not torch.isfinite(value[k]).all() for k in MODEL_FIELDS)):
        raise ValueError('only complete frozen MT facts/images without hidden are model inputs')
    return {k:value[k].to(device) if device is not None else value[k] for k in MODEL_FIELDS}


def history_cache(args, runtime):
    from .image_data import publish_record
    entries = {r['task']:r for r in read_json(args.root/'history_manifest.json')['training']}
    queue = args.root/'queues/history_features.sqlite'
    while (claim:=claim_next(queue,worker_id=args.worker_id)) is not None:
        try:
            entry = entries[claim.shard.task_id]
            file = history_file(args.root,entry['task'])
            if not file.exists():
                value = encode_history(runtime,entry)
                file.parent.mkdir(parents=True,exist_ok=True)
                temporary = file.with_suffix(f'.{args.worker_id}.tmp')
                torch.save(value,temporary);temporary.replace(file)
            else:
                value = torch.load(file,map_location='cpu',weights_only=False)
            model_history(value)
            receipt = args.root/'history_receipts'/f'{claim.shard.job_id}_{claim.attempt:03d}.json'
            write_json_atomic(receipt,dict(complete=True,task=entry['task'],file=str(file),
                source=value['source'],peak=memory_peak(runtime.device),execution_hidden=False,
                no_new_practice=True,teaching_Gemma_forward_count=0))
            publish_record(queue,claim,args.worker_id,receipt,1)
        except BaseException as error:
            fail_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,
                claim_token=claim.claim_token,error=repr(error))
            raise
