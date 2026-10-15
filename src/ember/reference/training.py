"""Source-only native FM, uniform episode/frame cursor and complete checkpoints."""
from pathlib import Path
import json
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import default_collate

from ember.batched_lora import BatchedLoRAInference
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_source_setup import load_policy
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_processing import Pi05LiberoProcessor
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank
from ember.writer.functional import prepare_frozen_writer_policy, task_logical_batch_policy_rng_seed
from ember.writer.data import FunctionalQueryDataset
from ember.writer.learning_data import load_learning_tasks
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample, mean_velocity_loss
from ember.writer.model import DirectLoRAParameters
from ember.writer.runtime import autocast
from .contract import ASSETS, SOURCE, EVALUATION, NORMALIZATION, TASKS, OPTIMIZER, SCHEMA, NODES, checkpoint


class Runtime:
    def __init__(self, root, device='cuda:0'):
        self.device, self.root = torch.device(device), Path(root)
        authority = load_evaluation_authorities(ASSETS / EVALUATION, ASSETS)
        source = ASSETS / SOURCE
        self.source = inspect_source_checkpoint(authority, source.parent.parent, source, evaluation_mode='formal')
        self.policy = load_policy(Path(self.source['model_path']), authority.source_base_config, self.device)
        self.lora = derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'), rank=8)
        self.identity = prepare_frozen_writer_policy(self.policy, self.lora)
        self.policy.model.gradient_checkpointing_disable()
        self.execution = BatchedLoRAInference(self.policy, self.lora)
        self.processor = Pi05LiberoProcessor(read_json(ASSETS/NORMALIZATION)['stats'],
            ASSETS/'models/tokenizers/openpi/paligemma_tokenizer.model', 200, str(self.device))
        self.tasks = load_learning_tasks(ASSETS, TASKS, protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
        if (len(self.lora.targets), self.lora.rank, self.lora.parameter_count) != (38,8,643584):
            raise ValueError('reference must be complete 38-target rank8')
        if any(p.requires_grad for p in self.policy.parameters()):
            raise ValueError('source and physical identity adapter must remain frozen')
        if any(torch.count_nonzero(v) for k,v in self.identity.items() if 'lora_B' in k):
            raise ValueError('fresh reference identity has a nonzero B')

    @staticmethod
    def task_parameters(state):
        return {'policy.'+k:v for k,v in state.items()}


class QueryStream:
    def __init__(self, runtime, task):
        self.task = int(task)
        self.authority = runtime.tasks[task]
        self.data = FunctionalQueryDataset((self.authority.authority,), demo_indices=tuple(range(50)),
            action_chunk_size=50, action_start_offset=1)
        self.rows = self.data.task_episode_rows[task]
        self.orders, self.frames = {}, {}

    def query(self, n):
        cycle, slot = divmod(n, 50)
        if cycle not in self.orders:
            self.orders = {cycle:np.random.default_rng(np.random.SeedSequence([7,self.task,0,cycle])).permutation(50)}
        demo = int(self.orders[cycle][slot])
        turn, position = divmod(cycle, self.authority.episode_lengths[demo]-1)
        key = demo, turn
        if key not in self.frames:
            self.frames[key] = np.random.default_rng(np.random.SeedSequence([7,self.task,1,demo,turn])).permutation(self.authority.episode_lengths[demo]-1)
        return dict(demo=demo, frame=int(self.frames[key][position]))

    def sample(self, runtime, update):
        queries = [self.query(update*28+i) for i in range(28)]
        batch = default_collate([self.data[self.rows[q['demo']][q['frame']]] for q in queries])
        batch = runtime.processor.training_batch(batch)
        seed = task_logical_batch_policy_rng_seed(optimization_seed=7,task_id=self.task,task_visit=update,
            demo_indices=[q['demo'] for q in queries],frame_indices=[q['frame'] for q in queries])
        sample = flow_sample(runtime.policy,batch,seed=seed,device=runtime.device,random_batch=28,offset=0)
        return sample, dict(update=update+1,queries=queries,noise_time_root=seed,next_query=(update+1)*28)


def slice_tree(value, start, end):
    if isinstance(value, torch.Tensor):
        return value[start:end]
    return type(value)(slice_tree(v,start,end) for v in value)


def save_checkpoint(runtime, task, update, model, optimizer, scheduler, microbatch, measures):
    destination = checkpoint(runtime.root,task,update)
    temporary = destination.with_name(destination.name+f'.partial_{os.getpid()}')
    temporary.mkdir(parents=True,exist_ok=False)
    save_file({k:v.detach().cpu().contiguous() for k,v in model().items()},str(temporary/'lora.safetensors'))
    state = dict(schema=SCHEMA,task=task,next_update=update,names=model.names,optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),sampler=dict(seed=7,next_query=update*28,logical_batch=28),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.random.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()),topology=dict(world_size=1,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        physical_microbatch=microbatch,physical_measurements=measures)
    torch.save(state,temporary/'state.pt')
    write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,task=task,update=update,
        source=str(ASSETS/SOURCE),rank=8,targets=38,complete=True,full_optimizer_rng_sampler=True))
    temporary.rename(destination)


def restore(runtime, task, start, model, optimizer, scheduler):
    saved = torch.load(checkpoint(runtime.root,task,start)/'state.pt',map_location='cpu',weights_only=False)
    if (saved['schema'],saved['task'],saved['next_update'],tuple(saved['names'])) != (SCHEMA,task,start,model.names):
        raise ValueError('reference resume schema/task/cursor changed')
    if saved['sampler'] != dict(seed=7,next_query=start*28,logical_batch=28):
        raise ValueError('reference logical query stream changed')
    weights=load_file(str(checkpoint(runtime.root,task,start)/'lora.safetensors'),device=str(runtime.device))
    with torch.no_grad():
        for k,v in model().items():v.copy_(weights[k])
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
    torch.random.set_rng_state(saved['rng']['cpu']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
    return saved['physical_microbatch'],saved['physical_measurements']


def update_model(runtime, owner, model, optimizer, sample, microbatch):
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.
    for first in range(0,28,microbatch):
        last=min(first+microbatch,28)
        part=FlowSample(slice_tree(sample.arguments,first,last),sample.target[first:last],sample.action_width)
        with autocast(runtime.device):
            prediction=torch.func.functional_call(owner,runtime.task_parameters(model()),(part,),strict=False)
            loss=mean_velocity_loss(prediction,part.target,part.action_width)*((last-first)/28)
        loss.backward();loss_value+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    return loss_value,float(norm)


def train(runtime, task, start, stop):
    if task not in TASKS or (start,stop) not in tuple(zip(NODES,NODES[1:])):
        raise ValueError('unregistered task/update segment')
    torch.manual_seed(7);random.seed(7);np.random.seed(7)
    model=DirectLoRAParameters({k:v.float() for k,v in runtime.identity.items()}).to(runtime.device)
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    microbatch,measures=(7,[])
    if start:microbatch,measures=restore(runtime,task,start,model,optimizer,scheduler)
    stream=QueryStream(runtime,task);owner=NativeFlowPrediction(runtime.policy)
    metrics=runtime.root/'training'/f'task_{task:04d}'/'metrics.jsonl'
    metrics.parent.mkdir(parents=True,exist_ok=True)
    if metrics.exists():
        with metrics.open('rb') as handle:
            handle.seek(max(0,metrics.stat().st_size-65536));last=handle.read().splitlines()[-1]
        if json.loads(last)['update']!=start:
            raise RuntimeError('unsaved optimizer updates cannot be silently replayed')
    try:
        for update in range(start,stop):
            candidate=(7,14,28)[update%3] if update<6 else microbatch
            tick=time.monotonic()
            sample,event=stream.sample(runtime,update)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            ooms=[]
            while True:
                try:
                    loss,norm=update_model(runtime,owner,model,optimizer,sample,candidate)
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                    ooms.append(candidate)
                    if candidate<=7:raise
                    candidate//=2
            optimizer.step();scheduler.step();torch.cuda.synchronize()
            duration=time.monotonic()-tick
            measure=dict(microbatch=candidate,seconds=duration,queries_per_second=28/duration,
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,oom_attempts=ooms)
            if update<6:
                measures.append(measure)
                # Second useful update at each physical size removes its cold
                # shape/kernel startup without adding or discarding training.
                warmed=measures[3:] or measures
                microbatch=max(warmed,key=lambda m:m['queries_per_second'])['microbatch']
            event.update(loss=loss,gradient_norm=norm,**measure)
            with metrics.open('a') as handle:handle.write(json.dumps(event)+'\n')
        save_checkpoint(runtime,task,stop,model,optimizer,scheduler,microbatch,measures)
        return dict(task=task,start=start,stop=stop,queries=(stop-start)*28,
                    physical_microbatch=microbatch,physical_measurements=measures)
    finally:
        stream.data.close()
