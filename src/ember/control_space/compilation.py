"""A fixed-theta batch of four complete adaptive parameter compilation paths."""
from pathlib import Path
import datetime,json,sys,time
import numpy as np
import torch
from safetensors.torch import save_file
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.reference.accounting import host_resources
from ember.writer.runtime import autocast
from .contract import condition_id,condition_root,checkpoint,PRACTICE_STEPS,SIGMA,BASIS
from .features import load_cached
from .model import load_basis
from .data import make_runtime
from .checkpoint import load


def emit(**value):print('EMBER_WORK_EVENT '+json.dumps(value),flush=True)
def archive(root,task):return condition_root(root,task)/'path.json'
def proposal_seed(task,ordinal):return int(np.random.SeedSequence([7,int(task),0,int(ordinal),202610161]).generate_state(1,dtype=np.uint64)[0]%(2**63-1))


def useful_mean(model,runtime,video,events,state,requested,event_cache=None):
    candidate=requested;ooms=[];tick=time.monotonic()
    if runtime.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    while True:
        try:
            with autocast(runtime.device):
                y0,memory=video;offset=0 if event_cache is None else event_cache.shape[1]
                if offset>len(events):raise ValueError('behavior H cache cannot go backwards')
                new=model.event_memory(events[offset:],memory,event_chunk=candidate,checkpointed=False,first_ordinal=offset)
                combined=new if event_cache is None else torch.cat((event_cache,new),1)
                encoded=model.causal_history(combined,checkpointed=False)
                mu=model.mean_from_encoded(y0,memory,encoded,len(events),torch.tensor(state['best_y'],device=runtime.device),
                    state['best_successes']/4,PRACTICE_STEPS-state['used_steps']).float()
            if runtime.device.type=='cuda':torch.cuda.synchronize(runtime.device)
            return mu,combined,dict(requested_chunk=requested,event_chunk=candidate,events=len(events)-offset,full_H_events=len(events),
                seconds=time.monotonic()-tick,OOM_attempts=ooms,
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if runtime.device.type=='cuda' else 0.,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if runtime.device.type=='cuda' else 0.)
        except torch.cuda.OutOfMemoryError:
            ooms.append(candidate);torch.cuda.empty_cache()
            if candidate<=4:raise
            candidate=max(4,candidate//2)


def physical_chunk(measures):
    candidates=(16,32,64,128,256,512)
    if len(measures)<len(candidates):return candidates[len(measures)]
    clean=[r for r in measures if not r['OOM_attempts']]
    return max(clean,key=lambda r:r['events']/max(1e-6,r['seconds']))['event_chunk'] if clean else 16


class Compilation:
    def __init__(self,root,event,update):
        self.root=Path(root);self.event=event;self.task=event['task'];self.update=update
        folder=condition_root(root,self.task);folder.mkdir(parents=True,exist_ok=True)
        self.file=archive(root,self.task)
        self.state=read_json(self.file) if self.file.exists() else dict(task=self.task,v=0,meta_update=update,teacher_demo=event['teacher_demo'],
            behavior_checkpoint=str(checkpoint(root,update)),history_paths=[],panels=[],proposals=[],used_steps=0,
            best_y=[0.]*64,best_successes=-1,best_panel=None,next_panel=0,query_complete=False,teaching_reads=0,frame_equivalent=0)
        if self.state['meta_update']!=update:raise ValueError('a behavior path cannot change theta during collection')
        self.video=None;self.event_cache=None

    def save(self):write_json_atomic(self.file,self.state)
    def request(self,role,y,panel):
        states=self.event['practice_states'] if role=='practice' else self.event['query_states']
        noise=self.event['practice_noise_root'] if role=='practice' else self.event['query_noise_root']
        identity=condition_id(self.task)+(f'_practice_{panel:03d}' if role=='practice' else '_query')
        value=dict(command='panel',identity=identity,request_id=identity,task=self.task,states=states,noise_root=noise,
            parameter_ref=condition_id(self.task)+f'_parameter_{panel:03d}',y=y,role=role,
            cache_root=str(self.root/'cache/H'/condition_id(self.task)),behavior_model=self.state['behavior_checkpoint'],
            meta_update=self.update,condition_id=condition_id(self.task),panel_ordinal=panel)
        self.state['pending']=value;self.save();return value

    def accept(self,output):
        request=self.state.pop('pending');rows=output['rows']
        if {r['init_state_id'] for r in rows}!=set(request['states']) or len(rows)!=4:raise ValueError('best/return needs one complete four-case panel')
        order={s:i for i,s in enumerate(request['states'])};rows.sort(key=lambda r:order[r['init_state_id']])
        if request['role']=='query':
            self.state.update(query_complete=True,query_rows=rows,R=sum(r['success'] for r in rows)/4,complete=True)
            self.save();return 'complete'
        from .practice import condition_spent
        steps=sum(r['environment_steps'] for r in rows)
        self.state['used_steps']=max(self.state['used_steps']+steps,condition_spent(self.root,self.task))
        if self.state['used_steps']>PRACTICE_STEPS:raise ValueError('condition practice exceeded its independent8192 budget')
        successes=sum(r['success'] for r in rows)
        self.state['panels'].append(dict(identity=request['identity'],y=request['y'],successes=successes,steps=steps,rows=rows))
        for row in rows:
            self.state['history_paths'].extend(str(p) for p in sorted(Path(row['history_path']).parent.glob('attempt_*.pt.gz'))
                if str(p) not in self.state['history_paths'])
        if successes>self.state['best_successes']:
            self.state.update(best_y=request['y'],best_successes=successes,best_panel=request['identity'])
        self.state['next_panel']+=1;self.save();return 'practice'

    def next(self,model,runtime,decoder,units,event_chunk):
        from .practice import model_events
        from .feedback import gaussian_log_prob
        if self.state['query_complete']:return None
        if 'pending' in self.state:return self.state['pending']
        if not self.state['panels']:return self.request('practice',[0.]*64,0)
        horizon=self.event['horizon'];remaining=PRACTICE_STEPS-self.state['used_steps']
        if self.state['best_successes']==4 or remaining<4*(horizon+10):
            y=torch.tensor(self.state['best_y'],device=runtime.device,dtype=torch.float32)
            state=decoder.decode(units*y);destination=condition_root(self.root,self.task)/'selected.safetensors'
            if not destination.exists():
                tmp=destination.with_suffix('.partial');save_file({k:v.detach().float().cpu().contiguous() for k,v in state.items()},str(tmp));tmp.replace(destination)
            self.state.update(stop_reason='all_four_success' if self.state['best_successes']==4 else 'insufficient_complete_worst_case_panel',selected_LoRA=str(destination))
            return self.request('query',self.state['best_y'],int(self.state['best_panel'].rsplit('_',1)[1]))
        if self.video is None:
            features=load_cached(Path(self.event['original']['cache_path']),runtime.device)
            if Path(features['source']['model']).resolve()!=Path(runtime.source['model_path']).resolve():
                raise ValueError('teaching prefix is not the canonical frozen source')
            with autocast(runtime.device):self.video=model.prepare_video(features['z_img'],features['indices'])
        events=model_events(runtime,self.state['history_paths'],cache_root=self.root/'cache/H'/condition_id(self.task),image_chunk=event_chunk)
        mu,self.event_cache,physical=useful_mean(model,runtime,self.video,events,self.state,event_chunk,self.event_cache)
        y0,memory=self.video
        ordinal=len(self.state['proposals']);seed=proposal_seed(self.task,ordinal)
        epsilon=torch.randn(64,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).to(runtime.device)
        y=mu+SIGMA*epsilon
        if not bool(torch.isfinite(y).all()):raise FloatingPointError('nonfinite proposal: preserve path without executing invalid factors')
        physical['resources']=host_resources()
        self.state['proposals'].append(dict(ordinal=ordinal,seed=seed,physical=physical,y=y.cpu().tolist(),mu=mu.cpu().tolist(),
            logp=float(gaussian_log_prob(y,mu)),prefix_length=len(events),history_paths=list(self.state['history_paths']),
            best_y=list(self.state['best_y']),best_score=self.state['best_successes']/4,remaining_steps=remaining,
            behavior_checkpoint=self.state['behavior_checkpoint']))
        self.state['teaching_reads']+=1;self.state['frame_equivalent']+=len(memory[0])//8
        return self.request('practice',y.cpu().tolist(),self.state['next_panel'])


@torch.no_grad()
def collect(args):
    from .feedback import FeedbackReader
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    runtime=make_runtime(args.root,context.device);saved=load(args.root,args.start)
    model=FeedbackReader().to(context.device);model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    decoder=load_basis(BASIS).to(context.device).requires_grad_(False);units=saved['code_units'].to(context.device)
    events=read_json(args.root/'batches'/f'meta_{args.start:08d}.json')['events']
    paths={e['task']:Compilation(args.root,e,args.start) for e in events};pending={}
    profile_file=args.root/'batches/H_profile.json'
    measures=read_json(profile_file)['measurements'] if profile_file.exists() else []
    chunk=physical_chunk(measures);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        emit(kind='controller_ready',meta_update=args.start,resources=host_resources())
        initial=[]
        for task,path in paths.items():
            request=path.next(model,runtime,decoder,units,chunk)
            if request:pending[request['identity']]=task;initial.append(request)
        emit(kind='panel_requests',work=initial)
        while pending:
            message=json.loads(sys.stdin.readline())
            if message['command']!='panel_done':raise RuntimeError('compiler collection interrupted; saved decisions and actual episodes preserved')
            output=message['output'];task=pending.pop(output['identity']);path=paths[task];path.accept(output)
            before=len(path.state['proposals']);chunk=physical_chunk(measures)
            request=path.next(model,runtime,decoder,units,chunk)
            if len(path.state['proposals'])>before:
                measures.append(path.state['proposals'][-1]['physical']);write_json_atomic(profile_file,dict(measurements=measures))
            if request:pending[request['identity']]=task;emit(kind='panel_request',work=request)
        total_steps=sum(p.state['used_steps']+sum(r['environment_steps'] for r in p.state['query_rows']) for p in paths.values())
        record=dict(complete=True,meta_update=args.start,conditions=list(paths),seconds=time.monotonic()-started,environment_steps=total_steps,
            event_chunk=chunk,peak_allocated_MiB=max([torch.cuda.max_memory_allocated()/2**20]+[p['physical']['peak_allocated_MiB'] for c in paths.values() for p in c.state['proposals']]),
            peak_reserved_MiB=max([torch.cuda.max_memory_reserved()/2**20]+[p['physical']['peak_reserved_MiB'] for c in paths.values() for p in c.state['proposals']]),resources=host_resources(),
            end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        from .practice import history_cache_info
        record['raw_history_cache']=history_cache_info()
        write_json_atomic(args.root/'batches'/f'collection_{args.start:08d}.json',record)
        emit(kind='collection_complete',**record)
    finally:runtime.videos.close();runtime.execution.close()
