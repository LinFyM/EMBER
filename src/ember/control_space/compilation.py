"""Atomic task-local incumbent updates; no shared learner or semantic H reader."""
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.accounting import connect
from .contract import PRACTICE_STEPS,SIGMA
from .practice import condition_spent,specifications,validate_row


def proposal_seed(task,ordinal):
    return int(np.random.SeedSequence([7,int(task),0,int(ordinal),202610161]).generate_state(1,dtype=np.uint64)[0]%(2**63-1))


def maximum_panels(horizon):
    # A nonterminal panel has at least one full-horizon scientific failure;
    # three successful episodes take at least1 control step plus settling each.
    return (PRACTICE_STEPS-4*(horizon+10))//(horizon+43)+1


class Compilation:
    def __init__(self,root,condition,decoder,units,task):
        self.root,self.condition,self.decoder,self.units,self.task=Path(root),condition,decoder,units,task
        self.id=condition['id'];self.file=self.root/'conditions'/self.id/'path.json'
        self.reused_steps=sum(r['environment_steps'] for r in condition.get('source_initial_rows',[]))
        if self.file.exists():
            self.state=read_json(self.file)
            if (self.state['id'],self.state['initial_y'])!=(self.id,condition['initial_y']):raise ValueError('condition restart changed its initial controller')
        else:
            self.state=dict(id=self.id,task=condition['task'],arm=condition['arm'],teacher_demo=condition.get('teacher_demo'),
                initial_y=condition['initial_y'],initial_LoRA=condition.get('initial_LoRA'),panels=[],proposals=[],proposal_cursor=0,
                incumbent_index=None,incumbent_y=condition['initial_y'],incumbent_successes=-1,reused_practice_steps=self.reused_steps,
                condition_steps=self.reused_steps,history_paths=[],frozen=False,query_complete=False,complete=False)
            if condition.get('source_initial_rows'):
                rows=condition['source_initial_rows']
                self.record_panel(dict(identity=self.id+'_practice_000',kind='initial',y=[0.]*64,parameter_ref='raw_source_identity',
                    ordinal=0,proposal_ordinal=None),rows,reused=True)
            self.save()

    def save(self):write_json_atomic(self.file,self.state)
    def spent(self):return condition_spent(self.root,self.id,self.reused_steps)

    def record_panel(self,request,rows,*,reused=False):
        states=self.condition['practice_states']
        if len(rows)!=4 or {r['init_state_id'] for r in rows}!=set(states):raise ValueError('a decision requires all four registered practice receipts')
        rows=sorted(rows,key=lambda r:states.index(r['init_state_id']));score=sum(r['success'] for r in rows)
        accepted=score>self.state['incumbent_successes'];index=len(self.state['panels'])
        panel=dict(ordinal=index,identity=request['identity'],kind=request['kind'],y=request['y'],
            parameter_ref=request['parameter_ref'],rows=rows,history_paths=[r['history_path'] for r in rows],
            successes=score,steps=sum(r['environment_steps'] for r in rows),proposal_ordinal=request['proposal_ordinal'],accepted=accepted,reused=reused)
        self.state['panels'].append(panel);self.state['history_paths'].extend(panel['history_paths'])
        if accepted:self.state.update(incumbent_index=index,incumbent_y=request['y'],incumbent_successes=score)
        self.state['condition_steps']=self.spent()
        if self.state['condition_steps']>PRACTICE_STEPS:raise RuntimeError('condition8192 exceeded; preserve all original evidence')

    def ensure_parameters(self,request):
        destination=Path(request['parameter_ref'])
        if destination.is_file():return
        if request['kind']=='initial' or request['role']=='query':raise FileNotFoundError('actual initial/practiced controller is missing')
        with connect(self.root) as db:
            prefix=request['role']+'_'+request['identity']+'_state_'
            count=db.execute('SELECT count(*) FROM attempts WHERE substr(case_id,1,?)=?',(len(prefix),prefix)).fetchone()[0]
        if count:raise RuntimeError('submitted controller lost its actual saved factors; do not silently rematerialize')
        state=self.decoder.decode(self.units*torch.tensor(request['y'],dtype=torch.float32))
        if any(not bool(v.isfinite().all()) for v in state.values()):raise FloatingPointError('nonfinite proposed controller')
        destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.partial')
        save_file({k:v.detach().float().contiguous() for k,v in state.items()},str(temporary));temporary.replace(destination)

    def request(self,role,y,ordinal,kind,proposal_ordinal=None):
        identity=self.id+(f'_practice_{ordinal:03d}' if role=='practice' else '_query')
        file=(self.condition.get('initial_LoRA') if kind=='initial' else
            str(self.root/'generated'/self.id/f'panel_{ordinal:03d}.safetensors'))
        if role=='query':file=self.state['selected_LoRA']
        request=dict(command='panel',identity=identity,request_id=identity,condition_id=self.id,arm=self.condition['arm'],
            task=self.condition['task'],states=self.condition['practice_states'] if role=='practice' else self.condition['query_states'],
            noise_root=self.condition['practice_noise_root'] if role=='practice' else self.condition['query_noise_root'],
            y=y,role=role,parameter_ref=file,LoRA=file,kind=kind,panel_ordinal=ordinal,ordinal=ordinal,
            proposal_ordinal=proposal_ordinal,reused_practice_steps=self.reused_steps)
        self.state['pending']=request;self.save();self.ensure_parameters(request);return request

    def next(self):
        if self.state['query_complete']:return None
        if 'pending' in self.state:self.ensure_parameters(self.state['pending']);return self.state['pending']
        if not self.state['panels']:return self.request('practice',self.condition['initial_y'],0,'initial')
        remaining=PRACTICE_STEPS-self.spent();score=self.state['incumbent_successes']
        if self.state['frozen'] or score==4 or remaining<4*(self.task['horizon']+10):return self.freeze()
        ordinal=self.state['proposal_cursor'];seed=proposal_seed(self.condition['task'],ordinal)
        epsilon=torch.randn(64,generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
        center=torch.tensor(self.state['incumbent_y'],dtype=torch.float32);y=(center+SIGMA*epsilon).tolist()
        self.state['proposals'].append(dict(ordinal=ordinal,seed=seed,epsilon=epsilon.tolist(),center_before=center.tolist(),y=y))
        self.state['proposal_cursor']+=1
        return self.request('practice',y,len(self.state['panels']),'proposal',ordinal)

    def freeze(self):
        if not self.state['frozen']:
            incumbent=self.state['panels'][self.state['incumbent_index']]
            self.state.update(frozen=True,condition_steps=self.spent(),selected_LoRA=(None if self.condition['arm']=='no_video' and
                self.state['incumbent_y']==[0.]*64 else incumbent['parameter_ref']),
                stop_reason='all_four_success' if self.state['incumbent_successes']==4 else 'insufficient_complete_worst_case_panel')
            self.save()
        if self.state['incumbent_y']==self.condition['initial_y']:
            from .comparison import initial_query_rows
            self.state.update(query_rows=initial_query_rows(self.condition['task'],self.condition['arm']),query_reused=True,query_complete=True,complete=True)
            self.save();return None
        return self.request('query',self.state['incumbent_y'],self.state['incumbent_index'],'query')

    def accept(self,output):
        request=self.state['pending'];rows=output['rows']
        if len(rows)!=4 or {r['init_state_id'] for r in rows}!=set(request['states']):raise ValueError('partial panel cannot change an incumbent or finish query')
        for spec,row in zip(specifications(request),sorted(rows,key=lambda r:request['states'].index(r['init_state_id'])),strict=True):
            validate_row(row,spec,self.task)
        if request['role']=='practice':self.record_panel(request,rows)
        else:self.state.update(query_rows=rows,query_reused=False,query_complete=True,complete=True)
        del self.state['pending'];self.save()
