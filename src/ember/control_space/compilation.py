"""Atomic deterministic H-conditioned task-local revision after fixed psi180."""
from pathlib import Path
import torch
from safetensors.torch import save_file
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.accounting import connect
from .contract import PRACTICE_STEPS,request_identity
from .practice import condition_spent,specifications,validate_row


def maximum_panels(horizon):
    # A nonterminal panel has at least one full-horizon scientific failure;
    # three successful episodes take at least1 control step plus settling each.
    return (PRACTICE_STEPS-4*(horizon+10))//(horizon+43)+1


class Compilation:
    def __init__(self,root,condition,decoder,units,task):
        self.root,self.condition,self.decoder,self.units,self.task=Path(root),condition,decoder,units,task
        self.id=condition['id'];self.file=self.root/'conditions'/self.id/'path.json'
        self.reused_steps=sum(r['environment_steps'] for r in condition.get('initial_rows',[]))
        if self.file.exists():
            self.state=read_json(self.file)
            if (self.state['id'],self.state['initial_y'])!=(self.id,condition['initial_y']):raise ValueError('condition restart changed its initial controller')
        else:
            self.state=dict(id=self.id,task=condition['task'],arm=condition['arm'],teacher_demo=condition.get('teacher_demo'),
                initial_y=condition['initial_y'],initial_LoRA=condition.get('initial_LoRA'),panels=[],proposals=[],proposal_cursor=0,
                incumbent_index=None,incumbent_y=condition['initial_y'],incumbent_successes=-1,reused_practice_steps=self.reused_steps,
                condition_steps=self.reused_steps,history_paths=[],decisions=[],frozen=False,query_complete=False,complete=False)
            if condition.get('initial_rows'):raise ValueError('validation initial practice must be new')
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
        identity=request_identity(self.condition,role,ordinal)
        directory=self.root/'generated'/self.id
        file=(self.condition.get('initial_LoRA') if kind=='initial' else
            str(directory/f'panel_{ordinal:03d}.safetensors'))
        if role=='query':file=self.state['selected_LoRA']
        request=dict(command='panel',identity=identity,request_id=identity,condition_id=self.id,arm=self.condition['arm'],
            task=self.condition['task'],states=self.condition['practice_states'] if role=='practice' else self.condition['query_states'],
            noise_root=self.condition['practice_noise_root'] if role=='practice' else self.condition['query_noise_root'],
            y=y,role=role,parameter_ref=file,LoRA=file,kind=kind,panel_ordinal=ordinal,ordinal=ordinal,
            proposal_ordinal=proposal_ordinal,reused_practice_steps=self.reused_steps)
        request['teacher_demo']=self.condition['teacher_demo']
        if role=='query':request['registered_scene']=self.condition['registered_scene']
        self.state['pending']=request;self.save();self.ensure_parameters(request);return request

    def next(self):
        if self.state['query_complete']:return None
        if 'pending' in self.state:self.ensure_parameters(self.state['pending']);return self.state['pending']
        if 'pending_proposal' in self.state:
            request=self.state['pending_proposal'];receipt=self.root/'proposals'/f"{request['identity']}.json"
            if receipt.exists():self.accept_proposal(read_json(receipt));return self.next()
            return request
        if not self.state['panels']:
            return self.request('practice',self.condition['initial_y'],0,'initial')
        if self.state.get('stopping_reason'):return self.freeze(self.state['stopping_reason'])
        remaining=PRACTICE_STEPS-self.spent();score=self.state['incumbent_successes']
        if score==4:return self.freeze('all_four_success')
        if remaining<4*(self.task['horizon']+10):return self.freeze('insufficient_complete_worst_case_panel')
        ordinal=len(self.state['panels']);identity=request_identity(self.condition,'practice',ordinal)
        request=dict(command='propose',identity=identity,condition_id=self.id,task=self.condition['task'],arm=self.condition['arm'],
            ordinal=ordinal,proposal_ordinal=ordinal-1,m=self.state['incumbent_y'],best_score=score/4,remaining_steps=remaining,
            history_paths=list(self.state['history_paths']),original=self.condition['original'])
        self.state['pending_proposal']=request;self.save();return request

    def accept_proposal(self,receipt):
        request=self.state['pending_proposal']
        if receipt['request']!=request:raise ValueError('actual deterministic proposal lost its saved causal input')
        y=torch.as_tensor(receipt['candidate'],dtype=torch.float32)
        if y.shape!=(64,) or not bool(y.isfinite().all()):raise FloatingPointError('nonfinite deterministic candidate cannot execute')
        y=y.tolist();repeat=next((p['ordinal'] for p in self.state['panels'] if p['y']==y),None)
        self.state['decisions'].append(dict(ordinal=request['proposal_ordinal'],y=y,center_before=request['m'],
            history_paths=request['history_paths'],best_score=request['best_score'],remaining_steps=request['remaining_steps'],
            repeated_practiced_ordinal=repeat,receipt=str(self.root/'proposals'/f"{request['identity']}.json")))
        del self.state['pending_proposal']
        if repeat is not None:
            self.state['repeated_code']=y;self.state['stopping_reason']='repeated_practiced_code';self.save();return False
        self.state['proposals'].append(dict(ordinal=request['proposal_ordinal'],y=y,center_before=request['m'],
            history_paths=request['history_paths'],best_score=request['best_score'],remaining_steps=request['remaining_steps']))
        self.state['proposal_cursor']+=1
        self.request('practice',y,request['ordinal'],'proposal',request['proposal_ordinal']);return True

    def freeze(self,reason):
        if not self.state['frozen']:
            incumbent=self.state['panels'][self.state['incumbent_index']]
            self.state.update(frozen=True,condition_steps=self.spent(),selected_LoRA=incumbent['parameter_ref'],stop_reason=reason)
            self.save()
        if (self.state['incumbent_y']==self.condition['initial_y']
                and self.state['selected_LoRA']==self.condition['initial_LoRA']):
            from .comparison import initial_query_rows
            self.state.update(query_rows=initial_query_rows(self.condition),query_reused=True,query_complete=True,complete=True)
            self.save();return None
        return self.request('query',self.state['incumbent_y'],self.state['incumbent_index'],'query')

    def accept(self,output):
        request=self.state['pending'];rows=output['rows']
        if len(rows)!=len(request['states']) or {r['init_state_id'] for r in rows}!=set(request['states']):raise ValueError('partial panel cannot change an incumbent or finish query')
        for spec,row in zip(specifications(request),sorted(rows,key=lambda r:request['states'].index(r['init_state_id'])),strict=True):
            validate_row(row,spec,self.task)
        if request['role']=='practice':self.record_panel(request,rows)
        else:self.state.update(query_rows=rows,query_reused=False,query_complete=True,complete=True)
        del self.state['pending'];self.save()
