"""Atomic endpoint compilation, then read-only paired seen/held evidence."""
from pathlib import Path
from itertools import combinations
import datetime,json,os,subprocess,time
import torch
from safetensors.torch import load_file,save_file
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.lora import validate_lora_state
from ember.reference.analysis import paired
from ember.proposal_writer.analysis import paired as checked_pair,billing
from ember.reference.accounting import summary,connect
from .contract import (SCHEMA,PHASE,FINAL,TASKS,VALIDATION_TASKS,VIDEO_ROOT,REFERENCE_ROOT,
    INITIAL_READER,ASSETS,SOURCE,NORMALIZATION,checkpoint_path)
from .analysis import grouped


def generated(root,role,panel):
    return Path(root)/'generated'/role/f'task_{panel["task"]:04d}'/f'state_{panel["init_state_id"]:03d}'


def generation_identity(root,role,panel):
    return dict(schema=SCHEMA,phase=PHASE,checkpoint=str(checkpoint_path(root,FINAL)),update=FINAL,
        source_checkpoint=str(ASSETS/SOURCE),normalization=str(ASSETS/NORMALIZATION),frozen_reader=str(INITIAL_READER),
        source_inputs=str(Path(root)/'source_inputs/state.pt'),role=role,task=panel['task'],init_state_id=panel['init_state_id'],
        teacher_demo=panel['teacher_demo'],noise_root=panel['noise_root'],registered_scene=panel.get('registered_scene'),
        rank=8,alpha=8,dropout=0.,targets=38,complete=True,held_offline_labels_read=False,
        teacher_input='complete_frozen_ordered_image_memory',interaction_adaptation=False,MT_loaded=False,
        path=str(generated(root,role,panel)/'lora.safetensors'))


def validate_generated(root,role,panel,*,factors=False):
    from .weights import topology
    folder=generated(root,role,panel);meta=read_json(folder/'metadata.json');identity=generation_identity(root,role,panel)
    if any(meta.get(k)!=v for k,v in identity.items()) or not (folder/'lora.safetensors').is_file():
        raise ValueError('saved per-video controller lost its endpoint/input identity')
    if factors:
        state=load_file(str(folder/'lora.safetensors'),device='cpu');validate_lora_state(state,topology())
        if any(not bool(v.isfinite().all()) for v in state.values()):raise ValueError('nonfinite complete generated LoRA')
    return meta


class Materializer:
    """Lazy generator/M owner. Only complete1180 and actual saved LoRAs are adopted."""
    def __init__(self,root,device,runtime,*,row_chunk=4096,pack=2,frame_chunk=32):
        self.root,self.device,self.runtime=Path(root),torch.device(device),runtime
        self.source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        self.row_chunk,self.pack,self.frame_chunk=int(row_chunk),int(pack),int(frame_chunk)
        if min(self.row_chunk,self.pack,self.frame_chunk)<1:raise ValueError('physical generation sizes must be positive')
        self.model,self.memory=None,None

    def _ensure(self):
        if self.model is not None:return
        from .model import SourceWeightCompiler
        from .weights import load_source_inputs
        from .video import MemoryStore
        folder=checkpoint_path(self.root,FINAL).parent;manifest=read_json(folder/'manifest.json')
        state=torch.load(folder/'state.pt',map_location='cpu',weights_only=False,mmap=True)
        expected=dict(schema=SCHEMA,phase=PHASE,next_update=FINAL,source_inputs=str(self.root/'source_inputs/state.pt'),
            frozen_reader=str(INITIAL_READER),frozen_source=str(ASSETS/SOURCE),normalization=str(ASSETS/NORMALIZATION))
        if (not manifest['complete'] or manifest['next_update']!=FINAL or any(state.get(k)!=v for k,v in expected.items())):
            raise ValueError('only the frozen complete new-generator1180 may compile')
        with torch.random.fork_rng(devices=[]):self.model=SourceWeightCompiler(load_source_inputs(self.root))
        self.model.load_state_dict(state['generator'],strict=True);del state
        self.model=self.model.to(self.device).eval().requires_grad_(False)
        self.memory=MemoryStore(self.root,self.device,self.runtime,frame_chunk=self.frame_chunk)

    def _adopt(self,role,panel):
        destination=generated(self.root,role,panel);temporary=destination.with_name(destination.name+'.partial')
        if destination.exists():validate_generated(self.root,role,panel,factors=True);return True
        if not temporary.exists():return False
        identity=generation_identity(self.root,role,panel)
        if not (temporary/'intent.json').exists():
            if (temporary/'lora.safetensors').exists() or (temporary/'metadata.json').exists():
                raise ValueError('pending complete controller has no attributable immutable intent')
            return False
        if read_json(temporary/'intent.json')!=identity:raise ValueError('partial compiler transaction belongs to another condition')
        if not (temporary/'lora.safetensors').exists():return False
        from .weights import topology
        state=load_file(str(temporary/'lora.safetensors'),device='cpu');validate_lora_state(state,topology())
        if any(not bool(v.isfinite().all()) for v in state.values()):raise ValueError('invalid pending complete controller')
        if not (temporary/'metadata.json').exists():
            write_json_atomic(temporary/'metadata.json',dict(identity,recovered_after_parameter_write=True))
        elif any(read_json(temporary/'metadata.json').get(k)!=v for k,v in identity.items()):
            raise ValueError('pending complete metadata lost its immutable compiler identity')
        temporary.rename(destination);validate_generated(self.root,role,panel);return True

    @torch.no_grad()
    def compile(self,role,panels):
        from ember.writer.runtime import autocast
        tick=time.monotonic();missing=[p for p in panels if not self._adopt(role,p)];measurements=[]
        if missing:self._ensure()
        for first in range(0,len(missing),self.pack):
            part=missing[first:first+self.pack];memories=[]
            for panel in part:
                M=(self.memory.training(panel['task'],panel['teacher_demo']) if role=='seen'
                    else self.memory.held(panel['task'],panel['init_state_id'],panel))
                memories.append(M.to(self.device))
            measured=self.memory.drain_measurements();measurements.extend(measured)
            folder=self.root/'materialization';folder.mkdir(exist_ok=True)
            with (folder/f'memory_{os.uname().nodename}_{os.getpid()}.jsonl').open('a') as stream:
                for measure in measured:stream.write(json.dumps(dict(role=role,**measure))+'\n')
            with autocast(self.device):states=self.model(memories,row_chunk=self.row_chunk,checkpointed=False)
            if len(states)!=len(part):raise ValueError('each video requires its own complete generated controller')
            for panel,state in zip(part,states,strict=True):
                destination=generated(self.root,role,panel);temporary=destination.with_name(destination.name+'.partial')
                temporary.mkdir(parents=True,exist_ok=True);identity=generation_identity(self.root,role,panel)
                if (temporary/'intent.json').exists() and read_json(temporary/'intent.json')!=identity:raise ValueError('pending generation identity changed')
                write_json_atomic(temporary/'intent.json',identity);validate_lora_state(state,self.model.lora)
                if any(not bool(v.isfinite().all()) for v in state.values()):raise ValueError('nonfinite generator output cannot execute')
                payload=temporary/'lora.safetensors.partial'
                save_file({k:v.detach().float().cpu().contiguous() for k,v in state.items()},str(payload))
                with payload.open('rb') as stream:os.fsync(stream.fileno())
                payload.replace(temporary/'lora.safetensors')
                write_json_atomic(temporary/'metadata.json',dict(identity,materialization_source_commit=self.source_commit,
                    generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                temporary.rename(destination)
            del states,memories
        receipt=dict(new_controllers=len(missing),adopted_controllers=len(panels)-len(missing),seconds=time.monotonic()-tick,
            materialize_pack=self.pack,row_chunk=self.row_chunk,frame_chunk=self.frame_chunk,
            memory=measurements)
        return [validate_generated(self.root,role,p) for p in panels],receipt

    def close(self):
        if self.memory is not None:self.memory.close()
        self.memory,self.model=None,None


def _index(rows,panels):
    keys={(p['task'],p['init_state_id']) for p in panels};result={}
    suites=('libero_spatial','libero_object','libero_goal','libero_10')
    for original in rows:
        row=dict(original);task=row.get('global_task_id')
        if task is None:task=10*suites.index(row['suite'])+int(row['task_id'])
        row['global_task_id']=int(task)  # Derived suite/local identity; never fabricate observations.
        key=int(task),int(row['init_state_id'])
        if key in result:raise ValueError('duplicate historical paired case')
        result[key]=row
    if set(result)!=keys:raise ValueError('historical reference is not the entire registered panel')
    return [result[p['task'],p['init_state_id']] for p in panels]


def case_task(row):
    if 'global_task_id' in row:return int(row['global_task_id'])
    return 10*('libero_spatial','libero_object','libero_goal','libero_10').index(row['suite'])+int(row['task_id'])


def historical(root,panels):
    """Actual raw rows, opened only after training/final consumers; never conditions."""
    run=read_json(Path(root)/'run_contract.json');bank={r['task']:r for r in run['references']}
    video=read_json(VIDEO_ROOT/'analysis/results.json');seen={a:[] for a in ('source','reference','U36')}
    linear=REFERENCE_ROOT.parent/'control_linear64_seed7'
    for p in panels['seen']:
        task,init=p['task'],p['init_state_id'];base=REFERENCE_ROOT/'selection'/f'task_{task:04d}'
        for arm,folder in (('source',base/'update_00000000'),('reference',base/f'update_{bank[task]["selected_update"]:08d}'),('U36',linear/'readout'/f'task_{task:04d}')):
            seen[arm].append(read_json(folder/f'state_{init:03d}.json'))
    for update in ('900','1800'):seen['Reader'+update]=video['checkpoints'][update]['raw_rows']['seen']
    paths=video['validation_references'];validation={}
    for update in ('900','1800'):validation['Reader'+update]=video['checkpoints'][update]['raw_rows']['validation']
    validation['MT153']=read_json(Path(paths['MT153']))['rows'];validation['G165']=read_json(Path(paths['G165']))['raw_rows']
    for arm in ('D72','D144'):validation[arm]=read_json(Path(paths[arm]))['arms']['operator_only']['validation']['raw_rows']
    psi_path=REFERENCE_ROOT.parent/'functional_feedback_validation180_seed7/analysis/results.json'
    validation['psi180']=read_json(psi_path)['raw_rows']['functional_feedback_validation180']
    result={role:{a:_index(v,panels[role]) for a,v in arms.items()} for role,arms in (('seen',seen),('validation',validation))}
    for role,expected in (('seen',dict(source=218,reference=502,U36=484,Reader900=334,Reader1800=413)),
            ('validation',dict(Reader900=117,Reader1800=111,MT153=153,G165=165,D72=175,D144=168,psi180=119))):
        if {a:sum(r['success'] for r in v) for a,v in result[role].items()}!=expected:raise ValueError('actual historical full-panel identity/count changed')
        from ember.pi05_eval.scene import validate_scene_row
        for arm,rows in result[role].items():
            for row,panel in zip(rows,panels[role],strict=True):
                validate_scene_row(row,row,dict(operator_read_write_scene=panel.get('registered_scene')))
                if arm.startswith('Reader') and row.get('teacher_demo')!=panel['teacher_demo']:
                    raise ValueError('historical Reader teaching differs from the same-condition panel')
    return result,dict(video=str(VIDEO_ROOT/'analysis/results.json'),reference=str(REFERENCE_ROOT),linear=str(linear),
        validation=paths,psi180=str(psi_path),MT_missing_initial_proprio='registered scene/RNG checked; no fabricated proprio',
        D_G='historical MT-based methods, not source-only matched ablations',psi='uses actual practice, not a matched zero-interaction ablation')


def grouped_evidence(arms,tasks,panels,identity):
    rows=[]
    for task in sorted(tasks):
        values={a:[r for r in cases if case_task(r)==task]
            for a,cases in arms.items()}
        sets={a:{r['init_state_id'] for r in v if r['success']} for a,v in values.items()};comparisons={}
        for a,b in combinations(values,2):
            checked=checked_pair(values[a],values[b]);item=paired(values[a],values[b]);union=sets[a]|sets[b]
            item.update(success_set_union=sorted(union),jaccard=len(sets[a]&sets[b])/len(union) if union else 1.,
                first_native_first5_RMSE=checked['first_native_first5_RMSE'])
            comparisons[a+'_to_'+b]=item
        group='validation8' if task in VALIDATION_TASKS else ('target24' if task<40 else 'source_seen_support12')
        item=dict(task=task,suite=tasks[task]['suite'],group=group,identity=task in identity,
            cases=len(values[next(iter(values))]),success_sets={a:sorted(v) for a,v in sets.items()},comparisons=comparisons)
        if 'reference' in sets:
            gains=sets['reference']-sets['source'];item.update(reference_new_gains=sorted(gains),reference_gains_preserved={a:sorted(v&gains) for a,v in sets.items()})
        rows.append(item)
    return grouped(rows)


def interaction_evidence(root,raw,panels,limits):
    """Every final row must own one complete physical receipt; unknowns stay charged."""
    with connect(root) as db:
        ledger={r[0]:r[1:] for r in db.execute('SELECT identity,case_id,status,control,dummy,row_path FROM attempts')}
    expected=set()
    from .readout import row_path
    for role in ('seen','validation'):
        for row,panel in zip(raw[role]['source_weight1180'],panels[role],strict=True):
            case=f'{role}_t{panel["task"]:04d}_s{panel["init_state_id"]:03d}';expected.add(case)
            actual=ledger.get(row['attempt_id'])
            wanted=(case,'complete',row['steps'],10,str(row_path(root,role,panel)))
            if actual!=wanted:raise ValueError('final valid row is not bound to its complete actual-cost ledger receipt')
    complete=[r for r in ledger.values() if r[1]=='complete']
    if len(complete)!=976 or {r[0] for r in complete}!=expected or any(r[0] not in expected for r in ledger.values()):
        raise ValueError('only976 unique complete final cases may finish')
    evidence=summary(root);control=sum(v['control'] for v in evidence.values());dummy=sum(v['dummy'] for v in evidence.values())
    physical=len(ledger);extra=physical-len({r[0] for r in ledger.values()})
    if (physical>limits['physical_submissions'] or control>limits['control_steps'] or dummy>limits['dummy_steps']
            or control+dummy>limits['total_steps'] or extra>limits['engineering_extra_attempts']):
        raise ValueError('actual and conservatively unknown final costs exceed the registered limits')
    return evidence


def paid_dependencies():
    source_root=(ASSETS/SOURCE).parent.parent;source_summary=read_json(source_root/'run_summary.json')
    reference=read_json(REFERENCE_ROOT/'analysis/results.json')['billing']
    video=read_json(VIDEO_ROOT/'analysis/results.json')['billing']
    return dict(shared_paid_source=str(ASSETS/SOURCE),source_summary=str(source_root/'run_summary.json'),
        source_training_only_GPU_hours_lower_bound=4*source_summary['wall_seconds']/3600,
        source_cost_scope='original4 ranks through last update; excludes failed final save/recovery and earlier preparation',
        shared_paid_reference=str(REFERENCE_ROOT),reference_family_billing=reference,
        shared_paid_Reader900=str(INITIAL_READER),Reader_family_billing=video,
        source_center_and_units='old basis saved identity/layout and control_space76S only; no old U/code/neural decoder forward',
        costs_are_not_from_zero='current-root billing excludes prior source/reference/Reader learning and old failures; do not add shared costs twice')


def analyze(root):
    from .readout import inputs,saved_case
    root=Path(root);run,tasks,panels=inputs(root);old,provenance=historical(root,panels);results={};raw={}
    identity={r['task'] for r in run['references'] if r['path'] is None}
    for role in ('seen','validation'):
        new=[saved_case(root,role,tasks[role][p['task']],p) for p in panels[role]]
        raw[role]=dict(old[role],source_weight1180=new)
        results[role]=grouped_evidence(raw[role],tasks[role],panels[role],identity)
    interactions=interaction_evidence(root,raw,panels,run['limits'])
    if len(raw['seen']['source_weight1180'])!=576 or len(raw['validation']['source_weight1180'])!=400:
        raise ValueError('only complete976 and the original finite physical budget can finish')
    result=dict(schema=SCHEMA,phase=PHASE,complete=True,checkpoint=str(checkpoint_path(root,FINAL)),update=FINAL,
        panels=results,raw_rows=raw,provenance=provenance,billing=billing(root),interactions=interactions,limits=run['limits'],
        new_episodes=976,zero_interaction=True,shared_updates=1180,held_offline_labels_read=False,Test=False,
        selection='unique preregistered1180; no mid-window/partial-held selection',automatic_successor=False,
        dependencies=paid_dependencies())
    write_json_atomic(root/'analysis/results.json',result);write_json_atomic(root/'completion.json',dict(complete=True,rows=976,result=str(root/'analysis/results.json')))
    lines=['# Source权重条件完整LoRA：唯一1180读出','',
        'seen576：'+str(results['seen']['total']['successes']),'','strict validation400：'+str(results['validation']['total']['successes']),'',
        '完整raw rows、每task/suite/target-support/identity、RGL/churn/Jaccard及参考新增能力保留见results.json。',
        '旧D/G含MT，ψ含实践；它们是历史强参照，不是匹配零交互机制消融。MT缺原初始proprio不补造。',
        '单点不能认证稳定最终能力；0中间选点/额外forward探针/controls/Test/自动续训。']
    (root/'analysis/report.md').write_text('\n'.join(lines)+'\n');return result
