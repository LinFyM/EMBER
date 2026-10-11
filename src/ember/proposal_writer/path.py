"""Complete bounded compilation, categorical records and stopped raw histories."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import gzip
import math
import time

import torch
from safetensors.torch import save_file

from ember.lora import validate_lora_state, LoRAContractError
from ember.writer.practice import History
from ember.writer.runtime import autocast
from ember.pi05_source_checkpoint import write_json_atomic

from .contract import seed, SELECTOR_SCHEMA, CHECKPOINT_SCHEMA
from .runtime import phase,finish_phases


def cpu_state(state):
    return {k: v.detach().float().cpu().contiguous() for k, v in state.items()}


def history_prefix(history, records, episodes):
    rows = history.records[:records]
    keys = {key for row in rows for key in (row['pre'], row['post'])}
    refs = {'MT300'} | {row['parameter_ref'] for row in rows}
    return History(records=rows, observations={k: history.observations[k] for k in keys},
        episodes=history.episodes[:episodes], image_features={k: history.image_features[k] for k in keys
                                                            if k in history.image_features},
        states={k: v for k, v in history.states.items() if k in refs})


def save_history(path, history):
    """Lossless compression bounds raw RGB storage; labels are not Reader inputs."""
    with gzip.open(path, 'wb', compresslevel=1) as f:
        torch.save(replace(history,image_features={}),f)


def load_history(path):
    with gzip.open(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=False)


@dataclass
class Compilation:
    task: int
    demo: int
    identity: str
    path_ordinal: int
    noise_root: int
    history: History
    states: dict
    practiced: list = field(default_factory=lambda: ['MT300'])
    valid: list = field(default_factory=lambda: ['MT300'])
    decisions: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    selected: str | None = None
    environment_steps: int = 0
    video_task: int | None = None
    kernel_identity: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    external_max_environment_steps: int = 0
    bootstrap_provenance: dict = field(default_factory=dict)
    sampling_identity: str = "deployment"
    completed_rounds: int = 0
    native_response_cache: dict = field(default_factory=dict)

    def manifest(self):
        return dict(task_id=self.task, teacher_demo=self.demo, video_task=self.video_task,
            identity=self.identity, path_ordinal=self.path_ordinal, noise_root=self.noise_root,
            practiced_P=self.practiced, valid_U=self.valid, selected=self.selected,
            decisions=self.decisions, attempts=self.attempts, boundaries=self.boundaries,
            environment_steps=self.environment_steps,final_locked=self.selected is not None,
            kernel_identity=self.kernel_identity,timings=self.timings,selector_schema=SELECTOR_SCHEMA,
            generator_schema=CHECKPOINT_SCHEMA,
            outputs=self.outputs,external_max_environment_steps=self.external_max_environment_steps,
            bootstrap_provenance=self.bootstrap_provenance,sampling_identity=self.sampling_identity,
            completed_rounds=self.completed_rounds)

    def save(self, root, *, retain_U=False):
        started=time.monotonic()
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        save_history(root / 'history.pt.gz', self.history)
        refs = set(self.valid if retain_U else self.practiced) | {'MT300', self.selected}
        refs |= {r['parameter_ref'] for r in self.history.records}
        for ref in refs - {'MT300', None}:
            save_file(self.states[ref], str(root / f'{ref}.safetensors'))
        self.timings['path_IO_seconds']=self.timings.get('path_IO_seconds',0.)+time.monotonic()-started
        write_json_atomic(root / 'path.json', {**self.manifest(), 'retained_U': retain_U,
            'untried_reconstruction': 'deterministic frozen ψ with exact chronological H, teaching and parent',
            'raw_history_compression': 'lossless gzip level1; frozen raw-image cache recomputed from actual RGB'})


def decision_inputs(runtime, path, decision, features):
    # Only source-frozen raw image embeddings are reused; θ encodings below remain fresh.
    history = history_prefix(path.history, decision['records'], decision['episodes'])
    rows=history.model_rows(runtime)
    path.history.image_features.update(history.image_features)
    refs = set(decision['candidates']) - {'STOP'}
    refs |= {row['parameter_ref'] for row in history.records}
    refs.add('MT300')
    states = {k: path.states[k] for k in refs}
    parent = decision.get('parent') if decision['kind'] == 'practice' else None
    if 'native_response_predictions' in decision:
        responses={key:torch.tensor(value,dtype=torch.float32) for key,value in
                   decision['native_response_predictions'].items()}
    else:
        if not hasattr(path,'native_response_cache'):path.native_response_cache={}
        responses=runtime.responses({k:states[k] for k in decision['candidates'] if k!='STOP'},
            history,runtime.tasks[path.task].authority.language,parent=parent,absolute_cache=path.native_response_cache)
        decision['native_response_predictions']={k:v.detach().cpu().tolist() for k,v in responses.items()}
        decision['prediction_source']='frozen source, complete candidate task factors, actual seen RGB and saved native noise; ten ODE calls'
    return rows, states, responses


def decision_logits(runtime, path, decision, features):
    rows, states, responses = decision_inputs(runtime, path, decision, features)
    with phase(runtime,'theta_forward'),autocast(runtime.device):
        return runtime.actor(decision['kind'], features, rows, states, decision['candidates'],
                             responses).float()


def conditional_parent_log_probability(logits,selected,candidates):
    """The complete Actor forward includes STOP; normalization alone conditions it out."""
    keep=[i for i,name in enumerate(candidates) if name!='STOP']
    if selected not in keep:raise ValueError('conditional continuation cannot sample STOP')
    return score_log_probability(logits[keep],keep.index(selected))


def categorical_record(runtime,path,kind,candidates,features,*,rng,uniform,parent=None,
                       baseline=None,conditional_continue=False,round_index=None):
    if not candidates:raise ValueError('no legal categorical choice')
    record=dict(kind=kind,candidates=list(candidates),records=len(path.history.records),
        episodes=len(path.history.episodes),parent=parent,temperature=1.,uniform_behavior=bool(uniform),
        conditional_continue=bool(conditional_continue),round_index=round_index)
    if conditional_continue and (kind!='parent' or candidates[-1]!='STOP'):
        raise ValueError('conditional collection requires the complete parent+STOP set')
    if len(candidates)==1:full=torch.ones(1)
    elif uniform:full=torch.full((len(candidates),),1/len(candidates))
    else:
        logits=decision_logits(runtime,path,record,features).detach().cpu();full=logits.softmax(-1)
    if not bool(torch.isfinite(full).all()):raise ValueError('nonfinite categorical probabilities')
    sampling=full.clone()
    if conditional_continue:
        sampling[-1]=0.
        sampling[:-1]=torch.full((len(candidates)-1,),1/(len(candidates)-1)) if uniform else logits[:-1].softmax(-1)
        record['stop_probability']=float(full[-1])
    chosen=int(torch.multinomial(sampling,1,generator=rng))
    record.update(selected=candidates[chosen],selected_index=chosen,probabilities=full.tolist(),
        sampling_probabilities=sampling.tolist(),log_probability=float(sampling[chosen].log()),
        forced=len(candidates)-(1 if conditional_continue else 0)==1)
    if baseline is not None:
        record['baseline_before_batch']=float(baseline(features,path.history.records).detach())
    path.decisions.append(record)
    return candidates[chosen]


def emit_output(runtime,path,features,*,uniform):
    ordinal=len(path.outputs)
    branch_seed=seed(33,path.noise_root,path.path_ordinal,ordinal)
    selected=categorical_record(runtime,path,'final',path.practiced,features,
        rng=torch.Generator().manual_seed(branch_seed),uniform=uniform)
    record=path.decisions[-1]
    path.outputs.append(dict(**record,environment_steps=path.environment_steps,
        attempts=len(path.attempts),branch_seed=branch_seed,decision_index=len(path.decisions)-1))
    path.selected=selected


def bootstrap(path,task,panel,*,history=None,provenance=None,snapshots=False):
    if history is None:
        state_id=panel['practice_states'][path.path_ordinal%len(panel['practice_states'])]
        request=dict(task=task,state=path.states['MT300'],parameter_ref='MT300',state_id=state_id,
            noise_root=seed(32,path.noise_root,0),episode_id=f'{path.identity}_bootstrap_MT',
            snapshots=snapshots,uses_teaching=True,remaining=path.external_max_environment_steps)
        output=yield request;history=output['history']
        path.bootstrap_provenance=dict(reused=False,actual_episode=request['episode_id'])
    else:
        if len(history.episodes)!=1 or not history.records or any(r['parameter_ref']!='MT300' for r in history.records):
            raise ValueError('bootstrap reuse requires one real complete MT-produced episode')
        path.bootstrap_provenance=dict(reused=True,source=provenance,new_environment_steps=0)
    if not history.records or history.episodes[0].get('model_failure'):
        raise ValueError('MT bootstrap did not provide a valid nonempty actual history')
    path.history.append(history)
    path.environment_steps+=sum(int(row['environment_steps']) for row in history.episodes)


def generate_candidates(runtime,path,parent,features):
    started=time.monotonic()
    proposals=[];refs={'MT300',parent}|{r['parameter_ref'] for r in path.history.records}
    states={key:path.states[key] for key in refs};rows=path.history.model_rows(runtime)
    with phase(runtime,'G_shared_context'),autocast(runtime.device):memory=runtime.generator.context(features,rows,states)
    if len(path.attempts)>=32:return proposals
    attempt=len(path.attempts);ref=f'G_{attempt:03d}';tick=time.monotonic()
    with phase(runtime,'G_candidate_endpoint'),autocast(runtime.device):
        candidate=runtime.generator.generate(path.states[parent],features,rows,states,memory=memory)
    try:
        validate_lora_state(candidate,runtime.lora)
        valid=all(bool(torch.isfinite(v).all()) for v in candidate.values())
    except LoRAContractError:valid=False
    item=dict(reference=ref,parent=parent,valid=valid,generation='deterministic_MT_centered_endpoint',
        records=len(path.history.records),episodes=len(path.history.episodes),decoder_calls=1,
        generation_and_materialization_seconds=time.monotonic()-tick)
    path.attempts.append(item)
    if not valid:raise ValueError('invalid deterministic endpoint; repeating the same condition is not recovery')
    path.states[ref]=cpu_state(candidate);path.valid.append(ref);proposals.append(ref)
    path.timings['generation_seconds']=path.timings.get('generation_seconds',0.)+time.monotonic()-started
    return proposals


def practice(path,task,panel,chosen,*,snapshots):
    n=len(path.history.episodes);state_id=panel['practice_states'][(path.path_ordinal+n)%len(panel['practice_states'])]
    request=dict(task=task,state=path.states[chosen],parameter_ref=chosen,state_id=state_id,
        noise_root=seed(32,path.noise_root,n),episode_id=f'{path.identity}_practice_{n:03d}',
        remaining=path.external_max_environment_steps-path.environment_steps,snapshots=snapshots,uses_teaching=True)
    output=yield request;row=output['row']
    path.environment_steps+=int(row['environment_steps']);path.history.append(output['history'])
    if row.get('model_failure'):
        if chosen!='MT300':
            if chosen in path.practiced:path.practiced.remove(chosen)
            if chosen in path.valid:path.valid.remove(chosen)
    elif chosen not in path.practiced:path.practiced.append(chosen)
    path.boundaries.append(dict(records=len(path.history.records),episodes=len(path.history.episodes),
        practiced=list(path.practiced),actual_parameter_ref=chosen,row=row,
        environment_steps=path.environment_steps,attempts=len(path.attempts)))
    return bool(row.get('budget_truncated'))


def begin_condition(runtime,task,demo,identity,path_ordinal,noise_root,max_rounds,
                    external_max_environment_steps,uniform,video_task,conditional_continue,student_only):
    if not runtime.psi_frozen:raise ValueError('compilation needs a fixed generation/read kernel')
    if max_rounds<1 or max_rounds>32:raise ValueError('registered depth exceeds proposal cap')
    if conditional_continue and uniform:raise ValueError('RL continuation must use actual current theta')
    cap=(max_rounds+1)*(task['horizon']+10) if external_max_environment_steps is None else int(external_max_environment_steps)
    mt=cpu_state(runtime.mt)
    identity_kind='student_refresh' if student_only else ('conditional_continue' if conditional_continue else ('uniform_offline' if uniform else 'deployment'))
    path=Compilation(task['global_task_id'],int(demo),identity,int(path_ordinal),int(noise_root),
        History(states={'MT300':mt}),{'MT300':mt},video_task=video_task,
        kernel_identity=getattr(runtime,'fixed_kernel_identity',{}),external_max_environment_steps=cap,
        sampling_identity=identity_kind,selected='MT300')
    return path,cap


def condition_steps(runtime,task,panel,demo,*,identity,path_ordinal,noise_root,
                      max_rounds,external_max_environment_steps=None,uniform=False,snapshots=False,
                      video_task=None,baseline=None,bootstrap_history=None,bootstrap_provenance=None,
                      conditional_continue=False,student_only=False):
    tick=time.monotonic()
    path,cap=begin_condition(runtime,task,demo,identity,path_ordinal,noise_root,
        max_rounds,external_max_environment_steps,uniform,video_task,conditional_continue,student_only)
    yield from bootstrap(path,task,panel,history=bootstrap_history,provenance=bootstrap_provenance,snapshots=snapshots)
    if path.history.episodes[-1].get('budget_truncated'):
        raise ValueError('bootstrap interrupted; no complete current selector state')
    features=runtime.teaching(path.task,demo,video_task=video_task)
    emit_output(runtime,path,features,uniform=uniform)
    rng=torch.Generator().manual_seed(seed(30,noise_root,path_ordinal))
    for round_index in range(max_rounds):
        if cap-path.environment_steps<11 or len(path.attempts)>=32:break
        # Full set attention must see STOP even when collecting its continuation branch.
        choices=['MT300'] if student_only else list(path.practiced)+['STOP']
        parent=categorical_record(runtime,path,'parent',choices,features,rng=rng,uniform=uniform,
            baseline=baseline,conditional_continue=(conditional_continue or uniform) and not student_only,round_index=round_index)
        if parent=='STOP':break
        proposals=generate_candidates(runtime,path,parent,features)
        if student_only and not proposals:raise ValueError('no valid actual G proposal for registered student data')
        practice_choices=[proposals[0]] if student_only else [parent,*proposals]
        chosen=categorical_record(runtime,path,'practice',practice_choices,features,parent=parent,
            rng=rng,uniform=uniform,baseline=baseline,round_index=round_index)
        truncated=yield from practice(path,task,panel,chosen,snapshots=snapshots)
        if truncated or path.history.episodes[-1].get('model_failure'):break
        path.completed_rounds+=1;emit_output(runtime,path,features,uniform=uniform)
    if conditional_continue and path.completed_rounds!=max_rounds:
        path.timings['incomplete_continuation']=True
        path.save(runtime.root/'incomplete_paths'/path.identity,retain_U=True)
        raise ValueError('actual continuation main path did not complete all registered rounds')
    if path.decisions[-1].get('selected')!='STOP':
        path.decisions.append(dict(kind='forced_exit',candidates=['STOP'],selected='STOP',forced=True,
            records=len(path.history.records),episodes=len(path.history.episodes),log_probability=0.,
            forced_reason='registered external depth/resource boundary; not a STOP label'))
    path.timings.update(compilation_residence_seconds=time.monotonic()-tick,
        new_environment_steps=path.environment_steps-(int(path.history.episodes[0]['environment_steps']) if bootstrap_history is not None else 0),
        generation_seconds=path.timings.get('generation_seconds',sum(a['generation_and_materialization_seconds'] for a in path.attempts)),
        completed_rounds=path.completed_rounds,sampling_identity=path.sampling_identity)
    return path


@torch.no_grad()
def compile_conditions(runtime,runner,specifications):
    """Interleave independent conditions in the existing persistent episode queue."""
    tick=time.monotonic();before=dict(runtime.components);env_start=runner.environment_steps
    active={i:condition_steps(runtime,**spec) for i,spec in enumerate(specifications)}
    completed={};owners={};requests=[]
    def advance(index,output=None):
        try:
            request=next(active[index]) if output is None else active[index].send(output)
        except StopIteration as done:
            completed[index]=done.value;del active[index];return []
        key=request['episode_id']
        if key in owners:raise ValueError('independent conditions require unique actual episode identities')
        owners[key]=index
        return [request]
    for index in sorted(active,key=lambda i:specifications[i]['task']['horizon'],reverse=True):
        requests.extend(advance(index))
    def receive(output):
        return advance(owners.pop(output['request']['episode_id']),output)
    for _ in runner.run(requests,on_completion=receive):pass
    if active:raise ValueError('condition driver exited with unfinished real episode dependencies')
    finish_phases(runtime)
    shared=dict(compilation_seconds=time.monotonic()-tick,new_environment_steps=runner.environment_steps-env_start,
        simultaneous_conditions=len(specifications),components_delta={k:v-before.get(k,0) for k,v in runtime.components.items()})
    paths=[completed[i] for i in range(len(specifications))]
    for path in paths:
        path.timings['shared_compilation_group']=shared
        path.timings['shared_group_members']=[p.identity for p in paths]
    return paths


def compile_condition(runtime,runner,task,panel,demo,**options):
    return compile_conditions(runtime,runner,[dict(task=task,panel=panel,demo=demo,**options)])[0]


def fixed_requests(path, task, state_ids, noise_root, *, references=None,existing_rows=()):
    if path.selected is None:
        raise ValueError('independent query cannot precede final parameter lock')
    requests = []
    arms = [(ref, ref) for ref in references] if references is not None else [('selected', path.selected), ('MT', 'MT300')]
    for arm, reference in arms:
        for state_id in state_ids:
            if (reference,state_id) in existing_rows:continue
            requests.append(dict(task=task, state=path.states[reference], parameter_ref=reference,
                state_id=state_id, noise_root=noise_root, snapshots=False, uses_teaching=False,
                episode_id=f'{path.identity}_query_{arm}_{state_id}', remaining=task['horizon'] + 10,
                arm=arm,capture_history=False,capture_hidden=False))
    return requests


def fixed_row(path,result):
    row = dict(**result['row'], condition=path.identity, teaching_demo=path.demo,
                   global_task_id=path.task, reference=result['request']['parameter_ref'],
                   arm=result['request']['arm'],
                   selected=path.selected, query_excluded_from_H=True)
    if result['history'].records and not row.get('model_failure'):
        first=result['history'].records[0]
        row.update(first_native_normalized_actions=first['normalized_actions'].tolist(),
                       first_noise_seed=first['noise_seed'],first_action_mask=first['executed'].tolist())
    return row


def evaluate_fixed(runtime, runner, path, task, state_ids, noise_root, *, references=None,existing_rows=()):
    return [fixed_row(path,result) for result in runner.run(
        fixed_requests(path,task,state_ids,noise_root,references=references,existing_rows=existing_rows))]


def restore_compilation(root, runtime):
    from safetensors.torch import load_file
    from ember.pi05_source_checkpoint import read_json
    root = Path(root); meta = read_json(root / 'path.json')
    if meta.get('parameter_lifecycle_complete') and not meta['parameter_retirement']['full_compilation_replay_available']:
        raise ValueError('consumed path parameters retired; resume the completed shared checkpoint, not old on-policy paths')
    if meta.get('selector_schema')!=SELECTOR_SCHEMA:raise ValueError('retired budget-conditioned paths are not current selector data')
    if meta.get('generator_schema')!=CHECKPOINT_SCHEMA:raise ValueError('retired stochastic generation paths are not endpoint data')
    history = load_history(root / 'history.pt.gz')
    states = dict(history.states)
    states['MT300'] = cpu_state(runtime.mt)
    states.update({file.stem: load_file(str(file)) for file in root.glob('*.safetensors')})
    refs={'MT300'}|{r['parameter_ref'] for r in history.records}
    history.states={k:states[k] for k in refs}
    path = Compilation(meta['task_id'], meta['teacher_demo'], meta['identity'], meta['path_ordinal'],
        meta['noise_root'],history,states,video_task=meta['video_task'],kernel_identity=meta.get('kernel_identity',{}),
        timings=meta.get('timings',{}),outputs=meta['outputs'],
        external_max_environment_steps=meta['external_max_environment_steps'],bootstrap_provenance=meta.get('bootstrap_provenance',{}),
        sampling_identity=meta['sampling_identity'],completed_rounds=meta['completed_rounds'])
    for attr, key in [('practiced', 'practiced_P'), ('valid', 'valid_U'), ('selected', 'selected'),
                      ('decisions', 'decisions'), ('attempts', 'attempts'), ('boundaries', 'boundaries'),
                      ('environment_steps', 'environment_steps')]:
        setattr(path, attr, meta[key])
    return path


def score_log_probability(logits, selected, *, forced=False):
    """Forced/single score is zero; complete paths sum these without length division."""
    if forced or len(logits) == 1:
        return logits.sum() * 0
    return logits.log_softmax(-1)[selected]
