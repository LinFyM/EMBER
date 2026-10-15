"""Real episode batching and bounded frozen raw-feature reuse contracts."""
import torch


def test_ready_wave_coalescing_preserves_each_actual_episode(monkeypatch):
    import numpy as np
    from ember.writer import practice
    from ember.pi05_eval_contract import policy_noise_seed
    class Slots:
        def __init__(self,*_):self.pending={};self.wave=0
        def submit(self,index,kind,**fields):self.pending[index]=(kind,fields)
        def receive(self,*,block=False,timeout=None):
            indices=list(self.pending)
            if self.wave==0:indices=indices[:1]
            self.wave+=1;output=[]
            for index in indices:
                kind,fields=self.pending.pop(index)
                raw=dict(images=torch.zeros(2,3,2,2),proprio=torch.zeros(8))
                row=dict(kind=kind,raw=raw,operation_seconds=0.,episode_ended=kind=='step')
                if kind=='start':row['settling_steps']=10
                else:row.update(executed=fields['actions'][:5],reward=0.,done=False,
                    row=dict(success=False,steps=5,environment_steps=15,settling_steps=10,init_state_id=index))
                output.append((index,row))
            return output
        def close(self):pass
    monkeypatch.setattr(practice,'EnvironmentSlots',Slots)
    groups=[]
    def infer(self,live,indices):
        groups.append(list(indices))
        for index in indices:
            slot=live[index];request=slot['request']
            value=policy_noise_seed(request['noise_root'],'spatial',0,request['state_id'],0)
            actions=np.full((50,7),value%13,dtype=np.float32)
            slot.update(ready=False,pending=dict(pre=slot['key'],hidden=None,snapshot=None,
                noise_seed=value,normalized_actions=torch.from_numpy(actions[:5]),
                planned_raw_actions=torch.from_numpy(actions[:5]),
                requested_actions=torch.from_numpy(actions[:5]),command_kind='model_prediction'))
            self.environments.submit(index,'step',actions=actions,noise_seed=value)
    monkeypatch.setattr(practice.EpisodeRunner,'_infer',infer)
    requests=[dict(state={'factor':torch.ones(1)},parameter_ref=f'P{i}',task=dict(suite='spatial',task_id=0),
        state_id=i,noise_root=7,episode_id=f'e{i}') for i in range(2)]
    runs=[]
    for delay in (0.,.1):
        runner=practice.EpisodeRunner(None,{},0,slots=2,coalesce_seconds=delay)
        groups.clear();outputs=list(runner.run(requests));runs.append((outputs,[*groups]))
        assert len(outputs)==2 and runner.environment_steps==30
        assert all(len(row['history'].records)==1 for row in outputs)
    assert runs[0][1]==[[0],[1]] and runs[1][1]==[[0,1]]
    originals={row['request']['episode_id']:row for row in runs[0][0]}
    for row in runs[1][0]:
        before=originals[row['request']['episode_id']]
        assert before['row']==row['row']
        assert before['history'].records[0]['noise_seed']==row['history'].records[0]['noise_seed']
        torch.testing.assert_close(before['history'].records[0]['actions'],row['history'].records[0]['actions'])


def test_current_batch_history_cache_reuses_raw_features_and_keeps_prefix_wall(monkeypatch):
    from ember.proposal_writer import learning
    from ember.writer.practice import History
    def no_decode(_):raise AssertionError('same current batch must not decode its saved H again')
    monkeypatch.setattr(learning,'load_history',no_decode)
    history=History(records=[dict(pre='a',post='b',parameter_ref='MT300'),
        dict(pre='b',post='c',parameter_ref='G')],observations={k:{} for k in 'abc'},
        episodes=[{},{}],image_features={k:torch.ones(4) for k in 'abc'},states={'MT300':{},'G':{}})
    cache=learning.HistoryCache(limit_bytes=64,max_entries=1);cache.seed('actual',history)
    early=cache.get(dict(path='actual',records=1,episodes=1))
    assert cache.cache_hits==1 and set(early.image_features)=={'a','b'} and 'G' not in early.states
    assert early.image_features['a'] is history.image_features['a']
    cache.seed('later',History(image_features={'x':torch.ones(20)}))
    assert not cache.items  # A single oversize H never becomes an unbounded worker cache.


def test_environment_slots_spawn_only_submitted_slots_with_child_thread_limits(monkeypatch):
    import os
    from ember.writer.practice import environment

    started, closed = [], []
    class Pipe:
        def send(self, command):self.command=command
        def recv(self):return dict(kind=self.command['kind'],operation_seconds=0.)
        def close(self):closed.append(self)
    class Process:
        exitcode=0
        def __init__(self, **kwargs):self.alive=False
        def start(self):
            started.append({name:os.environ.get(name) for name in environment._CHILD_THREADS})
            self.alive=True
        def join(self,timeout):self.alive=False
        def is_alive(self):return self.alive
        def close(self):closed.append(self)
    class Context:
        def Pipe(self):return Pipe(),Pipe()
    Context.Process=Process
    monkeypatch.setattr(environment.mp,'get_context',lambda method: Context() if method=='spawn' else None)
    monkeypatch.setattr(environment,'wait',lambda connections,timeout:connections)
    monkeypatch.setenv('OMP_NUM_THREADS','8')
    monkeypatch.delenv('OPENBLAS_NUM_THREADS',raising=False)
    before={name:os.environ.get(name) for name in environment._CHILD_THREADS}
    slots=environment.EnvironmentSlots({},0,8)
    assert not started
    slots.release_renderers()
    assert not started
    slots.submit(5,'start')
    assert len(started)==1 and set(started[0].values())=={'1'}
    assert {name:os.environ.get(name) for name in environment._CHILD_THREADS}==before
    assert slots.pipes[0] is None and slots.processes[5] is not None
    import pytest
    with pytest.raises(ValueError,match='operation'):slots.ensure_capacity(16)
    slots.receive(block=True)
    original_pipe,original_process=slots.pipes[5],slots.processes[5]
    from ember.writer.practice import EpisodeRunner
    runner=EpisodeRunner(None,{},0,slots=8)
    runner.environments=slots
    runner.slots=16
    assert list(runner.run([]))==[]
    slots.ensure_capacity(4)
    assert len(slots.pipes)==len(slots.processes)==16 and len(started)==1
    assert slots.pipes[5] is original_pipe and slots.processes[5] is original_process
    assert slots.pipes[15] is None
    for invalid in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='positive integer'):slots.ensure_capacity(invalid)
    slots.submit(15,'start');assert len(started)==2
    slots.receive(block=True)
    slots.close();closed_count=len(closed);slots.close()
    assert len(closed)==closed_count and all(process is None for process in slots.processes)
    with pytest.raises(ValueError,match='closed'):slots.submit(0,'start')


def test_environment_rgb_is_original_render_and_keeps_early_terminal(monkeypatch):
    from copy import deepcopy
    import numpy as np
    from ember.writer.practice import environment

    def observation(step):
        views=np.zeros((2,256,256,3),dtype=np.uint8)
        views[0,0,0]=[11,step,31]
        views[1,0,0]=[71,step,91]
        return dict(agentview_image=views[0],robot0_eye_in_hand_image=views[1],
            robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.array([0.,0.,0.,1.]),
            robot0_gripper_qpos=np.zeros(2))
    class Env:
        step_count=0
        def step(self,action):
            self.step_count+=1
            return observation(self.step_count),float(self.step_count==7),self.step_count==7,{}
    class Pool:
        def __init__(self,*args,**kwargs):self.env=Env()
        def switch(self,task):return [self.env],[]
        def close(self):pass
    contract=dict(parallel={},environment=dict(dummy_settling_steps=10,dummy_action=[0]*7,horizons={'spatial':220}))
    commands=iter([dict(kind='start',task=dict(language='move',suite='spatial'),contract=contract,
        state_id=0,noise_root=7,capture_rgb=True),
        dict(kind='step',actions=np.zeros((50,7)),noise_seed=17),
        dict(kind='step',actions=np.zeros((50,7)),noise_seed=18),dict(kind='close')])
    class Pipe:
        sent=[]
        def recv(self):return next(commands)
        def send(self,result):self.sent.append(deepcopy(result))
        def close(self):pass
    monkeypatch.setattr(environment,'PersistentTaskEnvironmentPool',Pool)
    monkeypatch.setattr(environment.torch,'set_num_threads',lambda count:None)
    monkeypatch.setattr(environment.torch,'set_num_interop_threads',lambda count:None)
    monkeypatch.setattr(environment,'capture_level',lambda *args:None)
    monkeypatch.setattr(environment,'record_passive_step',lambda *args:None)
    monkeypatch.setattr(environment,'start_fixed_episode',lambda **kwargs:dict(obs=observation(0),
        steps=0,replan_index=0,policy_noise_seeds=[],episode_done=False))
    monkeypatch.setattr(environment,'finish_episode_row',lambda **kwargs:dict(success=kwargs['slot']['episode_done']))
    pipe=Pipe();environment._serve(pipe,contract,0,[])
    assert not any('error' in result for result in pipe.sent)
    assert [result['control_step'] for result in pipe.sent]==[0,5,7]
    assert pipe.sent[-1]['episode_ended'] and pipe.sent[-1]['row']['environment_steps']==17
    for result in pipe.sent:
        assert result['rgb'].shape==(2,256,256,3) and result['rgb'].dtype==np.uint8
        assert result['rgb'][0,0,0].tolist()==[11,result['control_step'],31]
        assert result['rgb'][1,0,0].tolist()==[71,result['control_step'],91]
        # Existing policy images keep their own single official rotation.
        assert result['raw']['images'][0,:,255,255].tolist()==[11,result['control_step'],31]


def test_rgb_only_episode_capture_survives_close_and_reopen(monkeypatch):
    import numpy as np
    from ember.writer import practice
    from ember.pi05_eval_contract import policy_noise_seed

    instances=[]
    class Slots:
        def __init__(self,*args):self.pending={};self.steps=0;self.closed=False;instances.append(self)
        def submit(self,index,kind,**fields):self.pending[index]=(kind,fields)
        def receive(self,**kwargs):
            output=[]
            for index,(kind,fields) in list(self.pending.items()):
                del self.pending[index]
                if kind=='start':assert fields['capture_rgb']
                else:self.steps+=min(5,7-self.steps)
                pixels=np.full((2,256,256,3),self.steps,dtype=np.uint8)
                result=dict(kind=kind,raw=dict(images=torch.zeros(2,3,2,2),proprio=torch.zeros(8)),
                    rgb=pixels,control_step=self.steps,operation_seconds=0.,episode_ended=self.steps==7)
                if kind=='start':result['settling_steps']=10
                else:
                    count=5 if self.steps==5 else 2
                    result.update(executed=fields['actions'][:count],reward=float(self.steps==7),done=self.steps==7)
                    if self.steps==7:result['row']=dict(success=True,steps=7,environment_steps=17,settling_steps=10)
                output.append((index,result))
            return output
        def close(self):self.closed=True
    monkeypatch.setattr(practice,'EnvironmentSlots',Slots)
    seeds=[]
    def infer(self,live,indices):
        for index in indices:
            slot=live[index];request=slot['request'];actions=np.zeros((50,7),dtype=np.float32)
            noise=policy_noise_seed(request['noise_root'],'spatial',0,request['state_id'],slot['steps']//5)
            seeds.append(noise)
            slot.update(ready=False,pending=dict(pre=slot['key'],hidden=None,snapshot=None,noise_seed=noise,
                normalized_actions=torch.from_numpy(actions[:5]),planned_raw_actions=torch.from_numpy(actions[:5]),
                requested_actions=torch.from_numpy(actions[:5]),command_kind='model_prediction'))
            self.environments.submit(index,'step',actions=actions)
    monkeypatch.setattr(practice.EpisodeRunner,'_infer',infer)
    runner=practice.EpisodeRunner(None,{},0,slots=8,coalesce_seconds=0.)
    request=dict(state={'factor':torch.ones(1)},parameter_ref='source',task=dict(suite='spatial',task_id=0),
        state_id=0,noise_root=7,episode_id='rgb',capture_history=False,capture_rgb=True)
    outputs=[]
    for _ in range(2):
        outputs.append(list(runner.run([request]))[0]);runner.close();runner.close()
    assert len(instances)==2 and all(slots.closed for slots in instances)
    assert runner.environment_steps==34 and seeds[:2]==seeds[2:]
    for output in outputs:
        history=output['history']
        assert not history.observations and not history.states
        assert [frame['control_step'] for frame in history.rgb]==[0,5,7]
        assert [int(frame['images'][0,0,0,0]) for frame in history.rgb]==[0,5,7]
    merged=practice.History(episodes=[{}]);merged.append(outputs[0]['history'])
    assert {frame['episode'] for frame in merged.rgb}=={1}


def test_old_history_pickle_defaults_to_no_rgb():
    import pickle
    from ember.writer.practice import History
    original=History(episodes=[dict(success=False)])
    del original.rgb
    restored=pickle.loads(pickle.dumps(original))
    assert restored.rgb==[] and restored.episodes==original.episodes


