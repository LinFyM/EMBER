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
                    row=dict(success=False,steps=5,environment_steps=15,init_state_id=index))
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
                noise_seed=value,normalized_actions=torch.from_numpy(actions[:5])))
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


