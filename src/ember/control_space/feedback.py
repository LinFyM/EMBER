"""Fresh causal experience revision in the fixed64 control coordinates."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ember.proposal_writer.model import attention,positions

EVENT_SHAPES=dict(images=(2,512,2048),hidden=(2,50,1024),facts=(63,),y=(64,))


class RevisionReader(nn.Module):
    """Only trainable experience/decision operators; teaching memory is frozen."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.image_norm,self.hidden_norm=nn.LayerNorm(2048),nn.LayerNorm(1024)
            self.image,self.hidden=nn.Linear(2048,256),nn.Linear(1024,256)
            self.spatial,self.pre_post=nn.Embedding(512,256),nn.Embedding(2,256)
            self.flow_call,self.action_position=nn.Embedding(2,256),nn.Embedding(50,256)
            self.parameter,self.facts=nn.Linear(64,256),nn.Linear(63,256)
            self.event_query=nn.Parameter(torch.randn(1,256)/16)
            self.event_attention,self.video_attention=attention(),attention()
            layer=nn.TransformerEncoderLayer(256,4,1024,dropout=0.,batch_first=True,norm_first=True)
            self.history_time=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.decision_query=nn.Parameter(torch.randn(1,256)/16)
            self.decision=nn.Linear(66,256)
            self.decision_attention,self.decision_norm=attention(),nn.LayerNorm(256)
            self.delta_head=nn.Linear(256,64)
            nn.init.zeros_(self.delta_head.weight);nn.init.zeros_(self.delta_head.bias)

    def _event_chunk(self,events,memory,first):
        # CPU evidence enters the device inside the replayed chunk, never all H.
        device=self.image.weight.device
        def pack(name):
            value=torch.stack([event[name] for event in events]).to(device)
            return value if torch.is_autocast_enabled(device.type) else value.to(self.image.weight.dtype)
        images=self.image(self.image_norm(pack('images')))
        images=images+self.spatial.weight[None,None]+self.pre_post.weight[None,:,None]
        hidden=self.hidden(self.hidden_norm(pack('hidden')))
        hidden=hidden+self.flow_call.weight[None,:,None]+self.action_position.weight[None,None]
        tokens=torch.cat((images.flatten(1,2),hidden.flatten(1,2),self.parameter(pack('y'))[:,None]),1)
        ordinal=torch.arange(first,first+len(events),device=device)
        query=self.event_query[None]+self.facts(pack('facts'))[:,None]+positions(ordinal)[:,None]
        event=query+self.event_attention(query,tokens,tokens,need_weights=False)[0]
        teaching=memory.expand(len(events),-1,-1)
        event=event+self.video_attention(event,teaching,teaching,need_weights=False)[0]
        return event[:,0][None]

    def event_memory(self,events,memory,event_chunk=16,checkpointed=True,first_ordinal=0):
        if memory.ndim!=3 or memory.shape[0]!=1 or memory.shape[2]!=256 or not memory.shape[1]:
            raise ValueError('revision requires the complete frozen teaching memory[1,T*8,256]')
        if type(event_chunk) is not int or event_chunk<=0:raise ValueError('event_chunk must be positive')
        for event in events:
            if any(not isinstance(event[name],torch.Tensor) or event[name].shape!=shape
                    or not event[name].is_floating_point() for name,shape in EVENT_SHAPES.items()):
                raise ValueError('real H requires full image/hidden/facts63/behavior64 events')
        memory=memory.to(self.image.weight.device)
        if not torch.is_autocast_enabled(memory.device.type):memory=memory.to(self.image.weight.dtype)
        if not events:return memory.new_empty((1,0,256))
        chunks=[]
        for first in range(0,len(events),event_chunk):
            chunk=events[first:first+event_chunk]
            if checkpointed and torch.is_grad_enabled():
                value=checkpoint(self._event_chunk,chunk,memory,first+first_ordinal,use_reentrant=False,preserve_rng_state=False)
            else:value=self._event_chunk(chunk,memory,first+first_ordinal)
            chunks.append(value)
        return torch.cat(chunks,1)

    def causal_history(self,encoded,checkpointed=True):
        count=encoded.shape[1]
        if not count:return encoded
        mask=torch.ones(count,count,dtype=torch.bool,device=encoded.device).triu(1)
        if checkpointed and torch.is_grad_enabled():
            for layer in self.history_time.layers:
                encoded=checkpoint(layer,encoded,src_mask=mask,is_causal=True,use_reentrant=False,preserve_rng_state=False)
            return encoded
        return self.history_time(encoded,mask=mask,is_causal=True)

    def encode_events(self,events,memory,event_chunk=16,checkpointed=True):
        return self.causal_history(self.event_memory(events,memory,event_chunk,checkpointed),checkpointed)

    def candidate_from_encoded(self,memory,encoded,m,score_fraction,remaining,prefix_length=None):
        prefix_length=encoded.shape[1] if prefix_length is None else prefix_length
        if (m.shape!=(64,) or encoded.ndim!=3 or encoded.shape[0]!=1 or encoded.shape[2]!=256
                or memory.ndim!=3 or memory.shape[0]!=1 or memory.shape[2]!=256
                or type(prefix_length) is not int or not 0<=prefix_length<=encoded.shape[1]):
            raise ValueError('revision needs full incumbent64 and a legal causal decision prefix')
        device=self.decision.weight.device;m=m.to(device=device,dtype=torch.float32)
        if not prefix_length:return m
        facts=torch.cat((m,torch.as_tensor([score_fraction,remaining/8192],device=device,dtype=torch.float32)))
        query=self.decision_query[None]+self.decision(facts)[None,None]
        visible=encoded[:,:prefix_length]
        decision=query+self.decision_attention(query,visible,visible,need_weights=False)[0]
        delta=self.delta_head(self.decision_norm(decision))[0,0]
        return m+delta.float()

    def propose(self,memory,events,m,score_fraction,remaining,event_chunk=16,checkpointed=True):
        encoded=self.encode_events(events,memory,event_chunk,checkpointed)
        return self.candidate_from_encoded(memory,encoded,m,score_fraction,remaining)
