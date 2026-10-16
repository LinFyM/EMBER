"""Source-row/column addressed generation of every complete task LoRA factor."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX,LoRATarget
from .weights import KINDS,SCHEMA as INPUT_SCHEMA


def attention():return nn.MultiheadAttention(256,4,batch_first=True,dropout=0.)
def norm():return nn.LayerNorm(256,eps=1e-5)
def feedforward():return nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))


class WeightGroup(nn.Module):
    """Same-kind/geometry frozen source arrays; never included in a checkpoint."""
    def __init__(self,indices,targets):
        super().__init__();self.indices=tuple(indices);self.kind=targets[0]['kind']
        self.in_features=targets[0]['in_features'];self.out_features=targets[0]['out_features']
        self.register_buffer('rows',torch.stack([t['W']/t['rho'] for t in targets]),persistent=False)
        self.register_buffer('A0',torch.stack([t['A0'] for t in targets]),persistent=False)
        self.register_buffer('rank_inputs',torch.stack([t['A0'].T/t['rho_A'] for t in targets]),persistent=False)
        self.register_buffer('S_A',torch.tensor([t['S_A'] for t in targets],dtype=torch.float32),persistent=False)
        self.register_buffer('S_B',torch.tensor([t['S_B'] for t in targets],dtype=torch.float32),persistent=False)


class FusionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.self_norm=norm();self.memory_norm=norm();self.ff_norm=norm()
        self.self_attention=attention();self.cross_attention=attention();self.ff=feedforward()

    def forward(self,z,memory,padding):
        query=self.self_norm(z);z=z+self.self_attention(query,query,query,need_weights=False)[0]
        z=z+self.cross_attention(self.memory_norm(z),memory,memory,key_padding_mask=padding,need_weights=False)[0]
        return z+self.ff(self.ff_norm(z))


class SourceWeightCompiler(nn.Module):
    """All task dependence comes from ordered M; no task labels or reference inputs."""
    def __init__(self,inputs):
        super().__init__();targets=inputs['targets'];self.specs=tuple({k:t[k] for k in ('name','kind','in_features','out_features')} for t in targets)
        if (inputs['schema'],inputs['rank'],inputs['alpha'],inputs['dropout'],len(targets))!=(INPUT_SCHEMA,8,8,0.,38):
            raise ValueError('generator requires complete38/r8/alpha8/dropout0 source inputs')
        self.lora=SimpleNamespace(rank=8,alpha=8,dropout=0.,targets=tuple(LoRATarget(t['name'],t['in_features'],t['out_features']) for t in targets),
            parameter_count=8*sum(t['in_features']+t['out_features'] for t in targets),state_tensor_count=76)
        self.target_address=nn.Embedding(38,256);self.kind_address=nn.Embedding(4,256);self.side_address=nn.Embedding(2,256)
        self.queries=nn.Parameter(torch.empty(16,256));self.input_A=nn.ModuleDict();self.input_B=nn.ModuleDict();self.rank_input=nn.Linear(8,256,bias=False)
        groups={}
        for index,target in enumerate(targets):
            kind=target['kind'];di=target['in_features'];do=target['out_features']
            if kind not in KINDS or target['W'].shape!=(do,di) or target['A0'].shape!=(8,di):raise ValueError('source target shape/kind changed')
            groups.setdefault((kind,di,do),[]).append(index)
            if f'{kind}_{do}' not in self.input_A:self.input_A[f'{kind}_{do}']=nn.Linear(do,256,bias=False)
            if f'{kind}_{di}' not in self.input_B:self.input_B[f'{kind}_{di}']=nn.Linear(di,256,bias=False)
        self.groups=nn.ModuleList(WeightGroup(indices,[targets[i] for i in indices]) for indices in groups.values())
        self.compress_q_norm=norm();self.compress_x_norm=norm();self.compress_attention=attention()
        self.fusion=nn.ModuleList(FusionBlock() for _ in range(2))
        self.read_x_norm=norm();self.read_z_norm=norm();self.read_attention=attention()
        self.read_ff_norm=norm();self.read_ff=feedforward();self.output_norm=norm()
        self.head_A=nn.Linear(256,8);self.head_B=nn.Linear(256,8)
        for value in (self.target_address.weight,self.kind_address.weight,self.side_address.weight,self.queries):nn.init.normal_(value,std=1/16)
        for head in (self.head_A,self.head_B):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def _source(self,group):
        kind=group.kind;di=group.in_features;do=group.out_features
        indices=torch.tensor(group.indices,device=group.rows.device)
        address=self.target_address(indices)+self.kind_address.weight[KINDS.index(kind)]
        B=self.input_B[f'{kind}_{di}'](group.rows)+address[:,None]+self.side_address.weight[1]
        A=self.input_A[f'{kind}_{do}'](group.rows.transpose(1,2))+self.rank_input(group.rank_inputs)+address[:,None]+self.side_address.weight[0]
        X=torch.cat((B,A),1);Q=self.queries[None]+address[:,None]
        Z=Q+self.compress_attention(self.compress_q_norm(Q),self.compress_x_norm(X),X,need_weights=False)[0]
        return X,Z

    def _read_chunk(self,X,Z,side):
        batch,layers=Z.shape[:2];rows=X.shape[1]
        query=X[None].expand(batch,-1,-1,-1).reshape(batch*layers,rows,256)
        latent=Z.reshape(batch*layers,16,256)
        H=query+self.read_attention(self.read_x_norm(query),self.read_z_norm(latent),latent,need_weights=False)[0]
        H=H+self.read_ff(self.read_ff_norm(H))
        head=self.head_A if side=='A' else self.head_B
        return head(self.output_norm(H)).float().reshape(batch,layers,rows,8)

    def _factors(self,X,Z,group,row_chunk,checkpointed):
        sides={}
        for side,start,stop in (('B',0,group.out_features),('A',group.out_features,group.out_features+group.in_features)):
            values=[]
            for first in range(start,stop,row_chunk):
                part=X[:,first:min(first+row_chunk,stop)]
                if checkpointed:
                    # Bind side now: backward must not capture a later loop value.
                    def read(x,z,which=side):return self._read_chunk(x,z,which)
                    value=checkpoint(read,part,Z,use_reentrant=False,preserve_rng_state=False)
                else:value=self._read_chunk(part,Z,side)
                values.append(value)
            sides[side]=torch.cat(values,2)
        A=group.A0[None]+group.S_A[None,:,None,None]*sides['A'].transpose(-2,-1)
        B=group.S_B[None,:,None,None]*sides['B']
        return A,B

    def forward(self,memories,*,row_chunk=512,checkpointed=True):
        if not memories or type(row_chunk) is not int or row_chunk<1:raise ValueError('complete memories and positive physical row chunk are required')
        if any(m.ndim!=3 or m.shape[0]!=1 or m.shape[2]!=256 or m.shape[1]<1 for m in memories):raise ValueError('each ordered memory must be complete[1,L,256]')
        memory=pad_sequence([m[0] for m in memories],batch_first=True)
        lengths=torch.tensor([m.shape[1] for m in memories],device=memory.device)
        padding=torch.arange(memory.shape[1],device=memory.device)[None]>=lengths[:,None]
        checkpointed=bool(checkpointed and self.training and torch.is_grad_enabled())
        source=[];blocks=[None]*38
        for group in self.groups:
            if checkpointed:
                def project(current=group):return self._source(current)
                X,Z=checkpoint(project,use_reentrant=False,preserve_rng_state=False)
            else:X,Z=self._source(group)
            source.append(X)
            for offset,index in enumerate(group.indices):blocks[index]=Z[offset]
        Z=torch.stack(blocks).reshape(1,608,256).expand(len(memories),-1,-1)
        for block in self.fusion:
            Z=checkpoint(block,Z,memory,padding,use_reentrant=False,preserve_rng_state=False) if checkpointed else block(Z,memory,padding)
        Z=Z.reshape(len(memories),38,16,256);states=[{} for _ in memories]
        for group,X in zip(self.groups,source,strict=True):
            latent=Z[:,list(group.indices)]
            A,B=self._factors(X,latent,group,row_chunk,checkpointed)
            for local,index in enumerate(group.indices):
                name=self.specs[index]['name']
                for sample,state in enumerate(states):state[name+LORA_A_SUFFIX]=A[sample,local];state[name+LORA_B_SUFFIX]=B[sample,local]
        return states
