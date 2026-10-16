"""The single frozen linear64 factor encoder/decoder; no learned neural path."""
import torch
from torch import nn
from ember.lora import identity_lora_state
from ember.proposal_writer.model import FactorLayout
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.reference.contract import ASSETS
from .contract import BASIS_SCHEMA


class ControlSpace(nn.Module):
    """U lives only on valid coordinates; D receives only the continuous code."""
    def __init__(self,lora,center,units,basis):
        super().__init__();self.lora=lora
        self.layout=FactorLayout(lora,identity_lora_state(lora),units)
        if center.shape!=self.layout.center.shape or basis.shape!=(self.layout.count,64):
            raise ValueError('linear64 needs the complete saved38/r8 layout and 64 basis columns')
        self.layout.center.copy_(center.float())
        self.register_buffer('basis',basis.detach().float())
        if not bool(torch.isfinite(self.basis).all()) or not bool(torch.isfinite(center).all()):
            raise ValueError('nonfinite saved center/basis')

    def normalized(self,state):
        blocks=(self.layout.pack(state).to(self.layout.center)-self.layout.center)/self.layout.scale
        return blocks[self.layout.valid]

    def encode(self,state):return self.normalized(state)@self.basis

    def decode_vectors(self,codes):return codes@self.basis.T

    def from_vector(self,vector):
        if vector.shape!=(self.layout.count,):raise ValueError('wrong complete valid factor vector')
        blocks=torch.zeros_like(self.layout.center);blocks[self.layout.valid]=vector
        return self.layout.unpack(self.layout.center+self.layout.scale*blocks)

    def decode(self,code):
        if code.shape!=(64,):raise ValueError('decoder input must be one continuous64 code')
        return self.from_vector(self.decode_vectors(code))

    def forward(self,state):return self.decode(self.encode(state))

    def forward_pack(self,states):
        vectors=torch.stack([self.normalized(state) for state in states])
        return [self.from_vector(v) for v in self.decode_vectors(vectors@self.basis)]


def load_basis(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!=BASIS_SCHEMA or state['phase']!='linear64' or state['optimizer_updates']!=0:
        raise ValueError('not the unique fresh linear64 basis schema')
    lora=derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)
    return ControlSpace(lora,state['center'],state['units'],state['U'])
