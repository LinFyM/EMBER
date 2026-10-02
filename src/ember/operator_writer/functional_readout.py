"""Saved A28 sample and frozen official FM consumers shared by registered readouts."""
from __future__ import annotations
from pathlib import Path
import torch
from ember.lora import validate_lora_state
from ember.operator_writer import joint_readout as readout
from ember.writer.function_credit import FlowSample, NativeFlowPrediction
from ember.writer.runtime import autocast
from contextlib import contextmanager
from collections import defaultdict
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_

def fixed_flow(task: int, panel: dict) -> tuple[dict, Path]:
    path = readout.FIXED_PANEL / f"task{task:03d}_A_query_flow_target.pt"
    flow = torch.load(path, map_location="cpu", weights_only=False)
    shapes = {"action": (28, 50, 7), "noise": (28, 50, 32),
              "time": (28,), "FM_target": (28, 50, 32)}
    if (tuple(panel["teachers"]) != readout.TEACHERS[task] or flow["queries"] != panel["A"]
            or flow["flow_seed"] != panel["A_flow_seed"]
            or len(flow["queries"]) != 28
            or set(query["demo"] for query in flow["queries"]) & set(panel["teachers"])
            or any(tuple(flow[key].shape) != shape or not torch.isfinite(flow[key]).all()
                   for key, shape in shapes.items())):
        raise ValueError("fixed A28 query/Gaussian/tau/target source changed")
    return flow, path


@torch.no_grad()
def fm_prediction(runtime, state: dict, batch: dict, flow: dict, microbatch: int) -> torch.Tensor:
    """Read the original saved FM sample; never call the 10-step action sampler."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    validate_lora_state(state, runtime.lora)
    runtime.restore_identity()
    owner = NativeFlowPrediction(runtime.policy)
    predictions = []
    for start in range(0, 28, microbatch):
        stop = min(start + microbatch, 28)
        sliced = {key: (value[start:stop] if isinstance(value, torch.Tensor)
                       and value.ndim and len(value) == 28 else value) for key, value in batch.items()}
        sliced[ACTION] = flow["action"][start:stop].to(runtime.device)
        images, masks = runtime.policy._preprocess_images(dict(sliced))
        sample = FlowSample((images, masks, sliced[OBS_LANGUAGE_TOKENS],
                            sliced[OBS_LANGUAGE_ATTENTION_MASK], runtime.policy.prepare_action(sliced),
                            flow["noise"][start:stop].to(runtime.device),
                            flow["time"][start:stop].to(runtime.device)),
                           flow["FM_target"][start:stop].to(runtime.device), 7)
        with autocast(runtime.device):
            prediction = torch.func.functional_call(owner, {"policy." + key: value
                                                            for key, value in state.items()},
                                                    (sample,), strict=False)
        predictions.append(prediction[..., :7].detach().float().cpu())
    result = torch.cat(predictions)
    if result.shape != (28, 50, 7) or not torch.isfinite(result).all():
        raise ValueError("fixed A28 FM consumer lost finite 28x50x7 velocity")
    return result


def risk(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = (prediction.float() - target.float()).square()
    return {"full50": float(error.mean()), "first5": float(error[:, :5].mean()),
            "full50_motion6": float(error[..., :6].mean()),
            "full50_gripper1": float(error[..., 6].mean()),
            "first5_motion6": float(error[:, :5, :6].mean()),
            "first5_gripper1": float(error[:, :5, 6].mean()),
            "per_query_full50": error.mean(dim=(1, 2)).tolist(),
            "per_query_first5": error[:, :5].mean(dim=(1, 2)).tolist()}


def masked_risk(prediction, target, valid):
    """Keep full50 identity and separately score real future labels."""
    result = risk(prediction, target)
    valid = valid.cpu().bool()
    if valid.shape != prediction.shape[:2] or not valid.any():
        raise ValueError('action validity mask changed')
    error = (prediction.float()-target.float()).square()
    for label, horizon in [('valid_future',50),('valid_first5',5)]:
        mask=valid[:,:horizon,None]
        for suffix,channels in [('',slice(None)),('_motion6',slice(0,6)),('_gripper1',slice(6,7))]:
            e=error[:,:horizon,channels]
            result[label+suffix]=float((e*mask).sum()/(mask.sum()*e.shape[-1]))
    result['valid_steps_per_query']=valid.sum(-1).tolist()
    result['per_query_valid_future']=(error*valid[...,None]).sum((1,2)).div(valid.sum(-1)*7).tolist()
    return result


@torch.no_grad()
def generation_prediction(runtime,state,batch,flow,microbatch):
    """Official ten-step sampler, same saved Gaussian, no action labels passed."""
    from lerobot.utils.constants import ACTION
    validate_lora_state(state,runtime.lora)
    copy_task_lora_state_(runtime.policy,state,runtime.lora)
    predictions=[]
    for start in range(0,28,microbatch):
        stop=min(start+microbatch,28)
        sliced={k:(v[start:stop] if isinstance(v,torch.Tensor) and v.ndim and len(v)==28 else v)
                for k,v in batch.items() if k!=ACTION}
        prediction=runtime.policy.predict_action_chunk(sliced,
            noise=flow['noise'][start:stop].to(runtime.device),num_steps=10)
        predictions.append(prediction.detach().float().cpu())
    result=torch.cat(predictions)
    if result.shape!=(28,50,7) or not torch.isfinite(result).all():
        raise ValueError('official generated action lost finite28x50x7')
    return result


class LocalEffects:
    """Passive Original hidden effects; reduce on-device, retain no activations."""
    def __init__(self,policy,original,reexpressed,s_edits,device,microbatch):
        self.policy,self.microbatch=policy,microbatch
        self.original={k:v.to(device).float() for k,v in original.items()}
        self.reexpressed={k:v.to(device).float() for k,v in reexpressed.items()}
        self.s_edits={k:v.to(device).float() for k,v in s_edits.items()}
        self.names=tuple(k.removesuffix(LORA_A_SUFFIX) for k in original if k.endswith(LORA_A_SUFFIX))
        self.records=defaultdict(lambda:dict(E_squared=torch.zeros(28,device=device),
            original_squared=torch.zeros(28,device=device),S_squared=torch.zeros(28,device=device),
            first5_E_squared=torch.zeros(28,device=device),first5_original_squared=torch.zeros(28,device=device),
            max_absolute_E=torch.zeros(28,device=device)))
        self.calls={}

    def _hook(self,phase,expected_calls,name,arguments):
        h=arguments[0].detach().float()
        if h.ndim!=3 or h.shape[1]!=50:
            raise ValueError('execution target hidden lost full50 token grid')
        count=self.calls.get((phase,name),0)
        block,step=divmod(count,expected_calls)
        start,stop=block*self.microbatch,block*self.microbatch+len(h)
        if stop>28:
            raise ValueError('execution passive hook exceeded registered28 queries')
        a,b=self.original[name+LORA_A_SUFFIX],self.original[name+LORA_B_SUFFIX]
        at,bt=self.reexpressed[name+LORA_A_SUFFIX],self.reexpressed[name+LORA_B_SUFFIX]
        with torch.autocast(device_type=h.device.type,enabled=False):
            y=(h@a.T)@b.T
            e=y-(h@at.T)@bt.T
            sy=(h@self.s_edits[name].T)@b.T
        r=self.records[(phase,name)]
        for key,value in [('E_squared',e.square().sum((1,2))),
            ('original_squared',y.square().sum((1,2))),('S_squared',sy.square().sum((1,2))),
            ('first5_E_squared',e[:,:5].square().sum((1,2))),
            ('first5_original_squared',y[:,:5].square().sum((1,2)))]:
            r[key][start:stop]+=value
        r['max_absolute_E'][start:stop]=torch.maximum(r['max_absolute_E'][start:stop],e.abs().amax((1,2)))
        self.calls[(phase,name)]=count+1

    @contextmanager
    def capture(self,phase):
        expected=1 if phase=='FM' else 10
        handles=[]
        try:
            for name in self.names:
                handles.append(self.policy.get_submodule(name).register_forward_pre_hook(
                    lambda _module,args,site=name:self._hook(phase,expected,site,args)))
            yield
            wanted=((28+self.microbatch-1)//self.microbatch)*expected
            if any(self.calls.get((phase,n))!=wanted for n in self.names):
                raise ValueError('Original passive capture missed an execution target/flow step')
        finally:
            for h in handles:h.remove()

    def summary(self):
        result=[]
        for (phase,name),v in self.records.items():
            rows={k:value.cpu().tolist() for k,value in v.items()}
            rows['per_query_relative_E_to_original']=(v['E_squared']/v['original_squared'].clamp_min(1e-30)).sqrt().cpu().tolist()
            rows['relative_E_to_original']=float((v['E_squared'].sum()/v['original_squared'].sum().clamp_min(1e-30)).sqrt())
            rows['relative_E_to_S']=float((v['E_squared'].sum()/v['S_squared'].sum().clamp_min(1e-30)).sqrt())
            result.append(dict(phase=phase,site=name,calls=self.calls[(phase,name)],**rows))
        return result
