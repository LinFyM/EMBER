"""Source-H/native-operator compilation and full-factor cross-episode FM credit."""
from pathlib import Path
import json,os
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import read_json
from ember.writer.practice import processed
from ember.writer.function_credit import NativeFlowPrediction,FlowSample,mean_velocity_loss
from ember.writer.runtime import autocast
from .native import DenseExecution,NativeJacobian,public_probes,write_operators
from .video import MemoryStore


class Compiler:
    """One source Runtime, one frozen input cache, one complete operator writer."""
    def __init__(self,runtime,root,*,support_microbatch=4):
        from .history import SourceHistoryStore
        self.runtime,self.root = runtime,Path(root)
        self.history = SourceHistoryStore(self.root,runtime)
        self.memory = MemoryStore(self.root,runtime.device)
        self.dense = DenseExecution(runtime)
        self.probes = public_probes(runtime.lora,runtime.device)
        self.support_microbatch,self.F0 = support_microbatch,{}
        self.measurements = []

    def compile(self,model,task,demo):
        h = self.history.get(task)
        inputs = [processed(self.runtime,raw,self.runtime.tasks[task].authority.language) for raw in h['pre']]
        batch = {k:torch.cat([x[k] for x in inputs]) for k in inputs[0]}
        jacobian = NativeJacobian(self.runtime,self.dense,batch,h['noise'],microbatch=self.support_microbatch)
        try:
            if task not in self.F0:
                self.F0[task] = jacobian.response().detach().cpu()
            baseline = self.F0[task].to(self.runtime.device)
            memory = self.memory.training(task,demo).to(self.runtime.device)
            with autocast(self.runtime.device):
                pressure = model(memory,h['images'].to(self.runtime.device),h['hidden'].to(self.runtime.device),
                                 h['facts'].to(self.runtime.device),baseline).float()
            state,measure = write_operators(jacobian,pressure,self.probes,self.runtime.lora)
            measure.update(task=task,teacher_demo=demo,support=len(h['pre']),
                           physical_support_microbatch=self.support_microbatch,
                           support_positions=h['positions'],H_provenance=h['provenance'])
            self.measurements.append(measure)
            return state,jacobian,measure
        except BaseException:
            jacobian.close()
            raise

    def close(self):
        self.memory.close()
        self.dense.close()
        if hasattr(self.history,'close'):
            self.history.close()
        self.F0.clear()


def assignments(world,rank):
    """Sixteen real seven-query pieces, one global logical mean over112."""
    return [(t,[q for q in range(28) if ((t*4+q//7)*world)//16==rank]) for t in range(4)
            if any(((t*4+q//7)*world)//16==rank for q in range(28))]


def concatenate(values):
    if isinstance(values[0],torch.Tensor):return torch.cat(values)
    return type(values[0])(concatenate([v[i] for v in values]) for i in range(len(values[0])))


def native_credit(runtime,samples,states,microbatch):
    """Pack only useful query fragments, retaining one full A/B leaf per task."""
    leaves=[{k:v.detach().requires_grad_(True) for k,v in state.items()} for state in states]
    sample=FlowSample(concatenate([s.arguments for s in samples]),torch.cat([s.target for s in samples]),7)
    which=torch.tensor([i for i,s in enumerate(samples) for _ in s.target],device=runtime.device)
    owner=NativeFlowPrediction(runtime.policy);loss_value=0.;batches=[]
    for first in range(0,len(sample.target),microbatch):
        last=min(first+microbatch,len(sample.target))
        def cut(value):
            if isinstance(value,torch.Tensor):return value[first:last]
            return type(value)(cut(x) for x in value)
        part=FlowSample(cut(sample.arguments),sample.target[first:last],7)
        with autocast(runtime.device),runtime.execution.activate(leaves,batch_indices=which[first:last]):
            loss=mean_velocity_loss(owner(part),part.target,7)*((last-first)/112)
        loss.backward();loss_value+=float(loss.detach());batches.append(last-first)
    cotangents=[{k:(v.grad.detach() if v.grad is not None else torch.zeros_like(v)) for k,v in leaf.items()}
                for leaf in leaves]
    return cotangents,dict(FM=loss_value,query_rows=len(sample.target),physical_FM_batches=batches)


def distributed_credit(model,compiler,stream,events,context,*,microbatch=14,consumer=False):
    """Each compiler graph exists once; query pieces may use1–6 local ranks."""
    owned={};zero=lambda: {k:torch.zeros_like(v) for k,v in compiler.runtime.identity.items()}
    for index,event in enumerate(events):
        if index%context.world_size==context.rank:
            owned[index]=compiler.compile(model,event['task'],event['teacher_demo'])
    query_map=dict(assignments(context.world_size,context.rank))
    report=dict(FM=0.,query_rows=0,conditions=len(owned),physical_FM_batches=[],compilations=[],consumer=[])
    try:
        transmitted=[]
        for index,event in enumerate(events):
            owner=index%context.world_size
            state=owned[index][0] if context.rank==owner else zero()
            values={k:v.detach().clone().contiguous() for k,v in state.items()}
            if context.world_size>1:
                for value in values.values():dist.broadcast(value,src=owner)
            transmitted.append(values)
        selected=list(query_map)
        samples=[stream.sample(events[i],query_map[i]) for i in selected]
        cotangents,value=native_credit(compiler.runtime,samples,[transmitted[i] for i in selected],microbatch)
        local=dict(zip(selected,cotangents,strict=True))
        report.update(FM=value['FM'],query_rows=value['query_rows'],physical_FM_batches=value['physical_FM_batches'])
        for index,event in enumerate(events):
            owner=index%context.world_size
            cotangent=local[index] if index in local else {k:torch.zeros_like(v) for k,v in transmitted[index].items()}
            if context.world_size>1:
                for value in cotangent.values():dist.reduce(value,dst=owner,op=dist.ReduceOp.SUM)
            if context.rank==owner:
                actual,jacobian,measure=owned.pop(index)
                try:
                    live=[k for k,v in actual.items() if v.requires_grad]
                    if live:torch.autograd.backward(tuple(actual[k] for k in live),tuple(cotangent[k] for k in live))
                    if consumer:
                        output=jacobian.controller_response({k:v.detach() for k,v in actual.items()})
                        baseline=compiler.F0[event['task']].to(output)
                        report['consumer'].append(dict(task=event['task'],actual_F_out_finite=bool(output.isfinite().all()),
                            actual_F_out_source_MSE=float((output-baseline).square().mean()),
                            first_order_MSE=float((output-baseline+jacobian.calibration['alpha']*jacobian.calibration['r']).square().mean()),
                            pressure_dot_actual_change=float(((output-baseline)*jacobian.calibration['pressure']).sum()),
                            full_factors_consumed=True,extra_environment_episodes=0))
                    report['compilations'].append(dict(measure,native_calls=dict(jacobian.calls)))
                finally:
                    jacobian.close()
        return report
    finally:
        for _,jacobian,_ in owned.values():jacobian.close()


class FinalCompiler:
    def __init__(self,model,pipeline):self.model,self.pipeline=model,pipeline
    def close(self):self.pipeline.close()


def load_final(root,runtime):
    from .checkpoint import load_model
    from .contract import FINAL,ROOT
    model=load_model(root,FINAL,runtime.device)
    model.eval().requires_grad_(False)
    physical=read_json(Path(root)/'physical_selection.json')
    return FinalCompiler(model,Compiler(runtime,ROOT,support_microbatch=physical['support_microbatch']))


def materialize(runtime,compiler,task,demo):
    state,jacobian,measure=compiler.pipeline.compile(compiler.model,task,demo)
    try:
        folder=compiler.pipeline.root/'materialization';folder.mkdir(parents=True,exist_ok=True)
        from ember.pi05_source_checkpoint import write_json_atomic
        write_json_atomic(folder/f'task_{task:04d}.json',dict(measure,native_calls=dict(jacobian.calls),
            factor_L2_norm=float(torch.stack([v.detach().norm() for v in state.values()]).norm()),
            operator_L2_norm=measure['alpha']*measure['rank8_energy']**.5,new_environment_episodes=0))
        with (folder/f'memory_{os.getpid()}.jsonl').open('a') as handle:
            for item in compiler.pipeline.memory.drain_measurements():handle.write(json.dumps(item)+'\n')
        return {k:v.detach().float() for k,v in state.items()}
    finally:
        jacobian.close()
