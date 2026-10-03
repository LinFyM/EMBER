"""Fixed A28/B20 FM and the official full32 sampler, without extra forwards."""
from __future__ import annotations
from contextlib import contextmanager

import torch
from safetensors.torch import save_file

from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, flow_sample
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from .effect_labels import ROOT, TEACHERS
from .effect_learning import compile_B, label_batch, raw_batch

PRIOR = ROOT.parent / 'operator_chain_diagnosis_20260929/functional_credit_transport/group0'


@contextmanager
def capture_full_sample(policy):
    """Capture the existing official integration result before action7 unpadding."""
    original = policy.model.sample_actions
    values = []
    def sample(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.shape[1:] != (50, 32):
            raise ValueError('official native sampler lost 50x32 output')
        values.append(result.detach())
        return result
    policy.model.sample_actions = sample
    try:
        yield values
    finally:
        policy.model.sample_actions = original


@torch.no_grad()
def functional(runtime, data, positions, panels, cached, out, micro, *, arm, states=None):
    records = []
    for task, teachers in TEACHERS.items():
        panel = panels[str(task)]
        for teacher in teachers:
            identifier = f'task{task:03d}_teacher{teacher:02d}'
            state = states[(task, teacher)] if states is not None else compile_B(runtime, cached[(task, teacher)])
            validate_lora_state(state, runtime.lora)
            factor = out / 'bank' / f'{identifier}.safetensors'
            factor.parent.mkdir(parents=True, exist_ok=True)
            if arm != 'parent':
                save_file({k:v.cpu().contiguous() for k,v in state.items()}, str(factor))
            paths = {}
            for label in ('A', 'B'):
                if arm == 'parent' and label == 'A':
                    paths[label] = str(ROOT.parent/'conditional_A_reexpression_diagnostic_20261002/functional'/f'{identifier}_Original.pt')
                    continue
                queries = panel[label]; size = len(queries)
                raw = raw_batch(data, task, queries)
                batch = runtime.processor.training_batch(raw)
                displacement, valid = label_batch(positions, task, queries)
                refpath = PRIOR / f'task{task:03d}_{label}_query_{"flow" if label=="A" else "noise"}_target.pt'
                ref = torch.load(refpath, map_location='cpu', weights_only=False, mmap=True)
                if queries != ref['queries'] or int(ref['flow_seed']) != panel[f'{label}_flow_seed']:
                    raise ValueError('fixed offline query/noise identity changed')
                runtime.restore_identity()
                owner = NativeFlowPrediction(runtime.policy)
                predictions, targets, noises, times, generated = [], [], [], [], []
                for start in range(0, size, micro):
                    stop = min(start+micro, size)
                    sliced = {k:(v[start:stop] if isinstance(v,torch.Tensor) and v.ndim and len(v)==size else v)
                              for k,v in batch.items()}
                    with autocast(runtime.device):
                        sample = flow_sample(runtime.policy, sliced, seed=ref['flow_seed'], device=runtime.device,
                                             random_batch=size, offset=start)
                        if label == 'A':
                            # Actual retained query target/Gaussian/tau, never a reconstructed approximation.
                            clean = sample.arguments[4].clone(); clean[..., :7] = ref['action'][start:stop].to(clean)
                            sample = FlowSample((*sample.arguments[:4],clean,ref['noise'][start:stop].to(clean),
                                ref['time'][start:stop].to(runtime.device)), ref['FM_target'][start:stop].to(clean),7)
                        else:
                            noise = ref['noise'][start:stop].to(runtime.device)
                            sample = FlowSample((*sample.arguments[:5],noise,sample.arguments[6]),
                                                noise-sample.arguments[4],7)
                        if arm == 'J':
                            clean = sample.arguments[4].clone(); clean[..., 7:10] = displacement[start:stop].to(clean)
                            sample = FlowSample((*sample.arguments[:4],clean,*sample.arguments[5:]), sample.arguments[5]-clean,7)
                        prediction = torch.func.functional_call(owner,
                            {'policy.'+k:v for k,v in state.items()}, (sample,), strict=False)
                    predictions.append(prediction.float().cpu()); targets.append(sample.target.float().cpu())
                    noises.append(sample.arguments[5].cpu()); times.append(sample.arguments[6].cpu())
                if label == 'B':
                    copy_task_lora_state_(runtime.policy, state, runtime.lora)
                    with capture_full_sample(runtime.policy) as captured, autocast(runtime.device):
                        for start in range(0, size, micro):
                            stop = min(start+micro,size)
                            obs = {k:(v[start:stop] if isinstance(v,torch.Tensor) and v.ndim and len(v)==size else v)
                                   for k,v in batch.items() if k != 'action'}
                            runtime.policy.predict_action_chunk(obs,noise=ref['noise'][start:stop].to(runtime.device),num_steps=10)
                        generated = [v.float().cpu() for v in captured]
                    runtime.restore_identity()
                row = dict(schema='ember_joint_effect_functional_v1',arm=arm,task=task,teacher=teacher,queries=queries,
                    flow_seed=ref['flow_seed'],query_offset=1,target_ref=file_record(refpath),
                    prediction_FM=torch.cat(predictions), FM_target=torch.cat(targets),noise=torch.cat(noises),time=torch.cat(times),
                    action_target=batch['action'].cpu(), valid_action=~raw['action_is_pad'],
                    displacement_target=displacement,valid_displacement=valid,
                    action_generated_full32=torch.cat(generated) if generated else None,
                    generated_displacement_is_offline_demonstration_error_not_actual_physics=True)
                path = out / 'functional' / f'{identifier}_{label}.pt'; path.parent.mkdir(exist_ok=True)
                torch.save(row,path); paths[label] = str(path)
            records.append(dict(condition_id=identifier,global_task_id=task,teacher_demo=teacher,
                                factors=file_record(factor) if arm!='parent' else None,functional=paths))
    write_json_atomic(out/'functional/manifest.json',dict(records=records,arm=arm,queries={'A':28,'B':20}))
    return records
