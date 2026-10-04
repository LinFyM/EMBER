"""Endpoint25 complete banks and fixed640 paired native-precision B20 queries."""
from copy import deepcopy
import gc
import time
import torch
from safetensors.torch import load_file, save_file
from ember.batched_lora import BatchedLoRAInference
from ember.lora import LORA_A_SUFFIX
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ..data import FormalData
from .common import ROOT, OLD, ASSET, TASKS, HELD_TEACHERS, HELD_REFERENCE, load_fixed, load_labels, save_compact
from .model import compile_fixed
from .credit import raw_batch, samples, labels_for
from .observe import TenFlowObserver


@torch.no_grad()
def materialize(runtime, spec, binding, arm, identity):
    out = ROOT / 'banks' / arm
    if (out / 'manifest.json').exists():
        raise ValueError('endpoint25 already compiled; no repeated Writer calls')
    out.mkdir(exist_ok=False)
    # The source/normalization/lora recipe is inherited, while diagnostic
    # condition/scene mappings and actual training/reading identities are new.
    bank = deepcopy(read_json(HELD_REFERENCE)['adapter'])
    bank.update(conditions=[], tasks=[], native_role_centered_content=dict(study=ROOT.name, arm=arm),
        training_git=identity['commit'], materialization_git=identity['commit'],
        checkpoint=str(ROOT / arm / 'checkpoint64'),
        checkpoint_manifest=file_record(ROOT / arm / 'checkpoint64/checkpoint_manifest.json'),
        native_reading=file_record(OLD / 'native/manifest.json'),
        condition_factors='complete_A0_plus_S_B0_plus_M', diagnostic_only=True,
        condition_count=25, environment_case_count=55, formal400=False,
        evaluation_scope='fixed32 train +23 task16; scene/RNG source per actual case')
    for tasks, role in ((TASKS, 'train'), ((16,), 'validation')):
        data = FormalData(ASSET, spec, query_labels=False, task_ids=tasks, role=role)
        try:
            for task in tasks:
                meta = data.tasks[task]
                bank['tasks'].append(dict(global_task_id=task, suite=meta.suite, task_id=meta.suite_task_id,
                    language=meta.authority.language, split_role=role, episodes=[]))
                for teacher in ((0, 1) if task != 16 else HELD_TEACHERS):
                    fixed = load_fixed(task, teacher, arm, runtime.device)
                    factors, residual = compile_fixed(runtime, fixed, binding, components=True)
                    key = f'task{task:03d}_teacher{teacher:02d}'; path = out / (key + '.safetensors')
                    save_file({k: v.detach().cpu().contiguous() for k, v in factors.items()}, str(path))
                    content = torch.load(ROOT / 'content' / (key + '.pt'), map_location='cpu', mmap=True, weights_only=True)
                    save_compact(dict(R=residual, alpha_frame_mean=content['alpha'].mean(1)), out / (key + '_binding.pt'))
                    bank['conditions'].append(dict(condition_id=key, global_task_id=task, teacher_demo=teacher,
                        factors=file_record(path), sampled_frames=len(fixed['H']), full38=True,
                        native_reading=file_record(OLD / 'native' / (key + '.pt')),
                        content=file_record(ROOT / 'content' / (key + '.pt'))))
                    del fixed, factors, residual, content
        finally:
            data.close()
    bank['manifest'] = dict(path=str(out / 'manifest.json'))
    write_json_atomic(out / 'manifest.json', bank)
    return bank


@torch.inference_mode()
def readout(runtime, spec, arm, bank):
    data = FormalData(ASSET, spec, query_labels=True, task_ids=TASKS)
    labels = load_labels(); manifest = read_json(OLD / 'query_manifest.json')
    label_records = {(r['task'], r['demo']):r for r in read_json(OLD / 'labels/manifest.json')['records']}
    records, timings = [], []; runtime.restore_identity()
    batched = BatchedLoRAInference(runtime.policy, runtime.lora)
    torch.cuda.reset_peak_memory_stats()
    try:
        for task in TASKS:
            queries = manifest['B20'][str(task)]
            raw = raw_batch(data, task, queries); batch = runtime.processor.training_batch(raw)
            q = labels_for(labels, task, queries, runtime.device).repeat(2, 1, 1)
            conditions = [next(r for r in bank['conditions'] if r['condition_id'] == f'task{task:03d}_teacher{d:02d}') for d in (0,1)]
            factors = [load_file(c['factors']['path'], device='cpu') for c in conditions]
            packed = {k:torch.cat((v,v)) if isinstance(v,torch.Tensor) and v.ndim and len(v)==20
                      else v*2 if isinstance(v,(list,tuple)) and len(v)==20 else v for k,v in batch.items() if k!='action'}
            noise20 = samples(runtime, batch, queries).arguments[5]; noise = torch.cat((noise20,noise20))
            observer = TenFlowObserver(runtime.policy, q)
            if torch.is_autocast_enabled('cuda'):
                raise ValueError('full generated B20 must use original consumer without outer autocast')
            torch.cuda.synchronize(); started=time.monotonic()
            with batched.activate([factors[0]]*20+[factors[1]]*20), observer.capture():
                generated=runtime.policy.predict_action_chunk(packed,noise=noise,num_steps=10)
            torch.cuda.synchronize(); seconds=time.monotonic()-started; arrays=observer.arrays()
            timings.append(dict(task=task,queries=40,seconds=seconds,queries_per_second=40/seconds))
            for d,c in enumerate(conditions):
                selected=slice(d*20,(d+1)*20)
                row=dict(schema='centered_content_B20_v1',arm=arm,task=task,teacher=d,queries=queries,query_offset=1,
                    action_generated_normalized=generated[selected].float().cpu(),action_target=batch['action'].cpu(),
                    valid_mask=~raw['action_is_pad'],noise=noise[selected].cpu(),role_patch_weights=q[selected].cpu(),
                    role_scores=arrays['role_scores'][selected],image_mass=arrays['image_mass'][selected],
                    entities=label_records[task,queries[0]['demo']]['entities'],target_entity_index=0,
                    entity_total_mass_available=False,reason='retained B20 q is area-normalized; raw f was not retained',
                    extra_forward=False,Writer_removed=True,physical_query_batch=40,outer_autocast=False)
                path=ROOT/'predictions'/arm/(c['condition_id']+'_B20.pt')
                artifact=save_compact(row,path);records.append(dict(task=task,teacher=d,queries=20,**artifact))
            del arrays,observer,row,generated,packed,factors,noise,q,batch,raw
    finally:
        batched.close();data.close();runtime.restore_identity()
    write_json_atomic(ROOT/'predictions'/arm/'manifest.json',dict(complete=True,records=records,queries=320,
        forward_steps=10,additional_FM_forward=0,physical_query_batch=40,paired_teachers=2,timings=timings,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,outer_autocast=False,
        larger_same_task_batch_unavailable='all20 authorized queries ×both teachers consumed once'))


def remove_training_adapter(runtime):
    """Unwrap the existing identity PEFT modules, then use canonical evaluator injection."""
    runtime.restore_identity()
    names=[t.name for t in runtime.lora.targets]
    for name in names:
        module=runtime.policy.get_submodule(name)
        base=module.get_base_layer()
        parent,key=name.rsplit('.',1)
        setattr(runtime.policy.get_submodule(parent),key,base)
    if any('.lora_' in n for n,_ in runtime.policy.named_parameters()):
        raise ValueError('temporary Writer adapter unload incomplete')
    # No merge or source reload: the original frozen base_layer objects persist.
    del runtime.writer;gc.collect();torch.cuda.empty_cache()
