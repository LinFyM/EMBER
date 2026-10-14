"""Persistent ready endpoint consumers, shared held reading and strict paired rows."""
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_eval_queue import claim_next, fail_job
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import autocast
from .analysis import paired, billing, interaction_ledger
from .image_compiler import pure_features
from .image_data import feature_file, publish_record
from .image_history import encode_history, history_file, model_history
from .image_learning import load_compiler


def endpoint(root, arm):
    decision = read_json(root/'endpoint_decision.json')
    return root/'arms'/arm/'checkpoints'/f"update_{decision['endpoint']:08d}"


def compile_condition(root, runtime, models, entry, role, frame_chunk):
    """One complete video read supplies all ready arms; never combine their LoRAs."""
    task, demo = entry['task'],entry['teacher_demo']
    targets = {arm:root/'arms'/arm/'readouts'/role/'loras'/f'task_{task:04d}_demo_{demo:02d}.safetensors' for arm in models}
    missing = {arm:file for arm,file in targets.items() if not file.exists()}
    if missing:
        if role == 'seen':
            value = torch.load(feature_file(root,task,demo),map_location='cpu',weights_only=False)
        else:
            value = pure_features(runtime,task,demo,frame_chunk=frame_chunk)
        descriptor = entry['history_descriptor']
        history = (torch.load(history_file(root,task),map_location='cpu',weights_only=False)
                   if role=='seen' else encode_history(runtime,descriptor))
        h = model_history(history,runtime.device)
        z,indices,words = (value[name].to(runtime.device) for name in ('z_img','indices','language'))
        for arm,file in missing.items():
            with torch.no_grad(),autocast(runtime.device):
                state = models[arm](z,indices,h,words if arm=='VL' else None)
            if len(state)!=76 or any(not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('compiler must export all38 finite factors')
            file.parent.mkdir(parents=True,exist_ok=True)
            temporary=file.with_suffix('.tmp')
            save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},str(temporary));temporary.replace(file)
            write_json_atomic(file.with_suffix('.json'),dict(complete=True,checkpoint=str(endpoint(root,arm)),
                task=task,demo=demo,frames=len(indices),arm=arm,Gemma_language=False,
                Reader_language=arm=='VL',history=True,execution_hidden_input=False,
                task_parameters_input=False,fixed_real_MT_context=True,teacher_labels_input=False,
                real_source=value['source'],H_source=history['source'],unique_complete_LoRA=True,
                adaptation_episodes=1,new_adaptation_episodes=0,
                reused_MT_environment_steps=descriptor['environment_steps'],practice_state_id=descriptor['practice_state_id']))
    return targets


def readout_worker(args,runtime,runner):
    root,role=args.root,args.readout_role
    arms=('V','VL') if args.compiler_arm=='both' else (args.compiler_arm,)
    if role=='validation' and arms!=('V','VL'):raise ValueError('held shared reader exports both complete endpoints')
    models={arm:load_compiler(root,arm,endpoint(root,arm),runtime.device)[0].eval().requires_grad_(False) for arm in arms}
    panel=read_json(root/'readout_panels'/f'{role}.json')
    source = read_json(root/'history_manifest.json')['training' if role=='seen' else 'validation']
    histories={(e['task'],e.get('init_state_id') if role=='validation' else None):e for e in source}
    tasks={(t['suite'],t['task_id']):t for t in runner.contract['tasks']}
    queue=root/'queues'/f"{'paired_validation' if role=='validation' else arms[0]+'_seen'}.sqlite"
    while (claim:=claim_next(queue,worker_id=args.worker_id)) is not None:
        try:
            task=tasks[claim.task_key]
            entries=[dict(**e,history_descriptor=histories[e['task'],e['init_state_id'] if role=='validation' else None])
                for e in panel['rows'] if e['task']==task['global_task_id'] and e['init_state_id'] in claim.shard.init_state_ids]
            requests=[];by_id={}
            for entry in entries:
                targets=compile_condition(root,runtime,models,entry,role,args.frame_chunk)
                for arm,file in targets.items():
                    identity=f"{arm}_task_{entry['task']:04d}_state_{entry['init_state_id']:03d}"
                    output=root/'arms'/arm/'readouts'/role/'rows'/f'{identity}.json'
                    by_id[identity]=(arm,entry,file,output)
                    if output.exists():continue
                    requests.append(dict(task=task,state=load_file(str(file),device=str(runtime.device)),
                        parameter_ref=f'{arm}_{endpoint(root,arm).name}',state_id=entry['init_state_id'],
                        noise_root=entry['noise_root'],registered_scene=entry['registered_scene'],episode_id=identity,
                        remaining=task['horizon']+10,capture_history=False,capture_hidden=False,uses_teaching=False))
            for actual in runner.run(requests):
                arm,entry,file,output=by_id[actual['request']['episode_id']]
                row=dict(actual['row'])
                if actual['history'].records:
                    first=actual['history'].records[0]
                    row.update(first_native_normalized_actions=first['normalized_actions'].tolist(),first_noise_seed=first['noise_seed'])
                row.update(global_task_id=entry['task'],teacher_demo=entry['teacher_demo'],video_ordinal=entry.get('video_ordinal',0),
                    checkpoint=str(endpoint(root,arm)),compiler_arm=arm,complete_LoRA=str(file),
                    compiler_exits_before_final_episode=True,unique_complete_LoRA=True,adaptation_episodes=1,
                    new_adaptation_episodes=0,reused_MT_environment_steps=entry['history_descriptor']['environment_steps'],
                    H_source=entry['history_descriptor']['H'],H_execution_hidden=False,
                    teaching_in_final_episode=False,query_excluded_from_training=True)
                write_json_atomic(output,row)
            rows=[read_json(item[3]) for item in by_id.values()]
            receipt=root/'readout_shards'/queue.stem/f'{claim.shard.job_id}_{claim.attempt:03d}.json'
            write_json_atomic(receipt,dict(rows=rows,seconds=time.monotonic(),complete=True))
            publish_record(queue,claim,args.worker_id,receipt,len(rows),successes=sum(r['success'] for r in rows))
        except BaseException as error:
            fail_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,error=repr(error))
            raise


def reference_rows(root,role):
    manifest=read_json(root/'inputs/asset_manifest.json');parent=root.parent
    if role=='seen':
        mt,g,e=[],[],[]
        for task in manifest['tasks']:
            source=read_json(Path(task['MT_query_receipt']['path']))['rows']
            mt.extend(r for r in source if r['reference']=='MT300')
            g.extend(r for r in source if r['reference']=='G_000')
            receipt=read_json(Path(task['E_query_receipt']['path']))
            e.extend(receipt['rows'])
        d=read_json(parent/'operator_native_joint_learning/arms/operator_only/readouts/seen/results.json')['raw_rows']
        d144=read_json(parent/'operator_D_adjacent_stability/arms/operator_only/readouts/seen/results.json')['raw_rows']
        return dict(MT340=mt,E455=e,firstG427=g,D72_414=d,D144_414=d144)
    old=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261013/selector_transfer72_fixedG38349')
    mt=[e['MT_row'] for e in read_json(old/'formal_panel.json')['rows']]
    old_full=[read_json(p)['selected'] for p in sorted((old/'validation/RL_00000018/correct').glob('task_*/query.json'))]
    source=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/full_writer_after24849_20261013')
    g=read_json(source/'diagnostics/first_G_validation400_20261013/results.json')['raw_rows']
    return dict(MT153=mt,firstG165=g,old_full163=old_full,
        Phi167=read_json(parent/'independent_value_fixed_paths/results.json')['rows'],
        D72_175=read_json(parent/'operator_native_joint_learning/arms/operator_only/readouts/validation/results.json')['raw_rows'],
        D144_168=read_json(parent/'operator_D_adjacent_stability/arms/operator_only/readouts/validation/results.json')['raw_rows'])


def summarize(root):
    root=Path(root);result=dict(complete=True,arms={},Test=False,no_automatic_successor=True)
    for role,count in [('seen',576),('validation',400)]:
        refs=reference_rows(root,role);parts={}
        for arm in ('V','VL'):
            rows=[read_json(p) for p in sorted((root/'arms'/arm/'readouts'/role/'rows').glob('*.json'))]
            if len(rows)!=count or len({(r['global_task_id'],r['init_state_id']) for r in rows})!=count:
                raise ValueError('complete unique paired endpoint denominator required')
            tasks=sorted({r['global_task_id'] for r in rows})
            per_task=[dict(task=t,rows=sum(r['global_task_id']==t for r in rows),
                successes=sum(r['success'] for r in rows if r['global_task_id']==t)) for t in tasks]
            per_suite={s:dict(rows=sum(r['suite']==s for r in rows),successes=sum(r['success'] for r in rows if r['suite']==s))
                       for s in sorted({r['suite'] for r in rows})}
            parts[arm]=dict(complete=True,raw_rows=rows,per_task=per_task,per_suite=per_suite,
                successes=sum(r['success'] for r in rows),breadth=sum(t['successes']>0 for t in per_task),
                paired={name:paired(base,rows) for name,base in refs.items()},single_checkpoint=True,
                adaptation_episodes_per_condition=1,new_adaptation_episodes=0)
            write_json_atomic(root/'arms'/arm/'readouts'/role/'results.json',parts[arm])
            result['arms'].setdefault(arm,{})[role]=parts[arm]
        result['V_VL_'+role]=paired(parts['VL']['raw_rows'],parts['V']['raw_rows'])
    method=read_json(root/'history_manifest.json')
    result.update(interaction=interaction_ledger(root),billing=billing(root),endpoint=read_json(root/'endpoint_decision.json'),
        reused_MT_dependency={role:dict(episodes=len(entries),environment_steps=sum(e['environment_steps'] for e in entries),
            physical_reuse_shared_across_arms=True,source=str(root/'history_manifest.json'))
            for role,entries in [('training',method['training']),('validation',method['validation'])]},
        prior_scope_change=read_json(root/'run_contract.json')['prior_scope_change'],zero_interaction_method=False)
    write_json_atomic(root/'results.json',result)
    write_json_atomic(root/'complete.json',dict(complete=True,finished=1952,Test=False,no_automatic_successor=True))
    return result
