"""Phi180 original-prefix readouts, sealed400 choices, and only changed final episodes."""
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import (EvaluationTask, EvaluationShard, claim_next, fail_job,
    initialize_queue, build_cost_balanced_shards)
from ember.writer.runtime import autocast
from .value_data import ValueInputs, prefix_inputs, publish_value_record
from .value_learning import initialize_value, phi_logits
from .learning import memory_peak


def value_endpoint(root):
    checkpoint = Path(root)/'checkpoints/update_00000180'
    manifest = read_json(checkpoint/'manifest.json')
    if not manifest['complete'] or manifest['checkpoint_role'] != 'sole_scientific_endpoint':
        raise ValueError('scores and selection accept only the unique Phi180 endpoint')
    return checkpoint


def score_values(args, runtime):
    checkpoint = value_endpoint(args.root)
    initialize_value(runtime, checkpoint=checkpoint)
    runtime.phi.eval()
    inputs = ValueInputs(runtime)
    held = args.readout_role == 'validation'
    role = 'held' if held else 'nonheld'
    entries = ({e['id']: e for e in read_json(args.root/'input_manifest.json')['held']}
               if held else inputs.entries)
    queue = args.root/'queues'/f'{role}.sqlite'
    while (claim := claim_next(queue, worker_id=args.worker_id)) is not None:
        entry = entries[claim.shard.job_id]
        file = args.root/'scores'/role/f"{entry['id']}.json"
        try:
            started = time.monotonic()
            if not file.exists():
                prefixes = entry['prefixes'] if held else read_json(
                    args.root/'frozen_native'/f"{entry['id']}.json")['prefixes']
                labels = None if held else read_json(Path(entry['path'])/'query/complete.json')['rewards']
                rows = []
                path = None
                for prefix in prefixes:
                    candidates = prefix['candidates']
                    if held and candidates == ['MT300']:
                        rows.append(dict(selected='MT300', candidates=candidates, forced_single_candidate=True,
                            records=prefix['records'], episodes=prefix['episodes'], logits=None, probabilities=None))
                        continue
                    if path is None:
                        path = inputs.path(entry)
                        features = runtime.teaching(entry['task'], entry['demo'], video_task=path.video_task)
                    with torch.no_grad():
                        logits, counts = phi_logits(runtime, path, prefix, features)
                        probability = logits.sigmoid()
                        index = int(logits.argmax())
                        row = dict(selected=candidates[index], candidates=candidates, counts=counts,
                            logits=logits.cpu().tolist(), probabilities=probability.cpu().tolist(),
                            records=prefix['records'], episodes=prefix['episodes'], forced_single_candidate=False)
                        if not held:
                            responses = {key: torch.as_tensor(v) for key, v in prefix['native_response_predictions'].items()}
                            model_rows, states, responses, _ = prefix_inputs(runtime, path, prefix, responses)
                            with autocast(runtime.device):
                                beta = runtime.actor('final', features, model_rows, states, candidates, responses).float()
                            y = logits.new_tensor([labels[key] for key in candidates])
                            row.update(query_success_rates=y.cpu().tolist(), BCE=float(F.binary_cross_entropy_with_logits(logits,y)),
                                Brier=float((probability-y).square().mean()), selected_query_return=float(y[index]),
                                regret=float(y.max()-y[index]), beta_logits=beta.cpu().tolist(),
                                beta_selected=candidates[int(beta.argmax())],
                                beta_query_return=float(y[int(beta.argmax())]),
                                Phi0_defined_probability=.5, Phi0_selected=candidates[0],
                                Phi0_query_return=float(y[0]), query_label_consumer='readout only;not condition')
                        rows.append(row)
                write_json_atomic(file, dict(complete=True, checkpoint=str(checkpoint), path_source=entry['path'],
                    task=entry['task'], kind=entry['kind'], rows=rows, seconds=time.monotonic()-started,
                    peak=memory_peak(runtime.device), held_outcomes_read=False, new_native_queries=0))
            publish_value_record(queue,claim,args.worker_id,file,len(entry['prefixes']))
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token,error=repr(error))
            raise


def lock_choices(root):
    """Seal all400 choices before any original final/MT outcome is read."""
    root = Path(root)
    checkpoint = value_endpoint(root)
    entries = read_json(root/'input_manifest.json')['held']
    selections = []
    for entry in entries:
        score = read_json(root/'scores/held'/f"{entry['id']}.json")
        if not score['complete'] or score['checkpoint'] != str(checkpoint) or len(score['rows']) != 1:
            raise ValueError('whole400 fixed Phi choices required before reading final outcomes')
        choice = score['rows'][0]
        if choice['selected'] not in entry['prefixes'][0]['candidates']:
            raise ValueError('only a current practiced candidate can be deployed')
        weight = None if choice['selected'] == 'MT300' else str(Path(entry['path'])/f"{choice['selected']}.safetensors")
        if weight is not None and not Path(weight).is_file():
            raise ValueError('selected actual complete candidate weight is missing')
        selections.append(dict(id=entry['id'], task=entry['task'], init_state_id=entry['init_state_id'],
            demo=entry['demo'], path_source=entry['path'], selected=choice['selected'],
            qualified_parameter_ref=entry['identity']+':'+choice['selected'], weight_source=weight,
            forced_single_candidate=choice['forced_single_candidate'], logits=choice['logits'],
            probabilities=choice['probabilities'], candidates=choice['candidates']))
    if len(selections) != 400 or len({(e['task'],e['init_state_id']) for e in selections}) != 400:
        raise ValueError('strict unique400 selection denominator is incomplete')
    file = root/'held_selection_manifest.json'
    record = dict(complete=True, checkpoint=str(checkpoint), selections=selections,
                  final_outcomes_read=False, no_checkpoint_selection=True)
    if file.exists() and read_json(file) != record:
        raise ValueError('sealed selection manifest cannot change after final outcomes')
    write_json_atomic(file, record)
    return selections


def plan_final(root, *, workers=6, group=16):
    root = Path(root)
    selections = lock_choices(root)
    run = read_json(root/'run_contract.json')
    old = Path(run['old_root'])
    panel = read_json(old/'formal_panel.json')
    panel_rows = {(r['task'],r['init_state_id']):r for r in panel['rows']}
    entries = []
    for selected in selections:
        original = read_json(Path(selected['path_source'])/'path.json')
        p = panel_rows[selected['task'], selected['init_state_id']]
        if p['teacher_demo'] != selected['demo'] or original['identity'] != selected['qualified_parameter_ref'].rsplit(':',1)[0]:
            raise ValueError('original paired400 task/video/path identity changed')
        row_file = root/'final_rows'/f"{selected['id']}.json"
        if selected['selected'] == 'MT300':
            raw = dict(p['MT_row'])
            source = dict(kind='reused_official_MT400', file=p['MT_source'])
        elif selected['selected'] == original['selected']:
            query = Path(selected['path_source'])/'query.json'
            raw = read_json(query)['selected']
            if raw['reference'] != selected['selected']:
                raise ValueError('old final row was executed with another complete candidate')
            source = dict(kind='reused_same_actual_candidate', file=str(query), weight=selected['weight_source'])
        else:
            raw, source = None, dict(kind='new_final_episode', weight=selected['weight_source'])
        entry = dict(selected, noise_root=p['noise_root'], registered_scene=p['registered_scene'],
            old_selected=original['selected'], execution_source=source)
        entries.append(entry)
        if raw is not None and not row_file.exists():
            write_json_atomic(row_file, finalized_row(root, entry, raw))
    new = [entry for entry in entries if entry['execution_source']['kind'] == 'new_final_episode']
    tasks = [EvaluationTask(t['suite'],t['task_id'],t['horizon'], tuple(e['init_state_id'] for e in new
        if e['task']==t['global_task_id'])) for t in run['validation_environment']['tasks']]
    tasks = [task for task in tasks if task.init_state_ids]
    balanced = build_cost_balanced_shards(tasks, env_batch_size=group, physical_gpu_count=workers) if tasks else []
    parts = [(s,s.init_state_ids[first:first+group]) for s in balanced
             for first in range(0,len(s.init_state_ids),group)]
    shards = [EvaluationShard(f'{s.job_id}_{index:03d}',index,s.suite,s.task_id,s.horizon,
        ids,s.horizon*len(ids),s.preferred_gpu) for index,(s,ids) in enumerate(parts)]
    write_json_atomic(root/'final_plan.json', dict(complete=True, checkpoint=str(value_endpoint(root)), entries=entries,
        new_episode_count=len(new), reused_row_count=400-len(new), shards=[asdict(s) for s in shards],
        all_choices_sealed_before_final_outcomes=True))
    if shards:
        initialize_queue(root/'queues/final.sqlite', shards, contract_reference=str(root/'held_selection_manifest.json'))


def finalized_row(root, entry, raw):
    return dict(raw, global_task_id=entry['task'], teacher_demo=entry['demo'], video_ordinal=entry['init_state_id'],
        reference=entry['selected'], qualified_parameter_ref=entry['qualified_parameter_ref'],
        weight_source=entry['weight_source'], execution_source=entry['execution_source'],
        original_checkpoint=raw.get('checkpoint'), checkpoint=str(Path(root)/'checkpoints/update_00000180'),
        condition=entry['id'], query_excluded_from_H=True, fixed_unique_LoRA=True,
        compiler_exits_before_final_episode=True, unchanged_adaptation_source=entry['path_source'])


def final_episodes(args, runtime, runner):
    plan = read_json(args.root/'final_plan.json')
    if not plan['all_choices_sealed_before_final_outcomes'] or plan['new_episode_count'] > 400:
        raise ValueError('final execution requires the sealed bounded whole400 plan')
    tasks = {(t['suite'],t['task_id']):t for t in runner.contract['tasks']}
    while (claim := claim_next(args.root/'queues/final.sqlite', worker_id=args.worker_id)) is not None:
        try:
            task = tasks[claim.task_key]
            entries = [e for e in plan['entries'] if e['task'] == task['global_task_id']
                and e['init_state_id'] in claim.shard.init_state_ids]
            requests = [dict(task=task, state=load_file(entry['weight_source'],device=str(runtime.device)),
                parameter_ref=entry['selected'], state_id=entry['init_state_id'], noise_root=entry['noise_root'],
                registered_scene=entry['registered_scene'], episode_id=entry['id'], remaining=task['horizon']+10,
                capture_history=False, capture_hidden=False, uses_teaching=False) for entry in entries
                if not (args.root/'final_rows'/f"{entry['id']}.json").exists()]
            by_id = {e['id']:e for e in entries}
            for output in runner.run(requests):
                entry = by_id[output['request']['episode_id']]
                raw = dict(output['row'])
                if output['history'].records:
                    first = output['history'].records[0]
                    raw.update(first_native_normalized_actions=first['normalized_actions'].tolist(), first_noise_seed=first['noise_seed'])
                write_json_atomic(args.root/'final_rows'/f"{entry['id']}.json", finalized_row(args.root,entry,raw))
            rows = [read_json(args.root/'final_rows'/f"{entry['id']}.json") for entry in entries]
            file = args.root/'final_shards'/f'{claim.shard.job_id}_{claim.attempt:03d}.json'
            write_json_atomic(file, dict(rows=rows))
            publish_value_record(args.root/'queues/final.sqlite',claim,args.worker_id,file,len(rows),
                                 successes=sum(bool(row['success']) for row in rows))
        except BaseException as error:
            fail_job(args.root/'queues/final.sqlite', job_id=claim.shard.job_id, worker_id=args.worker_id,
                claim_token=claim.claim_token, error=repr(error))
            raise


def nonheld_summary(root):
    root = Path(root)
    manifest = read_json(root/'input_manifest.json')
    by_path = [read_json(root/'scores/nonheld'/f"{entry['id']}.json") for entry in manifest['paths']]
    fields = ('BCE','Brier','selected_query_return','beta_query_return','Phi0_query_return','regret')
    tasks = read_json(root/'run_contract.json')['training_tasks']
    per_task = []
    for task in tasks:
        paths = [p for p in by_path if p['task']==task]
        if len(paths) != 2:
            raise ValueError('nonheld endpoint requires both original pair and RL paths per task')
        per_task.append(dict(task=task, **{field:sum(sum(r[field] for r in p['rows'])/len(p['rows'])
            for p in paths)/2 for field in fields}))
    run = read_json(root/'run_contract.json')
    groups = dict(all=tasks, G_seen=run['G_seen_tasks'], G_novel=run['G_novel_tasks'])
    by_kind = {kind:{field:sum(sum(r[field] for r in p['rows'])/len(p['rows'])
        for p in by_path if p['kind']==kind)/72 for field in fields} for kind in ('pair','RL')}
    counterexamples = {'pair_MT_better':[], 'RL_final_MT_better_all_G':[]}
    for path in by_path:
        row = path['rows'][-1]
        rates = dict(zip(row['candidates'],row['query_success_rates'],strict=True))
        alternatives = [value for key,value in rates.items() if key!='MT300']
        if alternatives and rates['MT300'] > max(alternatives):
            group = 'pair_MT_better' if path['kind']=='pair' else 'RL_final_MT_better_all_G'
            counterexamples[group].append(dict(task=path['task'], path_source=path['path_source'], **row))
    if (len(counterexamples['pair_MT_better']),len(counterexamples['RL_final_MT_better_all_G'])) != (14,6):
        raise ValueError('original14 pair/6 RL fixed counterexample panel identity changed')
    result = dict(complete=True, paths=len(by_path), prefixes=sum(len(p['rows']) for p in by_path),
        candidate_items=sum(len(r['candidates']) for p in by_path for r in p['rows']), per_task=per_task,
        task_equal={group:{field:sum(p[field] for p in per_task if p['task'] in ids)/len(ids)
                          for field in fields} for group,ids in groups.items()}, pair_RL_task_equal=by_kind,
        original_counterexamples=counterexamples, raw_paths=by_path,
        training_domain_readout_not_generalization=True, no_checkpoint_selection=True)
    write_json_atomic(root/'nonheld_readout.json', result)
    return result
