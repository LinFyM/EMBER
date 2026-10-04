"""Matched fixed64 full-factor cotangent lifecycle; all profiles discarded."""
import gc
import json
import time
import torch
from safetensors.torch import save_file
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ..data import FormalData
from .common import ROOT, OLD, ASSET, TASKS, PARENT, LOCATOR, load_fixed, load_labels, rng_state, restore_rng, topology
from .model import ContentBinding, freeze_heads
from .credit import step


def optimizer(parameters):
    return torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)


def checkpoint(runtime, binding, opt, contract):
    path = ROOT / contract['arm'] / 'checkpoint64'; path.mkdir(exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in runtime.writer.state_dict().items()}, str(path / 'writer.safetensors'))
    save_file({k: v.detach().cpu().contiguous() for k, v in binding.state_dict().items()}, str(path / 'binding.safetensors'))
    torch.save(dict(schema=contract['schema'], optimizer=opt.state_dict(), scheduler=None, scaler=None,
        cursor=64, sampler=dict(manifest=str(OLD / 'query_manifest.json'), next_update=64), RNG=rng_state(),
        topology=contract['topology'], locator=file_record(LOCATOR), contract=contract), path / 'trainer_state.pt')
    write_json_atomic(path / 'checkpoint_manifest.json', dict(complete=True, next_update=64,
        schema=contract['schema'], locator_frozen_source=file_record(LOCATOR), locator_tensor_keys=['Pq.weight', 'Pk.weight'],
        files={p.name: file_record(p) for p in path.iterdir() if p.is_file()}))


def learn(runtime, spec, identity, affinity, arm):
    out = ROOT / arm; out.mkdir(exist_ok=True)
    if (out / 'checkpoint64').exists():
        raise ValueError('endpoint64 exists; do not train a second time')
    binding = ContentBinding(runtime.device)
    parameters = freeze_heads(runtime.writer) + tuple(binding.parameters())
    data = FormalData(ASSET, spec, query_labels=True, task_ids=TASKS)
    cached = {(t, d): load_fixed(t, d, arm, runtime.device) for t in TASKS for d in (0, 1)}
    labels = load_labels(); manifest = read_json(OLD / 'query_manifest.json')
    if len(manifest['steps']) != 64 or any(len(e['tasks']) != 4 for e in manifest['steps']):
        raise ValueError('exact retained64-event stream changed')
    contract = dict(schema='native_role_centered_content_training_v1', arm=arm, git=identity,
        parent=file_record(PARENT), parent_training_git='85919994aef11c17b49b7d0e70a2c110158bff61',
        source_training_git='b8ea00e9fbb86742ef076bac9dd35c5314cd5aed', updates=64,
        query_manifest=file_record(OLD / 'query_manifest.json'), labels=file_record(OLD / 'labels/manifest.json'),
        native=file_record(OLD / 'native/manifest.json'), content=file_record(ROOT / 'content/manifest.json'),
        frozen_locator=file_record(LOCATOR), locator_tensor_keys=['Pq.weight', 'Pk.weight'],
        P_initialization='18 identity', D_initialization='18 zero', binding_parameters=20054016,
        main_queries=14336, additional_tau1_queries=14336, query_offset=1, condition_weight=1/8,
        main_FM_weight=1, own_Q_weight=.1, teacher_selection_loss_weight=0,
        trainable_names=[n for n, p in runtime.writer.named_parameters() if p.requires_grad],
        frozen=['source', 'common_A0_B0', 'interpreter', 'native_X_H_c_d_K', 'locator', 'p'],
        replay='complete76 same-version; real A/h and upstream cotangents', topology=topology(affinity),
        optimizer=dict(kind='AdamW', lr=1e-4, betas=[.9, .95], eps=1e-8, weight_decay=1e-4, clip=1, scheduler=None))
    native_rows = read_json(OLD / 'native/manifest.json')['records']
    lengths = {(r['task'], r['teacher']): r['sampled_frames'] for r in native_rows}
    profile_entry = max(manifest['steps'], key=lambda e: sum(lengths[t,d] for t in e['tasks'] for d in (0,1)))
    initial = [p.detach().clone() for p in parameters]; initial_rng = rng_state(); profiles = []
    try:
        # Two registered real updates only; micro28 exhausts all logical queries.
        for micro in (14, 28):
            with torch.no_grad():
                for p, before in zip(parameters, initial, strict=True):
                    p.copy_(before)
            restore_rng(initial_rng); opt = optimizer(parameters)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                value = step(runtime, cached, data, labels, profile_entry, arm, binding, opt, parameters, micro)
            except torch.cuda.OutOfMemoryError as error:
                opt.zero_grad(set_to_none=True)
                value = dict(microbatch=micro, failed=True, error=str(error), seconds=time.monotonic()-started,
                             reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
            profiles.append(value); del opt
            write_json_atomic(out / 'profile.json', dict(discarded_updates=len(profiles), registered_event=profile_entry['update'], profiles=profiles))
        usable = [r for r in profiles if not r.get('failed')]
        if not usable:
            raise RuntimeError('two bounded real profiles failed; no extra search')
        selected = min(usable, key=lambda r:r['seconds'])
        # In-scope projection includes future materialization/readout; each arm
        # reserves <=3GPUh, leaving2 for shared crops/failures/delivery.
        if selected['seconds'] * 64 / 3600 + .65 > 3:
            raise RuntimeError('actual matched-profile projection threatens8GPUh; report boundary')
        with torch.no_grad():
            for p, before in zip(parameters, initial, strict=True):
                p.copy_(before)
        del initial; restore_rng(initial_rng); opt = optimizer(parameters)
        contract.update(profile_registered_event=profile_entry['update'], microbatch=selected['microbatch'], expected_training_seconds=64*selected['seconds'],
            profiles=profiles, profile_restored_all_parameters_optimizer_RNG=True,
            unused_K_GPU_cache=False, packed_teachers=2, largest_legal_same_task_query_microbatch=28,
            selection_reason='maximum actual updates/s; two independent arms concurrently, no extra queries')
        write_json_atomic(out / 'training_contract.json', contract)
        print(json.dumps(dict(event='profile_complete', arm=arm, microbatch=selected['microbatch'], seconds=selected['seconds'],
                              projected_training_GPUh=selected['seconds']*64/3600)), flush=True)
        with (out / 'metrics.jsonl').open('x') as log:
            for entry in manifest['steps']:
                value = step(runtime, cached, data, labels, entry, arm, binding, opt, parameters, selected['microbatch'])
                log.write(json.dumps(dict(update=entry['update'], **value))+'\n'); log.flush()
        checkpoint(runtime, binding, opt, contract)
        write_json_atomic(out / 'learning_completion.json', dict(complete=True, updates=64, reading_git=identity))
    finally:
        data.close()
    runtime.writer.zero_grad(set_to_none=True); runtime.writer.requires_grad_(False)
    binding.zero_grad(set_to_none=True); binding.requires_grad_(False)
    del opt, parameters, cached, labels; gc.collect(); torch.cuda.empty_cache()
    return binding
